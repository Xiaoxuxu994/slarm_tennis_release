#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Does physical extrapolation improve the frame24 landing? No retraining needed.

The evaluator builds frame24 by selecting ball pixels, unprojecting depth to a 3D
point, taking medians for pos15 and v15/a15/j15, then extrapolating with a
third-order Taylor term. a15 and j15 are free-form network predictions amplified
by dt^2 and dt^3, which is where the farthest bucket falls apart.

From the same extracted states, this compares three ways of doing the last step:

  free    the current one, pos + v*dt + 0.5*a*dt^2 + (1/6)*j*dt^3
  phys    pos + v*dt + 0.5*g*dt^2, with known gravity and no jerk
  linear  pos + v*dt

If phys has a much smaller p95 than free, free-form a/j is the culprit.

    SLARM_SINGLE_PROCESS=1 python tools/verify_physics_extrapolation.py \
        --config <config> --checkpoint <ckpt> --split validation --limit 100

Everything is in the rig frame, so --gravity is the rig-frame vector; the script
prints the measured GT ball acceleration so the sign and magnitude can be checked.
"""
import os
import sys
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")

from tools.stream25_runtime import (
    load_stream25_args,
    build_stream25_dataset,
    build_stream25_model,
    collate_and_prepare,
    slice_stream_observation,
)
from src.models.stream_session import StreamSession
from src.utils.stream25_metrics import (
    transform_position,
    transform_vector,
    finite_percentile,
    apply_ball_surface_offset,
    BALL_SURFACE_COEFFICIENT_MEASURED,
)


def _run_scene(model, prepared, device, dtype):
    """Stream six frames and return frame 15's predictions, rays and geometry."""
    session = StreamSession(model, mode="window", window_size=6)
    with torch.inference_mode():
        for obs_idx in range(6):
            obs = slice_stream_observation(prepared, obs_idx)
            session.forward_stream(obs, device, dtype)
        predictions = session.get_all_predictions()
        rays = model.plucker_embedder(
            prepared["target_intrinsics"],
            prepared["target_camtoworlds"],
            image_size=prepared["target_image"].shape[-2:],
        )
    render = predictions["render_results"]
    return {
        "depth15": render["rendered_depth"][0].float().cpu()[15],           # [V,H,W]
        "sem15": render["rendered_task_semantic"][0].long().cpu()[15],       # [V,H,W]
        "ms3_15": render["rendered_target_ms3"][0].float().cpu()[15],        # [V,H,W,9]
        "ray_o15": rays["origins"][0, 15].float().cpu(),                     # [V,H,W,3]
        "ray_d15": rays["dirs"][0, 15].float().cpu(),                        # [V,H,W,3]
    }


def _extract_ball_state(scene, canonical_to_rig, region_mask, ball_surface_offset=0.0):
    """Extract (pos15, v15, a15, j15) per view in the rig frame; None where the ball
    is absent. region_mask chooses predicted semantics or the GT ball mask.

    ball_surface_offset pushes the unprojected near-surface point back to the
    centre along the ray. That bias lies exactly along the depth direction, the
    "along" component in _pos15_decompose.
    """
    depth15, ms3_15 = scene["depth15"], scene["ms3_15"]
    positions15 = scene["ray_o15"] + scene["ray_d15"] * depth15[..., None]   # [V,H,W,3]
    positions15 = apply_ball_surface_offset(
        positions15, scene["ray_d15"], ball_surface_offset
    )
    per_eye = []
    for eye in range(depth15.shape[0]):
        mask = (
            region_mask[eye].bool()
            & torch.isfinite(depth15[eye])
            & (depth15[eye] > 0)
            & torch.isfinite(ms3_15[eye]).all(dim=-1)
            & torch.isfinite(positions15[eye]).all(dim=-1)
        )
        if not mask.any():
            per_eye.append(None)
            continue
        pos = transform_position(positions15[eye][mask].median(dim=0).values, canonical_to_rig)
        v, a, j = (
            transform_vector(ms3_15[eye, ..., o:o + 3][mask].median(dim=0).values, canonical_to_rig)
            for o in (0, 3, 6)
        )
        # Ray direction in the rig frame; depth error runs along it, which is how
        # the pos15 error is split into along and lateral
        view_dir = transform_vector(scene["ray_d15"][eye][mask].median(dim=0).values, canonical_to_rig)
        view_dir = view_dir / (view_dir.norm() + 1e-8)
        per_eye.append((pos, v, a, j, view_dir))
    return per_eye


def _scene_error(per_eye, gt_pos24, dt, gravity, strategy):
    """The evaluator's conservative rule: the largest valid error across views."""
    errs = []
    for state in per_eye:
        if state is None:
            continue
        pos, v, a, j = state[:4]
        if strategy == "free":
            pred = pos + v * dt + 0.5 * a * dt ** 2 + (1.0 / 6.0) * j * dt ** 3
        elif strategy == "phys":
            pred = pos + v * dt + 0.5 * gravity * dt ** 2
        elif strategy == "linear":
            pred = pos + v * dt
        errs.append(float((pred - gt_pos24).norm().item()))
    return max(errs) if errs else float("nan")


def gt_position_at(frame, gt_pos24, gt_v24, dt24, dt_target, gravity):
    """Truth at any target frame.

    These trajectories are exact parabolas, so truth past frame 24 needs no
    annotation: it follows from the last stored position and velocity plus gravity.
    That makes "how far can the prediction extrapolate" measurable without new GT.

    Only valid until the ball is caught or lands; past that the truth itself leaves
    the parabola and the numbers mean nothing.
    """
    d = dt_target - dt24
    return gt_pos24 + gt_v24 * d + 0.5 * gravity * d * d


def _pos15_error(per_eye, gt_pos15):
    """Starting-point error, the floor for frame24: however good the extrapolation,
    a wrong start costs at least this much, with coefficient 1."""
    errs = []
    for state in per_eye:
        if state is None:
            continue
        pos = state[0]
        errs.append(float((pos - gt_pos15).norm().item()))
    return max(errs) if errs else float("nan")


def _pos15_decompose(per_eye, gt_pos15):
    """Split the pos15 error into along-ray (depth) and lateral (localization)
    components, from the view with the largest total error."""
    best = None
    for state in per_eye:
        if state is None:
            continue
        pos, view_dir = state[0], state[4]
        err = pos - gt_pos15
        proj = err.dot(view_dir)
        along = float(proj.abs().item())
        lateral = float((err - proj * view_dir).norm().item())
        total = float(err.norm().item())
        if best is None or total > best[0]:
            best = (total, along, lateral)
    return (best[1], best[2]) if best is not None else None


def _v15_error(per_eye, gt_v15):
    """Velocity error at frame 15: medians of the MS3 velocity over the ball region,
    taken from the worst view."""
    if gt_v15 is None:
        return float("nan")
    errs = []
    for state in per_eye:
        if state is None:
            continue
        errs.append(float((state[1] - gt_v15).norm().item()))
    return max(errs) if errs else float("nan")


def _axis_errors(per_eye, gt_vec, state_index):
    """Split the error onto the three rig axes as (|ex|, |ey|, |ez|).

    This locates what dropping a camera costs. front_left/front_right form a
    horizontal baseline and lower_front the vertical one, so if removing it really
    hurts vertically the loss concentrates on z rather than spreading evenly --
    which decides whether to change camera placement or the model prior. The norm
    alone cannot answer that.
    """
    if gt_vec is None:
        return None
    best = None
    for state in per_eye:
        if state is None:
            continue
        err = state[state_index] - gt_vec
        total = float(err.norm().item())
        if best is None or total > best[0]:
            best = (total, [abs(float(err[i].item())) for i in range(3)])
    return best[1] if best is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes; 0 means all")
    ap.add_argument("--catch-frame", "--catch_frame", dest="catch_frame",
                    type=int, default=None,
                    help="frame at which the ball is caught, overriding the config. "
                         "It bounds where extrapolation is valid and is added to the targets")
    ap.add_argument("--target-frames", "--target_frames", dest="target_frames",
                    default="24",
                    help="comma-separated target frames, e.g. 24,30,40. Past 24 there is "
                         "no annotation and both sides extrapolate analytically, which "
                         "holds only until the ball is caught or lands")
    ap.add_argument("--gravity", default="0,0,-9.81", help="gravity vector in the rig frame, comma separated")
    ap.add_argument("--ball-radius-compensation", "--ball_radius_compensation",
                    dest="ball_radius_compensation", type=float, nargs="?",
                    const=BALL_SURFACE_COEFFICIENT_MEASURED, default=0.0,
                    help="push the unprojected near-surface point to the ball centre. "
                         "Takes a coefficient c; the offset is c x radius, and without a "
                         f"value it uses the measured {BALL_SURFACE_COEFFICIENT_MEASURED}. "
                         "Off by default; turning it on visibly lowers the along component.")
    ap.add_argument("--ball-radius", "--ball_radius", dest="ball_radius",
                    type=float, default=None,
                    help="ball radius in metres; defaults to the config's stream25_ball_radius")
    ap.add_argument("--ball-mask-source", choices=["pred", "gt", "both"], default="both",
                    help="ball region source: pred (predicted semantics) / gt (GT mask) / both")
    args_cli = ap.parse_args()

    gravity = torch.tensor([float(x) for x in args_cli.gravity.split(",")], dtype=torch.float32)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    args = load_stream25_args(args_cli.config, checkpoint_path=args_cli.checkpoint,
                              checkpoint_role="evaluation")
    # After args are parsed: this reads the config.
    ball_radius = args_cli.ball_radius
    if ball_radius is None:
        ball_radius = float(getattr(args, "stream25_ball_radius", 0.0325) or 0.0325)
    ball_surface_offset = float(args_cli.ball_radius_compensation) * float(ball_radius)
    if ball_surface_offset:
        print(f"[verify] ball-surface compensation ON: "
              f"{args_cli.ball_radius_compensation:.3f} x {ball_radius:.4f} m "
              f"= {ball_surface_offset * 100:.2f} cm along the view ray "
              f"(front surface -> centre). Not comparable to runs without it.",
              flush=True)
    else:
        print("[verify] ball-surface compensation off: pos15 is the ball FRONT SURFACE, "
              "GT is the CENTRE, so a constant bias sits in the 'along' component.",
              flush=True)

    # Catch frame: command line, then config, then unknown. It also bounds where
    # extrapolation still means anything.
    catch_frame = args_cli.catch_frame
    if catch_frame is None:
        catch_frame = int(getattr(args, "stream25_catch_frame", 0) or 0)
    catch_frame = catch_frame if catch_frame > 0 else None

    target_frames = sorted({int(x) for x in args_cli.target_frames.split(",") if x.strip()})
    if not target_frames:
        target_frames = [24]
    if catch_frame is not None and catch_frame not in target_frames:
        target_frames = sorted(target_frames + [catch_frame])

    # Past the catch the truth leaves the parabola, so drop those frames rather
    # than compute numbers nobody should read.
    if catch_frame is not None:
        beyond = [f for f in target_frames if f > catch_frame]
        if beyond:
            print(f"[skip] target frames {beyond} are past the catch frame "
                  f"{catch_frame}; the ball is no longer in free flight there and "
                  f"the ground truth stops following the parabola")
            target_frames = [f for f in target_frames if f <= catch_frame]
    elif max(target_frames) > 24:
        print(f"[warn] scoring past frame 24 without a catch frame. Set "
              f"stream25_catch_frame in the config (or pass --catch-frame) so the "
              f"extrapolation stops where the ball is actually caught.")
    if not target_frames:
        target_frames = [24]
    dataset = build_stream25_dataset(args, args_cli.split, online_feat=False)
    model = build_stream25_model(args, device)
    model.eval()

    n = len(dataset) if args_cli.limit <= 0 else min(args_cli.limit, len(dataset))
    print(f"[verify] {args_cli.split}: {n}/{len(dataset)} scenes | gravity(rig)={gravity.tolist()}", flush=True)

    per_scene = []          # (per_eye_states, gt_pos24, gt_pos15, gt_v15, dt, targets)
    gt_accel_samples = []   # GT ball acceleration, to check the gravity vector

    for index in range(n):
        t0 = time.time()
        input_dict, target_dict = collate_and_prepare(dataset[index], args, device)
        t_load = time.time()
        prepared = dict(input_dict)
        prepared.update(target_dict)

        scene = _run_scene(model, prepared, device, dtype)
        t_fwd = time.time()
        canonical_to_rig = prepared["context_canonical_to_rig"][0, -1].float().cpu()
        gt_pos24 = prepared["ball_position_rig"][0, 24].float().cpu()
        gt_pos15 = prepared["ball_position_rig"][0, 15].float().cpu()
        gt_v24 = prepared["ball_velocity_rig"][0, 24].float().cpu()
        dt = float(
            (prepared["target_time"][0, 24, 0] - prepared["context_time"][0, -1, 0]).item()
            * args.timespan
        )
        # Per-frame step derived from frame15 -> frame24; no fps assumed
        dt_per_frame = dt / (24 - 15)
        # target frame -> (truth position, dt from frame 15); past 24 is analytic
        targets = {}
        for tf in target_frames:
            dt_tf = (tf - 15) * dt_per_frame
            g = (prepared["ball_position_rig"][0, tf].float().cpu() if tf <= 24
                 else gt_position_at(tf, gt_pos24, gt_v24, dt, dt_tf, gravity))
            targets[tf] = (g, dt_tf)
        # Ball region from predicted semantics and/or the GT mask
        regions = {}
        if args_cli.ball_mask_source in ("pred", "both"):
            regions["pred"] = (scene["sem15"] == 1)
        if args_cli.ball_mask_source in ("gt", "both"):
            regions["gt"] = prepared["ball_ms3_mask"][0].bool().cpu()[15]
        scene_states = {
            src: _extract_ball_state(scene, canonical_to_rig, region,
                                     ball_surface_offset=ball_surface_offset)
            for src, region in regions.items()
        }
        # GT velocity and acceleration from the dense MS3 ball region median
        gt_v15 = None
        try:
            gt_ms3_15 = prepared["dense_ms3_gt"][0].float().cpu()[15]        # [V,H,W,9]
            ball_mask_15 = prepared["ball_ms3_mask"][0].bool().cpu()[15]     # [V,H,W]
            v_samples = []
            for eye in range(gt_ms3_15.shape[0]):
                m = ball_mask_15[eye]
                if m.any():
                    a_gt = gt_ms3_15[eye, ..., 3:6][m].median(dim=0).values
                    gt_accel_samples.append(transform_vector(a_gt, canonical_to_rig))
                    v_gt = gt_ms3_15[eye, ..., 0:3][m].median(dim=0).values
                    v_samples.append(transform_vector(v_gt, canonical_to_rig))
            if v_samples:
                gt_v15 = torch.stack(v_samples).mean(dim=0)
        except Exception:
            pass

        per_scene.append((scene_states, gt_pos24, gt_pos15, gt_v15, dt, targets))

        del input_dict, target_dict, prepared, scene
        print(
            f"  scene {index + 1}/{n}  load={t_load - t0:.1f}s "
            f"fwd+render={t_fwd - t_load:.1f}s extract={time.time() - t_fwd:.1f}s",
            flush=True,
        )

    if gt_accel_samples:
        g_mean = torch.stack(gt_accel_samples).mean(dim=0)
        print(f"\n[check] GT ball accel(rig) mean = {g_mean.tolist()}  (should be ~= gravity; use to verify --gravity)")

    sources = [s for s in ("pred", "gt") if per_scene and s in per_scene[0][0]]
    # 1. pos15 starting error: the floor for frame24
    print("\n" + "=" * 72)
    print(f"{'region':8s} {'metric':16s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for src in sources:
        errs = [_pos15_error(states[src], gp15) for (states, _g24, gp15, _gv, _dt, _bt, _tg) in per_scene]
        finite = [e for e in errs if e == e]
        med = finite_percentile(finite, 50) if finite else float("nan")
        p95 = finite_percentile(finite, 95) if finite else float("nan")
        print(f"{src:8s} {'pos15_error':16s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
    print("=" * 72)

    # 1b. Split into along-ray (depth) and lateral (localization). The compensation
    # setting goes in the header, not just at startup: two runs can print identical
    # along_med with different settings, and then neither can be compared.
    _comp = (f"   [ball-centre comp: {ball_surface_offset*100:.2f} cm]"
             if ball_surface_offset else "   [ball-centre comp: OFF -> along still carries the ball-radius bias]")
    print(f"{'region':8s} {'pos15_split':14s} {'along_med':>10s} {'along_p95':>10s}"
          f" {'lat_med':>10s} {'lat_p95':>10s}{_comp}")
    print("-" * 72)
    for src in sources:
        decs = [_pos15_decompose(states[src], gp15) for (states, _g24, gp15, _gv, _dt, _bt, _tg) in per_scene]
        decs = [d for d in decs if d is not None]
        along = [d[0] for d in decs]
        lateral = [d[1] for d in decs]
        am = finite_percentile(along, 50) if along else float("nan")
        ap = finite_percentile(along, 95) if along else float("nan")
        lm = finite_percentile(lateral, 50) if lateral else float("nan")
        lp = finite_percentile(lateral, 95) if lateral else float("nan")
        print(f"{src:8s} {'along/lateral':14s} {am:10.4f} {ap:10.4f} {lm:10.4f} {lp:10.4f}")
    print("=" * 72)

    # 1c. Velocity error, the main source of frame24 minus pos15
    print(f"{'region':8s} {'metric':16s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for src in sources:
        errs = [_v15_error(states[src], gv) for (states, _g24, _gp15, gv, _dt, _bt, _tg) in per_scene]
        finite = [e for e in errs if e == e]
        med = finite_percentile(finite, 50) if finite else float("nan")
        p95 = finite_percentile(finite, 95) if finite else float("nan")
        print(f"{src:8s} {'v15_error':16s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
    print("=" * 72)

    # 1d. Per-axis split; z is the gravity axis. A horizontal baseline constrains
    # the vertical direction least, so losing the vertical camera should show up on
    # z. Spread evenly, it is plain added noise and placement will not help.
    print(f"{'region':8s} {'axis_split':14s} {'x_med':>9s} {'y_med':>9s} {'z_med':>9s} "
          f"{'z_share':>9s} {'n':>6s}")
    print("-" * 72)
    for label, gt_index, state_index in (("pos15", 2, 0), ("v15", 3, 1)):
        for src in sources:
            # per_scene = (scene_states, gt_pos24, gt_pos15, gt_v15, dt, targets)
            rows = [
                _axis_errors(scene[0][src], scene[gt_index], state_index)
                for scene in per_scene
            ]
            rows = [r for r in rows if r is not None]
            if not rows:
                print(f"{src:8s} {label:14s} {'-':>9s} {'-':>9s} {'-':>9s} {'-':>9s} {0:6d}")
                continue
            meds = [finite_percentile([r[i] for r in rows], 50) for i in range(3)]
            total = sum(m * m for m in meds)
            z_share = (meds[2] * meds[2] / total) if total > 0 else float("nan")
            print(f"{src:8s} {label:14s} {meds[0]:9.4f} {meds[1]:9.4f} {meds[2]:9.4f} "
                  f"{z_share:9.1%} {len(rows):6d}")
    print("=" * 72)

    # 2. Landing error: target frame x extrapolation x region source. Past frame 24
    # both sides extrapolate analytically, and since these trajectories are exact
    # parabolas the truth side carries no approximation.
    print(f"{'region':8s} {'frame':>5s} {'extrap':12s} {'median':>10s} {'p95':>10s} {'n_valid':>8s}")
    print("-" * 72)
    for tf in target_frames:
        for src in sources:
            for strat, name in [("free", "free(current)"), ("phys", "phys(gravity)"), ("linear", "linear")]:
                errs = [
                    _scene_error(states[src], tg[tf][0], tg[tf][1], gravity, strat)
                    for (states, _g24, _gp15, _gv, _dt, _bt, tg) in per_scene
                ]
                finite = [e for e in errs if e == e]  # drop nan
                med = finite_percentile(finite, 50) if finite else float("nan")
                p95 = finite_percentile(finite, 95) if finite else float("nan")
                print(f"{src:8s} {tf:>5d} {name:12s} {med:10.4f} {p95:10.4f} {len(finite):8d}")
        if tf != target_frames[-1]:
            print("-" * 72)
    if max(target_frames) > 24:
        print("")
        if catch_frame is not None:
            print(f"  catch frame = {catch_frame} (config: stream25_catch_frame). "
                  f"Targets past it are dropped.")
        print(f"  frames past 24 have no annotation. Both sides are extrapolated with the")
        print(f"  same analytic ballistic, which is exact for this data (second difference")
        print(f"  of position is -9.8100, velocity agrees to 1e-5 m/s), so the comparison")
        print(f"  is valid -- but only until the ball is caught or lands. Past that the")
        print(f"  ground truth stops following the parabola and the numbers mean nothing.")

    print("Readout:")
    print("  pos15_error = frame24 floor; (pred - gt) = cost of ball-selection error")
    print("  along vs lateral: along = depth expected-value error, lateral = localization error")
    print("  pred x phys ~= pred x free  -> extrapolation (a/j) is NOT the bottleneck")
    print("  gt x * << pred x *  -> ball selection is the bottleneck -> ball token")
    print("  frame24 minus pos15  -> contribution of v15 (velocity) + extrapolation")
    print("  v15_error x dt(~0.3)  ~= that extrapolation contribution")
    print("  axis_split z_share: fraction of the squared median error on the gravity axis.")
    print("    Isotropic noise sits near 33%. Well above that means the vertical direction")
    print("    is the weak one, which is what dropping a vertical-baseline camera predicts,")
    print("    and the fix is camera placement. Near 33% means it is plain added noise and")
    print("    placement will not help -- spend the effort on the trajectory prior instead.")


if __name__ == "__main__":
    main()
