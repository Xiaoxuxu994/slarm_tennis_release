#!/usr/bin/env python
"""Per-scene ball trajectory: where the model puts the ball, frame by frame.

The rendered video cannot answer this. The ball is 2.66 px across in a
320x240 frame, seven thousandths of one percent of the image, so at full frame
it is a smudge whether the geometry is right or wrong. This reads the same
quantity the evaluator scores -- the median of the back-projected ball pixels,
per view, per frame -- and writes it out as a track that can be plotted.

It runs the single forward that render uses, not the six-step StreamSession, so
it can be asked for frames past 24 and show where the ball is predicted to be
at the catch. Truth is the stored trajectory up to the last exported frame and
the analytic ballistic continuation beyond it, marked as such in the output.

    python tools/export_ball_track.py \\
        --config configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml \\
        --checkpoint work_dirs/.../ckpt_013999.pth \\
        --scene 0 --num-frames 46 --output ball_track

Writes three files:
  <output>.csv     one row per frame and view
  <output>.html    the three axes and the error, openable in a browser
  <output>_3d.png  the trajectory in space, truth against prediction, with the
                   catch ring drawn where the ball is supposed to arrive

Two things worth looking for in the plot:
  - Does the error DRIFT smoothly across frames, or scatter about zero? A
    smooth drift is indistinguishable from velocity to a ballistic fit, and is
    the known reason the fitted velocity sits over twice its noise floor.
  - Do the three views agree with each other while all disagreeing with truth?
    That is a common-mode error, which averaging over views cannot remove.

All output is ASCII.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.dataset.stream25 import MS3_GRAVITY_RIG
from src.utils.stream25_metrics import transform_position


def ball_positions(depth, semantic, origins, directions, canonical_to_rig) -> List[Optional[List[float]]]:
    """Median of the back-projected ball pixels, per view. None where no ball."""
    points = origins + directions * depth[..., None]
    out: List[Optional[List[float]]] = []
    for eye in range(depth.shape[0]):
        mask = (
            (semantic[eye] == 1)
            & torch.isfinite(depth[eye])
            & (depth[eye] > 0)
            & torch.isfinite(points[eye]).all(dim=-1)
        )
        if not mask.any():
            out.append(None)
            continue
        centre = transform_position(points[eye][mask].median(dim=0).values, canonical_to_rig)
        out.append([float(x) for x in centre])
    return out


def plot_3d(path, rows, frames, views, scene, checkpoint, last_stored,
            catch_frame, ring_radius, ball_radius) -> Optional[str]:
    """The trajectory in space: truth as a line, each view's readings as points.

    The three separate axis plots show WHEN the track goes wrong; this shows
    WHERE. It also draws the catch ring at the true landing point, oriented
    across the ball's arrival direction, so the landing error can be seen
    against the aperture it has to fit through rather than compared to a
    tolerance in a table.

    Returns None when matplotlib is absent, since everything else still works.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    truth = np.array([r["gt"] for r in rows], dtype=float)
    figure = plt.figure(figsize=(13, 5.6))
    colours = ["#2E6389", "#B33D33", "#2B7A56"]

    # Left: the whole arc. Right: the catch, close enough to see the ring.
    # One panel cannot do both -- the arc spans about three metres and the ring
    # is 0.27, so at full extent the aperture is a smudge and the landing error
    # cannot be read against the thing it has to fit through.
    for position, (elev, azim, title, closeup) in enumerate((
            (18, -62, "whole flight", False),
            (14, -62, f"at the catch (frame {catch_frame})", True)), start=1):
        axes = figure.add_subplot(1, 2, position, projection="3d")
        span = (slice(max(0, catch_frame - 6), catch_frame + 1)
                if closeup and catch_frame < len(truth) else slice(None))
        axes.plot(truth[span, 0], truth[span, 1], truth[span, 2],
                  color="#171A18", linewidth=1.8,
                  label="truth" if position == 1 else None, zorder=2)
        for index, view in enumerate(views):
            subset = rows[span] if isinstance(span, slice) else rows
            points = np.array([r["views"][index] for r in subset
                               if r["views"][index] is not None], dtype=float)
            if len(points):
                axes.scatter(points[:, 0], points[:, 1], points[:, 2], s=9,
                             color=colours[index % len(colours)], alpha=0.8,
                             label=view if position == 1 else None,
                             depthshade=False, zorder=3)

        # The terminal observation is where every prediction is made from, and
        # the catch is where it is scored, so both are worth finding by eye.
        for frame, marker, colour, label in (
                (15, "s", "#C4D62C", "frame 15 (terminal)"),
                (catch_frame, "*", "#B33D33", f"frame {catch_frame} (catch)")):
            if frame < len(truth):
                axes.scatter(*truth[frame], s=110, marker=marker, color=colour,
                             edgecolors="#171A18", linewidths=0.8, zorder=5,
                             label=label if position == 1 else None)

        # The ring, drawn across the arrival direction at the true landing.
        if catch_frame < len(truth) and catch_frame >= 1:
            direction = truth[catch_frame] - truth[catch_frame - 1]
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                axis = direction / norm
                seed = np.array([1.0, 0.0, 0.0])
                if abs(axis[0]) > 0.9:
                    seed = np.array([0.0, 1.0, 0.0])
                u = np.cross(axis, seed)
                u /= np.linalg.norm(u)
                w = np.cross(axis, u)
                angle = np.linspace(0, 2 * np.pi, 80)
                ring = (truth[catch_frame][None, :]
                        + ring_radius * (np.cos(angle)[:, None] * u[None, :]
                                         + np.sin(angle)[:, None] * w[None, :]))
                axes.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#2E6389",
                          linewidth=2.4, zorder=4,
                          label="catch ring" if position == 1 else None)

        # The predicted landing itself, which is what the success criterion
        # scores. Only drawn in the close-up, where it is not a dot on a dot.
        if closeup and catch_frame < len(rows):
            for index, view in enumerate(views):
                landing = rows[catch_frame]["views"][index]
                if landing is not None:
                    axes.scatter(*landing, s=90, marker="o",
                                 color=colours[index % len(colours)],
                                 edgecolors="#171A18", linewidths=0.8, zorder=6)

        # Equal scale on all three axes, or a 3 m arc in a 0.3 m box reads as a
        # different shape entirely.
        if closeup and catch_frame < len(truth):
            centre = truth[catch_frame]
            reach = ring_radius * 2.2
        else:
            spans = np.stack([truth.min(axis=0), truth.max(axis=0)])
            centre = spans.mean(axis=0)
            reach = max(float((spans[1] - spans[0]).max()) * 0.55, 0.1)
        axes.set_xlim(centre[0] - reach, centre[0] + reach)
        axes.set_ylim(centre[1] - reach, centre[1] + reach)
        axes.set_zlim(centre[2] - reach, centre[2] + reach)
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=elev, azim=azim)
        axes.set_xlabel("x (m)"); axes.set_ylabel("y (m)"); axes.set_zlabel("z (m)")
        axes.set_title(title, fontsize=10)
        axes.tick_params(labelsize=7)

    figure.suptitle(
        f"Ball track  |  scene {scene}  |  {checkpoint}  |  "
        f"truth stored through frame {last_stored}, ballistically continued after",
        fontsize=10)
    figure.legend(*figure.axes[0].get_legend_handles_labels(),
                  loc="lower center", ncol=6, fontsize=8, frameon=False)
    figure.tight_layout(rect=(0, 0.06, 1, 0.96))
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return str(path)


def svg_plot(rows: List[Dict[str, Any]], frames: List[int], axis: int, name: str,
             views: List[str]) -> str:
    """One axis of the track: truth as a line, each view's reading as dots."""
    width, height, pad = 620, 190, 44
    truth = [(f, r["gt"][axis]) for f, r in zip(frames, rows) if r["gt"] is not None]
    readings = [(f, r["views"][i][axis])
                for f, r in zip(frames, rows)
                for i in range(len(views)) if r["views"][i] is not None]
    if not truth and not readings:
        return ""
    values = [v for _, v in truth] + [v for _, v in readings]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    def sx(f: int) -> float:
        return pad + (f - frames[0]) / max(1, frames[-1] - frames[0]) * (width - pad - 14)

    def sy(v: float) -> float:
        return height - pad + 14 - (v - lo) / (hi - lo) * (height - pad - 14)

    colours = ["#2E6389", "#B33D33", "#2B7A56"]
    parts = [f'<svg viewBox="0 0 {width} {height}" class="plot">']
    parts.append(f'<line x1="{pad}" y1="{height-pad+14:.1f}" x2="{width-14}" y2="{height-pad+14:.1f}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
    parts.append(f'<text x="{pad}" y="16" class="t lab">{name}</text>')
    parts.append(f'<text x="{width-14}" y="16" class="t dim" text-anchor="end">'
                 f'{lo:.2f} .. {hi:.2f} m</text>')
    if truth:
        d = " ".join(f"{'M' if i == 0 else 'L'} {sx(f):.1f} {sy(v):.1f}"
                     for i, (f, v) in enumerate(truth))
        parts.append(f'<path d="{d}" fill="none" stroke="var(--ink)" stroke-width="2"/>')
    for i, view in enumerate(views):
        for f, r in zip(frames, rows):
            if r["views"][i] is None:
                continue
            parts.append(f'<circle cx="{sx(f):.1f}" cy="{sy(r["views"][i][axis]):.1f}" r="2.6" '
                         f'fill="{colours[i % len(colours)]}" fill-opacity="0.85"/>')
    parts.append(f'<text x="{pad}" y="{height-6}" class="t dim">frame {frames[0]}</text>')
    parts.append(f'<text x="{width-14}" y="{height-6}" class="t dim" text-anchor="end">'
                 f'frame {frames[-1]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_html(path: Path, rows, frames, views, scene, checkpoint, last_stored) -> None:
    colours = ["#2E6389", "#B33D33", "#2B7A56"]
    legend = " ".join(
        f'<span class="key"><i style="background:{colours[i % len(colours)]}"></i>{v}</span>'
        for i, v in enumerate(views)
    )
    plots = "".join(svg_plot(rows, frames, a, n, views)
                    for a, n in enumerate(("x (m)", "y (m)", "z (m)")))
    errors = [(f, max((r["err"][i] for i in range(len(views)) if r["err"][i] is not None),
                      default=None))
              for f, r in zip(frames, rows)]
    err_rows = "".join(
        f"<tr><td>{f}</td><td>{'n/a' if e is None else f'{e*100:.2f}'}</td>"
        f"<td>{'stored' if f <= last_stored else 'continued'}</td></tr>"
        for f, e in errors)
    path.write_text(f"""<!doctype html><meta charset="utf-8">
<title>Ball track scene {scene}</title>
<style>
 :root {{ --paper:#F4F6F3; --panel:#fff; --ink:#171A18; --muted:#6E756F; --rule:#DCE1DA; }}
 @media (prefers-color-scheme: dark) {{ :root {{
   --paper:#131614; --panel:#1B1F1C; --ink:#E9ECE7; --muted:#99A29B; --rule:#2C322D; }} }}
 body {{ background:var(--paper); color:var(--ink); margin:0; padding:28px 20px 48px;
   font:14px/1.7 system-ui,sans-serif; }}
 .wrap {{ max-width:700px; margin:0 auto; display:flex; flex-direction:column; gap:18px; }}
 h1 {{ font-size:20px; margin:0; }} p {{ margin:0; color:var(--muted); }}
 .plot {{ display:block; width:100%; height:auto; background:var(--panel);
   border:1px solid var(--rule); }}
 .t {{ font:11px ui-monospace,monospace; }} .lab {{ fill:var(--ink); font-weight:600; }}
 .dim {{ fill:var(--muted); }}
 .key {{ display:inline-flex; align-items:center; gap:6px; margin-right:14px;
   font:12px ui-monospace,monospace; color:var(--muted); }}
 .key i {{ width:10px; height:10px; border-radius:50%; display:inline-block; }}
 table {{ border-collapse:collapse; font:12px ui-monospace,monospace; }}
 td, th {{ padding:3px 12px; border-bottom:1px solid var(--rule); text-align:right; }}
 th:last-child, td:last-child {{ text-align:left; }}
 .scroll {{ max-height:340px; overflow:auto; background:var(--panel);
   border:1px solid var(--rule); }}
</style>
<div class="wrap">
<h1>Ball track &mdash; scene {scene}</h1>
<p>{checkpoint}<br>Black line is truth; dots are each view's reading.
Truth is stored through frame {last_stored} and ballistically continued after it.</p>
<div>{legend}<span class="key"><i style="background:var(--ink)"></i>truth</span></div>
{plots}
<h1 style="font-size:16px">Worst-view error per frame (cm)</h1>
<div class="scroll"><table><thead><tr><th>frame</th><th>error</th><th>truth</th></tr></thead>
<tbody>{err_rows}</tbody></table></div>
</div>
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", type=int, default=0, help="index into the eval manifest")
    parser.add_argument("--num-frames", type=int, default=46,
                        help="render this many frames; past the stored ones the model "
                             "is extrapolating and truth is continued analytically")
    parser.add_argument("--output", type=Path, default=Path("ball_track"))
    parser.add_argument("--ring-diameter", type=float, default=0.27,
                        help="catch ring diameter in metres, drawn at the landing "
                             "point so the error can be seen against the aperture")
    cli = parser.parse_args()

    from scripts.render_stream25_base import configure_reconstruction_timeline
    from src.dataset.data_utils import to_batch_tensor, prepare_inputs_and_targets
    from src.utils.stream25_metrics import set_camera_order
    from src.utils.misc import camera_names_from_arguments
    from tools.stream25_runtime import (
        build_stream25_dataset, build_stream25_model, load_stream25_args,
    )

    args = load_stream25_args(cli.config, checkpoint_path=cli.checkpoint,
                              checkpoint_role="evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_stream25_dataset(args, "validation", online_feat=False)
    if not 0 <= cli.scene < len(dataset):
        parser.error(f"--scene must be in [0, {len(dataset)})")
    model = build_stream25_model(args, device)
    model.eval()

    sample = dataset[cli.scene]
    scene_name = sample.get("scene_name", str(cli.scene))

    batch = to_batch_tensor(sample)
    batch["num_max_cams"] = int(batch["num_max_cams"][0]) if not isinstance(
        batch["num_max_cams"], int) else batch["num_max_cams"]
    input_dict, target_dict = prepare_inputs_and_targets(
        batch, device, v=batch["num_max_cams"], timespan=args.timespan, feat_extractor=None)
    stored = int(target_dict["target_image"].shape[1])
    input_dict = configure_reconstruction_timeline(input_dict, num_frames=cli.num_frames)

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        prediction = model(input_dict)

    render = prediction["render_results"]
    depth = render["rendered_depth"][0].float().cpu()
    semantic = prediction["rendered_task_semantic"][0].long().cpu()
    rays = model.plucker_embedder(input_dict["target_intrinsics"], input_dict["target_camtoworlds"],
                                  image_size=depth.shape[-2:])
    origins = rays["origins"][0].float().cpu()
    directions = rays["dirs"][0].float().cpu()
    canonical_to_rig = input_dict["context_canonical_to_rig"][0, -1].float().cpu()

    truth_all = target_dict.get("ball_position_rig")
    velocity_all = target_dict.get("ball_velocity_rig")
    if truth_all is None or velocity_all is None:
        parser.error("this dataset has no ball_position_rig / ball_velocity_rig")
    truth_all = truth_all[0].float().cpu()
    gravity = torch.tensor(MS3_GRAVITY_RIG, dtype=torch.float32)
    step = float(args.timespan) / 24.0
    last = min(stored, truth_all.shape[0]) - 1
    tail_p = truth_all[last]
    tail_v = velocity_all[0, last].float().cpu()

    # Real camera names, so a column in the CSV can be matched against the
    # per-view rows of an evaluation report without counting positions.
    view_names = list(set_camera_order(
        camera_names_from_arguments(args, role="evaluation")
    ))[: depth.shape[1]]
    while len(view_names) < depth.shape[1]:
        view_names.append(f"view{len(view_names)}")
    frames = list(range(min(cli.num_frames, depth.shape[0])))
    rows: List[Dict[str, Any]] = []
    for frame in frames:
        per_view = ball_positions(depth[frame], semantic[frame], origins[frame],
                                  directions[frame], canonical_to_rig)
        if frame <= last:
            gt = [float(x) for x in truth_all[frame]]
        else:
            dt = (frame - last) * step
            gt = [float(x) for x in (tail_p + tail_v * dt + 0.5 * gravity * dt * dt)]
        errors = [None if p is None else math.dist(p, gt) for p in per_view]
        rows.append({"views": per_view, "gt": gt, "err": errors})

    csv_path = cli.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as handle:
        handle.write("frame,view,pred_x,pred_y,pred_z,gt_x,gt_y,gt_z,error_m,truth_source\n")
        for frame, row in zip(frames, rows):
            source = "stored" if frame <= last else "continued"
            for i, name in enumerate(view_names):
                p = row["views"][i]
                cells = ",".join("" if p is None else f"{x:.6f}" for x in (p or [0, 0, 0]))
                err = "" if row["err"][i] is None else f"{row['err'][i]:.6f}"
                handle.write(f"{frame},{name},{cells},"
                             f"{row['gt'][0]:.6f},{row['gt'][1]:.6f},{row['gt'][2]:.6f},"
                             f"{err},{source}\n")

    html_path = cli.output.with_suffix(".html")
    write_html(html_path, rows, frames, view_names, scene_name,
               Path(cli.checkpoint).name, last)

    ball_radius = float(getattr(args, "stream25_ball_radius", 0.0325) or 0.0325)
    png_path = plot_3d(
        cli.output.parent / f"{cli.output.name}_3d.png", rows, frames, view_names,
        scene_name, Path(cli.checkpoint).name, last,
        catch_frame=min(int(getattr(args, "stream25_catch_frame", 45) or 45),
                        len(frames) - 1),
        ring_radius=cli.ring_diameter / 2, ball_radius=ball_radius)

    finite = [e for row in rows for e in row["err"] if e is not None]
    missing = sum(1 for row in rows for e in row["err"] if e is None)
    print(f"scene        : {scene_name} (index {cli.scene})")
    print(f"frames       : {frames[0]}..{frames[-1]}  "
          f"(truth stored through {last}, continued after)")
    print(f"ball found   : {len(finite)} of {len(frames) * len(view_names)} view-frames"
          + (f", missing in {missing}" if missing else ""))
    if finite:
        print(f"error        : median {sorted(finite)[len(finite)//2]*100:.2f} cm, "
              f"max {max(finite)*100:.2f} cm")
    print(f"wrote        : {csv_path}")
    print(f"               {html_path}")
    if png_path:
        print(f"               {png_path}")
    else:
        print("               (no 3D png: matplotlib is not installed)")
    print("")
    print("In the plot: a smoothly DRIFTING error is indistinguishable from velocity")
    print("to a ballistic fit, and is the known reason the fitted velocity sits over")
    print("twice its noise floor. Scatter about zero is harmless by comparison.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
