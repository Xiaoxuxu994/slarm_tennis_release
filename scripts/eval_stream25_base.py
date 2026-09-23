"""True-stream evaluator for Stream25 (Task 9, spec sec.8).

Instantiates a fresh StreamSession per scene, streams [0,3,6,9,12,15], renders
all 25 native tri-view targets after observation 6, and applies frozen absolute
gates. Includes the final-test guard with O_CREAT|O_EXCL sentinel.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

WORKTREE = Path(__file__).resolve().parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

from src.utils.stream25_metrics import (
    ACCEPTANCE_TABLE,
    # Frozen tri-view defaults, re-exported for tests only; at run time always use
    # get_camera_order() / get_required_eval_scopes(), or a two-view config raises.
    CAMERA_ORDER,
    REQUIRED_EVAL_SCOPES,
    get_camera_order,
    get_required_eval_scopes,
    set_camera_order,
    TIME_BUCKETS,
    worst_normalized_ratio,
    compute_depth_absrel,
    compute_depth_rmse,
    compute_ball_region_iou,
    compute_iou,
    compute_macro_dice,
    compute_miou,
    compute_psnr,
    compute_ssim,
    finite_percentile,
    integrate_frame24_position,
    integrate_frame24_position_physics,
    fit_ballistic_state,
    apply_ball_surface_offset,
    BALL_SURFACE_COEFFICIENT_MEASURED,
    transform_position,
    transform_vector,
)
from src.dataset.stream25 import (
    MS3_GRAVITY_RIG,
    STREAM25_ALL_TARGET_FRAMES,
    STREAM25_CONTEXT_FRAMES,
)

#: Index of the terminal observation in the TARGET LIST, never a frame number.
#  ball_position_rig / ball_velocity_rig are indexed by target, so their axis is
#  25-offset long; adding the offset here reads past the end (or, worse, reads a
#  plausible wrong instant). Absolute frame 15 + offset is only for time arithmetic.
TERMINAL_TARGET_INDEX = STREAM25_CONTEXT_FRAMES[-1]

# Side-by-side landing readouts; not gated, for comparison only.
SIDE_METRIC_NAMES = (
    # Refit (pos15, v15) from the ball centres rendered at frames 0..15 instead of
    # taking v15 from the MS3 head. Every checkpoint can compute this.
    "frame24_position_fit",
    "ball_pos15_error_fit",
    "ball_vel15_error_fit",
    # Per-frame position error, split into a constant offset and frame-to-frame
    # scatter. Only the scatter reaches the fitted velocity; the constant cancels.
    "pixel_pos_error_frame0", "pixel_pos_error_frame3", "pixel_pos_error_frame6",
    "pixel_pos_error_frame9", "pixel_pos_error_frame12", "pixel_pos_error_frame15",
    "pixel_pos_error_constant_m",
    "pixel_pos_error_scatter_m",
    # Landing at the catch frame -- the only landing metric comparable across
    # context offsets, since frame24_* measures a horizon that shrinks as the
    # window slides while this one is pinned to an absolute instant.
    "catch_position",
    "catch_position_inplane",
    "catch_position_axial",
    "catch_horizon_s",
)
# Their p95 sub-key is the 95th percentile across scenes, as for frame24_position.
_P95_ACROSS_SCENES = ("frame24_position",) + SIDE_METRIC_NAMES

FINAL_TEST_SENTINEL_DIR = str(
    WORKTREE / "data" / "SLARM_data" / ".stream25_final_test_registry"
)


def set_evaluation_seed(seed: int) -> int:
    """Match rank-0 training initialization before constructing new heads."""
    from src.utils.misc import fix_random_seeds

    seed = int(seed)
    fix_random_seeds(seed)
    return seed


def check_final_test_sentinel(sentinel_dir: str, experiment_hash: str) -> None:
    """Atomically create sentinel; refuse if experiment hash already exists."""
    os.makedirs(sentinel_dir, exist_ok=True)
    sentinel_path = os.path.join(sentinel_dir, f"{experiment_hash}.json")
    fd = os.open(sentinel_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)
    with open(sentinel_path, "w") as f:
        json.dump({"started": True, "experiment_hash": experiment_hash}, f)


def apply_acceptance_gates(
    metrics: Dict[str, Any],
    acceptance_table: Dict[str, Dict] = ACCEPTANCE_TABLE,
    *,
    valid_counts: Optional[Dict[str, Dict[str, int]]] = None,
    minimum_valid_count: int = 1,
) -> Dict[str, Any]:
    """Apply one scope's frozen gates without converting N/A values to zero."""
    if minimum_valid_count < 1:
        raise ValueError("minimum_valid_count must be positive")
    ratios = []
    all_pass = True
    missing_gates = []
    gate_results = {}

    for gate_name, thresholds in acceptance_table.items():
        if gate_name not in metrics:
            missing_gates.append(gate_name)
            ratios.append(float("inf"))
            all_pass = False
            continue
        metric_val = metrics[gate_name]
        if not isinstance(metric_val, dict):
            metric_val = {"median": metric_val}
        upper = not gate_name.startswith(("rgb", "semantic", "ball_iou", "ball_rgb"))
        for subkey, limit in thresholds.items():
            gate_id = f"{gate_name}.{subkey}"
            if subkey not in metric_val:
                missing_gates.append(gate_id)
                ratios.append(float("inf"))
                gate_results[gate_id] = "MISSING"
                all_pass = False
                continue
            val = metric_val[subkey]
            count = None
            if valid_counts is not None:
                count = valid_counts.get(gate_name, {}).get(subkey)
                if not isinstance(count, int) or count < minimum_valid_count:
                    missing_gates.append(gate_id)
                    ratios.append(float("inf"))
                    gate_results[gate_id] = {
                        "status": "INSUFFICIENT_SAMPLES",
                        "value": val,
                        "limit": limit,
                        "valid_count": count,
                        "minimum_valid_count": minimum_valid_count,
                        "passed": False,
                        "ratio": float("inf"),
                    }
                    all_pass = False
                    continue
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                ratios.append(float("inf"))
                gate_results[gate_id] = "NONFINITE"
                all_pass = False
                continue
            passed = val <= limit if upper else val >= limit
            ratio = worst_normalized_ratio(val, limit, upper_bound=upper)
            ratios.append(ratio)
            gate_results[gate_id] = {
                "status": "PASS" if passed else "FAIL",
                "value": val,
                "limit": limit,
                "valid_count": count,
                "passed": passed,
                "ratio": ratio,
            }
            all_pass = all_pass and passed

    worst = max(ratios) if ratios else float("inf")
    return {
        "all_gates_pass": all_pass and not missing_gates,
        "worst_ratio": worst,
        "missing_gates": missing_gates,
        "gates": gate_results,
    }


def apply_scoped_acceptance_gates(
    scope_metrics: Dict[str, Dict[str, Any]],
    scope_valid_counts: Dict[str, Dict[str, Dict[str, int]]],
    acceptance_table: Dict[str, Dict] = ACCEPTANCE_TABLE,
    *,
    minimum_valid_count: int = 1,
) -> Dict[str, Any]:
    """Require aggregate and every named view to pass independently."""
    required_scopes = get_required_eval_scopes()
    if tuple(scope_metrics) != required_scopes:
        raise ValueError(
            "Stream25 scope order must be "
            f"{required_scopes}, got {tuple(scope_metrics)}"
        )
    if tuple(scope_valid_counts) != required_scopes:
        raise ValueError(
            "Stream25 valid-count scope order must be "
            f"{required_scopes}, got {tuple(scope_valid_counts)}"
        )

    scope_reports = {}
    combined_gates = {}
    combined_missing = []
    ratios = []
    for scope in required_scopes:
        report = apply_acceptance_gates(
            scope_metrics[scope],
            acceptance_table,
            valid_counts=scope_valid_counts[scope],
            minimum_valid_count=minimum_valid_count,
        )
        scope_reports[scope] = {
            "metrics": scope_metrics[scope],
            "valid_counts": scope_valid_counts[scope],
            **report,
        }
        combined_gates.update(
            {f"{scope}.{name}": value for name, value in report["gates"].items()}
        )
        combined_missing.extend(
            f"{scope}.{name}" for name in report["missing_gates"]
        )
        ratios.append(report["worst_ratio"])

    all_pass = all(
        report["all_gates_pass"] for report in scope_reports.values()
    )
    return {
        "scope_reports": scope_reports,
        "gates": combined_gates,
        "missing_gates": combined_missing,
        "all_gates_pass": all_pass,
        "worst_ratio": max(ratios) if ratios else float("inf"),
    }


def rendered_ball_positions_per_view(
    depth: torch.Tensor,
    semantic: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    *,
    ball_surface_offset: float = 0.0,
) -> List[Optional[torch.Tensor]]:
    """One rig-frame ball centre per view at a single rendered frame, or None.

    This is the same read the frame-24 metric performs at frame 15 -- predicted
    semantic picks the ball pixels, the rendered depth is back-projected, and the
    per-pixel median pools them -- factored out so the earlier context frames can
    be read the same way.
    """
    positions = ray_origins + ray_directions * depth[..., None]
    positions = apply_ball_surface_offset(positions, ray_directions, ball_surface_offset)
    out: List[Optional[torch.Tensor]] = []
    for eye in range(depth.shape[0]):
        mask = (
            (semantic[eye] == 1)
            & torch.isfinite(depth[eye])
            & (depth[eye] > 0)
            & torch.isfinite(positions[eye]).all(dim=-1)
        )
        if not mask.any():
            out.append(None)
            continue
        out.append(transform_position(
            positions[eye][mask].median(dim=0).values, canonical_to_rig))
    return out


def compute_rendered_history_fit_metrics(
    pred_depth: torch.Tensor,
    pred_sem: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    target_time: torch.Tensor,
    gt_pos24: torch.Tensor,
    gt_positions: Optional[torch.Tensor],
    gt_v15: Optional[torch.Tensor],
    *,
    dt: float,
    timespan: float,
    ball_surface_offset: float = 0.0,
    fit_frames: Optional[Sequence[int]] = None,
    context_offset: int = 0,
) -> Dict[str, float]:
    """Refit (pos15, v15) from the ball's rendered position at several frames.

     Why this is not circular. terminal_context_extrapolation makes frame 15 the
      sole owner of targets at or after frame 15 (slarm.py clears the earlier
      context frames there), but targets *before* frame 15 are still rendered by
      the nearby context frames' own Gaussians under time_mask_backward. So the
      ball's position at frames 0..12 is an observation, not frame 15's velocity
      integrated backwards, and fitting a velocity to them learns something the
      MS3 head did not already assert.

     Why it can beat the MS3 head. A position error component that is constant
      across the six frames cancels exactly out of the fitted velocity, so only
      the frame-to-frame scatter matters. pixel_pos_error_constant_m and
      pixel_pos_error_scatter_m split the measured error that way; the ratio
      between them is what decides how much this can win, and it is reported
      rather than assumed.

    Conservative pooling matches frame24_position: fit each view separately and
    keep the worst finite error, so the two numbers are directly comparable.
    """
    out: Dict[str, float] = {}
    frames = list(STREAM25_CONTEXT_FRAMES)
    if pred_depth.shape[0] <= frames[-1]:
        return out
    gravity = gt_pos24.new_tensor(MS3_GRAVITY_RIG)
    times = torch.tensor(
        [float(target_time[0, frame, 0].item()) for frame in frames],
        dtype=torch.float64,
    )
    times = ((times - times[-1]) * float(timespan)).float()

    per_frame: List[List[Optional[torch.Tensor]]] = [
        rendered_ball_positions_per_view(
            pred_depth[frame], pred_sem[frame],
            ray_origins[frame], ray_directions[frame], canonical_to_rig,
            ball_surface_offset=ball_surface_offset,
        )
        for frame in frames
    ]

    # Per-frame position error, and its split into a constant offset and the
    # frame-to-frame scatter. Only the scatter propagates into a fitted velocity.
    if gt_positions is not None:
        residuals: List[torch.Tensor] = []
        for index, frame in enumerate(frames):
            finite = [p for p in per_frame[index] if p is not None and torch.isfinite(p).all()]
            if not finite:
                continue
            truth = gt_positions[frame]
            worst = max(float((p - truth).norm().item()) for p in finite)
            if math.isfinite(worst):
                out[f"pixel_pos_error_frame{frame}"] = worst
            residuals.append(torch.stack(finite).mean(dim=0) - truth)
        if len(residuals) >= 2:
            stacked = torch.stack(residuals)
            constant = stacked.mean(dim=0)
            out["pixel_pos_error_constant_m"] = float(constant.norm().item())
            out["pixel_pos_error_scatter_m"] = float(
                (stacked - constant).norm(dim=-1).mean().item())

    selected = [frames.index(f) for f in fit_frames] if fit_frames else list(range(len(frames)))
    if len(selected) < 2:
        return out

    errors, vel_errors, pos_errors = [], [], []
    for eye in range(pred_depth.shape[1]):
        usable = [i for i in selected
                  if per_frame[i][eye] is not None and torch.isfinite(per_frame[i][eye]).all()]
        if len(usable) < 2:
            continue
        fitted = fit_ballistic_state(
            torch.stack([per_frame[i][eye] for i in usable]), times[usable], gravity)
        pos15, v15 = fitted[:3], fitted[3:]
        predicted = integrate_frame24_position_physics(pos15, v15, dt, gravity)
        errors.append(float((predicted - gt_pos24).norm().item()))
        if gt_positions is not None:
            pos_errors.append(float((pos15 - gt_positions[frames[-1]]).norm().item()))
        if gt_v15 is not None:
            vel_errors.append(float((v15 - gt_v15).norm().item()))
    for name, values in (("frame24_position_fit", errors),
                         ("ball_pos15_error_fit", pos_errors),
                         ("ball_vel15_error_fit", vel_errors)):
        finite = [v for v in values if math.isfinite(v)]
        if finite:
            out[name] = max(finite)
    return out


def compute_rendered_frame24_position_errors(
    depth15: torch.Tensor,
    semantic15: torch.Tensor,
    ms3_15: torch.Tensor,
    ray_origins15: torch.Tensor,
    ray_directions15: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    gt_pos24: torch.Tensor,
    *,
    dt: float,
    ball_surface_offset: float = 0.0,
) -> List[float]:
    """Return one rendered frame-24 position error per named view.

    ``ball_surface_offset`` in metres pushes the unprojected point from the ball's
    near surface to its centre. It changes what frame24_position means, so it is
    off by default and must be asked for.
    """
    predicted = rendered_landing_positions_per_view(
        depth15, semantic15, ms3_15, ray_origins15, ray_directions15,
        canonical_to_rig, dt=dt, ball_surface_offset=ball_surface_offset,
    )
    return [
        float("nan") if p is None else float((p - gt_pos24).norm().item())
        for p in predicted
    ]


def rendered_landing_positions_per_view(
    depth15: torch.Tensor,
    semantic15: torch.Tensor,
    ms3_15: torch.Tensor,
    ray_origins15: torch.Tensor,
    ray_directions15: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    *,
    dt: float,
    ball_surface_offset: float = 0.0,
) -> List[Optional[torch.Tensor]]:
    """Predicted landing position per named view, or None where no ball rendered.

    The landing ERROR is what the gates read, but a norm cannot be split into
    the part that makes the ball miss the ring and the part that only makes it
    arrive early. Returning the position keeps that decomposition available to
    the caller without changing how the error is computed.
    """
    positions15 = ray_origins15 + ray_directions15 * depth15[..., None]
    positions15 = apply_ball_surface_offset(
        positions15, ray_directions15, ball_surface_offset
    )
    out: List[Optional[torch.Tensor]] = []
    for eye in range(depth15.shape[0]):
        mask = (
            (semantic15[eye] == 1)
            & torch.isfinite(depth15[eye])
            & (depth15[eye] > 0)
            & torch.isfinite(ms3_15[eye]).all(dim=-1)
            & torch.isfinite(positions15[eye]).all(dim=-1)
        )
        if not mask.any():
            out.append(None)
            continue
        pos = transform_position(
            positions15[eye][mask].median(dim=0).values,
            canonical_to_rig,
        )
        parts = [
            transform_vector(
                ms3_15[eye, ..., offset : offset + 3][mask]
                .median(dim=0)
                .values,
                canonical_to_rig,
            )
            for offset in (0, 3, 6)
        ]
        out.append(integrate_frame24_position(pos, *parts, dt))
    return out


def compute_rendered_frame24_position_error(
    depth15: torch.Tensor,
    semantic15: torch.Tensor,
    ms3_15: torch.Tensor,
    ray_origins15: torch.Tensor,
    ray_directions15: torch.Tensor,
    canonical_to_rig: torch.Tensor,
    gt_pos24: torch.Tensor,
    *,
    dt: float,
    ball_surface_offset: float = 0.0,
) -> float:
    """Return the conservative worst finite named-view frame-24 error."""
    errors = compute_rendered_frame24_position_errors(
        depth15,
        semantic15,
        ms3_15,
        ray_origins15,
        ray_directions15,
        canonical_to_rig,
        gt_pos24,
        dt=dt,
        ball_surface_offset=ball_surface_offset,
    )
    finite = [value for value in errors if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def _aggregate_scene_scope_metrics(
    scene_scopes: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Dict[str, int]]]:
    metric_names = {
        name
        for scope in scene_scopes
        for name in scope.get("metrics", {})
    }
    metrics: Dict[str, Any] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for metric_name in sorted(metric_names):
        subkeys = {
            subkey
            for scope in scene_scopes
            for subkey in scope.get("metrics", {}).get(metric_name, {})
        }
        metrics[metric_name] = {}
        counts[metric_name] = {}
        for subkey in sorted(subkeys):
            values = [
                scope.get("metrics", {}).get(metric_name, {}).get(subkey)
                for scope in scene_scopes
            ]
            finite_values = [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and math.isfinite(value)
            ]
            percentile = (
                95
                if metric_name in _P95_ACROSS_SCENES and subkey == "p95"
                else 50
            )
            metrics[metric_name][subkey] = finite_percentile(
                finite_values, percentile
            )
            counts[metric_name][subkey] = sum(
                int(scope.get("valid_counts", {}).get(metric_name, {}).get(subkey, 0))
                for scope in scene_scopes
            )

    if "rgb_psnr" in metrics:
        metrics["rgb_psnr_p10"] = {}
        counts["rgb_psnr_p10"] = {}
        for bucket in metrics["rgb_psnr"]:
            values = [
                scope.get("metrics", {}).get("rgb_psnr", {}).get(bucket)
                for scope in scene_scopes
            ]
            metrics["rgb_psnr_p10"][bucket] = finite_percentile(
                [
                    float(value)
                    for value in values
                    if isinstance(value, (int, float)) and math.isfinite(value)
                ],
                10,
            )
            counts["rgb_psnr_p10"][bucket] = counts["rgb_psnr"][bucket]
    return metrics, counts


def summarize_stream25_scene_results(
    scene_results: List[Dict[str, Any]],
    *,
    acceptance_table: Dict[str, Dict] = ACCEPTANCE_TABLE,
) -> Dict[str, Any]:
    """Aggregate scenes into independent aggregate and named-view gate tables."""
    if not scene_results:
        raise ValueError("Stream25 evaluation requires at least one scene")
    scope_metrics: Dict[str, Dict[str, Any]] = {}
    scope_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    required_scopes = get_required_eval_scopes()
    for scope in required_scopes:
        scene_scopes = []
        for scene in scene_results:
            scopes = scene.get("scopes")
            if not isinstance(scopes, dict) or tuple(scopes) != required_scopes:
                raise ValueError(
                    "each scene must contain aggregate and three named-view scopes"
                )
            scene_scopes.append(scopes[scope])
        metrics, counts = _aggregate_scene_scope_metrics(scene_scopes)
        scope_metrics[scope] = metrics
        scope_counts[scope] = counts
    return apply_scoped_acceptance_gates(
        scope_metrics,
        scope_counts,
        acceptance_table,
    )


def evaluate_scene(
    model,
    data_dict,
    device,
    timespan: float = 0.8,
    *,
    ball_surface_offset: float = 0.0,
    fit_frames: Optional[Sequence[int]] = None,
    context_offset: int = 0,
    catch_frame: int = 0,
) -> Dict[str, Any]:
    """Evaluate one scene through a fresh StreamSession and return per-bucket metrics."""
    from src.models.stream_session import StreamSession

    session = StreamSession(model, mode="window", window_size=6)
    inference_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for obs_idx in range(6):
        obs_dict = _extract_observation(data_dict, obs_idx)
        session.forward_stream(obs_dict, device, inference_dtype)

    predictions = session.get_all_predictions()
    target_rays = model.plucker_embedder(
        data_dict["target_intrinsics"],
        data_dict["target_camtoworlds"],
        image_size=data_dict["target_image"].shape[-2:],
    )
    metrics = compute_stream25_scene_metrics(
        predictions,
        data_dict,
        timespan,
        target_ray_origins=target_rays["origins"],
        target_ray_directions=target_rays["dirs"],
        ball_surface_offset=ball_surface_offset,
        fit_frames=fit_frames,
        context_offset=context_offset,
        catch_frame=catch_frame,
    )
    return metrics


def _extract_observation(data_dict: Dict, obs_idx: int) -> Dict:
    from tools.stream25_runtime import slice_stream_observation
    return slice_stream_observation(data_dict, obs_idx)


def _summarize_scene_scope(
    records: List[Dict[str, Any]],
    context_values_by_view: Dict[str, List[List[float]]],
    frame24_errors: List[float],
    eye_indices: tuple[int, ...],
    history_fit_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    selected = [record for record in records if record["eye"] in eye_indices]
    metrics: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    scalar_names = (
        "rgb_psnr",
        "rgb_ssim",
        "ball_rgb_psnr",
        "depth_absrel",
        "depth_rmse",
        "semantic_miou",
        "semantic_dice",
        "ball_iou",
    )
    for name in scalar_names:
        metrics[name] = {}
        counts[name] = {}
        for bucket, frames in TIME_BUCKETS.items():
            values = [
                float(record[name])
                for record in selected
                if record["frame"] in frames
                and record.get(name) is not None
                and math.isfinite(record[name])
            ]
            metrics[name][bucket] = (
                sum(values) / len(values) if values else float("nan")
            )
            counts[name][bucket] = len(values)

    for suffix, percentile in (("median", 50), ("p95", 95)):
        metric_name = f"ball_depth_error_{suffix}"
        metrics[metric_name] = {}
        counts[metric_name] = {}
        for bucket, frames in TIME_BUCKETS.items():
            values = [
                value
                for record in selected
                if record["frame"] in frames
                for value in record["ball_depth_errors"]
                if math.isfinite(value)
            ]
            metrics[metric_name][bucket] = finite_percentile(values, percentile)
            counts[metric_name][bucket] = len(values)

    for prefix in ("ball", "static"):
        for component in ("velocity", "acceleration", "jerk"):
            name = f"ms3_{prefix}_{component}"
            values = [
                value
                for record in selected
                for value in record[name]
                if math.isfinite(value)
            ]
            metrics[name] = {
                "median": finite_percentile(values, 50),
                "p95": finite_percentile(values, 95),
            }
            counts[name] = {"median": len(values), "p95": len(values)}

            context_name = f"context_{name}"
            context_values = [
                value
                for eye in eye_indices
                for value in context_values_by_view[context_name][eye]
                if math.isfinite(value)
            ]
            metrics[context_name] = {
                "median": finite_percentile(context_values, 50),
                "p95": finite_percentile(context_values, 95),
            }
            counts[context_name] = {
                "median": len(context_values),
                "p95": len(context_values),
            }

    finite_frame24 = [
        frame24_errors[eye]
        for eye in eye_indices
        if math.isfinite(frame24_errors[eye])
    ]
    conservative_frame24 = max(finite_frame24) if finite_frame24 else float("nan")
    metrics["frame24_position"] = {
        "median": conservative_frame24,
        "p95": conservative_frame24,
    }
    frame24_sample_count = int(bool(finite_frame24))
    counts["frame24_position"] = {
        "median": frame24_sample_count,
        "p95": frame24_sample_count,
    }

    # One value per scene, independent of view, so every scope records the same
    # number; the percentile spread only appears when scenes are aggregated.
    combined_scalar_metrics = dict(history_fit_metrics or {})
    for name in SIDE_METRIC_NAMES:
        value = combined_scalar_metrics.get(name)
        if value is None or not math.isfinite(value):
            continue
        metrics[name] = {"median": value, "p95": value}
        counts[name] = {"median": 1, "p95": 1}

    return {"metrics": metrics, "valid_counts": counts}


def compute_stream25_scene_metrics(
    predictions: Dict,
    data_dict: Dict,
    timespan: float,
    *,
    target_ray_origins: torch.Tensor,
    target_ray_directions: torch.Tensor,
    ball_surface_offset: float = 0.0,
    fit_frames: Optional[Sequence[int]] = None,
    context_offset: int = 0,
    catch_frame: int = 0,
) -> Dict[str, Any]:
    render = predictions["render_results"]
    pred_rgb = render["rendered_image"][0].float().cpu()
    pred_depth = render["rendered_depth"][0].float().cpu()
    pred_sem = render["rendered_task_semantic"][0].long().cpu()
    pred_ms3 = render["rendered_target_ms3"][0].float().cpu()
    gt_rgb = data_dict["target_image"][0].permute(0, 1, 3, 4, 2).float().cpu()
    gt_depth = data_dict["target_depth"][0].float().cpu()
    gt_sem = data_dict["task_semantic"][0].long().cpu()
    gt_ms3 = data_dict["dense_ms3_gt"][0].float().cpu()
    ball_mask = data_dict["ball_ms3_mask"][0].bool().cpu()
    static_mask = data_dict["static_ms3_mask"][0].bool().cpu()
    dilated_ball_mask = torch.nn.functional.max_pool2d(
        ball_mask.float().reshape(-1, 1, *ball_mask.shape[-2:]),
        kernel_size=11,
        stride=1,
        padding=5,
    ).reshape_as(ball_mask).bool()
    num_views = pred_rgb.shape[1]
    camera_order = get_camera_order()
    if num_views != len(camera_order):
        raise ValueError(
            f"Stream25 evaluator expects {len(camera_order)} named views "
            f"{camera_order}, got {num_views}. Call set_camera_order() with the "
            f"camera_list entry for this config's num_max_cameras."
        )
    view_tensors = (
        pred_depth,
        pred_sem,
        pred_ms3,
        gt_rgb,
        gt_depth,
        gt_sem,
        gt_ms3,
        ball_mask,
        static_mask,
    )
    if any(tensor.shape[1] != num_views for tensor in view_tensors):
        raise ValueError("Stream25 evaluator received inconsistent target view axes")

    # Offset k leaves 25-k targets, so loop over what actually arrived; a hardcoded
    # 25 runs off the end. The count is still checked, to catch other truncation.
    num_targets = int(gt_depth.shape[0])
    expected_targets = len(STREAM25_ALL_TARGET_FRAMES) - int(context_offset)
    if num_targets != expected_targets:
        raise ValueError(
            f"Stream25 evaluator received {num_targets} targets; a window at "
            f"offset +{context_offset} must render {expected_targets}"
        )

    records: List[Dict[str, Any]] = []
    for frame in range(num_targets):
        for eye in range(num_views):
            valid_depth = torch.isfinite(gt_depth[frame, eye]) & (gt_depth[frame, eye] > 0)
            ball = ball_mask[frame, eye]
            dilated = dilated_ball_mask[frame, eye]
            rec = {
                "frame": frame,
                "eye": eye,
                "view": camera_order[eye],
                "ball_visible": bool(ball.any()),
                "rgb_psnr": compute_psnr(pred_rgb[frame, eye].permute(2, 0, 1), gt_rgb[frame, eye].permute(2, 0, 1)),
                "rgb_ssim": compute_ssim(pred_rgb[frame, eye].permute(2, 0, 1), gt_rgb[frame, eye].permute(2, 0, 1)),
                "depth_absrel": compute_depth_absrel(pred_depth[frame, eye], gt_depth[frame, eye]),
                "depth_rmse": compute_depth_rmse(pred_depth[frame, eye], gt_depth[frame, eye]),
                "semantic_miou": compute_miou(pred_sem[frame, eye], gt_sem[frame, eye]),
                "semantic_dice": compute_macro_dice(pred_sem[frame, eye], gt_sem[frame, eye]),
                "ball_iou": compute_ball_region_iou(
                    pred_sem[frame, eye], gt_sem[frame, eye]
                ),
            }
            rec["ball_rgb_psnr"] = (
                compute_psnr(pred_rgb[frame, eye][dilated], gt_rgb[frame, eye][dilated])
                if dilated.any() else None
            )
            rec["ball_depth_errors"] = (
                (pred_depth[frame, eye][ball & valid_depth] - gt_depth[frame, eye][ball & valid_depth]).abs().tolist()
                if (ball & valid_depth).any() else []
            )
            for cls, name in enumerate(("background", "ball", "floor", "obstacle")):
                rec[f"{name}_iou"] = (
                    compute_ball_region_iou(
                        pred_sem[frame, eye], gt_sem[frame, eye]
                    )
                    if cls == 1
                    else compute_iou(
                        pred_sem[frame, eye], gt_sem[frame, eye], cls
                    )
                )
            for mask, prefix in ((ball, "ball"), (static_mask[frame, eye], "static")):
                for offset, component in ((0, "velocity"), (3, "acceleration"), (6, "jerk")):
                    rec[f"ms3_{prefix}_{component}"] = (
                        (pred_ms3[frame, eye, ..., offset:offset + 3][mask]
                         - gt_ms3[frame, eye, ..., offset:offset + 3][mask]).norm(dim=-1).tolist()
                        if mask.any() else []
                    )
            records.append(rec)

    context_pred_ms3 = predictions["gs_params"]["forward_ms3"][0].float().cpu()
    context_gt_ms3 = data_dict["context_dense_ms3_gt"][0].float().cpu()
    if context_pred_ms3.shape[1] != num_views or context_gt_ms3.shape[1] != num_views:
        raise ValueError("Stream25 evaluator received inconsistent context view axes")
    context_values_by_view: Dict[str, List[List[float]]] = {}
    for prefix in ("ball", "static"):
        context_mask = data_dict[f"context_{prefix}_ms3_mask"][0].bool().cpu()
        if context_mask.shape[1] != num_views:
            raise ValueError("Stream25 evaluator received inconsistent context masks")
        for offset, component in ((0, "velocity"), (3, "acceleration"), (6, "jerk")):
            name = f"context_ms3_{prefix}_{component}"
            context_values_by_view[name] = []
            for eye in range(num_views):
                mask = context_mask[:, eye]
                values = (
                    (
                        context_pred_ms3[:, eye, ..., offset : offset + 3][mask]
                        - context_gt_ms3[:, eye, ..., offset : offset + 3][mask]
                    )
                    .norm(dim=-1)
                    .tolist()
                    if mask.any()
                    else []
                )
                context_values_by_view[name].append(values)

    # Landing target. At offset 0 this is the frozen frame 24 and nothing below
    # changes. A slid window eats targets off the end of range(25), so index the
    # target list that actually arrived and convert to the absolute frame that
    # ball_position_rig is indexed by.
    landing_index = min(
        STREAM25_ALL_TARGET_FRAMES[-1], int(data_dict["target_time"].shape[1]) - 1
    )
    gt_pos24 = data_dict["ball_position_rig"][0, landing_index].float().cpu()
    canonical_to_rig = data_dict["context_canonical_to_rig"][0, -1].float().cpu()
    dt = float(
        (
            data_dict["target_time"][0, landing_index, 0]
            - data_dict["context_time"][0, -1, 0]
        ).item()
        * timespan
    )
    # At offset 9 the last stored frame IS the terminal observation, so
    # frame24_position would degrade into "error at the terminal frame" -- a
    # smaller number that reads like the sliding window helped. Emit nothing and
    # let the gate report MISSING rather than publish a flattering wrong value.
    if dt <= 0:
        print(f"[eval] frame24_* skipped: the window at offset +{context_offset} "
              f"leaves no extrapolation to the last stored frame "
              f"(landing index {landing_index} is the terminal observation). "
              f"catch_position is unaffected.", flush=True)
    frame24_errors = [] if dt <= 0 else compute_rendered_frame24_position_errors(
        pred_depth[15],
        pred_sem[15],
        pred_ms3[15],
        target_ray_origins[0, 15].float().cpu(),
        target_ray_directions[0, 15].float().cpu(),
        canonical_to_rig,
        gt_pos24,
        dt=dt,
        ball_surface_offset=ball_surface_offset,
    )

    # The pixel-path multi-frame fit; every checkpoint can compute it.
    _gt_positions = data_dict.get("ball_position_rig")
    history_fit_metrics = compute_rendered_history_fit_metrics(
        pred_depth, pred_sem,
        target_ray_origins[0].float().cpu(),
        target_ray_directions[0].float().cpu(),
        canonical_to_rig,
        data_dict["target_time"],
        gt_pos24,
        None if _gt_positions is None else _gt_positions[0].float().cpu(),
        # Terminal GT velocity. ball_velocity_rig is indexed by ABSOLUTE frame,
        # so a literal 15 would score the slid window's terminal state against
        # the wrong instant -- quietly, with a plausible-looking number.
        None if data_dict.get("ball_velocity_rig") is None
        else data_dict["ball_velocity_rig"][0, TERMINAL_TARGET_INDEX].float().cpu(),
        dt=dt, timespan=timespan,
        ball_surface_offset=ball_surface_offset,
        fit_frames=fit_frames,
        context_offset=context_offset,
    )


    # ---- catch-frame landing: the only number comparable across window offsets --
    # frame24_* extrapolates from the terminal observation to a fixed TARGET, so
    # its horizon shrinks as the window slides and the numbers stop meaning the
    # same thing. The catch frame is fixed in absolute time, so the horizon
    # (catch_frame - terminal) is exactly what sliding the window buys, and every
    # configuration is scored against the same instant.
    #
    # GT at the catch frame is the analytic continuation of the GT terminal state.
    # That is exact here, not an approximation, because the simulator has no air
    # drag. Real drag would add roughly 10 cm over 1 s at this ball speed, so the
    # day the simulator gains drag this continuation has to be replaced by stored
    # GT (see docs/EXPERIMENTS_AND_ERROR_BUDGET.md).
    catch_metrics: Dict[str, float] = {}
    # Two different things: TERMINAL_TARGET_INDEX is a tensor index, always 15;
    # terminal_frame is the absolute frame, 15 + offset, for time arithmetic only.
    terminal_frame = STREAM25_CONTEXT_FRAMES[-1] + int(context_offset)
    _gt_v_all = data_dict.get("ball_velocity_rig")
    if catch_frame and _gt_positions is not None and _gt_v_all is not None:
        span = STREAM25_ALL_TARGET_FRAMES[-1] - STREAM25_ALL_TARGET_FRAMES[0]
        catch_dt = (int(catch_frame) - terminal_frame) * float(timespan) / float(span)
        gravity = torch.tensor(MS3_GRAVITY_RIG, dtype=torch.float32)
        gt_catch = integrate_frame24_position_physics(
            _gt_positions[0, TERMINAL_TARGET_INDEX].float().cpu(),
            _gt_v_all[0, TERMINAL_TARGET_INDEX].float().cpu(),
            catch_dt, gravity,
        )
        catch_metrics["catch_horizon_s"] = catch_dt
        predicted = rendered_landing_positions_per_view(
            pred_depth[15], pred_sem[15], pred_ms3[15],
            target_ray_origins[0, 15].float().cpu(),
            target_ray_directions[0, 15].float().cpu(),
            canonical_to_rig, dt=catch_dt,
            ball_surface_offset=ball_surface_offset,
        )
        finite = [p for p in predicted if p is not None and torch.isfinite(p).all()]
        if finite:
            catch_metrics["catch_position"] = max(
                float((p - gt_catch).norm().item()) for p in finite
            )

        # ---- the component that actually decides whether the ball goes in ----
        #
        # The ring is placed at the predicted point and waits, and it faces the
        # incoming ball, so its axis is the ball's velocity at the catch. Split
        # the error against that axis:
        #
        #   along  the axis  -> the true trajectory still passes through the ring
        #                       centre, just early or late. Not a miss.
        #   across the axis  -> the ball crosses the ring plane off-centre by
        #                       this much. This is the miss.
        #
        # catch_position uses the full 3D norm and therefore counts axial error
        # as if it were lateral, which makes any success rate read off it a
        # LOWER bound. catch_position_inplane is the geometrically honest one.
        # Their ratio also says which way the error points, which is checkable
        # against the error budget's claim that it is mostly along the view ray.
        #
        # Gravity is known and the flight is ballistic, so the catch-time
        # velocity comes from the terminal one without another prediction.
        velocity_catch = _gt_v_all[0, TERMINAL_TARGET_INDEX].float().cpu() + gravity * catch_dt
        speed = float(velocity_catch.norm().item())
        if finite and math.isfinite(speed) and speed > 1e-6:
            axis = velocity_catch / speed
            lateral, axial = [], []
            for p in finite:
                error = p - gt_catch
                along = float((error * axis).sum().item())
                axial.append(abs(along))
                lateral.append(float((error - along * axis).norm().item()))
            catch_metrics["catch_position_inplane"] = max(lateral)
            catch_metrics["catch_position_axial"] = max(axial)
    # Flat scalar dicts are merged wholesale at _summarize_scene_scope, so folding
    # the catch metrics in here needs no new plumbing.
    history_fit_metrics.update(catch_metrics)

    scope_eye_indices = {
        "aggregate": tuple(range(num_views)),
        **{
            camera_name: (eye,)
            for eye, camera_name in enumerate(camera_order)
        },
    }
    scopes = {
        scope: _summarize_scene_scope(
            records,
            context_values_by_view,
            frame24_errors,
            eye_indices,
            history_fit_metrics,
        )
        for scope, eye_indices in scope_eye_indices.items()
    }
    aggregate_metrics = scopes["aggregate"]["metrics"]
    bucket_metric_names = (
        "rgb_psnr",
        "rgb_ssim",
        "ball_rgb_psnr",
        "depth_absrel",
        "depth_rmse",
        "semantic_miou",
        "semantic_dice",
        "ball_iou",
        "ball_depth_error_median",
        "ball_depth_error_p95",
    )
    bucket_metrics = {
        bucket: {
            metric_name: aggregate_metrics[metric_name][bucket]
            for metric_name in bucket_metric_names
        }
        for bucket in TIME_BUCKETS
    }
    ms3_names = (
        "ms3_ball_velocity",
        "ms3_ball_acceleration",
        "ms3_ball_jerk",
        "ms3_static_velocity",
        "ms3_static_acceleration",
        "ms3_static_jerk",
    )
    scene_result = {
        "records": records,
        "scopes": scopes,
        "buckets": bucket_metrics,
        "ms3": {name: aggregate_metrics[name] for name in ms3_names},
        "context_ms3": {
            f"context_{name}": aggregate_metrics[f"context_{name}"]
            for name in ms3_names
        },
        "frame24_position_error": aggregate_metrics["frame24_position"]["median"],
        "considered_frame_eyes": len(records),
        "visible_ball_frame_eyes": sum(r["ball_visible"] for r in records),
    }
    return scene_result


def _compact_scene_result(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Keep per-frame/eye evidence without serializing every selected pixel error."""
    compact = dict(scene)
    compact_records = []
    for record in scene["records"]:
        item = dict(record)
        depth_errors = item.pop("ball_depth_errors")
        item["ball_depth_valid_count"] = len(depth_errors)
        item["ball_depth_error_median"] = finite_percentile(depth_errors, 50)
        item["ball_depth_error_p95"] = finite_percentile(depth_errors, 95)
        for prefix in ("ball", "static"):
            for component in ("velocity", "acceleration", "jerk"):
                key = f"ms3_{prefix}_{component}"
                values = item.pop(key)
                item[f"{key}_valid_count"] = len(values)
                item[f"{key}_median"] = finite_percentile(values, 50)
        compact_records.append(item)
    compact["records"] = compact_records
    return compact


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _single_sample_collate(batch):
    """Return the single sample unchanged; batch_size is 1.

    Reading happens in worker processes while the main process collates and moves
    to device, so IO overlaps the GPU without changing any number."""
    return batch[0]


def _render_failing_gates(result: Dict[str, Any]) -> List[str]:
    """List the gates that actually made overall FAIL.

    The metrics table above shows only the aggregate scope while gates are judged
    per camera too, so a report can read all-green and still FAIL. Two of the three
    failure kinds say nothing about model quality: INSUFFICIENT_SAMPLES means that
    scope never saw the ball in that bucket, which more training cannot fix, and
    NONFINITE is usually a division by zero upstream.
    """
    lines: List[str] = []
    for gate_id, rec in (result.get("gates") or {}).items():
        if rec == "NONFINITE":
            lines.append(f"- `{gate_id}` - **NONFINITE** (metric is NaN or inf)")
            continue
        if not isinstance(rec, dict) or rec.get("passed", True):
            continue
        status = rec.get("status", "FAIL")
        if status == "INSUFFICIENT_SAMPLES":
            lines.append(
                f"- `{gate_id}` - **INSUFFICIENT_SAMPLES**"
                f" ({rec.get('valid_count')} valid < {rec.get('minimum_valid_count')} required)"
                " - this view cannot see the ball in this bucket; training will not fix it"
            )
        else:
            lines.append(
                f"- `{gate_id}` - **{status}**: {rec.get('value'):.4f} "
                f"vs limit {rec.get('limit'):.4f}"
            )
    if not lines:
        lines.append(
            "(No gate failed. If Overall is still FAIL, check `missing_gates`: "
            "a gate in ACCEPTANCE_TABLE whose metric was never computed also counts as a loss.)"
        )
    missing = result.get("missing_gates") or []
    if missing:
        lines += ["", f"`missing_gates` ({len(missing)}): " +
                  ", ".join(f"`{m}`" for m in missing[:20]) +
                  (" ..." if len(missing) > 20 else "")]
    return lines


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    split: str = "validation",
    *,
    selection_report: Optional[Dict] = None,
    output_json: Optional[str] = None,
    output_markdown: Optional[str] = None,
    device: str = "cuda",
    reference: bool = False,
    render_chunk: Optional[int] = None,
    num_workers: int = 8,
    ball_radius_compensation: float = 0.0,
    ball_radius: Optional[float] = None,
    fit_frames: Optional[Sequence[int]] = None,
    context_offset: int = 0,
) -> Dict[str, Any]:
    """Run full evaluation on a split. For final-test, require a selection report."""
    if split == "final-test":
        if selection_report is None:
            raise ValueError("final-test evaluation requires a frozen selection report")
        if selection_report.get("status") != "PASS":
            raise RuntimeError(
                "final-test requires a validation PASS selection report; "
                f"got status={selection_report.get('status')!r}"
            )
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"evaluation config is missing: {config_path}")
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"evaluation checkpoint is missing: {checkpoint_path}")

    if split == "final-test":
        from tools.select_stream25_checkpoint import (
            compute_experiment_hash,
            FINAL_TEST_MANIFEST_HASH_KEY,
            _sha256_file,
        )
        actual_checkpoint_hash = _sha256_file(checkpoint_path)
        actual_config_hash = _sha256_file(config_path)
        actual_evaluator_hash = _sha256_file(str(Path(__file__).resolve()))
        actual_acceptance_hash = _sha256_file(
            str(WORKTREE / "src" / "utils" / "stream25_metrics.py")
        )
        frozen_pairs = (
            ("checkpoint_hash", actual_checkpoint_hash),
            ("config_hash", actual_config_hash),
            ("evaluator_hash", actual_evaluator_hash),
            ("acceptance_table_hash", actual_acceptance_hash),
        )
        for key, actual in frozen_pairs:
            if not selection_report.get(key) or selection_report[key] != actual:
                raise RuntimeError(f"final-test frozen {key} does not match current artifact")
        exp_hash = compute_experiment_hash(
            checkpoint=actual_checkpoint_hash,
            config=actual_config_hash,
            evaluator=actual_evaluator_hash,
            acceptance=actual_acceptance_hash,
            final_manifest=selection_report.get(FINAL_TEST_MANIFEST_HASH_KEY, ""),
        )
        check_final_test_sentinel(FINAL_TEST_SENTINEL_DIR, exp_hash)

    from tools.stream25_runtime import (
        annotation_path,
        build_stream25_dataset,
        build_stream25_model,
        collate_and_prepare,
        load_stream25_args,
        sha256_file,
    )

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    args = load_stream25_args(
        config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_role="evaluation",
    )
    evaluation_seed = set_evaluation_seed(args.seed)

    # Gates are reported per camera name, so the list has to come from this run's
    # camera_list, not the module's frozen tri-view default.
    from src.utils.misc import camera_names_from_arguments

    camera_order = set_camera_order(
        camera_names_from_arguments(args, role="evaluation")
    )

    # Push the unprojected near-surface point to the ball centre. Off by default so
    # frame24_position stays comparable; the radius comes from the config.
    if ball_radius is None:
        ball_radius = float(getattr(args, "stream25_ball_radius", 0.0325) or 0.0325)
    ball_surface_offset = float(ball_radius_compensation) * float(ball_radius)
    if ball_surface_offset:
        print(
            f"[eval] ball-surface compensation ON: "
            f"{ball_radius_compensation:.3f} x {ball_radius:.4f} m "
            f"= {ball_surface_offset*100:.2f} cm along the view ray. "
            f"frame24_position is NOT comparable to runs without it.",
            flush=True,
        )
    print(
        f"[eval] named views ({len(camera_order)}): {', '.join(camera_order)}",
        flush=True,
    )

    manifest = annotation_path(args, split)
    if split == "final-test":
        from tools.select_stream25_checkpoint import FINAL_TEST_MANIFEST_HASH_KEY
        expected_manifest_hash = selection_report.get(FINAL_TEST_MANIFEST_HASH_KEY, "")
        actual_manifest_hash = sha256_file(manifest)
        if not expected_manifest_hash or actual_manifest_hash != expected_manifest_hash:
            raise RuntimeError(
                "final-test manifest hash differs from the frozen selection report"
            )
    # Slide the observation window later in the clip. get_frame measures time
    # from the window's own first frame, so the trunk receives bit-identical time
    # values at any offset and no retraining is involved -- only nearer images.
    # catch_position is the number to read across offsets; frame24_* changes its
    # horizon with the window and stops being comparable.
    context_offset = int(context_offset or 0)
    args.stream25_context_offset = context_offset
    catch_frame = int(getattr(args, "stream25_catch_frame", 0) or 0)
    if context_offset:
        terminal_frame = STREAM25_CONTEXT_FRAMES[-1] + context_offset
        print(
            f"[eval] context window slid +{context_offset}: "
            f"{tuple(f + context_offset for f in STREAM25_CONTEXT_FRAMES)}, "
            f"terminal frame {terminal_frame}. frame24_* is NOT comparable to offset 0; "
            f"read catch_position instead.",
            flush=True,
        )
        # Remaining extrapolation: last stored frame minus terminal observation,
        # which shrinks with the offset and reaches zero at +9.
        span = STREAM25_ALL_TARGET_FRAMES[-1] - STREAM25_ALL_TARGET_FRAMES[0]
        remaining = STREAM25_ALL_TARGET_FRAMES[-1] - terminal_frame
        print(
            f"[eval] frame24 horizon: {remaining} frames "
            f"({remaining * float(args.timespan) / span:.3f} s); "
            f"catch horizon: {catch_frame - terminal_frame} frames. "
            + ("frame24_* will be skipped." if remaining <= 0 else ""),
            flush=True,
        )
        if not catch_frame:
            raise ValueError(
                "a slid window needs stream25_catch_frame in the config: without "
                "it there is no offset-independent landing metric to compare"
            )
    dataset = build_stream25_dataset(args, split, online_feat=False)
    model = build_stream25_model(args, torch_device)
    model.eval()
    if render_chunk is not None:
        model.render_target_chunk_size = int(render_chunk)
        print(
            f"[eval] render_target_chunk_size override -> {model.render_target_chunk_size}",
            flush=True,
        )

    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,               # sequential, so results stay reproducible
        num_workers=num_workers,
        collate_fn=_single_sample_collate,
    )
    if num_workers > 0:
        # Prefetch so reading overlaps the GPU instead of leaving it idle.
        loader_kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    scene_results = []
    with torch.inference_mode():
        for index, sample in enumerate(loader):
            input_dict, target_dict = collate_and_prepare(sample, args, torch_device)
            prepared = dict(input_dict)
            prepared.update(target_dict)
            scene_result = evaluate_scene(
                model, prepared, torch_device, args.timespan,
                ball_surface_offset=ball_surface_offset,
                fit_frames=fit_frames,
                context_offset=context_offset,
                catch_frame=catch_frame,
            )
            scene_result["scene_index"] = index
            scene_result["scene_name"] = input_dict.get("scene_name", [str(index)])[0]
            scene_results.append(_compact_scene_result(scene_result))
            print(
                f"Evaluated Stream25 {split} scene {index + 1}/{len(dataset)}",
                flush=True,
            )
            del input_dict, target_dict, prepared, sample

    return _finalize_and_write(
        scene_results,
        split=split,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        manifest=manifest,
        evaluation_seed=evaluation_seed,
        reference=reference,
        output_json=output_json,
        output_markdown=output_markdown,
        ball_surface_offset=ball_surface_offset,
        fit_frames=fit_frames,
        ball_radius=ball_radius,
        ball_radius_compensation=ball_radius_compensation,
        context_offset=context_offset,
        catch_frame=catch_frame,
    )


def _finalize_and_write(
    scene_results,
    *,
    split,
    checkpoint_path,
    config_path,
    manifest,
    evaluation_seed,
    reference,
    output_json,
    output_markdown,
    ball_surface_offset=0.0,
    fit_frames=None,
    ball_radius=None,
    ball_radius_compensation=0.0,
    context_offset=0,
    catch_frame=0,
):
    """Aggregate every scene result and write the report.

    This is a top-level function so several shards can be merged, which means
    run_evaluation's locals are NOT visible here: everything it needs must be
    passed in. Referring to one by accident is a NameError on every eval.
    """
    from tools.stream25_runtime import sha256_file

    # Sort by scene_index so the result is independent of shard order.
    scene_results = sorted(scene_results, key=lambda s: s.get("scene_index", 0))

    gate_result = summarize_stream25_scene_results(scene_results)
    scope_reports = gate_result["scope_reports"]
    metrics = scope_reports["aggregate"]["metrics"]
    valid_counts = scope_reports["aggregate"]["valid_counts"]
    ball_visibility_counts = {
        bucket: {
            camera_name: sum(
                1
                for scene in scene_results
                for record in scene["records"]
                if record["view"] == camera_name
                and record["frame"] in frames
                and record["ball_visible"]
            )
            for camera_name in get_camera_order()
        }
        for bucket, frames in TIME_BUCKETS.items()
    }

    result: Dict[str, Any] = {
        "split": split,
        "role": "reference" if reference else "candidate",
        "seed": evaluation_seed,
        "checkpoint": checkpoint_path,
        "checkpoint_hash": sha256_file(checkpoint_path),
        "config_hash": sha256_file(config_path),
        "manifest": str(manifest),
        "manifest_hash": sha256_file(manifest),
        "scene_count": len(scene_results),
        "considered_frame_eyes": sum(s["considered_frame_eyes"] for s in scene_results),
        "visible_ball_frame_eyes": sum(s["visible_ball_frame_eyes"] for s in scene_results),
        "ball_visibility_counts": ball_visibility_counts,
        "frame24_position_method": (
            "rendered_depth_ms3_predicted_semantic_frame15"
            + ("_ball_center_compensated" if ball_surface_offset else "")
        ),
        "ball_surface_offset_m": ball_surface_offset,
        "ball_radius_m": ball_radius,
        "ball_radius_compensation": ball_radius_compensation,
        "fit_frames": list(fit_frames) if fit_frames else None,
        # Which window this run observed. A non-zero offset moves the terminal
        # observation, so frame24_* is measured over a different horizon and must
        # not be compared across offsets -- catch_position is the one that can.
        "context_offset": int(context_offset),
        "context_frames": [f + int(context_offset) for f in STREAM25_CONTEXT_FRAMES],
        "catch_frame": int(catch_frame),
        "metrics": metrics,
        "valid_counts": valid_counts,
        "scope_reports": scope_reports,
        "per_scene": scene_results,
        "gates": gate_result["gates"],
        "missing_gates": gate_result["missing_gates"],
        "all_gates_pass": gate_result["all_gates_pass"],
        "worst_ratio": gate_result["worst_ratio"],
        "overall": "PASS" if gate_result["all_gates_pass"] else "FAIL",
    }

    result = _json_safe(result)
    result = _json_safe(result)
    if output_json:
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2, allow_nan=False)
    if output_markdown:
        from src.utils.stream25_report import render_single_markdown
        lines = [
            "# Stream25 Evaluation",
            "",
            f"- Role: **{result['role']}**",
            f"- Split: `{split}`",
            f"- Scenes: **{result['scene_count']}**",
            f"- Considered frame-eyes: **{result['considered_frame_eyes']}**",
            f"- Overall: **{result['overall']}**",
            (f"- Context window: **{result['context_frames']}**"
             f" (offset +{result['context_offset']}); compare **catch_position**,"
             f" not frame24_*"
             if result["context_offset"] else
             f"- Context window: **frozen {list(STREAM25_CONTEXT_FRAMES)}**"),
            (f"- Ball-surface compensation: **{result['ball_surface_offset_m']*100:.2f} cm**"
             f" ({result['ball_radius_compensation']:.3f} x r={result['ball_radius_m']:.4f} m)"
             " -- frame24_position is NOT comparable with runs that had this off"
             if result["ball_surface_offset_m"] else
             "- Ball-surface compensation: off (frame24_position measures the ball's near surface, not its centre)"),
            "",
            "## Key metrics (aggregate)",
            "",
            render_single_markdown(result["metrics"]),
            "",
            "## Failing gates",
            "",
            *_render_failing_gates(result),
            "",
            "## Gate metrics (full)",
            "",
            "```json",
            json.dumps(result["metrics"], indent=2, allow_nan=False),
            "```",
        ]
        Path(output_markdown).write_text("\n".join(lines) + "\n")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--output-markdown", default=None)
    parser.add_argument("--selection-report", default=None)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--render-chunk", type=int, default=None,
                        help="override render_target_chunk_size; larger renders faster")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers; 8 suits a local disk, 0 loads in the main process")
    parser.add_argument("--ball-radius-compensation", "--ball_radius_compensation",
                        dest="ball_radius_compensation", type=float, nargs="?",
                        const=BALL_SURFACE_COEFFICIENT_MEASURED, default=0.0,
                        help="push the unprojected near-surface point to the ball centre, "
                             "removing a constant bias. Takes a coefficient c; the offset is "
                             f"c x radius. Without a value it uses the measured "
                             f"{BALL_SURFACE_COEFFICIENT_MEASURED}. Default 0 keeps "
                             "frame24_position comparable with past runs.")
    parser.add_argument("--fit-frames", "--fit_frames",
                        "--balltoken-fit-frames", "--balltoken_fit_frames",
                        dest="fit_frames", default=None,
                        help="Context frames whose rendered ball centres are refit into "
                             "(pos15, v15), e.g. '0,3,6,9,12,15'. Default: all of them. "
                             "Dropping the earliest frames shortens the time span and makes the "
                             "fitted velocity worse (0.50 s -> 0.30 s costs 1.87x), so only drop "
                             "them when pixel_pos_error_frame* shows they are >1.6x worse.")
    parser.add_argument("--ball-radius", "--ball_radius", dest="ball_radius",
                        type=float, default=None,
                        help="ball radius in metres; defaults to the config's stream25_ball_radius")
    parser.add_argument("--context-offset", "--context_offset", dest="context_offset",
                        type=int, default=0,
                        help="slide the observation window N frames later (evaluation only). "
                             "0 is the frozen contract; 9 makes context 9,12,...,24. "
                             "Across offsets compare catch_position, never frame24_*")
    args = parser.parse_args()

    sel = None
    if args.selection_report:
        with open(args.selection_report) as f:
            sel = json.load(f)

    result = run_evaluation(
        args.config, args.checkpoint, args.split,
        selection_report=sel, output_json=args.output,
        output_markdown=args.output_markdown, reference=args.reference,
        render_chunk=args.render_chunk, num_workers=args.num_workers,
        ball_radius_compensation=args.ball_radius_compensation,
        ball_radius=args.ball_radius,
        fit_frames=(
            [int(x) for x in args.fit_frames.split(",") if x.strip()]
            if args.fit_frames else None
        ),
        context_offset=args.context_offset,
    )
    print(json.dumps(result, indent=2))
