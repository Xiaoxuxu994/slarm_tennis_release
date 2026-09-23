"""Stream25 pure evaluation metrics for the six-context, 25-frame contract.

All functions are pure and tested independently of the model/renderer so the
frozen acceptance contract can be verified without CUDA.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

# Tri-view is the historical default and the basis of every published gate report,
# so these names stay fixed for the frozen acceptance table.
#
# The active list is settable at run time: a two-view model renders a view axis of
# length 2, while the evaluator reports one gate per named view, so a hardcoded
# list raises at the num_views check. The evaluation entry point resolves
# camera_list[num_max_cameras] from the config and calls set_camera_order(); every
# other caller should go through get_camera_order() / get_required_eval_scopes().
# Unset means tri-view, byte for byte as before.
CAMERA_ORDER: Tuple[str, ...] = (
    "front_left",
    "front_right",
    "lower_front",
)
REQUIRED_EVAL_SCOPES: Tuple[str, ...] = ("aggregate",) + CAMERA_ORDER
DEFAULT_CAMERA_ORDER: Tuple[str, ...] = CAMERA_ORDER
_ACTIVE_CAMERA_ORDER: Tuple[str, ...] = DEFAULT_CAMERA_ORDER


def set_camera_order(names: Sequence[str]) -> Tuple[str, ...]:
    """Set the named views this process evaluates. Returns the resolved tuple."""
    global _ACTIVE_CAMERA_ORDER
    resolved = tuple(str(name) for name in names)
    if not resolved:
        raise ValueError("camera order must name at least one view")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"camera order has duplicate names: {resolved}")
    _ACTIVE_CAMERA_ORDER = resolved
    return _ACTIVE_CAMERA_ORDER


def reset_camera_order() -> Tuple[str, ...]:
    """Restore the frozen tri-view default (used by tests and between runs)."""
    global _ACTIVE_CAMERA_ORDER
    _ACTIVE_CAMERA_ORDER = DEFAULT_CAMERA_ORDER
    return _ACTIVE_CAMERA_ORDER


def get_camera_order() -> Tuple[str, ...]:
    """Named views in render order (view axis index -> name)."""
    return _ACTIVE_CAMERA_ORDER


def get_required_eval_scopes() -> Tuple[str, ...]:
    """Gate scopes: the aggregate plus one per named view."""
    return ("aggregate",) + _ACTIVE_CAMERA_ORDER

TIME_BUCKETS: Dict[str, List[int]] = {
    "anchor": [0, 3, 6, 9, 12, 15],
    "interpolation": [1, 2, 4, 5, 7, 8, 10, 11, 13, 14],
    "near": list(range(16, 18)),
    "mid": list(range(18, 20)),
    "far": list(range(20, 22)),
    "farthest": list(range(22, 25)),
}

ACCEPTANCE_TABLE: Dict[str, Dict[str, Any]] = {
    "rgb_psnr": {
        "anchor": 25.0, "interpolation": 24.0, "near": 23.0,
        "mid": 22.0, "far": 21.0, "farthest": 20.0,
    },
    "rgb_ssim": {
        "anchor": 0.90, "interpolation": 0.88, "near": 0.86,
        "mid": 0.84, "far": 0.82, "farthest": 0.80,
    },
    "rgb_psnr_p10": {
        "anchor": 23.0, "interpolation": 22.0, "near": 21.0,
        "mid": 20.0, "far": 19.0, "farthest": 18.0,
    },
    "ball_rgb_psnr": {
        "interpolation": 22.0, "near": 18.0, "mid": 18.0,
        "far": 18.0, "farthest": 18.0,
    },
    "depth_absrel": {
        "anchor": 0.08, "interpolation": 0.10, "near": 0.12,
        "mid": 0.14, "far": 0.16, "farthest": 0.18,
    },
    "ball_depth_error_median": {name: 0.10 for name in TIME_BUCKETS},
    "ball_depth_error_p95": {name: 0.25 for name in TIME_BUCKETS},
    "semantic_miou": {
        "anchor": 0.80, "interpolation": 0.75, "near": 0.70,
        "mid": 0.65, "far": 0.60, "farthest": 0.55,
    },
    "ball_iou": {
        "anchor": 0.75, "interpolation": 0.65, "near": 0.60,
        "mid": 0.55, "far": 0.50, "farthest": 0.45,
    },
    "ms3_ball_velocity": {"median": 0.25, "p95": 0.75},
    "ms3_ball_acceleration": {"median": 0.50, "p95": 1.50},
    "ms3_ball_jerk": {"median": 1.00, "p95": 3.00},
    "ms3_static_velocity": {"median": 0.05, "p95": 0.20},
    "ms3_static_acceleration": {"median": 0.10, "p95": 0.50},
    "ms3_static_jerk": {"median": 0.20, "p95": 1.00},
    "frame24_position": {"median": 0.15, "p95": 0.30},
}
for _context_key in (
    "ms3_ball_velocity", "ms3_ball_acceleration", "ms3_ball_jerk",
    "ms3_static_velocity", "ms3_static_acceleration", "ms3_static_jerk",
):
    ACCEPTANCE_TABLE[f"context_{_context_key}"] = dict(ACCEPTANCE_TABLE[_context_key])


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt)
    if mse < 1e-12:
        return 120.0
    return float(10.0 * math.log10(1.0 / mse.item()))


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = pred.mean(dim=(-2, -1))
    mu_g = gt.mean(dim=(-2, -1))
    var_p = pred.var(dim=(-2, -1))
    var_g = gt.var(dim=(-2, -1))
    cov = ((pred - mu_p.unsqueeze(-1).unsqueeze(-1)) * (gt - mu_g.unsqueeze(-1).unsqueeze(-1))).mean(dim=(-2, -1))
    ssim = ((2 * mu_p * mu_g + C1) * (2 * cov + C2)) / ((mu_p ** 2 + mu_g ** 2 + C1) * (var_p + var_g + C2))
    return float(ssim.mean().item())


def compute_depth_absrel(pred: torch.Tensor, gt: torch.Tensor) -> float:
    valid = torch.isfinite(gt) & (gt > 0)
    if not valid.any():
        return float("nan")
    abs_err = (pred[valid] - gt[valid]).abs()
    rel_err = abs_err / gt[valid].clamp(min=0.1)
    return float(rel_err.mean().item())


def compute_depth_rmse(pred: torch.Tensor, gt: torch.Tensor) -> float:
    valid = torch.isfinite(gt) & (gt > 0)
    if not valid.any():
        return float("nan")
    return float(torch.sqrt(F.mse_loss(pred[valid], gt[valid])).item())


def finite_percentile(values: List[float], q: float) -> float:
    tensor = torch.tensor([v for v in values if math.isfinite(v)], dtype=torch.float64)
    if tensor.numel() == 0:
        return float("nan")
    return float(torch.quantile(tensor, q / 100.0).item())


def compute_iou(pred: torch.Tensor, gt: torch.Tensor, class_id: int) -> Optional[float]:
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    union = pred_mask | gt_mask
    if union.sum() == 0:
        return None
    intersection = (pred_mask & gt_mask).sum()
    return float(intersection.item() / union.sum().item())


def compute_ball_region_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    class_id: int = 1,
) -> Optional[float]:
    """Return N/A whenever the ground-truth ball is off-screen."""
    if not (gt == class_id).any():
        return None
    return compute_iou(pred, gt, class_id=class_id)


def compute_miou(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> float:
    ious = []
    for cls in range(num_classes):
        iou = compute_iou(pred, gt, class_id=cls)
        if iou is not None:
            ious.append(iou)
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


def compute_macro_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> float:
    dices = []
    for cls in range(num_classes):
        p = (pred == cls).float()
        g = (gt == cls).float()
        intersection = (p * g).sum()
        union = p.sum() + g.sum()
        if union > 0:
            dices.append(float((2.0 * intersection / (union + 1e-8)).item()))
    if not dices:
        return 0.0
    return sum(dices) / len(dices)


def compute_ms3_vector_error(pred: torch.Tensor, gt: torch.Tensor) -> float:
    err = (pred - gt).norm(dim=-1)
    return float(err.mean().item())


def integrate_frame24_position(
    pos15: torch.Tensor,
    v15: torch.Tensor,
    a15: torch.Tensor,
    j15: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Integrate terminal context (frame 15) MS3 to frame 24 position."""
    return pos15 + v15 * dt + 0.5 * a15 * dt ** 2 + (1.0 / 6.0) * j15 * dt ** 3


def integrate_frame24_position_physics(
    pos15: torch.Tensor,
    v15: torch.Tensor,
    dt: float,
    gravity_rig: torch.Tensor,
) -> torch.Tensor:
    """Integrate frame-15 ball state to frame 24 under known gravity (a=g, j=0).

    Companion to :func:`integrate_frame24_position` for the ball-token path: the
    network only supplies ``pos15``/``v15`` and the physical prior supplies the
    second order term, so nothing free-form is amplified by ``dt**2``/``dt**3``.
    All three arguments live in the scene-fixed rig frame.
    """
    return pos15 + v15 * dt + 0.5 * gravity_rig * dt ** 2


# Ball centre versus ball near surface: a systematic bias on the pixel path.
#
# The depth map is a z-buffer recording the first opaque surface, so unprojecting
# the ball mask gives the ball's NEAR SURFACE, while ball_trajectory's
# position_rig is the simulator's ball CENTRE. The gap is constant and points at
# the camera -- the same direction that carries 95.5% of the error.
#
# The coefficient depends on how the sphere is pooled:
#     1.000  nearest point only
#     0.707  disc median,  -r/sqrt(2)
#     0.667  disc mean,    -(2/3)r
#     0.646  measured on 0903_2k, matching the evaluator's own median pooling
#     0.000  off
#
# Measured 2026-09-09 and left OFF by default: it buys nothing at the median.
# Over 100 scenes, compensation 21.0 mm:
#
#                    off       on        delta   of compensation
#     pred along_med  0.0304  0.0292     -1.2mm        6%
#     pred along_p95  0.0947  0.0803    -14.4mm       69%
#     gt   along_med  0.0327  0.0277     -5.0mm       24%
#     gt   along_p95  0.1135  0.0925    -21.0mm      100%
#
# Only the tail behaves like a near-surface bias. The GT .tif is a clean z-buffer,
# but the MODEL's rendered depth is not: 3DGS returns an alpha-weighted expected
# depth, and the ball is 2.66 px of nothing but edge pixels, so the background
# behind it drags the value past the centre. Assuming the render was a z-buffer
# because the GT is one was the mistake.
#
BALL_SURFACE_COEFFICIENT_MEASURED = 0.646
BALL_SURFACE_COEFFICIENT_DISC_MEAN = 2.0 / 3.0
BALL_SURFACE_COEFFICIENT_DISC_MEDIAN = 0.5 ** 0.5


def apply_ball_surface_offset(
    positions: torch.Tensor,
    directions: torch.Tensor,
    offset_meters: float,
) -> torch.Tensor:
    """Push an unprojected near-surface point along the ray to the ball centre.

    ``directions`` is normalised here because ``dirs`` from embedders.py is NOT a
    unit vector (its camera-frame z is 1, to pair with planar z-depth). Using it
    raw scales the offset by ||dirs||, up to 1.3x in the corners of the image.

    Args:
        positions: ``[..., 3]`` unprojected surface points.
        directions: ``[..., 3]`` ray directions, need not be normalised.
        offset_meters: metres to move away from the camera, ``coefficient * radius``.
    """
    if not offset_meters:
        return positions
    unit = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    return positions + unit * offset_meters


def fit_ballistic_state(
    positions: torch.Tensor,
    times: torch.Tensor,
    gravity: torch.Tensor,
) -> torch.Tensor:
    """Least-squares (pos, vel) at ``t = 0`` from several observed positions.

    Removing the known gravity term linearizes the trajectory::

        q(t) = p(t) - 0.5 * g * t**2 = p0 + v0 * t

    so a plain two-parameter fit recovers the state. Two properties matter here,
    and both are tested:

     A **constant** position bias cancels out of the velocity exactly. It shifts
      q(t) by the same amount at every t, so it lands entirely in p0 and leaves
      v0 untouched. Learned position heads carry systematic biases -- this repo
      already found one, the 2.1 cm between the ball's front surface and its
      centre -- and this fit is immune to that whole class.

     The velocity error scales as ``sigma_p / (dt * sqrt(n(n^2-1)/12))``, which
      is dominated by the **time span**, not the sample count. Dropping the two
      earliest observations from the frozen six takes the span from 0.50 s to
      0.30 s and makes the fitted velocity 1.87x worse -- enough to lose to
      direct regression. Only drop early observations when their position error
      is more than about 1.6x the later ones; ball_prefix_pos_error_frame* is
      reported so that ratio is visible rather than assumed.

    Args:
        positions: ``[N, 3]`` observed positions, in any single frame.
        times: ``[N]`` observation times in seconds, relative to the target
            instant, so ``p0`` comes out at that instant. Must be distinct.
        gravity: ``[3]`` in the same frame as ``positions``.

    Returns:
        ``[6]`` = ``(p0, v0)``.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must be [N, 3]")
    if times.ndim != 1 or times.shape[0] != positions.shape[0] or times.shape[0] < 2:
        raise ValueError("times must be [N] with N >= 2 and match positions")
    if gravity.shape != (3,):
        raise ValueError("gravity must be a three-vector")
    t = times.double()
    if torch.unique(t).numel() != t.numel():
        raise ValueError("Observation times must be distinct")
    corrected = positions.double() - 0.5 * (t ** 2)[:, None] * gravity.double()
    design = torch.stack((torch.ones_like(t), t), dim=-1)          # [N, 2]
    solution = torch.linalg.lstsq(design, corrected).solution      # [2, 3]
    return solution.reshape(6).to(positions.dtype)


def transform_position(position: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous rigid transform to one or more 3-D positions."""
    if position.shape[-1] != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError(
            "position/transform must end in (3,) and (4, 4), got "
            f"{tuple(position.shape)} and {tuple(transform.shape)}"
        )
    homogeneous = torch.cat((position, torch.ones_like(position[..., :1])), dim=-1)
    return torch.matmul(transform, homogeneous.unsqueeze(-1)).squeeze(-1)[..., :3]


def transform_vector(vector: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Rotate one or more 3-D vectors without applying rigid translation."""
    if vector.shape[-1] != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError(
            "vector/transform must end in (3,) and (4, 4), got "
            f"{tuple(vector.shape)} and {tuple(transform.shape)}"
        )
    return torch.matmul(
        transform[..., :3, :3], vector.unsqueeze(-1)
    ).squeeze(-1)


def worst_normalized_ratio(metric: float, limit: float, upper_bound: bool = True) -> float:
    if upper_bound:
        return metric / limit
    if metric <= 0.0:
        return float("inf")
    return limit / metric


def _checkpoint_step(name: str) -> int:
    """Parse common checkpoint names such as ``5k`` and ``004999``."""
    normalized = name.strip().lower()
    if normalized.endswith("k"):
        try:
            return int(float(normalized[:-1]) * 1_000)
        except ValueError:
            pass
    digits = "".join(character for character in normalized if character.isdigit())
    return int(digits) if digits else 2**63 - 1


def checkpoint_report_worst_ratio(report: Dict[str, Any]) -> Optional[float]:
    """Return the minimax score only when every required scope passes."""
    scope_reports = report.get("scope_reports")
    if not isinstance(scope_reports, dict):
        return None
    required_scopes = get_required_eval_scopes()
    if set(scope_reports) != set(required_scopes):
        return None
    ratios = []
    for scope in required_scopes:
        scope_report = scope_reports[scope]
        if not isinstance(scope_report, dict) or not scope_report.get(
            "all_gates_pass", False
        ):
            return None
        ratio = scope_report.get("worst_ratio")
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            return None
        ratios.append(float(ratio))
    return max(ratios)


def select_checkpoint(
    checkpoints: Dict[str, Dict[str, Any]],
    *,
    tie_threshold: float = 0.01,
) -> Optional[str]:
    """Select the passing checkpoint with the best four-scope minimax score."""
    candidates = []
    for name, report in checkpoints.items():
        ratio = checkpoint_report_worst_ratio(report)
        if ratio is not None:
            candidates.append((name, ratio, _checkpoint_step(name)))
    if not candidates:
        return None

    candidates.sort(key=lambda candidate: (candidate[1], candidate[2]))
    true_best_ratio = candidates[0][1]
    denominator = max(abs(true_best_ratio), 1e-12)
    tied = [
        candidate
        for candidate in candidates
        if abs(candidate[1] - true_best_ratio) / denominator < tie_threshold
    ]
    return min(tied, key=lambda candidate: candidate[2])[0]
