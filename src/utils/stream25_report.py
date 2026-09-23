"""Key-metric extraction and the markdown tables built from it.

Shared by scripts/eval_stream25_base.py and tools/compare_evaluations.py. Standard
library only; the threshold table is imported lazily so this never pulls in torch.

`metrics` is evaluation.json's top-level "metrics", the aggregate scope:
  metrics[<scalar>][<bucket>]          rgb_psnr, semantic_miou, ball_iou, depth_absrel
  metrics["ball_depth_error_median"][<bucket>]
  metrics["ms3_ball_velocity"]["median" | "p95"]
  metrics["frame24_position"]["median" | "p95"]
bucket is one of anchor, interpolation, near, mid, far, farthest.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# (label, direction, [(metric_key, subkey), ...], threshold_key)
#   direction: "up" means larger is better, "down" means smaller is better
#   threshold_key: a key in ACCEPTANCE_TABLE, or None for an ungated row
KEY_METRICS: List[Tuple[str, str, List[Tuple[str, str]], Optional[str]]] = [
    ("frame24 position med / p95",    "down", [("frame24_position", "median"), ("frame24_position", "p95")], "frame24_position"),
    # Landing at the catch frame: the only row comparable across context offsets,
    # because frame24_* measures a horizon that shrinks as the window slides.
    ("catch position med / p95",        "down", [("catch_position", "median"), ("catch_position", "p95")], None),
    # The landing error split along the ring axis. Axial error only makes the ball
    # early or late; only the in-plane part decides whether it clears the opening.
    # catch_position uses the 3D norm and so counts axial error as a miss, which
    # makes any success rate derived from it a lower bound.
    ("catch position inplane med / p95", "down", [("catch_position_inplane", "median"), ("catch_position_inplane", "p95")], None),
    ("catch position axial med",        "down", [("catch_position_axial", "median")], None),
    ("catch horizon s",                 "down", [("catch_horizon_s", "median")], None),
    ("pixel fit frame24 med / p95",     "down", [("frame24_position_fit", "median"), ("frame24_position_fit", "p95")], None),
    ("pixel fit pos15 med / p95",       "down", [("ball_pos15_error_fit", "median"), ("ball_pos15_error_fit", "p95")], None),
    ("pixel fit vel15 med / p95",       "down", [("ball_vel15_error_fit", "median"), ("ball_vel15_error_fit", "p95")], None),
    ("pixel pos err const / scatter",   "down", [("pixel_pos_error_constant_m", "median"), ("pixel_pos_error_scatter_m", "median")], None),
    ("pixel pos err f0 / f15",          "down", [("pixel_pos_error_frame0", "median"), ("pixel_pos_error_frame15", "median")], None),
    ("ball velocity med / p95",       "down", [("ms3_ball_velocity", "median"), ("ms3_ball_velocity", "p95")], "ms3_ball_velocity"),
    ("ball acceleration med / p95",   "down", [("ms3_ball_acceleration", "median"), ("ms3_ball_acceleration", "p95")], "ms3_ball_acceleration"),
    ("ball jerk med / p95",           "down", [("ms3_ball_jerk", "median"), ("ms3_ball_jerk", "p95")], "ms3_ball_jerk"),
    ("ball depth farthest med / p95", "down", [("ball_depth_error_median", "farthest"), ("ball_depth_error_p95", "farthest")], None),
    ("ball IoU anchor",               "up",   [("ball_iou", "anchor")], "ball_iou"),
    ("ball IoU farthest",             "up",   [("ball_iou", "farthest")], "ball_iou"),
    ("ball RGB PSNR anchor",          "up",   [("ball_rgb_psnr", "anchor")], "ball_rgb_psnr"),
    ("semantic mIoU anchor",          "up",   [("semantic_miou", "anchor")], "semantic_miou"),
    ("semantic mIoU farthest",        "up",   [("semantic_miou", "farthest")], "semantic_miou"),
    ("RGB PSNR anchor",               "up",   [("rgb_psnr", "anchor")], "rgb_psnr"),
    ("RGB PSNR farthest",             "up",   [("rgb_psnr", "farthest")], "rgb_psnr"),
    ("RGB p10 farthest",              "up",   [("rgb_psnr_p10", "farthest")], "rgb_psnr_p10"),
    ("depth absrel farthest",         "down", [("depth_absrel", "farthest")], "depth_absrel"),
]


def _get(metrics: Dict[str, Any], key: str, sub: str) -> float:
    node = metrics.get(key) if isinstance(metrics, dict) else None
    if isinstance(node, dict):
        try:
            return float(node.get(sub))
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.3f}"


def extract_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for label, direction, paths, thr_key in KEY_METRICS:
        vals = [_get(metrics, k, s) for (k, s) in paths]
        rows.append({
            "label": label, "direction": direction,
            "values": vals, "paths": paths, "threshold_key": thr_key,
        })
    return rows


def _threshold(thr_key: Optional[str], sub: str):
    if not thr_key:
        return None
    from src.utils.stream25_metrics import ACCEPTANCE_TABLE  # deferred: that module needs torch
    row = ACCEPTANCE_TABLE.get(thr_key)
    return row.get(sub) if isinstance(row, dict) else None


def render_single_markdown(metrics: Dict[str, Any]) -> str:
    """One eval's key metrics: metric | value | threshold | pass."""
    out = ["| metric | value | threshold | pass |", "| --- | ---: | ---: | :---: |"]
    for row in extract_rows(metrics):
        vals, direction, paths = row["values"], row["direction"], row["paths"]
        val_str = " / ".join(_fmt(v) for v in vals)
        thr = _threshold(row["threshold_key"], paths[0][1])
        thr_str = _fmt(thr) if thr is not None else "-"
        ok = "-"
        if thr is not None and not (isinstance(vals[0], float) and math.isnan(vals[0])):
            passed = (vals[0] >= thr) if direction == "up" else (vals[0] <= thr)
            ok = "yes" if passed else "NO"
        out.append(f"| {row['label']} | {val_str} | {thr_str} | {ok} |")
    return "\n".join(out)


#: Tag -> displayed text. The comparison logic tests the tag, never the wording.
_VERDICT_TEXT = {
    "n/a": "-",
    "better": "improved",
    "worse": "worse",
    "much_worse": "much worse",
    "same": "about the same",
}


def _verdict(base: float, cand: float, direction: str, tol: float = 0.02) -> str:
    """Return a tag from _VERDICT_TEXT, not display text."""
    if math.isnan(base) or math.isnan(cand):
        return "n/a"
    denom = abs(base) if base != 0 else 1e-9
    rel = (cand - base) / denom
    better = (rel > tol) if direction == "up" else (rel < -tol)
    worse = (rel < -tol) if direction == "up" else (rel > tol)
    if better:
        return "better"
    if worse:
        return "much_worse" if abs(rel) > 0.5 else "worse"
    return "same"


def render_compare_markdown(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any],
                            label_a: str = "A", label_b: str = "B") -> str:
    """Two runs side by side, flagging a p95 tail the median hides."""
    out = [f"| metric | {label_a} | {label_b} | verdict |", "| --- | ---: | ---: | --- |"]
    for ra, rb in zip(extract_rows(metrics_a), extract_rows(metrics_b)):
        va = " / ".join(_fmt(v) for v in ra["values"])
        vb = " / ".join(_fmt(v) for v in rb["values"])
        verdict = _verdict(ra["values"][0], rb["values"][0], ra["direction"])
        note = ""
        if len(ra["values"]) > 1:
            v_med = _verdict(ra["values"][0], rb["values"][0], ra["direction"])
            v_p95 = _verdict(ra["values"][1], rb["values"][1], ra["direction"])
            worse_p95 = v_p95 in ("worse", "much_worse")
            worse_med = v_med in ("worse", "much_worse")
            if worse_p95 and not worse_med:
                note = " (p95 tail)"
            elif v_p95 == "much_worse" and v_med != "much_worse":
                note = " (heavier p95 tail)"
        out.append(f"| {ra['label']} | {va} | {vb} | {_VERDICT_TEXT[verdict]}{note} |")
    return "\n".join(out)
