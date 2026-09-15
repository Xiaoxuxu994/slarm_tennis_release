#!/usr/bin/env python
"""Split an evaluation report by which cameras actually saw the ball.

Why this exists. The 0903_2k rig has three named views, but a catch net
occludes `lower_front` in most scenes, so a large share of them are solved
from two views rather than three. A newer dataset opens that ring, so the
bottom view sees the ball. The obvious question -- what is the third view
worth -- does not need new data or a new run to answer: the existing report
already records, per scene and per (frame, view), whether the ball was
visible. Grouping the scenes by that and comparing the landing error is a
natural experiment on data already in hand.

Read the result as an ESTIMATE, not a measurement of the new dataset. Scenes
where the bottom view happens to see the ball may differ in other ways (a
trajectory nearer that camera is both more visible and better triangulated by
the front pair). It bounds the effect; it does not isolate it.

    python tools/split_by_view_visibility.py <evaluation.json> [--view lower_front]

All output is ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# Landing error at the catch frame is the number the task cares about; the
# others are reported because they localise where a difference comes from.
METRICS = (
    ("catch_position", "landing at the catch frame (m)"),
    ("frame24_position", "landing at frame 24 (m)"),
    ("ball_pos15_error_fit", "fitted terminal position (m)"),
    ("ball_vel15_error_fit", "fitted terminal velocity (m/s)"),
    ("pixel_pos_error_scatter_m", "frame-to-frame scatter (m)"),
    ("pixel_pos_error_constant_m", "constant offset (m)"),
)


def scene_metric(scene: Dict[str, Any], name: str) -> Optional[float]:
    """Read one aggregate-scope scalar, tolerating reports that lack it."""
    scopes = scene.get("scopes")
    if not isinstance(scopes, dict):
        return None
    metrics = (scopes.get("aggregate") or {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    node = metrics.get(name)
    if isinstance(node, dict):
        node = node.get("median")
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def saw_ball(scene: Dict[str, Any], view: str, frame: int) -> Optional[bool]:
    """Did `view` see the ball at `frame`? None when the report cannot say."""
    records = scene.get("records")
    if not isinstance(records, list):
        return None
    for record in records:
        if record.get("view") == view and record.get("frame") == frame:
            return bool(record.get("ball_visible"))
    return None


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("--view", default="lower_front",
                        help="camera to split on (default: lower_front)")
    parser.add_argument("--frame", type=int, default=15,
                        help="target index to test visibility at (default: 15, "
                             "the terminal observation under any window offset)")
    cli = parser.parse_args()

    if not cli.report.is_file():
        parser.error(f"not a file: {cli.report}")
    with cli.report.open() as handle:
        report = json.load(handle)
    scenes = report.get("per_scene")
    if not isinstance(scenes, list) or not scenes:
        parser.error("report has no per_scene entries")

    groups: Dict[str, List[Dict[str, Any]]] = {"visible": [], "occluded": []}
    unknown = 0
    for scene in scenes:
        seen = saw_ball(scene, cli.view, cli.frame)
        if seen is None:
            unknown += 1
            continue
        groups["visible" if seen else "occluded"].append(scene)

    total = len(groups["visible"]) + len(groups["occluded"])
    if unknown:
        print(f"WARNING: {unknown} scenes carry no record for view "
              f"{cli.view} at frame {cli.frame}; they are excluded.")
    if not total:
        print(f"[FAIL] no scene records mention view {cli.view}. "
              f"Views present: {sorted({r.get('view') for s in scenes for r in s.get('records', [])})}")
        return 1

    share = len(groups["visible"]) / total
    print(f"Report      : {cli.report}")
    print(f"Split view  : {cli.view} at frame {cli.frame}")
    print(f"Scenes      : {len(groups['visible'])} visible, "
          f"{len(groups['occluded'])} occluded, {total} usable "
          f"({share:.0%} see three views)")
    print("")
    print(f"{'metric':<34}{'3 views':>10}{'2 views':>10}{'ratio':>9}  n(3)/n(2)")
    print("-" * 78)

    for name, label in METRICS:
        cells = {}
        for key, scene_list in groups.items():
            values = [v for v in (scene_metric(s, name) for s in scene_list) if v is not None]
            cells[key] = (percentile(values, 50), len(values))
        vis, occ = cells["visible"][0], cells["occluded"][0]
        ratio = f"{occ / vis:.2f}x" if (vis and occ and vis > 0) else "n/a"
        text = lambda v: "n/a" if v is None else f"{v:.4f}"
        print(f"{label:<34}{text(vis):>10}{text(occ):>10}{ratio:>9}  "
              f"{cells['visible'][1]}/{cells['occluded'][1]}")

    print("")
    print("ratio = 2-view median / 3-view median. Above 1.00 means the third")
    print("view helps. A landing ratio near 1.00 says opening the catch ring")
    print("buys little and the gain in the new dataset must come from")
    print("elsewhere; well above 1.00 makes the third view a real lever and")
    print("means results on the new data are not comparable to the old.")
    print("")
    print("Medians only. With a hundred scenes split two ways, a p95 would be")
    print("the second or third worst sample in a group and would not carry a")
    print("conclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
