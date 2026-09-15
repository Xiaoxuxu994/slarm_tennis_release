#!/usr/bin/env python
"""Print an evaluation report's per-camera scopes side by side.

Why this exists. The markdown report renders only the aggregate scope, but
`evaluation.json` carries one scope per named view, and the headline
frame24/catch numbers take the WORST view. When a report degrades, whether the
loss is spread across the rig or concentrated in one camera is the difference
between "the model needs to adapt everywhere" and "one view broke", and those
call for different responses. The aggregate number cannot tell them apart.

    python tools/show_per_view.py <evaluation.json> [<second.json> ...]

All output is ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# (label, metric key, sub key). Reconstruction quality first because a view
# that lost the ball while still rendering the room is the informative case.
ROWS: Tuple[Tuple[str, str, str], ...] = (
    ("ball IoU anchor", "ball_iou", "anchor"),
    ("ball IoU farthest", "ball_iou", "farthest"),
    ("ball depth med farthest", "ball_depth_error_median", "farthest"),
    ("ball depth p95 farthest", "ball_depth_error_p95", "farthest"),
    ("ball RGB PSNR anchor", "ball_rgb_psnr", "anchor"),
    ("semantic mIoU anchor", "semantic_miou", "anchor"),
    ("semantic mIoU farthest", "semantic_miou", "farthest"),
    ("RGB PSNR anchor", "rgb_psnr", "anchor"),
    ("RGB PSNR farthest", "rgb_psnr", "farthest"),
    ("depth absrel farthest", "depth_absrel", "farthest"),
    ("ms3 ball velocity med", "ms3_ball_velocity", "median"),
    ("ms3 ball velocity p95", "ms3_ball_velocity", "p95"),
)


def read_scopes(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open() as handle:
        report = json.load(handle)
    scopes = report.get("scope_reports")
    if not isinstance(scopes, dict) or not scopes:
        raise SystemExit(f"[FAIL] {path} has no scope_reports")
    return {name: entry.get("metrics", {}) for name, entry in scopes.items()}


def value(metrics: Dict[str, Any], key: str, sub: str) -> Optional[float]:
    node = metrics.get(key)
    if isinstance(node, dict):
        node = node.get(sub)
    try:
        number = float(node)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path)
    cli = parser.parse_args()

    for path in cli.reports:
        if not path.is_file():
            raise SystemExit(f"[FAIL] not a file: {path}")
        scopes = read_scopes(path)
        # aggregate first, then the named views in report order
        names: List[str] = (["aggregate"] if "aggregate" in scopes else [])
        names += [n for n in scopes if n != "aggregate"]

        width = max(12, max(len(n) for n in names) + 2)
        header = f"{'metric':<26}" + "".join(f"{n:>{width}}" for n in names)
        print("=" * len(header))
        print(path)
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for label, key, sub in ROWS:
            cells = [value(scopes[n], key, sub) for n in names]
            if all(c is None for c in cells):
                continue
            text = "".join(
                f"{'n/a' if c is None else format(c, '.4f'):>{width}}" for c in cells
            )
            print(f"{label:<26}{text}")
        print("")
    print("The headline frame24 and catch numbers take the WORST named view, so")
    print("one broken camera sets them by itself. A ball metric that collapses in")
    print("a single column while the RGB/depth columns stay level means that view")
    print("stopped rendering the ball, not that the scene got harder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
