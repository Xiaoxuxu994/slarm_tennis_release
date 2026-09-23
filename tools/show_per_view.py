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


def find_report(path: Path) -> Path:
    """Accept the json itself, or a directory holding exactly one."""
    if path.is_file():
        if path.suffix == ".json":
            return path
        # The markdown report sits beside the json and is the easier name to
        # reach for; it holds only the aggregate scope, which is the very table
        # this tool exists to go beyond.
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            print(f"note: {path.name} is the markdown report; reading {sibling.name}.")
            return sibling
        raise SystemExit(f"[FAIL] {path} is not JSON and {sibling.name} is not beside it")
    if path.suffix and not path.exists():
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            return sibling
    if path.is_dir():
        found = sorted(path.rglob("evaluation.json"))
        if len(found) == 1:
            return found[0]
        if not found:
            raise SystemExit(f"[FAIL] no evaluation.json under {path}")
        raise SystemExit("[FAIL] several reports under that directory; name one:\n  "
                         + "\n  ".join(str(f) for f in found[:10]))
    raise SystemExit(f"[FAIL] no such file or directory: {path}\n"
                     f"       eval.sh writes to "
                     f"output/stream25_eval/<config>/<ckpt>/evaluation.json")


def read_scopes(path: Path) -> Dict[str, Dict[str, Any]]:
    """Per-view metrics, from the top-level scopes or rebuilt from per_scene.

    Reports written before scope_reports existed still carry every scope inside
    each scene, so falling back to those keeps this usable on older runs rather
    than failing on a report that does contain the answer.
    """
    with path.open() as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise SystemExit(f"[FAIL] {path} is not a JSON object")

    scopes = report.get("scope_reports")
    if isinstance(scopes, dict) and scopes:
        return {name: entry.get("metrics", {}) for name, entry in scopes.items()}

    scenes = report.get("per_scene")
    if isinstance(scenes, list) and scenes:
        names: List[str] = []
        for scene in scenes:
            for name in (scene.get("scopes") or {}):
                if name not in names:
                    names.append(name)
        if names:
            print(f"note: {path.name} has no scope_reports; "
                  f"taking the median across {len(scenes)} scenes instead.")
            rebuilt: Dict[str, Dict[str, Any]] = {}
            for name in names:
                merged: Dict[str, Any] = {}
                for key, sub in {(k, s) for _, k, s in ROWS}:
                    values = []
                    for scene in scenes:
                        node = ((scene.get("scopes") or {}).get(name) or {}).get("metrics", {})
                        number = value(node, key, sub)
                        if number is not None:
                            values.append(number)
                    if values:
                        merged.setdefault(key, {})[sub] = sorted(values)[len(values) // 2]
                rebuilt[name] = merged
            return rebuilt

    raise SystemExit(
        f"[FAIL] {path} carries neither scope_reports nor per_scene scopes.\n"
        f"       top-level keys present: {sorted(report)[:14]}"
    )


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
        path = find_report(path)
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
