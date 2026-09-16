#!/usr/bin/env python
"""Turn a Stream25 report's landing errors into a catch success rate.

The rig catches the ball through a ring, so success is geometric: the ball
passes only while its whole body clears the opening, which puts its centre
within (ring radius - ball radius) of the ring centre. At the current 27 cm
ring and 6.5 cm ball that is 0.1025 m.

That is TIGHTER than the 0.1196 m threshold scattered through this repo, which
corresponds to a 30.4 cm ring and is documented in those tools as a diagnostic
distance rather than robot success. Hit rates quoted against it are optimistic
by one ring size; --threshold reproduces them when comparing with old notes.

No rerun is needed: every report already stores catch_position per scene.

    python tools/catch_success_rate.py <evaluation.json> [<more.json> ...]
    python tools/catch_success_rate.py report.json --ring-diameter 0.30

## What this number is, and is not

It counts scenes whose predicted landing sits within the aperture of a ring
centred on the true landing. That is a necessary condition for a catch and a
useful single number, but it is not the robot's success rate:

  - By default it uses the 3D distance, while only the component IN the ring
    plane decides whether the ball clears the opening. Error along the ring
    axis arrives as early or late rather than wide, so the 3D rate is a LOWER
    bound. Reports produced after the decomposition landed also carry
    catch_position_inplane, which is the geometrically honest number:

        python tools/catch_success_rate.py report.json \
          --metric catch_position_inplane

    Older reports have only the norm, and the split cannot be recovered from
    it, so they have to be re-evaluated to get the better number.
  - It assumes the ring is placed exactly at the true landing point. A real
    arm also has to get there, with its own reach, timing and control error.

Read it as a lower bound on the perception-limited rate.

All output is ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RING_DIAMETER_M = 0.27
BALL_DIAMETER_M = 0.065
LEGACY_THRESHOLD_M = 0.1196


def find_report(path: Path) -> Path:
    if path.is_file():
        if path.suffix == ".json":
            return path
        sibling = path.with_suffix(".json")
        if sibling.is_file():
            return sibling
        raise SystemExit(f"[FAIL] {path} is not JSON and {sibling.name} is not beside it")
    if path.is_dir():
        found = sorted(path.rglob("evaluation.json"))
        if len(found) == 1:
            return found[0]
        if not found:
            raise SystemExit(f"[FAIL] no evaluation.json under {path}")
        raise SystemExit("[FAIL] several reports there; name one:\n  "
                         + "\n  ".join(str(f) for f in found[:10]))
    raise SystemExit(f"[FAIL] no such file or directory: {path}")


def scene_values(report: Dict[str, Any], metric: str) -> List[float]:
    """Per-scene aggregate-scope values for one scalar metric."""
    out: List[float] = []
    for scene in report.get("per_scene") or []:
        node = ((scene.get("scopes") or {}).get("aggregate") or {}).get("metrics", {})
        value = node.get(metric)
        if isinstance(value, dict):
            value = value.get("median")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def wilson(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """95% interval for a proportion. At a hundred scenes this is +-8 points,
    which is wider than most differences between neighbouring checkpoints."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


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
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--ring-diameter", type=float, default=RING_DIAMETER_M,
                        help=f"inner diameter of the catch ring, metres (default {RING_DIAMETER_M})")
    parser.add_argument("--ball-diameter", type=float, default=BALL_DIAMETER_M,
                        help=f"ball diameter, metres (default {BALL_DIAMETER_M})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the geometric tolerance, metres; use "
                             f"{LEGACY_THRESHOLD_M} to reproduce older notes")
    parser.add_argument("--metric", default="catch_position",
                        help="per-scene landing metric (default catch_position)")
    cli = parser.parse_args()

    if min(cli.ring_diameter, cli.ball_diameter) <= 0:
        parser.error("diameters must be positive")
    if cli.ball_diameter >= cli.ring_diameter:
        parser.error("the ball does not fit through the ring")

    # Resolve every path before printing anything, so a typo fails on its own
    # rather than after a header that looks like the run started.
    paths = [find_report(raw) for raw in cli.reports]

    ring_r, ball_r = cli.ring_diameter / 2, cli.ball_diameter / 2
    clears = ring_r - ball_r
    tolerance = cli.threshold if cli.threshold is not None else clears
    if tolerance <= 0:
        parser.error("threshold must be positive")

    print(f"ring {cli.ring_diameter * 100:.1f} cm, ball {cli.ball_diameter * 100:.1f} cm")
    print(f"  ball fully clears the opening   centre within {clears:.4f} m   <- criterion")
    print(f"  ball centre still inside ring   centre within {ring_r:.4f} m")
    print(f"  ball edge grazes ring edge      centre within {ring_r + ball_r:.4f} m")
    if cli.threshold is not None:
        print(f"  OVERRIDDEN to {tolerance:.4f} m")
    print("")

    label_width = max(28, max(len(p.parent.name) for p in paths) + 2)
    print(f"{'report':<{label_width}}{'n':>5}{'success':>10}{'95% CI':>16}"
          f"{'median':>9}{'p95':>9}")
    print("-" * (label_width + 49))

    for path in paths:
        with path.open() as handle:
            report = json.load(handle)
        values = scene_values(report, cli.metric)
        label = path.parent.name or path.stem
        if not values:
            print(f"{label:<{label_width}}{'-':>5}  no {cli.metric} in per_scene "
                  f"(older report? try --metric frame24_position)")
            continue
        hits = sum(1 for v in values if v < tolerance)
        low, high = wilson(hits, len(values))
        med, p95 = percentile(values, 50), percentile(values, 95)
        print(f"{label:<{label_width}}{len(values):>5}{hits / len(values):>9.1%}"
              f"{f'[{low:.1%}, {high:.1%}]':>16}{med:>9.4f}{p95:>9.4f}")

    print("")
    print("Sensitivity: the same scenes scored against the other two criteria,")
    print("so a rate that moves a lot between them is one where many scenes sit")
    print("near the rim rather than comfortably inside or outside.")
    print("")
    for path in paths:
        with path.open() as handle:
            values = scene_values(json.load(handle), cli.metric)
        if not values:
            continue
        cells = "".join(
            f"{sum(1 for v in values if v < limit) / len(values):>12.1%}"
            for limit in (clears, ring_r, ring_r + ball_r, LEGACY_THRESHOLD_M)
        )
        print(f"{(path.parent.name or path.stem):<{label_width}}{cells}")
    print(f"{'':<{label_width}}{'clears':>12}{'centre in':>12}{'grazes':>12}{'legacy':>12}")
    print("")
    print("A catch also needs the arm to reach the point in time; this counts")
    print("only whether perception put the landing inside the ring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
