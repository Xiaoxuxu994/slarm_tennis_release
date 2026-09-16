#!/usr/bin/env python
"""Check the two assumptions behind catch_position_inplane, against the data.

The evaluator builds the catch point and the ring axis like this:

    gt_catch = gt_pos[15] + gt_v[15] * dt + 0.5 * g * dt^2
    ring axis = (gt_v[15] + g * dt) normalised

Both are ASSUMPTIONS, and this reads the scene annotations to test them.

1. Is the catch frame stored? position_rig / velocity_rig are per frame, so if
   the annotation runs past the catch frame the truth is on disk and does not
   have to be extrapolated. The analytic continuation is exact only while the
   simulator has no drag; comparing the two measures whatever it does have.

2. How much does the flight direction vary between scenes? The evaluator
   assumes the ring faces the incoming ball, which is only reasonable if the
   ring can be oriented per throw. If every scene arrives from nearly the same
   direction, a fixed ring orientation is equally defensible and the two agree.
   If the spread is wide, the assumption has to come from the rig, not here.

3. How much does the answer move if the axis is wrong? A ring held at a fixed
   orientation differs from a velocity-aligned one by the spread in 2, and the
   in-plane component changes by roughly (axial error) x sin(that angle).

    python tools/check_catch_geometry.py --data-root data/slarm_data \\
        --annotation scene_list/ball_catch_triview_0908_10k_validation.txt \\
        --catch-frame 45 [--limit 200]

All output is ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

GRAVITY_RIG = (0.0, 0.0, -9.81)
TERMINAL_FRAME = 15


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


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def unit(v: Sequence[float]) -> Tuple[float, ...]:
    n = norm(v)
    return tuple(x / n for x in v) if n > 0 else tuple(v)


def angle_between(a: Sequence[float], b: Sequence[float]) -> float:
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(unit(a), unit(b)))))
    return math.degrees(math.acos(dot))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True,
                        help="scene list, relative to data-root or absolute")
    parser.add_argument("--catch-frame", type=int, default=45)
    parser.add_argument("--limit", type=int, default=200)
    cli = parser.parse_args()

    manifest = cli.annotation if cli.annotation.is_absolute() else cli.data_root / cli.annotation
    if not manifest.is_file():
        raise SystemExit(f"[FAIL] no such manifest: {manifest}")
    entries = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    if not entries:
        raise SystemExit(f"[FAIL] manifest is empty: {manifest}")
    entries = entries[: cli.limit]

    stored_frames: List[int] = []
    directions: List[Tuple[float, ...]] = []
    from_vertical: List[float] = []
    speeds: List[float] = []
    extrapolation_error: List[float] = []
    skipped = 0

    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = cli.data_root / entry
        if not path.is_file():
            skipped += 1
            continue
        with path.open() as handle:
            scene = json.load(handle)
        frames = (scene.get("ball_trajectory") or {}).get("frames")
        times = scene.get("normalized_time")
        if not isinstance(frames, list) or not isinstance(times, list) or len(frames) <= TERMINAL_FRAME:
            skipped += 1
            continue
        stored_frames.append(len(frames))

        p15 = frames[TERMINAL_FRAME].get("position_rig")
        v15 = frames[TERMINAL_FRAME].get("velocity_rig")
        if not (isinstance(p15, list) and isinstance(v15, list)):
            skipped += 1
            continue

        # normalized_time is in seconds despite the name (see check_dataset_contract).
        step = float(times[1]) - float(times[0]) if len(times) > 1 else 0.0
        if step <= 0:
            skipped += 1
            continue
        dt = (cli.catch_frame - TERMINAL_FRAME) * step

        v_catch = [a + b * dt for a, b in zip(v15, GRAVITY_RIG)]
        directions.append(unit(v_catch))
        speeds.append(norm(v_catch))
        from_vertical.append(angle_between(v_catch, (0.0, 0.0, -1.0)))

        # Only when the catch frame is actually on disk can the continuation be checked.
        if len(frames) > cli.catch_frame:
            stored = frames[cli.catch_frame].get("position_rig")
            if isinstance(stored, list):
                predicted = [p + v * dt + 0.5 * g * dt * dt
                             for p, v, g in zip(p15, v15, GRAVITY_RIG)]
                extrapolation_error.append(norm([a - b for a, b in zip(predicted, stored)]))

    if not directions:
        raise SystemExit(f"[FAIL] no usable scenes ({skipped} skipped). Check --data-root.")

    print(f"manifest    : {manifest}")
    print(f"scenes read : {len(directions)}" + (f"  ({skipped} skipped)" if skipped else ""))
    print(f"catch frame : {cli.catch_frame}, terminal frame {TERMINAL_FRAME}")
    print("")

    shortest = min(stored_frames)
    print("1. IS THE CATCH FRAME ON DISK?")
    print(f"   stored frames per scene: min {shortest}, max {max(stored_frames)}")
    if extrapolation_error:
        print(f"   frame {cli.catch_frame} IS stored in {len(extrapolation_error)} scenes, so the")
        print("   analytic continuation can be checked against it:")
        print(f"     median {percentile(extrapolation_error, 50):.4f} m   "
              f"p95 {percentile(extrapolation_error, 95):.4f} m   "
              f"max {max(extrapolation_error):.4f} m")
        print("   Anything above a millimetre is unmodelled physics (drag), and the")
        print("   evaluator should read the stored truth instead of extrapolating.")
    else:
        print(f"   frame {cli.catch_frame} is NOT stored (scenes end at {shortest - 1}).")
        print("   The catch truth has to be extrapolated, which is exact only while")
        print("   the simulator has no drag. Ask the data side to export past it.")
    print("")

    print("2. DOES THE BALL ALWAYS ARRIVE FROM THE SAME DIRECTION?")
    print(f"   angle from straight down: median {percentile(from_vertical, 50):.1f} deg, "
          f"p5 {percentile(from_vertical, 5):.1f}, p95 {percentile(from_vertical, 95):.1f}")
    print(f"   catch speed            : median {percentile(speeds, 50):.2f} m/s")
    mean_direction = unit([sum(d[i] for d in directions) for i in range(3)])
    spread = [angle_between(d, mean_direction) for d in directions]
    print(f"   spread about the mean  : median {percentile(spread, 50):.1f} deg, "
          f"p95 {percentile(spread, 95):.1f} deg, max {max(spread):.1f} deg")
    print("")

    print("3. HOW MUCH WOULD A WRONG AXIS COST?")
    print("   In-plane error moves by about (axial error) x sin(axis error).")
    axial_median_m = 0.0427     # measured on ckpt_013999, 200 validation scenes
    for label, degrees in (("p95 spread", percentile(spread, 95)),
                           ("median angle from vertical", percentile(from_vertical, 50))):
        shift = axial_median_m * math.sin(math.radians(degrees))
        print(f"   {label:<28} {degrees:5.1f} deg  ->  {shift * 100:.2f} cm "
              f"({shift / 0.0550:.0%} of the 5.50 cm in-plane median)")
    print("")
    print("Read 2 this way: a tight spread means a fixed ring orientation and a")
    print("velocity-aligned one are nearly the same thing, so the decomposition is")
    print("safe. A wide spread means the ring cannot face every throw, and the")
    print("orientation has to come from the rig rather than from this assumption.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
