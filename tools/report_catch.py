#!/usr/bin/env python
"""One command from evaluation.json to the numbers a report needs.

Reads the landing errors and prints, per checkpoint: the frame-24 error, the
catch-frame error, and the catch success rate under both distance criteria,
plus the velocity and position errors that explain where a change came from.

    python tools/report_catch.py <evaluation.json> [<more.json> ...]
    python tools/report_catch.py work_dirs/slarm/stream25_eval/<cfg>/ --markdown

Directories are searched for evaluation.json, so a whole checkpoint sweep can
be passed at once and comes back as one table in checkpoint order.

Success is geometric: the ball passes only while its whole body clears the
opening, so its centre must land within (ring radius - ball radius) of the true
point, 0.1025 m for a 27 cm ring and a 6.5 cm ball. Two distances are scored
against that same threshold:

  3D        the full distance between predicted and true landing. Counts error
            along the flight direction as a miss when it is really early or
            late arrival, so this is a LOWER BOUND, true whatever the ring's
            orientation.
  in-plane  only the component across the flight direction, which is what
            carries the ball off-centre through the opening. Assumes the ring
            faces the incoming ball; measured throw-direction spread is a few
            degrees, so a fixed ring and a velocity-aligned one nearly agree.

Report the 3D number and footnote the in-plane one. For exploring other
criteria (ball centre inside the rim, edge grazing the rim) see
tools/catch_success_rate.py.

All output is ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RING_DIAMETER_M = 0.27
BALL_DIAMETER_M = 0.065


def find_reports(paths: List[Path]) -> List[Path]:
    found: List[Path] = []
    for path in paths:
        if path.is_file():
            found.append(path if path.suffix == ".json" else path.with_suffix(".json"))
        elif path.is_dir():
            inside = sorted(path.rglob("evaluation.json"))
            if not inside:
                raise SystemExit(f"[FAIL] no evaluation.json under {path}")
            found.extend(inside)
        else:
            raise SystemExit(f"[FAIL] no such file or directory: {path}")
    for path in found:
        if not path.is_file():
            raise SystemExit(f"[FAIL] not a file: {path}")
    # ckpt_009999 must sort before ckpt_013999, which string order gets wrong.
    def key(path: Path) -> Tuple[int, str]:
        digits = re.findall(r"\d+", path.parent.name)
        return (int(digits[-1]) if digits else -1, str(path))
    return sorted(dict.fromkeys(found), key=key)


def scene_values(report: Dict[str, Any], metric: str) -> List[float]:
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


def summary(report: Dict[str, Any], metric: str, sub: str) -> Optional[float]:
    node = (report.get("metrics") or {}).get(metric)
    if isinstance(node, dict):
        node = node.get(sub)
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile, matching the evaluator's finite_percentile.

    Computed from per_scene rather than read off the report's own aggregate, so
    that median, p90 and p95 all come from one place. The report only ever
    aggregates 50 and 95, and p90 is the one a reader asks for next: p95 at two
    hundred scenes sits on the tenth worst case and moves a lot between runs,
    while p90 rests on twenty and is far steadier.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    low = int(math.floor(position))
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def wilson(hits: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def rate(values: List[float], tolerance: float) -> Optional[Tuple[float, float, float, int]]:
    if not values:
        return None
    hits = sum(1 for v in values if v < tolerance)
    low, high = wilson(hits, len(values))
    return (hits / len(values), low, high, len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--ring-diameter", type=float, default=RING_DIAMETER_M)
    parser.add_argument("--ball-diameter", type=float, default=BALL_DIAMETER_M)
    parser.add_argument("--markdown", action="store_true",
                        help="emit markdown tables to paste into a report")
    cli = parser.parse_args()

    if cli.ball_diameter >= cli.ring_diameter or min(cli.ring_diameter, cli.ball_diameter) <= 0:
        parser.error("need a positive ball diameter smaller than the ring")
    tolerance = (cli.ring_diameter - cli.ball_diameter) / 2

    rows = []
    for path in find_reports(cli.reports):
        with path.open() as handle:
            report = json.load(handle)
        landing = scene_values(report, "catch_position")
        inplane = scene_values(report, "catch_position_inplane")
        frame24 = scene_values(report, "frame24_position")
        rows.append({
            "f24": {q: percentile(frame24, q) for q in (50, 90, 95)},
            "catch": {q: percentile(landing, q) for q in (50, 90, 95)},
            "ip": {q: percentile(inplane, q) for q in (50, 90, 95)},
            "label": path.parent.name or path.stem,
            "scenes": report.get("scene_count"),
            "split": report.get("split"),
            "horizon": summary(report, "catch_horizon_s", "median"),
            "f24_med": summary(report, "frame24_position", "median"),
            "f24_p95": summary(report, "frame24_position", "p95"),
            "catch_med": summary(report, "catch_position", "median"),
            "catch_p95": summary(report, "catch_position", "p95"),
            "inplane_med": summary(report, "catch_position_inplane", "median"),
            "axial_med": summary(report, "catch_position_axial", "median"),
            "rate3d": rate(scene_values(report, "catch_position"), tolerance),
            "rate_ip": rate(scene_values(report, "catch_position_inplane"), tolerance),
            "vel": summary(report, "ms3_ball_velocity", "median"),
            "pos": summary(report, "ball_pos15_error_fit", "median"),
            "iou": summary(report, "ball_iou", "anchor"),
            "overall": report.get("overall"),
        })
    if not rows:
        raise SystemExit("[FAIL] nothing to report")

    def metres(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    def percent(cell) -> str:
        return "n/a" if cell is None else f"{cell[0]:.1%}"

    def interval(cell) -> str:
        return "n/a" if cell is None else f"{cell[1]:.1%}-{cell[2]:.1%}"

    scenes = {row["scenes"] for row in rows if row["scenes"]}
    horizons = {round(row["horizon"], 3) for row in rows if row["horizon"]}

    print("=" * 96)
    print("Stream25 catch report")
    print("=" * 96)
    print(f"  split      : {'/'.join(sorted({str(r['split']) for r in rows}))}"
          f"   scenes: {'/'.join(str(s) for s in sorted(scenes)) if scenes else 'n/a'}")
    print(f"  criterion  : centre within {tolerance:.4f} m "
          f"(ring {cli.ring_diameter*100:.0f} cm minus ball {cli.ball_diameter*100:.1f} cm; "
          f"the ball fully clears the opening)")
    print(f"  horizon    : {'/'.join(f'{h:.3f}' for h in sorted(horizons)) if horizons else 'n/a'} s "
          f"from the terminal observation to the catch frame")
    print(f"  statistics : median across scenes, worst named view, Wilson 95% interval")
    if len(rows) > 1:
        print(f"  note       : at these scene counts an interval spans about +-6 points, "
              f"wider than the gap between neighbouring checkpoints")
    print("")

    width = max(22, max(len(row["label"]) for row in rows) + 2)

    print("Landing error, metres")
    header = (f"{'model':<{width}}"
              f"{'frame24 med':>13}{'p90':>9}{'p95':>9}"
              f"{'catch med':>12}{'p90':>9}{'p95':>9}"
              f"{'in-plane med':>15}{'p90':>9}{'p95':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = "".join(metres(row[group][q]).rjust(w)
                        for group, first in (("f24", 13), ("catch", 12), ("ip", 15))
                        for q, w in ((50, first), (90, 9), (95, 9)))
        print(f"{row['label']:<{width}}{cells}")
    print("")

    print(f"Catch success, ball centre within {tolerance:.4f} m")
    header = (f"{'model':<{width}}{'3D':>10}{'95% CI':>17}"
              f"{'in-plane':>12}{'95% CI':>17}{'scenes':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        scored = row["rate3d"][3] if row["rate3d"] else 0
        print(f"{row['label']:<{width}}{percent(row['rate3d']):>10}{interval(row['rate3d']):>17}"
              f"{percent(row['rate_ip']):>12}{interval(row['rate_ip']):>17}{scored:>9}")
    print("")

    print("Where the landing error comes from. It is dominated by velocity:")
    print("  catch ~ sqrt((velocity error x horizon)^2 + (position error)^2)")
    print("")
    sub = (f"{'model':<{width}}{'velocity':>11}{'position':>10}{'predicted':>11}"
           f"{'measured':>10}{'ball IoU':>10}{'gates':>8}")
    print(sub)
    print("-" * len(sub))
    for row in rows:
        predicted = None
        if row["vel"] is not None and row["pos"] is not None and row["horizon"]:
            predicted = math.hypot(row["vel"] * row["horizon"], row["pos"])
        print(f"{row['label']:<{width}}{metres(row['vel']):>11}{metres(row['pos']):>10}"
              f"{metres(predicted):>11}{metres(row['catch_med']):>10}"
              f"{metres(row['iou']):>10}{str(row['overall'] or 'n/a'):>8}")
    print("")
    print("  'predicted' rebuilds the catch error from velocity and position alone.")
    print("  It should sit near 'measured'; a large gap means one of them is not")
    print("  measuring what the name says.")
    print("")

    if any(row["rate_ip"] for row in rows):
        print("Footnote for the report:")
        print("  The success rate above uses the 3D distance between predicted and true")
        print("  landing and is a conservative LOWER BOUND: error along the flight")
        print("  direction only makes the ball arrive early or late, it does not carry")
        print("  it off-centre. Scored on the across-flight component instead, the rate")
        print("  is the in-plane column.")
        for row in rows:
            if row["axial_med"] is not None and row["catch_med"]:
                delay = row["axial_med"] / 7.72      # catch speed, m/s
                print(f"  {row['label']}: axial {row['axial_med']:.4f} m = {delay*1000:.1f} ms of timing.")
        print("")
    print("  The catch frame is not in the exported data (scenes end at frame 24), so")
    print("  the catch truth is an analytic continuation that assumes no air drag and")
    print("  has never been checked against stored ground truth.")

    if cli.markdown:
        print("")
        print("=" * 96)
        print("Markdown")
        print("=" * 96)
        print("")
        print("| model | catch med | catch p90 | catch p95 | success (3D) | 95% CI |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            print(f"| {row['label']} | {metres(row['catch'][50])} | {metres(row['catch'][90])} "
                  f"| {metres(row['catch'][95])} | {percent(row['rate3d'])} | {interval(row['rate3d'])} |")
        print("")
        print("| model | velocity (m/s) | position (m) | ball IoU anchor |")
        print("| --- | ---: | ---: | ---: |")
        for row in rows:
            print(f"| {row['label']} | {metres(row['vel'])} | {metres(row['pos'])} "
                  f"| {metres(row['iou'])} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
