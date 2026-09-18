#!/usr/bin/env python
"""Rank the evaluated scenes, so a visualization run knows which to render.

Rendering twenty scenes to find a good one spends a GPU on a question the
evaluation report already answered: it carries every metric for every scene.
This reads them and ranks, which costs nothing and covers all two hundred
rather than the twenty someone had patience for.

    python tools/pick_scenes.py <evaluation.json> [--top 5]

## Best is usually the wrong one to show

A slide built from the best scene says what the model can do at its luckiest,
and the first person to run the system on their own clip will not see it. So
this prints the best, the median and the worst together. The median scene is
the honest headline; the best and worst belong beside it, labelled, as the
range. Cherry-picking is only a problem when it is silent.

Two rankings, because they answer different questions:

  landing   catch_position, how far the predicted landing lands from the true
            one. This is the task metric.
  ball      ball_iou at the anchor and far frames, how well the ball itself is
            reconstructed. A scene can land well while rendering the ball badly
            -- the landing only needs the centre, not the shape.

All output is ASCII.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_report(path: Path) -> Path:
    if path.is_file():
        return path if path.suffix == ".json" else path.with_suffix(".json")
    if path.is_dir():
        found = sorted(path.rglob("evaluation.json"))
        if len(found) == 1:
            return found[0]
        if not found:
            raise SystemExit(f"[FAIL] no evaluation.json under {path}")
        raise SystemExit("[FAIL] several reports there; name one:\n  "
                         + "\n  ".join(str(f) for f in found[:10]))
    raise SystemExit(f"[FAIL] no such file or directory: {path}")


def metric(scene: Dict[str, Any], name: str, sub: str = "median") -> Optional[float]:
    node = ((scene.get("scopes") or {}).get("aggregate") or {}).get("metrics", {}).get(name)
    if isinstance(node, dict):
        node = node.get(sub)
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def rank_of(values: List[Optional[float]], lower_is_better: bool) -> List[Optional[float]]:
    """Percentile rank in [0, 1], 1 being best, None where the metric is absent."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return [None] * len(values)
    out: List[Optional[float]] = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        below = sum(1 for p in present if p < value)
        fraction = below / max(1, len(present) - 1)
        out.append(1.0 - fraction if lower_is_better else fraction)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", type=Path)
    parser.add_argument("--top", type=int, default=5, help="how many to list per ranking")
    parser.add_argument("--ball-metric", default="ball_iou",
                        help="reconstruction metric to rank on (default ball_iou)")
    parser.add_argument("--ball-bucket", default="anchor",
                        choices=("anchor", "interpolation", "near", "mid", "far", "farthest"))
    cli = parser.parse_args()

    path = find_report(cli.report)
    with path.open() as handle:
        report = json.load(handle)
    scenes = report.get("per_scene")
    if not isinstance(scenes, list) or not scenes:
        raise SystemExit(f"[FAIL] {path} has no per_scene entries")

    landing = [metric(s, "catch_position") for s in scenes]
    if not any(v is not None for v in landing):
        landing = [metric(s, "frame24_position") for s in scenes]
        landing_name = "frame24_position"
        print("note: no catch_position in this report; ranking on frame24_position.")
    else:
        landing_name = "catch_position"
    ball = [metric(s, cli.ball_metric, cli.ball_bucket) for s in scenes]

    landing_rank = rank_of(landing, lower_is_better=True)
    ball_rank = rank_of(ball, lower_is_better=False)

    rows = []
    for index, scene in enumerate(scenes):
        combined = [r for r in (landing_rank[index], ball_rank[index]) if r is not None]
        rows.append({
            "index": scene.get("scene_index", index),
            "name": str(scene.get("scene_name", index)),
            "landing": landing[index],
            "ball": ball[index],
            "combined": sum(combined) / len(combined) if combined else None,
        })

    print(f"report : {path}")
    print(f"scenes : {len(rows)}")
    print(f"ranked : {landing_name} (lower better), "
          f"{cli.ball_metric}.{cli.ball_bucket} (higher better)")
    print("")

    def show(title: str, ordered: List[Dict[str, Any]], note: str = "") -> None:
        print(f"{title}{('   ' + note) if note else ''}")
        print(f"  {'scene':>6}  {'name':<22}{'landing':>10}{'ball':>9}{'combined':>10}")
        for row in ordered:
            landing_text = "n/a" if row["landing"] is None else f"{row['landing']:.4f}"
            ball_text = "n/a" if row["ball"] is None else f"{row['ball']:.4f}"
            combined_text = "n/a" if row["combined"] is None else f"{row['combined']:.2f}"
            print(f"  {row['index']:>6}  {row['name'][:22]:<22}"
                  f"{landing_text:>10}{ball_text:>9}{combined_text:>10}")
        print("")

    scored = [r for r in rows if r["combined"] is not None]
    scored.sort(key=lambda r: r["combined"], reverse=True)
    if scored:
        show("BEST overall", scored[:cli.top],
             "-- shows what the model does at its luckiest")
        middle = len(scored) // 2
        span = max(1, cli.top // 2)
        show("TYPICAL (median)", scored[max(0, middle - span):middle + span + 1],
             "-- this is the honest headline")
        show("WORST overall", list(reversed(scored[-cli.top:])),
             "-- the failure mode a viewer will meet")

    by_landing = sorted((r for r in rows if r["landing"] is not None),
                        key=lambda r: r["landing"])
    if by_landing:
        show("Smallest landing error", by_landing[:cli.top])
    by_ball = sorted((r for r in rows if r["ball"] is not None),
                     key=lambda r: r["ball"], reverse=True)
    if by_ball:
        show("Best ball reconstruction", by_ball[:cli.top])

    overlap = ({r["index"] for r in by_landing[:cli.top]}
               & {r["index"] for r in by_ball[:cli.top]})
    print(f"In both top {cli.top}: "
          + (", ".join(str(i) for i in sorted(overlap)) if overlap else "none"))
    print("A scene can land well while rendering the ball badly: the landing only")
    print("needs the ball's centre, not its shape. Little overlap is normal and")
    print("means one scene will not illustrate both.")
    print("")
    if scored:
        best = scored[0]
        print("Render it with:")
        print(f"  python tools/export_ball_track.py --config <cfg> --checkpoint <ckpt> \\")
        print(f"      --scene {best['index']} --num-frames 46 --output track_{best['index']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
