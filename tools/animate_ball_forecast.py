#!/usr/bin/env python
"""Animate the forecast tightening as observations arrive, on the pixel path.

The still 3D plot shows where the ball went. It cannot show the thing the task
actually turns on: with only the first two observations the landing is a guess,
and each further observation should pull it onto the truth. Watching that
happen is how you tell a forecast that converges from one that was lucky.

There is an existing animation of this for the ball token
(tools/ball_token_viz_plot.render_trajectory_animation), which reads
ball_prefix_states. The pixel-path models have no ball token, so on every
current checkpoint it has nothing to read. This is the same idea driven by the
quantity the pixel path does produce: the rendered ball centre.

It runs on the CSV that tools/export_ball_track.py writes, not on a model, so
the animation can be rebuilt on a laptop and re-timed without spending a GPU
again:

    python tools/export_ball_track.py --config ... --checkpoint ... \\
        --scene 0 --num-frames 46 --output ball_track_019999     # needs a GPU
    python tools/animate_ball_forecast.py ball_track_019999.csv  # does not

## What each step shows

At observation k the ball's rendered centres at frames 0..k are fitted to a
ballistic arc under known gravity -- the same readout the evaluator scores as
`pixel fit` -- and that arc is continued to the catch frame. Nothing after
frame k is used, so the sequence is causal: it is what the model could have
said at the time.

Views are pooled by averaging before the fit, which is not the evaluator's
worst-view rule, so the error printed here is a little kinder than the reported
one. It is the right choice for a picture about convergence and the wrong one
for a number, and the caption says so.

All terminal output is ASCII.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

OBSERVATION_FRAMES = (0, 3, 6, 9, 12, 15)
TRUTH_COLOUR = "#171A18"
PRED_COLOUR = "#B33D33"
RING_COLOUR = "#2E6389"
BALL_COLOUR = "#C4D62C"


def read_track(csv_path: Path) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[float]]]:
    """Per-frame view readings and truth, from the exporter's CSV."""
    readings: Dict[int, List[List[float]]] = defaultdict(list)
    truth: Dict[int, List[float]] = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            truth[frame] = [float(row[f"gt_{a}"]) for a in "xyz"]
            if row["pred_x"]:
                point = [float(row[f"pred_{a}"]) for a in "xyz"]
                if all(math.isfinite(v) for v in point):
                    readings[frame].append(point)
    if not truth:
        raise SystemExit(f"[FAIL] no rows in {csv_path}")
    return readings, truth


def fit_ballistic(points: Sequence[Sequence[float]], times: Sequence[float],
                  gravity: Sequence[float]):
    """Least squares for (p0, v0) with the known gravity term removed first.

    Subtracting 0.5*g*t^2 leaves a straight line in t, so the fit never has to
    estimate an acceleration it already knows; letting it would amplify noise
    by dt^2 for nothing.
    """
    import numpy as np
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or len(set(times.tolist())) < 2:
        return None
    corrected = (np.asarray(points, dtype=float)
                 - 0.5 * (times ** 2)[:, None] * np.asarray(gravity, dtype=float))
    design = np.stack([np.ones_like(times), times], axis=-1)
    solution, *_ = np.linalg.lstsq(design, corrected, rcond=None)
    return solution[0], solution[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("track", type=Path, help="the .csv written by export_ball_track.py")
    parser.add_argument("--output", type=Path, default=None,
                        help="default: <track>_forecast.mp4")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="playback fps; one step per observation, so slow is right")
    parser.add_argument("--hold", type=int, default=4,
                        help="repeat the final step this many times, so the last "
                             "forecast is readable before the clip loops")
    parser.add_argument("--dpi", type=int, default=150)
    cli = parser.parse_args()

    if not cli.track.is_file():
        raise SystemExit(f"[FAIL] no such file: {cli.track}")
    meta_path = cli.track.with_suffix(".json")
    if not meta_path.is_file():
        raise SystemExit(
            f"[FAIL] {meta_path.name} is missing. export_ball_track.py writes it "
            f"beside the CSV; without it the frame rate, catch frame and ring "
            f"size would have to be guessed."
        )
    meta = json.loads(meta_path.read_text())
    output = cli.output or cli.track.with_name(f"{cli.track.stem}_forecast.mp4")
    if output.suffix.lower() not in (".mp4", ".gif"):
        parser.error("output must end in .mp4 or .gif")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit(f"[FAIL] needs matplotlib, numpy and imageio: {exc}")

    readings, truth = read_track(cli.track)
    step = float(meta["step_seconds"])
    catch = int(meta["catch_frame"])
    gravity = list(meta["gravity_rig"])
    ring_radius = float(meta["ring_diameter_m"]) / 2
    tolerance = ring_radius - float(meta["ball_radius_m"]) / 2

    truth_frames = sorted(truth)
    truth_points = np.array([truth[f] for f in truth_frames])
    if catch not in truth:
        raise SystemExit(
            f"[FAIL] the track stops at frame {truth_frames[-1]} and the catch is "
            f"at {catch}; re-run export_ball_track.py with --num-frames {catch + 1}"
        )
    catch_truth = np.array(truth[catch])

    # One step per observation that can support a fit.
    steps = []
    for last in OBSERVATION_FRAMES:
        used = [f for f in OBSERVATION_FRAMES if f <= last and readings.get(f)]
        if len(used) < 2:
            continue
        pooled = [np.mean(readings[f], axis=0) for f in used]
        times = [(f - last) * step for f in used]
        fit = fit_ballistic(pooled, times, gravity)
        if fit is None:
            continue
        p0, v0 = fit
        horizon = np.arange(0.0, (catch - last) * step + 1e-9, step)
        arc = (p0[None, :] + v0[None, :] * horizon[:, None]
               + 0.5 * np.asarray(gravity)[None, :] * (horizon ** 2)[:, None])
        steps.append({"last": last, "used": used, "pooled": np.array(pooled),
                      "arc": arc, "landing": arc[-1],
                      "error": float(np.linalg.norm(arc[-1] - catch_truth))})
    if not steps:
        raise SystemExit("[FAIL] fewer than two observation frames carry a ball reading")

    # One extent for every step, or the arc would appear to move when only the
    # axes did.
    everything = np.concatenate([truth_points] + [s["arc"] for s in steps])
    centre = (everything.min(axis=0) + everything.max(axis=0)) / 2
    reach = max(float((everything.max(axis=0) - everything.min(axis=0)).max()) * 0.55, 0.2)

    seed = np.array([1.0, 0.0, 0.0])
    arrival = catch_truth - np.array(truth[catch - 1])
    arrival = arrival / max(float(np.linalg.norm(arrival)), 1e-9)
    if abs(arrival[0]) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    u = np.cross(arrival, seed); u /= np.linalg.norm(u)
    w = np.cross(arrival, u)
    angle = np.linspace(0, 2 * np.pi, 90)
    ring = (catch_truth[None, :] + ring_radius * (np.cos(angle)[:, None] * u[None, :]
                                                  + np.sin(angle)[:, None] * w[None, :]))

    writer_options = ({"duration": 1000.0 / cli.fps, "loop": 0}
                      if output.suffix.lower() == ".gif"
                      else {"fps": cli.fps, "codec": "libx264", "macro_block_size": 2})
    order = list(range(len(steps))) + [len(steps) - 1] * max(0, cli.hold)

    with plt.rc_context({"font.size": 10}):
        with imageio.get_writer(output, **writer_options) as writer:
            for position in order:
                state = steps[position]
                figure = plt.figure(figsize=(9, 6.4))
                axes = figure.add_subplot(111, projection="3d")

                axes.plot(*truth_points.T, color=TRUTH_COLOUR, linewidth=1.6,
                          label="truth", zorder=2)
                axes.plot(*ring.T, color=RING_COLOUR, linewidth=2.4,
                          label=f"catch ring {meta['ring_diameter_m'] * 100:.0f} cm", zorder=3)
                axes.scatter(*catch_truth, marker="*", s=150, color=BALL_COLOUR,
                             edgecolors=TRUTH_COLOUR, linewidths=0.8, zorder=6,
                             label="true landing")
                axes.scatter(*state["pooled"].T, s=26, color=RING_COLOUR, zorder=4,
                             label=f"observations through f{state['last']}")
                axes.plot(*state["arc"].T, color=PRED_COLOUR, linewidth=2.4, zorder=5,
                          label="forecast")
                axes.scatter(*state["landing"], marker="X", s=110, color=PRED_COLOUR,
                             edgecolors=TRUTH_COLOUR, linewidths=0.8, zorder=7,
                             label="predicted landing")
                axes.plot(*np.stack([catch_truth, state["landing"]]).T,
                          color=PRED_COLOUR, linestyle=":", linewidth=1.4, zorder=6)

                axes.set_xlim(centre[0] - reach, centre[0] + reach)
                axes.set_ylim(centre[1] - reach, centre[1] + reach)
                axes.set_zlim(centre[2] - reach, centre[2] + reach)
                axes.set_box_aspect((1, 1, 1))
                axes.view_init(elev=16, azim=-62)
                axes.set_xlabel("x (m)"); axes.set_ylabel("y (m)"); axes.set_zlabel("z (m)")
                axes.tick_params(labelsize=7)
                axes.legend(loc="upper left", fontsize=8, frameon=False)

                verdict = "inside the ring" if state["error"] < tolerance else "outside"
                figure.suptitle(
                    f"Forecast from {len(state['used'])} observations "
                    f"(through frame {state['last']})", x=0.06, ha="left",
                    fontsize=15, fontweight="semibold")
                figure.text(0.06, 0.905,
                            f"scene {meta['scene']}   {meta['checkpoint']}   "
                            f"landing error {state['error'] * 100:.1f} cm -- {verdict} "
                            f"(clears at {tolerance * 100:.2f} cm)",
                            fontsize=10, color=TRUTH_COLOUR)
                figure.text(0.06, 0.03,
                            "Views pooled before the fit; the evaluator takes the worst "
                            "view, so its error is larger than this.",
                            fontsize=8, color="#6E756F")
                figure.tight_layout(rect=(0, 0.05, 1, 0.88))
                figure.canvas.draw()
                writer.append_data(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
                plt.close(figure)

    print(f"scene       : {meta['scene']}   {meta['checkpoint']}")
    print(f"steps       : {len(steps)} observations, {len(order)} frames at {cli.fps:g} fps")
    print(f"{'observations':<14}{'landing error':>15}{'':>3}{'verdict':<16}")
    for state in steps:
        verdict = "inside" if state["error"] < tolerance else "outside"
        print(f"through f{state['last']:<6}{state['error'] * 100:>13.1f} cm   {verdict:<16}")
    print(f"wrote       : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
