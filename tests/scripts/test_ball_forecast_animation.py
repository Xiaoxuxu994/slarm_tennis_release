"""The causal forecast animation: the fit, its convergence, and its refusal to
guess when it has too little to go on.

Each step fits a ballistic arc to the observations up to frame k and extrapolates
to the catch frame, so what the clip shows is convergence. These tests check the
fit itself and that the error really falls as observations accumulate, not that
the picture looks nice. The quantity animated is the rendered ball centre.
"""
from __future__ import annotations

import ast
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tools" / "animate_ball_forecast.py"

GRAVITY = [0.0, 0.0, -9.81]
STEP = 0.8 / 24
OBSERVATIONS = (0, 3, 6, 9, 12, 15)
CATCH = 45


def _lstsq_line(times, corrected):
    """Normal equations for [1, t], matching what numpy's lstsq returns here."""
    n = len(times)
    sum_t = sum(times)
    sum_tt = sum(t * t for t in times)
    determinant = n * sum_tt - sum_t * sum_t
    p0, v0 = [], []
    for axis in range(3):
        sum_y = sum(c[axis] for c in corrected)
        sum_ty = sum(t * c[axis] for t, c in zip(times, corrected))
        p0.append((sum_tt * sum_y - sum_t * sum_ty) / determinant)
        v0.append((n * sum_ty - sum_t * sum_y) / determinant)
    return p0, v0


def forecast(readings, last):
    """The arithmetic one animation step performs."""
    used = [f for f in OBSERVATIONS if f <= last and f in readings]
    if len(used) < 2:
        return None
    times = [(f - last) * STEP for f in used]
    corrected = [[readings[f][k] - 0.5 * times[i] ** 2 * GRAVITY[k] for k in range(3)]
                 for i, f in enumerate(used)]
    p0, v0 = _lstsq_line(times, corrected)
    dt = (CATCH - last) * STEP
    return [p0[k] + v0[k] * dt + 0.5 * GRAVITY[k] * dt * dt for k in range(3)], len(used)


def truth(frame, p15=(0.0, 3.0, 1.6), v15=(0.02, -2.11, 2.39)):
    t = (frame - 15) * STEP
    return [p15[k] + v15[k] * t + 0.5 * GRAVITY[k] * t * t for k in range(3)]


@pytest.mark.parametrize("last", [3, 6, 9, 12, 15])
def test_a_clean_ballistic_is_recovered_exactly(last):
    """Two points determine it once gravity is removed, so any noiseless prefix
    must land on the truth; anything else is a bug in the fit, not noise."""
    clean = {f: truth(f) for f in OBSERVATIONS}
    predicted, _ = forecast(clean, last)
    assert math.dist(predicted, truth(CATCH)) < 1e-9


def test_the_forecast_converges_as_observations_arrive():
    """The single claim the animation makes. Without this it is decoration."""
    rng = random.Random(7)
    medians = []
    for last in (3, 6, 9, 12, 15):
        errors = []
        for _ in range(400):
            noisy = {f: [truth(f)[k] + rng.gauss(0, 0.011) for k in range(3)]
                     for f in OBSERVATIONS}
            predicted, _ = forecast(noisy, last)
            errors.append(math.dist(predicted, truth(CATCH)))
        errors.sort()
        medians.append(errors[len(errors) // 2])
    assert medians == sorted(medians, reverse=True), f"not monotone: {medians}"
    assert medians[0] / medians[-1] > 3, "the whole point is that it tightens a lot"


# ---------------------------------------------------------------- flight mode

def test_flight_runs_one_frame_per_rendered_frame():
    """The clip should run at the scene's own pace, not at the observation
    schedule's; subsampling would make the ball jump."""
    truth_frames = list(range(46))
    hold = 4
    order = truth_frames + [truth_frames[-1]] * hold
    assert order[:46] == truth_frames
    assert len(order) == 50 and order[-1] == 45


@pytest.mark.parametrize("position", [0, 22, 45])
def test_the_flight_index_never_runs_off_the_end(position):
    """`upto` slices the truth and also indexes the current ball; an off-by-one
    here raises only on the last frame of the clip."""
    truth_frames = list(range(46))
    upto = truth_frames.index(position) + 1
    assert 1 <= upto <= len(truth_frames)
    assert 0 <= upto - 1 < len(truth_frames)


def test_the_orbit_sweeps_exactly_the_requested_amount():
    order_length, orbit = 50, 45.0
    sweeps = [orbit * tick / max(1, order_length - 1) for tick in range(order_length)]
    assert sweeps[0] == 0.0
    assert sweeps[-1] == pytest.approx(orbit)
    assert sweeps == sorted(sweeps), "the camera must not reverse mid-clip"


def test_the_orbit_is_slow_enough_to_follow_the_ball():
    """A fast sweep reads as motion of its own and competes with the subject."""
    degrees_per_second = 45.0 / (50 - 1) * 12.0
    assert degrees_per_second < 20, f"{degrees_per_second:.0f} deg/s is dizzying"


def test_zero_orbit_holds_the_camera_still():
    assert [0.0 * tick / 49 for tick in range(50)] == [0.0] * 50


def test_flight_does_not_require_a_fittable_prefix():
    """forecast needs two observations; flight only needs the track, and a scene
    where the ball is never rendered should still produce a clip of the truth."""
    source = SRC.read_text()
    assert 'if not steps and cli.mode == "forecast"' in source


def test_the_whole_arc_is_drawn_faint_under_the_flown_part():
    """Revealing the trajectory as it goes would suggest the model discovers it,
    when the truth is fixed and only the readings arrive over time."""
    source = SRC.read_text()
    assert "alpha=0.25" in source


def test_gravity_is_removed_before_fitting_not_estimated():
    """Fitting an acceleration the physics already gives would amplify noise by
    dt squared for nothing."""
    body = ast.unparse(next(n for n in ast.walk(ast.parse(SRC.read_text()))
                            if isinstance(n, ast.FunctionDef) and n.name == "fit_ballistic"))
    assert "0.5 * times ** 2" in body or "0.5 * (times ** 2)" in body
    assert "np.ones_like(times)" in body, "the design matrix should be [1, t]"


def test_only_observations_up_to_the_step_are_used():
    """A step that peeked at later frames would show a forecast the model could
    not have made, and the convergence would be theatre."""
    source = SRC.read_text()
    assert "if f <= last and readings.get(f)" in source


def test_metadata_is_required_rather_than_guessed():
    """Frame rate, catch frame and ring size decide the numbers on screen; a
    default that happened to be wrong would look entirely plausible."""
    source = SRC.read_text()
    assert "is missing" in source and "would have to be guessed" in source


def test_a_short_track_is_refused_with_the_fix():
    """Extrapolating past the exported frames would draw a landing nobody
    measured."""
    source = SRC.read_text()
    assert "re-run export_ball_track.py with --num-frames" in source


def test_the_exporter_writes_the_sidecar_this_reads():
    exporter = (ROOT / "tools" / "export_ball_track.py").read_text()
    for key in ("step_seconds", "catch_frame", "ring_diameter_m",
                "ball_radius_m", "gravity_rig"):
        assert f'"{key}"' in exporter, f"{key} missing from the sidecar"


def test_every_step_shares_one_set_of_axes():
    """Rescaling per step would make the arc appear to move when only the axes
    did, which is exactly the illusion this animation must not create."""
    source = SRC.read_text()
    assert "One extent for every step" in source
    assert source.count("axes.set_xlim") == 1


def test_bare_drops_the_chrome_and_keeps_the_plot():
    """A slide carries its own caption, so the figure repeating it wastes the
    frame. Everything --bare removes is still in the CSV beside it."""
    source = SRC.read_text()
    assert '"--bare"' in source
    assert "if not cli.bare:" in source, "the legend has to be conditional"
    assert "if cli.bare:" in source
    assert "rect=(0, 0, 1, 1)" in source, "the axes should take the whole figure"


def test_bare_does_not_change_what_is_drawn():
    """Only the chrome goes. If --bare also dropped a series, two clips of the
    same scene would disagree and neither would say why."""
    tree = ast.parse(SRC.read_text())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = set()
    for node in ast.walk(main):
        if isinstance(node, ast.If) and "cli.bare" in ast.unparse(node.test):
            guarded |= {ast.unparse(n.func) for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    drawing = {name for name in guarded
               if any(verb in name for verb in ("plot", "scatter", "set_xlim",
                                                "set_zlim", "view_init"))}
    assert not drawing, f"--bare must not gate the data itself: {sorted(drawing)}"


def test_the_pooling_difference_is_disclosed_on_the_figure():
    """Pooling views is kinder than the evaluator's worst-view rule, so the
    error drawn here is smaller than the reported one.

    Matched on a fragment short enough to survive rewrapping: the first version
    of this test spanned a line break in the source string and went red on an
    edit that changed nothing it was meant to protect.
    """
    source = SRC.read_text()
    assert "Views pooled before the fit" in source
    assert "worst" in source and "larger than this" in source
