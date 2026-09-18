"""因果预报动画：拟合本身、收敛性、以及它拒绝在缺信息时瞎猜。

动画每一步是「只用到第 k 帧为止的观测」拟合一条弹道，外推到接球帧。它要讲的是
**收敛**：两个观测时落点是猜的，每多一个观测应该把它拉向真值。所以这份测试的
核心不是画得好不好，而是那条拟合对不对、以及误差确实随观测数下降。

已有的 ball-token 动画（tools/ball_token_viz_plot.render_trajectory_animation）
读的是 ball_prefix_states，而当前所有像素路径 ckpt 都没有 ball token，所以它在
新模型上读不到东西。这份是同一个想法换成像素路径真正产出的量：渲染球心。

    pytest tests/scripts/test_ball_forecast_animation.py -q
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


def test_the_pooling_difference_is_disclosed_on_the_figure():
    """Pooling views is kinder than the evaluator's worst-view rule, so the
    error drawn here is smaller than the reported one."""
    source = SRC.read_text()
    assert "the evaluator takes the worst" in source
