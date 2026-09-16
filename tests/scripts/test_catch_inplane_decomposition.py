"""落点误差按环轴的分解。

存在的理由。接球环放在预测点上等球来，而且正对来球，所以它的轴就是接球时刻的
球速方向。误差沿这个轴偏，球的真实轨迹**照样穿过环心**，只是早到或晚到；只有
垂直于这个轴的分量才会让球从环口偏开。

`catch_position` 用的是三维模，把轴向也当成了打偏，所以由它算出来的成功率是
**下界**。`catch_position_inplane` 才是几何上诚实的那个。

这份测试盯三件事：
  1. 分解本身正确（纯轴向 -> 横向为 0；纯横向 -> 轴向为 0；勾股恒等）
  2. 一个纯轴向的大误差会被 3D 判据误判成"没进"，而平面判据判"进"
     —— 这正是"下界"这个说法的全部内容
  3. 新指标在三处都注册了：指标名、报表行、跨 ckpt 对照表
     （漏任何一处，指标会被算出来然后丢掉，不报错）

    pytest tests/scripts/test_catch_inplane_decomposition.py -q
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Sequence, Tuple

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVAL_SRC = ROOT / "scripts" / "eval_stream25_base.py"

RING_CLEARS_M = 0.27 / 2 - 0.065 / 2      # 0.1025


def decompose(error: Sequence[float], axis: Sequence[float]) -> Tuple[float, float, float]:
    """The arithmetic the evaluator performs, in plain python."""
    norm_axis = math.sqrt(sum(a * a for a in axis))
    unit = [a / norm_axis for a in axis]
    along = sum(e * u for e, u in zip(error, unit))
    lateral = [e - along * u for e, u in zip(error, unit)]
    return abs(along), math.sqrt(sum(v * v for v in lateral)), math.sqrt(sum(e * e for e in error))


def unit_perpendicular_to(axis: Sequence[float]) -> Tuple[float, ...]:
    norm = math.sqrt(sum(a * a for a in axis))
    unit = [a / norm for a in axis]
    seed = [1.0, 0.0, 0.0] if abs(unit[0]) < 0.9 else [0.0, 1.0, 0.0]
    dot = sum(s * u for s, u in zip(seed, unit))
    perpendicular = [s - dot * u for s, u in zip(seed, unit)]
    length = math.sqrt(sum(v * v for v in perpendicular))
    return tuple(v / length for v in perpendicular)


# Terminal velocity of the measured order, carried to the catch by known gravity.
AXIS = (0.0, -2.1, 2.4 - 9.81 * 1.0)


def test_pure_axial_error_has_no_lateral_component():
    unit = [a / math.sqrt(sum(x * x for x in AXIS)) for a in AXIS]
    axial, lateral, norm = decompose([0.10 * u for u in unit], AXIS)
    assert lateral == pytest.approx(0.0, abs=1e-9)
    assert axial == pytest.approx(0.10, abs=1e-9)
    assert norm == pytest.approx(0.10, abs=1e-9)


def test_pure_lateral_error_has_no_axial_component():
    perpendicular = unit_perpendicular_to(AXIS)
    axial, lateral, norm = decompose([0.10 * p for p in perpendicular], AXIS)
    assert axial == pytest.approx(0.0, abs=1e-9)
    assert lateral == pytest.approx(0.10, abs=1e-9)


@pytest.mark.parametrize("mix", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_two_components_are_orthogonal(mix):
    """Pythagoras, so neither part can quietly absorb the other."""
    unit = [a / math.sqrt(sum(x * x for x in AXIS)) for a in AXIS]
    perpendicular = unit_perpendicular_to(AXIS)
    error = [0.10 * (mix * u + (1 - mix) * p) for u, p in zip(unit, perpendicular)]
    axial, lateral, norm = decompose(error, AXIS)
    assert math.hypot(axial, lateral) == pytest.approx(norm, rel=1e-9)


def test_a_large_axial_error_is_a_miss_only_to_the_3d_criterion():
    """This single case is what "lower bound" means."""
    unit = [a / math.sqrt(sum(x * x for x in AXIS)) for a in AXIS]
    axial, lateral, norm = decompose([0.30 * u for u in unit], AXIS)
    assert norm > RING_CLEARS_M, "the 3D distance calls this a miss"
    assert lateral < RING_CLEARS_M, "the ball still passes through the ring centre"
    # It arrives late rather than wide; the delay is the axial error over speed.
    speed = math.sqrt(sum(a * a for a in AXIS))
    assert 0.0 < axial / speed < 0.1, "tens of milliseconds, not a miss"


def test_inplane_can_never_exceed_the_full_distance():
    """So a success rate from the norm can only understate, never overstate."""
    perpendicular = unit_perpendicular_to(AXIS)
    unit = [a / math.sqrt(sum(x * x for x in AXIS)) for a in AXIS]
    for weight in (0.0, 0.3, 0.6, 1.0):
        error = [0.12 * (weight * u + (1 - weight) * p) for u, p in zip(unit, perpendicular)]
        _, lateral, norm = decompose(error, AXIS)
        assert lateral <= norm + 1e-12


def test_catch_axis_uses_the_catch_time_velocity_not_the_terminal_one():
    """Gravity turns the ball over during a one second flight; the ring faces
    where it is going when it arrives, not where it was going at frame 15."""
    source = EVAL_SRC.read_text()
    assert "velocity_catch" in source
    assert "+ gravity * catch_dt" in source


@pytest.mark.parametrize("name", ["catch_position_inplane", "catch_position_axial"])
def test_new_metrics_are_registered_everywhere_they_are_consumed(name):
    """A metric missing from the name tuple is computed and then dropped."""
    assert f'"{name}",' in EVAL_SRC.read_text()
    assert f'("{name}", "median")' in (ROOT / "src/utils/stream25_report.py").read_text()
    assert f'"{name}"' in (ROOT / "tools/compare_evaluations.py").read_text()


def test_landing_positions_helper_keeps_the_error_path_identical():
    """The error function must go through the new helper, not duplicate it."""
    tree = ast.parse(EVAL_SRC.read_text())
    errors_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                     and n.name == "compute_rendered_frame24_position_errors")
    body = ast.unparse(errors_fn)
    assert "rendered_landing_positions_per_view" in body
    assert "median(dim=0)" not in body, "the median readout should live in one place"
