"""Decomposing the landing error along the catch-ring axis.

The ring faces the incoming ball, so its axis is the ball's velocity direction at
the catch. Error along that axis still passes through the ring, just early or
late; only the perpendicular part moves the ball off the opening. catch_position
uses the 3D norm and so counts axial error as a miss, making any success rate from
it a lower bound, while catch_position_inplane is the geometrically honest one.

Checked here: the decomposition itself, that a large purely axial error is a miss
by the 3D criterion and a hit in plane, and that the new metric is registered in
all three places -- name, report row, cross-checkpoint table -- since missing any
one computes it and then throws it away without an error.
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
