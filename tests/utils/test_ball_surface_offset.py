"""Compensation from the ball's near surface to its centre.

The depth map is a z-buffer, so unprojecting the ball mask gives the near surface,
while ball_trajectory's position_rig is the centre. The gap is constant and points
at the camera -- the direction carrying 95.5% of the error.

    pytest tests/utils/test_ball_surface_offset.py -q
"""
from __future__ import annotations

import pytest
import torch

from src.utils.stream25_metrics import (
    BALL_SURFACE_COEFFICIENT_DISC_MEAN,
    BALL_SURFACE_COEFFICIENT_DISC_MEDIAN,
    BALL_SURFACE_COEFFICIENT_MEASURED,
    apply_ball_surface_offset,
)

R = 0.065 / 2


def test_off_by_default_is_bitwise_identity():
    """offset=0 must return the input untouched, or past numbers drift."""
    pos = torch.randn(4, 5, 3)
    out = apply_ball_surface_offset(pos, torch.randn(4, 5, 3), 0.0)
    assert out is pos


def test_direction_is_normalized_before_use():
    """This is why the function normalises.

    ``dirs`` from embedders.py is not a unit vector -- its camera-frame z is 1, to
    pair with planar z-depth -- so multiplying by it raw scales the offset by
    ||dirs||, which reaches 1.3 in the corners: 30% too far.
    """
    pos = torch.zeros(1, 3)
    d = torch.tensor([[0.6, 0.8, 1.0]])          # ||d|| = sqrt(2) != 1
    assert float(d.norm()) == pytest.approx(2 ** 0.5)
    out = apply_ball_surface_offset(pos, d, 0.05)
    assert float((out - pos).norm()) == pytest.approx(0.05, abs=1e-7)   # not 0.0707


def test_moves_away_from_the_camera():
    """The offset must move the point AWAY from the camera, not towards it."""
    origin = torch.zeros(1, 3)
    d = torch.tensor([[0.0, 0.0, 3.0]])
    surface = origin + d * 1.0                    # 3 m in front of the camera
    out = apply_ball_surface_offset(surface, d, BALL_SURFACE_COEFFICIENT_MEASURED * R)
    assert float(out.norm()) > float(surface.norm())
    assert float(out.norm() - surface.norm()) == pytest.approx(
        BALL_SURFACE_COEFFICIENT_MEASURED * R, abs=1e-7
    )


def test_recovers_a_known_centre():
    """Known centre -> take the near surface -> compensation returns the centre."""
    origin = torch.zeros(1, 3)
    unit = torch.tensor([[0.6, 0.0, 0.8]])        # already unit length
    centre = origin + unit * 4.0
    c = BALL_SURFACE_COEFFICIENT_MEASURED
    surface = centre - unit * (c * R)             # near surface, same coefficient
    out = apply_ball_surface_offset(surface, unit * 7.3, c * R)   # deliberately not a unit vector
    torch.testing.assert_close(out, centre, atol=1e-6, rtol=0)


def test_the_measured_coefficient_sits_between_the_theoretical_ones():
    """The measured 0.646 sits below the theoretical values, as expected.

    Disc mean is 0.667, disc median 0.707, nearest point 1.0. The measurement comes
    out lower because the mask includes edge pixels, where the sphere is thin.
    """
    assert BALL_SURFACE_COEFFICIENT_DISC_MEAN == pytest.approx(2 / 3)
    assert BALL_SURFACE_COEFFICIENT_DISC_MEDIAN == pytest.approx(0.70711, abs=1e-5)
    assert BALL_SURFACE_COEFFICIENT_MEASURED < BALL_SURFACE_COEFFICIENT_DISC_MEAN
    assert BALL_SURFACE_COEFFICIENT_MEASURED * R == pytest.approx(0.0210, abs=2e-4)


def test_batched_shapes_are_preserved():
    """The evaluator passes [V,H,W,3]; the shape must survive."""
    pos = torch.randn(3, 8, 9, 3)
    d = torch.randn(3, 8, 9, 3) + 3.0
    out = apply_ball_surface_offset(pos, d, 0.02)
    assert out.shape == pos.shape
    step = (out - pos).norm(dim=-1)
    torch.testing.assert_close(step, torch.full_like(step, 0.02), atol=1e-6, rtol=0)
