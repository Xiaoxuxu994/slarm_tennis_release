"""Velocity scale is loss-space normalization, never a decoder gain."""

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.utils.ball_residual_diagnostics import residual_diagnostics, velocity_unit_diagnostics
from src.utils.stream25_losses import ball_vel_scale_from_timespan


ROOT = Path(__file__).resolve().parents[2]


def test_default_velocity_contract():
    assert ball_vel_scale_from_timespan(.8) == pytest.approx(1 / 3)
    # The decoder build must not receive the loss normalization argument.
    tree = ast.parse((ROOT / "engine_tools.py").read_text())
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        assert all(kw.arg != "stream25_ball_vel_scale" for kw in call.keywords)


@pytest.mark.parametrize("scale", [1 / 3, .2, .1])
def test_production_loss_normalizes_both_sides_without_mutating_physical_state(scale):
    tree = ast.parse((ROOT / "src/utils/stream25_losses.py").read_text())
    statement = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "ball_vel_loss" for t in n.targets))
    base = torch.tensor([[1., 2., 3.]])
    delta = torch.tensor([[.03, -.06, .09]], requires_grad=True)
    final = base.detach() + delta
    env = {"F": F, "vel_pred": final, "vel_gt": base, "_vel_scale": scale}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), "physical_loss", "exec"), env)
    expected = delta.square().sum() / (6 * scale**2)
    torch.testing.assert_close(env["ball_vel_loss"], expected)
    env["ball_vel_loss"].backward()
    torch.testing.assert_close(delta.grad, delta.detach() / (3 * scale**2))
    torch.testing.assert_close(final.detach(), base + delta.detach())


def test_dual_unit_logs_are_detached_and_do_not_change_physics():
    base = torch.tensor([[1., 2., 3.]])
    delta = torch.tensor([[.03, -.06, .09]], requires_grad=True)
    final = base + delta
    zero = torch.zeros_like(base)
    diagnostics = residual_diagnostics(base, delta, final, base, zero, zero, dt24=.3, dt45=1.)
    for scale in (1 / 3, .2):
        values = velocity_unit_diagnostics(diagnostics, scale)
        torch.testing.assert_close(values["delta_x_mps"], delta.detach()[:, 0])
        torch.testing.assert_close(values["final_x_normalized"], final.detach()[:, 0] / scale)
        assert not any(v.requires_grad for v in values.values())
        assert not any("loss" in k for k in values)
    assert "final_frame45_error_normalized" not in values
    torch.testing.assert_close(diagnostics["final_frame45_error"], delta.detach().norm(dim=-1))
