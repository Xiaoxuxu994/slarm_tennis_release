"""Checkpoints must survive a change in camera count.

aggregator.affine_token is the only parameter shaped by camera count, and the
adaptation is by NAME, not by position. Calling load_state_dict on the raw
checkpoint instead of going through prepare_checkpoint_state_for_model gives

    size mismatch for aggregator.affine_token: copying a param with shape
    [1, 3, 768] from checkpoint, the shape in current model is [1, 2, 768]

which reads as "incompatible checkpoint" when the checkpoint is fine. That was
a real false alarm from tools/check_model_init.py; these tests pin the shared
path both it and misc.load_model now use.
"""
from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from src.utils.misc import prepare_checkpoint_state_for_model

TRIVIEW = ["front_left", "front_right", "lower_front"]
STEREO = ["front_left", "front_right"]
DATASET = "ball_catch_triview_v3_0829"
EMBED = 8


class _Args:
    def __init__(self, num_cams: int):
        self.dataset = [DATASET]
        self.num_max_cameras = num_cams


def _affine(num_cams: int) -> torch.Tensor:
    # Camera i's row is all i, so the surviving rows are obvious after reordering
    return torch.stack([torch.full((EMBED,), float(i)) for i in range(num_cams)]).unsqueeze(0)


def _state(num_cams: int, grid: int) -> OrderedDict:
    return OrderedDict(
        [
            ("aggregator.affine_token", _affine(num_cams)),
            ("aggregator.plucker_embedder.x", torch.full((grid, grid), float(grid))),
            ("aggregator.blocks.0.weight", torch.ones(EMBED, EMBED)),
        ]
    )


def _rows(tensor: torch.Tensor) -> list[float]:
    return [float(row[0]) for row in tensor[0]]


def test_triview_checkpoint_adapts_to_a_two_view_model():
    checkpoint = {"model": _state(3, grid=4), "args": _Args(3)}
    model_state = _state(2, grid=4)

    state, report = prepare_checkpoint_state_for_model(
        checkpoint, model_state, _Args(2)
    )

    affine = state["aggregator.affine_token"]
    assert affine.shape == model_state["aggregator.affine_token"].shape
    # front_left / front_right survive and lower_front is dropped, by name not position
    assert _rows(affine) == [0.0, 1.0]
    assert report["camera_action"] == "remap"
    assert report["indices"] == [0, 1]
    assert report["source_camera_names"] == TRIVIEW
    assert report["target_camera_names"] == STEREO


class _TwoViewStub(torch.nn.Module):
    """A minimal module with the real key names, to actually run load_state_dict."""

    def __init__(self):
        super().__init__()
        self.aggregator = torch.nn.Module()
        self.aggregator.affine_token = torch.nn.Parameter(_affine(2))
        self.aggregator.register_buffer("plucker_embedder_x", torch.zeros(4, 4))


def test_raw_load_fails_but_the_prepared_state_loads():
    module = _TwoViewStub()
    checkpoint = {"model": _state(3, grid=4), "args": _Args(3)}
    raw = OrderedDict(
        [("aggregator.affine_token", checkpoint["model"]["aggregator.affine_token"])]
    )

    # What check_model_init.py used to do; it reported a usable checkpoint as broken
    with pytest.raises(RuntimeError, match="size mismatch"):
        module.load_state_dict(raw, strict=False)

    state, _ = prepare_checkpoint_state_for_model(
        checkpoint, module.state_dict(), _Args(2)
    )
    result = module.load_state_dict(
        OrderedDict(
            [("aggregator.affine_token", state["aggregator.affine_token"])]
        ),
        strict=False,
    )
    assert not result.unexpected_keys
    assert _rows(module.aggregator.affine_token.detach()) == [0.0, 1.0]


def test_stereo_checkpoint_still_expands_to_three_views():
    """The existing path still works: lower_front starts as the mean of the two."""
    checkpoint = {"model": _state(2, grid=4), "args": _Args(2)}
    model_state = _state(3, grid=4)

    state, report = prepare_checkpoint_state_for_model(
        checkpoint, model_state, _Args(3)
    )

    affine = state["aggregator.affine_token"]
    assert affine.shape == model_state["aggregator.affine_token"].shape
    assert _rows(affine) == [0.0, 1.0, 0.5]
    assert report["camera_action"] == "expand"


def test_same_camera_count_leaves_the_token_untouched():
    checkpoint = {"model": _state(3, grid=4), "args": _Args(3)}
    model_state = _state(3, grid=4)

    state, report = prepare_checkpoint_state_for_model(
        checkpoint, model_state, _Args(3)
    )

    assert _rows(state["aggregator.affine_token"]) == [0.0, 1.0, 2.0]
    assert report["camera_action"] == "remap"
    assert report["indices"] == [0, 1, 2]


def test_resolution_dependent_buffers_come_from_the_target_model():
    """A 320x240 checkpoint into a 480x640 run: the plucker grid must be the model's."""
    checkpoint = {"model": _state(3, grid=4), "args": _Args(3)}
    model_state = _state(2, grid=8)

    state, _ = prepare_checkpoint_state_for_model(
        checkpoint, model_state, _Args(2)
    )

    buffer = state["aggregator.plucker_embedder.x"]
    assert buffer.shape == model_state["aggregator.plucker_embedder.x"].shape
    assert torch.equal(buffer, model_state["aggregator.plucker_embedder.x"])
