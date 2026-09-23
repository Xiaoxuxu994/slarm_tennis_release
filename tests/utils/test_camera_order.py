"""Named-view list must follow the config, not a frozen tri-view constant.

Regression guard for two-view (num_max_cameras: 2) evaluation: before this,
CAMERA_ORDER was a module constant and scripts/eval_stream25_base.py raised
"requires 3 named views" on any two-view checkpoint.
"""
from __future__ import annotations

import pytest

from src.utils.stream25_metrics import (
    CAMERA_ORDER,
    DEFAULT_CAMERA_ORDER,
    REQUIRED_EVAL_SCOPES,
    checkpoint_report_worst_ratio,
    get_camera_order,
    get_required_eval_scopes,
    reset_camera_order,
    set_camera_order,
)

STEREO = ("front_left", "front_right")
TRIVIEW = ("front_left", "front_right", "lower_front")


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    reset_camera_order()


def test_default_is_the_frozen_triview_order():
    assert get_camera_order() == TRIVIEW
    assert get_required_eval_scopes() == ("aggregate",) + TRIVIEW
    # The frozen constants do not move: published gate reports compare against them
    assert CAMERA_ORDER == TRIVIEW
    assert DEFAULT_CAMERA_ORDER == TRIVIEW
    assert REQUIRED_EVAL_SCOPES == ("aggregate",) + TRIVIEW


def test_two_views_narrow_the_scope_list():
    assert set_camera_order(STEREO) == STEREO
    assert get_camera_order() == STEREO
    assert get_required_eval_scopes() == ("aggregate", "front_left", "front_right")


def test_setting_does_not_mutate_the_frozen_constants():
    set_camera_order(STEREO)
    assert CAMERA_ORDER == TRIVIEW
    assert REQUIRED_EVAL_SCOPES == ("aggregate",) + TRIVIEW


def test_reset_restores_triview():
    set_camera_order(STEREO)
    assert reset_camera_order() == TRIVIEW
    assert get_required_eval_scopes() == ("aggregate",) + TRIVIEW


@pytest.mark.parametrize("bad", [(), [], ["a", "a"]])
def test_rejects_empty_or_duplicate_names(bad):
    with pytest.raises(ValueError):
        set_camera_order(bad)


def _passing_report(scopes):
    return {"scope_reports": {s: {"all_gates_pass": True, "worst_ratio": 0.9}
                              for s in scopes}}


def test_worst_ratio_follows_the_active_scope_list():
    stereo_scopes = ("aggregate",) + STEREO
    report = _passing_report(stereo_scopes)

    # Under the tri-view default this two-view report lacks lower_front
    assert checkpoint_report_worst_ratio(report) is None

    set_camera_order(STEREO)
    assert checkpoint_report_worst_ratio(report) == pytest.approx(0.9)


def test_triview_report_is_rejected_once_two_views_are_active():
    report = _passing_report(("aggregate",) + TRIVIEW)
    assert checkpoint_report_worst_ratio(report) == pytest.approx(0.9)

    set_camera_order(STEREO)
    # An extra lower_front scope means the report came from a different setup
    assert checkpoint_report_worst_ratio(report) is None
