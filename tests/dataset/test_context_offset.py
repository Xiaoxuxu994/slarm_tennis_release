"""The sliding observation window (context offset).

This is a data-side switch only. get_frame computes dt against
context_frames[0], so when the whole window moves later the source moves with it
and the model receives byte-identical times; only the images change, with the ball
nearer. No retraining is involved.

Three things here would quietly ruin an experiment:

1. Offset 0 must equal the frozen contract byte for byte, or every past number
   stops meaning anything.
2. The invariant targets[15] == context[-1]. Every terminal readout in the
   evaluator hardcodes target index 15, so breaking it reads the wrong frame and
   returns a plausible wrong number instead of an error.
3. The training path must refuse a non-zero offset: its target scheduler still
   speaks frozen frame numbers, and would pair a shifted context with unshifted
   targets.

    pytest tests/dataset/test_context_offset.py -q
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = Path(__file__).resolve().parents[2]

# stream25.py imports cv2 and torch at the top, so the arithmetic is pulled out of
# the AST and executed on its own; that runs anywhere.
_BASE_CONTEXT = (0, 3, 6, 9, 12, 15)
_BASE_TARGETS = tuple(range(25))


def _standalone_shifted_contract():
    source = (ROOT / "src" / "dataset" / "stream25.py").read_text()
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "shifted_contract"
    )
    namespace = {
        "Tuple": tuple,
        "STREAM25_CONTEXT_FRAMES": _BASE_CONTEXT,
        "STREAM25_ALL_TARGET_FRAMES": _BASE_TARGETS,
    }
    exec(compile(ast.Module([fn], []), "<shifted_contract>", "exec"), namespace)
    return namespace["shifted_contract"]


shifted_contract = _standalone_shifted_contract()


def test_zero_offset_is_the_frozen_contract_byte_for_byte():
    context, targets = shifted_contract(0)
    assert context == _BASE_CONTEXT
    assert targets == _BASE_TARGETS


@pytest.mark.parametrize("offset", [0, 1, 3, 6, 9])
def test_target_index_fifteen_is_always_the_terminal_observation(offset):
    """The evaluator hardcodes target index 15; breaking this reads a wrong frame."""
    context, targets = shifted_contract(offset)
    assert targets[15] == context[-1] == 15 + offset


@pytest.mark.parametrize("offset", [0, 1, 3, 6, 9])
def test_window_keeps_its_shape_and_stays_inside_stored_frames(offset):
    context, targets = shifted_contract(offset)
    assert len(context) == len(_BASE_CONTEXT)
    assert [b - a for a, b in zip(context, context[1:])] == [3] * 5
    assert targets[0] == offset and targets[-1] == _BASE_TARGETS[-1]
    assert len(targets) == len(_BASE_TARGETS) - offset
    assert list(targets) == sorted(set(targets))


@pytest.mark.parametrize("bad", [-1, 10, 25, 100])
def test_offset_past_the_stored_frames_is_refused(bad):
    with pytest.raises(ValueError):
        shifted_contract(bad)


@pytest.mark.parametrize("bad", [1.0, "9", None, True])
def test_non_integer_offset_is_refused(bad):
    """True is an int in Python; an offset of True would silently mean +1."""
    with pytest.raises(TypeError):
        shifted_contract(bad)


def test_times_the_trunk_sees_are_identical_at_every_offset():
    """The whole no-retraining claim in one assertion.

    get_frame subtracts time_in_seconds[context_frames[0]], so reproduce that
    arithmetic on a 30 fps clock and check the six values never move.
    """
    fps = 30.0
    baseline = None
    for offset in (0, 1, 3, 6, 9):
        context, _ = shifted_contract(offset)
        source = context[0]
        times = tuple(round((f - source) / fps, 9) for f in context)
        if baseline is None:
            baseline = times
        assert times == baseline
    assert baseline == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def test_training_path_refuses_a_slid_window():
    """The training target scheduler still speaks frozen frame numbers."""
    source = (ROOT / "src" / "dataset" / "datasets.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Stream25Dataset"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert "context_offset" in {arg.arg for arg in init.args.kwonlyargs + init.args.args}
    raises = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Raise)
        and "context_offset" in ast.unparse(node)
    ]
    assert raises, "a non-zero offset on the training path must raise"


def test_getitem_anchors_source_frame_to_the_window_not_the_clip():
    """source_frame_idx must follow the window, or the times would shift."""
    source = (ROOT / "src" / "dataset" / "datasets.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Stream25Dataset"
    )
    getitem = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"
    )
    body = ast.unparse(getitem)
    assert "source_frame_idx = context_frames[0]" in body
    assert "STREAM25_CONTEXT_FRAMES[0]" not in body


# ---------------------------------------------------------------------------
# Evaluator wiring, checked statically because importing it pulls in gsplat.
#
# The offset crosses five calls. A layer that drops the parameter indexes the GT
# at offset 0 silently, comparing the terminal state against the wrong instant and
# returning a number that looks fine.
# ---------------------------------------------------------------------------

_EVAL_SRC = ROOT / "scripts" / "eval_stream25_base.py"

_MUST_ACCEPT_OFFSET = (
    "compute_stream25_scene_metrics",
    "evaluate_scene",
    "run_evaluation",
    "compute_rendered_history_fit_metrics",
    "_finalize_and_write",
)


def _eval_tree():
    return ast.parse(_EVAL_SRC.read_text())


@pytest.mark.parametrize("name", list(_MUST_ACCEPT_OFFSET))
def test_every_link_in_the_chain_takes_the_offset(name):
    fn = next(
        node for node in ast.walk(_eval_tree())
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert "context_offset" in params, f"{name} would silently index GT at offset 0"


def test_eval_source_compiles_not_just_parses():
    """compile() rejects duplicate argument names; ast.parse does not."""
    compile(_EVAL_SRC.read_text(), str(_EVAL_SRC), "exec")


def test_gt_lookups_do_not_add_the_offset():
    """GT tensors are TARGET-indexed, not absolute-frame indexed.

    ball_position_rig / ball_velocity_rig go through
    _collate_stream25_frames' preserve_time_axes, so their time axis is the
    target list: length 25 - offset, entry i holding the truth for absolute
    frame i + offset. At offset 0 the index equals the frame number, which is
    why treating them as absolute-frame tensors survived every earlier run and
    then blew up on the first slid window:

        IndexError: index 24 is out of bounds for dimension 1 with size 22

    shifted_contract pins targets[15] to the terminal observation at every
    offset, so the correct index is the constant 15; adding the offset is the
    bug, not the fix.

     This check walks the AST rather than scanning lines. A line-based regex
      was written first and missed two of the four real spellings: one where
      the tensor name sat on the previous line of a wrapped call, and one where
      the offset was folded into a variable that was then used as the index.
      Guards that enumerate spellings keep losing to the spellings nobody
      enumerated.
    """
    tree = _eval_tree()
    gt_names = ("ball_position_rig", "ball_velocity_rig", "gt_pos15", "gt_v15",
                "gt_v", "gt_positions", "truth")

    # Taint per function scope, not per module: two functions both have a local
    # called terminal, one a target index and one an absolute frame.
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    offenders = []
    for function in functions:
        tainted = {
            target.id
            for node in ast.walk(function) if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and "context_offset" in ast.unparse(node.value)
        }
        _scan(function, gt_names, tainted, offenders)
    assert not offenders, f"GT lookup indexed with the offset: {offenders}"


def _scan(function, gt_names, tainted, offenders):
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        base = ast.unparse(node.value)
        if not any(name in base for name in gt_names):
            continue
        # Take the identifiers that actually appear; splitting the text leaves
        # "terminal)" with a bracket attached and never matches.
        used = {n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name)}
        index = ast.unparse(node.slice)
        if "context_offset" in index or used & tainted:
            offenders.append(f"{base}[{index}]")


def test_terminal_target_index_is_a_constant_not_an_offset_expression():
    source = _EVAL_SRC.read_text()
    assert "TERMINAL_TARGET_INDEX = STREAM25_CONTEXT_FRAMES[-1]" in source


def test_absolute_frame_is_used_only_for_time_arithmetic():
    """terminal_frame = 15 + offset is right for seconds, wrong for indexing."""
    import re
    source = _EVAL_SRC.read_text()
    for match in re.finditer(r"\[0, terminal_frame\]", source):
        raise AssertionError("terminal_frame is an absolute frame, not an index")
    assert "terminal_frame = STREAM25_CONTEXT_FRAMES[-1] + int(context_offset)" in source


@pytest.mark.parametrize("offset,length", [(0, 25), (3, 22), (6, 19), (9, 16)])
def test_gt_tensor_length_is_the_target_count(offset, length):
    """The length that the IndexError reported: 22 at offset 3, not 25."""
    _, targets = shifted_contract(offset)
    assert len(targets) == length
    assert targets[15] == 15 + offset, "the terminal must stay at index 15"
    # Index 24 only exists at offset 0; that is exactly what blew up.
    assert (24 < len(targets)) == (offset == 0)


def test_catch_metric_is_registered_everywhere_it_is_aggregated():
    """A metric absent from the name tuple is computed and then dropped."""
    assert '"catch_position",' in _EVAL_SRC.read_text()
    report = (ROOT / "src" / "utils" / "stream25_report.py").read_text()
    assert '("catch_position", "median")' in report
    compare = (ROOT / "tools" / "compare_evaluations.py").read_text()
    assert '"catch_position"' in compare


# ---------------------------------------------------------------------------
# StreamSession's observation validator.
#
# The first offset-9 eval failed here:
#     ValueError: Expected context frame 0, got [3]
# the validator had (0,3,6,9,12,15) hardcoded in __init__ and did not recognise a
# slid window. The fix is not to loosen it: sliding the whole window is allowed,
# changing its SHAPE is not, so the first observation fixes the scene's offset and
# every later step is still checked against the contract's stride. The tests below
# hold that a skipped frame, a reordering and a repeat are all still caught.
#
# The method body is executed from the AST; importing stream_session needs gsplat.
# ---------------------------------------------------------------------------


def _validate_observation_fn():
    torch = pytest.importorskip("torch")
    source = (ROOT / "src" / "models" / "stream_session.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "StreamSession"
    )
    fn = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_observation"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([fn], []), "<validate>", "exec"), namespace)
    return namespace["_validate_observation"], torch


class _Model:
    """A plain pixel-path model: no flag makes the contract guard apply."""

    terminal_context_extrapolation = True
    num_cams = 3


class _Session:
    """Only the attributes _validate_observation actually reads."""

    def __init__(self):
        self.model = _Model()
        self.mode = "window"
        self.window_size = 6
        self.expected_context_frames = (0, 3, 6, 9, 12, 15)
        self.num_streamed_observations = 0
        self.context_frame_offset = None


def _stream(validate, torch, frames):
    """Feed frames one at a time exactly as forward_stream does."""
    session = _Session()
    image = torch.zeros(1, 1, 3, 3, 4, 4)
    for frame in frames:
        validate(session, {
            "context_image": image,
            "context_frame_idx": torch.tensor([[frame]] * 3),
        })
        session.num_streamed_observations += 1
    return session


@pytest.mark.parametrize("offset", [0, 3, 6, 9])
def test_a_slid_window_streams_without_tripping_the_validator(offset):
    """This is the assertion the offset-9 eval crash would have failed."""
    validate, torch = _validate_observation_fn()
    session = _stream(validate, torch, [f + offset for f in (0, 3, 6, 9, 12, 15)])
    assert session.context_frame_offset == offset


def test_a_skipped_frame_is_still_caught():
    validate, torch = _validate_observation_fn()
    with pytest.raises(ValueError):
        _stream(validate, torch, [3, 6, 12, 15, 18, 21])


def test_a_wrong_stride_is_still_caught():
    validate, torch = _validate_observation_fn()
    with pytest.raises(ValueError):
        _stream(validate, torch, [3, 5, 7, 9, 11, 13])


def test_a_repeated_frame_is_still_caught():
    validate, torch = _validate_observation_fn()
    with pytest.raises(ValueError):
        _stream(validate, torch, [3, 3, 6, 9, 12, 15])


def test_the_offset_is_fixed_by_the_first_observation_not_per_step():
    """A window that slides mid-scene is a bug, not a slid window."""
    validate, torch = _validate_observation_fn()
    with pytest.raises(ValueError):
        _stream(validate, torch, [3, 6, 9, 12, 15, 21])


def test_a_window_starting_before_the_contract_is_refused():
    validate, torch = _validate_observation_fn()
    with pytest.raises(ValueError):
        _stream(validate, torch, [-3, 0, 3, 6, 9, 12])


def test_the_offset_resets_between_scenes():
    """_clear_cache must reset it, or scene two inherits scene one's window."""
    source = (ROOT / "src" / "models" / "stream_session.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "StreamSession"
    )
    clear = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_clear_cache"
    )
    assert "context_frame_offset = None" in ast.unparse(clear)
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert "self.clear()" in ast.unparse(init), "__init__ must go through clear()"


# ---------------------------------------------------------------------------
# This guard must apply to every model.
#
# It did not. Every check in _validate_observation was written as
#
#     if strict_ball:
#         raise ValueError(...)
#     pass
#
# and strict_ball was only true for a model with a ball token. On the pixel path,
# the only one that ships, a skipped frame, a reordering and a mid-scene slide were
# all accepted silently. The tests above passed only because _Model pretended to
# carry a ball token.
#
# The ball-token branch is gone, so the guard has no reason to vary by model. The
# two tests below hold that, and that no bare `pass` swallows a check again.
# ---------------------------------------------------------------------------


def test_the_guard_does_not_depend_on_any_model_flag():
    """A bare model -- no flags at all -- must still be held to the contract."""
    validate, torch = _validate_observation_fn()

    class _Bare:
        num_cams = 3

    session = _Session()
    session.model = _Bare()
    image = torch.zeros(1, 1, 3, 3, 4, 4)
    with pytest.raises(ValueError):
        validate(session, {
            "context_image": image,
            "context_frame_idx": torch.tensor([[3]] * 3),
        })


def test_no_check_in_the_validator_falls_through_to_pass():
    """A bare `pass` where a raise belongs is how the guard was neutered."""
    source = (ROOT / "src" / "models" / "stream_session.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "StreamSession"
    )
    fn = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_observation"
    )
    assert not [node for node in ast.walk(fn) if isinstance(node, ast.Pass)]
    assert [node for node in ast.walk(fn) if isinstance(node, ast.Raise)]


# ---------------------------------------------------------------------------
# The fourth bug of this kind: assuming there are always 25 targets.
#
#     IndexError: index 22 is out of bounds for dimension 0 with size 22
#
# Sliding eats targets from the tail, leaving 25-k, so any loop over range(25)
# runs off the end. The first three were hardcoded frame NUMBERS, this one a
# hardcoded COUNT -- the same mistake wearing different clothes.
# ---------------------------------------------------------------------------


def test_no_loop_over_a_hardcoded_target_count():
    """A slid window renders 25 - offset targets, never a literal 25."""
    source = _EVAL_SRC.read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "range(25)" in line and not line.strip().startswith("#")
    ]
    assert not offenders, f"loop assumes 25 targets: {offenders}"


def test_the_target_count_is_checked_against_the_offset():
    """Bounding by the tensor alone would hide a truncation from another cause."""
    source = _EVAL_SRC.read_text()
    assert "expected_targets" in source
    assert "len(STREAM25_ALL_TARGET_FRAMES) - int(context_offset)" in source


@pytest.mark.parametrize("offset,targets", [(0, 25), (3, 22), (6, 19), (9, 16)])
def test_rendered_target_count_matches_the_shifted_contract(offset, targets):
    """The count the evaluator asserts is the count the dataset actually yields."""
    _, shifted = shifted_contract(offset)
    assert len(shifted) == targets == len(_BASE_TARGETS) - offset


@pytest.mark.parametrize("offset", [0, 3, 6, 9])
def test_the_anchor_bucket_still_lands_on_the_context_frames(offset):
    """TIME_BUCKETS index the target list, so anchor must stay the six observations."""
    context, targets = shifted_contract(offset)
    anchor_indices = [0, 3, 6, 9, 12, 15]
    assert [targets[i] for i in anchor_indices] == list(context)


@pytest.mark.parametrize("offset,empty", [(0, []), (3, ["farthest"]),
                                          (9, ["near", "mid", "far", "farthest"])])
def test_extrapolation_buckets_empty_out_as_the_window_slides(offset, empty):
    """Not a bug: a slid window has no targets left past its terminal.

    Those buckets carry acceptance gates, so `overall` goes FAIL-by-missing at
    large offsets. catch_position is computed from the terminal render plus an
    analytic extrapolation and does not depend on any bucket.
    """
    buckets = {"near": range(16, 18), "mid": range(18, 20),
               "far": range(20, 22), "farthest": range(22, 25)}
    _, targets = shifted_contract(offset)
    gone = [name for name, idx in buckets.items()
            if all(i >= len(targets) for i in idx)]
    assert gone == empty


# ---------------------------------------------------------------------------
# The zero-extrapolation trap.
#
# landing_index is the last stored frame, absolute 24. The further the window
# slides the closer it gets to the terminal observation, and at offset 9 they
# coincide:
#
#   offset  targets  landing_idx  landing_frame  terminal  frame24 horizon
#      +0     25        24            24           15        0.300 s
#      +3     22        21            24           18        0.200 s
#      +6     19        18            24           21        0.100 s
#      +9     16        15            24           24        0.000 s
#
# frame24_position then degrades into the error at the terminal frame, a smaller
# number that reads like the sliding window helped. Emitting nothing is safer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset,horizon_frames", [(0, 9), (3, 6), (6, 3), (9, 0)])
def test_frame24_horizon_shrinks_and_reaches_zero(offset, horizon_frames):
    _, targets = shifted_contract(offset)
    landing_index = min(_BASE_TARGETS[-1], len(targets) - 1)
    terminal = targets[15]
    assert targets[landing_index] == _BASE_TARGETS[-1], "landing is the last stored frame"
    assert targets[landing_index] - terminal == horizon_frames


def test_zero_horizon_suppresses_the_frame24_metrics():
    """A flattering wrong number is worse than a missing one."""
    source = _EVAL_SRC.read_text()
    assert "frame24_errors = [] if dt <= 0 else" in source, (
        "frame24_* must be skipped when the window leaves no extrapolation"
    )


def test_startup_prints_both_horizons():
    """The operator should see the remaining horizon before the run, not after."""
    source = _EVAL_SRC.read_text()
    assert "frame24 horizon" in source and "catch horizon" in source


@pytest.mark.parametrize("offset,seconds", [(0, 1.0), (3, 0.9), (6, 0.8), (9, 0.7)])
def test_catch_horizon_stays_positive_and_shrinks_with_the_offset(offset, seconds):
    """catch_position targets an absolute instant, so it stays well defined.

    The per-frame step is timespan / (last target - first target) = 0.8 / 24,
    so frame 15 to frame 45 is 1.000 s, which is what the report prints as
    `catch horizon s`. Sliding the window is exactly what shortens it, and that
    shortening is the point: the velocity error enters the landing multiplied
    by this number.
    """
    catch_frame, span, timespan = 45, 24, 0.8
    _, targets = shifted_contract(offset)
    horizon = (catch_frame - targets[15]) * timespan / span
    assert horizon > 0
    assert abs(horizon - seconds) < 1e-9
