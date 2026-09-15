"""滑动观测窗口（context offset）。

存在的理由：这是一个**纯数据侧**开关。`datasets.py` 的 `get_frame` 算的是

    dt = time_in_seconds[frame_idx] - time_in_seconds[source_frame_idx]

而 `source_frame_idx = context_frames[0]`，所以模型收到的时间是相对**窗口自己的
第一帧**的。窗口整体后移、source 跟着移，模型看到的时间值逐字节不变（0, 0.1,
..., 0.5 秒），连续的 time_embedder 分辨不出两个窗口。变的只有图像里球更近。

这份测试盯住三件会悄悄毁掉实验的事：

1. offset 0 必须**逐字节**等于冻结契约。任何漂移都会让所有历史数字失效。
2. 不变量 `targets[15] == context[-1]`。eval 里每一处终端读出都写死了目标下标
   15（`pred_depth[15]`、`target_ray_origins[0, 15]`），这条不成立就读错帧，
   而且不会报错，只会给出一个看起来合理的错数。
3. 训练路径必须拒绝非零 offset。训练的 target scheduler 仍然说冻结帧号，
   滑窗后会把移位的 context 和没移位的 target 配在一起。

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

# stream25.py 顶上 import cv2/torch，没装依赖的机器跑不了 import 那几条；
# 纯算术的部分从 AST 里取出来单独执行，任何机器都能跑。
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
    """eval 的终端读出写死了目标下标 15。这条不成立就静默读错帧。"""
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
# eval 侧的接线。纯静态，不 import（eval 会一路拉到 gsplat）。
#
# 存在的理由和 test_eval_report_wiring.py 一样：commit a325081 在一个顶层函数体
# 里引用了没有对应参数的名字，于是每一次 eval 都 NameError。offset 要穿过五层
# 调用，任何一层漏掉参数，指标都会**静默**地按 offset 0 去索引 GT —— 不报错，
# 只是拿终端状态去和错误的时刻比，得到一个看起来正常的数。
# ---------------------------------------------------------------------------

_EVAL_SRC = ROOT / "scripts" / "eval_stream25_base.py"

_MUST_ACCEPT_OFFSET = (
    "compute_stream25_scene_metrics",
    "evaluate_scene",
    "run_evaluation",
    "compute_balltoken_frame24_metrics",
    "_balltoken_fit_metrics",
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

    ★ This check walks the AST rather than scanning lines. A line-based regex
      was written first and missed two of the four real spellings: one where
      the tensor name sat on the previous line of a wrapped call, and one where
      the offset was folded into a variable that was then used as the index.
      Guards that enumerate spellings keep losing to the spellings nobody
      enumerated.
    """
    tree = _eval_tree()
    gt_names = ("ball_position_rig", "ball_velocity_rig", "gt_pos15", "gt_v15",
                "gt_v", "gt_positions", "truth")

    # 污点按**函数作用域**算，不是整模块：两个函数里都有叫 terminal 的局部变量，
    # 一个是目标序下标、一个是绝对帧号，模块级的污点分析会把前者一起误报。
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
        # 取下标里真正出现的标识符，别用字符串切分 —— "(0, terminal)" 被
        # split() 切出来是 ["(0,", "terminal)"]，带括号就永远匹配不上。
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
# StreamSession 的观测校验器
#
# 存在的理由：offset 9 的第一次 eval 在这里炸了 ——
#     ValueError: Expected context frame 0, got [3]
# 校验器把 (0,3,6,9,12,15) 写死在 __init__ 里，dataset 那边滑了窗它不认。
#
# 修法不是把校验关掉。窗口整体后移是允许的，窗口的**形状**不允许变，所以第一次
# 观测确定本场景的 offset，之后每一步仍按契约的步长核对。下面四条把"原来能抓到
# 的现在还能抓到"钉住：跳帧、乱序、重复帧，一条都不能漏。
#
# 方法体从 AST 里取出来单独跑，不用 import（stream_session 会一路拉到 gsplat）。
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
    ball_temporal_refine = True
    ball_prefix_supervision = False
    ball_velocity_residual = False
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
# SLARM._validate_ball_temporal_observation
#
# 存在的理由：这是滑窗撞上的**第二道**写死契约的校验（第一道在 StreamSession）。
#     ValueError: Temporal ball history must match frames 0,3,6,9,12,15
# 它比 StreamSession 那道更难修，因为模型跨调用只靠 cache 传状态，所以 offset
# 必须随 cache 一起走 —— 不然第二次观测就不知道本场景滑了多少。
#
# 这里钉住的正是那条链路：offset 由第一次观测确定、存进 cache、后续从 cache 读。
# 如果 forward 忘了把 offset 放回新 cache（每次调用子模块都返回全新的 dict），
# 第二步就会退回 offset 0 并报错 —— test_offset_must_survive_the_cache_handoff
# 就是为这个失败模式写的。
# ---------------------------------------------------------------------------


def _ball_temporal_validator():
    torch = pytest.importorskip("torch")
    from src.utils.frame_indices import normalize_frame_indices

    source = (ROOT / "src" / "models" / "slarm.py").read_text()
    tree = ast.parse(source)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_ball_temporal_observation"
    )
    # 方法体里那句 `from src.utils.frame_indices import ...` 会自己执行，
    # 这里只需要 torch 和 Tensor 在命名空间里。
    namespace = {"torch": torch, "Tensor": torch.Tensor,
                 "normalize_frame_indices": normalize_frame_indices}
    exec(compile(ast.Module([fn], []), "<validate_ball>", "exec"), namespace)
    return namespace["_validate_ball_temporal_observation"], torch


def _stream_ball(validate, torch, frames, *, drop_offset=False):
    """Six single-observation calls, threading the cache exactly as forward does."""
    session = object()
    cache = None
    for step, frame in enumerate(frames):
        data = {
            "context_image": torch.zeros(1, 1, 3, 3, 4, 4),
            "context_frame_idx": torch.tensor([[[frame]] * 3]),
        }
        offset = validate(session, data, cache, True, aggregator_cache=None)
        cache = {"num_steps": step + 1}
        if not drop_offset:
            cache["context_frame_offset"] = offset
    return cache


@pytest.mark.parametrize("offset", [0, 3, 6, 9])
def test_ball_temporal_validator_accepts_a_slid_window(offset):
    validate, torch = _ball_temporal_validator()
    cache = _stream_ball(validate, torch, [f + offset for f in (0, 3, 6, 9, 12, 15)])
    assert cache["context_frame_offset"] == offset


def test_offset_must_survive_the_cache_handoff():
    """The sub-modules return a fresh cache dict, so forward has to re-attach it."""
    validate, torch = _ball_temporal_validator()
    with pytest.raises(ValueError):
        _stream_ball(validate, torch, [9, 12, 15, 18, 21, 24], drop_offset=True)


def test_ball_temporal_validator_still_catches_a_truncated_history():
    validate, torch = _ball_temporal_validator()
    with pytest.raises(ValueError):
        _stream_ball(validate, torch, [3, 6, 12, 15, 18, 21])


def test_ball_temporal_validator_still_catches_a_reordered_history():
    validate, torch = _ball_temporal_validator()
    with pytest.raises(ValueError):
        _stream_ball(validate, torch, [3, 6, 9, 15, 12, 18])


def test_forward_reattaches_the_offset_to_the_outgoing_cache():
    """Static guard: the sub-module's fresh dict must be stamped every call."""
    source = (ROOT / "src" / "models" / "slarm.py").read_text()
    assert 'context_frame_offset"] = ball_window_offset' in source
    tree = ast.parse(source)
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_ball_temporal_observation"
    )
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, "the validator must hand the derived offset back to forward"


# ---------------------------------------------------------------------------
# 第四次同类 bug：假设"永远有 25 个 target"
#
#     IndexError: index 22 is out of bounds for dimension 0 with size 22
#
# 滑窗会从尾部吃掉 target（offset k 只剩 25-k 个），所以任何按 range(25) 遍历
# 渲染结果的循环都会越界。前三次是写死的**帧号**，这次是写死的**数量** —— 同一
# 个毛病的两种长相，所以下面两条都扫。
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
# 零外推陷阱
#
# landing_index 取的是**存下来的最后一帧**（绝对帧 24）。窗口越往后滑，它离终端
# 观测越近，到 offset 9 就重合了：
#
#   offset  targets  landing_idx  landing_frame  terminal  frame24 外推
#      +0     25        24            24           15        0.300 s
#      +3     22        21            24           18        0.200 s
#      +6     19        18            24           21        0.100 s
#      +9     16        15            24           24        0.000 s
#
# 零外推下 frame24_position 退化成"终端帧的位置误差"，数值比真的落点误差**小**，
# 扫表的人会读成滑窗效果拔群。这比缺一个指标危险，所以那种情况下干脆不发。
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
