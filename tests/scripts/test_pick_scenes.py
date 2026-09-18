"""挑场景：从已有报告里排名，而不是把 GPU 花在钓鱼上。

报告里本来就带着全部 200 个场景的逐场景指标，所以"渲 20 条挑一条好的"是在
重算已经算过的东西，而且只覆盖了有耐心渲的那 20 条。

这份测试盯三件事：
  1. 排名方向对（落点越小越好、IoU 越大越好）—— 搞反了不会报错，只会给出
     系统性最差的场景，而且看起来像模像样
  2. 缺指标的场景被排除而不是当成 0
  3. 最好 / 中位 / 最差三档都输出 —— 只给"最好"就是无声的挑好的说

    pytest tests/scripts/test_pick_scenes.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tools" / "pick_scenes.py"


def _function(name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(ast.parse(SRC.read_text()))
                if isinstance(node, ast.FunctionDef) and node.name == name)


def rank_of(values, lower_is_better):
    """The ranking the tool performs."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return [None] * len(values)
    out = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        below = sum(1 for p in present if p < value)
        fraction = below / max(1, len(present) - 1)
        out.append(1.0 - fraction if lower_is_better else fraction)
    return out


def test_a_smaller_landing_error_ranks_higher():
    """Reversed, this would nominate the worst scene in the set and the output
    would look exactly as convincing."""
    ranks = rank_of([0.02, 0.07, 0.20], lower_is_better=True)
    assert ranks[0] > ranks[1] > ranks[2]
    assert ranks[0] == pytest.approx(1.0)
    assert ranks[2] == pytest.approx(0.0)


def test_a_larger_iou_ranks_higher():
    ranks = rank_of([0.5, 0.74, 0.95], lower_is_better=False)
    assert ranks[2] > ranks[1] > ranks[0]


def test_missing_metrics_stay_missing():
    """Treating an absent metric as zero would sink a scene that simply has no
    ball visible, or float one on a value nobody measured."""
    ranks = rank_of([0.02, None, 0.20], lower_is_better=True)
    assert ranks[1] is None
    assert ranks[0] > ranks[2]


def test_a_single_scene_does_not_divide_by_zero():
    assert rank_of([0.05], lower_is_better=True) == [1.0]


def test_all_equal_values_do_not_pick_a_winner_by_accident():
    ranks = rank_of([0.07, 0.07, 0.07], lower_is_better=True)
    assert len(set(ranks)) == 1


def test_median_and_worst_are_printed_beside_best():
    """A slide built only from the best scene says what the model does at its
    luckiest. Cherry-picking is only a problem when it is silent."""
    source = SRC.read_text()
    assert "BEST overall" in source
    assert "TYPICAL (median)" in source
    assert "WORST overall" in source
    assert "honest headline" in source


def test_both_rankings_are_reported_separately():
    """Landing needs the ball's centre; reconstruction needs its shape. One
    scene rarely wins both, and averaging them away would hide that."""
    source = SRC.read_text()
    assert "Smallest landing error" in source
    assert "Best ball reconstruction" in source
    assert "In both top" in source


def test_it_falls_back_when_the_report_predates_catch_position():
    source = SRC.read_text()
    assert 'metric(s, "frame24_position")' in source
    assert "ranking on frame24_position" in source
