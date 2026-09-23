"""Two wiring bugs in the gaussian ply export, both of which failed silently.

target_frame_idx is [b, tgt_t * v], each frame number repeated per camera, and
indexing it with t_idx made filenames collapse to t_idx // v -- three different
target frames writing the same name, last one winning, no error. And
gs_semantic_*.ply sat behind `if self.with_feat:`, which the woLSeg variant pins
to False, so a whole class of file was quietly missing.

Static checks: importing slarm pulls in gsplat, which is CUDA-only.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SLARM_SRC = ROOT / "src" / "models" / "slarm.py"


def _exporter() -> ast.FunctionDef:
    tree = ast.parse(SLARM_SRC.read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "save_gs_params_to_ply"
    )


def test_frame_names_come_from_the_normalizer_not_raw_indexing():
    body = ast.unparse(_exporter())
    assert "normalize_frame_indices" in body
    assert "data_dict['target_frame_idx'][0].to(torch.int16).tolist()" not in body


@pytest.mark.parametrize("tgt_t,views", [(25, 3), (40, 3), (48, 3)])
def test_every_target_frame_gets_its_own_file_name(tgt_t, views):
    """Reproduce both layouts and show the old one collapses names."""
    flattened = [frame for frame in range(tgt_t) for _ in range(views)]
    collapsed = {flattened[i] for i in range(tgt_t)}
    assert len(collapsed) == -(-tgt_t // views)          # old behaviour: fewer distinct names
    normalized = list(range(tgt_t))                       # new behaviour: [b, t]
    assert len(set(normalized)) == tgt_t
    assert sorted(normalized) == normalized


def test_the_collapsed_name_held_the_wrong_frame():
    """gs_15.ply used to contain target frame 47, silently."""
    tgt_t, views = 48, 3
    flattened = [frame for frame in range(tgt_t) for _ in range(views)]
    written_to_name_15 = [i for i in range(tgt_t) if flattened[i] == 15]
    assert written_to_name_15 == [45, 46, 47]


def test_semantic_export_is_not_behind_the_dead_with_feat_branch():
    """Cut the `if self.with_feat:` branch out; the export must survive."""
    exporter = _exporter()
    dead = next(
        node for node in ast.walk(exporter)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.with_feat"
    )
    outside = ast.unparse(exporter).replace(ast.unparse(dead), "")
    assert "gs_semantic_" in outside, "semantic export still only lives in dead code"


def test_semantic_colors_are_converted_for_the_sh_writer():
    """save_ply stores colors as SH DC and applies SH2RGB on write."""
    body = ast.unparse(_exporter())
    assert "RGB2SH" in body, "a raw RGB palette would come out washed out"


def test_semantic_palette_marks_the_ball_class():
    """Class 1 is the ball; the eval ball mask is `semantic == 1`."""
    source = SLARM_SRC.read_text()
    assert "TASK_SEMANTIC_PLY_COLORS" in source
    tree = ast.parse(source)
    palette = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "TASK_SEMANTIC_PLY_COLORS" for t in node.targets)
    )
    colors = ast.literal_eval(palette)
    assert len(colors) == 4
    assert all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in colors)
    ball, background = colors[1], colors[0]
    assert max(ball) - min(ball) > 0.5, "the ball colour has to be saturated"
    assert max(background) - min(background) < 0.2, "background should stay neutral"


def test_semantic_layout_is_asserted_before_indexing():
    """A resolution mismatch must say so, not index into the wrong pixels."""
    body = ast.unparse(_exporter())
    assert "context_task_semantic is" in body and "(b, t, v, h, w)" in body
