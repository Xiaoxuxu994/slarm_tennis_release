"""Wiring of the report-writing path.

_finalize_and_write is a top-level function, not a closure inside run_evaluation,
so a reference added to its body without a matching parameter is a NameError on
every eval -- which is what happened once, unconditionally, even with the feature
it referenced turned off. The AST guard that existed then covered the computation
chain and missed the report chain, and no test actually called the function.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.stream25_metrics import CAMERA_ORDER  # noqa: E402


def _load_finalizer():
    """Import the eval script; it pulls in gsplat, so skip without CUDA. The
    static test below needs no import and always runs."""
    pytest.importorskip("gsplat", reason="eval_stream25_base imports the CUDA rasterizer")
    from scripts.eval_stream25_base import _finalize_and_write
    return _finalize_and_write

EVAL_SRC = Path(__file__).resolve().parents[2] / "scripts" / "eval_stream25_base.py"


def _scene_results():
    scopes = {}
    for index, scope in enumerate(("aggregate",) + CAMERA_ORDER):
        scopes[scope] = {
            "metrics": {"rgb_psnr": {"anchor": 28.0 - index}},
            "valid_counts": {"rgb_psnr": {"anchor": 6}},
        }
    return [{"scopes": scopes, "scene_index": 0}]


def _write(tmp_path, **kwargs):
    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"not a real checkpoint, only hashed")
    out = tmp_path / f"result_{kwargs.get('ball_radius_compensation', 0)}.json"
    return _load_finalizer()(
        _scene_results(),
        split="validation",
        checkpoint_path=str(ckpt),
        config_path="configs/does_not_need_to_exist.yml",
        manifest=None,
        evaluation_seed=0,
        reference=False,
        output_json=str(out),
        output_markdown=str(out.with_suffix(".md")),
        **kwargs,
    ), out


def test_writing_the_report_does_not_raise(tmp_path):
    """The basic one: the function is actually called. A NameError lands here."""
    result, out = _write(tmp_path)
    assert out.exists()
    assert json.loads(out.read_text())["overall"] in ("PASS", "FAIL")


def test_compensation_off_is_recorded_as_off(tmp_path):
    result, _ = _write(tmp_path)
    assert result["ball_surface_offset_m"] == 0.0
    assert result["frame24_position_method"].endswith("frame15")


def test_compensation_on_is_stamped_into_the_method_name(tmp_path):
    """With compensation on, the setting must be written beside the numbers, or
    two runs cannot be compared."""
    result, out = _write(
        tmp_path,
        ball_surface_offset=0.021,
        ball_radius=0.0325,
        ball_radius_compensation=0.646,
    )
    assert result["ball_surface_offset_m"] == pytest.approx(0.021)
    assert result["ball_radius_m"] == pytest.approx(0.0325)
    assert result["frame24_position_method"].endswith("_ball_center_compensated")
    assert "compensation" in out.with_suffix(".md").read_text()


def test_no_top_level_function_references_an_unresolvable_name():
    """Static net for a whole class of bug: an edit that landed in the wrong
    function scope.

    The change looked like it was inside run_evaluation but was in
    _finalize_and_write, where no closure exists. Checking every function's free
    variables is far cheaper than an end-to-end test for each one.
    """
    source = EVAL_SRC.read_text(encoding="utf-8")
    # compile(), not ast.parse(): duplicate argument names are a compile-time
    # error that parse lets through, surviving until import.
    compile(source, str(EVAL_SRC), "exec")
    tree = ast.parse(source)
    module_names = set(dir(builtins)) | {"__name__", "__file__"}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            module_names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)

    problems = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        bound |= {a.arg for a in fn.args.posonlyargs}
        for extra in (fn.args.vararg, fn.args.kwarg):
            if extra is not None:
                bound.add(extra.arg)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                bound.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                bound |= {a.asname or a.name.split(".")[0] for a in sub.names}
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(sub.name)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                bound.add(sub.name)
            elif isinstance(sub, (ast.comprehension,)):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(sub, ast.Lambda):
                bound |= {a.arg for a in sub.args.args}
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id not in bound and sub.id not in module_names:
                    problems.append(f"{fn.name}() line {sub.lineno}: {sub.id}")
    assert not problems, "unresolvable free variables:\n  " + "\n  ".join(sorted(set(problems)))
