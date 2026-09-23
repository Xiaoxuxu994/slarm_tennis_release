"""Every attribute __init__ reads must be one __init__ has set.

pyflakes cannot see this: `self.ball_velocity_only_train` is an attribute access,
not a name, so removing the assignment and leaving the read behind is invisible to
every static check the repository runs -- and to every test, because constructing
SLARM needs CUDA. It surfaced as an AttributeError on the first line of a four-GPU
training run, after the data was loaded.

Reading the source rather than importing it keeps this runnable anywhere; the
model's own module pulls in gsplat.

    pytest tests/models/test_model_attributes.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: Inherited from nn.Module, so reachable without this class assigning them.
_INHERITED = {
    "training", "apply", "register_buffer", "register_parameter", "named_parameters",
    "named_children", "named_modules", "parameters", "children", "modules",
    "state_dict", "load_state_dict", "to", "cuda", "cpu", "train", "eval",
    "requires_grad_", "zero_grad", "add_module", "get_submodule", "forward",
}

#: Base classes outside this repository, and what they give a subclass.
_EXTERNAL_BASES = {"ConcatDataset": {"datasets", "cumulative_sizes"}}


def _all_classes():
    """Every class defined under src/, by name, so base classes can be resolved."""
    found = {}
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                found.setdefault(node.name, (path.relative_to(ROOT), node))
    return found


_ALL = _all_classes()


def _classes():
    for path, node in _ALL.values():
        if any(isinstance(b, ast.FunctionDef) and b.name == "__init__" for b in node.body):
            yield path, node


def _inherited(cls, seen=None):
    """Names a subclass reaches through its bases: attributes they set, and their
    methods. Correct to read even though this class never assigns them."""
    seen = seen if seen is not None else set()
    out = set()
    for base in cls.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if not name or name in seen:
            continue
        seen.add(name)
        out |= _EXTERNAL_BASES.get(name, set())
        if name not in _ALL:
            continue
        parent = _ALL[name][1]
        out |= set(_self_attrs(parent, ast.Store))
        out |= {b.name for b in parent.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
        out |= _inherited(parent, seen)
    return out


def _self_attrs(node, ctx):
    return {
        n.attr: n.lineno
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
        and isinstance(n.ctx, ctx)
    }


@pytest.mark.parametrize(
    "path,cls", [(p, c) for p, c in _classes()], ids=lambda v: getattr(v, "name", str(v))
)
def test_init_only_reads_attributes_it_has_set(path, cls):
    init = next(b for b in cls.body if isinstance(b, ast.FunctionDef) and b.name == "__init__")
    methods = {b.name for b in cls.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
    stored = set(_self_attrs(cls, ast.Store))
    stored |= {t.id for b in cls.body if isinstance(b, ast.Assign)
               for t in b.targets if isinstance(t, ast.Name)}
    # getattr(self, "x", default) never raises, so a guarded read is not a crash.
    guarded = {
        call.args[1].value
        for call in ast.walk(init)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        and call.func.id == "getattr" and len(call.args) >= 3
        and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str)
    }
    known = stored | methods | guarded | _INHERITED | _inherited(cls)
    dangling = {a: n for a, n in _self_attrs(init, ast.Load).items() if a not in known}
    assert not dangling, (
        f"{path}:{cls.name}.__init__ reads attributes it never sets: "
        + ", ".join(f"self.{a} (line {n})" for a, n in sorted(dangling.items(), key=lambda x: x[1]))
    )
