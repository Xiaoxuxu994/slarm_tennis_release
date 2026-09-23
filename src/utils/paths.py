"""One place that decides where a run writes.

The same path used to be assembled by hand in three files -- main_slarm.py,
scripts/inference_stream.py and tools/stream25_runtime.py. Three copies of one
rule is how a checkpoint ends up written somewhere the evaluator does not look,
and nothing reports an error: the trainer saves, the evaluator finds no file
and says so in a way that reads like a missing checkpoint rather than a
disagreement about the layout.

The layout is

    <output_dir>/<exp_name>/checkpoints/ckpt_*.pth
                           /logs/
                           /videos/
                           /tensorboard/

`project` is deliberately NOT part of it. It is the Weights & Biases project
name (src/utils/logging.py), and grouping every experiment under one more
directory named "slarm" inside a repository already called slarm bought a level
of nesting and no information.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def run_dir(args: Any) -> Path:
    """The directory holding everything one run produces."""
    exp_name = getattr(args, "exp_name", None)
    if not exp_name:
        raise ValueError("exp_name is required to locate a run directory")
    return Path(args.output_dir) / exp_name


def checkpoint_dir(args: Any) -> Path:
    return run_dir(args) / "checkpoints"
