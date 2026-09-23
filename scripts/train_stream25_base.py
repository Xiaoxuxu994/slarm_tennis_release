"""Tri-view Stream25 launcher that records what it actually launched.

run_sh/train.sh is the everyday entry point; this one exists for the single
thing it does that train.sh does not -- print the sha256 of the config and of
the initial checkpoint before exec. A run whose log carries those two hashes
can be traced back to exactly what produced it, which matters once several
checkpoints share a name.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml


WORKTREE = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = WORKTREE / "configs/ball_training.yml"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _consume_config(arguments: Sequence[str]) -> tuple[Path | None, list[str]]:
    requested = None
    forwarded: list[str] = []
    iterator = iter(arguments)
    for argument in iterator:
        if argument == "--config":
            requested = Path(next(iterator))
        elif argument.startswith("--config="):
            requested = Path(argument.split("=", 1)[1])
        else:
            forwarded.append(argument)
    return requested, forwarded


def build_launch_command(extra_args: Sequence[str]) -> list[str]:
    requested, forwarded = _consume_config(extra_args)
    config_path = (requested or DEFAULT_CONFIG).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"No such config: {config_path}")
    return [
        sys.executable,
        str(WORKTREE / "main_slarm.py"),
        f"--config={config_path}",
        *forwarded,
    ]


def main(extra_args: Sequence[str]) -> list[str] | None:
    command = build_launch_command(extra_args)
    config_path = Path(command[2].split("=", 1)[1])
    config = _load_config(config_path)
    checkpoint = Path(str(config["load_from"]))
    if not checkpoint.is_absolute():
        checkpoint = WORKTREE / checkpoint
    print(f"[Stream25] config={config_path}")
    print(f"[Stream25] config_sha256={_sha256(config_path)}")
    if checkpoint.is_file():
        print(f"[Stream25] initial_checkpoint_sha256={_sha256(checkpoint)}")
    print(f"[Stream25] command={' '.join(command)}")
    if os.environ.get("STREAM25_LAUNCH_DRY_RUN") == "1":
        return command
    os.environ.setdefault("SLARM_OFFLOAD_TARGET_FEAT", "1")
    os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")
    os.execvp(sys.executable, command)
    return None


if __name__ == "__main__":
    main(sys.argv[1:])
