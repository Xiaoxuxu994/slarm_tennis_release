#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Register a new dataset in src/dataset/constants.py.

Three places need the name and it must match the annotation JSON's "dataset"
field exactly: DATASETS (coordinate mapping), DATASET_DICT (camera_list,
ref_camera, scene_list) and the training config. Typing it three times turns one
wrong character into an obscure KeyError, so this reads the real name from the
data.

It also reports what you want to know before training: the camera list, frame
count, timespan, and whether the name still starts with ball_catch -- datasets.py
branches on that prefix in four places, and a name that misses it silently skips
ball trajectory, semantics and MS3 supervision.

Prints only by default; --write edits constants.py after saving a .bak.

    python tools/register_dataset.py --data-root data/slarm_data [--write]

Standard library only. All output is ASCII English.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
CONSTANTS = WORKTREE / "src" / "dataset" / "constants.py"

# The frozen Stream25 contract, to check the new data has the same shape
EXPECTED_CAMERAS = {
    2: ["front_left", "front_right"],
    3: ["front_left", "front_right", "lower_front"],
}


def _dict_span(src: str, name: str) -> tuple[int, int]:
    """Return the positions of `name = {` and its matching `}`."""
    start = src.index(name)
    brace = src.index("{", start)
    depth, i = 0, brace
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return brace, i
        i += 1
    raise ValueError(f"unbalanced braces in {name}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Register a new dataset in constants.py, reading its real name "
                    "from the annotation JSON instead of retyping it.")
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--dataset", default=None,
                    help="restrict to scene_list files whose name contains this. "
                         "Required once a data_root holds more than one dataset, "
                         "otherwise the first list wins and the wrong one is read")
    ap.add_argument("--write", action="store_true",
                    help="apply the edit (constants.py.bak is kept)")
    args = ap.parse_args()

    root: Path = args.data_root
    lists = sorted((root / "scene_list").glob("*.txt"))
    if not lists:
        print(f"[FAIL] no scene_list/*.txt under {root}")
        return 2

    # One data_root can hold several datasets. Without this filter the first
    # readable annotation may belong to another one, and the tool then reports
    # "already registered" -- a success that did nothing.
    if args.dataset:
        scoped = [l for l in lists if args.dataset in l.stem]
        if not scoped:
            print(f"[FAIL] no scene_list/*.txt matching {args.dataset!r} under {root}")
            print(f"       available: {[l.name for l in lists]}")
            print("       Build them first: tools/make_scene_list.py")
            return 2
        lists = scoped
    else:
        stems = {l.stem.replace("_train", "").replace("_validation", "")
                 .replace("_final_test", "") for l in lists}
        if len(stems) > 1:
            print(f"[FAIL] {root}/scene_list holds more than one dataset: {sorted(stems)}")
            print("       Pass --dataset <name> so the right one is read.")
            return 2

    # First annotation that parses
    js = first = None
    for lst in lists:
        for line in lst.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            p = p if p.is_absolute() else root / p
            if p.exists() and p.suffix == ".json":
                js, first = json.loads(p.read_text(encoding="utf-8")), p
                break
        if js:
            break
    if not js:
        print(f"[FAIL] could not read any annotation JSON listed in {[l.name for l in lists]}")
        print("       If the scene_list holds bare scene names rather than JSON paths,")
        print("       run tools/fix_scene_list.py first.")
        return 2

    name = js.get("dataset", "")
    cams = js.get("camera_list") or list(js.get("camera_to_world", {}).keys())
    n = js.get("num_timesteps")
    t = js.get("normalized_time")
    span = (float(t[24]) - float(t[0])) if isinstance(t, list) and len(t) > 24 else None

    print("=" * 74)
    print("Dataset registration")
    print("=" * 74)
    print(f"data_root   : {root}")
    print(f"annotation  : {first.relative_to(root)}")
    print(f"dataset name: {name!r}")
    print(f"cameras     : {cams}")
    print(f"num_timesteps: {n}")
    print(f"timespan    : {span:.6f}" if span else "timespan    : could not derive")
    print(f"scene_list  : {[l.name for l in lists]}")
    print("")

    problems = []
    if not name:
        problems.append("annotation has no 'dataset' field")
    elif not name.startswith("ball_catch"):
        problems.append(
            f"dataset name {name!r} does not start with 'ball_catch'. datasets.py branches "
            "on startswith('ball_catch') in 4 places; the ball trajectory, semantics and "
            "MS3 would all be skipped silently")
    expect = EXPECTED_CAMERAS.get(len(cams))
    if expect is None:
        problems.append(f"{len(cams)} cameras -- Stream25 contract only covers 2 or 3")
    elif list(cams) != expect:
        problems.append(f"camera list {cams} != contract {expect} (order matters)")
    if isinstance(n, int) and n < 25:
        problems.append(f"num_timesteps={n}, Stream25 needs at least 25")

    for p in problems:
        print(f"[FAIL] {p}")
    if problems:
        print("")
        print("Fix these before registering; a registration that hides one of them just")
        print("moves the failure later, into training, where nothing reports it.")
        return 1

    src = CONSTANTS.read_text(encoding="utf-8")
    if f'"{name}"' in src:
        print(f"[ OK ] {name!r} is already registered in constants.py -- nothing to do.")
        print("")
        print("Config lines for this dataset:")
        print(f"    dataset: [{name}]")
        print(f"    data_root: {root}")
        for l in lists:
            kind = ("train" if "train" in l.stem else
                    "validation" if "valid" in l.stem else None)
            if kind:
                print(f"    {'train' if kind == 'train' else 'eval'}_annotation: scene_list/{l.name}")
        return 0

    train_txt = next((l.name for l in lists if "train" in l.stem), f"{name}_train.txt")
    val_txt = next((l.name for l in lists if "valid" in l.stem), f"{name}_validation.txt")

    entry_datasets = (
        f'    "{name}": {{"opencv2dataset": opencv2waymo, "canonical_to_flu": np.eye(4)}},\n'
    )
    cam_lines = "".join(
        f"            {k}: {v!r},\n" for k, v in sorted(EXPECTED_CAMERAS.items())
    )
    entry_dict = (
        f'\n    "{name}": {{\n'
        f'        "size": [320, 240],\n'
        f'        "temporal": True,\n'
        f'        "num_context_timesteps": 6,\n'
        f'        "num_target_timesteps": 7,\n'
        f'        "annotation_txt_file_train": "scene_list/{train_txt}",\n'
        f'        "annotation_txt_file_val": "scene_list/{val_txt}",\n'
        f'        "camera_list": {{\n{cam_lines}        }},\n'
        f'        "ref_camera": "front_left",\n'
        f'    }},\n'
    )

    print("Will insert into DATASETS:")
    print("    " + entry_datasets.strip())
    print("")
    print("Will insert into DATASET_DICT:")
    for line in entry_dict.strip().split("\n"):
        print("    " + line)
    print("")

    if not args.write:
        print("Dry run. Re-run with --write to apply (constants.py.bak is kept first).")
        print("")
        print("Then, in order:")
        print(f"    python tools/check_dataset_contract.py --data-root {root} \\")
        print(f"        --annotation scene_list/{train_txt} --limit 10")
        print(f"    python tools/check_dataset_contract.py --data-root {root} \\")
        print(f"        --annotation scene_list/{train_txt} --visibility-summary --limit 0")
        return 0

    _, end_datasets = _dict_span(src, "DATASETS = {")
    src = src[:end_datasets] + entry_datasets + src[end_datasets:]
    _, end_dict = _dict_span(src, "DATASET_DICT = {")
    src = src[:end_dict] + entry_dict + src[end_dict:]

    shutil.copy2(CONSTANTS, CONSTANTS.with_suffix(".py.bak"))
    CONSTANTS.write_text(src, encoding="utf-8")
    print(f"[ OK ] written (backup at {CONSTANTS.name}.bak)")
    print("")
    print("Note: constants.py is tracked by git. This edit is uncommitted, so any")
    print("  `git reset --hard` or `git checkout` throws it away, and the next run")
    print("  dies with KeyError on the dataset name -- often long after the pull that")
    print("  caused it. Commit it now:")
    print("")
    print("      git add src/dataset/constants.py")
    print(f"      git commit -m 'Register {name}'")
    print("")
    print(f"    python tools/check_dataset_contract.py --data-root {root} --limit 10")
    print(f"    python tools/check_dataset_contract.py --data-root {root} --visibility-summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
