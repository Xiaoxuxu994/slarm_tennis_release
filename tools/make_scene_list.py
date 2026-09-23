#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build scene_list/*.txt for a freshly dropped dataset tree.

First step of onboarding: scene_list -> register_dataset -> check_dataset_contract
-> inspect_trajectory -> train. Every line must be the annotation JSON's path
RELATIVE TO data_root, which is what datasets.py opens. The validation split is
taken at even intervals over sorted scene names, never as a trailing block: scene
numbers usually track a generation parameter, so the tail is a corner of the
parameter space rather than a sample of the training distribution.

    python tools/make_scene_list.py --data-root data/slarm_data \
        --dataset ball_catch_triview_0908_10k --val-count 20 [--write]

Standard library only. All output is ASCII English.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Identify annotations by these keys, not by filename: scene directories also hold
# camera parameters and render logs, and collecting every *.json would put those in
# scene_list and crash training at the first __getitem__.
ANNOTATION_KEYS = ("dataset", "num_timesteps", "relative_image_path")


def looks_like_annotation(path: Path, dataset: str) -> bool:
    """Is this JSON a scene annotation for this dataset?"""
    try:
        js = json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                        # noqa: BLE001
        return False
    if not isinstance(js, dict):
        return False
    if js.get("dataset") != dataset:
        return False
    return all(k in js for k in ANNOTATION_KEYS)


def find_annotation_jsons(root: Path, dataset: str) -> tuple[list[Path], str]:
    """Find this dataset's annotation JSONs; returns (paths, how they were found).

    Tries the conventional directory first, then scans. Both paths filter on the
    "dataset" field, which is what the dataloader actually keys on and is more
    reliable than the directory layout.
    """
    for sub in ("annotations", "datasets"):
        base = root / sub / dataset
        if base.is_dir():
            found = sorted(p for p in base.rglob("*.json")
                           if looks_like_annotation(p, dataset))
            if found:
                return found, f"{sub}/{dataset}/**/*.json"

    # Fallback scan: slow, and only reached when the layout is unconventional.
    found = sorted(p for p in root.rglob("*.json")
                   if dataset in str(p) and looks_like_annotation(p, dataset))
    return found, 'rglob + "dataset" field match'


def split_indices(n: int, val_count: int) -> tuple[list[int], list[int]]:
    """Even-interval validation split; deterministic across runs."""
    if val_count <= 0:
        return list(range(n)), []
    if val_count >= n:
        return [], list(range(n))
    step = n / val_count
    val = sorted({min(int(i * step + step / 2), n - 1) for i in range(val_count)})
    train = [i for i in range(n) if i not in set(val)]
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build scene_list/*.txt for a freshly dropped dataset tree.")
    ap.add_argument("--data-root", required=True, type=Path,
                    help="e.g. data/slarm_data")
    ap.add_argument("--dataset", required=True,
                    help="dataset name, must match the 'dataset' field in the "
                         "annotation JSON exactly")
    ap.add_argument("--val-count", type=int, default=5,
                    help="scenes held out for validation, sampled at even spacing "
                         "(default 5)")
    ap.add_argument("--write", action="store_true",
                    help="write the files (default: report only)")
    args = ap.parse_args()

    root: Path = args.data_root
    if not root.is_dir():
        print(f"[FAIL] data-root does not exist: {root}")
        return 2

    found, how = find_annotation_jsons(root, args.dataset)
    print("=" * 74)
    print("Scene list builder")
    print("=" * 74)
    print(f"data_root : {root}")
    print(f"dataset   : {args.dataset}")
    print(f"searched  : {how}")
    print(f"found     : {len(found)} annotation JSON(s)")
    if not found:
        print("")
        print("[FAIL] no annotation JSON matched. A file counts only when its top level")
        print(f"       has {list(ANNOTATION_KEYS)} and 'dataset' equals {args.dataset!r}.")
        print("")
        # List what is actually here; the usual failure is a name that is close
        # but not exact.
        dirs = sorted({d.name for sub in ("annotations", "datasets")
                       if (root / sub).is_dir()
                       for d in (root / sub).iterdir() if d.is_dir()})
        if dirs:
            print(f"       Dataset directories under {root}:")
            for d in dirs:
                mark = "  <- closest to what you passed" if (
                    args.dataset in d or d in args.dataset) else ""
                print(f"         {d}{mark}")
            print("       Re-run with --dataset set to the one you want.")
        else:
            print(f"       No annotations/ or datasets/ directory under {root} at all.")
            print("       Check the data-root against the actual tree.")
        return 2

    # Already filtered by dataset field; this just lays out the facts to check.
    sample = json.loads(found[0].read_text(encoding="utf-8"))
    print(f"sample    : {found[0].relative_to(root)}")
    print(f"declared  : {sample.get('dataset')!r}")
    print(f"cameras   : {sample.get('camera_list')}")
    print(f"timesteps : {sample.get('num_timesteps')}")
    n_ts = sample.get("num_timesteps")
    if not isinstance(n_ts, int) or n_ts < 25:
        print("")
        print(f"[FAIL] num_timesteps={n_ts}; Stream25 needs at least 25 frames")
        return 1
    print("")

    train_idx, val_idx = split_indices(len(found), args.val_count)
    rel = [str(p.relative_to(root)) for p in found]
    out_dir = root / "scene_list"
    targets = {
        out_dir / f"{args.dataset}_train.txt": [rel[i] for i in train_idx],
        out_dir / f"{args.dataset}_validation.txt": [rel[i] for i in val_idx],
    }

    for path, entries in targets.items():
        print(f"{path.relative_to(root)}  ({len(entries)} scenes)")
        for e in entries[:3]:
            print(f"    {e}")
        if len(entries) > 3:
            print(f"    ... {len(entries) - 3} more")
    print("")

    if not args.write:
        print("Dry run. Re-run with --write to create the files.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for path, entries in targets.items():
        path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")
        print(f"[ OK ] wrote {path}")
    print("")
    print("Next, in order:")
    print(f"    python tools/register_dataset.py --data-root {root} "
          f"--dataset {args.dataset}")
    print(f"    python tools/register_dataset.py --data-root {root} "
          f"--dataset {args.dataset} --write")
    print(f"    python tools/check_dataset_contract.py --data-root {root} "
          f"--annotation scene_list/{args.dataset}_train.txt --limit 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
