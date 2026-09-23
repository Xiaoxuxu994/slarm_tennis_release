#!/usr/bin/env bash
# Stream25 launcher that logs the config/checkpoint sha256 before exec.
# Usage: train_stream25_base.sh [--config <path>] [extra main_slarm.py args]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"
DEVICE_NUM="${DEVICE_NUM:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export DEVICE_NUM CUDA_VISIBLE_DEVICES FEAT_DIST=1

exec "$PYTHON" scripts/train_stream25_base.py "$@"
