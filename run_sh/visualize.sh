#!/usr/bin/env bash
set -euo pipefail

# Look at one scene of a dataset before training on it.
#
#   bash run_sh/visualize.sh                  # everything, including the point cloud
#   bash run_sh/visualize.sh --skip-3d        # 2D images and ball coverage only, fast
#   bash run_sh/visualize.sh --list-labels    # print the semantic histogram and exit
#
# Extra arguments pass through to tools/visualize_dataset.py.

DATA_ROOT="data/slarm_data_catch45"
DATASET="ball_catch_6.5cm_triview_catch45"   # must match the annotation JSON's dataset field
SPLIT="training"                             # training / validation
SCENE="scene_5000"
BALL_LABEL=1                                 # confirm with --list-labels

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SCENE_ROOT="${DATA_ROOT}/datasets/${DATASET}/${SPLIT}/${SCENE}"
OUT_DIR="vis_out/${DATASET}/${SCENE}"

[ -d "${SCENE_ROOT}" ] || {
    echo "scene directory not found: ${SCENE_ROOT}"
    echo "check that DATA_ROOT / DATASET / SPLIT / SCENE match the actual layout:"
    echo "  <DATA_ROOT>/datasets/<DATASET>/<SPLIT>/<SCENE>/{front_left,front_right,lower_front}/vis/"
    exit 1
}

mkdir -p "${OUT_DIR}"

echo "scene: ${SCENE_ROOT}"
echo "out:   ${OUT_DIR}"
echo "ball_label: ${BALL_LABEL}"
echo ""

PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" tools/visualize_dataset.py \
    --root "${SCENE_ROOT}" \
    --out "${OUT_DIR}" \
    --ball-label "${BALL_LABEL}" \
    "$@"
