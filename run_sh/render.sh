#!/usr/bin/env bash
set -euo pipefail

# Render frames [0,N) of a scene to mp4 in one forward pass. One GPU.
# Each config/checkpoint pair gets its own output directory.

GPUS="0"
CONFIG="configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml"
CKPT="output/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/checkpoints/ckpt_019999.pth"
SCENE_IDS="0,1,2"      # indices into the validation manifest, not global scene numbers
NUM_FRAMES=46          # past 25 the model extrapolates and there is no GT; 46 reaches
                       # the catch frame 45, the one the task cares about

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"
export SLARM_SINGLE_PROCESS=1

CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"
TAG="$(basename "${CKPT}" .pth)"
OUT_DIR="output/stream25_render/${CONFIG_NAME}/${TAG}"
mkdir -p "${OUT_DIR}"

[ -f "${CKPT}" ] || { echo "checkpoint not found: ${CKPT}"; exit 1; }

echo "config:     ${CONFIG}"
echo "ckpt:       ${CKPT}"
echo "scene_ids:  ${SCENE_IDS}"
echo "num_frames: ${NUM_FRAMES}"
echo "out:        ${OUT_DIR}"
echo "GPU:        ${GPUS}"
echo ""

bash run_sh/render_stream25_base.sh \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --scene_ids "${SCENE_IDS}" \
    --num_frames "${NUM_FRAMES}" \
    --output_dir "${OUT_DIR}" \
    "$@"

echo ""
echo "done -> ${OUT_DIR}/ (scene_XXXX.mp4)"
