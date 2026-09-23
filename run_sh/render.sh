#!/usr/bin/env bash
set -euo pipefail

# 重建渲染启动：单卡。整段前向渲染 [0,N) 帧成 mp4，看重建 / 外推效果。
# 输出目录按 config 名 + ckpt 名自动生成，切换实验 / 权重时不会互相覆盖。

GPUS="0"
CONFIG="configs/ball_training.yml"
CKPT="output/ball_training/checkpoints/ckpt_019999.pth"
SCENE_IDS="0,1,2"      # validation manifest 内的局部下标（不是全局 scene 编号）
NUM_FRAMES=46          # 渲染 [0,N)；>25 为外推（无 GT）。46 一直画到接球帧 45，
                       # 那是任务真正关心、也是没有存储真值的那一帧

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"
export SLARM_SINGLE_PROCESS=1

# 输出路径 = output/stream25_render/<config名>/<ckpt名>/
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
