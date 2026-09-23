#!/usr/bin/env bash
set -euo pipefail

# Train. One GPU runs python, several run torchrun. Extra arguments pass through
# to main_slarm.py.
#
# GPU count is part of an experiment's identity: batch_size is per-GPU, there is
# no gradient accumulation, and lr is fixed in the config, so halving the cards
# halves the global batch and doubles the per-sample step at once.

GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml}"
RESUME="${RESUME:-0}"

# Check the checkpoint really loaded before a long run:
#   SLARM_SINGLE_PROCESS=1 python tools/check_model_init.py --config "${CONFIG}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEVICE_NUM=$(awk -F',' '{print NF}' <<< "${GPUS}")

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FEAT_DIST=1

echo "[train] config=${CONFIG} gpus=${GPUS} resume=${RESUME}"

EXTRA_ARGS=(--enable_tensorboard)

if [ "${RESUME}" = "1" ]; then
    # --auto_resume picks the newest checkpoint by file mtime, not by step, so a
    # copied or rsynced checkpoint can resume from the wrong one. Then pass
    # --resume_from <path> instead.
    EXTRA_ARGS+=(--auto_resume)
    echo "[resume] resuming from the newest checkpoint (load_from in the config is ignored)"
fi

if [ "${DEVICE_NUM}" -gt 1 ]; then
    exec torchrun --nproc_per_node="${DEVICE_NUM}" --master_port 16818 \
        main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
else
    exec python main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
fi
