#!/usr/bin/env bash
set -euo pipefail

# Evaluate. One GPU is enough. Each checkpoint gets its own output directory, so
# nothing overwrites anything.

GPUS="0"
CONFIG="configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml"

# A glob evaluates every matching checkpoint in name order and prints one table
# across them at the end. Sweeping the middle checkpoints is the direct way to
# see overfitting: a metric that falls and then rises only shows up side by side.
#
# Weights from before the output root moved are still under
# work_dirs/slarm/<exp_name>/checkpoints/; swap the prefix below to evaluate those.
CKPTS=(
    "output/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/checkpoints/ckpt_*.pth"
)

# Observation window offset; each value writes to its own directory.
#   0 = frozen contract, context (0,3,6,9,12,15), comparable with every past number
#   9 = context (9,12,15,18,21,24), terminal frame 24
# Only catch_position is comparable across offsets: the frame24_* horizon shrinks
# as the window slides, so those measure different instants. A sweep: OFFSETS=(0 3 6 9)
OFFSETS=(0)

# 1 skips a checkpoint that already has evaluation.json.
SKIP_EXISTING=0

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"

# output/stream25_eval/<config>/<ckpt>/ -- %.* strips any extension, .yml or .yaml.
CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"

# nullglob alone is not enough: it drops a glob that matches nothing, but leaves a
# literal path untouched, so a typo would be evaluated as if it were a checkpoint.
shopt -s nullglob
EXPANDED=()
for pattern in "${CKPTS[@]}"; do
    matches=( ${pattern} )
    [ ${#matches[@]} -gt 0 ] || matches=( "${pattern}" )
    for match in "${matches[@]}"; do
        if [ -f "${match}" ]; then
            EXPANDED+=( "${match}" )
        else
            echo "[skip] checkpoint not found: ${match}"
        fi
    done
done
shopt -u nullglob
[ ${#EXPANDED[@]} -gt 0 ] || { echo "[FAIL] no checkpoint matched CKPTS"; exit 1; }
IFS=$'\n' EXPANDED=( $(printf '%s\n' "${EXPANDED[@]}" | sort -u) ); unset IFS

echo "config : ${CONFIG}"
echo "ckpts  : ${#EXPANDED[@]}"
echo "GPU    : ${GPUS}"
echo ""

# The report also refits (pos15, v15) from the rendered ball centres of all six
# context frames, as "pixel fit *". Choose other frames with
#   bash run_sh/eval.sh --fit-frames 3,6,9,12,15
# Fewer frames shorten the time base and cost accuracy (0.50s -> 0.30s is 1.87x),
# so only drop frames that pixel_pos_error_frame* shows to be much worse.

# Offsets come from the OFFSETS array, never the command line: the output
# directory is named from it, so a flag here would make the name lie.
for arg in "$@"; do
    case "${arg}" in
        --context-offset*|--context_offset*)
            echo "[FAIL] pass offsets through the OFFSETS array, not the command line"
            echo "       (the output directory is named from it; a flag here would lie)"
            exit 1
            ;;
    esac
done
[ ${#OFFSETS[@]} -gt 0 ] || { echo "[FAIL] OFFSETS is empty"; exit 1; }

REPORTS=()
for CKPT in "${EXPANDED[@]}"; do
  for OFF in "${OFFSETS[@]}"; do
    TAG="$(basename "${CKPT}" .pth)"
    # offset 0 keeps the plain name, so past results stay comparable in place.
    if [ "${OFF}" = "0" ]; then
        OUT_DIR="output/stream25_eval/${CONFIG_NAME}/${TAG}"
    else
        OUT_DIR="output/stream25_eval/${CONFIG_NAME}/${TAG}_off${OFF}"
    fi
    mkdir -p "${OUT_DIR}"
    REPORTS+=( "${OUT_DIR}/evaluation.json" )

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_DIR}/evaluation.json" ]; then
        echo "[keep] ${TAG} off${OFF} -> ${OUT_DIR}/evaluation.json"
        continue
    fi

    echo "=========================================================="
    echo "eval   : ${TAG}   context offset +${OFF}"
    echo "out    : ${OUT_DIR}"
    echo "=========================================================="
    bash run_sh/eval_stream25_base.sh \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --split validation \
        --context-offset "${OFF}" \
        --output "${OUT_DIR}/evaluation.json" \
        --output-markdown "${OUT_DIR}/evaluation.md" \
        "$@"
    echo ""
  done
done

echo ""
if [ ${#REPORTS[@]} -gt 1 ]; then
    "${PYTHON:-python}" tools/compare_evaluations.py "${REPORTS[@]}"
else
    echo "done -> ${REPORTS[0]}"
fi
