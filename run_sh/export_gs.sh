#!/usr/bin/env bash
set -euo pipefail

# Export Gaussian .ply files for viewing: bash run_sh/export_gs.sh
#
# One call per scene, deliberately: save_gs_params_to_ply names files by frame
# only (gs_15.ply) and gaussian_save_path is fixed when the model is built, so
# several scenes in one call would overwrite each other without an error.

GPUS="0"
CONFIG="configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml"
CKPT="output/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/checkpoints/ckpt_019999.pth"

SCENE_IDS="0"    # indices into the validation manifest, not global scene numbers
NUM_FRAMES=25    # three ply files per frame, roughly 30 MB each

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPUS}"
export SLARM_SINGLE_PROCESS=1

[ -f "${CONFIG}" ] || { echo "[FAIL] config not found: ${CONFIG}"; exit 1; }
[ -f "${CKPT}" ]   || { echo "[FAIL] checkpoint not found: ${CKPT}"; exit 1; }

CONFIG_NAME="$(basename "${CONFIG}")"; CONFIG_NAME="${CONFIG_NAME%.*}"
TAG="$(basename "${CKPT}" .pth)"
ROOT="output/gs_ply/${CONFIG_NAME}/${TAG}"

IFS=',' read -ra SCENES <<< "${SCENE_IDS}"
EST=$(( ${#SCENES[@]} * NUM_FRAMES * 3 * 30 ))

echo "config     : ${CONFIG}"
echo "ckpt       : ${TAG}"
echo "scenes     : ${SCENE_IDS}  (${#SCENES[@]} total)"
echo "num_frames : ${NUM_FRAMES}"
echo "out        : ${ROOT}/scene_XXXX/"
echo "estimated  : about ${EST} MB"
echo ""
AVAIL=$(df -Pm . | awk 'NR==2{print $4}')
if [ "${AVAIL}" -lt "$(( EST * 2 ))" ]; then
    echo "[FAIL] only ${AVAIL} MB free here, less than twice the estimate."
    echo "       Reduce NUM_FRAMES or SCENE_IDS."
    exit 1
fi

for SCENE in "${SCENES[@]}"; do
    OUT_DIR="${ROOT}/scene_$(printf '%04d' "${SCENE}")"
    mkdir -p "${OUT_DIR}"    # save_gs_params_to_ply will not create it
    echo "=========================================================="
    echo "scene ${SCENE} -> ${OUT_DIR}"
    echo "=========================================================="
    bash run_sh/render_stream25_base.sh \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --scene_ids "${SCENE}" \
        --num_frames "${NUM_FRAMES}" \
        --output_dir "${OUT_DIR}" \
        --save_gaussian \
        --gaussian_save_path "${OUT_DIR}" \
        "$@"
    echo ""
done

echo "=========================================================="
PLY_COUNT=$(find "${ROOT}" -name '*.ply' | wc -l | tr -d ' ')
echo "${PLY_COUNT} ply files, $(du -sh "${ROOT}" | cut -f1) total -> ${ROOT}/"
echo ""
echo "Three files per target frame:"
echo "  gs_<frame>.ply           gaussians, already moved to that instant by MS3"
echo "  gs_rgb_<frame>.ply       RGB coloured"
echo "  gs_semantic_<frame>.ply  semantic coloured; the ball is class 1"
echo ""
echo "Viewing:"
echo "  MeshLab     opens them as a point cloud only -- vertex colours are right,"
echo "              opacity/scale/rotation ignored, no splatting."
echo "  SuperSplat  superspl.at/editor, drag the file in, real gaussian rendering."
echo "              Standard INRIA format: scale in log, opacity in logit."
