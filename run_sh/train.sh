#!/usr/bin/env bash
set -euo pipefail

# 通用训练启动：单卡走 python，多卡自动走 torchrun。
# GPUS 里写几张卡就用几张；命令行额外参数会透传给 main_slarm.py。
#
# ★ 卡数是实验口径的一部分，不要顺手改。
#   batch_size 是 per-GPU，全局 batch = batch_size x 卡数，代码里没有梯度累积
#   （main_slarm.py:702，全仓库无 accum_iter），而且各 config 都写死了 lr，
#   `if args.lr is None: args.lr = args.blr * global_batch_size / 256` 这条
#   自动缩放不会触发 —— 所以卡数减半 = 全局 batch 减半 + 每样本步长翻倍，
#   两个变量一起动。
#
#   6.5cm 这一系列（exp0827_003 退火 / exp0827_001,002 A-B / exp0829_001 in-trunk）
#   全部是 2 卡跑的，互相可比。改成别的卡数，跟这些的横比就不成立了。
#   核实某次实验实际用了几张：
#       grep "Global batch size" work_dirs/slarm/<exp_name>/logs/log.txt
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml}"
# 默认：0908_10k 像素路径微调。起点是 exp0908_001/ckpt_019999.pth（config 里的
# load_from），不恢复旧 optimizer/步数。
#
# 开跑前核对权重确实加载上了（strict=False 会静默丢弃不匹配的 key）：
#   SLARM_SINGLE_PROCESS=1 python tools/check_model_init.py --config "${CONFIG}"
# 换卡：GPUS=4,5,6,7 bash run_sh/train.sh
# 换权重：bash run_sh/train.sh --load_from /path/to/ckpt.pth

# 断点续训：改成 1，其他什么都不用动（CONFIG/GPUS 保持和中断那次一致即可）。
# ckpt 目录由 output_dir/project/exp_name 推出来，会自动挑最新的一个接着跑，
# 权重/optimizer/loss_scaler/迭代数/采样器进度全部恢复，config 里的 load_from 会被忽略。
RESUME="${RESUME:-0}"

# ★ constant LR 跑到底的训练，末点 ckpt 只是盆地里的一次采样 —— exp0827_003 补了
#   6k 步 cosine（1e-4 → 1e-6）之后，pos15 median 改善 33%、p95 改善 50%。
#   此后每一组训练都带 LR 衰减。

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEVICE_NUM=$(awk -F',' '{print NF}' <<< "${GPUS}")

export CUDA_VISIBLE_DEVICES="${GPUS}"
export FEAT_DIST=1

echo "[train] config=${CONFIG} gpus=${GPUS} resume=${RESUME}"

# 渲染分块 stream25_render_target_chunk_size 由各 config 控制（不在此硬编码覆盖，
# 否则命令行会盖过 YAML）；这里只开 TensorBoard。
# TensorBoard event 写到 <output_dir>/<project>/<exp_name>/tensorboard/
EXTRA_ARGS=(--enable_tensorboard)

if [ "${RESUME}" = "1" ]; then
    # --auto_resume 是按**文件 mtime**挑最新 ckpt（misc.load_model），不是按步数。
    # 如果 ckpt 被 cp/rsync/scp 动过，mtime 顺序会和步数顺序脱节，可能续错点。
    # 那种情况下别用 auto，直接写死：bash run_sh/train.sh --resume_from <ckpt路径>
    # （命令行参数在 EXTRA_ARGS 之后透传，resume_from 优先级高于 auto_resume）
    EXTRA_ARGS+=(--auto_resume)
    echo "[resume] resuming from the newest checkpoint (load_from in the config is ignored)"
fi

if [ "${DEVICE_NUM}" -gt 1 ]; then
    exec torchrun --nproc_per_node="${DEVICE_NUM}" --master_port 16818 \
        main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
else
    exec python main_slarm.py --config="${CONFIG}" "${EXTRA_ARGS[@]}" "$@"
fi
