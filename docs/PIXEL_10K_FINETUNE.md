# Pixel-only 0908 10k Fine-tune

Config: `configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml`.
Initialize from the original pixel-only exp0908_001 checkpoint019999, NOT the
ball-token checkpoint007999, Exp004 or residual010. Override `--load_from` only
if the matching pixel checkpoint is elsewhere. Start fresh optimizer/scheduler,
not resume. Trunk and pixel heads both train; all ball-token modules are off.
RGB/depth/semantic/MS3 supervision is unchanged. The catch-frame config is
evaluation metadata here: ball-token landing/trajectory losses are disabled.

## Data Contract and Manifests

The provided annotation `annotations/ball_catch_triview_0908_10k/training/scene_20000.json`
was not available locally. Registration assumes existing 0903_2k conventions:
320x240 (H,W), three named views, frames0/3/6/9/12/15, 30fps, same rig axes and
gravity. These are assumptions, not a completed new-data audit. The JSON dataset
field must equal `ball_catch_triview_0908_10k`. Default data_root is `data/slarm_data`;
the annotation path above is relative to it. Images resolve under
`datasets/ball_catch_triview_0908_10k/` according to each JSON's image paths.

The loader needs text manifests, not a single scene JSON as train_annotation:

```
scene_list/ball_catch_triview_0908_10k_train.txt
scene_list/ball_catch_triview_0908_10k_validation.txt
```

Entries are annotation paths relative to data_root. Preserve an existing intended
train/validation split; do not resplit it. If there is NO preassigned split, the
existing helper can create a held-out500 split (first inspect its dry run):

```bash
python tools/make_scene_list.py --data-root data/slarm_data \
  --dataset ball_catch_triview_0908_10k --val-count 500
# Only after confirming that a new split is appropriate:
python tools/make_scene_list.py --data-root data/slarm_data \
  --dataset ball_catch_triview_0908_10k --val-count 500 --write
```

This helper scans the dataset and selects evenly spaced holdouts; it does not
preserve directory-based splits or group correlated trajectories. If scenes
share underlying trajectories, split by trajectory family instead. Keep old
validation trajectories out of the new training set to preserve comparisons.

```bash
python tools/check_dataset_contract.py --data-root data/slarm_data \
  --annotation scene_list/ball_catch_triview_0908_10k_train.txt --limit 20
python tools/check_dataset_contract.py --data-root data/slarm_data \
  --annotation scene_list/ball_catch_triview_0908_10k_validation.txt --limit 20
SLARM_SINGLE_PROCESS=1 python tools/check_model_init.py \
  --config configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml
```

Expect no missing/unexpected model keys for the matching pixel checkpoint.
Check image loading, intrinsics, time and gravity before committing GPU time.

## Train

```bash
GPUS=0,1,2,3 RESUME=0 \
CONFIG=configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml \
bash run_sh/train.sh
```

Head LR2e-5, trunk LR2e-6, cosine, 500-step warmup, 20k steps, batch1/GPU.
Epoch count is `steps * global_batch / actual_training_scene_count`, about8
for10k training scenes; change budgets before starting, not midway through cosine.
No augmentation, loss or architecture experiment is mixed into this data run.

Evaluate the starting pixel checkpoint and intermediate checkpoints on identical
held-out scenes. Track pixel velocity and catch median/p95, not only RGB or total
loss. Also preserve the old100 validation set as a separate cross-dataset check.
The training loop's infrequent built-in evaluation is not Stream25 acceptance;
use the existing `scripts/eval_stream25_base.py` with the new config and matching
checkpoint. Real data/checkpoint testing still requires the server.
