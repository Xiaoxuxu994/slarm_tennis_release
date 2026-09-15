# 0908 10k 数据验证命令

在仓库根目录运行，不需要 GPU。以下假设数据根目录是
`data/slarm_data`；如果实际位置不同，只修改 `DATA_ROOT`。

先检查数据，再开始 pixel-only 微调。这些命令不修改模型或 checkpoint。

## 1. 确认标注和场景清单

```bash
export DATA_ROOT=data/slarm_data
export DATASET=ball_catch_triview_0908_10k

ls -lh "$DATA_ROOT/annotations/$DATASET/training/scene_20000.json"

ls -lh "$DATA_ROOT/scene_list/${DATASET}"*.txt
```

后续命令需要以下两个清单，清单内每行是相对 `DATA_ROOT` 的标注 JSON 路径：

```text
scene_list/ball_catch_triview_0908_10k_train.txt
scene_list/ball_catch_triview_0908_10k_validation.txt
```

如果已有正确清单，直接进入第 2 步，不重新划分数据。

如果没有清单，先扫描，只查看结果，不写文件：

```bash
python tools/make_scene_list.py \
  --data-root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --val-count 500
```

**如果数据已有 training/validation 目录划分，或需要按轨迹分组划分，
先停在这里并提供扫描结果，不要执行下面的写入命令。**
现有工具会重新选择留出场景，不会保留目录划分，也不会按轨迹家族分组。

仅在确认没有既定划分、同意新留出 500 个场景后，才执行：

```bash
python tools/make_scene_list.py \
  --data-root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --val-count 500 \
  --write
```

不要让旧验证轨迹或它们的重复版本进入新训练清单。

## 2. 全量标注契约检查

`--limit 0` 检查清单中的全部场景。`--json-only` 跳过数据集注册检查，
适合注册代码尚未同步时先检查标注；`--no-images` 跳过图像检查以加快扫描。

```bash
python tools/check_dataset_contract.py \
  --data-root "$DATA_ROOT" \
  --annotation "scene_list/${DATASET}_train.txt" \
  --timespan 0.8 --limit 0 --json-only --no-images
```

```bash
python tools/check_dataset_contract.py \
  --data-root "$DATA_ROOT" \
  --annotation "scene_list/${DATASET}_validation.txt" \
  --timespan 0.8 --limit 0 --json-only --no-images
```

`timespan=0.8` 是拟用配置中 frame0 到 frame24 的时长假设，不代表新数据
已经核验通过。遇到时间、相机或坐标相关错误时，先保留错误输出，不要为了
通过检查直接修改标注或跳过检查。

## 3. 抽查图像与语义标注

此次不要加 `--no-images`。每个清单检查前 20 个场景，不是随机抽样，
也不代表全量图片已经验证。

```bash
for SPLIT in train validation; do
  python tools/check_dataset_contract.py \
    --data-root "$DATA_ROOT" \
    --annotation "scene_list/${DATASET}_${SPLIT}.txt" \
    --timespan 0.8 --limit 20 --json-only --show-pass
done
```

## 4. 全量三视角球可见性统计

统计模式扫描整个清单；它不能代替第 2、3 步的契约和图像检查。
这里只输出统计，不生成剔除名单，也不修改训练数据。

```bash
for SPLIT in train validation; do
  python tools/check_dataset_contract.py \
    --data-root "$DATA_ROOT" \
    --annotation "scene_list/${DATASET}_${SPLIT}.txt" \
    --visibility-summary --num-cams 3
done
```

## 5. 需要回传的结果

- 扫描得到的场景总数、清单数量，以及是否已有目录或轨迹分组划分。
- train/validation 全量契约检查汇总，以及所有 FAIL/WARN。
- 图像抽查中的异常信息。
- train/validation 三视角可见性统计完整输出。

如果第 1 步没有清单或发现已有目录划分，先提供第 1 步结果即可。
如果命令报错，保留完整 traceback，不需要继续强行跑后面的步骤。

这些检查不证明 train/validation 无重复轨迹，也不证明两者运动分布一致。
新数据集注册、实际数据加载和 pixel checkpoint 初始化仍需在配置同步后
单独检查。详细训练方案见 `docs/PIXEL_10K_FINETUNE.md`（若该文档尚未
同步，不影响本页已有验证工具的运行）。
