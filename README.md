# SLARM 流式重建系统文档（woLSeg_v2）

本仓是 SLARM 的精简版：只保留**流式重建（Stream25 base）**的训练 / 评估 / 推理，
移除了下游接球模块（CatchStateReader）、STORM、data_gen 采集链路，以及 LSeg 特征监督。

## 1. 系统目标

系统接收六次同步三目观测：

```text
frame 0 → 3 → 6 → 9 → 12 → 15
每次观测：front_left + front_right + lower_front 的 RGB + 相机内外参 + 物理时间
```

端到端流程：

```text
三目稀疏观测(6 次)
  → ViT/Aggregator 因果时空聚合(window_6)
  → 逐帧 Gaussian + MS3 运动
  → 渲染 25 帧:RGB / metric depth / 四类语义 / MS3
  → frame 16–24 由 frame-15 terminal context 外推
```

即：用六次观测重建 `[0,25)` 共 25 帧的 RGB、metric depth、四类语义和 MS3，并从最后一次观测外推到 frame 24。

## 2. 冻结的三目与时序合同

### 2.1 相机

相机顺序不可改变：

| 顺序 |     名称      | rig/FLU offset（m）  | Pitch  |
| :--: | :-----------: | :------------------: | :----: |
|  0   | `front_left`  | `[0.00,+0.20,0.00]`  |  `0°`  |
|  1   | `front_right` | `[0.00,-0.20,0.00]`  |  `0°`  |
|  2   | `lower_front` | `[+0.30,0.00,-1.00]` | `+27°` |

`front_left` 是 canonical/reference camera。Pitch 只属于 `lower_front` 的相机外参，rig/FLU 坐标系本身没有旋转。rig 原点位于离地 1.5 m，因此 `lower_front` 的世界高度为 0.5 m。

### 2.2 图像与 token

| 项目                 |              数值 |
| -------------------- | ----------------: |
| 输入高×宽            |         `320×240` |
| Patch size           |               `8` |
| 每目 patch           |      `40×30=1200` |
| Aggregator embed dim |             `768` |

### 2.3 时间

- 原始仿真以 30 FPS 记录 30 帧，用于接触边界和数据完整性审计；
- SLARM 训练/评估使用半开区间 `[0,25)`，即 frame 0–24；
- 只有 `[0,3,6,9,12,15]` 进入因果上下文；
- 每个训练 step 采样 7 个 target：1 anchor、2 interpolation、4 extrapolation；
- validation/test 对 25 帧完整评估；
- frame 16–24 由 frame-15 terminal context 独占外推责任。

## 3. 相对原始 SLARM 的网络改动

### 3.1 真流式 `window_6`

[`src/models/stream_session.py`](src/models/stream_session.py) 提供真正的六步因果会话：每次只接收 `[B,1,V,C,H,W]`，维护 Aggregator/CameraHead KV cache，并强制帧序为 `[0,3,6,9,12,15]`。

在 `terminal_context_extrapolation=True` 时，前五次调用只累计上下文和 Gaussian；第六次调用后才统一渲染全部 targets。

> 注：训练与 `render_stream25_base.py` 走「整段一次前向 `model(input_dict)`」，`eval_stream25_base.py` 与 `inference_stream.py` 走「StreamSession 逐帧」。两条路径靠 window 因果 mask 设计上等价。

### 3.2 Terminal-context 外推

[`src/models/slarm.py`](src/models/slarm.py) 的 `terminal_context_extrapolation`：

- anchor/interpolation 仍由相应上下文表征负责；
- frame 16–24 的动态 Gaussian 由 frame-15 表征推进；
- MS3 对 Gaussian 的位置进行连续时间更新；
- `render_target_chunk_size` 支持按 target 分块渲染，降低峰值显存。

### 3.3 四类任务语义

`enable_task_semantic_head` 让每个 Gaussian/patch 预测四类 logits：

```text
0 background   1 ball   2 floor   3 obstacle
```

语义 logits 与 Gaussian 使用同一几何和 opacity 渲染，因此 RGB、深度、语义和 MS3 在像素上保持对齐。类别权重必须由正式 train split 的像素频率计算，采用 inverse-square-root weighting，并 cap 到 10。

### 3.4 Dense MS3 重建

MS3 使用 9 个通道：

```text
[vx,vy,vz, ax,ay,az, jx,jy,jz]
```

球区域监督仿真速度、重力 `[0,0,-9.81]` 和零 jerk；有效静态区域监督零运动。
运动状态随同 Gaussian 几何一起渲染到 target view，而不是单独预测一张无几何约束的运动图。

## 4. 仓库布局

```text
main_slarm.py          训练主程序（入口 + 全部 argparse + 训练循环）
engine_tools.py        build_model / evaluate 等共享库（被多处 import，留在根）

scripts/               面向使用的入口 py
  ├─ train_stream25_base.py    记录 config/ckpt sha256 后 exec main_slarm.py
  ├─ eval_stream25_base.py     流式重建评估（acceptance 指标）
  ├─ render_stream25_base.py   生成指定帧长的重建视频（整段前向）
  └─ inference_stream.py       StreamSession 流式推理演示

run_sh/                一键启动 sh（内部 cd 到仓根后调用上面的入口）
  ├─ train.sh                  训练：单卡 python / 多卡 torchrun 自动
  ├─ eval.sh                   评估：扫 ckpt / offset，输出目录自动分开
  ├─ render.sh                 渲染三视角未来帧预测
  ├─ visualize.sh              数据集可视化
  ├─ export_gs.sh              导出高斯点云序列
  ├─ verify_sweep.sh           用 verify_physics_extrapolation 扫一串 ckpt 看趋势
  └─ *_stream25_base.sh        上面几个壳调用的底层入口

tools/                 数据准备、评测读数与可视化
  ├─ make_scene_list.py / register_dataset.py / fix_scene_list.py   数据接入
  ├─ check_dataset_contract.py / inspect_trajectory.py              标注体检
  ├─ report_catch.py / catch_success_rate.py / compare_evaluations.py  结果读数
  ├─ pick_scenes.py / export_ball_track.py / animate_ball_forecast.py  出图
  └─ stream25_runtime.py / common.py / export_ply.py                共享库

src/                   核心代码：models / dataset / utils / visualization
configs/               实验 YAML
```

## 5. 关键文件

| 文件 | 用途 |
|---|---|
| `configs/slarm_stream25_24cm_triview_window6.yaml` | 三目 base 模板 |
| `configs/exp0827_003_slarm_stream25_6.5cm_triview_window6_nolseg_anneal.yml` | backbone（cosine 退火终点） |
| `configs/exp0908_001_slarm_stream25_0903_2k_triview_window6_nolseg_4gpu.yml` | 2k 数据前置训练，`exp0915_001` 的起点 |
| `configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml` | **当前主线**：10k 数据像素路径微调 |
| `configs/exp0908_001_eval_on_train_scenes.yml` | 泛化性对照：同一权重在训练场景上评 |
| `run_sh/train.sh` | 训练启动（单/多卡自动） |
| `run_sh/eval.sh` | 评估启动（输出路径自动） |
| `src/models/slarm.py` | 主模型（Gaussian / MS3 / terminal 外推） |
| `src/models/stream_session.py` | 六步因果流式会话 |
| `src/utils/stream25_losses.py` | 重建损失组装 |
| `src/utils/stream25_metrics.py` | 评估指标 / acceptance 门 |

## 6. 环境搭建

在一台干净的机器上，下面这段照抄即可。gsplat 要现场编译，是整个过程里唯一可能出问题的一步。

```bash
git clone git@github.com:Xiaoxuxu994/slarm_tennis_release.git
cd slarm_tennis_release

conda create -n SLARM python=3.10 -y
conda activate SLARM

# gsplat 需要 nvcc。系统已有 CUDA 12.1 可跳过这两行。
conda install -c conda-forge mamba -y
mamba install nvidia/label/cuda-12.1.1::cuda-toolkit -c nvidia/label/cuda-12.1.1 -y
export CUDA_HOME=$CONDA_PREFIX

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# gsplat 固定在这个 commit。换版本会改变光栅化数值，历史指标不再可比。
pip install --no-build-isolation \
    git+https://github.com/nerfstudio-project/gsplat.git@937e29912570c372bed6747a5c9bf85fed877bae

pip install -r requirements.txt
pip install pytest

# 可选：可微体素化
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
```

### 6.1 验证环境（不需要数据）

```bash
pytest tests -q
python -c "import torch, gsplat; print(torch.__version__, torch.cuda.is_available())"
```

17 个文件、197 条用例，几十秒跑完。整个测试链路只依赖 `torch` / `numpy` / `pytest`
/ `yaml` —— 不读数据集、不碰 CUDA，**也不 import gsplat**（要碰模型的测试一律从
AST 里把单个函数取出来执行，正是为了不把 gsplat 拉进来）。

所以这两条命令能分开诊断两件事：

- `pytest` 挂了 = 代码或 torch/numpy 环境有问题
- `pytest` 过了但第二条挂了 = 只是 gsplat 没编译成功，改 `CUDA_HOME` 重装那一条即可

第二条期望输出 `2.3.1+cu121 True`。两条都过再往下走。

## 7. 数据接入

训练和评估都需要数据，仓库里不含数据。拿到一棵裸数据树之后，按下面四步接进来。
**顺序不能换**：后一步依赖前一步的产物。

数据树的形状（`datasets.py` 按这个读）：

```text
<data_root>/
  scene_list/<dataset>_train.txt             每行一个标注 JSON 的相对路径
  scene_list/<dataset>_validation.txt
  datasets/<dataset>/<relative_image_path>   图像，路径由标注 JSON 的字段给出
  <标注 JSON 若干>                            位置任意，scene_list 里写相对 data_root 的路径
```

`<dataset>` 这个名字必须与标注 JSON 里的 `"dataset"` 字段**逐字一致**，而且要以
`ball_catch` 开头 —— `datasets.py` 有 4 处 `startswith("ball_catch")` 分流，名字不匹配
的话球轨迹、语义和 MS3 监督会被**静默跳过**，训练照常跑，只是学不到球。

```bash
# ① 生成 scene_list（等间隔抽验证集，不取末尾连续一段）
python tools/make_scene_list.py --data-root data/slarm_data \
    --dataset ball_catch_triview_0908_10k --val-count 20

# ② 注册到 src/dataset/constants.py（改 DATASETS 和 DATASET_DICT 两处）
python tools/register_dataset.py --data-root data/slarm_data

# ③ 体检：相机数、帧数、timespan、重力、位置与速度是否自洽
python tools/check_dataset_contract.py --data-root data/slarm_data

# ④ 逐帧检查轨迹
python tools/inspect_trajectory.py --data-root data/slarm_data
```

① 和 ② 默认**只打印不落盘**，确认输出无误后再加 `--write` 重跑一次
（② 会先备份 `constants.py.bak`）。

③ 取的是全帧加速度均值，末尾一两帧异常（球落地、被接住、轨迹被截断）会被平均掉，
这是它的盲点，所以 ④ 才要逐帧看。③ 装了 `opencv-python` 会额外校验语义图与可见性
标注是否一致，值得装。

四步都通过之后，把 config 里的四个字段指向它。当前主线 config `exp0915_001` 用的是：

```yaml
dataset: [ball_catch_triview_0908_10k]
data_root: data/slarm_data
train_annotation: scene_list/ball_catch_triview_0908_10k_train.txt
eval_annotation: scene_list/ball_catch_triview_0908_10k_validation.txt
```

> **待补**：数据本身的产出／同步方式（仿真器导出参数、rsync 来源），以及初始权重
> `ckpt_019999.pth` 的分发方式。两者都不在仓库里，换机器时要单独搬。

## 8. 训练

```bash
bash run_sh/train.sh
```

默认：4 卡、`configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml`，从 config 里 `load_from` 指向的 `ckpt_019999.pth`
初始化。改卡数或 config 用环境变量，不必编辑脚本：

```bash
GPUS=0                  bash run_sh/train.sh        # 单卡，自动走 python 而非 torchrun
GPUS=4,5,6,7            bash run_sh/train.sh
CONFIG=configs/<别的>.yml bash run_sh/train.sh
RESUME=1                bash run_sh/train.sh        # 断点续训，其他都不用动
bash run_sh/train.sh --num_iterations 30000         # 额外参数透传给 main_slarm.py
```

**开跑前先确认权重真的加载上了**：

```bash
SLARM_SINGLE_PROCESS=1 python tools/check_model_init.py \
    --config configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml
```

输出全部落在 `output/<exp_name>/`，由 config 的 `exp_name` 决定：

```text
output/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/
  checkpoints/ckpt_*.pth
  logs/log.txt
  tensorboard/
  videos/
```

换到别处用 `bash run_sh/train.sh --output_dir /abs/path`。

> **关于 `work_dirs/`**：输出根目录是这次整改才从 `work_dirs/slarm/` 改成 `output/` 的，
> `exp_name` 没动，所以新旧两条路径只差一个根。已经训好的权重仍在
> `work_dirs/slarm/<exp_name>/checkpoints/`，没有搬动 —— 各 config 的 `load_from`
> 指向那里是对的。想统一就把旧目录搬过去：
>
> ```bash
> mkdir -p output && mv work_dirs/slarm/* output/
> ```
>
> 搬完之后把各 config 里 `load_from` 的 `work_dirs/slarm/` 前缀改成 `output/`。

### 8.1 卡数是实验口径的一部分

`batch_size` 是 per-GPU，全局 batch = `batch_size × 卡数`，代码里**没有梯度累积**，
而且各 config 的 `lr` 是写死的、不随卡数自动缩放。所以卡数减半 = 全局 batch 减半
**且**每样本步长翻倍，两个变量一起动，跟别的实验就不可比了。核对某次实验实际用了几张：

```bash
grep "Global batch size" output/<exp_name>/logs/log.txt
```

### 8.2 loss

`src/utils/stream25_losses.py` 组装以下重建 loss（woLSeg 版已移除 LSeg 特征监督）：

| Loss | 权重 |
| --- | ---: |
| full RGB | `1.00` |
| LPIPS | `0.05` |
| ball RGB | `0.50` |
| full depth relative | `1.00` |
| ball metric depth | `0.02` |
| four-class semantic | `1.00` |
| ball MS3 | `1.00` |
| static MS3 | `0.25` |
| opacity regularization | `0.10` |

球在 frame 16–24 某一目离屏时，只把该 frame-eye 的 ball-region loss/metric 记为 N/A；
全图 loss 仍然有效，也不会冻结该视角后续全部帧。

## 9. 评估

独立评估，不继续训练。按 `[0,3,6,9,12,15]` 真流式（StreamSession）输入，重建 frame 0–24。

编辑 `run_sh/eval.sh` 顶部的 `CONFIG` 和 `CKPTS`（支持通配符，会逐个评测并自动出跨
ckpt 对照表），然后：

```bash
bash run_sh/eval.sh
python tools/report_catch.py output/stream25_eval/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/ --markdown
```

第二条把整个 ckpt 扫描读成一张表：frame24 与接球点的落点误差（中位／p90／p95）和成功率。
输出在 `output/stream25_eval/<config名>/<ckpt名>/`，含 `evaluation.json`
与 `evaluation.md`，切换 config／ckpt 不会互相覆盖。

### 9.1 成功率怎么算

判据是几何的：球必须整体穿过圆环，所以球心到真值落点的距离要小于

```
圆环半径 − 球半径 = 0.135 − 0.0325 = 0.1025 m
```

报表对同一阈值给两个距离：**3D 距离**（保守，是成功率的数学下界）和**平面内距离**
（把误差沿球到达方向分解，只算垂直于该方向的分量 —— 沿轴的误差只是早到晚到，
不影响能不能穿过环）。汇报时用哪个都行，但要说清是哪个。

### 9.2 滑动观测窗口

`OFFSETS=(0 3 6 9)` 可以把观测窗口整体后移而**不需要重训** —— 时间嵌入是相对窗口
第一帧算的，窗口后移时模型收到的时间值逐字节不变，变的只有图像里球更近。

| offset | context | 窗口中点 Z | 终端帧 | 到 frame 45 的外推 |
|---|---|---|---|---|
| 0 | 0,3,…,15 | 4.78 m | 15 | 1.005 s |
| 9 | 9,12,…,24 | 3.81 m | 24 | 0.700 s |

★ 跨 offset **只能比 `catch_position`**。`frame24_*` 的外推时长随窗口滑动而变，
测的是不同时刻的量，横着比没有意义。offset 0 逐字节等于冻结契约，历史数字仍可比。

## 10. 可视化

```bash
# 三视角未来帧预测视频（整段前向，默认渲染到 frame 45 的外推段）
bash run_sh/render.sh

# 从 evaluation.json 里挑场景：落点最准 + 球重建最好
python tools/pick_scenes.py \
    output/stream25_eval/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/<ckpt名>/evaluation.json

# 导出球轨迹（.csv/.json/.html/_3d.png）。--scene 给逗号列表时一次模型加载跑多个场景，
# 文件名带 _scene0003 后缀；单场景时就用 --output 给的名字。
python tools/export_ball_track.py \
    --config configs/exp0915_001_slarm_stream25_0908_10k_pixel_finetune.yml \
    --checkpoint output/exp0915_001_slarm_stream25_0908_10k_pixel_finetune/checkpoints/ckpt_019999.pth \
    --scene 3 --output output_vis/track

# 轨迹动画（只读 csv，不用 GPU，可在笔记本上重排版）
python tools/animate_ball_forecast.py output_vis/track.csv --bare
```

`--bare` 去掉标题和图例，适合贴进有独立说明文字的 PPT。

## 11. 结果

> 待补：训练精度与评估结果。

