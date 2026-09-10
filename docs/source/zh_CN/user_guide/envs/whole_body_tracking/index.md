# 全身动作跟踪环境

`ManagerEnv` 是 MotrixLab 面向人形机器人的通用全身动作跟踪（Whole-Body Tracking，WBT）环境。策略在物理仿真中
逐帧跟踪一段参考动作，任务同时约束参考身体的全局位姿、多个身体部位的相对位姿、身体速度和关节可行性。
机器人模型与物理限制由 `RobotCfg` 及其资产提供；`WbtManagerEnvCfg` 选择 motion、跟踪身体、控制缩放、奖励和终止条件。
同一套环境实现因此可以支持不同机器人和不同动作片段。

## 效果演示

以下视频分别展示 Dex-EVT 和 Unitree G1 的舞蹈跟踪，以及 Booster K1 的任意球动作跟踪效果。

::::{grid} 1 1 2 3
:gutter: 2 2 2 2

:::{grid-item-card} Dex-EVT 舞蹈

```{video} /_static/videos/dex-evt-wbt-dance.mp4
:alt: 16 台 Dex-EVT 人形机器人跟踪舞蹈动作
:class: wbt-demo-video
:poster: /_static/images/poster/dex-evt-wbt-dance.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Unitree G1 舞蹈

```{video} /_static/videos/g1-wbt-dance.mp4
:alt: 16 台 Unitree G1 人形机器人跟踪舞蹈动作
:class: wbt-demo-video
:poster: /_static/images/poster/g1-wbt-dance.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Booster K1 任意球

```{video} /_static/videos/k1-wbt-freekick.mp4
:alt: 16 台 Booster K1 人形机器人跟踪任意球动作
:class: wbt-demo-video
:poster: /_static/images/poster/k1-wbt-freekick.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

::::

```{toctree}
:hidden:
:maxdepth: 1

env_design
motion_format
adding_wbt_task

```

## 内置任务

所有内置任务均使用 MotrixSim NumPy 后端，并提供一份 `motrix.fastsac` 训练配置。`algo.asynchronous` 用于选择同步或
异步执行，不改变算法身份。每个 Env ID 选择一份完整配置，包括机器人、motion、跟踪身体、奖励和终止条件。点击
训练曲线缩略图可查看 SVG 大图。

:::{div} task-table
| Env ID | 机器人 | 参考动作 | 时长 | 已提供的训练配置 | 训练曲线 |
| --- | --- | --- | ---: | --- | --- |
| `g1-29dof-wbt-largebox` | Unitree G1 29-DoF | `sub3_largebox_003.npz` | 6.50&nbsp;s | `motrix.fastsac` | — |
| `g1-wbt-dance` | Unitree G1 29-DoF | `dance1_subject2.npz` | 19.98&nbsp;s | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="g1-wbt-dance-curve" aria-label="放大 Unitree G1 舞蹈 WBT 训练曲线"><img src="../../../_static/images/performance/g1-wbt-dance.svg" alt="Unitree G1 舞蹈 WBT 训练曲线" width="180"></button> |
| `dex-evt-wbt-dance` | Dex-EVT | `dance1_easy.npz` | 39.72&nbsp;s | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="dex-evt-wbt-dance-curve" aria-label="放大 Dex-EVT 舞蹈 WBT 训练曲线"><img src="../../../_static/images/performance/dex-evt-wbt-dance.svg" alt="Dex-EVT 舞蹈 WBT 训练曲线" width="180"></button> |
| `k1-wbt-freekick` | Booster K1 | `freekick_shoot_arc_02.npz` | 2.50&nbsp;s | `motrix.fastsac` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="k1-wbt-freekick-curve" aria-label="放大 Booster K1 任意球 WBT 训练曲线"><img src="../../../_static/images/performance/k1-wbt-freekick.svg" alt="Booster K1 任意球 WBT 训练曲线" width="180"></button> |
:::

<dialog id="g1-wbt-dance-curve" class="training-curve-dialog" aria-labelledby="g1-wbt-dance-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../../_static/images/performance/g1-wbt-dance.svg" alt="Unitree G1 舞蹈 WBT 训练曲线">
  <p id="g1-wbt-dance-curve-caption">Unitree G1（<code>g1-wbt-dance</code>）训练曲线</p>
</dialog>

<dialog id="dex-evt-wbt-dance-curve" class="training-curve-dialog" aria-labelledby="dex-evt-wbt-dance-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../../_static/images/performance/dex-evt-wbt-dance.svg" alt="Dex-EVT 舞蹈 WBT 训练曲线">
  <p id="dex-evt-wbt-dance-curve-caption">Dex-EVT（<code>dex-evt-wbt-dance</code>）训练曲线</p>
</dialog>

<dialog id="k1-wbt-freekick-curve" class="training-curve-dialog" aria-labelledby="k1-wbt-freekick-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../../_static/images/performance/k1-wbt-freekick.svg" alt="Booster K1 任意球 WBT 训练曲线">
  <p id="k1-wbt-freekick-curve-caption">Booster K1（<code>k1-wbt-freekick</code>）训练曲线</p>
</dialog>

表中的曲线展示 FastSAC 异步训练过程：横轴为累计环境步数并标注训练墙钟时间，纵轴为平均 Episode 回报；部分曲线
还给出相对于训练时限的存活率。平均回报和存活率在训练早期快速提升，约 6 ～ 7 分钟内即可进入有效跟踪阶段，继续训练
后逐步趋于稳定，体现出并行仿真对动作跟踪任务的快速迭代能力。不同 motion 的时长、难度和奖励尺度不同，曲线数值
适合用于观察各自的学习进展，不宜直接横向比较。

训练 CLI 不支持用 `motion_file=...` 临时替换动作；接入新动作时应按照[新增 WBT 训练任务](adding_wbt_task.md)注册新的
Env ID。

## 运行内置任务

Motion 文件由 Git LFS 管理。克隆后若 `.npz` 仍是 pointer 文本，先执行 `git lfs pull`。

### 运动学 Replay

先用目标机器人的 `RobotCfg` 检查参考动作：

```bash
python scripts/motion/replay.py \
  --robot g1-29dof \
  --motion motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/g1/dance1_subject2.npz
```

`replay.py` 直接写入 motion 的浮动根和关节状态并运行正向运动学，不执行策略、物理控制、奖励或 WBT 终止逻辑。
训练前应确认根轨迹连续、左右肢体和关节方向正确，并且整段动作没有名称映射或四元数错误。Motion 的字段与转换方法见
[Motion 文件格式](motion_format.md)。

### 训练

从“内置任务”表格中选择 Env ID 和对应的训练配置，并替换下列命令中的 `ENV_ID` 与 `TRAINING_CONFIG`：

```bash
python scripts/train.py task=ENV_ID/TRAINING_CONFIG
```

例如，使用 FastSAC 异步训练 G1 舞蹈跟踪任务：

```bash
python scripts/train.py task=g1-wbt-dance/motrix.fastsac algo.asynchronous=true
```

内置配置默认使用 2048 个并行环境，并按任务设置学习迭代次数。只验证 registry、shape 和训练入口时，可以使用不会
生成可用策略的小规模 smoke test：

```bash
python scripts/train.py task=g1-wbt-dance/motrix.fastsac \
  algo.asynchronous=true num_envs=64 algo.trainer.num_learning_iterations=100
```

训练期间，环境从 motion 的不同时间点开始，并利用失败记录提高困难片段的采样概率。奖励分项写入
`info["Reward"]`，bad-tracking 比例、motion 进度和自适应采样统计写入 `info["metrics"]`；各项定义见
[任务环境设计](env_design.md)。

### 回放策略

```bash
python scripts/play.py env=ENV_ID num_envs=16
```

`play.py` 自动选择该环境最新一次 metadata-backed run 的最佳策略。WBT 的 play 配置从 motion 第 0 帧开始，关闭
reset noise 和自适应采样，并移除 10 秒训练时限；播放到 clip 末尾后从第 0 帧重新开始。训练产物和 checkpoint 选择规则见
[训练产物：runs 目录与 checkpoint 结构](../../tutorial/runs_and_checkpoints.md)。
