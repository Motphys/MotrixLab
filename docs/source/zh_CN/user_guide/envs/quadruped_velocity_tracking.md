# 通用四足速度跟踪环境

`QuadrupedWalkTask` 是 MotrixLab 面向四足机器人的通用速度跟踪环境。它将机体坐标系下的
`[vx, vy, yaw_rate]` 命令、四脚对角小跑参考、动作与观察构造、奖励、终止和 reset 统一为一套共享任务，
不绑定具体机器人模型，也不假设固定的关节数量。接入新机器人时，只需提供 `QuadrupedRobotCfg`、任务所需传感器
和一份 `QuadrupedWalkEnvCfg` 配置，无需复制环境实现。

## 粗糙地形演示

以下视频分别回放 Go1、Go2 和 ANYmal-C 的 `rslrl.ppo` 策略；每段视频同时展示 16 个并行环境。

::::{grid} 1 1 3 3
:gutter: 2 2 2 2

:::{grid-item-card} Unitree Go1

```{video} /_static/videos/go1-walk-rough.mp4
:poster: /_static/images/poster/go1-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} Unitree Go2

```{video} /_static/videos/go2-walk-rough.mp4
:poster: /_static/images/poster/go2-walk-rough.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

:::

:::{grid-item-card} ANYmal-C

```{video} /_static/videos/anymalc-walk-rough.mp4
:poster: /_static/images/poster/anymalc-walk-rough.jpg
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

quadruped_velocity_tracking/adding_robot
quadruped_velocity_tracking/env_design
quadruped_velocity_tracking/config_tuning

```

## 内置机器人

MotrixLab 已为下列机器人提供平地和程序化粗糙高度场配置。所有 Env ID 都注册到同一个
`QuadrupedWalkTask`，但分别选择一份完整配置。点击训练曲线缩略图可查看 SVG 大图。

:::{div} task-table
| Env ID | 机器人 | 地形 | 已提供的训练配置 | 训练曲线 |
| --- | --- | --- | --- | --- |
| `go1-walk-flat` | Unitree Go1 | 平地 | `rslrl.ppo`、`skrl.ppo` | — |
| `go1-walk-rough` | Unitree Go1 | 程序化粗糙高度场 | `rslrl.ppo`、`skrl.ppo` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="go1-walk-rough-curve" aria-label="放大 Unitree Go1 粗糙地形训练曲线"><img src="../../_static/images/performance/go1-walk-rough.svg" alt="Unitree Go1 粗糙地形 RSL-RL PPO 训练曲线" width="180"></button> |
| `go2-walk-flat` | Unitree Go2 | 平地 | `motrix.fastsac`、`rslrl.ppo`、`skrl.ppo` | — |
| `go2-walk-rough` | Unitree Go2 | 程序化粗糙高度场 | `motrix.fastsac`、`rslrl.ppo`、`skrl.ppo` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="go2-walk-rough-curve" aria-label="放大 Unitree Go2 粗糙地形训练曲线"><img src="../../_static/images/performance/go2-walk-rough.svg" alt="Unitree Go2 粗糙地形 RSL-RL PPO 训练曲线" width="180"></button> |
| `anymalc-walk-flat` | ANYmal-C | 平地 | `rslrl.ppo`、`skrl.ppo` | — |
| `anymalc-walk-rough` | ANYmal-C | 程序化粗糙高度场 | `rslrl.ppo`、`skrl.ppo` | <button type="button" class="training-curve-thumbnail" data-training-curve-dialog="anymalc-walk-rough-curve" aria-label="放大 ANYmal-C 粗糙地形训练曲线"><img src="../../_static/images/performance/anymalc-walk-rough.svg" alt="ANYmal-C 粗糙地形 RSL-RL PPO 训练曲线" width="180"></button> |
:::

<dialog id="go1-walk-rough-curve" class="training-curve-dialog" aria-labelledby="go1-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../_static/images/performance/go1-walk-rough.svg" alt="Unitree Go1 粗糙地形 RSL-RL PPO 训练曲线">
  <p id="go1-walk-rough-curve-caption">Unitree Go1（<code>go1-walk-rough</code>）训练曲线</p>
</dialog>

<dialog id="go2-walk-rough-curve" class="training-curve-dialog" aria-labelledby="go2-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../_static/images/performance/go2-walk-rough.svg" alt="Unitree Go2 粗糙地形 RSL-RL PPO 训练曲线">
  <p id="go2-walk-rough-curve-caption">Unitree Go2（<code>go2-walk-rough</code>）训练曲线</p>
</dialog>

<dialog id="anymalc-walk-rough-curve" class="training-curve-dialog" aria-labelledby="anymalc-walk-rough-curve-caption">
  <button type="button" class="training-curve-dialog-close" data-training-curve-close aria-label="关闭训练曲线">×</button>
  <img src="../../_static/images/performance/anymalc-walk-rough.svg" alt="ANYmal-C 粗糙地形 RSL-RL PPO 训练曲线">
  <p id="anymalc-walk-rough-curve-caption">ANYmal-C（<code>anymalc-walk-rough</code>）训练曲线</p>
</dialog>

同一机器人的平地与粗糙地形配置复用机器人、控制、命令、奖励、出生范围和终止逻辑，仅在场景地形上区分。
表中的曲线展示粗糙地形任务的训练过程：横轴为累计环境步数，纵轴为平均 Episode 回报，可用于观察策略的学习进展
和收敛趋势。整体回报在训练早期快速提升并趋于稳定，说明策略能够较快学会持续的粗糙地形行走。在当前训练配置与
硬件条件下，本页展示的任务约 1 ～ 2 分钟内即可收敛，体现出并行仿真与训练流程的高吞吐效率。

## 执行命令

从“内置机器人”表格中选择 Env ID 和对应的训练配置，并替换下列命令中的 `ENV_ID` 与 `TRAINING_CONFIG`：

```bash
python scripts/view.py env=ENV_ID num_envs=1
python scripts/train.py task=ENV_ID/TRAINING_CONFIG
python scripts/play.py env=ENV_ID num_envs=16
```

例如，使用 RSL-RL PPO 训练 Go2 粗糙地形任务：

```bash
python scripts/train.py task=go2-walk-rough/rslrl.ppo
```

`view.py` 使用随机动作，仅用于检查场景和模型；训练后的步态需要通过 `play.py` 回放策略查看。
