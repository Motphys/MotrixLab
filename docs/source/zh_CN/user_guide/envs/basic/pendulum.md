# 单摆训练示例

单摆（Pendulum）是经典的单关节摆起并倒立保持任务，目标是用一个电机扭矩把摆甩起并稳定在倒立位置。

```{video} /_static/videos/pendulum.mp4
:poster: _static/images/poster/pendulum.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## 任务描述

单摆由一段杆体和一个铰接关节组成，关节由单个电机驱动（gear 可配置）。电机施加的扭矩使摆杆在平面内旋转，实现从任意初始角度的摆起、倒立和保持。关节力矩受限于 ctrlrange，通过控制扭矩大小与方向，策略需要完成能量累积（swing-up）并在倒立位置维持平衡，同时抑制角速度带来的震荡。

---

## 动作空间

| 项目     | 详情                            |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (1,), float32)` |
| **维度** | 1                               |

---

## 观察空间

| 项目     | 详情                            |
| -------- | ------------------------------- |
| **类型** | `Box(-inf, inf, (3,), float32)` |
| **维度** | 3                               |

顺序：`cos(theta), sin(theta), 角速度`。

---

## 奖励设计

-   倒立奖励：鼓励角度围绕 π（倒立）
-   能量 shaping：能量接近倒立位置
-   惩罚：`角速度^2`、`ctrl^2`、`(ctrl - prev_ctrl)^2`，抑制震荡与过猛动作

---

## 初始状态

-   角度随机于 `[-pi, pi]`
-   角速度小噪声（如配置）
-   控制历史 `prev_ctrl` 初始化为 0

## Episode 终止条件

-   无跌倒终止；仅 NaN 检查
-   Episode 长度由 `max_episode_seconds` 限制

---

### 1. 环境预览

```bash
python scripts/view.py env=pendulum
```

### 2. 开始训练

```bash
# 默认参数训练
python scripts/train.py task=pendulum/skrl.ppo

# 自定义并行环境数
python scripts/train.py task=pendulum/skrl.ppo num_envs=1024

# 开启训练时渲染
python scripts/train.py task=pendulum/skrl.ppo render=true
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/pendulum
```

### 4. 测试训练结果

```bash
# 自动寻找最新/最优策略（推荐）
python scripts/play.py env=pendulum

# 手动指定带 metadata 的 run 中的 checkpoint
python scripts/play.py env=pendulum policy=/path/to/run/checkpoints/policy-file
```

> **提示**：策略默认在 `runs/pendulum/` 下自动发现。使用 `policy=...` 时，checkpoint 所属 run 必须包含 `metadata.json`。

---

## 配置参数

### 环境配置（示例）

```{literalinclude} ../../../../../motrix_envs/src/motrix_envs/basic/pendulum/cfg.py
:language: python
:start-after: '# -- docs-tag-start: pendulum-env-cfg --'
:end-before: '# -- docs-tag-end: pendulum-env-cfg --'
```

### 训练配置（示例 PPO）

```{literalinclude} ../../../../../configs/task/pendulum/skrl.ppo.yaml
:language: yaml
```

---

## 预期训练结果

1. 摆能主动摆起并停留在倒立附近
2. 倒立处的震荡由角速度与控制变化惩罚抑制
