# Acrobot 双连杆

Acrobot 是一个双连杆摆动和平衡任务。目标是使用一个电机扭矩摆动双臂并到达目标位置。

```{video} /_static/videos/acrobot.mp4
:poster: _static/images/poster/acrobot.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## 任务描述

双连杆 Acrobot 由一个铰链关节驱动，该关节由单个电机控制。电机安装在肘关节处，这是系统中唯一的驱动关节。电机的扭矩使连杆在平面内旋转，实现从任意初始角度摆起并到达目标位置。扭矩受限于执行器的 ctrlrange；通过调节其大小和方向，策略必须积累能量以摆起并到达目标，同时保持稳定性。

## 动作空间

| 项目     | 详情                            |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (1,), float32)` |
| **维度** | 1                               |

---

## 观察空间

| 项目     | 详情                            |
| -------- | ------------------------------- |
| **类型** | `Box(-inf, inf, (6,), float32)` |
| **维度** | 6                               |

顺序：`upper_arm_horizontal, lower_arm_horizontal, upper_arm_vertical, lower_arm_vertical, shoulder_velocity, elbow_velocity`。

---

## 奖励函数设计

-   基础稀疏奖励：鼓励末端进入目标区域（半径 = 0.2）
-   持续奖励：在目标区域内每步提供 0.1 的奖励
-   距离奖励：0.3 \* (1.0 - clip(distance / 2.0, 0, 1.0)) 鼓励向目标移动
-   速度惩罚：0.01 \* max(0, velocity_magnitude - 2.0) 惩罚过高的速度

---

## 初始状态

-   肩关节角度随机于 `[-pi, pi]`
-   肘关节角度随机于 `[-pi, pi]`
-   角速度初始化为零

## Episode 终止条件

-   Episode 长度由 `max_episode_seconds` 限制
-   对观察值进行 NaN 检查

---

### 1. 环境预览

```bash
python scripts/view.py env=acrobot
```

### 2. 开始训练

```bash
# 使用默认参数训练
python scripts/train.py task=acrobot/skrl.ppo

# 自定义并行环境数
python scripts/train.py task=acrobot/skrl.ppo num_envs=1024

# 开启训练时渲染
python scripts/train.py task=acrobot/skrl.ppo render=true
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/acrobot
```

### 4. 测试训练结果

```bash
# 自动发现最佳策略（推荐）
python scripts/play.py env=acrobot

# 手动指定带 metadata 的 run 中的 checkpoint
python scripts/play.py env=acrobot policy=/path/to/run/checkpoints/policy-file
```

> **提示**：策略会从 `runs/acrobot/` 中自动选择。使用 `policy=...` 时，checkpoint 所属 run 必须包含 `metadata.json`。

---

## 配置参数

### 环境配置

```{literalinclude} ../../../../../motrix_envs/src/motrix_envs/basic/acrobot/cfg.py
:language: python
:start-after: '# -- docs-tag-start: acrobot-env-cfg --'
:end-before: '# -- docs-tag-end: acrobot-env-cfg --'
```

### 训练配置（PPO 示例）

```{literalinclude} ../../../../../configs/task/acrobot/skrl.ppo.yaml
:language: yaml
```

---

## 预期训练结果

1. Acrobot 能够摆动双臂到达目标位置
2. 末端能够稳定地停留在目标区域内
3. 过高的震荡通过速度惩罚得到减少
4. 策略能够以平滑的动作高效地接近目标
