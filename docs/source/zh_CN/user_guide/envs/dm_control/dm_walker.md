# 二维步行机器人

二维步行机器人（Walker2D）是 DeepMind Control Suite 中的经典双足行走任务。其目标是训练一个模拟的双足机器人，通过控制其关节力矩，实现站立、行走和奔跑。

```{video} /_static/videos/dm_walker.mp4
:poster: _static/images/poster/dm_walker.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

Walker2D 是一个二维平面的双足机器人任务。由多个身体部位组成，拥有多个受控关节，智能体通过向这些关节施加扭矩作为动作，让机器人实现站立平衡、向前行走或快速奔跑。该任务要求协调的双足步态和平衡控制能力。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (6,), float32)` |
| **维度** | 6                               |

动作对应如下：

| 序号 | 动作含义（施加在关节的力矩） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ---------------------------- | :----: | :----: | :-------------: |
|    0 | 右大腿关节驱动扭矩           |   -1   |   1    |  `right_thigh`  |
|    1 | 右小腿关节驱动扭矩           |   -1   |   1    |   `right_leg`   |
|    2 | 右脚关节驱动扭矩             |   -1   |   1    |  `right_foot`   |
|    3 | 左大腿关节驱动扭矩           |   -1   |   1    |  `left_thigh`   |
|    4 | 左小腿关节驱动扭矩           |   -1   |   1    |   `left_leg`    |
|    5 | 左脚关节驱动扭矩             |   -1   |   1    |   `left_foot`   |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (17,), float32)` |
| **维度** | 17                               |

Walker2D 环境的观测空间由以下部分组成（按顺序）：

| 部分     | 内容说明                   | 维度 | 备注                 |
| -------- | -------------------------- | ---- | -------------------- |
| **qpos** | 各身体关节与根部的位置信息 | 9    | 包括躯干高度和角度   |
| **qvel** | 各身体关节与根部的速度信息 | 8    | 所有关节及躯干的速度 |

---

## 奖励函数设计

walker 的奖励函数由以下几个部分组成：

### 基础站立奖励

```python
# 高度奖励：保持躯干在目标高度
# 直立奖励：保持躯干直立
# 总奖励 = 高度奖励 + 直立奖励
```

### 移动奖励（行走和奔跑任务）

```python
# 速度奖励：追踪目标前进速度
# 站立奖励：保持躯干在目标高度
# 直立奖励：保持躯干直立
# 总奖励 = 速度奖励 + 站立奖励 + 直立奖励
```

---

## 初始状态

-   重置所有有限关节角度到其允许范围内的随机值
-   保持无限幅度关节默认状态

## Episode 终止条件

-   机器人的状态观测值出现异常数值（NaN）
-   机器人躯干接触地面（跌倒）

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=dm-stander
python scripts/view.py env=dm-walker
python scripts/view.py env=dm-runner
```

### 2. 开始训练

```bash
python scripts/train.py task=dm-stander/skrl.ppo
python scripts/train.py task=dm-walker/skrl.ppo
python scripts/train.py task=dm-runner/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-walker
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-stander
python scripts/play.py env=dm-walker
python scripts/play.py env=dm-runner
```

---

## 预期训练结果

### 站立任务 (dm-stander)

1. 躯干高度保持在 1.0-1.4m 范围
2. 躯干直立角度偏差小于 15 度
3. 能够稳定站立不倒

### 行走任务 (dm-walker)

1. 实际行走速度接近 1.0 m/s
2. 步态协调，无明显摔倒
3. 能够持续稳定行走

### 奔跑任务 (dm-runner)

1. 奔跑速度达到 4.0-5.0 m/s
2. 出现飞行相（双脚同时离地）
3. 步态协调稳定
