# 半猎豹机器人

半猎豹机器人(Cheetah)是 DeepMind Control Suite 中的经典连续控制任务。其目标是训练一个模拟的双足机器人，通过控制其关节力矩，实现高速、稳定地奔跑

```{video} /_static/videos/dm_cheetah.mp4
:poster: _static/images/poster/dm_cheetah.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

HalfCheetah 是一个二维半身猎豹奔跑任务。，由 7 个主要 Body 部位组成（1 个躯干和前后腿各 3 节），拥有 6 个受控关节(前后大腿[连接躯干]、胫骨[连接大腿]和脚[连接胫骨])，智能体通过向这些关节施加扭矩作为动作，让猎豹尽可能快速且稳定地向前奔跑。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (6,), float32)` |
| **维度** | 6                               |

关节对应如下：

| 序号 | 动作含义（施加在关节的力矩） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ---------------------------- | :----: | :----: | :-------------: |
|    0 | 后腿大腿关节驱动扭矩         |   -1   |   1    |    `bthigh`     |
|    1 | 后腿小腿关节驱动扭矩         |   -1   |   1    |     `bshin`     |
|    2 | 后腿脚部关节驱动扭矩         |   -1   |   1    |     `bfoot`     |
|    3 | 前腿大腿关节驱动扭矩         |   -1   |   1    |    `fthigh`     |
|    4 | 前腿小腿关节驱动扭矩         |   -1   |   1    |     `fshin`     |
|    5 | 前腿脚部关节驱动扭矩         |   -1   |   1    |     `ffoot`     |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (17,), float32)` |
| **维度** | 17                               |

HalfCheetah 环境的观测空间由以下部分组成（按顺序）：
| 部分 | 内容说明 | 维度 | 备注 |
| -------- | ------------- | -- | ------------ |
| **qpos** | 各身体关节与根部的位置信息 | 8 | 默认不包括根部 x 坐标 |
| **qvel** | 各身体关节与根部的速度信息 | 9 | 速度为位置导数 |

| 序号     | 观察量          | 最小值 | 最大值 | XML 名称 | 关节  | 类型 (单位)    |
| -------- | --------------- | ------ | ------ | -------- | ----- | -------------- |
| 0        | 前端 z 坐标     | -Inf   | Inf    | rootz    | slide | 位置 (m)       |
| 1        | 前端角度        | -Inf   | Inf    | rooty    | hinge | 角度 (rad)     |
| 2        | 后腿大腿角度    | -Inf   | Inf    | bthigh   | hinge | 角度 (rad)     |
| 3        | 后腿小腿角度    | -Inf   | Inf    | bshin    | hinge | 角度 (rad)     |
| 4        | 后脚角度        | -Inf   | Inf    | bfoot    | hinge | 角度 (rad)     |
| 5        | 前腿大腿角度    | -Inf   | Inf    | fthigh   | hinge | 角度 (rad)     |
| 6        | 前腿小腿角度    | -Inf   | Inf    | fshin    | hinge | 角度 (rad)     |
| 7        | 前脚角度        | -Inf   | Inf    | ffoot    | hinge | 角度 (rad)     |
| 8        | 前端 x 坐标速度 | -Inf   | Inf    | rootx    | slide | 速度 (m/s)     |
| 9        | 前端 z 坐标速度 | -Inf   | Inf    | rootz    | slide | 速度 (m/s)     |
| 10       | 前端角速度      | -Inf   | Inf    | rooty    | hinge | 角速度 (rad/s) |
| 11       | 后腿大腿角速度  | -Inf   | Inf    | bthigh   | hinge | 角速度 (rad/s) |
| 12       | 后腿小腿角速度  | -Inf   | Inf    | bshin    | hinge | 角速度 (rad/s) |
| 13       | 后脚角速度      | -Inf   | Inf    | bfoot    | hinge | 角速度 (rad/s) |
| 14       | 前腿大腿角速度  | -Inf   | Inf    | fthigh   | hinge | 角速度 (rad/s) |
| 15       | 前腿小腿角速度  | -Inf   | Inf    | fshin    | hinge | 角速度 (rad/s) |
| 16       | 前脚角速度      | -Inf   | Inf    | ffoot    | hinge | 角速度 (rad/s) |
| excluded | 前端 x 坐标     | -Inf   | Inf    | rootx    | slide | 位置 (m)       |

---

## 奖励函数设计

cheetah 的奖励函数由以下几个部分组成：

```python
# 速度奖励：追踪目标速度
# 姿势奖励：保持稳定的姿势
# 总奖励 = 速度奖励 + 姿势奖励
```

---

## 初始状态

-   重置所有有限关节角度到其允许范围内的随机值，并保持无限幅度关节默认状态。
-   通过多步物理模拟稳定躯干和腿部位置，生成初始观测向量返回。

## Episode 终止条件

-   **无跌倒终止条件**（不会因失稳直接结束）

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=dm-cheetah
```

### 2. 开始训练

```bash
python scripts/train.py task=dm-cheetah/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-cheetah
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-cheetah
```

---

## 预期训练结果

1. 以接近或超过 10.0 m/s 的稳定水平速度奔跑
2. 保持躯体稳定性且步态协调，长距离奔跑不摔倒
3. 奔跑姿态接近真实猎豹，奔跑过程具有伸展感
