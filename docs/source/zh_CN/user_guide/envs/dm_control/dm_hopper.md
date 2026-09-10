# 单脚跳跃机器人

单脚跳跃机器人（Hopper）是 DeepMind Control Suite 中的经典连续控制任务。其目标是训练一个模拟的单腿跳跃机器人，通过控制其关节力矩，实现站立平衡或向前跳跃。

```{video} /_static/videos/dm_hopper.mp4
:poster: _static/images/poster/dm_hopper.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务描述

Hopper 是一个二维单腿跳跃机器人任务。由 5 段身体结构组成（torso 躯干、pelvis 骨盆、thigh 大腿、calf 小腿、foot 脚部），拥有 3 个受控关节（thigh_joint 大腿关节、leg_joint 小腿关节、foot_joint 脚部关节），智能体通过向这些关节施加扭矩作为动作，让机器人完成站立平衡或向前跳跃。

---

## 动作空间（Action Space）

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (3,), float32)` |
| **维度** | 3                               |

关节对应如下：

| 序号 | 动作含义（施加在关节的力矩） | 最小值 | 最大值 | 对应 XML 中名称 |
| ---: | ---------------------------- | :----: | :----: | :-------------: |
|    0 | 大腿转子驱动扭矩             |   -1   |   1    |  `thigh_joint`  |
|    1 | 小腿转子驱动扭矩             |   -1   |   1    |   `leg_joint`   |
|    2 | 脚部转子驱动扭矩             |   -1   |   1    |  `foot_joint`   |

---

## 观察空间

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (13,), float64)` |
| **维度** | 13                               |

Hopper 环境的观测空间由以下部分组成（按顺序）：

| 部分                | 内容说明             | 维度 | 备注                            |
| ------------------- | -------------------- | ---- | ------------------------------- |
| **qpos**            | 关节角度与躯干高度   | 5    | 不包括 torso x 坐标（默认隐藏） |
| **qvel**            | 关节角速度与躯干速度 | 6    | 所有关节及躯干的速度            |
| **contact sensors** | 脚尖与脚跟触地传感器 | 2    | 使用 `log1p` 正规化处理         |

| 序号 | 观察量          | 最小值 | 最大值 | XML 名称    | 关节   | 类型 (单位)    |
| ---- | --------------- | ------ | ------ | ----------- | ------ | -------------- |
| 0    | 躯干 z 坐标     | -Inf   | Inf    | rootz       | slide  | 位置 (m)       |
| 1    | 躯干角度        | -Inf   | Inf    | rooty       | hinge  | 角度 (rad)     |
| 2    | 大腿关节角度    | -Inf   | Inf    | thigh_joint | hinge  | 角度 (rad)     |
| 3    | 小腿关节角度    | -Inf   | Inf    | leg_joint   | hinge  | 角度 (rad)     |
| 4    | 脚部关节角度    | -Inf   | Inf    | foot_joint  | hinge  | 角度 (rad)     |
| 5    | 躯干 x 坐标速度 | -Inf   | Inf    | rootx       | slide  | 速度 (m/s)     |
| 6    | 躯干 z 坐标速度 | -Inf   | Inf    | rootz       | slide  | 速度 (m/s)     |
| 7    | 躯干角速度      | -Inf   | Inf    | rooty       | hinge  | 角速度 (rad/s) |
| 8    | 大腿关节角速度  | -Inf   | Inf    | thigh_joint | hinge  | 角速度 (rad/s) |
| 9    | 小腿关节角速度  | -Inf   | Inf    | leg_joint   | hinge  | 角速度 (rad/s) |
| 10   | 脚部关节角速度  | -Inf   | Inf    | foot_joint  | hinge  | 角速度 (rad/s) |
| 11   | 前脚触地传感器  | -Inf   | Inf    | touch_toe   | sensor | 压力 (无量纲)  |
| 12   | 后脚触地传感器  | -Inf   | Inf    | touch_heel  | sensor | 压力 (无量纲)  |

---

## 奖励函数设计

hopper 的奖励函数由以下几个部分组成：

### stand 任务（站立）

```python
# 站立奖励：维持目标站立高度稳定性
# 高度奖励：保持躯干在目标高度附近
# 总奖励 = 站立奖励
```

### hop 任务（跳跃）

```python
# 站立奖励：维持目标站立高度稳定性
# 跳跃奖励：达到目标前进速度
# 腿部运动奖励：适度的腿部摆动（抑制过度运动）
# 膝盖伸展奖励：适度的膝盖伸展动作
# 足部接触奖励：保持适当的足部接触力
# 总奖励 = 站立奖励 + 跳跃奖励 + 腿部运动奖励 + 膝盖伸展奖励 + 接触奖励
```

---

## 初始状态

-   重置部分关节角度到其允许范围内的随机值
-   保持有限幅度关节默认状态

## Episode 终止条件

-   机器人的状态观测值出现异常数值（NaN）

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=dm-hopper-stand
python scripts/view.py env=dm-hopper-hop
```

### 2. 开始训练

```bash
python scripts/train.py task=dm-hopper-stand/skrl.ppo
python scripts/train.py task=dm-hopper-hop/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-hopper-hop
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-hopper-stand
python scripts/play.py env=dm-hopper-hop
```

---

## 预期训练结果

### 站立任务 (dm-hopper-stand)

1. 保持机器人稳定站立
2. 躯干高度保持在 0.6 米附近

### 跳跃任务 (dm-hopper-hop)

1. 机器人实现稳定的向前跳跃
2. 跳跃运动达到 2.0 m/s 的目标速度
3. 步态协调，无明显摔倒
