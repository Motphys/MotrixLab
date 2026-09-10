# Stewart 平台平衡控制

Stewart 平台任务用于训练六自由度并联平台在倾斜控制下保持球体稳定。当前仓库提供两个主要变体：

-   `stewart-static`：静态平台平衡任务
-   `stewart-disturb-xy`：带平面扰动的平衡任务

其中 `stewart` 当前与 `stewart-static` 使用相同配置，可视为同一静态任务的别名。

```{video} /_static/videos/stewart_static.mp4
:poster: _static/images/poster/stewart.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

```{video} /_static/videos/stewart_disturb_xy.mp4
:poster: _static/images/poster/stewart.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

---

## 任务描述

该任务的场景由 Stewart 并联平台、顶部托盘和一个自由滚动的小球组成。智能体并不直接控制 6 条支腿长度，而是输出二维倾斜控制指令：

-   横滚方向目标倾角
-   俯仰方向目标倾角

环境会将这两个控制量映射为平台目标姿态，再进一步计算 6 条滑动支腿的长度控制，使平台实现期望倾斜。小球在重力和平台姿态变化作用下滚动，策略需要把小球重新带回平台中心，并在速度足够小的情况下稳定停住。

对于扰动版本，环境还会在平台底座上叠加低频平移扰动，并可将扰动状态写入观察向量。

---

## 动作空间

所有 Stewart 任务都使用相同动作空间：

| 项目     | 详细信息                        |
| -------- | ------------------------------- |
| **类型** | `Box(-1.0, 1.0, (2,), float32)` |
| **维度** | 2                               |

| 序号 | 动作含义             | 最小值 | 最大值 | 说明                    |
| ---: | -------------------- | :----: | :----: | ----------------------- |
|    0 | 横滚方向倾斜控制输入 |  -1.0  |  1.0   | 映射到目标 `roll` 倾角  |
|    1 | 俯仰方向倾斜控制输入 |  -1.0  |  1.0   | 映射到目标 `pitch` 倾角 |

动作不会直接写入 6 个 actuator，而是先映射为目标平台姿态，再由逆向几何计算得到 6 条支腿的控制长度。

---

## 观察空间

### stewart-static / stewart

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (15,), float32)` |
| **维度** | 15                               |

观察向量由以下部分拼接而成：

| 部分                 | 含义                           | 维度 |
| -------------------- | ------------------------------ | ---- |
| **rel**              | 小球在平台局部坐标系下的位置   | 3    |
| **rel_vel**          | 小球相对速度                   | 3    |
| **platform_tilt**    | 平台当前 roll / pitch 归一化值 | 2    |
| **platform_ang_vel** | 平台局部角速度                 | 3    |
| **target_tilt**      | 当前目标倾角指令               | 2    |
| **action_exec**      | 平滑后的实际执行动作           | 2    |

### stewart-disturb-xy

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(-inf, inf, (25,), float32)` |
| **维度** | 25                               |

在静态版本 15 维观察基础上，额外增加 10 维扰动状态：

-   `disturb_pos`：底座平移扰动
-   `disturb_lin_vel`：扰动线速度
-   `disturb_rot_deg`：扰动姿态角
-   `disturb_ang_vel_deg`：扰动角速度

---

## 奖励函数设计

Stewart 任务当前使用三项主奖励和一项终止惩罚：

```python
center_score = k_center * clip(1 - rel_xy / platform_radius, 0, 1)
zero_vel_closer = k_progress * zero_improve_norm
still_bonus = k_still if success else 0

reward = center_score + zero_vel_closer + still_bonus
reward = fall_penalty if fallen else reward
```

各项含义如下：

-   **中心奖励**：小球越靠近平台中心，奖励越高
-   **低速逼近奖励**：当小球速度足够小且相比历史零速参考点更接近中心时，给予额外进度奖励
-   **稳定奖励**：当小球连续多个控制步都满足“位置接近中心且速度足够小”时，给予稳定停驻奖励
-   **跌落惩罚**：如果小球滚出平台或掉落，直接给予终止惩罚

---

## 初始状态

每次重置时，环境会：

-   将 Stewart 平台设置到一个小幅随机初始倾角
-   先执行一段短暂 settle 过程，使平台支腿和约束达到稳定状态
-   将小球随机放置在平台中心附近的圆形区域内
-   将所有初始线速度、角速度和动作历史清零

其中：

-   初始平台倾角由 `min_init_tilt_deg ~ init_tilt_deg` 采样
-   小球初始半径由 `platform_radius * init_ball_radius_ratio` 控制

---

## Episode 终止条件

### Termination

当以下任一条件满足时，episode 终止：

-   小球滚出平台可接受半径
-   小球高度低于跌落阈值
-   小球连续多个控制步满足稳定条件，视为任务成功

### Truncation

-   达到最大 episode 时长时截断  
    当前默认 `max_episode_seconds = 24.0s`

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=stewart-static
python scripts/view.py env=stewart-disturb-xy
```

### 2. 开始训练

```bash
python scripts/train.py task=stewart-static/skrl.ppo
python scripts/train.py task=stewart-disturb-xy/skrl.ppo
```

### 训练说明

-   当前默认训练参数主要用于提供可运行的基线配置
-   受默认训练时长和超参数限制，训练效果未必达到该任务的最佳水平
-   若希望获得更高成功率或更平滑的稳定效果，可进一步调整训练步数、学习率、rollouts、mini-batches 等 PPO 参数

### 3. 查看训练进度

```bash
tensorboard --logdir runs/stewart-static
tensorboard --logdir runs/stewart-disturb-xy
```

### 4. 测试训练结果

```bash
python scripts/play.py env=stewart-static
python scripts/play.py env=stewart-disturb-xy
```

---

## 预期训练结果

### stewart-static

1. 小球能够被快速带回平台中心附近
2. 小球相对速度逐渐减小并稳定停住
3. 平台不会出现明显的持续大幅振荡

### stewart-disturb-xy

1. 在低频平面扰动下，小球仍能保持在平台中央附近
2. 策略能够对缓慢外扰做出补偿
3. 成功率会低于静态版本，但稳定性会随训练逐步提升
