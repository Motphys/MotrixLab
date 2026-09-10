# 四足机器人

四足机器人（Quadruped）是 DeepMind Control Suite 中的经典连续控制任务。在 MotrixLab 中，`motrix_envs/src/motrix_envs/basic/quadruped` 目录当前注册了四个可直接训练的任务：平地行走 `dm-quadruped-walk`、平地奔跑 `dm-quadruped-run`、复杂地形逃离 `dm-quadruped-escape`，以及平地推球到目标区 `dm-quadruped-fetch`。

## 任务预览

### Walk

```{video} /_static/videos/dm_quadruped_walk.mp4
:poster: _static/images/poster/dm_quadruped_walk.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

### Run

```{video} /_static/videos/dm_quadruped_run.mp4
:poster: _static/images/poster/dm_quadruped_run.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

### Escape

```{video} /_static/videos/dm_quadruped_escape.mp4
:poster: _static/images/poster/dm_quadruped_escape.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

### Fetch

```{video} /_static/videos/dm_quadruped_fetch.mp4
:poster: _static/images/poster/dm_quadruped_fetch.png
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

## 任务概览

| 环境 ID               | 任务目标                         | 模型文件               | 目标速度 | 观察维度 |
| --------------------- | -------------------------------- | ---------------------- | -------- | -------- |
| `dm-quadruped-walk`   | 在平地上稳定向前行走并保持朝向   | `quadruped_walk.xml`   | 0.5 m/s  | 54       |
| `dm-quadruped-run`    | 在平地上高速奔跑并保持稳定姿态   | `quadruped_walk.xml`   | 5.0 m/s  | 54       |
| `dm-quadruped-escape` | 在起伏地形上尽快向外逃离原点区域 | `quadruped_escape.xml` | 3.0 m/s  | 57       |
| `dm-quadruped-fetch`  | 在平地上将球体推动到目标区域     | `quadruped_fetch.xml`  | 2.0 m/s  | 66       |

## 任务描述

Quadruped 是一个三维四足机器人任务。机器人主体由一个躯干和四条腿组成，每条腿具有偏航、抬升和伸展相关的控制能力。底层 XML 中定义了髋部、膝部和踝部的关节结构，而动作层采用每条腿 3 个执行器的耦合设计：

-   `yaw`：控制腿部偏航
-   `lift`：通过 tendon 耦合控制抬腿动作
-   `extend`：通过 tendon 耦合控制腿部伸展/收缩

`walk` 与 `run` 使用相同的平地模型，区别主要在目标速度；`escape` 使用带高度场的 `quadruped_escape.xml`，要求机器人在崎岖地形中快速远离世界原点，同时保持躯干直立和运动稳定；`fetch` 使用 `quadruped_fetch.xml`，在场景中额外引入自由球体与目标区域，要求机器人先调整到合适站位，再将球向目标方向推进。

---

## 动作空间（Action Space）

| 项目     | 详细信息                         |
| -------- | -------------------------------- |
| **类型** | `Box(low, high, (12,), float32)` |
| **维度** | 12                               |

动作按照四条腿依次排列，每条腿包含 `yaw / lift / extend` 三个执行器：

| 序号 | 动作含义       | 最小值 | 最大值 | 对应执行器           |
| ---: | -------------- | :----: | :----: | -------------------- |
|    0 | 左前腿偏航控制 |  -1.0  |  1.0   | `yaw_front_left`     |
|    1 | 左前腿抬升控制 |  -1.0  |  1.1   | `lift_front_left`    |
|    2 | 左前腿伸展控制 |  -0.8  |  0.8   | `extend_front_left`  |
|    3 | 右前腿偏航控制 |  -1.0  |  1.0   | `yaw_front_right`    |
|    4 | 右前腿抬升控制 |  -1.0  |  1.1   | `lift_front_right`   |
|    5 | 右前腿伸展控制 |  -0.8  |  0.8   | `extend_front_right` |
|    6 | 右后腿偏航控制 |  -1.0  |  1.0   | `yaw_back_right`     |
|    7 | 右后腿抬升控制 |  -1.0  |  1.1   | `lift_back_right`    |
|    8 | 右后腿伸展控制 |  -0.8  |  0.8   | `extend_back_right`  |
|    9 | 左后腿偏航控制 |  -1.0  |  1.0   | `yaw_back_left`      |
|   10 | 左后腿抬升控制 |  -1.0  |  1.1   | `lift_back_left`     |
|   11 | 左后腿伸展控制 |  -0.8  |  0.8   | `extend_back_left`   |

---

## 观察空间

| 环境                  | 详细信息                         |
| --------------------- | -------------------------------- |
| `dm-quadruped-walk`   | `Box(-inf, inf, (54,), float32)` |
| `dm-quadruped-run`    | `Box(-inf, inf, (54,), float32)` |
| `dm-quadruped-escape` | `Box(-inf, inf, (57,), float32)` |
| `dm-quadruped-fetch`  | `Box(-inf, inf, (66,), float32)` |

四种任务共享绝大部分本体观测，`escape` 额外增加了 3 维与原点相关的任务信息，`fetch` 额外增加了球体状态与目标位置信息：

| 部分                   | 内容说明                             | 维度 | `walk/run` | `escape` | `fetch` |
| ---------------------- | ------------------------------------ | ---- | ---------- | -------- | ------- |
| **egocentric dof pos** | 身体广义位置状态                     | 16   | 是         | 是       | 是      |
| **egocentric dof vel** | 身体广义速度状态                     | 16   | 是         | 是       | 是      |
| **actuator ctrl**      | 当前 12 维执行器控制量               | 12   | 是         | 是       | 是      |
| **torso velocity**     | 躯干线速度传感器 `velocimeter`       | 3    | 是         | 是       | 是      |
| **torso upright**      | 躯干朝上程度标量                     | 1    | 是         | 是       | 是      |
| **imu**                | IMU 加速度与角速度                   | 6    | 是         | 是       | 是      |
| **origin**             | 世界原点在本体坐标系中的位置         | 3    | 否         | 是       | 否      |
| **ball state**         | 球相对本体的位置、相对线速度与角速度 | 9    | 否         | 否       | 是      |
| **target**             | 目标区域在本体坐标系中的相对位置     | 3    | 否         | 否       | 是      |

环境还在 XML 中定义了足端力/力矩传感器与质心传感器，但当前实现的默认观测并未直接拼接这些量。

---

## 奖励函数设计

四种任务都以“保持躯干直立”为核心约束。实现中首先根据 `torso_upright` 计算 `upright_reward`，要求机器人主体保持接近直立的姿态。

### Walk / Run

`dm-quadruped-walk` 与 `dm-quadruped-run` 使用相同的奖励结构，只是目标速度不同：

-   `walk` 追踪 `0.5 m/s`
-   `run` 追踪 `5.0 m/s`

总奖励由以下部分组成：

```python
# 速度奖励：沿前向达到目标速度
# 姿态奖励：保持躯干直立
# 辅助奖励：高度、横向稳定性、朝向一致性、动作平滑性
# 惩罚项：后退、竖直速度过大、横滚/俯仰角速度过大、偏离默认姿态
total_reward = upright_reward * move_reward + shaping_terms - penalty_terms
```

其中主要 shaping/penalty 项包括：

-   `height_reward`：鼓励躯干保持在站立高度附近
-   `lateral_reward`：抑制过大的横向速度
-   `heading_reward`：鼓励机器人持续朝 +X 方向前进
-   `smooth_reward`：惩罚相邻时刻动作变化过大
-   `backward_penalty`：抑制向后运动
-   `lin_vel_z_penalty` 与 `ang_vel_xy_penalty`：抑制上下颠簸和躯干横滚/俯仰过大
-   `similar_to_default_penalty`：鼓励关节姿态不要偏离默认站姿太远

### Escape

`dm-quadruped-escape` 在运动控制奖励基础上，增加了“尽快逃离原点区域”的任务项。该任务使用 `quadruped_escape.xml` 中的高度场地形：

```python
# 基础运动控制奖励
# + 远离原点奖励
# + 径向外逃速度奖励
total_reward = locomotion_reward + upright_reward * escape_reward + radial_speed_reward
```

额外任务项包括：

-   `escape_reward`：根据机器人离原点区域的距离给出奖励
-   `radial_speed_reward`：鼓励沿着“远离原点”的方向加速前进

这使得 `escape` 不仅要求机器人跑得快，还要求它在复杂地形上沿正确方向脱离中心区域。

### Fetch

`dm-quadruped-fetch` 使用一套面向“站位 + 推球”的专用 shaping 结构。当前实现中，奖励主要围绕机器人相对球和目标的几何关系展开：

```python
# 站位阶段：鼓励机器人先移动到球后方或侧后方
# 就绪阶段：鼓励机器人面向球、贴近球，并与球-目标连线对齐
# 推球阶段：鼓励球向目标方向滚动，并最终进入目标区域
# 惩罚项：反向移动、把球推离目标方向、腿部与球体过近
total_reward = stage_terms + ready_terms + push_terms - penalty_terms
```

其中主要项包括：

-   `stage_move`：鼓励机器人朝当前阶段目标点移动
-   `stage_reach`：鼓励机器人先到达球后方或侧后方的站位点
-   `behind_align`：鼓励机器人站到球和目标连线的后方
-   `face_ball`：鼓励机体朝向球体
-   `near_ball`：鼓励机器人接近球体
-   `ready` 与 `ready_gate`：综合站位、朝向和距离关系，决定是否进入更积极的推球阶段
-   `fetch`：鼓励球体靠近目标区域
-   `push`：鼓励球沿目标方向滚动
-   `backward`：惩罚朝阶段目标的反向运动
-   `away`：惩罚把球推向远离目标的方向
-   `leg_ball`：惩罚腿部几何体与球体过近，减少“缠球”或挤球现象

此外，`fetch` 还通过 `stability` 门控同时约束躯干直立程度和机体高度，避免机器人通过明显跌倒或趴地的方式获取任务奖励。

---

## 初始状态

-   `walk`、`run` 和 `escape` 都从 XML 中的默认四足站姿开始重置
-   这三个任务的根关节朝向固定为初始朝向，不随机旋转
-   `fetch` 会在平面上随机初始化机器人位置与偏航角，同时随机初始化球体在地面上的位置
-   所有任务的关节速度与球体速度都初始化为 0
-   重置时会自动抬高机体，直到机器人与地面不发生初始穿透/碰撞

## Episode 终止条件

-   最大时长为 20 秒
-   当观测中出现 `NaN` 时，episode 终止
-   `walk`、`run` 与 `escape` 当前实现没有单独设置“跌倒即终止”的条件
-   `fetch` 在机体明显跌倒时会提前终止，条件包括躯干直立程度过低或躯干高度过低
-   当前实现尚未单独设置“球进入目标区域即成功终止”的条件

---

## 使用指南

### 1. 环境预览

```bash
python scripts/view.py env=dm-quadruped-walk
python scripts/view.py env=dm-quadruped-run
python scripts/view.py env=dm-quadruped-escape
python scripts/view.py env=dm-quadruped-fetch
```

### 2. 开始训练

```bash
python scripts/train.py task=dm-quadruped-walk/skrl.ppo
python scripts/train.py task=dm-quadruped-run/skrl.ppo
python scripts/train.py task=dm-quadruped-escape/skrl.ppo
python scripts/train.py task=dm-quadruped-fetch/skrl.ppo
```

### 3. 查看训练进度

```bash
tensorboard --logdir runs/dm-quadruped-walk
tensorboard --logdir runs/dm-quadruped-run
tensorboard --logdir runs/dm-quadruped-escape
tensorboard --logdir runs/dm-quadruped-fetch
```

### 4. 测试训练结果

```bash
python scripts/play.py env=dm-quadruped-walk
python scripts/play.py env=dm-quadruped-run
python scripts/play.py env=dm-quadruped-escape
python scripts/play.py env=dm-quadruped-fetch
```

---

## 预期训练结果

### 行走任务（`dm-quadruped-walk`）

1. 稳定维持接近 `0.5 m/s` 的平地前进速度
2. 身体姿态平稳，横向摆动较小
3. 能够持续保持朝 +X 方向行走

### 奔跑任务（`dm-quadruped-run`）

1. 速度提升到接近或超过 `5.0 m/s`
2. 步幅明显增大，动作具有更强爆发性
3. 高速运动下仍能保持较好的躯干稳定性

### 逃离任务（`dm-quadruped-escape`）

1. 能够快速离开原点附近区域
2. 在高度场地形上保持稳定落脚，不易侧翻
3. 运动方向以向外逃离为主，而不是原地打转

### 推球任务（`dm-quadruped-fetch`）

1. 能够先调整到球和目标连线的合理站位，而不是直接从侧面乱撞球
2. 能够稳定地将球向目标区域推进，而不是把球踢飞或持续推离目标
3. 在推球过程中保持较好的机体稳定性，减少趴地、翻倒和腿部缠球现象
