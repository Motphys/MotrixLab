# 任务环境设计

`ManagerEnv` 在每个控制周期推进一帧参考 motion，并将机器人状态与该帧目标进行比较。`WbtManagerEnvCfg` 指定 motion、
`tracked_body_names` 和 `reference_body_name`；`scene.objs.robot` 提供机器人模型、默认 key pose、基座 link 与 actuator。
参考关节状态是策略观察中的命令，但动作仍是相对于机器人默认姿态的位置残差，奖励主要根据 body 位姿和速度误差计算。

令 $A$ 为 position actuator 数量，$B$ 为 `tracked_body_names` 中的身体数量。

## 动作空间

动作空间是 $A$ 维连续 `Box`，每个维度对应一个 position actuator。令 $q_{default,i}$ 为默认关节角，
$[q_{min,i},q_{max,i}]$ 为 actuator 直接声明或从目标 joint 继承的 `ctrl_range`，
$\alpha=control\_config.action\_scale$。动作边界为

$$
b_i=\frac{\max\left(\left|q_{min,i}-q_{default,i}\right|,
\left|q_{max,i}-q_{default,i}\right|\right)}{\alpha},
\qquad a_i\in[-b_i,b_i].
$$

环境写入 position actuator 的目标角为

$$
q_{target,i}=q_{default,i}+s_i a_i,
$$

其中 $s_i=\alpha$；当 `action_scales_by_effort_limit_over_p_gain=True` 时，
$s_i=\alpha\,e_i/k_{p,i}$。对于 actuator `force_range` $(f_{min,i},f_{max,i})$，
$e_i=\max(|f_{min,i}|,|f_{max,i}|)$，$k_{p,i}$ 来自同一个 position actuator。开启此模式时，每个 position
actuator 都必须定义 `force_range`。RobotCfg/MJCF 是物理 force limit 的单一来源：同一个运行时 range 同时作为该目标
缩放的输入，并执行实际的 actuator force clamp。

## 观察空间

参考 joint 数组会先按模型 actuator 顺序重排。位置和姿态差在当前 `reference_body_name` 的局部坐标系中表达；姿态使用
旋转矩阵前两列组成的 6D 表示。基座速度在机器人 base link 坐标系中表达。

| 观察                         | Actor | Critic | 含义                                             |
| ---------------------------- | ----: | -----: | ------------------------------------------------ |
| 参考关节位置与速度           |  $2A$ |   $2A$ | 当前 motion 帧的 `joint_pos`、`joint_vel`        |
| 参考身体相对位置             |     — |      3 | motion 参考身体相对当前机器人参考身体的位置      |
| 参考身体相对姿态             |     6 |      6 | motion 参考身体相对当前机器人参考身体的 6D 姿态  |
| 当前 tracked bodies 相对位置 |     — |   $3B$ | 各 tracked body 相对当前机器人参考身体的位置     |
| 当前 tracked bodies 相对姿态 |     — |   $6B$ | 各 tracked body 相对当前机器人参考身体的 6D 姿态 |
| 基座局部线速度               |     — |      3 | base link 坐标系中的线速度                       |
| 基座局部角速度               |     3 |      3 | base link 坐标系中的角速度                       |
| 关节位置残差                 |   $A$ |    $A$ | 当前关节角减默认姿态                             |
| 关节速度                     |   $A$ |    $A$ | 当前关节速度                                     |
| 当前动作                     |   $A$ |    $A$ | 最近一次策略动作                                 |

Actor 维度为 $5A+9$，Critic 维度为 $5A+9B+15$。当前内置任务的实际维度如下：

| 机器人            | $A$ | $B$ | Actor | Critic |
| ----------------- | --: | --: | ----: | -----: |
| Unitree G1 29-DoF |  29 |  14 |   154 |    286 |
| Dex-EVT           |  23 |  14 |   124 |    256 |
| Booster K1        |  22 |  13 |   119 |    242 |

Actor 的参考身体姿态、基座角速度、关节位置和关节速度会分别叠加 `observation_noise` 中配置的均匀噪声；Critic
观察不加噪声。当前实现不额外进行 observation scale 归一化。

## 奖励设计

全局参考项直接比较 motion 与机器人 `reference_body_name` 的世界坐标状态；相对 body 项先用当前机器人参考身体的
水平朝向和位置对齐 motion，再比较所有 tracked bodies。这使相对身体构型的奖励不依赖当前水平平移和偏航。

| 奖励项                                       | 计算方式                                                            | 设计目的                         |
| -------------------------------------------- | ------------------------------------------------------------------- | -------------------------------- |
| `motion_global_ref_position_error_exp`       | 对参考身体世界位置的平方误差应用指数核                              | 跟踪 motion 的整体平移轨迹和高度 |
| `motion_global_ref_orientation_error_exp`    | 对参考身体旋转距离的平方应用指数核                                  | 跟踪整体身体朝向                 |
| `motion_relative_body_position_error_exp`    | 对所有 tracked bodies 的平均相对位置平方误差应用指数核              | 复现四肢与躯干的空间构型         |
| `motion_relative_body_orientation_error_exp` | 对所有 tracked bodies 的平均旋转距离平方应用指数核                  | 复现各身体部位的姿态             |
| `motion_global_body_lin_vel`                 | 对 tracked bodies 的平均世界线速度平方误差应用指数核                | 匹配动作节奏与平移速度           |
| `motion_global_body_ang_vel`                 | 对 tracked bodies 的平均世界角速度平方误差应用指数核                | 匹配身体转动速度                 |
| `action_rate_l2`                             | 计算当前动作与上一动作之差的平方和                                  | 减少控制目标突变                 |
| `limits_dof_pos`                             | 累加超出 soft joint range 的距离，并用 `limits_dof_pos_cap` 截断    | 避免策略持续逼近或越过关节限位   |
| `undesired_contacts`                         | 统计净接触力超过阈值且不在 `allowed_contact_links` 中的机器人 links | 抑制动作不需要的身体接触         |

每个原始项先乘以 `WbtRewardScales` 中的权重，再乘以 `ctrl_dt`；`action_rate_l2`、`limits_dof_pos` 和
`undesired_contacts` 通过负权重成为惩罚。最终加权项写入 `state.reward_terms`。

## 终止条件

| 类型             | 状态         | 条件                                                                               | 含义                                                       |
| ---------------- | ------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 参考高度偏差     | `terminated` | 参考身体世界 Z 误差超过 `bad_ref_pos_threshold`                                    | 机器人整体高度已明显偏离 motion                            |
| 参考朝向偏差     | `terminated` | motion 与机器人参考身体的投影重力 Z 分量差超过 `bad_ref_ori_threshold`             | 机器人倾倒或整体朝向严重偏离                               |
| 关键身体高度偏差 | `terminated` | `bad_motion_body_pos_body_names` 中最大 Z 误差超过 `bad_motion_body_pos_threshold` | 脚或手等关键部位严重失配                                   |
| 关节位置异常     | `terminated` | 关节位置非有限，或超出 hard limit 的最大距离超过阈值                               | 阻止无效或发散的关节状态继续传播                           |
| 关节速度异常     | `terminated` | 关节速度非有限，或最大绝对值超过 `bad_dof_vel_abs`                                 | 在数值速度尖峰扩大前结束回合                               |
| 时间上限         | `truncated`  | 训练回合达到 `max_episode_seconds`，内置配置为 10 s                                | 正常达到训练时限，不表示 bad tracking                      |
| Motion 末帧      | 两者都不是   | `motion_steps` 到达 clip 末尾                                                      | 训练时重采样起始帧并重置 motion 状态；play 时从第 0 帧重播 |

`undesired_contacts` 只产生奖励惩罚，不直接终止回合。各类终止比例与误差均值记录在 `state.metrics`。

## 重置逻辑

重置时，环境从某个 motion 帧复制浮动根位姿与速度、关节位置与速度，重置模型并重新计算运动学状态；当前动作和上一动作
同时清零。

| 随机化项             | 采样方式                                                                                            | 生效时机                   | 作用                                       |
| -------------------- | --------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------ |
| Motion 起始帧        | 默认按 adaptive sampler 的概率分布采样；`start_at_timestep_zero_prob` 可强制一部分环境从第 0 帧开始 | 回合重置和训练 clip 结束时 | 覆盖整段动作，并增加失败集中片段的训练概率 |
| 关节位置             | 在参考关节角上叠加均匀噪声并裁剪到 hard joint range                                                 | 训练重置时                 | 提高对初始姿态误差的恢复能力               |
| 根节点位置与姿态     | 按 `reset_noise.root_pos`、`root_rot` 叠加均匀噪声                                                  | 训练重置时                 | 扩大初始整体位姿分布                       |
| 根节点线速度与角速度 | 按 `reset_noise.root_lin_vel`、`root_ang_vel` 叠加均匀噪声                                          | 训练重置时                 | 提高对初始速度扰动的鲁棒性                 |
| Actor 观察噪声       | 对参考姿态、基座角速度、关节位置和关节速度叠加均匀噪声                                              | 每次构造观察时             | 提高策略对观测误差的鲁棒性                 |

Adaptive sampler 只把 `terminated` 的 motion 帧记录为失败，并通过指数移动平均与均匀探索底线更新采样概率。上述行为属于
初始状态、观察和任务课程随机化；当前 WBT 环境不随机化质量、惯量、摩擦、PD gain 或 actuator delay 等物理参数。
