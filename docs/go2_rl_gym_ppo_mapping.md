# Go2 RL Gym PPO 到 MotrixLab 的对齐记录

该表记录第一阶段迁移的接口边界。训练仍由 MotrixLab 的 backend-neutral 环境和 RSLRL PPO provider 负责。

| 项目 | go2_rl_gym legacy PPO | MotrixLab `go2-walk-flat` | 第一阶段处理 |
| --- | --- | --- | --- |
| 动作 | 12 维关节位置残差，`scale=0.25` | 12 维 asymmetric residual action，`scale=0.25` | 保持关节顺序和尺度一致 |
| 策略观测 | 45 维：角速度、投影重力、关节位置差、关节速度、上一动作、命令 | 当前为同类本体量加命令和四脚相位 | 记录维度与顺序；不直接加载 legacy checkpoint |
| critic 观测 | privileged state 可包含线速度和地形量 | policy 观测 + 局部线速度 | 继续使用 value observation 分组 |
| 控制 | P 控制，`kp=20`、`kd=0.5`、动作降采样 | position actuator、统一 `ctrl_dt` | 训练配置保持一致，部署 profile 读取环境增益 |
| 命令 | 随机平面速度，legacy 支持单轴和边界课程 | 独立随机平面速度 | 增加单轴与边界覆盖概率，改善直行训练分布 |
| 随机化 | 摩擦、质量、COM、PD、噪声、动作延迟 | 已有对应 reset randomization | 第一阶段保留现有范围，后续再做课程 |
| 奖励 | 速度跟踪、姿态、垂向速度、角速度、动作变化、接触与关节约束 | 同类奖励，并加入平衡姿态、站立、足端滑移、关节加速度和动作平滑 | 先用固定命令指标验证，不用总回报替代跟踪指标 |

固定命令评估输出到 `walk_analysis/velocity_metrics.json`，包括命令、平均速度、稳态误差、RMSE、P95、正交漂移、跌倒率、姿态 RMS 和高度波动。
