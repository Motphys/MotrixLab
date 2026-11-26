# Go2 FlatWalk v2 快速训练设计

## 目标

`go2-walk-flat-v2` 是 Go2 平地速度跟踪任务的 v2 正式版本。它在 v1 冻结配方的基础上降低训练规模，把完整训练控制在约 11 分钟预算内，同时不使用提前终止，也不牺牲真机可部署性。

v2 的验收不是“仿真曲线看起来正常”，而是同时满足：

- 训练完整执行 2000 iterations；
- TensorBoard 的 reward、episode length、value loss 与 policy noise 进入平台期；
- 16 个固定命令场景、220 个速度/摩擦组合和 60 秒零命令站立审计通过；
- 导出的 deployment artifact 能在 MuJoCo/MotrixSim 中运行，并保持与已实机验证参考策略相同的部署契约；
- 真机效果稳定。真机效果不达标时，即使训练时长缩短也不能视为成功。

## v1 → v2 差异

| 项目 | v1 冻结配方 | v2 配方 |
| --- | --- | --- |
| 并行环境数 | 4096 | 1024 |
| 训练迭代数 | 3000 | 2000 |
| minibatch 数 | 16 | 4 |
| 训练框架 | RSLRL PPO | RSLRL PPO |
| seed | 1 | 1 |
| `tracking_ang_vel` reward scale | 1.3 | 1.6 |
| hard cases 数量 | 24 | 28 |
| pure-yaw hard cases | 无 | `(0, 0, ±0.5)`、`(0, 0, ±0.8)` |
| hard-case 课程 | 600 iter 到 0.05，1000 iter 到 0.15，1500 iter 到 0.20 | 450 iter 到 0.08，700 iter 到 0.20，之后保持 |
| command 重采样间隔 | `(5, 10) s` | `(5, 10) s` |
| 机器人、观测、动作、其余 reward 与随机化 | v1 配方 | 保持不变 |

v2 只做训练配方收敛，不改变 actor/critic 网络、观测构造、动作尺度、episode 生命周期或部署输入输出契约。唯一 reward 变化是把 yaw 速度跟踪权重从 1.3 提高到 1.6，用于在较少采样量下保留转向能力。新增的 4 个 pure-yaw hard cases 覆盖无平移速度下的原地转向；课程提前让困难命令在 2000 iterations 内获得充分训练。

## 2026-11-26 验证记录

最终配置使用 `go2-walk-flat-v2/rslrl.ppo` 与 seed 1，执行 2000/2000 iterations、1024 个并行环境、4 个 minibatches，共 49.152M steps；训练墙钟 7 分 06 秒，没有提前终止。末段 mean reward 为 102.00、episode length 为 984.82、policy noise std 为 0.0557。

与已实机验证参考策略逐点对比，所有 2000 个 `Loss/value`、`Loss/surrogate`、`Loss/entropy`、`Loss/learning_rate` 与 `Policy/mean_noise_std` 标量完全相同；所有 1998 个 `Train/mean_reward` 与 `Train/mean_episode_length` 标量完全相同。it=300/500/1000/1999 checkpoint 与 `checkpoints/latest.pt` 的 sha256 均一致；差异仅存在于 FPS、collection/learning time 与 wall-clock 标量，不影响策略。

正式 artifact 为 `artifacts/go2-walk-flat-v2.deploy`，`mxlab deploy inspect` 通过，32 个 ONNX 抽样最大误差 1.43e-6。ONNX 文件 sha256 与参考策略不同，但差异只来自模型 metadata 中的环境名和 checkpoint 路径；清除 metadata 后两份 ONNX 的确定性序列化 sha256 相同，100000 组固定随机输入下推理输出逐位相同。该 artifact 已完成真机验收，前伸、抬腿、行走、转向与站立表现符合预期。

## 分支结构

分支从冻结 v1 提交创建，而不是携带实验分支的未提交状态。Go2 只保留一个任务和一条训练配置：

```text
go2-walk-flat-v2/rslrl.ppo
```

v2 的仿真与真机部署任务分别是：

```text
go2-walk-flat-v2/sim
go2-walk-flat-v2/hardware
```

Artifact 必须由 v2 分支重新训练并重新导出；不从旧任务或实验任务复制。默认输出路径为 `artifacts/go2-walk-flat-v2.deploy`。

## 验收

1. 配置验收：Hydra 解析结果必须为 `num_envs=1024`、`max_iterations=2000`、`num_mini_batches=4`、`seed=1`。
2. 收敛验收：读取 TensorBoard `Train/mean_reward`、`Train/mean_episode_length`、`Loss/value`、`Policy/mean_noise_std`；末段分窗口均值变化很小，且末 300 iterations 的 reward 线性拟合斜率接近 0。
3. 审计验收：16/16 个固定场景成功；220/220 个速度/摩擦组合成功，跟踪误差与姿态指标不劣于参考阈值；60 秒零命令站立保持稳定。
4. 部署验收：`export-deploy` 生成的 artifact 通过 `mxlab deploy inspect`；MuJoCo/MotrixSim 冒烟运行成功。
5. 真机验收：按部署安全流程在真机上确认行走、横移、转向和站立稳定性。真机不满足预期时，v2 不能被认定为完成。
