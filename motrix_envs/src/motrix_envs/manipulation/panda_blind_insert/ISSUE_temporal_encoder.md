# Feature: temporal observation encoding (sequence encoder hook) for manipulation tasks

## Summary

`motrix_rl` 的三个 provider（rslrl / skrl / builtin-fastsac）在配置层只支持**扁平向量 MLP**。
对需要时序信息的操作任务（接触搜索、盲插装配、力/阻抗感知），当前唯一做法是把 N 帧历史
在 env 侧展平后整体喂 MLP——首层参数量随帧数线性膨胀，且没有任何时间结构归纳偏置。
建议在模型配置层增加**时序编码器钩子**（1D-Conv / TCN / 帧池化），默认关闭、完全向后兼容。

## Evidence（panda-blind-insertion，升级前的基线）

任务：`motrix_envs/manipulation/panda_blind_insert`（本分支）——Franka Panda（Apache-2.0）

- 参数化 NIST 风格方孔插座（34 mm 方销、36 mm 孔，每侧 1 mm 间隙），30 帧本体感知
  历史（含 7 维 `target−qpos` 阻抗残差 + 3 维末端线速度），两阶段课程
  （cap=2mm 1500 iter → cap=10mm 2000 iter warm-start），512 局 first-episode-only
  确定性评测，cap=10mm、30/30/40 半径分层：

| 分桶 | 展平 MLP（升级前实测） |
|---|---|
| 0-3 mm | 98.7% |
| 3-6 mm | 96.1% |
| 6-10 mm | **80.6%** |
| **总体** | **0.904** |

**关键信号：远距桶（6-10mm）的失败 100% 是 timeout（800 步搜索未果），penetration 0.000、
drop 0**——在焊接销变体（夹持失败结构性归零，见下）下，远距瓶颈被纯化为**纯搜索效率**问题：
策略的单帧均值决策能覆盖全分布 90% 的局，但剩余 ~20% 的远距局需要在长程面板搜索中利用
时序局部性（速度估计 / 接触事件时间戳 / 扫过区域记忆）——这正是展平 MLP 缺失、
时序卷积天然具备的归纳偏置。

内部对照（非公开资产，仅作参考刻度）：同语义任务在内部 24 段圆孔 + 摩擦夹持变体上，
29 维观测（无阻抗残差）展平 MLP 仅 24.0%（6-10mm 桶 0.0%），TCN 时序卷积 + 三阶段课程
为 67.8%（6-10mm 37.5%）。**物理边界差异说明**：开源版使用焊接销（无 drop 通道）、
倒角方孔（捕获漏斗更宽）、平面指垫接触，桶结构与内部版不可直接等价比较；
两版的共同结论一致——观测语义（阻抗残差）决定下限，时序编码决定远距上限。

## Proposal

1. **配置面**：`ActorCfg`/`CriticCfg` 增加 `encoder: mlp | conv1d | tcn`（默认 `mlp`，现行为不变），
   附 `encoder_channels` / `encoder_kernel` / `encoder_stride` 等参数。
2. **接入点**：provider 在 `_make_model`/模型构造处按 encoder 分派；encoder 输出低维 latent
   （如 32-64 维）再接现有 MLP 头。模型可以用 `ObservationSpec` 中的帧空间和历史长度将
   扁平观测还原为 `[T, D]`。
3. **优先级**：rslrl（本实验所用）> skrl > fastsac（其 buffer 需要相应支持 obs 分组切片）。
4. **不做**：不引入 RNN/LSTM（部署导出复杂），conv1d/TCN 保持前馈、ONNX 可导出。

## Reproduction

- 任务：`motrix_envs/manipulation/panda_blind_insert`（本分支，全部开源资产）
- 第一阶段将环境的 `reset.offset_cap` 设为 `0.002`，运行 `mxlab train task=panda-blind-insertion/rslrl.ppo num_envs=256`。
- 第二阶段使用默认 `reset.offset_cap=0.010`，训练命令如下。`resume` 指向第一阶段的最终 checkpoint，
  learner 恢复 actor、critic 和观测归一化统计，并重新初始化优化器和探索噪声。

  ```bash
  mxlab train task=panda-blind-insertion/rslrl.ppo num_envs=256 algo.max_iterations=2000 \
    resume=/path/to/stage1/checkpoints/latest.pt algo.warm_start=true algo.actor.init_noise_std=0.5
  ```
- 评测协议：first-episode-only + 初始半径 30/30/40 分桶（防止 auto-reset 幸存者偏差）
- 旧版仿真与 reset、穿透判定下的基线为 Stage-1 cap=2 mm、0.932，Stage-2 cap=10 mm、0.904（98.7/96.1/80.6）。升级到 MotrixSim 1.0 后需要重新训练和测量。

相关诊断方法（本任务开发中验证有效的）：oracle 雅可比直压（分离"几何/动力可行性"与
"策略可学性"）、零动作奖励审计（暴露被动收入泄漏）、逐帧穿透审计（暴露判据误杀）。
