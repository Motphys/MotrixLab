# Feature: temporal observation encoding (sequence encoder hook) for manipulation tasks

## Summary

`motrix_rl` 的三个 provider（rslrl / skrl / builtin-fastsac）在配置层只支持**扁平向量 MLP**。
对需要时序信息的操作任务（接触搜索、盲插装配、力/阻抗感知），当前唯一做法是把 N 帧历史
在 env 侧展平后整体喂 MLP——首层参数量随帧数线性膨胀，且没有任何时间结构归纳偏置。
建议在模型配置层增加**时序编码器钩子**（1D-Conv / TCN / 帧池化），默认关闭、完全向后兼容。

## Evidence（panda-blind-insertion，全开源资产，可 100% 复现）

任务：`motrix_envs/manipulation/panda_blind_insert`（本分支）——Franka Panda（Apache-2.0）
+ 参数化 NIST 风格方孔插座（34mm 方销 / 30mm 孔，1mm/侧间隙，孔口倒角），30 帧本体感知
历史（含 7 维 `target−qpos` 阻抗残差 + 3 维末端线速度），两阶段课程
（cap=2mm 1500 iter → cap=10mm 2000 iter warm-start），512 局 first-episode-only
确定性评测，cap=10mm、30/30/40 半径分层：

| 分桶 | 展平 MLP（本分支实测） |
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
   （如 32-64 维）再接现有 MLP 头。obs 侧需要区分 `[T, D]` 帧布局与扁平布局（env 的
   `NpObs` 增加 frames 语义或约定 reshape）。
3. **优先级**：rslrl（本实验所用）> skrl > fastsac（其 buffer 需要相应支持 obs 分组切片）。
4. **不做**：不引入 RNN/LSTM（部署导出复杂），conv1d/TCN 保持前馈、ONNX 可导出。

## Reproduction

- 任务：`motrix_envs/manipulation/panda_blind_insert`（本分支，全部开源资产）
- 训练：`mxlab train task=panda-insert/rslrl.ppo num_envs=256`（两阶段课程）
- 评测协议：first-episode-only + 初始半径 30/30/40 分桶（防止 auto-reset 幸存者偏差）
- 验收数字：Stage-1 cap=2mm 0.932；Stage-2 cap=10mm 0.904（98.7/96.1/80.6）

相关诊断方法（本任务开发中验证有效的）：oracle 雅可比直压（分离"几何/动力可行性"与
"策略可学性"）、零动作奖励审计（暴露被动收入泄漏）、逐帧穿透审计（暴露判据误杀）。
