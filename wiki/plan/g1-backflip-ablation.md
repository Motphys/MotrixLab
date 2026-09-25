# G1 Backflip 消融实验计划

## 摘要

针对 `g1-wbt-backflip` 任务（`G1BackflipWbtEnvCfg`，`motrix_envs/src/motrix_envs/locomotion/wbt/g1.py`）的奖励 / 终止 / 采样消融实验计划。目标是量化任务相对常规 G1 WBT（`g1-wbt-dance` 等使用的 `G1WbtEnvCfg` 基线）新增的每个机制对"完成整周后空翻"的贡献，剔除非承重项、确认承重项，为最终配方瘦身提供依据。

## Baseline

- 分支 / 提交：`feat/g1-wbt-backflip` @ `f147bc2`（assets 提交 `ae9684a` 不影响训练配置）
- Run：`runs/g1-wbt-backflip/motrix/torch/fastsac/26-09-25_12-59-20-141793`
- 设置：FastSAC，2048 envs，60k iterations，RTX 5090 单卡，约 30 分钟（~33 it/s）
- 结果：final rollout mean return ≈ 95.2；play 录像 16 env 稳定站姿
- 曲线：`docs/source/_static/images/g1-backflip/reward-vs-walltime.png`（前段快速上升，后半段进入平台期）

## 统一实验协议

- 每个消融只改一个因子，其余与 baseline 完全一致
- 预算：40k iterations（沿用 v1–v7 历史消融的可比预算）
- 判据（按优先级）：
  1. `metrics/flight_pitch_rotation`（40k 时的峰值旋转角，能否 ≥ 2π 是成败线）
  2. `metrics/flight_max_pelvis_z`（起跳高度，apex 参考 1.19 m）
  3. `metrics/landed_upright` / frame-0-start episodes 的 play 表现（训练期均值被 mid-clip start 稀释，只能作参考）
  4. episode length（过早终止密度）
- 结果记录到本文档 TODO 表格，40k 曲线与 play 视频存到 `runs/` 对应目录并在本页链接 run 路径

## 消融项（按预期承重程度排序）

| # | 因子 | baseline 值 | 消融值 | 依据 / 预期 |
|---|---|---|---|---|
| A1 | `bad_ref_z` phased 终止 | 窗外 0.5 / 窗内 0.25 | 固定 0.5（v5 对照） | 代码注释：站立 pelvis-z 误差 0.44 < 0.5 时站立可存活，策略会放弃起跳；预期差异最大 |
| A2 | `flight_rotation_progress` | 0.3 | 0 | 唯一稠密旋转信号；exp ang-vel kernel 无法从局部尝试 bootstrap 旋转 |
| A3 | `flight_tuck` | 4.0, σ=1.0 | 0（另跑一组 σ=2.5 对照） | v1 历史数据：episode 326 vs 138；收腿决定最后 1/3 旋转 |
| A4 | 混合采样 `start_at_timestep_zero_prob` | 0.2 | 0.0（纯 RSI）/ 1.0（纯 frame-0） | 代码注释：frame-0-only 饿死飞行窗口；uniform_ratio sweep 已否定 adaptive |
| A5 | root 速度 reset 噪声 | 0（归零） | 恢复默认（lin ±0.5, ang ±0.52/0.78） | 空中瞬移需落在参考弹道；预期 mid-air start 质量下降 |
| A6 | `motion_ee_body_pos` | 3.0, σ=0.3 | 0 / 或换 z-only（启用 `motion_ee_body_pos_z`） | 代码注释：z-only 留 0.12 m 落点 xy 误差；全 3D 是落点精度来源 |
| A7 | `action_rate_l2` | -0.01 | -0.5 / -1.0 | 注释：仅 -0.01 在 40k 内解锁旋转（pitch 1.04 vs 0.04–0.25）；复验 |
| A8 | `motion_global_ref_position_error_exp` σ | 1.0 | 0.3 | 注释：0.3 在漂移 1.28 m 时 kernel ≈ 0；验证全局位置梯度必要性 |

### 非消融确认项

- `motion_ee_body_pos_z` 当前 weight=0：代码注释称对起跳 bootstrap 承重（30k 时 max_z 0.43 vs 0.09），但 v6 配方已弃用。仅在 A6 组内作为对照观察，不单独立项
- `limits_dof_pos` / `undesired_contacts`：`G1BackflipRewardsCfg` 未显式声明，实际继承 `RewardsCfg` 默认（-10.0 / -0.1）。需先确认继承行为符合预期，再决定是否纳入消融
- 机器人增益：cfg 注释声称使用官方增益（hip 100 / knee 150 / ankle 40），但 `UnitreeG129Dof()` 写法与常规任务相同，需核实增益改在资产层何处

## TODO

- [ ] 确认 `limits_dof_pos` / `undesired_contacts` 在 backflip 中的实际生效值（继承默认 vs 被覆盖）
- [ ] 核实官方增益的实际来源（资产层 / cfg 层）
- [x] A1：bad_ref_z 固定 0.5 — **结论：phased 窗口不必要，已固化为固定 `BadRefZTerminationCfg(threshold=0.5)`**
  - Run：`runs/g1-wbt-backflip/motrix/torch/fastsac/26-09-25_19-55-21-307302`（40k）
  - 与 baseline 40k 对齐对比：旋转 2.95 vs 2.91 rad、离地高度 0.66 vs 0.66 m 完全持平；`landed_upright` 0.65 vs 0.18 显著更稳
  - v5 注释中"固定 0.5 导致放弃起跳"的机制在当前配方（20% frame-0 混合采样 + tuck/rotation 奖励）下未出现
  - 两者均停在 ~3.1 rad 半圈平台，说明最后 1/3 旋转的限制不在 bad_ref_z → 支持 A3（flight_tuck）为下一优先
  - 对比图：`a1_vs_baseline.png`（run 目录内）；配置改动已合入 `g1.py`
  - 保留项：单 seed 单预算，60k 长程行为未验证
- [ ] A2：flight_rotation_progress=0
- [ ] A3：flight_tuck=0（+ σ=2.5 对照）
- [ ] A4：start_at_timestep_zero_prob 0.0 / 1.0 两组
- [ ] A5：恢复 root 速度 reset 噪声
- [ ] A6：motion_ee_body_pos=0 与 z-only 对照
- [ ] A7：action_rate_l2=-0.5 / -1.0
- [ ] A8：global_ref_position σ=0.3
- [ ] 汇总 40k 结果表格，给出最终精简配方
- [ ] （可选）最终配方 60k 全预算复跑 + play 录像对比 baseline
