# G1 Backflip 飞行指标口径设计

## 摘要

本文定义 `g1-wbt-backflip` 任务飞行技能指标（`flight_pitch_rotation` / `flight_max_pelvis_z` /
`landed_upright` / `clip_ended`）在 mixed/RSI 采样下的正确口径：指出当前实现把不可达判定点的
episode 计入均值造成的系统性低估，并给出条件指标 + 覆盖率、二值成功率、物理落地事件、
checkpoint 选择解耦的修正设计。修正尚未实施；实施前训练期均值只可用于趋势诊断，
策略真实水平以 play 口径（帧 0 起跑、零噪声）评估为准。

## 问题定义

指标由 `WbtMotionCommand`（`motrix_envs/locomotion/wbt/mdp/command.py`）按 episode 累积：
飞行窗 `[flight_start, flight_end]` 由 clip root-z 剖面推导（本 clip 为帧 108–137），
`land_check_step = flight_end + 0.5s`（帧 162），`hold_at_clip_end` 下 episode 钉住末帧到超时。
采样使用 mixed（10% 帧 0 + 90% 全 clip 均匀，含空中帧 RSI 传送）。

当前口径的偏差：

1. **不可达判定点稀释均值**。起点晚于判定帧的 episode 恒计 False/0：起点 >162 者占
   均匀采样约 17%（`landed_upright`），起点 >137 者约 31%（`flight_pitch_rotation`）；
   早死 episode 同理。指标是"全部 episode 的均值"而非"有资格被评判 episode 的条件值"，
   系统性低估技能。
2. **固定帧判定与物理落地错位**。落地判定取参考时间轴固定帧的 pelvis pitch，但策略真实
   落地时刻随起跳高度偏移；`land_check_step` 假设策略以参考弹道穿过飞行窗，恰是技能未
   成型时不成立的假设。
3. **连续均值不可解读**。`flight_pitch_rotation` 均值无法区分"人人小角度"与"少数完整翻转"，
   两种策略技能水平天差地别；技能形成的本质是成功率曲线。
4. **`clip_ended` 失去信息量**。尾部帧起始的 episode 几步即到 clip 末端，指标长期停在
   ~0.7，不再表示"完整跑完一次 clip"。
5. **训练均值与 checkpoint 选择耦合**。训练期指标受采样分布与非平稳策略影响，直接用于
   best checkpoint 选择会偏向"起始帧分布有利"的时段而非真实技能高点。

实证信号：同一配方可观察到训练均值与 play GIF 明显背离（如训练 `landed_upright` 均值
回撤而帧 0 起跑的 play 表现接近），即为上述稀释机理的表征。

## 修正设计

按优先级，均不进入 reward、不影响训练行为：

1. **条件指标 + 覆盖率成对上报**。`reset_env` 记录 per-env `episode_start_step`；每个技能
   指标配对输出 `*_cond`（仅在到达判定点的 episode 中统计）与 `*_reached`（判定点到达率）。
   条件值与覆盖率必须一起读：前者被低覆盖率欺骗，后者被混合均值稀释。
2. **二值成功率指标**。`flip_success = |flight_pitch_acc| > 0.85 * 2π`（方向正确且圈数
   足够）、`apex_reached = flight_max_z > 0.85 * ref_apex`，按条件口径上报。成功率对
   SAC 非平稳分布远比均值稳健，且直接对齐任务定义。
3. **落地判定改物理事件**。飞行窗结束后首次 `pelvis_z` 回落至站立高度阈值以下（或踝部
   接触力超阈，可复用 `undesired_contact_forces` query）触发落地事件；事件后 0.5s 内判定
   pelvis pitch upright、非期望 body 无触地、存活满 1s 三项，替代固定帧判定。
4. **`clip_ended` 条件化**。改为 `started_before_flight & reached_clip_end`。
5. **checkpoint 选择与训练均值解耦**。best checkpoint 由 play 口径批量评估
   （帧 0 起跑、零噪声、多 env）的 `flip_success_cond` / `landed_upright_cond` 决定。

## 实现评估

1/2/4 为 `command.py` 增加字段、`reset_env`/`update` kernel 若干行与 host 侧 `reset()`
的条件统计；3 需要跨 ctrl step 的落地状态机，工作量略大，可后置。
