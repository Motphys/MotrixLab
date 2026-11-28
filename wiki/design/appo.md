# APPO 异步 actor–learner

## 摘要

Torch APPO 的实现保留在独立 `motrix_rl_appo` 包中，由 `motrix_rl_builtin` 注册为公开算法 `motrix.appo`；builtin 同时拥有该算法的公开 ConfigStore 名称和默认配置。APPO 实现包不是 public framework provider，也不提供自己的 entry point 或公开算法名。Go2 flat-walk 的现有 `rslrl.ppo` 配方是效果基准。算法以 PPO、IMPALA/V-trace 和 IMPACT 为依据，RLlib 与 UniLab 仅作为交叉参考，不以复刻某个项目为正确性标准。

## 算法

参考 [IMPACT](https://arxiv.org/abs/1912.00167)、[IMPALA](https://arxiv.org/abs/1802.01561) 和 RLlib APPO（参考 checkout `976013db03b0`）。令 μ 为采样策略、π̄ 为冻结 target actor、π 为当前 actor。

- 价值修正：ρ=min(ρ_max, π̄/μ)，c=min(c_max, π̄/μ)。δ=ρ(r+discount·V_next−V)，反向递推 v=V+δ+discount·c·(v_next−V_next)。
- PG advantage：A=ρ_pg(r+discount·v_next−V)，ρ_pg=min(pg_rho_max,π̄/μ)。实现可按整批标准化该 detached advantage，不能在计算 V-trace 前打乱时间轴。
- IMPACT surrogate ratio：R=min(μ/π̄,worker_target_clip)·π/μ。使用 PPO 的 min(R·A,clip(R,1−ε,1+ε)·A)。两次 importance weighting 是参考 estimator 的组成部分，不能仅因出现两次而删掉其中之一。
- target actor 的参数与 normalizer 作为整体快照按更新频率同步；第一版硬同步，保留明确频率参数，不添加未需求驱动的 soft-update 变体。
- 每次 learner update 在固定 rollout batch 上计算 detached targets，再做多轮 shuffled minibatch 优化。normalizer 每批新采样仅更新一次，放在该批优化结束的发布边界，随后同步到期的完整 target 快照，避免首次优化前改变 policy 的输入坐标；loss 缓存使用 no_grad，不向 autograd 表达式泄漏 inference tensor。

## 数据和所有权

单 collector、单 learner。collector 进程创建环境，启动握手报告 policy/value/action 维度；learner 创建网络、weight channel 和 transition ring 并回传。`transition_transport=auto` 在learner与collector inference为同一CUDA设备时使用已有CUDA-IPC ring，其余使用host shared-memory ring；`host`/`cuda_ipc`可用于显式对照，不可用的显式IPC配置报错。权重通道独立通过`weight_transport=auto|host|cuda_ipc`选择：auto同GPU用GpuIpcWeightSender，其余host，显式不可用IPC报错，不依赖transition transport选择。参数slots由learner持有到collector退出；actor/critic参数IPC时D2D，host时经host；normalizer mean/std/var与int64 count始终走既有host共享快照，与参数同一seqlock版本发布，不暴露独立统计A/B开关。GPU统计方案的收益/维护取舍及后续验证由[issue #197](https://github.com/Motphys/MotrixLab/issues/197)追踪，当前不增加独立GPU bundle类型。不把两项传输优化混为一谈。复用 `motrix_rl.worker` 与 transport，不依赖 rsl-rl 或 Ray。

CUDA-IPC ring由父learner拥有fused float32 device slots和host cursors，collector只接收schema/slots/cursors并构造endpoint，CUDA events不跨进程pickle。producer支持CPU pinned staging的一次fused H2D，以及同GPU常驻chunk的直接D2D发布。CPU仿真观测/reward/done仍须H2D，动作仍须D2H给CPU环境，但GPU carrier避免为了保存obs/action/logp而D2H并再次H2D。push完成后poll event publication，不把未发布slot留到下一次chunk结束；这同时保证collector复用自有chunk前所有ring copy已完成。consumer按schema还原展平多维字段与bool dtype，自有D2D copy在commit event之前入流，producer仅在event完成后复用slot。owner保持device slots存活直至子进程退出。

预分配collector carrier随已解析transition transport选择：IPC使用同GPU存储，host使用CPU，不暴露独立chunk设备A/B开关。collect返回的chunk是collector自有可复用buffer的借用视图，仅到下一次collect前有效；caller必须完成publication后再复用，不承诺每次返回独立快照。

一个 slot 是固定 T 步、N 环境的 chunk。字段为起点 policy/value observation、action、原始 action 的 behavior log-prob、reward、done，以及独立 endpoint value observation。collector 必须在环境借用输出有效期内复制起点数据。action/log-prob/observation 保持同一时间 t。

每步 truncation（非 termination）优先使用稀疏 `state.final_obs` 中的 pre-reset critic observation，在 collector 用同一个快照 critic 计算 bootstrap 并折叠进 reward；若本步final_obs为None，则使用已保存的动作前critic输入计算V(s_t)近似，并在首次发生时告警一次。仅在能力完全缺失时使用该近似，已提供但缺少timeout行的final_obs仍报错。terminated即便同时truncated也不bootstrap；done切断递推。原始 reward 用于回合统计，不含 bootstrap 修正。普通 chunk 尾用独立 endpoint observation 由 learner 当前 critic 计算 V(s_T)。不能拿最后起点 observation 冒充 endpoint，也不能用 reset observation 代替 timeout final observation。

每次更新消费一个 chunk，固定训练预算 `max_iterations*T*N`，不由 queue 深度改变 batch/预算。learner 先 clone/copy 所有 ring view 到自有存储，再 commit。ring 满时 collector 有界背压；每 chunk 边界拉取一次完整 actor/critic/normalizer 快照。记录权重版本滞后、queue 深度、采样步数与耗时。

## 配置和集成

`motrix_rl_appo` 拥有带默认值的 dataclass 配置、provider 实现和训练实现；`motrix_rl_builtin` 拥有公开 provider 注册以及 ConfigStore `algo_base/motrix.appo` 节点。实现包不依赖 builtin，builtin 直接依赖实现包并统一通过自身的 `motrix_rl.frameworks` entry point 注册，依赖方向与 SuperMimic 一致。网络自包含：Gaussian actor（learnable log std，确保 std 正值）、value MLP、正确的 pooled running mean/variance。Go2 使用与 RSLRL 相同 512/256/128 ELU 网络、初始 std 0.5、1024 环境、24 步、2000 次更新作为起点，保持环境与 reward 配置不变。

默认 builtin 安装闭包包含 APPO，不使用单独的 APPO extra。cartpole/Go2 配方分别使用 `task=cartpole/motrix.appo` 和 `task=go2-walk-flat/motrix.appo`；公开算法的 framework 为 `motrix`、method 为 `appo`，训练 backend 为 Torch。play 使用 deterministic mean。ONNX、多 collector、多 GPU 不在当前范围。learner 可选编译/图执行的契约见下节。训练 render 能力不做静默忽略：若未实现则显式拒绝。

checkpoint 周期保存与最终保存均登记 manifest。保存 actor/critic/target、optimizer、normalizer、update/env-step counter、随机状态。恢复为 warm resume：learner 状态恢复，collector/environment 重新启动；不宣称轨迹级精确续接。

## Learner 执行优化

`algorithm.execution` 选择 `eager`、`compile` 或 `cuda_graph`；通用 provider 与 CartPole 默认 eager/unfused，保持 CPU 可用。Go2 flat、Go2 rough、ANYmal-C flat 三个已完成效果验证的任务配方默认 `cuda_graph` + `fused_optimizer=true`，无需额外CLI覆写即可使用实测配置。CPU运行这些四足配方需显式覆写为eager；不做隐式设备回退。显式选择不可用模式在边界报错，不静默回退。算法目标、adaptive KL 调整频率、每 epoch 随机 minibatch、梯度裁剪、normalizer/target 发布边界与 checkpoint 不因模式改变。

loss 使用解析 Gaussian log-probability、entropy、KL 的 tensor-only 核心，避免 distribution 分派与 Python 数值分支造成 graph break。compile 使用 fullgraph tensor loss；fused Adam 合并 optimizer kernel。metrics 在 device 累加，完整 update 结束后统一回传；adaptive LR 是 device tensor，通过 torch.where 实现同样逐 minibatch 调度，避免每次 minibatch 的 CPU 同步。

CUDA Graph 捕获固定形状单 minibatch 的 forward/backward、梯度裁剪与 fused Adam，输入来自私有静态 buffer；每次 replay 前更新其内容，保持同一存储地址。graph 路径只支持 CUDA 与均分 minibatch，并要求 fused optimizer。warmup/capture 不能额外训练或污染随机流：备份并恢复参数、optimizer 状态与 RNG，捕获后 in-place 恢复所引用的存储。checkpoint load 会替换 optimizer 张量地址，因此销毁旧捕获，恢复后再建立图，并重新绑定当前模式的 LR/capturable 设置。

编译融合会改变浮点运算次序，特别是严格 `KL > 0` 的 adaptive 调度边界可能因零附近舍入产生不同分支；不引入未要求的新epsilon阈值，也不宣称跨执行模式逐位一致。固定LR模式用于严格数值parity，adaptive模式另外验证非边界分支和真实训练。

实际性能结果与端到端瓶颈见 [APPO性能评测](../research/appo-evaluation.md#4-learner执行优化)。性能验收必须分别记录初次 compile/capture 成本与同步计时的 steady-state 完整 update；用相同 Go2 batch/network/epoch 规格比较，不能只测单个 kernel 或把异步提交耗时当GPU执行耗时。CPU correctness、CUDA eager/compile/graph多步数值parity、随机排列更新与checkpoint跨模式恢复分别覆盖；实际GPU验证不以skip替代。

## 验证和比较

低成本测试覆盖 off-policy 手算、terminal/timeout/endpoint、pooled variance、action/logp 对齐、ring wrap-around 所有权、target snapshot、checkpoint round-trip、Hydra/provider 安装与发现。真实进程/仿真测试归 integration。

先运行现有 Go2 RSLRL 配方建立当前 checkout 的基准，历史 runs 只作参考。阶段性短跑用于成本估算和错误排查；最终对照相同环境步数、seed、环境与 reward 配置，比较 deterministic evaluation 的原始回报、回合长度/失败率和速度跟踪指标，同时报告 wall-clock 与训练曲线。单 seed 结果只报告单 seed，不宣称统计显著性；若 APPO 明显落后，应诊断和调参并如实记录，不能以有限 loss 或冒烟通过声称效果相当。

实施及运行验收结果见 [APPO训练效果与性能评测](../research/appo-evaluation.md)。
