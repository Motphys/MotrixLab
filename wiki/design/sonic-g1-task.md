# SONIC G1 任务迁移设计

## 目标与边界

本设计将 SONIC 的 G1 动作跟踪能力接入 MotrixLab 当前九包 workspace，同时保持现有
Manager、simulator registry、Hydra Task 和 FastSAC 公共契约。不引入旧仓库的兼容层，
也不改变 workspace package 版本。

迁移包含：

- Manager-based G1 环境与 canonical 10 future-frame 时序契约；
- 版本化、只读 mmap 的 SONIC packed motion store；
- 通过 `PolicyVariant` 注册的 SONIC FastSAC actor、命名辅助损失和同步/异步训练；
- MotrixLab checkpoint 导出与回放；
- 小型 smoke store、行为测试和双语用户文档。

## 环境与数据

环境注册名为 `g1-sonic`，没有 profile 或兄弟变体。配置使用 typed
Manager group，term factory 返回 `ObsTerm`、`RewardTerm`、`TerminationTerm` 或
`ResetTerm`；所有 fused-kernel 入口使用 `@dispatch`。

`SonicMotionClip` 在通用 `WbtMotionClip` 数组之外保存 SMPL reference 和逐帧 clip
边界。运行时只读取 `motrixlab_sonic_packed_v1`，四元数统一为 `xyzw`，关节和 body
数组按 task contract 排列。环境在未设置变量时回退到仓库内的小型 store，以满足
registry 和集成测试契约；有效训练所需的完整动作集由 `SONIC_PACKED_STORE` 外部提供。
`configs/task/g1-sonic/motrix.fastsac.yaml` 是唯一 task recipe，直接内联
`algo.variant`。`model.num_future_frames=10` 同时驱动环境 observation 布局和
SONIC actor 输入；小规模验证只覆盖 CLI 的环境数、播放环境数、checkpoint 间隔和迭代数。

Action manager 在每次 `process` 前调用 `ActionTerm.prepare(sim_data)`。SONIC action
利用该 hook 保存动作应用前的足部关节速度，从而让 acceleration reward 使用正确的时间点。

## FastSAC

`FastSacCfg.policy_variant` 通过中性 `policy_variant_registry` 选择专用 actor；
`FastSacCfg.variant` 是由 SONIC 变体自解析的映射，FastSAC 不理解其字段。普通
FastSAC 路径不变。SONIC observation 最后两个 encoder selector 维度绕过经验归一化，
actor 自己输出归一化动作，具体 `effort/kp` 缩放由环境 action term 完成。
训练 checkpoint 写入 `policy_variant` 与 `policy_variant_metadata`，后者包含完整
模型映射和辅助权重，使 export/play 不依赖当前 YAML。训练 task recipe 不包含
`g1_control_decoder_hidden_dims`：本地 `SonicActor` 不构造该 decoder，并用
FastSAC policy head 替代上游 PPO control head。

异步 collector/learner 快照同时传输 actor parameters、persistent buffers 和 observation
normalizer state。这样 SONIC 的归一化 buffer 与 learner 保持一致，而普通 actor 的
buffer snapshot 大小仍为零。

MotrixLab 训练 checkpoint 的回放继续依赖 run metadata，与其他任务一致。

## MotrixSim 兼容依据

当前 workspace 固定 `motrixsim===0.10.1.dev121620`。公开文档没有完全相同的 tag，
迁移按最近且不高于目标版本的 `v0.10.0` API/MJCF 文档核对。控制写入必须保持
environment 轴和声明的 actuator 轴顺序；完整 actuator 写入会先转换为 native 顺序和
C-contiguous 布局。

## 发布约束

Bundled smoke store 的公开再分发已由用户确认，但可审计的授权引用、权利人和具体许可
文本尚未提供。该状态记录在 `THIRD_PARTY_NOTICES.md`，补齐凭据前不得将相关数据纳入
公开 release。
