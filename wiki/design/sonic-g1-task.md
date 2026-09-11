# SONIC G1 任务迁移设计

## 目标与边界

本设计将 SONIC 的 G1 动作跟踪能力接入 MotrixLab 当前九包 workspace，同时保持现有
Manager、simulator registry、Hydra Task 和 FastSAC 公共契约。不引入旧仓库的兼容层，
也不改变 workspace package 版本。

迁移包含：

- Manager-based G1 环境与三种 temporal profile；
- 版本化、只读 mmap 的 SONIC packed motion store；
- SONIC FastSAC actor、辅助损失和同步/异步训练；
- MotrixLab checkpoint 导出及官方 SONIC checkpoint 的受限回放入口；
- 小型 smoke store、行为测试和双语用户文档。

## 环境与数据

环境注册名为 `g1-sonic`、`g1-sonic-lafan` 和 `g1-sonic-smoke`。配置使用 typed
Manager group，term factory 返回 `ObsTerm`、`RewardTerm`、`TerminationTerm` 或
`ResetTerm`；所有 fused-kernel 入口使用 `@dispatch`。

`SonicMotionClip` 在通用 `WbtMotionClip` 数组之外保存 SMPL reference 和逐帧 clip
边界。运行时只读取 `motrixlab_sonic_packed_v1`，四元数统一为 `xyzw`，关节和 body
数组按 task contract 排列。三个环境在未设置变量时都回退到仓库内的小型 store，以满足
registry 和集成测试契约；有效训练所需的完整动作集由 `SONIC_PACKED_STORE` 外部提供。

Action manager 在每次 `process` 前调用 `ActionTerm.prepare(sim_data)`。SONIC action
利用该 hook 保存动作应用前的足部关节速度，从而让 acceleration reward 使用正确的时间点。

## FastSAC

`FastSacCfg.sonic` 选择专用 actor 和模型 profile。普通 FastSAC 路径不变。SONIC
observation 最后两个 encoder selector 维度绕过经验归一化，actor 自己输出归一化动作，
具体 `effort/kp` 缩放由环境 action term 完成。

异步 collector/learner 快照同时传输 actor parameters、persistent buffers 和 observation
normalizer state。这样 SONIC 的归一化 buffer 与 learner 保持一致，而普通 actor 的
buffer snapshot 大小仍为零。

MotrixLab 自身训练 checkpoint 继续依赖 run metadata。只有用户显式指定
`env=g1-sonic policy=<file>` 且文件上方不存在 `metadata.json` 时，play CLI 才进入
官方 SONIC release loader；其他环境不接受该 fallback。

## MotrixSim 兼容依据

当前 workspace 固定 `motrixsim===0.10.1.dev121620`。公开文档没有完全相同的 tag，
迁移按最近且不高于目标版本的 `v0.10.0` API/MJCF 文档核对。控制写入必须保持
environment 轴和声明的 actuator 轴顺序；完整 actuator 写入会先转换为 native 顺序和
C-contiguous 布局。

## 发布约束

Bundled smoke store 的公开再分发已由用户确认，但可审计的授权引用、权利人和具体许可
文本尚未提供。该状态记录在 `THIRD_PARTY_NOTICES.md`，补齐凭据前不得将相关数据纳入
公开 release。
