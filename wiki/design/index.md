# Design

## 摘要

本目录保存已经形成的设计方案和架构说明。

## 文档列表

- [项目整体架构设计](./architecture.md)
  系统总体分层、核心原则（Model/Data/State 分离、环境与 RL 框架解耦、注册表驱动、配置层次继承）、仿真后端抽象、RL 框架集成方式和数据流总览。
- [Manager 设计](./manager/index.md)
  Manager-based environment 的核心设计与开发指南，包括运行时边界、配置接口、kernel ABI、仿真数据查询和 action 所有权。
  192 核 EPYC 上 ManagerEnv task kernel 加速比从桌面 4x 塌缩到 1.3x 的根因诊断：受控因子实验把损失分解为 NUMA 跨 node 远端内存（H1）、默认 omp 相对 tbb 的线程层劣势（H2）、每核内存带宽/主频硬件上限（H3）；H1/H2 可修（numactl 钉扎 + tbb + 线程数按 NUMA 拓扑），H3 是硬件上限，剩余差距只能靠减少 kernel 访存量。
- [ONNX 模型导出设计](./onnx-export.md)
  基于 `RlFramework`/`AgentProvider` capability 的统一 ONNX policy exporter：从 metadata-backed run 恢复不同训练后端的 deterministic policy，烘焙 observation normalization，并执行 ONNX Runtime parity validation。
- [Motrix Deploy 框架设计](./motrix-deploy.md)
  独立策略部署包 `motrix_deploy` 的统一 ONNX export、deployment artifact、RobotState/RobotCommand/RobotInterface 契约、inference control loop、组件扩展边界与 MuJoCo sim2sim vertical slice。
- [Deploy Runtime Command Input 分层设计](./deploy-command-input.md)
  面向 Go2 的最小 command input 抽象：batch-first `PlanarVelocityCommand` 统一表达 training 与 deploy；core 内置 keyboard/gamepad/constant bindings，带 standing probability 的 `RandomPlanarVelocityBinding` 作为 task-specific 训练策略留在四足环境模块。
- [RL 多算法架构设计](./rl-multi-algorithm-architecture.md)
  在 RL 集成层引入 `rllib/train_backend/algo` 正交维度的架构：配置注册表、`RlFramework`/`AgentProvider`、`TrainerContext` 与通用 runner、算法配置与模型/memory 工厂、run metadata、checkpoint manifest 与 play 自动发现，以及内置 framework 矩阵（skrl / rslrl / motrix）。
- [FastSAC 异构（Collector/Learner 分进程）训练器设计](./fastsac-async-heterogeneous-trainer.md)
  把仿真采样与网络训练拆到两进程、经共享内存交换数据的异构 FastSAC 训练器：统一注册为 `motrix.fastsac`，由 `algo.asynchronous` 选择执行拓扑，并包含 SPSC 有界背压 replay 环、update-to-data 比例治理、seqlock 双缓冲权重/normalizer 快照，以及进程生命周期与 checkpoint 兼容。
- [Framework / Task 配置分离设计](./framework-task-split.md)
  训练入口通过 Hydra task group 直接组合环境、算法与运行配置；外部应用使用自己的 Hydra config root，不维护 Python task registry 或额外 config-root 状态。
- [ConfigClass 配置装饰器设计](./configclass.md)
  定义标准 dataclass 兼容的配置装饰器语义：直接声明可变容器与嵌套配置默认值、逐实例深拷贝隔离、显式 `field()` 优先，以及类型检查和 Hydra/OmegaConf 兼容契约。
- [Scene / Robot 配置设计](./scene-model-config.md)
  使用字段式 `SceneAssetsCfg` 与 `SceneObjsCfg` 声明具名 asset 和有序 scene object，通过 `SceneVisualCfg` 配置全局视觉环境，并由可继承的 `StandardSceneCfg` 提供标准 skybox、地面、材质、贴图、haze 和方向光。
- [G1 WBT 环境设计](./g1-wbt-env.md)
  MotrixLab 中 G1 whole-body tracking 环境（direct `DirectEnv`，注册 `g1-29dof-wbt-largebox` / `g1-wbt-dance`）的设计：`WbtMotion` 数据加载与 name-based 状态映射、reset/step 时间推进、effort-scaled PD 控制、actor/critic observation、tracking reward、bad-tracking termination、adaptive timestep sampler、`motrix.fastsac` 执行拓扑切换与 play 变体。
- [G1 Backflip 飞行指标口径设计](./g1-backflip-flight-metrics.md)
  `g1-wbt-backflip` 飞行技能指标在 mixed/RSI 采样下的口径修正：不可达判定点对均值的稀释、固定帧判定与物理落地错位、连续均值不可解读等问题定义，以及条件指标 + 覆盖率、二值成功率、物理落地事件与 checkpoint 选择解耦的修正设计。
- [MotrixLab Motion NPZ Schema 设计](./motrixlab-motion-npz-schema.md)
  MotrixLab 自己的 WBT `.npz` 动作文件格式 v1：在 BeyondMimic 原生 schema 上加 `schema_version / joint_names / body_names / num_frames` 与 `ext_*` 扩展槽位，四元数约定切到 xyzw 与 MotrixSim 内部 API 对齐。提供通用 `MotrixMotion` loader、Holosoma/BeyondMimic → MotrixLab 一次性 converter、`scripts/motion/replay.py` 通用化。
- [用户文档 Envs 栏目设计](./user-docs-envs-part.md)
  在现有双语用户指南的 Tutorials 之后建立独立 `Envs` 一级栏目：明确 Environment、Robot、Training Task 与 Tutorial Workflow 的边界，基于 registry、双语 metadata 和 Hydra Task 配置自动生成 Env ID、description 与支持训练算法的 overview 表格，以 Whole Body Tracking、Humanoid Locomotion、Quadruped Locomotion 等任务主题组织环境，并通过先建导航、再统一页面、最后迁移文件的方式渐进落地。

## 使用说明

- 设计文档应优先放入对应主题目录，而不是直接平铺在本层。
- 新增或迁移文档时，需要同步更新本页以及对应子目录的 `index.md`。
