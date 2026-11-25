# Manager 设计

## 摘要

本目录集中保存 Manager-based environment 的核心设计与开发指南，包括运行时边界、任务配置、仿真后端和 query 编译。

## 文档列表

### 核心设计

- [Manager Runtime 设计](./runtime.md)
  ManagerEnv 的运行时边界、配置到 term 的构造链路、step/reset 生命周期、term 协议和执行/性能约束。
- [Manager Task API](./task-authoring.md)
  面向任务开发者的 ManagerBasedEnvCfg、manager groups、term 创建、query 声明、action、observation、reward、termination 和 reset 指南。

### Simulator 集成

- [Manager SimBackend 设计](./sim-backend.md)
  定义 backend-neutral SimBackend、模型查询、读写编译、reset 和 backend registry 边界。
- [读侧编译：SimBackend.compile_reads 与 PhysicsReadProgram](./sim-read-program.md)
  定义 query declaration、read program、稳定 logical view、physical deduplication 和 partial read 契约。
- [SimDataQuery Backend 接入指南](./sim-query-backend-guide.md)
  面向 simulator backend 接入者，说明 typed query resolver、physical planning、logical view 和测试验收。

### 专题设计

- [History Observation 与单帧传输契约](./observation-history.md)
  定义 group 级历史配置、step 返回 stacked obs 与按静态描述提取尾帧、reset/final 生命周期，以及 RL 侧 ring payload 组装、FastSAC 和 SuperMimic 的接入边界。
- [Term 自声明 Sim Query 设计](./term-required-queries.md)
  定义 term 通过 `SimDataQuery` 参数自声明 sim 数据需求、与任务声明合并的 key 单一属主规则，以及 resolver 到批量 sim 视图的映射。
- [IsaacLab ManagerCfg Adapter 设计](./isaaclab-manager-cfg-adapter.md)
  定义 IsaacLab-like ManagerCfg 到 MotrixLab native term/query 配置的适配边界；本文是目标设计，尚未表示全部 API 已实现。
- [Torch Manager 执行器设计](./torch-executor.md)
  定义 Torch 执行器（唯一数值执行器）的调度语义、批量 term ctx 约定与 resolver 分工。
- [训练随机性与 Seed 所有权](../randomness-and-seeding.md)
  `ctx.rand` 的默认 Torch 流、训练入口的 seed 职责、SuperMimic 模型初始化/探索隔离、preview 共享随机消耗与恢复语义。

## 使用说明

- Manager 核心接口、运行时和数据契约的设计文档放在本目录。
- 具体任务设计、跨模块通用设计和性能实证仍放在上级 `design/` 目录。
- Manager 功能的实施计划放在 `wiki/plan/`，不与设计文档混放。
