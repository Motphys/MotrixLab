# Manager 设计

## 摘要

本目录集中保存 Manager-based environment 的核心设计与开发指南，包括运行时边界、任务配置、kernel ABI、仿真后端和 query 编译。

## 文档列表

### 核心设计

- [Manager Runtime 设计](./runtime.md)
  ManagerEnv 的运行时边界、配置到 term 的构造链路、step/reset 生命周期、term 协议和 Numba 性能约束。
- [Manager Task API](./task-authoring.md)
  面向任务开发者的 ManagerBasedEnvCfg、manager groups、term 创建、query 声明、action、observation、reward、termination 和 reset 指南。
- [Manager Kernel ABI](./kernel-abi.md)
  定义 ManagerContext、KernelData tree、leaf scope、term ownership、metric backing 和 fused kernel fingerprint。

### Simulator 集成

- [Manager SimBackend 设计](./sim-backend.md)
  定义 backend-neutral SimBackend、模型查询、读写编译、reset 和 backend registry 边界。
- [读侧编译：SimBackend.compile_reads 与 PhysicsReadProgram](./sim-read-program.md)
  定义 query declaration、read program、稳定 logical view、physical deduplication 和 partial read 契约。
- [SimDataQuery Backend 接入指南](./sim-query-backend-guide.md)
  面向 simulator backend 接入者，说明 typed query resolver、physical planning、logical view 和测试验收。

### 专题设计

- [Interval Events](./interval-events.md)
  定义 interval 专用事件协议、逐环境 timer、pre-physics 写入和 episode reset 生命周期。

- [Term 自声明 Sim Query 设计](./term-required-queries.md)
  定义 observation term 的 `required_queries()`、任务声明与 term 贡献的合并和冲突规则。
- [IsaacLab ManagerCfg Adapter 设计](./isaaclab-manager-cfg-adapter.md)
  定义 IsaacLab-like ManagerCfg 到 MotrixLab native term/query 配置的适配边界；本文是目标设计，尚未表示全部 API 已实现。

## 使用说明

- Manager 核心接口、运行时和数据契约的设计文档放在本目录。
- 具体任务设计、跨模块通用设计和性能实证仍放在上级 `design/` 目录。
- Manager 功能的实施计划放在 `wiki/plan/`，不与设计文档混放。
