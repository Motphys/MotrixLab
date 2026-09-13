# 教程总览

本节按照"先建立整体图景，再逐步深入细节"的顺序组织，左侧目录即完整的阅读结构：

- **基础框架**：MotrixLab 的分层架构与运行流程，理解环境、RL provider、Hydra 配置与 Trainer 如何协作。
- **构建环境**：先写环境（DirectEnv、ManagerEnv 两篇指南），再引入两个横切概念
  （SceneCfg 场景与仿真配置、SimBackend 与仿真器解耦）；入口是环境构建总览
  （环境组成、生命周期、工作流选型）。
- **训练与结果**：从创建 Task 配置、执行训练到分析 `runs/` 产物与 checkpoint。
- **进阶主题**：按需深入的独立能力——ONNX 导出、真机部署、指令输入架构与自定义训练后端。

如果还没有跑通第一个训练，建议先阅读[入门指南](../getting_started/installation.md)；
如果只想了解环境实现，直接从[环境构建总览](building_envs/index.md)开始。
