# Plan

## 摘要

本目录保存已经完成设计确认、正在进入编码落地阶段的 feature 计划文档。每个计划文档都应基于对应的 design 文档编写，并包含可持续更新的 TODO list。

## 使用规则

- 只有在相关设计已经与用户确认并写入 `wiki/design/` 后，才能在本目录新增或更新 plan 文档。
- 如果已经存在相关主题的 plan 文档，应优先直接修改原文档，而不是创建重复主题的多个计划。
- TODO 条目在完成后必须及时打钩，保证计划状态可追踪。
- 当某个 plan 对应的功能已经完整实现时，应删除该 plan 文档，并同步更新本页索引。

## 文档列表

- [fastsac-async-multi-learner.md](fastsac-async-multi-learner.md) — FastSAC 单机多卡（多 learner × DDP）实施计划
- [motion-multi-clip-infra.md](motion-multi-clip-infra.md) — 多 motion clip 训练基础设施（MotionLibrary + WbtMotionCommand 原地泛化）分阶段实施计划
