# Research

## 摘要

本目录保存研究性质文档，包括理论背景、数学推导、算法可行性探索，以及尚未形成明确方案的技术积累。

## 文档

- [MotrixSim CUDA 能力调查与实测](./motrixsim-gpu-backend-review.md)
- [SAC 与 FastSAC 算法原理入门](./sac-and-fastsac-primer.md)：面向初学者解释 SAC 的 soft return、critic target、actor loss、温度系数，以及当前 FastSAC 实现中的代码对应关系。
- [NumbaWbtTask.update_state 远端 96 核扩展性研究](./numba-wbt-remote-96c-scaling.md)：NumbaWbtTask 远端 EPYC 96 核瓶颈分析、融合 kernel 优化、线程调优与阶段性端到端基准。
- [FastSAC GPU 利用率分析与优化研究](./fastsac-gpu-analysis.md)：异步训练 GPU 不满问题的分层评测方法（面板统计/microbenchmark/CUDA Event/双进程 trace/真实训练 A/B）、idle 根因定位、compile 粒度与 CUDA Graph 实验结论，以及评测 KPI 选择教训。

## 使用说明

- 仅当文档仍处于探索阶段、结论未定时，才应放入本目录。
- 一旦形成明确方案，应迁移到 `design/` 并更新索引。
