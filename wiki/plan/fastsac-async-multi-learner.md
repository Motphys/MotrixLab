# FastSAC 单机多卡（多 learner × DDP）实施计划

## 摘要

落地 [design 文档 §9](../design/fastsac-async-heterogeneous-trainer.md)：`num_learners > 1` 时每 GPU 一个 learner 进程做 DDP 数据并行，collector 按 `num_collectors % num_learners == 0` 归属到各 learner；`world_size == 1` 时行为与现有单 learner 逐字节一致。

## TODO

- [x] config：`num_learners` / `learner_devices`，yaml 同步；NUMA node 自动分配（collector round-robin、learner GPU 本地），无手工列表
- [x] 校验：整除关系、设备索引唯一、`batch_size % num_learners == 0`
- [x] agent：`world_size` 参数；DDP 包装 actor/qnet；`log_alpha` 梯度手动 allreduce；本地 batch = `batch_size // num_learners`；多 learner 禁 learner 侧 compile
- [x] learner：`maybe_train_global(gstep)`——update 次数从全局进度基准增量推导（跨 rank 锁步）
- [x] worker：`run_learner_process` 增加 rank / world / rendezvous / device；rank 0 建全部 weight sender 并 ship 全部 slot queue、drain 全部 stats queue、独占 checkpoint/TB；退出前 barrier
- [x] train：spawn N 个 learner；ring 按 `i // k` 归属分区；`collector_inference_device: "cuda"` 解析为归属 learner 的卡
- [x] 测试：分区映射、锁步 due 计算、config 校验、collector 设备解析（CPU 可跑，不依赖 N 卡）
- [x] ruff + 全量 motrix_rl 测试 + 提交推送
- [ ] 真实多卡（≥2 GPU）环境端到端验证（本机单卡，仅验证了校验路径与单/多 collector 全链路）
