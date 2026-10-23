# 异步训练目标架构实施

## 摘要

落实 [异步训练架构目标与验收](../design/rl-multi-algorithm-architecture.md#异步训练架构目标与验收)。最终以 A1–A9 逐项验收，不以现有 Base 继承或测试数量代替职责迁移。不实现 APPO，不改变 FastSAC 数据路径与预算语义。

## 分阶段 TODO

- [x] 定义纯声明 WorkerSpec / TrainingPlan、只读 TrainingProgress / WorkerReports 和 TrainingDecision；公共 WorkerRuntime 实现 ready/report/finish/stop 事件协议。
- [x] 公共 BuildRuntime 持有 spawn context 和通信资源作用域；公共 worker 外壳负责日志/placement/seed/信号和错误上报，错误优先且保留 traceback。
- [x] AsyncTrainerBase 仅通过 build_training/evaluate_progress/collect_metrics 驱动；独占 Process 构造、状态表、readiness/退出检查、消息消费及最终回收。
- [x] 公共 metrics/presentation 对象拥有 logger/Live/系统采样；算法提供指标映射和有效性，不提供生命周期闭包。
- [x] FastSAC 仅构建布局、资源和 worker 描述；移除 context/Process/raw exitcode 与 error/readiness queue 管理。
- [x] FastSAC worker 接入公共外壳，算法保留 slot 交付内容、CUDA初始化、DDP、采样/学习循环、checkpoint 与进度含义。
- [x] 迁移既有 FastSAC 指标/生命周期测试，并补独立 toy/真实 spawn 的 ready/finish/exit、失败和取消契约。
- [ ] A1–A9 全面验收；更新当前差距、API文档和证据；完成后删除本计划。

## 具体契约与约束

- 声明 spec 的 target 为模块级可 pickle callable，调用形式 target(runtime, *args, **kwargs)；worker ID 唯一且稳定，算法不能持有 process。
- 公共 runtime 的控制消息与生产 ring/snapshot 数据通道分离，report 仅用于低频进度、指标与产物通知，不代替 tensor 数据路径。
- WorkerRuntime.ready 报告初始化完成并等待父进程放行；finish 为算法完成声明；公共外壳不自动把普通函数 return 转换为 finish。
- BuildRuntime 的资源工具统一 queue/事件及 ExitStack 所有权。算法需要的共享 tensor resource 可登记清理，但不取得进程工厂。
- 暂存新公共契约期间不得破坏现有 FastSAC 可运行状态；迁移完成后删除旧 session API，不保留 legacy fallback。

## 阶段验收记录

- 公共目标基础：`training.py` 纯描述与只读状态、`building.py` 资源作用域、`workers.py` 公共外壳、`messages.py` 状态消息消费、`presentation.py` 输出生命周期、`rates.py` 滚动速率已分别落地。
- `plan_execution.py` 已独立构造真实 spawn worker，并覆盖 ready/finish、未 ready 退出、退出竞态、错误优先及 bounded peer completion。
- `FastSacMetrics` 已提为纯计算聚合，FastSAC parent/worker/handshake 已切换到 `TrainingPlan`、`WorkerRuntime` 与 typed reports；公共/FastSAC 相关迁移测试已分组验证。
- worker topology 已抽到 `motrix_rl.execution.topology`；FastSAC topology 现在只保留 transition/weight IPC 策略，common transport primitive 继续由 `motrix_rl.transport` 提供。新增 `EndpointWiring` / `wire_endpoints` / `cuda_ipc_capable` 描述通用 endpoint wiring 与 capability 选择；拓扑与 ring 专项当前 `109 passed`，endpoint 契约另有 `18 passed`。
- `PlanExecution` 现在在公共层生成 `run_started`、`worker_crashed` 事件并持久化 role 与 worker ID 错误日志；`TrainingPresentation` 对禁用 live sink 安全处理。
- 当前全套快速回归仍有 FastSAC live fixture 的最终事件/计数时序差异，故 A1–A9 尚未宣称完成；需先收敛 fixture 与最终 checkpoint/metrics drain 语义，再执行全套和 `prek run --all-files`。

## 完整验收入口
- 每阶段执行针对性测试；完整阶段执行 `.venv/bin/python -m pytest motrix_rl/tests motrix_rl_builtin/tests -m 'not slow and not numba' -q` 和 `prek run --all-files`；真实硬件与性能结论另列证据。
