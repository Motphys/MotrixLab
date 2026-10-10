# Interval Events 性能优化计划

## 摘要

记录 interval events（[设计文档](../design/manager/interval-events.md)）的性能实测结论与后续优化计划。当前事件开销在典型配置下可忽略，本计划预置两档优化设计与 benchmark 工具，待服务器实测数据决定是否启动。

## 背景：2026-10 本地实测结论

### 开销量级（fake write backend，仅 manager 侧）

4096 envs、ctrl dt 0.02s、错开 timer：

| 配置 | apply() 每步开销 |
|---|---|
| 2 事件 | ~40 μs |
| 8 事件 | ~134 μs（其中 timer 扫描 ~31 μs） |

事件 kernel 是 CPU Numba njit（selected-row 语义），无 GPU launch 开销；大规模训练下（4096 envs、interval 5-10s）每步期望 ~11 行到期，事件几乎每步触发但 kernel 只处理少量行。典型任务 1-2 个事件的开销占毫秒级 physics step 不到 5%。

### 负结果：numpy 批处理 timer 扫描（勿重试）

将逐事件 timer 扫描（in-place `-= dt` + 小数组 `flatnonzero`）批量化为 2D `nonzero` + `bincount` + `cumsum` 是**负优化**，慢 2-6 倍（临时数组分配与调用开销超过省下的小调用）。per-event 扫描在所有实测配置下最快。数据可用 `scripts/bench_interval_events.py` 复测。

## 优化设计（已确认方向，未实施）

### 档 1：事件间共享 write（manager 层，优先）

将事件按 **writes 签名**（同 write 类型 + 同 body）分组，组内共享一个 delta buffer 和一个 WriteProgram：

1. timer 扫描不变（逐事件 `-= dt` + `flatnonzero`）；
2. 组内有 due 行时：取组内 due 行并集，清零 buffer 的并集行；
3. 逐事件 kernel 对各自 due 行采样并 `buffer[row] += delta`（累加顺序 = 声明顺序，保持确定性）；
4. 每组每步只执行一次 `program.execute(union)`。

收益：E 次 program.execute → 组数 G 次；E=1-2 时 G==E 零收益，E=8 时约省 60-80 μs/step。不改 codegen，改动集中在 `IntervalEventManager`，行为对现有测试基本透明。

### 档 2：timer 融入 action kernel（compiler 层，最后考虑）

timer 布局改为 `(num_envs, num_events)`（per-env 行连续，匹配按环境遍历的 fused kernel），把 timer 递减、due 检测、interval 重采样、delta 采样全部并入 action kernel 的 per-env 循环；kernel 内将 due 行写入预分配的 compact `env_ids` 输出数组 + 计数器，host 侧只做 `program.execute(due_ids[:count])`，完全绕开 host 侧扫描。

已知代价：action kernel 编译指纹包含事件配置（改 kick 参数需重编 action kernel）；action/event 模块边界消失；RNG 消耗顺序变化（版本内可复现性不受影响）。收益上限 ~50-110 μs/step（8 事件 @4096 envs）。

与 action **WriteProgram** 合并不可行：program 的 `env_ids` 是整 program 一套，action 需要全行、事件只要 due 行，除非事件改全行填零（backend 写入量放大，负收益）。

## Benchmark 计划

`scripts/bench_interval_events.py`（已就绪，fake write backend，仅测 manager 侧开销；motrixsim 原生 write 成本用真实训练的 `perf` scope `interval_events` 单独观察）：

```bash
python scripts/bench_interval_events.py                        # 默认网格
python scripts/bench_interval_events.py --num-envs 4096 --events 8 --steps 1000
```

- 场景：`hot`（interval≈一个 step，全部行每步到期，上界）/ `staggered`（错开 timer，典型大规模）/ `cold`（无到期，纯 timer 扫描）；
- 附带 timer 扫描策略对比（per_event / batched_2d / batched_flat），在新硬件上复核负结果。

### 判读标准

- staggered 场景 E=1-2 的 μs/step 占整步时间 <2%：维持现状；
- 任务实际配置 ≥8 事件且档 1 预估节省（≈(E-G)×program.execute 成本）>整步 5%：实施档 1；
- 档 1 落地后 profile 仍显示 timer/扫描为热点：再评估档 2。

## TODO

- [ ] 服务器运行 `scripts/bench_interval_events.py` 全网格，回填本档实测数字
- [ ] 依据判读标准决定是否启动档 1
- [ ] （视档 1 结果）评估档 2
