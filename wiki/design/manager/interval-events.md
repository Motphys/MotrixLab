# Interval Events

## 摘要

IntervalEvent 为 Manager 环境提供逐环境独立的周期性 simulator 写入。首个具体事件是世界坐标 root linear velocity kick；不提供 mode、external wrench 或持续力生命周期。事件运行于 action 写入之后、physics 之前，遵循 [Manager Runtime](./runtime.md) 的 frontend/backend 边界。

## 配置与职责

`ManagerBasedEnvCfg.interval_events` 是有序的 `dict[str, IntervalEventCfg]`。`IntervalEventCfg` 使用 `@configclass`，声明正数秒区间并通过 BuildContext 构造一个 `IntervalEvent`。事件描述对象声明 `@dispatch` 数值入口、静态参数和命名 SimWrite 输出。IntervalEventManager 持有 timer、每个事件独立的编译 kernel 和 WriteProgram；simulator handle 不进入 kernel。

每个事件独立编译 selected-row kernel，按原始 env id 读取 per-env 随机状态并填写原始行的 write buffers。入口数值逻辑使用 @dispatch，Numba 编译由 manager compiler 管理。事件不融合进 reward/observation/termination kernel。每项事件的 write program 固定 reset=False、forward_kinematics=False；不同事件按声明顺序执行，对相同 body 的增量可自然累加。

## 调度与生命周期

每个事件的每个 environment 保存一个 remaining-time timer。episode reset 只为对应行重新采样 interval；sim-only reset 不修改 timer。初始化通过正常 episode reset 初始化 timer，编译预热不得消耗真实随机状态。

每个 control step：action -> interval_events.apply(ctrl_dt) -> physics -> normal read/evaluate/reset/observe。timer 减去 ctrl_dt，remaining <= 0 的行触发一次，并重新采样 interval；不补发同一步内的多个历史事件。无到期行时不执行事件数值入口或 simulator read/write。事件不进行 simulator read；physics 之后的正常 read 刷新派生状态。

采样复用现有 per-env PRNG；相同 seed、配置顺序和调用序列下结果可复现，线程调度不影响每行随机序列。

事件的 `ctx.sim` 与 SimDataQuery 参数使用最近一次正常 read 的 snapshot；事件本身不刷新 simulator 数据。因此不同事件之间的写入不能通过这些缓存查询立即相互观察，增量写入直接作用于 backend 当前状态。

## Velocity kick

RandomVelocityKickCfg 声明目标 body、interval_range_s 和 world XYZ 的 velocity_delta_x/y/z（每轴独立 (low, high)，默认 (0.0, 0.0)）。每次触发独立均匀采样各轴，填写 AddBodyLinearVelocityWrite 的 (N, B, 3) float32 buffer。它表示 velocity += delta，不是 absolute velocity、force 或 impulse。首版不为其他模式建立通用事件抽象。

## 配置示例

```python
from motrix_env_core.mdp.events import RandomVelocityKickCfg

cfg.interval_events = {
    "push": RandomVelocityKickCfg(
        interval_range_s=(5.0, 10.0),
        body=cfg.scene.objs.robot.resolved_base_link_name,
        velocity_delta_x=(-1.0, 1.0),
        velocity_delta_y=(-1.0, 1.0),
    ),
}
```

使用空映射禁用事件。通用框架不自动修改具体任务的训练或 play preset。

## 验证

验证空触发、selected-row 写入隔离、重复累加、多事件独立触发、partial reset、sim-only reset、seed 可复现与 warmup 随机状态不变。前端集成测试验证 kick 影响当前 physics step 的 reward/termination/observation；backend 保持中立。

## 性能

`apply()` 每事件依次执行：timer 递减与 `flatnonzero` 到期扫描 → 串行 njit kernel 只遍历 due 行（重采样 interval、采样 delta、累加写入 buffer）→ `program.execute(due)` 提交 backend。事件 kernel 是 **selected-row 串行 kernel**：每步只处理少量 due 行，若以 `numba.prange` 编译，parallel 启动开销随线程池规模线性增长（192 核机器上每次调用固定 ~150 μs，即使 0 行 due），远超实际计算量，因此 manager 编译器对 interval event kernel 强制串行编译。

实测数据（`scripts/bench_interval_events.py`，fake write backend，仅 manager 侧开销，192 核 EPYC 9J14，dt=0.02s，staggered 场景）：

| 事件数 | num_envs | 串行化前 (μs/step) | 串行化后 (μs/step) |
|---|---|---|---|
| 1 | 4096 | 198 | 31 |
| 2 | 4096 | 368 | 63 |
| 8 | 4096 | 1602 | 262 |
| 16 | 4096 | — | 520 |

串行化后成本随事件数线性（~32 μs/事件 @4096 envs），其中 timer 扫描 ~4.6 μs、kernel 与提交占其余；纯扫描（无到期）~3.7 μs/事件。毫秒级 physics step 下典型 1-2 个事件的开销占比 <1%。

已验证的负结果与边界：

- 将逐事件 timer 扫描批量化为 2D `nonzero` + `bincount` + `cumsum` 是负优化（慢 2-6 倍）；flat 批量在 ≥8 事件时快约 2 倍，但绝对收益仅 ~16 μs/step，未达实施阈值。
- 事件间共享 WriteProgram（按 writes 签名分组，E 次 execute → G 次）预估仅省 ~5 μs × (E−G)/步，未达整步 5% 阈值，不实施。
- timer 融入 per-env parallel kernel（消除 host 侧扫描）的收益上限实测为 ~5.7 μs/事件（扫描 3.9 + kernel dispatch 1.8）；剩余 ~70% 成本是 `program.execute` 的 backend 写入（~23 μs/事件，需压缩 due_ids 后 host 侧提交，无法移入 kernel）。且该方案要求一个 physics step 之前的 per-env kernel 作为融合宿主——action term 是 host 侧可扩展 Python 管线（`ActionTerm.process`），不是 Numba kernel；evaluate/observe kernel 又在 physics 之后运行，时序不匹配。综合收益（8 事件@4096 约 262→210 μs/step）与代价（action/event 边界消失、编译指纹耦合、RNG 消耗顺序改为 env-major）不成比例，不实施。

可在目标硬件上用 `scripts/bench_interval_events.py` 复测（脚本开头默认设置 `OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`，避免 OpenMP 自旋干扰微秒级计时）。
