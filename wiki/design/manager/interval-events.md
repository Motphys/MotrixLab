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

manager 侧开销的实测数据与后续优化计划见 [interval-events-optimization](../../plan/interval-events-optimization.md)；可用 `scripts/bench_interval_events.py` 在目标硬件上复测。
