# Manager SimBackend 设计

## 摘要

`SimBackend` 是 ManagerEnv 和 DirectEnv 共用的 backend-neutral simulator 边界。backend 在构造期把
`SceneCfg` 编译为自己的模型和批量状态；前端只通过模型 query compiler、读程序、写程序、`step`、可选渲染和地形采样访问仿真，
不接触具体 simulator 类型。

Manager runtime 与任务配置见 [Manager Runtime](./runtime.md) 和 [Manager Task API](./task-authoring.md)。读侧细节见
[PhysicsReadProgram](./sim-read-program.md)，reset write 的声明式设计见 [Reset Write Program 设计](../reset-program.md)。

## 1. 边界

```text
DirectEnv / ManagerEnv
    │ 仅依赖
    ▼
SimBackend(scene, sim, num_envs)
    ├── model_compiler
    ├── compile_reads -> PhysicsReadProgram
    ├── write_compiler -> WriteProgram (普通写入 / reset=True)
    ├── step(substeps)
    ├── 可选 create_renderer / sample_terrain_height
    └── backend 私有 simulator model/data/state
```

`motrix_env_core` 不 import 具体 simulator。MotrixSim、MuJoCo 等 backend 在自己的发行包中注册，前端通过
`motrix_env_core.sim.registry` 按 registry 的 `sim` 参数选择；未指定时使用注册的默认 backend。

## 2. 当前接口

`SimBackend` 构造函数是 `(scene, sim, num_envs)`，构造完成后以下能力立即可用：

```python
class SimBackend(abc.ABC):
    @property
    def num_dof_pos(self) -> int: ...

    @property
    def num_dof_vel(self) -> int: ...

    @property
    def num_actuators(self) -> int: ...

    @property
    def model_compiler(self) -> SimModelCompiler: ...

    def compile_reads(self, queries) -> PhysicsReadProgram: ...

    @property
    def write_compiler(self) -> SimWriteCompiler: ...

    def step(self, substeps: int) -> None: ...

    def create_renderer(self, config, *, num_envs, render_spacing, system_camera) -> SimRenderer: ...

    def sample_terrain_height(self, geom_name, env_ids, xy) -> np.ndarray: ...
```

模型表面通过 `model_compiler.compile(scene, queries)` 组装为 typed `SimModel`（`bodies` 由 SceneCfg 无条件驱动，`others` 由声明的 query 驱动）。`SimModel` 只提供通用静态模型表面：

- `actuators`：按 canonical actuator order 排列的 `ActuatorSpec`；
- `init_dof_pos`：默认 DOF position；
- `others`：环境显式声明的 `ModelQuery` 结果。

backend 的内部 model/data/layout 不属于公共接口。需要 simulator 数据时，环境必须声明 `SimDataQuery` 并使用编译后的
`PhysicsReadProgram`；不得从 backend 取得原始 state。

## 3. 读写职责

- `compile_reads` 接收框架解析后的完整 query mapping，backend 负责 resolver、physical planning、duplicate deduplication、
  stable logical views 和 full/partial execute；详见 [读侧编译](./sim-read-program.md)。
- `write_compiler` 将声明式 write op 编译为 `WriteProgram`。program 持有自己的值 buffer，环境填充 buffer 后调用
  `execute(env_ids)`；名字和 target 在构造期校验。
- `step(substeps)` 只推进 physics。时间属于 control/training loop，reset 不通过 step 推进物理。
- `create_renderer` 和 `sample_terrain_height` 是可选 backend capability；不支持时必须显式报错。

Reset 使用 `write_compiler.compile(writes, reset=True)` 生成的 reset-mode `WriteProgram`，不再要求 frontend 构造 scene-global canonical DOF rows。backend 私有 layout、default restore、局部状态应用和派生运动学刷新都属于该 program；完整设计见 [Reset Write Program](../reset-program.md)。

## 4. Backend 注册与注入

```python
backend_name = sim or default_sim_backend_name()
factory = create_sim_backend(backend_name)
sim = factory(cfg.scene, cfg.sim, num_envs)
```

backend package 通过 `motrix_env.sim_backends` entry point 注册到 `motrix_env_core.sim.registry`。注册采用 lazy factory，导入
core 或 registry 不应启动具体 simulator。`registry.make(..., sim=name)` 会把选定名称作为构造参数传给环境；未指定时传入注册的默认 backend。

## 5. 不变量与测试

- backend 身份在环境生命周期内不变，构造期完成 scene translation；
- 具体 simulator 类型不跨越 `motrix_env_core` 边界；
- `SimModel.others` 只包含环境声明的 model query 结果；
- read/write/reset program 自己拥有运行期 buffer，前端只通过 program surface 访问；
- full/partial read 和 reset 保持原始 environment row mapping；
- fake backend 测试覆盖构造、query、write、step、partial reset 和可选 capability 的行为；
- import purity 测试确保 import `motrix_env_core`/manager 不加载具体 simulator。
