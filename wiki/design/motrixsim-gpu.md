# MotrixSim GPU 仿真与原生后端架构设计

## 摘要

本文定义 MotrixLab 引入 NVIDIA CUDA 物理仿真（`motrixsim-gpu`）的完整架构契约：保持“环境前端张量设备始终跟随 `sim.device`”的单一设备模型，通过 `EnvCfg.backend` 与统一工厂 `create_sim_backend(name, device)` 完成后端与设备的单点解析；在后端执行层区分 CPU 暂存批处理模型与 GPU 常驻零拷贝视图/掩码执行模型；并在 `motrix_rl` 单进程与异步多进程拓扑中统一对齐 `sim_device` 局部性。验收范围见 [MotrixSim GPU 实施与验收](../plan/motrixsim-gpu.md)。

---

## 1. 单一设备模型与后端解析契约

### 1.1 单一设备不变量（Single-Device Invariant）

- `SimCfg.device: str = "cpu"` 是单个环境实例唯一的设备配置源，并在 `SimCfg.validate()` 中完成合法设备语法与设备类型（`"cpu"` / `"cuda"`）校验。
- `ArrayEnv`（含 `DirectEnv` 与 `ManagerEnv`）在构造期将 `self.device` 规范化为明确的 `torch.device`（未带索引的 `"cuda"` 解析为当前可见默认卡 `cuda:<current>`）。环境生命周期张量、Manager 上下文与持久状态、动作输入、观测输出以及物理后端数据均驻留在同一 `self.device` 上，不拆分独立的“物理设备 vs 前端张量设备”双轴。
- `registry.make(..., sim=..., sim_device=...)` 与 `EnvBuildSpec.make(..., sim_device=...)` 作为运行期入口覆写 `env_cfg.sim.device`，显式参数优先于环境静态默认配置（内置任务默认 `"cpu"`）。[环境注册入口](../../motrix_env_core/src/motrix_env_core/registry.py)

### 1.2 统一后端配置与工厂派发

- `EnvCfg` 基类统一定义 `backend: str | None = None`（`DirectEnvCfg` 与 `ManagerBasedEnvCfg` 共享同一字段，子类不重复声明）。`DirectEnv` 与 `ManagerEnv` 均按 `sim_arg or cfg.backend or default_sim_backend_name()` 解析后端名称。
- `motrix_env_core.sim.registry` 提供单点后端类解析入口：

```python
def create_sim_backend(
    name: str | None = None,
    device: torch.device | str = "cpu",
) -> type[SimBackend]: ...
```

- 后端发行包通过 `motrix_env.sim_backends` entry point 注册惰性解析器 `(device: torch.device) -> type[SimBackend]`：
  - `motrix-env-motrixsim` 在 `device.type == "cpu"` 时返回 `MotrixSimBackend`；在 `device.type == "cuda"` 时校验可选依赖 `motrixsim.gpu` 与目标卡可用性，直接返回 `MotrixCudaBackend`。
  - 仅安装 CPU 依赖时可正常导入插件并运行全部 CPU 任务；显式请求不支持的设备类型、缺少 `motrix-env-motrixsim[gpu]` 或卡号越界时在工厂解析期立即给出明确异常。[插件依赖声明](../../motrix_env_motrixsim/pyproject.toml)
- 所有 `SimBackend` 子类遵循统一构造签名 `(scene: SceneCfg, sim: SimCfg, num_envs: int, *, render: RenderConfig | None = None)`。

---

## 2. 双轨后端执行模型：CPU 暂存批处理 vs GPU 原生常驻掩码执行

任务与前端层保持统一的声明式接口（`SimDataQuery`、`ModelQuery`、`SimWrite`、`ManagerResetCfg` 及 `DirectEnv` 生命周期钩子），底层物理后端按设备内存形态采用各自的原生执行模型：

| 维度 | CPU 后端（`MotrixSimBackend`） | GPU 原生后端（`MotrixCudaBackend`） |
| :--- | :--- | :--- |
| **底层数据形态** | C++ AoS 世界实例内存，跨语言边界访问需聚合拷贝 | Warp / CUDA SoA 批量设备数组（`gpu.SceneData`），天然与 Torch 共享显存 |
| **物理读取模型** | `mtx.query.QueryProgram` 将多字段打包拷贝至单一连续 Host NumPy Arena，暴露只读视图 | **常驻字段直接暴露 `wp.to_torch` 零拷贝设备视图**；仅非连续拼接与派生接触/传感器核写入紧凑持久缓冲，并与 Warp 核一并捕获进单一静态 CUDA Graph |
| **物理写入模型** | `mtx.write.WriteProgram` 持有 Host 暂存缓冲，单次 C++ 调用批量散射并归一化/刷新 FK | 普通写入使用暂存缓冲并保持行隔离。ActionManager 的全批专用通道可直写连续的 `data.actuator_ctrls`；球关节四元数在设备端归一化；执行器增益、连杆质量/质心与几何体滑动摩擦覆盖全部在设备端原位掩码更新并单核重算刚体惯量 |
| **局部 Reset 粒度** | 主机提取 `env_ids = nonzero(done)`（`(K,)`），仅复位 `K` 个世界以节省标量开销 | **固定形状 `world_mask: bool[N]` 原位掩码执行**：单核并行清零被选世界的 DOF/控制与双 bank 求解器缓存并按掩码刷新派生接触读数；纯写入 `ResetTerm(stateless=True)` 与命令掩码钩子合入静态 `CUDAGraph` 直接调用 `execute_masked`，仅在近截断步或存在有状态自定义 `ResetTerm` 时启用主机复位门控 |
| **流与计时边界** | `nullcontext`，CPU 单调时钟直接反映真实耗时 | `physics_stream()` 在控制步边界与调用方 Torch 流做单次进出事件交接；性能剖析可在各计时区间边界同步设备 |

### 2.1 读侧视图生命周期契约

- 前端在每个控制步的 `physics_step()` 之后（`compute_transition`）以及 `reset` 之后（`compute_observation`）刷新并消费仿真读视图；前端不依赖“在 `writes.execute()` 之后、`reads.execute()` 之前旧视图仍保留写前快照”的暂存副作用。
- 由此，GPU 后端对 `gpu.SceneData` 中布局已对齐的连续状态直接返回零拷贝张量视图；仅对需要跨列收集或物理后处理（接触合力矩、几何体碰撞对、Touch 传感器）的查询分配输出缓冲，且所有派生读取 kernel 均接受可选 `world_mask: bool[N]`，局部刷新时跳过未复位世界。MotrixSim GPU 1.0.0 的 FK 入口接收整批数据，写入和接触投影按掩码执行。

### 2.2 写侧与掩码 Reset 契约

- 声明层继续使用 `SimWrite` 与 `ResetTerm` 描述写入目标。
- 执行层支持固定批次宽度 `N` 的布尔掩码 `world_mask: torch.Tensor`（`bool[N]`）：
  - CPU 后端在边界将 `world_mask` 转为紧凑 `env_ids` 调用原生 C++ 批量接口；
  - CUDA Manager 使用掩码复位求解器、动作、命令和观测状态，并按掩码刷新派生读数。对声明 `stateless=True` 的纯输出 `ResetTerm`，执行器在静态 `(N,)` `CUDAGraph` 内求值并通过 `execute_masked(world_mask)` 原位掩码提交；对默认 `stateless=False` 的自定义 `ResetTerm`，经主机门控提取实际复位行的紧凑 `env_ids` 与 `sim_writes`。截断终态在近截断窗口经单次门控以紧凑快照保存。
- 所有对 `motrixsim-gpu` 内部 `.low` 求解器域（双 bank warmstart lambdas、碰撞计数、接触冲量转力矩、增益/参数覆盖）的访问必须收敛封装在 `motrix_env_motrixsim` 单一内部原语模块中，与上层读写编译器解耦，作为未来上游 SDK 原生算子到位后的唯一替换面。
- 模型静态元数据（`SimModel`、`BodyModel`、`KinematicTree`、`HeightFieldGrid`）在 `SimModelCompiler` 编译期直接按目标 `device` 构造张量。

### 2.3 每世界接触预算

任务通过 `SimCfg.max_contacts` 与 `SimCfg.max_contact_constraints` 配置 CUDA 接触缓冲容量。三维摩擦接触每个接触点会展开为 3–4 条单边约束行，因此内置任务按接触复杂度分类预留余量：四足、G1 WBT、平衡球和轻量抓取/插销任务采用 `128 / 384`，台阶四足、常规平地人形行走、柜体抽屉和灵巧手操作任务采用 `256 / 768`，G1 平地/起伏地形行走与常规起伏地形人形任务采用 `512 / 1536`，高密度网格自碰撞人形（Dex-EVT / K1 平地与 WBT）采用 `1024 / 4096`、起伏地形采用 `1536 / 6144`。未指定的配置使用 SDK 默认值。MotrixSim GPU 1.0.0 通过 `SceneModel.from_cmodel` 提供这两项公开设置，凸包候选对容量由 SDK 管理。

定标时用默认环境批量分别检查随机探索、典型跌倒和稳定策略轨迹中的逐世界引擎错误，以及峰值显存。接触预算应为瞬时接触峰值留余量，避免溢出触发异常复位并改变训练样本分布。训练吞吐按稳定训练窗口内实际采集的环境步数除以窗口墙钟时间统计。

---

## 3. RL 训练控制面与异构/同构拓扑对齐

### 3.1 无同步阻塞的控制步与指标归约

- 控制步热路径不得对环境指标（`state.metrics`、`state.reward_terms`）逐项调用 `.item()` 或 `.cpu()`。
- CUDA 仿真下，每步指标保留为设备张量或在设备侧窗口累加器中更新，SKRL Torch 和 FastSAC 在日志窗口结束时批量归约。SKRL Torch 也在需要按回报保存 checkpoint 时读取窗口统计。
- SKRL Torch、SKRL JAX（经 DLPack 零拷贝桥接观测、动作与奖励分解项）、RSLRL 与同步 FastSAC 在用户指定索引卡 `sim_device=cuda:k` 且未显式指定算法侧 `device` 时，默认将训练设备共置到 `cuda:k`。

### 3.2 异步 FastSAC 拓扑与进程隔离

- **父进程零 CUDA 上下文不变量**：异步训练父进程在 `mp.get_context("spawn")` 前仅在 CPU 上探测环境维度并分配主机共享内存游标；所有 CUDA 上下文、Warp 流、`GpuIpcTransitionRing` 槽位与 `FinalObsStorage` 设备缓冲均在子进程完成 CPU/NUMA 亲和性绑定后分配并通过 `StartupHandshake` 交接。[FastSAC 异步训练设计](./fastsac-async-heterogeneous-trainer.md)
- **统一设备拓扑求解**：`resolve_async_trainer_topology` 同时接收 `learner_devices`、`collector_inference_device` 与 `sim_device`，一次性确定每个 Collector 的 `(sim_device, infer_device, owner_learner, use_cuda_ipc)`：

| 仿真设备请求 | 拓扑放置与传输行为 |
| :--- | :--- |
| `"cpu"` | 沿用 NUMA 本地 CPU 亲和性绑定；推理按 `collector_inference_device` 执行，经主机环或同卡 IPC 环送往所属 Learner。 |
| `"cuda"`（未指定卡号） | 每个 Collector 的仿真分片绑定到所属 Learner 所在 GPU；若推理也在该卡则全程设备内零主机中转（CUDA IPC Ring）。 |
| `"cuda:k"`（显式指定卡号） | 所有 Collector 仿真分片固定到 `cuda:k`；单 Learner 未显式绑卡时自动共置到 `cuda:k`，跨卡或主机环模式下 Collector 内部缓冲仍驻留 `cuda:k`，仅在跨设备边界做批量传输。仿真、推理和 Learner 同卡时使用 IPC，否则使用主机环。 |

### 3.3 运行记录、续训与回放

- `metadata.json` 记录实际解析后的 `sim` 与 `sim_device`（异步多卡同时记录 `collector_sim_devices`）。
- `mxlab play` 与 `mxlab train resume=...` 在命令行未显式传入 `sim` / `sim_device` 时，均优先继承原运行记录中的 `sim` 与 `sim_device`；旧运行记录缺失该字段时回退环境默认值。
