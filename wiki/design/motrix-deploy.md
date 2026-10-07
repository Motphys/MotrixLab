# Motrix Deploy 框架设计

## 摘要

本文定义 `motrix_deploy` 的框架边界、公共契约、deployment artifact、统一控制循环，以及 MuJoCo sim2sim 与 Unitree Go2
sim2real vertical slice。`motrix_deploy` 是独立于训练框架的策略部署包：训练侧先通过统一的
[ONNX policy exporter](./onnx-export.md) 把 metadata-backed run 转换为经过 parity 验证的自包含 ONNX，
再由 `motrix_envs.deploy` compiler 补全 robot、task 和 control 契约并写出 artifact。部署侧使用带版本的
`DeployTask`、`PolicyRuntime` 和通用控制循环驱动仿真或真实机器人。运行时 command input 的
`InputDevice`、公共 `PlanarVelocityCommand` 和 `CommandBinding` 契约由独立的
[Deploy Runtime Command Input 分层设计](./deploy-command-input.md) 定义。sim2sim 与 sim2real 只替换
`RobotInterface`，不得分别维护 policy tensor 拼接、command mapping 或 action 后处理逻辑。

当前实现包含公共框架、MuJoCo 与原生 MotrixSim plugin，以及 Unitree SDK2 DDS plugin；Unitree 链路已通过注入式假 SDK 验证，
吊架真机 smoke test 仍待执行。当前 `motrix-deploy/v1` 固定 Go2 locomotion vertical slice 所需的单输入、单输出和
单关节控制组契约；面向多型号机器人与多任务的 general manifest 演进方向见 8.4 节。

## 1. 背景与架构定位

现有 `scripts/play.py` 从 run metadata 和 checkpoint manifest 发现训练产物，再创建对应 RL framework 的
trainer 进行回放。这个入口适合在训练环境内评估策略，但它仍依赖 SKRL、RSLRL 或 FastSAC 的模型与环境
封装，不能作为稳定的跨仿真器或真机部署接口。

`motrix_deploy` 在训练和设备之间建立一个独立边界：

```text
motrix_rl checkpoint + run metadata + task snapshot
                         |
                         | unified ONNX policy export
                         v
               validated ONNX + tensor report
                         |
                         | deployment profile compilation
                         v
                deployment artifact
                         |
                         v
 CommandBinding.read_command(batch_size=1) -> PlanarVelocityCommand
                                                       |
                                                       v
                                                  PolicyContext
                                                       |
     RobotState ----------------------------+----> DeployTask.build_observation
                                                   |
                                                   v
                          DeployTask -> PolicyRuntime -> DeployTask.process_action
                               ^                                      |
                               |                                      v
                               +----- RobotInterface <---------- RobotCommand
                                             |
                               +-------------+-------------+
                               |                           |
                        MuJoCoInterface              VendorInterface
                           (sim2sim)                    (sim2real)
```

`scripts/play.py` 继续承担训练框架内 playback，不改名、不转化为部署入口。两条路径可以消费相同的源
checkpoint，但职责不同：play 验证训练实现，deploy 验证可移植 artifact 和设备控制链路。

## 2. 设计目标

1. 新增可独立安装和导入的 `motrix_deploy` workspace package，核心运行时不依赖任何 RL framework。
2. 定义稳定的机器人状态、命令、设备生命周期、任务、策略执行和动作处理契约。
3. deployment artifact 完整记录模型、机器人、版本化 task 标识及 config、坐标系和控制频率，运行时不猜测文件名、
   目录结构或 tensor 语义。
4. sim2sim 与 sim2real 使用同一个控制循环、artifact 和 `DeployTask` 实现。
5. backend 只负责设备生命周期以及 `RobotState` / `RobotCommand` 与底层 SDK 或仿真器之间的转换。
6. artifact、模型元数据与 backend 配置的静态不匹配在对应边界 fail fast；实际 observation/action 数组在 policy/task 边界校验，运行期超时、非有限数值和 backend 故障产生明确的停止原因。
7. backend 和 policy runtime 保持通用；`motrix_deploy` 通过 `motrix_deploy.tasks` entry-point group 发现具体任务插件，按需加载，不修改主循环。
8. MuJoCo sim2sim 支持确定性初始化、无界面运行、基础指标和可判定的进程退出状态。
9. deploy 固定使用 keyboard binding，训练 task 使用自己的 random binding；二者都产生公共
   `PlanarVelocityCommand`，Go2 task 不感知 command 的来源。

## 3. 非目标

- 不在自动化测试或开发环境连接真实硬件；真机验证必须在吊架和现场急停条件下执行。
- 第一阶段不支持仓库中所有 robot、task、policy format 和控制模式。
- 不把训练环境的 reward、termination、curriculum 或 domain randomization 搬入部署核心。
- 不让 backend 直接读取 policy tensor、拼 observation 或解释 raw action。
- 不把 ROS 2 作为核心依赖；后续可将其实现为 `InputDevice` adapter。
- 不要求 deployment artifact 打包完整仿真场景；仿真模型和场景参数属于 backend 配置，但必须与 artifact
  中的机器人契约一致。
- 不在本设计中规定网络训练或 checkpoint 内部格式；相关转换由 `motrix_rl` 的统一 ONNX policy exporter 及其
  framework-owned adapter 负责，deployment profile compiler 和 `motrix_deploy` 均不解析 checkpoint 内部结构。

## 4. 核心原则与不变量

### 4.1 依赖方向单向

核心依赖方向为：

```text
motrix_env_core <- motrix_deploy <- motrix_deploy_tasks
motrix_deploy <- motrix_rl
motrix_deploy <- motrix_envs deploy integration
```

`motrix_deploy` 依赖 `motrix_env_core.input` 中的 device、公共 command 和 binding contract，但不依赖
`motrix_envs` 或 `motrix_rl`。`motrix_deploy_tasks` 依赖 `motrix_deploy`，Go2 task 直接消费公共
`PlanarVelocityCommand`。训练侧 integration 可以同时导入
`RobotCfg` 和 `motrix_deploy.artifact`，将训练时的机器人配置解析为自包含 `RobotSpec`；部署运行时只读取
artifact 中的 RobotSpec，而非重新推导控制契约。仿真应用单独通过现有 robot registry 选择模型并组装场景。
`motrix_deploy.cli` 读取 artifact、通过 task 插件发现选择具体实现并装配 runtime；
`motrix_deploy_tasks` 提供具体 task 类与已安装 Hydra recipe/search-path plugin，不反向承担 core CLI bootstrap。
Hydra 配置搜索发现与版本化 Python task 实现发现互相独立。
deployment export integration 应调用 `motrix_deploy.artifact` 的写入
API；policy checkpoint 只由 `motrix_rl.deploy` 的 framework-owned adapter 解析，不能交给部署包或 profile
compiler 处理。

artifact 生成侧进一步保持单向依赖：

```text
motrix_rl framework adapter -> motrix_rl.deploy -> motrix_deploy.artifact
       checkpoint restore       ONNX + report       validate / write
                                      ^
                                      |
                           motrix_envs.deploy compiler
                             environment semantics
```

`motrix_rl.deploy` 复用现有 `RlFramework` / `AgentProvider` 发现具体 policy exporter。deployment profile compiler
只按稳定的 task/profile/schema 选择运行时语义，不按 `rllib`、train backend 或 algorithm 再建立一套 exporter
registry。manifest 中的 framework 来源只用于追踪，不能成为部署 runtime 或 profile 选择的分派键。

`RobotCfg` 与 `KeyPoseCfg` 定义在 core；`motrix_robots` 独立提供仿真模型配置、模型自有关键姿态及完整
MJCF/URDF、mesh 和纹理资产，仅依赖 core。Unitree G1、Go1、Go2 与其他内置模型均在所属模型类中内联声明
`KeyPoseCfg`，不使用跨仿真与硬件的共享姿态 factory。
`motrix_deploy_unitree.hardware` 定义 `HardwareRobotCfg` 与直接声明 Go2 自有 `KeyPoseCfg` 的
`UnitreeGo2HardwareCfg`；`.sensor` 定义 `@configclass` 的 `HardwareSensorBindings` 与 frozen dataclass
`HardwareSensorBinding`，`.actuation` 定义 `Actuation` 与 `UnitreeMotorBinding`。硬件配置导入独立于模型和资产，
不会提前加载 simulator 或 Unitree SDK。模型与硬件默认姿态不要求相同；artifact 的关节顺序、默认姿态和限幅仍由
policy 契约拥有，硬件接线和 shutdown pose 不替代该契约。

### 4.2 关节顺序只有一个权威来源

artifact 中的 `robot.joint_names` 是 policy 与部署运行时的规范顺序。`RobotState`、`RobotCommand`、action
scale、默认姿态、限幅和 gains 都按该顺序排列。

backend 在 `open()` 阶段按名称建立一次映射，之后控制循环只使用预计算索引。名称重复、缺失、多余或
DoF 不兼容时禁止启动，不能按底层数组当前位置静默对齐。

### 4.3 单位和坐标系显式化

公共契约默认使用 SI 单位：弧度、弧度每秒、牛顿米、米、米每秒和秒。四元数顺序、旋转方向、角速度
表达坐标系、重力投影约定等必须记录在 artifact 中，并在加载时与 backend capability 校验。

第一版规范使用：

- 四元数顺序：`xyzw`；
- base orientation：从 body frame 到 world frame 的旋转；
- IMU angular velocity：body frame；
- base linear velocity：按具体 task version 明确选择 world 或 body frame；
- 时间戳：单调时钟纳秒，不使用 wall clock 计算状态新鲜度。

### 4.4 控制循环不包含 backend 分支

主循环不得出现 `if backend == "mujoco"` 或 `if real_robot`。backend 差异由 `RobotInterface` 和 capability
校验吸收。真机额外的 enable、安全确认和急停也通过接口生命周期与 safety policy 表达，而不是复制一套
inference loop。

### 4.5 artifact 是运行时事实来源

运行时不能根据 checkpoint 后缀猜模型格式，不能从环境类名推断 observation，不能从数组长度猜关节顺序，
也不能在缺字段时用训练配置默认值补齐。旧 schema 只有在存在显式迁移器时才能加载。

### 4.6 输入设备、任务命令和映射严格分层

command input 必须遵循 [独立分层设计](./deploy-command-input.md)：device 只产生原始输入，command dataclass 只表达
高层任务目标，binding 负责把输入机制转换成 command。当前 deploy 固定创建 keyboard binding；带 standing
probability 的 random binding 由训练 task 持有。

## 5. 包与模块划分

第一阶段采用以下逻辑结构；实现时可在不改变公共边界的前提下合并过小的内部模块：

```text
motrix_env_core/
└── src/motrix_env_core/
    └── input/
        ├── __init__.py     # 稳定公共入口和 re-export
        ├── device.py       # InputDevice、KeyboardDevice、GamePadDevice
        ├── command.py      # PlanarVelocityCommand
        └── bindings.py     # CommandBinding 与内置 bindings

motrix_deploy/
├── pyproject.toml
├── src/motrix_deploy/
│   ├── artifact/           # schema、reader/writer、checksum、静态校验
│   ├── contracts.py        # RobotSpec、RobotState、RobotCommand、capability
│   ├── profile.py          # DeploymentProfile、compiler registry 和统一 build 入口
│   ├── task.py             # DeployTask 接口与版本化 task entry-point discovery
│   ├── env.py              # robot-free 世界 discovery 与 assemble_deploy_scene(env ID, robot ID)
│   ├── policy/             # PolicyRuntime 与 runtime registry
│   ├── runtime/            # PolicyContext、lifecycle、control loop、scheduler、result/metrics
│   │   ├── base.py         # DeploymentRuntime / SimulationRuntime、bind_control_session
│   │   ├── control.py      # ControlSession：共享 start/tick/stop
│   │   ├── hardware.py     # HardwareRuntime：墙钟 scheduler
│   │   └── factory.py      # motrix_deploy.backends discovery；返回 prepared runtime
│   ├── robot/              # robot I/O 与可选 input-device capability；精简 initializer
│   │   ├── interface.py    # RobotInterface、KeyboardDeviceProvider、GamePadDeviceProvider
│   ├── config/             # deploy.yaml：task: null（通用帮助）；inspect.yaml：artifact 检查
│   └── cli.py              # 单一 Hydra 部署入口、inspect 子命令与 runtime 装配
└── tests/

motrix_deploy_mujoco/
├── src/motrix_deploy_mujoco/
│   ├── runtime.py          # MujocoRuntime：世界资源、模型编译与 physics loop
│   ├── interface.py        # MuJoCo RobotInterface：读取状态、缓存控制目标
│   ├── transform.py        # position actuator -> torque motor MjSpec transform
│   └── plugin.py           # motrix_deploy.backends entry point
└── tests/

motrix_deploy_unitree/
├── src/motrix_deploy_unitree/
│   ├── hardware.py         # HardwareRobotCfg / UnitreeGo2HardwareCfg；硬件自有 KeyPoseCfg
│   ├── sensor.py           # HardwareSensorBindings configclass / frozen HardwareSensorBinding
│   ├── actuation.py        # Actuation / UnitreeMotorBinding
│   ├── interface.py        # Unitree SDK2 LowState/LowCmd DDS adapter
│   ├── remote.py           # 遥控器按键与摇杆解码
│   └── plugin.py           # unitree_go2 backend entry point
└── tests/                 # 注入式假 SDK 安全与映射测试

motrix_deploy_tasks/
├── src/motrix_deploy_tasks/
│   ├── tasks/
│   │   └── go2_walk.py     # Go2WalkTaskSpec 与 Go2WalkDeployTaskV1 同置
│   ├── envs/               # flat.py / rough.py：独立、robot-free 部署世界
│   ├── config/task/        # go2-walk-flat/{sim,hardware}.yaml、go2-walk-rough/sim.yaml
│   └── __init__.py         # 精简 package initializer；task 类由 entry point 声明
├── src/hydra_plugins/
│   └── motrix_deploy_tasks_searchpath.py  # 将 pkg://motrix_deploy_tasks.config 加入 Hydra 搜索路径
└── tests/

motrix_envs/src/motrix_envs/deploy/
├── go2_walk.py             # Go2 EnvCfg -> DeploymentProfile，并向 motrix_deploy 注册
└── robot.py                # RobotCfg -> RobotSpec
```

依赖按能力分组：

- core：`motrix_env_core`、`numpy`、typed task spec 校验与 codec 所需的 Pydantic、应用 YAML 加载/合并/解析所需的 OmegaConf，以及 artifact 解析所需的标准库；
- `onnx` extra：`onnxruntime`，只在创建 ONNX policy runtime 时导入；
- `motrix-deploy-mujoco`：独立安装的 backend plugin，依赖 SceneCfg compiler 与 MuJoCo Python package；
- `motrix-deploy-tasks`：直接依赖并复用 `motrix_env_core.input`，Go2 不实现专用 binding；
- vendor SDK：由外部 adapter package 自己声明，不能进入 core 依赖。

安装 core 后 `import motrix_deploy` 不应触发 ONNX Runtime、MuJoCo、训练框架或厂商 SDK 的 import。

## 6. 公共运行时契约

运行时数据对象使用普通 `@dataclass`；typed task 配置使用 Pydantic `TaskSpec`，不使用面向声明式环境配置的 `@configclass`。

### 6.1 RobotSpec

`RobotSpec` 是 artifact 内自包含、经过验证的机器人运行时契约。它由 deployment profile compiler 从当时的
`RobotCfg`、编译模型和 deployment profile 生成，但定义和加载过程不依赖这些来源 package：

```python
@dataclass
class RobotSpec:
    base_link_name: str
    joint_names: tuple[str, ...]
    default_joint_position: np.ndarray
    position_lower: np.ndarray
    position_upper: np.ndarray
    torque_limit: np.ndarray
```

它只包含控制循环和 backend 校验所需的稳定语义，不持有仿真 model、`RobotCfg` 或 SDK handle。
数组形状必须等于 joint count，数值必须有限，lower 不得大于 upper。训练侧无法从 `RobotCfg` 或
编译模型获得的电机限制必须由 deployment profile 显式提供，不能猜测。

### 6.2 RobotState

```python
@dataclass
class RobotState:
    sample_time_ns: int
    receive_time_ns: int
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    base_orientation_xyzw: np.ndarray
    base_angular_velocity: np.ndarray
    base_linear_acceleration: np.ndarray
    base_position: np.ndarray | None = None
    base_linear_velocity: np.ndarray | None = None
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)
```

- joint 数组使用 `RobotSpec.joint_names` 顺序。
- `sample_time_ns` 表示底层采样时间；没有可靠设备时钟时可等于本机接收时间。
- `receive_time_ns` 使用本机单调时钟，用于 timeout 判断。
- optional 字段只有在 observation 或 metric 需要时才成为 backend 必需能力。
- `extras` 用稳定名称承载足底力、里程计等扩展；artifact 必须声明实际消费的 key、shape、dtype 和单位。

### 6.3 Command input

机器人状态不是 observation 的唯一输入。Go2 速度跟踪策略还需要平面目标速度。runtime 使用以下单向数据流：

```text
CommandBinding.read_command(batch_size=1) -> PlanarVelocityCommand -> PolicyContext -> DeployTask
```

最小 device/binding 接口、包归属、配置和测试边界统一由
[Deploy Runtime Command Input 分层设计](./deploy-command-input.md) 定义。本框架只要求 `ControlLoop` 调用已经装配的
binding，不直接读取 device，也不解释 `PlanarVelocityCommand` 字段。公共 command 始终携带显式 batch 维；当前
deploy runtime 只接受 `values.shape == (1, 3)`，Go2 task 校验 batch size 后取出唯一一行。

### 6.4 RobotCommand

```python
@dataclass
class RobotCommand:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    feedforward_torque: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
```

所有数组均使用规范 joint order。第一版统一表达关节 PD + feed-forward torque：

```text
tau = kp * (q_target - q) + kd * (qd_target - qd) + tau_ff
```

backend 可以把该命令传给原生 hybrid controller，也可以在本地计算 torque，但必须声明相应 capability。
`DeployTask.process_action()` 在写入 backend 前完成 action 和目标 position 裁剪，并生成显式 gain。backend 仍需执行最后
一道设备级硬限制，不能扩大 artifact 的限幅。

### 6.5 RobotInterface 与 capability

`motrix_deploy.robot.interface` 定义机器人 I/O 边界及可选输入设备 capability；它不是 runtime 插件 discovery 模块。
`RobotInterface` 绑定 `RobotSpec`，负责 state read、command write、health 与机器人生命周期。
`KeyboardDeviceProvider` 和 `GamePadDeviceProvider` 是可选 protocol，提供由 provider 管理生命周期的输入设备，
不承担 runtime factory 职责。测试目录中的 `FakeRobotInterface` 提供确定性内存实现，不作为运行时 adapter 发布。
`motrix_deploy.robot` 的 initializer 保持精简，使用方从定义模块导入：

```python
from motrix_deploy.robot.interface import GamePadDeviceProvider, KeyboardDeviceProvider, RobotInterface
```

```python
class RobotInterface(ABC):
    @property
    def capabilities(self) -> RobotCapabilities: ...

    @property
    def spec(self) -> RobotSpec: ...

    def open(self) -> None: ...
    def enable(self) -> None: ...
    def read_state(self, timeout_s: float) -> RobotState: ...
    def write_command(self, command: RobotCommand) -> None: ...
    def health(self) -> HealthStatus: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
```

`RobotCapabilities` 至少声明：

- 支持的控制模式；
- 可用 state 字段和 extra sensors；
- 是否支持渲染；
- 最大推荐 command rate；
- 是否需要显式 enable，以及 stop 的安全语义。

仿真 backend 在 `open()` 中完成自身的确定性初始化；真机 backend 建立通信后由 `read_state()` 返回当前状态。
`enable()` 不接收 initial command；Unitree backend 自行构造到 `RobotSpec.default_joint_position` 的姿态过渡，
目标速度与 feedforward torque 为零，使用解析后的 `runtime.kp` / `runtime.kd`，`null` 时保留 artifact 增益。
不通过 task action 推导启动命令。`ControlSession.start()` 不调用 task `process_action()` 生成零动作启动命令；
初始状态的 safety/termination 检查仍先于 enable，Start/A 门控与 enable 后新状态检查保持有效。
`stop()` 和 `close()` 必须幂等，任何启动后异常都按 `stop -> close` 收尾。`HealthStatus` 包含状态、原因和最后
成功通信时间，不能只返回无语义的 bool。

### 6.6 DeployTask

```python
class DeployTask(ABC, Generic[CommandT]):
    spec_type: ClassVar[type[TaskSpec]]

    def reset(self, state: RobotState, context: PolicyContext[CommandT]) -> None: ...
    def build_observation(self, state: RobotState, context: PolicyContext[CommandT]) -> np.ndarray: ...
    def process_action(self, action: np.ndarray) -> RobotCommand: ...
    def validate_command(self, command: CommandT) -> None: ...
```

artifact 的 task wire format 只记录版本化 task 标识和 config；内存中的 `TaskSpec` 是具体任务的 Pydantic model，
其 codec 与 class discovery 见 [Manifest 内容](#82-manifest-内容)。observation/action 维度只记录在 `policy.input/output.shape`。
`DeployTask` 不声明 `observation_size` 或 `action_size`；实际 observation 数组由 policy 输入边界校验，
实际 action 数组由 task 的 `process_action()` 边界校验，CLI 不提前比较 task 维度。具体 observation 顺序、
previous action、phase 与 action 后处理算法由 `motrix_deploy_tasks` 中一个带版本的 task 类直接实现，不在 manifest
中构造通用计算 DSL。修改已有语义必须新增 task version，不能静默改变旧实现。部署时关闭训练 observation noise；
训练框架的 observation normalizer 必须烘焙进 ONNX。

### 6.7 PolicyRuntime

```python
class PolicyRuntime(ABC):
    def reset(self) -> None: ...
    def infer(self, observation: np.ndarray) -> np.ndarray: ...
```

第一版内置 `OnnxPolicyRuntime`，使用确定性 action 输出。runtime 创建时验证模型输入输出名称、dtype 和 shape；
每次 `infer()` 验证输出无 NaN/Inf。接口保留 `reset()` 以支持未来 recurrent policy，但第一版 artifact 只接受
单输入、单输出、无隐藏状态的 ONNX policy。

`DeployTask.build_observation()` 的输出是训练环境的原始 actor observation。训练框架的 observation normalizer 和 deterministic
policy head 必须烘焙进 ONNX，不能在 runtime 外保留 framework-specific preprocessor；不满足自包含约束的模型
拒绝导出为第一版 artifact。

### 6.8 Task-specific action

第一版 `go2_walk/v1` 在自己的 `DeployTask` 中实现 joint-position target：raw action 经过 finite check、clip、
per-joint scale 和 default pose offset，再应用 position limit，生成零目标速度、零 feed-forward torque 和配置的
KP/KD。task 同时保存 previous action；sim2sim 与 sim2real 必须复用同一版本化 task 标识及 config。

## 7. 统一控制循环

运行时生命周期为：

```text
CREATED -> VALIDATED -> OPEN -> READY -> RUNNING -> STOPPING -> CLOSED
                                      \-> FAILED -> STOPPING -> CLOSED
```

`ControlLoop` 接收 application 已装配的 `CommandBinding`。device 的 provider-specific 创建和资源释放不进入
control-loop contract；runtime 只调用 `read_command(batch_size=1)`，具体 device-backed binding 在内部为
event-based device 执行一次 `poll()`。`ControlLoop.run()` 的单步顺序固定为：

```text
1. LoopScheduler 到达当前 tick deadline
2. CommandBinding.read_command(batch_size=1) 产生 PlanarVelocityCommand，并构造经过 task 校验的 PolicyContext
3. 从 RobotInterface 读取 RobotState，检查 backend health、状态时效、shape 和 finite values
4. DeployTask.build_observation(state, context)
5. PolicyRuntime.infer(obs)
6. DeployTask.process_action(action)
7. safety limiter 验证 RobotCommand，RobotInterface.write_command(command)
8. 记录 input / binding / read / obs / inference / action / write / loop timing
```

`ControlLoop` 接收 `LoopScheduler`，但不判断当前是否为仿真：

- `RealtimeScheduler` 使用绝对 monotonic deadline，避免每步相对 sleep 累积漂移，用于真机和 viewer sim2sim；
- `FixedStepScheduler` 每次推进 artifact 定义的固定 `dt`，不执行 wall-clock sleep，用于默认 headless sim2sim
  和确定性 CI。

两种 scheduler 向 `PolicyContext` 提供相同的 step 和 elapsed control time；wall-clock latency 单独计量。
仿真的 realtime/fixed-step scheduler 由 `runtime.realtime` 选择；`null` 跟随 `runtime.viewer`。硬件始终在内部实时调度，不从 device 推断。
Fixed-step 模式记录“若实时运行是否 overrun”，但默认不因此终止 rollout。第一版 policy rate 与 command rate
相同；未来需要多速率时通过 scheduler 扩展，不能由 backend 私自重复或插值 policy action。

以下情况立即进入失败停止流程：

- state timeout 或 timestamp 倒退；
- backend health 不是 healthy；
- observation、policy output 或 command 出现 NaN/Inf；
- policy shape、joint mapping 或 capability 在运行中改变；
- 启用 realtime 约束时，loop overrun 连续超过配置阈值；
- backend 读写异常或用户触发停止；
- binding/device 异常或无效的 `PlanarVelocityCommand`。

`RolloutResult` 记录结构化 `exit_reason`、完成步数、仿真时间、wall time、real-time factor、overrun 数量，以及
各阶段延迟的 min/mean/max/percentile。正常达到 step/duration 上限才返回 success；安全停止、数值错误和 backend
错误返回非零 CLI 状态。

## 8. Deployment artifact

### 8.1 目录结构

第一版 artifact 是一个不可变目录：

```text
go2_walk.deploy/
├── manifest.json
└── policy/
    └── model.onnx
```

`manifest.json` 中的文件路径只能是 artifact root 下的相对路径，禁止绝对路径和 `..`。所有载荷记录 SHA-256。
压缩包分发可以后续增加，但解包后必须得到相同逻辑结构。

### 8.2 Manifest 内容

manifest 至少包含以下部分。下面是结构节选，为便于阅读省略了长数组，不是可直接加载的完整文件：

```json
{
  "schema_version": "motrix-deploy/v1",
  "source": {
    "framework": "rslrl.ppo/torch",
    "run_id": "26-08-02_17-48-23-164697",
    "checkpoint": "checkpoints/latest.pt"
  },
  "policy": {
    "component_version": "onnx/v1",
    "payload_path": "policy/model.onnx",
    "sha256": "...",
    "input": {"name": "obs", "dtype": "float32", "shape": [1, 49]},
    "output": {"name": "actions", "dtype": "float32", "shape": [1, 12]}
  },
  "robot": {
    "joint_names": ["FL_hip_joint"],
    "default_joint_position": [],
    "position_lower": [],
    "position_upper": [],
    "torque_limit": []
  },
  "task": {
    "name": "go2_walk/v1",
    "config": {
      "action_scale": [],
      "action_lower": [],
      "action_upper": [],
      "kp": [],
      "kd": [],
      "gait_frequency_hz": 2.0,
      "feet_phase_offsets": [0.0, 0.5, 0.5, 0.0],
      "command_lower": [],
      "command_upper": [],
      "command_scale": [0.5, 0.5, 0.5]
    }
  },
  "control": {
    "period_s": 0.02,
    "state_timeout_s": 0.1,
    "quaternion_order": "xyzw",
    "base_orientation": "body_to_world",
    "angular_velocity_frame": "body"
  }
}
```

示例中的长数组被省略，只表达 schema 形状；真实 manifest 不允许使用空数组代替必要值。task config 只保存
该次训练可能变化且部署必须复现的数值；observation/action 算法固定在对应版本化 task 标识的直接实现中。

`schema_version` 管理 manifest 结构；版本化 task 标识管理具体任务语义。wire format 保持
`task: {name, config}`，内存中则使用具体 typed spec，而非 dict config：

- `motrix_deploy.artifact.schema.TaskSpec` 基于 Pydantic `BaseModel`，使用严格类型、`extra="forbid"`、
  frozen 字段赋值和有限数值约束；提供公共 `to_dict()` / `from_dict()` codec。具体 spec 声明
  `task_name: ClassVar[str]`，通过 `Annotated` / `Field` 约束字段，通过 `model_validator` 约束跨字段关系。
  构造与 `model_validate()` 自动校验，不另设手写 `validate()` 或 dataclass `__post_init__`。
- `Go2WalkTaskSpec` 与 `Go2WalkDeployTaskV1` 同置于 `motrix_deploy_tasks.tasks.go2_walk`；task 类通过
  `spec_type = Go2WalkTaskSpec` 关联配置类型。`action_scale` 是按 canonical joint order 排列的
  `list[float]` 向量；训练侧 compiler 在导出时把 scalar scale 展开为逐关节向量，codec 不接受 scalar fallback。
- manifest 读取按 `task.name` 加载唯一 `motrix_deploy.tasks` entry point 指向的 task **类**，不构造 runtime task，
  再按其 `spec_type.model_validate()` 解码、校验具体 model。只使用这一组 task entry points，不增加独立 spec entry point，
  也不保留原始 dict fallback。未知或重复插件、未知字段、缺少必需字段和非法类型在读取边界拒绝。

Frozen 只阻止字段赋值，不使嵌套 list 不可变；`to_dict()` 在序列化边界重新校验当前字段内容。
Artifact codec 不解析插值。应用侧继续通过 OmegaConf 加载、合并 YAML，并使用
`OmegaConf.to_container(cfg, resolve=True)` 得到普通字典，再传给具体 `Spec.model_validate()`。

控制循环只消费已构造的 task；具体 task 从 typed spec 字段读取参数，不直接读取原始 dict。

### 8.3 Artifact 生成边界

artifact 生成分为三个职责独立的阶段：

1. **Policy export**：`motrix_rl.deploy` 读取 `metadata.json`、`task_config.yaml` 和
   `checkpoints/manifest.json`，通过 `RlFramework` / `AgentProvider` 选择 framework-owned adapter，把具体
   checkpoint 转换为经过 ONNX Runtime parity 验证的自包含 ONNX，并返回实际 input/output tensor contract。
2. **Deployment profile compilation**：`motrix_envs.deploy` 根据稳定的环境 compiler 解析当次训练的
   task snapshot、`RobotCfg` 和编译模型，生成 `RobotSpec`、`TaskSpec` 和 `ControlSpec`；`motrix_rl.deploy`
   通过注入的 profile builder 将 policy tensor contract 与编译环境的 observation/action size 做交叉校验。
   这些维度仅作为 `DeploymentProfile.observation_size` / `action_size` 的导出期运行时数据，不序列化到 `TaskSpec`。
3. **Artifact writing**：`motrix_deploy.artifact` 校验完整 manifest，复制 policy payload、计算 checksum，并原子
   写入最终 artifact 目录。

现有 [ONNX 模型导出设计](./onnx-export.md) 负责第一阶段。它已经统一 RSL-RL、SKRL 和 FastSAC 等训练来源，
deployment export 不得再按 `(rllib, train_backend, algo)` 实现 `export_rslrl_*`、`export_skrl_*` 等平行编排。
新增训练 backend 只增加对应的 ONNX adapter，不修改 profile compiler、artifact reader 或部署控制循环；新增
task/robot/action 语义时，在 `motrix_envs.deploy` 增加 profile compiler，在 `motrix_deploy_tasks` 增加 task 实现，
compiler 通过 import-time 注册接入 profile registry，并直接构造该 task 的具体 spec；task 类通过唯一的
`motrix_deploy.tasks` entry point 接入，如
`"go2_walk/v1" = "motrix_deploy_tasks.tasks.go2_walk:Go2WalkDeployTaskV1"`，由类上的 `spec_type` 同时支持
manifest 解码和运行时构造，不另设 spec 插件。

组合导出接口直接消费经过验证的 ONNX model bytes 和 `OnnxExportReport`。独立导出 `.onnx` 的用户入口继续
原子落盘，但 deployment artifact 生成不依靠临时文件在两个阶段之间传递 policy。ONNX exporter 的动态
batch contract `(None, obs_dim) -> (None, action_dim)` 在 profile compilation 阶段校验；当前 v1 runtime 明确固定
为单实例 batch 1，并在 manifest 中记录 `(1, obs_dim) -> (1, action_dim)`，不得静默 reshape 其他 batch 语义。

ONNX 中只记录模型来源和 tensor metadata。版本化 task 标识及 config、关节顺序、坐标系和控制频率仍属于
deployment manifest；单独的 `.onnx` 文件不是可运行 deployment artifact。

### 8.4 面向多机器人、多任务的 General Manifest 演进

#### 8.4.1 当前 v1 的适用范围

`motrix-deploy/v1` 是已经验证的 locomotion vertical slice，而不是所有机器人任务的最终统一结构。它有以下
顶层假设：

| v1 假设 | 对后续场景的限制 |
|---|---|
| policy 只有一个输入和一个输出 | 不适合 recurrent policy、视觉输入和多个 action head |
| robot 是一个扁平 joint vector | 不便表达腿、手臂、夹爪、轮子等不同资源和控制模式 |
| observation 只有一个拼接向量 | 不适合 image、point cloud、reference motion 等独立输入 |
| action 只有一个 processor | 不支持腿用 torque、手臂用 position、夹爪用独立位置命令 |
| command spec 由具体 task version 与 config 解析 | artifact 尚未显式自描述跨 task command interface |
| runtime 只有一个 control period | 不支持 policy、低层控制和 command input 的多频率调度 |

general manifest 的目标不是允许任意 JSON，也不是在 manifest 中实现一个通用计算图语言，而是提供严格、稳定
的编排外壳，把机器人和任务差异放在带版本的 component config 中。逻辑边界分为：

```text
DeploymentManifest
├── Artifact/Payload         模型载荷、checksum 和训练来源
├── RobotSpec                机器人资源组、frame、限制和控制能力
├── TaskInterfaceSpec        command、observation pipeline、action pipeline
├── PolicyInterfaceSpec      runtime、多输入、多输出和显式 policy state
└── RuntimeRequirements      时钟、timeout 和 backend capability
```

#### 8.4.2 RobotSpec 使用资源组

未来 RobotSpec 应从单个扁平 joint vector 扩展为命名资源组：

```json
{
  "robot": {
    "spec_version": "motrix.robot/v1",
    "id": "mobile-manipulator-a",
    "base_frame": "base",
    "groups": {
      "base": {
        "resource_type": "wheel",
        "names": ["left_wheel", "right_wheel"],
        "supported_command_modes": ["joint_velocity"]
      },
      "arm": {
        "resource_type": "joint",
        "names": ["joint_1", "joint_2"],
        "default_position": [],
        "position_lower": [],
        "position_upper": [],
        "effort_limit": [],
        "supported_command_modes": ["joint_position", "joint_pd", "joint_torque"]
      },
      "gripper": {
        "resource_type": "joint",
        "names": ["gripper_joint"],
        "supported_command_modes": ["joint_position"]
      }
    },
    "frames": {
      "base": {"parent": "world"},
      "tool": {"parent": "arm_link_end"}
    }
  }
}
```

每个 group 内的名称顺序仍是唯一 canonical order。Go2 等单控制模式机器人可自然表示为一个 `legs` group；
多部件机器人可以让每个 action pipeline 绑定不同 group 和 command mode。

仿真配置与资产由独立的 `motrix_robots` package 提供；general manifest 仍不能只保存可变的 registry name。应同时
保存 package/name/version 引用和 resolved RobotSpec snapshot，并对 snapshot 计算 checksum：引用用于发现和
一致性检查，snapshot 用于离线部署和历史复现。

#### 8.4.3 Command 是一等 task 接口

task command 不再隐藏在特定 observation term 中，而是先定义命名、类型、单位、frame 和范围。以下 schema
描述单个 batch element；runtime batch 维由 command contract 单独约束：

```json
{
  "command": {
    "command_type": "planar_velocity",
    "api_version": "v1",
    "fields": {
      "linear_velocity_x_mps": {
        "dtype": "float32",
        "shape": [],
        "unit": "m/s",
        "frame": "base",
        "lower": -1.0,
        "upper": 1.0
      },
      "linear_velocity_y_mps": {
        "dtype": "float32",
        "shape": [],
        "unit": "m/s",
        "frame": "base",
        "lower": -0.5,
        "upper": 0.5
      },
      "yaw_rate_rad_s": {
        "dtype": "float32",
        "shape": [],
        "unit": "rad/s",
        "frame": "base",
        "lower": -1.0,
        "upper": 1.0
      }
    }
  }
}
```

observation pipeline 只引用 `task.command.linear_velocity_x_mps` 等稳定 signal。artifact 记录的是 command schema，
不记录部署现场的 keyboard 状态；input mechanism 与 binding 属于 runtime。当前 deploy 固定使用 keyboard binding，
它产生的 task command 不改变 task observation 的计算语义。

#### 8.4.4 Observation 和 Action 使用命名 Pipeline

一个 artifact 可以声明多个命名 observation pipeline：

```text
observations.proprioception
observations.front_camera
observations.reference_motion
observations.actor
```

每个 pipeline 仍由版本化组件实现，例如 `term_pipeline/v1`、`image_preprocess/v1` 或
`reference_motion/v1`。term pipeline 是有序的 typed transform 列表，不允许把任意 Python 表达式写入
manifest。term 必须声明输入 signal、输出 dtype/shape、单位和 frame；未知 kind 或版本继续 fail fast。

action 同样改为命名 pipeline，并显式绑定 policy output、robot group 和控制模式：

```json
{
  "actions": {
    "legs": {
      "kind": "joint_torque",
      "api_version": "v1",
      "source": "policy.outputs.leg_actions",
      "target": "robot.groups.legs"
    },
    "arm": {
      "kind": "joint_position",
      "api_version": "v1",
      "source": "policy.outputs.arm_actions",
      "target": "robot.groups.arm"
    },
    "gripper": {
      "kind": "joint_position",
      "api_version": "v1",
      "source": "policy.outputs.gripper_action",
      "target": "robot.groups.gripper"
    }
  }
}
```

#### 8.4.5 Policy 支持命名多输入、多输出

PolicyInterfaceSpec 不再把 ONNX 限制为单输入、单输出：

```json
{
  "policy": {
    "runtime": {
      "kind": "onnxruntime",
      "api_version": "v1",
      "payload": "artifact.payloads.actor"
    },
    "inputs": {
      "proprioception": {"source": "task.observations.proprioception", "shape": [1, 128]},
      "camera": {"source": "task.observations.front_camera", "shape": [1, 3, 224, 224]},
      "hidden_in": {"source": "runtime.policy_state.hidden", "shape": [1, 256]}
    },
    "outputs": {
      "leg_actions": {"tensor_name": "legs", "shape": [1, 12]},
      "arm_actions": {"tensor_name": "arm", "shape": [1, 7]},
      "hidden_out": {"tensor_name": "hidden", "shape": [1, 256]}
    }
  }
}
```

recurrent state 必须作为显式输入/输出及 reset state 记录，不能由特定 runtime 在 manifest 外隐式维护。

#### 8.4.6 Runtime 和 Backend Requirements

多频率任务使用命名 clock，例如 `policy`、`low_level_control` 和 `command`。所有 pipeline 显式绑定 clock，
control loop 根据 artifact 构造 scheduler；backend 不得私自重复 policy action 或插值 command。

artifact 不包含 MuJoCo scene、厂商连接地址等 backend 配置，但应声明启动所需 capability：

```json
{
  "runtime": {
    "clocks": {
      "policy": {"period_s": 0.02},
      "low_level_control": {"period_s": 0.002}
    },
    "state_timeout_s": 0.1
  },
  "backend_requirements": {
    "state_channels": ["base_orientation", "joint_position", "joint_velocity"],
    "command_modes": {
      "legs": "joint_torque",
      "arm": "joint_position"
    },
    "quaternion_order": "xyzw"
  }
}
```

backend 在 `open()` 阶段证明自己满足 state channel、group mapping、command mode、rate、frame 和 limit 要求，
不满足时仍须在第一条 command 前失败。

#### 8.4.7 v1 到 v2 的兼容策略

- `go2_walk/v1` 语义保持不可变，继续服务当前 Go2 locomotion artifact；修改语义时新增 task version。
- runtime 内部先引入与 JSON schema 解耦的通用 IR，例如 `RobotGroupSpec`、`TaskInterfaceSpec`、
  `PolicyInterfaceSpec` 和 `RuntimeRequirements`。
- v1 loader 转换为该 IR；统一 control loop 只依赖 IR，不在循环中判断 manifest 版本。
- v2 writer 应至少用两个差异明显的真实 profile 验证后再定稿；v1 只覆盖 Go2 locomotion，不提前设计第二种
  任务 command。
- task config 增加兼容字段时可保持 task version；修改 observation/action 计算语义时必须升级 task version。
- 不提供未知字段静默降级、自动 reshape、自动 group 重排或缺失 signal 零填充。

第一版 command input 不增加独立 command spec；versioned `DeployTask` 直接定义并校验具体 command dataclass。
general manifest 真正落地时，再把跨 task command interface 显式写为 `TaskInterfaceSpec`，不能提前让 v1 runtime
承担尚未出现的动态发现需求。其余 v2 优先落地顺序是：先支持 policy 命名多输入/多输出，再引入 robot resource
groups。多频率调度和图像等大数据 channel 在真实任务需要时扩展，不作为 v2 schema 定稿的前置条件。

## 9. 启动前验证

CLI 读取 artifact 并实例化具体 `DeployTask`，不在创建 backend runtime 前比较 task observation/action
维度。policy 输入边界校验 task 实际构造的 observation 数组，task action 转换边界校验实际 policy 输出。
`TaskSpec` 不序列化维度；manifest tensor specs 保持模型元数据，`DeploymentProfile` 维度仅用于导出期交叉校验。

静态验证分为以下五层；实际 observation/action 兼容性在首个控制 tick 校验，不属于 enable 前验证：

1. **Artifact validation**：schema、相对路径、checksum、枚举、数组长度、范围、频率和版本。
2. **Policy validation**：ONNX 可加载，输入输出名称、dtype、shape 与 manifest 一致，零输入 smoke inference
   返回有限值。
3. **RobotSpec validation**：registry name、base link 和 joint names 非空且唯一；default pose 与全部 limit
   数组 shape 一致、数值有限且范围合法；source fingerprint 只用于来源追踪，不触发运行时 registry 查询。
4. **Backend validation**：硬件或模型 joint 能完整、唯一地映射到 canonical joint order；必需 state 字段、sensor、
   control mode 和 rate 可用；
   backend 模型或设备范围必须覆盖 RobotSpec 所需命令范围，最终 backend hard limiter 不得放宽 RobotSpec 限制；
   仿真 timestep 能组成整数个 control substeps。
5. **Command input validation**：concrete binding constructor 校验自身配置，首个 command 由对应 `DeployTask`
   校验具体类型、shape、finite value 和范围；第一版不依赖 device capability 或 command spec。

错误信息必须同时指出期望值和实际值，例如具体缺失 joint、错误 command type、shape、limit 或 sensor。校验器
不允许截断数组、填零、重排未知名称或采用近似 frequency。

## 10. Registry 与扩展机制

只为确实需要按 artifact、环境或 runtime recipe 选择实现的组件保留 registry。`motrix_deploy.profile` 提供
profile compiler registry；`motrix_envs.deploy` import 具体 compiler 时通过
`register_profile_compiler()` 装饰器按环境名自动注册。task runtime 通过 `motrix_deploy.tasks` Python entry-point
group 按版本化 task 标识发现。`available_tasks()` 只读取已安装插件元数据；`load_task_type()` 懒加载唯一
`DeployTask` 类，manifest codec 读取其 `spec_type` 而不实例化 task。`create_task()` 再将已解码的具体
`TaskSpec` 和 `RobotSpec` 传给该类构造运行时 task。未知名称或重复插件报错，不按 import 顺序选择实现。
`motrix_deploy` core 发布指向 `motrix_deploy.cli:main` 的 console script，不反向 import 具体环境、task 或 SDK；
具体部署 recipe 通过已安装 Hydra search-path plugin 提供的 `task` group 选择；外部应用也可显式指定 Hydra config path/name。

runtime 插件由 `motrix_deploy.runtime.factory` 通过 `motrix_deploy.backends` Python entry-point group 发现；
recipe 用 `runtime.backend` 选择插件，具体包为 `motrix_deploy_mujoco`、`motrix_deploy_motrixsim` 与 `motrix_deploy_unitree`。
core 只持有 `motrix_deploy.robot.interface.RobotInterface`、factory context
和严格的重复/未知插件校验，不 import 具体仿真器。command input 第一版由 application factory 直接创建，暂不
建立 device/binding plugin registry，详见 [Command Input 配置与创建](./deploy-command-input.md#7-配置与创建)。
policy runtime 由 CLI 装配；observation 与 action 语义由具体 `DeployTask` 直接实现，也不拆分成
term/processor registry。

注册冲突、未知组件和版本不兼容在 artifact validation 阶段报错。注册的具体实现只获得公共 contract，不获得
`PolicyRuntime` 或 raw observation，因此无法绕过统一 task 路径。

## 11. MuJoCo sim2sim 设计

### 11.1 首个 vertical slice

第一阶段以 `go2-walk` 为目标任务、`go2` 为 robot、MuJoCo 为 cross-simulator backend。Go2 walk 的 actor
observation 由 IMU、joint state、previous action、速度指令和足端 phase 组成，适合先验证通用部署闭环。首个
rollout 支持平地与确定性 procedural height-field 世界，分别通过通用部署世界 ID `flat`、`rough` 选择。
世界模块独立于 Go2 robot namespace 和训练任务。

首个闭环必须完整经过：

```text
CommandBinding.read_command(batch_size=1) -> PlanarVelocityCommand --+
                                                                   v
MuJoCo state -> RobotState -> Go2WalkDeployTaskV1 -> ONNX PolicyRuntime
             -> Go2WalkDeployTaskV1 -> RobotCommand -> MuJoCo torque/actuator
```

不允许为了尽快跑通而直接调用 `QuadrupedWalkTask._compute_obs()` 或 `apply_action()`；deployment
task compiler 应把 run-varying 参数写入 artifact，并通过数值一致性测试证明 task 实现与训练语义一致。

### 11.2 Backend 配置

仿真场景配置独立于 artifact。MuJoCo backend 接收单一完整 `SceneCfg`，机器人只在 `scene.objs.robot`
声明；不持有 world/robot 分离配置、composition 模式、环境 ID 或训练 registry。CLI 的应用层配置使用
`runtime.deploy_env_id` 与现有 robot registry ID `runtime.robot_id`，无需场景 `_target_`：

```yaml
runtime:
  kind: simulation
  backend: mujoco
  deploy_env_id: flat
  robot_id: go2
  robot_translation: [0.0, 0.0, -0.114]
  robot_rotation: [0.0, 0.0, 0.0, 1.0]
```

Placement 唯一来源是 `scene.objs.robot.translation`（Vec3）与 `.rotation`（xyzw），它们是 attachment transform，
与模型固有 base pose 组合一次，并非绝对世界 root 位姿。CLI 的可选 `runtime.robot_translation` /
`runtime.robot_rotation` 由应用层消费并写入该 robot，不进入 runtime backend config。
Go2 固有 base 高度为 0.445 m；identity rotation 下，flat 的 -0.114 m 偏移得到 0.331 m，rough 的
-0.025 m 偏移得到 0.42 m。runtime 保留复制场景中的 translation / rotation；`mj_resetData` 恢复编译的
`model.qpos0`（含组合后的 root placement），只用绑定 `RobotSpec` 的关键姿态覆写 articulation 关节。
`SimulationRuntimeConfig` 负责完整场景、渲染、实时节奏与 state-source 设置；其 `physics` 使用
`motrix_env_core.config.sim.SimCfg`，统一负责 `dt`、`solver_iterations`、`solver_tolerance` 与 `gravity`。
Runtime 默认 physics 为 `SimCfg(dt=0.002, solver_iterations=100)`，不定义 backend 专用 physics schema。
YAML 仍使用 `runtime.physics`；`runtime.sensor_bindings` 保留角色到模型本地 sensor 名称的映射，
在配置边界通过 `SensorBindings(**mapping)` 转换。程序化 `SimulationRuntimeConfig.sensor_bindings`
使用 `motrix_deploy.runtime.config.SensorBindings`，而非 Python 字典。
该 frozen dataclass 只接受 `base_angular_velocity`、`base_linear_acceleration`、`base_linear_velocity`
三个角色，各字段类型为 `str | None`，默认为 `None`；字符串必须非空且至少包含一个非空白字符，
未知角色与非字符串值均被拒绝。字段缺省或显式 `None` 表示没有绑定；MuJoCo 在打开机器人时拒绝
这三个必需角色的未绑定状态。Backend 继续校验 sensor 是否存在、类型、维度与坐标系，不为其他角色
合成状态，也不把所有 `RobotState` 字段视为 sensor-bound。
程序化 `SimulationRuntimeConfig.scene` 接收同样的完整场景，physics 与 sensor bindings 保持独立。
可选 solver/gravity 字段为 `None` 时保留模型来源值，显式值覆盖来源设置。
物理配置仅在启动前设置：runtime 读取编译使用的共享 `SimCfg`，应在 runtime 构造前完成配置，
不支持编译后修改 `dt` 来动态调参。
fall termination 属于 artifact task，而非 backend。

`motrix_deploy.env` 通过 `motrix_deploy.envs` entry-point group 发现部署世界。
`available_deploy_envs()` 只读取 metadata；`create_deploy_env(id)` 按需加载唯一零参数 factory，要求返回
`SceneCfg` 且 `scene.objs.robot` 为空。未知 ID、重复 provider、返回类型不符或 robot slot 已占用均报错。
内置 `flat`、`rough` 分别位于 `motrix_deploy_tasks.envs.flat` 与 `.rough`，只使用 core floor/hfield scene
配置，不 import 训练任务、机器人模型或训练 contact 配置。

`motrix_deploy.env.assemble_deploy_scene(deploy_env_id, robot_id)` 创建无机器人世界，以
`motrix_env_core.registry.make_robot_config(robot_id)` 填入 `scene.objs.robot`。core registry 惰性发现已安装的
`motrix_env_core.robots` entry points，训练与部署复用同一机制，不重复注册模型。
不创建通用 core scene-factory registry。世界扩展仍使用上述零参数 factory entry point。

CLI 仿真应用从 backend 配置中取出 `deploy_env_id`、`robot_id` 与可选 robot attachment overrides，
调用组装 helper 并将 overrides 写入 scene robot，再把完整场景传给
simulation plugin；backend 不通过 Hydra instantiate 解释场景 YAML。用户仍可选择完整的
`task=go2-walk-flat/sim` 或 `task=go2-walk-rough/sim` recipe。覆写 `runtime.deploy_env_id=rough` 仅替换世界，
保留 artifact 与机器人模型；attachment placement 不随世界单字段 override 改变，
需要不同净空时应显式调整 `runtime.robot_translation`。

程序化调用沿用 `SimulationRuntimeConfig`：

```python
from motrix_deploy.env import assemble_deploy_scene
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_env_core.config.sim import SimCfg

scene = assemble_deploy_scene("flat", "go2")
scene.objs.robot.translation = (0.0, 0.0, -0.114)
scene.objs.robot.rotation = (0.0, 0.0, 0.0, 1.0)
config = SimulationRuntimeConfig(
    scene=scene,
    physics=SimCfg(dt=0.002, solver_iterations=100),
    sensor_bindings=SensorBindings(
        base_angular_velocity="gyro",
        base_linear_acceleration="accelerometer",
        base_linear_velocity="global_linvel",
    ),
)
```

Runtime factory 只选择并准备宿主，不通过回调构造控制器。创建返回后，`runtime.robot` 及其只读
`spec` 在 `open()` 之前已可用；应用显式构造自己的 task、policy、binding 和 `ControlSession`：

```python
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime

runtime = create_simulation_runtime("mujoco", config)
robot = runtime.robot
control = ControlSession(
    robot=robot,
    task=task,
    policy=policy,
    command_binding=command_binding,
    period_s=0.02,
)
runtime.bind_control_session(control)
with runtime:
    result = runtime.run(steps=50)
```

这里的 task、policy 与 command_binding 由应用持有；依赖机器人契约的 task 可在 runtime 创建后从
`robot.spec` 构造。硬件由 `create_hardware_runtime(name, config, context)` 返回 prepared host 后，使用同样的
显式组装和绑定流程。`run(control, steps=...)` 仍可直接接收 session；绑定和传入的 session 必须属于该
runtime 的 robot，不能覆盖另一个已绑定的 session。`ControlSession.period_s` 是控制周期的唯一来源；
`SimulationRuntimeConfig` 不重复声明控制周期，仿真宿主在绑定或解析 session 时校验它与 physics timestep 的整数倍关系。
Task 只由 control session 持有，不重复传给 runtime。

`motrix_deploy` core 不查询训练环境 registry，也不依赖仿真器。MuJoCo plugin 将完整场景与
`config.physics` 直接传给 `MuJoCoSceneCompiler.create_spec()`，编译 terrain、robot、sensor 与视觉配置，
再按部署控制语义改写 actuator；MotrixSim compiler 同样直接消费共享 `SimCfg`，不重建或丢弃可选字段。
Backend 不读取训练 `EnvCfg`。场景仍须保持 canonical joint names，
并通过 RobotSpec、DoF、range 与 actuator 结构校验，不能用位置索引掩盖模型差异。

### 11.3 状态与命令转换

`MujocoRuntime` 准备机器人端口，`open()` 持有最终模型构造和绑定：

- 程序化创建时通过 `MuJoCoSceneCompiler.create_spec()` 组装 SceneCfg 并派生物理 `RobotSpec`；
- 打开时校验 canonical source motor 或 position actuator，把 position actuator 转换为 torque motor，
  按 `RobotSpec.torque_limit` 设置 control/force range；
- 编译经过 deployment transform 的最终 model；
- 通过 joint name 解析 qpos/qvel address；
- 从 base body 得到 orientation，并按显式 `sensor_bindings` 解析 IMU 与 world-frame linear velocity；
- 校验 joint range、actuator force range 和外部 RobotSpec limit 的包含关系。

`RobotInterface.open()` 只激活该 runtime 已准备的机器人绑定，不构造或持有世界。

每个 physics substep 都读取最新 joint position/velocity，按
`tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff` 计算 torque，再按 `RobotSpec.torque_limit` 硬限幅并写入
motor control。当前 MuJoCo deployment physics timestep 为 2 ms、control period 为 20 ms，因此每个 control
tick 固定执行十个 `mj_step`，每步使用 100 次 solver iterations；backend 必须显式覆盖并验证 XML 中的原始
timestep。初始化时调用 `mj_resetData` 恢复组合后的 `model.qpos0`，只覆写 articulation joint pose、
清空 velocity，然后执行 `mj_forward`。

### 11.4 运行终止与指标

sim2sim 的 Hydra 配置将 artifact、duration_s 和 command 放在顶层，仿真设置直接放在 runtime 下。
`runtime.backend=mujoco` 选择 MuJoCo；`runtime.viewer` 和 `runtime.realtime` 只用于仿真。`duration_s` 按 artifact 的
控制周期换算为 `ceil(duration_s / control_period_s)` 个控制 tick，不是墙钟超时。未设置 duration_s 时持续运行，
直到 viewer 关闭、用户按 Esc/Ctrl-C、机器人跌倒或发生错误。运行时还须：

- 使用 task 从 artifact 读取的 command range 乘以 scale 后创建 keyboard binding；
- reset 后从 step 0 重新初始化 phase、previous action 和 scheduler；
- 用户中断时受控停止；
- 输出 JSON-compatible `RolloutResult`；
- 可配置任务指标：base height、fall、速度跟踪误差等，但这些指标不进入通用控制循环。

deployment backend 使用最小 GLFW viewer：每次 `sync()` 只更新 MuJoCo scene 并 render，不拥有仿真步进；同一
window 的 callbacks 提供 keyboard event frame 与 mouse camera control，失焦后立即释放 held keys。viewer 模式默认使用
realtime scheduler，可通过 `runtime.realtime=false` 关闭实时节奏控制。headless CLI 使用配方中的 constant command，可通过 `command.velocity` 覆盖；
`runtime.realtime=null` 时跟随 `runtime.viewer`，也可显式配置实时节奏。`ControlLoop.run(steps=...)` 保留为
程序化确定步数运行能力，不进入 CLI 配置。

### 11.5 原生 MotrixSim runtime

`motrix_deploy_motrixsim` 通过同一个 `motrix_deploy.backends` entry-point group 注册 `motrixsim`。
已安装仿真 recipe 默认选择 MuJoCo；使用同一 task、artifact 和完整场景切换原生 backend：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=false runtime.realtime=false duration_s=1.0
```

粗糙地形同样使用 `task=go2-walk-rough/sim`，不复制 backend-specific recipe。程序化 factory 使用
`create_simulation_runtime("motrixsim", config)`，返回 prepared `runtime.robot` 后显式构造并绑定 session。
`ControlSession.period_s` 是唯一控制周期来源，绑定或解析 session 时验证 physics timestep 的整数倍关系。

`MotrixSimRuntime` 持有 `MotrixSimSceneCompiler` 产生的 MSD world、原生 model/data、reset、physics loop 与
SDK `RenderApp` viewer，不借用 MuJoCo runtime，不实例化训练环境。Native robot port 只读状态、缓存控制目标，
不推进或关闭世界。Position actuator 转换为 unit-gear torque motor，native motor 同样支持；每个 physics substep
使用当前关节状态重算 PD + feedforward 并限幅。Direct torque 绕过 PD，但保留契约限幅。

Go2 使用相同的显式 model-local sensor 名称 `gyro`、`accelerometer`、`global_linvel`，分别校验 local
`FrameAngVel`、local `FrameLinAcc` 和 world `FrameLinVel`。IMU site 必须属于 base body 并与其坐标轴对齐；
world velocity 必须指向 base link 或对齐的 base-body site。名称通过 scene robot `resolve_name()` 处理 affixes，
缺少或不兼容的传感器报错，不伪造状态。Base position 与 linear velocity 标记为 privileged simulation state。

Reset 重新分配 native data，恢复编译的初始 placement，再覆写 articulation key pose、清空 velocity 与缓存控制；
不重复施加 attachment transform。公开状态是 query buffer 的独立快照。Sample timestamp 使用 simulation time，
receive timestamp 与 realtime pacing 使用 monotonic clock。Native viewer 提供 keyboard capability，关闭窗口中断控制，
离开 runtime context 释放 viewer/model/data；headless 不需要图形窗口。

模块职责如下：

```text
motrix_deploy_motrixsim/src/motrix_deploy_motrixsim/
├── plugin.py       # motrixsim entry point；共享 SimulationRuntimeConfig / artifact context
├── runtime.py      # native world/model/data、physics、reset 与 ControlSession 调度
├── interface.py    # semantic sensor binding、query/write、非 owning RobotInterface
└── viewer.py       # SDK RenderApp 与 viewer-owned KeyboardDevice
```

## 12. CLI 设计

package 暴露统一入口：

```bash
# 只检查 artifact，不创建设备
motrix-deploy inspect artifact=path/to/go2_walk.deploy

# sim2sim 有界无窗口检查
motrix-deploy task=go2-walk-flat/sim \
  artifact=path/to/go2_walk.deploy \
  runtime.viewer=false duration_s=2.0

# 实时可视化
motrix-deploy task=go2-walk-flat/sim \
  artifact=path/to/go2_walk.deploy \
  runtime.viewer=true

# 真机入口
motrix-deploy task=go2-walk-flat/hardware \
  artifact=path/to/go2_walk.deploy \
  runtime.network_interface=enp5s0
```

`inspect` 只执行 artifact、RobotSpec 和 policy 静态校验，不查询 robot registry，也不打开 backend。
Hydra 配置的公共执行字段是 artifact、duration_s 和 command；task 配方选择互斥的 runtime/sim 或 runtime/hardware 配置组。
所有目标专用设置直接位于 runtime 下：`runtime.backend` 是标量字符串选择器，取值为 `mujoco` 或 `unitree_go2`，
不是嵌套配置映射或 Hydra 配置组；场景、physics、sensor_bindings、network_interface、kp、kd 等字段也直接位于 runtime 下。viewer 和 realtime 仅用于仿真，硬件由 runtime 在内部强制实时调度。
application 从 runtime 的可选 keyboard device capability 创建 binding。真机不使用静态确认布尔值，
必须保留 adapter 的启动等待、enable、急停和安全停止流程；默认配置等待操作员按下启动和使能按钮。
duration_s 换算为控制 tick 上限，不是墙钟超时；程序化 runtime.run(steps=...) 保留确定步数运行。

具体 task/runtime recipe 属于应用层，由 `motrix_deploy_tasks` 插件安装，提供 flat/rough 场景、机器人和 command 默认值。
`motrix_deploy` 包内保留通用配置；任务 defaults 选择 runtime/sim 或 runtime/hardware 配置组。
安装任务和目标 runtime 插件后，部署可在仓库外运行，不需要 workspace 配置路径。

## 13. 与现有 MotrixLab 的集成

### 13.1 Run metadata 与 checkpoint manifest

policy exporter 与 deployment profile compiler 复用现有：

- `metadata.json`：训练环境、framework、backend、算法和版本来源；
- `task_config.yaml`：已解析的训练配置快照；
- `checkpoints/manifest.json`：按语义定位 `best_policy`，不按文件名搜索 checkpoint。

deployment manifest 是新的运行时契约，不能直接给现有 checkpoint manifest 增加大量 obs/action 字段。两者的
生命周期不同：checkpoint manifest 管理训练 run 内产物，deployment manifest 管理可移植、不可变的部署产物。

### 13.2 RobotCfg 导出集成

deployment profile compiler 从当次训练的 `EnvCfg` 取得 `RobotCfg`，从中读取 base link、canonical key pose，
并结合编译模型的 joint/actuator 信息生成完整 `RobotSpec`。这段转换逻辑属于
`motrix_rl` / `motrix_envs` integration，依赖方向指向 `motrix_deploy.artifact`；部署 core 不反向依赖训练环境。

canonical joint 到 actuator 的解析是环境 profile compiler 共用的 robot 能力，由 `motrix_envs.deploy.robot`
提供。解析只依赖高层 `motrixsim.Actuator` 的 target 信息，不依赖具体 actuator 子类，也不创建 `SceneData` 或读取 `SceneModel.low`。
profile 可以从高层 actuator 读取公开的静态模型属性（例如 `ctrl_range`）；真机控制所需但未由高层通用接口公开的
力矩限制和 PD 参数必须作为显式、可追踪的 deployment profile 参数提供，不能借 runtime override state 推导。
sim backend 打开时负责校验 profile 参数与其模型一致，真机 backend 则把同一份显式 contract 映射到设备 SDK。

artifact 记录规范化 `RobotCfg` 来源 fingerprint，便于追踪生成时使用的配置。第一阶段部署运行时不要求安装
该配置的 provider，也不重新计算 fingerprint；实际兼容性通过 artifact `RobotSpec` 与 backend model/device
直接校验。独立 `motrix_robots` package 提供仿真模型，但不要求部署 runtime 按 registry name 重新解析或重算 fingerprint。

### 13.3 训练 obs/action 一致性

deployment-enabled task 必须在 `motrix_deploy_tasks` 提供一个带版本的 `DeployTask`，并在拥有源环境配置的
`motrix_envs.deploy` 提供对应 compiler。compiler 统一返回由 `robot / task / control` 组成的
`DeploymentProfile`；其中 `RobotSpec` 由通用 robot builder 从 `RobotCfg` 构建，`TaskSpec` 只保存版本化 task
标识和 config。profile 单独保留编译环境的 observation/action 维度供导出校验，不写入 task serialization；
artifact 的维度以 policy tensor shapes 为唯一来源。compiler 不恢复 policy checkpoint，也不执行 ONNX 导出。
每个 task version 至少有一组 golden state/context/action probe，
同时在训练环境和具体 `DeployTask` 上运行，断言 observation 和 command 数值一致。

这项一致性检查防止训练环境继续演进后 deployment profile 静默落后。reward、critic observation 和训练噪声
不属于部署 profile；训练框架的 observation normalizer 已由 policy exporter 烘焙进 ONNX，`DeployTask`
不得重复应用。

### 13.4 Deployment export 入口

统一入口根据环境选择最新的 run 并生成默认 artifact 路径，不暴露训练框架专属子命令：

```bash
python scripts/export_deploy.py env=go2-walk-rough
```

入口先调用统一 policy export，再依据 task metadata 选择 deployment profile，最后交给 artifact writer。CLI 和
公共 service 统一调用 metadata-driven `export_deploy_run`，不暴露同时绑定训练框架与任务的函数。不支持 policy export 时，
错误列出 `RlFramework` 当前支持的组合；缺少 deployment profile 时，错误单独指出 task/profile 不受支持，避免把
模型导出能力和任务部署能力混为一谈。

## 14. 测试策略

### 14.1 单元测试

- `RobotSpec`、state、command 的 shape、dtype、finite、limit 和 joint-order 校验；
- artifact round trip、checksum、path traversal、schema/version 和缺字段错误；
- 具体 task 的 observation 顺序、scale、frame、reset 和 previous-action 语义；
- action scale/default pose/clip/rate limit/PD gains；
- ONNX input/output mismatch 和非有限输出；
- scheduler deadline、overrun、timeout、异常停止和幂等 close。

### 14.2 Contract test

`FakeRobotInterface` 是测试目录内共享的确定性 robot I/O 替身，不作为部署包的运行时 adapter 发布。控制面测试使用它验证：

- lifecycle 调用顺序；
- 不同状态反馈响应下的控制循环与确定性闭环；
- state timeout、backend unhealthy、write failure 和启动校验失败；
- stop/close 在正常结束、异常和用户中断下都执行。

name-based joint mapping、模型关节缺失以及 reset 等具体 adapter 行为由对应后端的测试验证，
不使用 fake 的名字集合检查替代真实映射测试。

command input 的 device contract suite 独立维护在
[Command Input 第一版测试边界](./deploy-command-input.md#9-第一版测试边界) 中。

### 14.3 集成测试

- 用 fixture ONNX 和 fake backend 完成确定步数闭环；
- 同一 deployment profile 分别打包来自两个已支持训练后端、tensor contract 相同的 ONNX，不修改 artifact
  reader、runtime 或 profile compiler；
- 同一 artifact 在 fake 和 MuJoCo backend 上使用同一 builder/processor 配置；
- `go2-walk` deployment profile 的 golden probe 与训练 actor obs/action 数值一致；
- MuJoCo headless 与 GLFW viewer 能确定性初始化、运行并输出结构化结果；内部测试可用固定 steps 截断；
- 缺 joint、交换 joint、错误 quaternion、错误 policy shape 和超出 capability 时全部在第一条 command 前失败。

CI smoke test 使用短 rollout，不以训练 reward 达标作为框架正确性的唯一条件。策略质量回归可以在 #139 基于
同一 CLI 和 `RolloutResult` 增加独立阈值。

Go2 vertical slice 的验收基线为：确定性初始化的 1000-tick headless rollout trace SHA-256
`8db7e74506dbe36147746696cb15a57b0a4bd2ddf95096f3dce597e94dececda`；GLFW viewer 的 2 秒真实 GUI smoke
完成 100 ticks、无 overrun，并正常退出。viewer 与 headless 的同长度 trace SHA-256 均为
`b7b94aaa2cfba6e4700e92b650d408da13dd73091ce870c55e326e2012ce21a0`，两种模式共用同一 backend stepping 路径。

## 15. 分阶段范围

### 第一阶段：框架与 sim2sim

1. 建立包含通用 CLI 的 `motrix_deploy` library，以及通过 `motrix_deploy.tasks` entry points 提供具体任务类的
   `motrix_deploy_tasks` workspace package。
2. 实现公共 contracts、artifact schema/reader/validator 和 component registry。
3. 实现 `DeployTask` 接口、ONNX `PolicyRuntime` 和直接编码 obs/action 的 `go2_walk/v1`。
4. 实现统一 control loop、scheduler、metrics、fake backend 与 contract tests。
5. 为一个现有 `go2-walk-rough/rslrl.ppo` checkpoint 生成可验证 artifact，并完成训练/部署 golden probe。
6. 实现 entry-point MuJoCo backend plugin、SceneCfg backend config，以及 headless/GLFW-viewer deployment run。
7. 为 #139 提供可复用 CLI 与 machine-readable result，不另写临时 runner。

### Command input 分层

该阶段的领域契约、组合方式与验收边界统一见
[Deploy Runtime Command Input 分层设计](./deploy-command-input.md)，不在框架总设计中重复维护。

### 第二阶段：sim2real vertical slice（软件实现完成，吊架验证待执行）

1. 已实现独立 Unitree Go2 SDK2 plugin 和 fake-SDK contract tests。
2. 已增加显式 enable、默认姿态过渡、状态/命令 timeout、阻尼停止和急停入口。
3. 待用第一阶段同一个 artifact、builder、runtime 和 processor 完成吊架 smoke test。
4. 已补充安全清单和操作文档；真机周期统计待吊架测试采集。

### 后续扩展

- recurrent policy 和多输入 ONNX；
- 多速率 policy/control scheduler；
- ROS 2 input device 或 service adapter；
- artifact 压缩签名、远程分发和更多 vendor backend。

Go2 MuJoCo sim2sim 与 Unitree sim2real 软件链路已实现；真机吊架 smoke test 仍是发布前置条件。
