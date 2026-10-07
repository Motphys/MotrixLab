# Go2 平地行走：从训练到真机部署

将导出的 Go2 策略先在 MuJoCo 中检查，再尝试真机部署。安装任务和 MuJoCo 插件后，可用以下命令进行有界无窗口检查：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

下面介绍如何生成 artifact、交互控制，以及现有 Unitree 真机工作流。

## 1. 安装与训练

在仓库根目录执行：

```bash
sh install.sh --all
source .venv/bin/activate
python scripts/train.py task=go2-walk-flat/rslrl.ppo
```

训练结果保存在 `runs/go2-walk-flat/`。已有包含 metadata 的 run 时可跳过训练。

## 2. 导出并检查 artifact

```bash
python scripts/export_deploy.py env=go2-walk-flat
motrix-deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

导出会选择最近一次 run，并创建 `artifacts/go2-walk-flat.deploy/`。用 `run=<run-dir>` 替代 `env` 可选择指定 run，
`output=<new-artifact-dir>` 可更改输出位置。已有 artifact 目录不会被覆盖。确认检查结果为 `valid: true`。

artifact 包含 ONNX 策略、机器人与控制设置，以及任务规格。部署还需要安装 `motrix_deploy_tasks` 插件，提供行走任务实现、
部署场景和 Hydra 配方，以及目标对应的运行时插件。配方安装在插件中，在仓库外运行也不需要 workspace 配置路径。
任务 defaults 选择 Hydra 运行时组（`sim` 或 `hardware`），无需另行覆盖运行时组。
顶层公共设置为 `artifact`、`duration_s` 和 `command`。所有目标专用设置直接位于 `runtime` 下；
`runtime.kind` 选择仿真或硬件执行路径；`runtime.backend` 选择插件（仿真为 `mujoco` 或 `motrixsim`，硬件为 `unitree_go2`）。
`runtime.backend` 是标量字符串选择器，不是嵌套配置映射或 Hydra 配置组。

## 3. 在 MuJoCo 中检查

使用开头的无窗口命令进行有界运行，或打开 viewer：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

按住 `W/S` 前后移动，`A/D` 横移，`Q/E` 转向。关闭窗口、按 Esc 或 Ctrl-C 退出。
配方选择平地场景和 `go2` 机器人。无窗口模式使用固定零速度；添加 `'command.velocity=[0.5,0.0,0.0]'` 可检查移动。
通过 `duration_s` 限制运行长度：按 artifact 的控制周期换算为
`ceil(duration_s / control_period_s)` 个控制 tick，而不是墙钟超时计时器。
启动和关闭可能额外消耗墙钟时间。

仿真专用设置为 `runtime.viewer` 和 `runtime.realtime`。
`runtime.realtime=null` 跟随 `runtime.viewer`；用 `runtime.realtime=true` 让无窗口运行按实时节奏执行，
用 `runtime.realtime=false` 则不进行实时节奏控制。

### 在 Python 中组装运行时

完整示例 `examples/deploy_to_sim.py` 组合场景、行走任务、示例 ONNX 策略和输入 binding：

```bash
python examples/deploy_to_sim.py --headless --steps 100
```

它通过 `create_simulation_runtime` 创建 MuJoCo 运行时，用 `runtime.bind_control_session(control)` 连接
`ControlSession`，并在 `with runtime:` 中运行。控制会话负责输入、观察、推理和机器人指令；运行时负责物理推进、
调度和 viewer。直接使用 ONNX 文件时需要显式指定任务预处理和动作设置；artifact 则为 CLI 携带这些设置。

### 选择原生 MotrixSim

安装 `motrix-deploy-motrixsim` 后，将 backend 切换为 `motrixsim`，即可复用同一 artifact 和场景配方：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=true
```

Python 示例同样支持原生 MotrixSim：

```bash
python examples/deploy_to_sim.py --backend motrixsim
```

## 4. 运行真机

将机器人悬空，切换到低层/调试模式，连接网线，并准备好独立急停和操作人员。启动过程可能在遥控器 Start 门控前
让机器人执行站下动作。先检查 artifact，再将 `enp5s0` 换成实际网卡名称：

```bash
motrix-deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.network_interface=enp5s0
```

真机运行在内部强制按实时节奏执行；真机运行时不提供 `runtime.viewer` 或 `runtime.realtime` 字段。
按 Start，等待默认姿态过渡完成后按 A；按住 L1 并移动摇杆发送运动指令。
B 请求趴下，Select 触发急停。停止时先发送阻尼指令，再关闭 DDS。软件测试不能替代悬空真机检查。

安装的真机配方设置 `runtime.kp=50`、`runtime.kd=1`。把二者设为 `null` 可保留 artifact 增益，也可传入非负标量
或按规范关节顺序排列的逐关节值。

只读诊断不创建指令 publisher：

```bash
motrix-deploy-unitree read-lowstate enp5s0
```

在同样的硬件安全准备下，可进行有界单关节检查：

```bash
motrix-deploy-unitree joint-control enp5s0 FL_thigh_joint 0.9 \
  --artifact artifacts/go2-walk-flat.deploy
```

该工具使用 artifact 中的机器人、增益、时序和限位设置。等待 Start 和 A 后，移动关节、保持、返回默认姿态，
最后以阻尼模式关闭。

## 进阶说明

Artifact 检查通过 `motrix_deploy.tasks` 解析版本化 task 类并校验其 Pydantic `spec_type`，
不构造 task，也不 import 训练环境或仿真后端。JSON wire format 保持 `{name, config}`。
Python 使用 `spec.kp`、`spec.action_scale` 等直接属性；action scale 与上下界是规范关节顺序的向量。
Policy action 上下界先于缩放和默认姿态偏移生效；机器人 position limits 约束最终关节目标。

### 从 YAML 加载 task spec

程序化应用可在 `task.yaml` 中保存完整 Go2 任务字段（即 `task.config` 的内容，不包含 manifest envelope）。
OmegaConf 负责加载、合并配置和解析插值，随后由 Pydantic 校验普通字典：

```python
from omegaconf import OmegaConf
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkTaskSpec

cfg = OmegaConf.load("task.yaml")
spec = Go2WalkTaskSpec.model_validate(OmegaConf.to_container(cfg, resolve=True))
```

`Go2WalkTaskSpec` 使用字段约束和 model validator 校验关节向量长度及上下界顺序，无需单独调用 `validate()`。
完整 deployment manifest 应通过 artifact reader 读取。

### 启动与数组校验

`RobotInterface.enable()` 不接收初始命令。Unitree 自行过渡到 `RobotSpec.default_joint_position`，
目标速度和 feedforward torque 为零，使用解析后的启动增益。`runtime.kp` / `runtime.kd` 覆盖 artifact 增益；
`null` 保留 artifact 增益。启动不通过 task `process_action()` 转换零动作。
初始状态 safety 与 task termination 检查仍先于 enable，之后重新检查新状态；Start/A 门控不变。

`DeployTask` 不声明 `observation_size` 或 `action_size`。policy 输入边界校验实际 observation，
task action 转换边界校验实际 policy 输出。CLI 组装不提前比较 task 维度。manifest tensor specs
保持模型元数据；`DeploymentProfile` 维度仅用于导出时与编译环境交叉校验。

### 真机传感器与电机接线

`motrix_deploy_unitree.hardware.UnitreeGo2HardwareCfg` 独立持有不依赖资产的硬件姿态与接线。
`HardwareSensorBinding` 映射 SDK IMU 字段，包括将 `wxyz` 四元数转换为规范 `xyzw`；
`UnitreeMotorBinding` 将规范关节名映射到 SDK 电机索引。这些定义属于硬件插件的 `sensor` 和
`actuation` 模块，而不是仿真机器人模型。artifact 仍拥有策略关节顺序、默认姿态和限幅。
导入硬件配置不会提前加载模型资产、仿真器或 SDK，也不伪造 base position 或 linear velocity。

### 选择部署世界

通过 `runtime.deploy_env_id` 选择世界，通过现有 registry ID `runtime.robot_id` 选择机器人。
CLI 调用 `motrix_deploy.env.assemble_deploy_scene(deploy_env_id, robot_id)`：创建无机器人世界，
通过 `motrix_env_core.registry.make_robot_config(robot_id)` 填入 `scene.objs.robot`。
core registry 惰性发现已安装的 `motrix_env_core.robots` entry points，训练与部署复用同一机制。部署世界 `flat` 和 `rough` 位于独立的
`motrix_deploy_tasks.envs` 模块，使用 core 地面/高度场配置，不 import 训练任务、机器人模型或训练 contact 配置。

保留同一 Go2 recipe 和 artifact，仅切换世界：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.deploy_env_id=rough
```

仅覆写世界不会改变机器人 attachment placement。新地形需要不同净空时应调整 `runtime.robot_translation`；
完整的 `task=go2-walk-rough/sim` recipe 自带对应的 attachment 默认值。

已安装的平地 recipe 使用 ID 选择世界和机器人：

```yaml
runtime:
  kind: simulation
  backend: mujoco
  deploy_env_id: flat
  robot_id: go2
  robot_translation: [0.0, 0.0, -0.114]
  robot_rotation: [0.0, 0.0, 0.0, 1.0]
```

`runtime.physics` 与 `runtime.sensor_bindings` 仍是独立设置。MuJoCo 只接收完整 `SceneCfg`，
不选择环境/机器人组合，不组装场景，也不查询训练 registry。程序化 `SimulationRuntimeConfig.scene` 接收同一个完整场景，
机器人只在 `scene.objs.robot` 中声明一次。

Placement 只有一个来源：`RobotCfg.translation`（Vec3）与 `rotation`（xyzw）是 attachment transform，
与模型固有 base pose 组合一次，并非绝对世界 root 位姿。Go2 固有 base 高度为 0.445 m；identity rotation 下，
平地 recipe 的 -0.114 m 偏移得到 0.331 m，粗糙地形 recipe 的 -0.025 m 偏移得到 0.42 m。
runtime 在复制的场景中保留这些变换。`mj_resetData` 恢复编译后的 `model.qpos0`，包括组合后的 root placement；
初始化只用绑定 `RobotSpec` 的关键姿态覆写 articulation 关节。

外部世界插件在 `motrix_deploy.envs` entry-point group 中声明零参数 `SceneCfg` factory。
`motrix_deploy.env.available_deploy_envs()` 无需加载 factory 即可列出已安装 ID；
`create_deploy_env(id)` 只加载选中的 factory，并要求 `scene.objs.robot` 为空。
core 组装 helper 添加由现有 robot registry 选择的机器人；不引入通用 scene-factory registry，也不通过
Hydra 实例化场景。CLI 消费 `deploy_env_id`、`robot_id` 与可选 `robot_translation` / `robot_rotation`，
把 attachment overrides 写入 `scene.objs.robot.translation` / `.rotation` 后，再把完整场景传给仿真插件。
这些应用层选项不是 runtime backend 字段。

`SimulationRuntimeConfig` 负责场景、渲染、实时节奏与状态来源设置；其 `physics` 字段使用共享的
`motrix_env_core.config.sim.SimCfg` 配置 `dt`、`solver_iterations`、`solver_tolerance` 与 `gravity`，
默认值为 `SimCfg(dt=0.002, solver_iterations=100)`。两个 backend compiler 直接消费该字段：
可选 solver/gravity 字段为 `None` 时保留模型来源中的值，显式配置则覆盖。
YAML 仍使用 `runtime.physics`，与场景、state sensors 保持独立。
物理参数应在 runtime 构造前设置：runtime 读取编译使用的共享 `SimCfg`；不支持编译后修改 `dt` 来动态调参。

程序化应用使用同一 helper，并把结果传给 `SimulationRuntimeConfig`：

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

`SimulationRuntimeConfig.sensor_bindings` 使用 frozen dataclass `SensorBindings`，
而非 Python 字典。它只支持 `base_angular_velocity`、`base_linear_acceleration` 与
`base_linear_velocity`，类型均为 `str | None`，默认为 `None`。名称必须非空且至少包含
一个非空白字符；未知角色与非字符串值均被拒绝。YAML `runtime.sensor_bindings` 保留同样的
映射键，在配置边界通过 `SensorBindings(**mapping)` 转换。字段缺省或显式 `None` 表示
没有绑定；MuJoCo 在打开机器人时会拒绝这三个必需角色的任一种未绑定状态。
Backend 仍校验 sensor 是否存在、类型、维度与坐标系；不会为其他角色合成状态，
也不是每个 `RobotState` 字段都通过 sensor 绑定。

### 显式组装 runtime 与控制器

Runtime factory 选择 backend 并准备 `runtime.robot`；绑定的 `RobotSpec` 在 `open()` 之前即可读取。
应用自行创建 task、policy 与 command binding，再显式构造并绑定 control session。
沿用上面的 `config` 以及应用持有的 `task`、`policy`、`command_binding`：

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

依赖机器人契约的 task 应在 runtime 创建后从 `robot.spec` 构造。仿真 runtime 只从 scene 派生
`RobotSpec`，不接收外部 spec。Artifact 部署由 CLI 在组装 control 前，将 artifact 的机器人契约
与 `runtime.robot.spec` 对照校验。
`ControlSession.period_s` 是控制周期的
唯一配置位置，`SimulationRuntimeConfig` 不重复声明；仿真 runtime 在绑定或解析 session 时校验该周期与 physics timestep
的关系。Runtime 创建不接收控制器构造回调。
Runtime context 持有世界资源并推进仿真；`ControlSession` 执行共享控制逻辑；`RobotInterface`
负责机器人 I/O，不推进或关闭世界。硬件应用先调用 `create_hardware_runtime(name, config, context)`，
再使用同样的显式构造与绑定流程。直接调用方也可向 `runtime.run(control, steps=...)` 传入 session。

### 选择原生 MotrixSim 部署

已安装的仿真 recipe 默认使用 MuJoCo。安装 `motrix-deploy-motrixsim` 后，可用同一 recipe 和 artifact
选择原生 backend：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=false runtime.realtime=false duration_s=1.0
```

同样的 override 适用于 `task=go2-walk-rough/sim`，无需复制 task recipe。程序化应用调用
`create_simulation_runtime("motrixsim", config)`，然后沿用上面的显式 session 绑定流程。
Go2 state mappings 仍为 `gyro`、`accelerometer` 与 `global_linvel`：原生 sensor 校验 local angular
velocity/acceleration 与 world-frame linear velocity。Runtime 持有原生 MSD 编译、model/data、reset、physics
与可选 SDK `RenderApp` 渲染，不创建 MuJoCo runtime 或训练环境。在可用的图形环境中移除 `runtime.viewer=false`
即可启用原生 viewer 与其 keyboard provider。

程序化 ONNX 示例可运行
`python examples/deploy_to_sim.py --backend motrixsim --headless --steps 50`，
准备匹配的 ONNX 策略；策略文件选项见示例。
这条路径加载 raw ONNX，不读取 `.deploy` manifest：任务预处理、tensor specs 和控制参数由 Python
示例显式声明，必须与模型匹配。其 scene 与控制器和默认 MuJoCo 路径共用。
Headless 使用零速度指令；窗口模式在可用时使用 viewer keyboard provider。

### 使用外部 Hydra 配置

上面的原生命令直接使用已安装的任务包，无需 checkout 中的配置目录。内置 runtime recipe 的唯一维护位置是
`motrix_deploy_tasks/config/task/{task}/{sim,hardware}.yaml`，优先通过 `task=...` 选择。自定义应用仍可使用 Hydra 标准的
`--config-path` 与 `--config-name` 管理自己的配置。以下示例假定应用已自行编写
`/absolute/path/to/my_app/config/deploy.yaml`，不是选择或复制内置 recipe：

```bash
motrix-deploy \
  --config-path /absolute/path/to/my_app/config \
  --config-name deploy \
  artifact=artifacts/go2-walk-flat.deploy
```

请使用绝对配置路径：Hydra 相对于 CLI 模块而非 shell 当前目录解析相对路径。
外部配置自行声明 defaults 和完整的 runtime/backend 配置，不必使用已安装的 `task` group。仿真配置必须显式提供
defaults 中的 `runtime=sim`、`runtime.backend`、`runtime.deploy_env_id`、`runtime.robot_id`，以及选中 backend 所需的全部设置（包括 MuJoCo
state-sensor mappings）。artifact 不提供这些 runtime 选择。沿用 Hydra 原生选项，不增加额外的 `--config` 参数。
