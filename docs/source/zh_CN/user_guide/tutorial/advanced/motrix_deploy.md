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

artifact 目录自包含：

```text
go2-walk-flat.deploy/
├── manifest.json        # 机器人/控制/任务设置与 payload 校验和
├── policy/model.onnx    # 导出的 actor（含输入归一化）
└── task/                # 任务自有 payload（例如 G1 的 motion NPZ）
```

部署还需要安装 `motrix_deploy_tasks` 插件，提供行走任务实现、
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

示例在外部加载模型并注入 task；task 计算机器人命令，control session 负责机器人 I/O 与执行统计，
runtime 负责物理推进、调度与 viewer。[进阶的显式组装](#显式组装-runtime-与任务)一节逐步展示同样的组合方式。
直接使用 ONNX 文件时需要显式指定任务预处理和动作设置；artifact 则为 CLI 携带这些设置。

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

## 部署 G1 WBT 舞蹈

G1 全身跟踪舞蹈任务使用同样的导出 → artifact → `motrix-deploy` 流程。artifact 内嵌伺服增益、action scale、
preparation 增益与时序、安全阈值以及完整的 motion 片段（以 `payloads/motion.npz` payload 形式，校验和记录在
manifest 中），部署既不需要 Torch 也不需要 RL 环境。导出需要训练 provider 与任务环境包，但**不**执行训练：

```bash
source .venv/bin/activate
RUN=runs/g1-wbt-dance/motrix/torch/fastsac/<run-dir>
python scripts/export_deploy.py run="$RUN" \
  output=artifacts/g1-wbt-dance-software-pd.deploy validation.atol=3e-5

motrix-deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy
```

无窗口验证与 artifact 检查：

```bash
motrix-deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy \
  runtime.viewer=false
motrix-deploy inspect artifact=artifacts/g1-wbt-dance.deploy
```

默认时长来自内嵌的 motion 片段；更短的运行用 `duration_s` 缩短，超长时长会在打开 backend 前被拒绝，
不会循环或保持。无需速度指令：motion 由 artifact 持有。同一 artifact 可运行在 MuJoCo 与 MotrixSim
（`runtime.backend=motrixsim`，配合训练预算使用 `runtime.physics.solver_iterations=3`）。
它不是真机部署 recipe：当前 Unitree 传输仅支持 Go2，且 sim2sim 成功不代表真机安全。

任务按真实部署实践编排四个阶段：`prepare`（实测关节 ramp 到默认站立位，就绪门连续满足）、`policy_hold`、
`playback` 与 `damping`。默认就绪后自动接管；程序化应用可用 `automatic_start=False`、请求方法或附加键盘
（`p` 启动 policy、`m` 提前启动 motion、`o` 停止）手动分步。注意舞蹈以后倾姿态结束，该姿态只有跟踪中的
policy 能维持平衡：播放预算完成后 task 直接进入实测 damping 收尾，若要以站立收场，应让 motion 本身以
站立姿态结束。

## 进阶说明

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

### 真机传感器与电机接线

`motrix_deploy_unitree.hardware.UnitreeGo2HardwareCfg` 独立持有不依赖资产的硬件姿态与接线。
`HardwareSensorBinding` 映射 SDK IMU 字段，包括将 `wxyz` 四元数转换为规范 `xyzw`；
`UnitreeMotorBinding` 将规范关节名映射到 SDK 电机索引。artifact 仍拥有策略关节顺序、默认姿态和限幅。

### 选择部署世界

通过 `runtime.deploy_env_id` 选择世界，通过现有 registry ID `runtime.robot_id` 选择机器人。
保留同一 Go2 recipe 和 artifact，仅切换世界：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.deploy_env_id=rough
```

仅覆写世界不会改变机器人 attachment placement。新地形需要不同净空时应调整 `runtime.robot_translation`；
完整的 `task=go2-walk-rough/sim` recipe 自带对应的 attachment 默认值。
`RobotCfg.translation`/`rotation` 是与模型固有 base pose 组合一次的 attachment transform：
Go2 固有 base 高度为 0.445 m，identity rotation 下，平地 recipe 的 -0.114 m 偏移得到 0.331 m，
粗糙地形 recipe 的 -0.025 m 偏移得到 0.42 m。

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

`runtime.physics` 使用共享的 `SimCfg`（`dt`、`solver_iterations`、`solver_tolerance`、`gravity`，
默认 `dt=0.002, solver_iterations=100`）；可选字段设为 `None` 时保留模型来源中的值。
物理参数应在 runtime 构造前设置——不支持编译后修改 `dt` 动态调参。

`runtime.sensor_bindings` 把状态角色映射到模型内 sensor 名称，只支持 `base_angular_velocity`、
`base_linear_acceleration` 与 `base_linear_velocity`（`str | None`）。MuJoCo 要求绑定角速度和加速度；
可选的 `base_linear_velocity` 未绑定时保持 `None`。

程序化应用组装同一场景并传给 `SimulationRuntimeConfig`：

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

更多世界来自已安装的插件：在 `motrix_deploy.envs` entry-point group 中声明零参数 `SceneCfg` factory，
`motrix_deploy.env.available_deploy_envs()` 可列出已安装 ID。

### 显式组装 runtime 与任务

Runtime factory 选择 backend 并准备 `runtime.robot`；`RobotSpec` 在 `open()` 前即可读取。
在外部加载模型，然后构造并绑定 session。沿用上面的 `config` 以及应用持有的
`task_spec`、`model` 与 Go2 `command_binding`：

```python
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy.task import create_task

runtime = create_simulation_runtime("mujoco", config)
robot = runtime.robot
task = create_task(task_spec, robot.spec, model, steps=50)
session = ControlSession(
    robot=robot,
    controller=task,
    command_binding=command_binding,
    period_s=0.02,
    state_timeout_s=0.1,
)
runtime.bind_session(session)
with runtime:
    result = runtime.run()
```

`create_task(spec, robot, policy, steps=None)` 根据 spec 的任务名解析具体 task class；引用 artifact payload 的
任务（例如 G1 的 motion NPZ）也通过同一调用获得 artifact。控制周期必须是 physics timestep 的整数倍。
`command_binding` 默认值为 `None`：G1 舞蹈这类自主任务可省略，Go2 行走则需要外部组装速度 binding，
其值由 task 按速度上下界校验。硬件应用使用 `create_hardware_runtime(name, config, context)` 并按同样方式组装 session。

`runtime.run()` 返回 rollout 结果，包含成功标志、退出原因、模型完成步数、墙钟耗时与延迟统计；
仿真运行额外报告 `real_time_factor`。

### 选择原生 MotrixSim 部署

已安装的仿真 recipe 默认使用 MuJoCo。安装 `motrix-deploy-motrixsim` 后，可用同一 recipe 和 artifact
选择原生 backend：

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim runtime.viewer=false runtime.realtime=false duration_s=1.0
```

同样的 override 适用于 `task=go2-walk-rough/sim`，无需复制 task recipe。程序化应用调用
`create_simulation_runtime("motrixsim", config)`，然后沿用上面的显式 session 组装流程。
Go2 state mappings 仍为 `gyro`、`accelerometer` 与 `global_linvel`：原生 sensor 校验 local angular
velocity/acceleration 与 world-frame linear velocity。在可用的图形环境中移除 `runtime.viewer=false`
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
