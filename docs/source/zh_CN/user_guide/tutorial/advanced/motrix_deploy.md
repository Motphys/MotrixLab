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
