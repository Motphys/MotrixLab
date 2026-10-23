# Go2 平地步行：sim2sim 与 sim2real

本页描述 `go2-walk-flat` 从训练策略到部署的完整路径：导出 artifact，在 MuJoCo 或 MotrixSim 中检查，再通过 Unitree SDK2 DDS 运行真机。
artifact 统一保存策略、关节顺序、默认姿态、动作缩放和控制周期；仿真通过不代表真机安全。

## 1. 训练策略

```bash
source .venv/bin/activate
mxlab train task=go2-walk-flat/rslrl.ppo
```

训练结果位于 `runs/go2-walk-flat/`，训练后的步态需要通过 `mxlab play` 回放策略查看。

## 2. 导出并检查 artifact

```bash
mxlab export-deploy env=go2-walk-flat output=artifacts/go2-walk-flat.deploy
mxlab deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

固定某次实验时用 `run_dir=/path/to/run` 替代 `env`，二者只能设置一个。检查输出必须为 `valid: true`。
artifact 目录自包含：

```text
go2-walk-flat.deploy/
├── manifest.json        # 机器人/控制/任务设置与 payload 校验和
├── policy/model.onnx    # 导出的 actor（含输入归一化）
└── task/                # 任务自有 payload（例如 G1 的 motion NPZ）
```

部署还需要安装 `motrix_deploy_tasks` 插件，提供行走任务实现、
部署场景和 Hydra 配方，以及目标对应的运行时插件。配方安装在插件中，在仓库外运行也不需要 workspace 配置路径。
任务 defaults 组合类型化运行时配置，无需另行覆盖运行时组。
顶层公共设置为 `artifact` 和 `duration_s`。运行时设置位于 `runtime` 下；Unitree 机器人专用选项
位于 `runtime.robot` 下，而 `runtime.network_interface` 等传输设置仍直接位于 `runtime` 下。
配置类型区分仿真与硬件执行路径；`runtime.backend` 选择插件（仿真为 `mujoco` 或 `motrixsim`，硬件为 `unitree`）。
`runtime.backend` 是标量字符串选择器，不是嵌套配置映射或 Hydra 配置组。

## 3. sim2sim：MuJoCo

无窗口短跑：

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.viewer=false duration_s=2.0
```

有窗口交互：查看时省略 `runtime.viewer=false`，默认使用MuJoco viewer。

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy
```

若要选择原生的MotrixSim，安装 `motrix-deploy-motrixsim` 后可复用同一 artifact：

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim
```
窗口中 `W/S` 前后移动、`A/D` 横移、`Q/E` 转向，Esc 或 Ctrl-C 退出。

```{video} /_static/videos/go2_sim2sim.mp4
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 60%
:align: center
```

使用 `task=go2-walk-rough/sim` 可选择随包提供的粗糙地形场景。更换场景不代表平地策略适合粗糙地形，测试前需确认策略适用范围。

同时仿真中的速度跟踪并不等于真机跟踪，仿真中还涉及动作延迟、摩擦、质量、质心、观测噪声等随机化。即使 sim 中能走，也不能直接说明真机一定稳定。
Go2 的仿真训练和 artifact 默认增益约为 `kp=35, kd=0.5`，而真机配方覆盖成了 `kp=50, kd=1`，这也是 sim2sim 到 sim2real 的一个明显差异。

## 4. 真机部署sim2real：Unitree Go2

事前准备：准备可靠承重支撑、所有腿的活动空间、操作员就绪的独立急停，以及另一名操作人员。停止其他所有低层指令发布程序。
阻尼和部署停止都不能维持站立支撑；没有受支撑的恢复方案时不要继续。
真机前将 Go2 悬空，切换低层/调试模式，准备独立急停和现场操作员。

查找连接机器人的有线网卡，把后续指令的`enp5s0` 换成实际网卡：

```bash
ip -br addr
```
将PC与Go2通过网线连接后，需要观察是否能够ping通，需将电脑 Ethernet 地址设置为与机器狗同一网段，如：

```text
PC 地址：192.168.123.100
Go2 地址：192.168.123.161
```
为了测试用户电脑与Go2机器人内置电脑是否正常连接，可在终端中输入:

```bash
ping 192.168.123.161
```

### 只读状态检查：

只读诊断不创建指令 publisher，把 `enp5s0` 换成实际网卡。：
```bash
motrix-deploy-unitree read-lowstate enp5s0
```
检查是否持续收到有效状态，以及是否有 CRC 或解码错误。该工具不发送电机指令，也不释放官方控制权。
官方控制仍活动时，遥控器按键可能同时触发官方动作，不要将按键检查视为无副作用操作。

### 有界单关节检查：

此测试不需要 artifact 或策略，但会先让全身过渡到保持姿态，因此即使 `--offset 0`，全部 12 个关节仍可能运动。
必须准备支撑和活动空间，并使用已针对当前机器人和支撑方式核验的增益，如 `kp=50`、`kd=1`。

```bash
motrix-deploy-unitree joint-control enp5s0 FL_thigh_joint 0.9 \
  --artifact artifacts/go2-walk-flat.deploy
```
`Start` 开始保持姿态过渡，`A` 开始所选关节测试。`B` 返回保持姿态后停止；尚未开始或仍在启动过渡时，`B` 直接取消并阻尼。

### 正式真机运行：

首次运行保持 `kp=50`、`kd=1`，不要在悬空检查前提高增益。为了防止DDS断联Go2摔落，可先按两次L2+A，让机器人趴下，`enp5s0` 换成实际网卡，运行：

```bash
mxlab deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.network_interface=enp5s0
```

按 `Start`，等待姿态过渡完成后按 `A` 接管策略；按住 `L1` 并移动摇杆发送指令。
`B` 请求趴下，`Select` 急停。结束时先发送阻尼，再关闭 DDS。

```{video} /_static/videos/go2_sim2real_h264.mp4
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 50%
:align: center
```
