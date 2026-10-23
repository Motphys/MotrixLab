# 真机部署

本页介绍如何将 `go2-walk-flat` 策略部署到 Unitree Go2：先导出并检查 deployment artifact，再在 MuJoCo 或 MotrixSim 中检查策略，最后按安全流程连接真机。仿真成功不代表真机安全；首次运行必须准备承重支撑、独立急停和现场操作员。

```{video} /_static/videos/go2_hardware_deployment.mp4
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
:align: center
```

## 1. 训练并导出策略

如果还没有训练结果，在仓库根目录执行：

```bash
source .venv/bin/activate
mxlab train task=go2-walk-flat/rslrl.ppo
```

训练结果位于 `runs/go2-walk-flat/`。导出最近一次包含 metadata 的实验，或使用 `run_dir` 固定某次实验；两者只能设置一个。未指定 `output` 时，artifact 默认保存到 `artifacts/<env>.deploy`：

```bash
mxlab export-deploy env=go2-walk-flat
# 或：
mxlab export-deploy run_dir=/path/to/run

mxlab deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

检查结果必须包含 `valid: true`。如需自定义输出位置，再添加 `output=<artifact-path>`。

部署需要安装 `motrix_deploy_tasks`，并安装目标对应的后端插件。部署配方使用 `runtime.transport.network_interface` 指定真机网卡。

## 2. 部署前的仿真检查

先使用 MuJoCo viewer 检查策略：

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy
```

按 `W/S` 前后移动、`A/D` 横移、`Q/E` 转向；关闭窗口、按 Esc 或 Ctrl-C 退出。

安装 `motrix-deploy-motrixsim` 后，也可以复用同一 artifact：

```bash
mxlab deploy task=go2-walk-flat/sim \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim
```

`task=go2-walk-rough/sim` 使用粗糙地形场景，但平地策略不因此就适用于粗糙地形。Go2 平地训练默认使用约 `kp=35, kd=0.5`，真机配方使用 `kp=50, kd=1`；这类差异需要在真机前确认。

## 3. 真机连接与安全准备

开始前必须完成以下准备：

- 使用可靠承重支撑，并确认所有腿有足够活动空间；
- 准备操作员可立即触达的独立急停，并安排第二名操作人员；
- 停止其他所有低层指令发布程序；
- 确认阻尼或部署停止不能维持站立支撑，并准备受支撑的恢复方案；
- 首次运行先让 Go2 悬空，切换到低层/调试模式。

查找连接机器人的有线网卡：

```bash
ip -br addr
```

将下面命令中的 `enp5s0` 替换为实际网卡。PC 与 Go2 需要位于同一网段，例如：

```text
PC 地址：192.168.123.100
Go2 地址：192.168.123.161
```

连接网线后先确认通信：

```bash
ping 192.168.123.161
```

## 4. 策略运行前的只读检查

只读诊断不创建指令 publisher，也不发送电机指令：

```bash
motrix-deploy-unitree read-lowstate enp5s0 --robot go2 --duration-s 10
```

确认持续收到有效状态，并且没有 CRC 或解码错误。官方控制仍活动时，遥控器按键可能触发官方动作，不要将按键检查视为无副作用操作。

## 5. 启动真机策略

首次运行保持 `kp=50`、`kd=1`，不要在悬空检查前提高增益。停止时仍需承重保护；失联、状态超时、Ctrl-C 或其他故障不会执行正常趴下轨迹。

```bash
mxlab deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.transport.network_interface=enp5s0
```

如果官方控制仍活动，部署命令会要求两次确认：仅在准备好授权官方 StandDown 动作时输入 `yes`；观察动作完成且机体已有可靠支撑后，再输入 `down` 授权释放。官方控制已经释放时会跳过这一步。释放后发生错误或中断不会自动恢复官方控制，不要同时启用两个控制器。

准备和操作按键如下：

| 操作 | 作用 |
| --- | --- |
| `Start` | 开始向默认站立姿态过渡。 |
| `A` | 准备完成并确认实测状态后启用策略。 |
| 按住 `L1` 并移动摇杆 | 发送速度指令；首次运行使用小幅指令。 |
| `B` | 向内置趴下姿态过渡，再阻尼结束。尚未开始准备时只执行阻尼。 |
| `Select` | 锁存软件急停阻尼，不能替代独立急停。 |

Go2 遥控器输入由策略任务解释：左摇杆控制平面移动，右摇杆控制转向，必须按住 `L1` 才会发送摇杆指令。结束时先确认低层发布已停止，再按机器人支持的流程恢复官方控制。
