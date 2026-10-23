# Go2 平地行走：从训练到真机部署

使用同一份 deployment artifact，先在仿真中检查 Go2 策略，再尝试真机部署。已有 artifact 并安装任务与 MuJoCo 插件后，可先做有界无窗口检查：

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy runtime.viewer=false duration_s=2.0
```

下面按训练、导出、仿真、人工确认真机部署和基础诊断的顺序操作。仿真成功不代表真机安全。

## 1. 安装与训练

在仓库根目录执行：

```bash
sh install.sh --all
source .venv/bin/activate
mxlab train task=go2-walk-flat/rslrl.ppo
```

训练结果保存在 `runs/go2-walk-flat/`。已有包含 metadata 的 run 时可跳过训练。
导出需要该 run 使用的训练 provider 和环境包；部署需要任务包及目标对应的后端插件。

## 2. 导出与检查

```bash
mxlab export-deploy env=go2-walk-flat
mxlab deploy inspect artifact=artifacts/go2-walk-flat.deploy
```

`env=go2-walk-flat` 选择最近一次包含 metadata 的 run。用 `run_dir=<run-dir>` 替代可选择指定 run；两者必须且只能设置一个。
已有 artifact 目录不会被覆盖，再次导出时应选择新的输出路径。确认检查结果为 `valid: true`。

artifact 包含策略及解释策略所需的机器人和任务设置。迁移时复制完整目录，不要只复制 ONNX 文件。
`mxlab export-onnx` 导出独立模型，不能代替 deployment artifact 导出。

## 3. 在仿真中检查

打开 MuJoCo viewer：

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

按住 `W/S` 前后移动，`A/D` 横移，`Q/E` 转向。关闭窗口、按 Esc 或 Ctrl+C 退出。
无窗口模式使用零速度指令。`duration_s` 限制策略执行时长，而非整个进程的墙钟时间；启动和停止可能额外耗时。

安装原生 MotrixSim 部署插件后，可复用同一 artifact 和配方：

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.backend=motrixsim
```

使用 `task=go2-walk-rough/sim` 可选择随包提供的粗糙地形场景。更换场景不代表平地策略适合粗糙地形，测试前需确认策略适用范围。

## 4. 运行 Go2 真机

### 准备机器人与连接

准备可靠承重支撑、所有腿的活动空间、操作员就绪的独立急停，以及另一名操作人员。停止其他所有低层指令发布程序。
阻尼和部署停止都不能维持站立支撑；没有受支撑的恢复方案时不要继续。

查找连接机器人的有线网卡：

```bash
ip -br addr
```

将下面的 `enp5s0` 替换为实际网卡名。先执行第 5 节的只读检查，再启动部署：

```bash
mxlab deploy task=go2-walk-flat/hardware \
  artifact=artifacts/go2-walk-flat.deploy \
  runtime.transport.network_interface=enp5s0
```

### 交接官方控制权

官方控制仍活动时，命令要求两次确认：

1. 仅在准备好授权官方 StandDown 动作时输入 `yes`。
2. 观察动作确实结束、机体已有可靠支撑，再输入 `down` 授权释放。

StandDown 请求被接受不代表动作已经完成。官方控制已释放时，会跳过这些提示和官方动作。
在确认处拒绝、EOF 或 Ctrl+C 会中止部署；在 `down` 处取消会保留官方控制权，但不会撤销 StandDown。

**退出后不自动恢复官方控制。** 释放后的报错或中断也不会恢复控制权。这套交接与确认仅适用于 Go2，不是 G1 的启动流程。

### 准备、接管与停止

| 操作 | 效果 |
| --- | --- |
| Start | 开始向默认站立姿态过渡。 |
| A | 实测准备就绪后启用策略；需在准备完成后重新按下，提前按住不能绕过检查。 |
| 按住 L1 并移动摇杆 | 发送运动指令，先在保护下使用小幅指令。 |
| B | 准备启动后，向内置趴下姿态过渡，再阻尼结束；尚未按 Start 时仅阻尼结束。 |
| Select | 锁存软件急停阻尼，不能替代独立急停。 |

B 的目标受 artifact 关节限位约束。启动时若提示目标受限，实际可能只达到浅趴，不能保证机体贴地或稳定性。
停止时仍需承重保护。失联、状态超时、Ctrl+C 和其他故障**不执行正常趴下轨迹**。

退出后，先确认所有低层发布程序均已停止，再按机器人支持的流程恢复官方控制，不要同时启用两个控制器。

## 5. 策略运行前的诊断

### 只读状态，不发送运动指令

```bash
motrix-deploy-unitree read-lowstate enp5s0 --robot go2 --duration-s 10
```

检查是否持续收到有效状态，以及是否有 CRC 或解码错误。该工具不发送电机指令，也不释放官方控制权。
官方控制仍活动时，遥控器按键可能同时触发官方动作，不要将按键检查视为无副作用操作。

## 6. 常用设置与故障处理

在选定任务后附加覆写参数，例如：

```bash
mxlab deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy \
  runtime.viewer=false runtime.realtime=true duration_s=5.0
```

| 设置 | 用途 |
| --- | --- |
| `artifact` | 选择完整的部署 artifact 目录。 |
| `duration_s` | 限制策略执行时长，不是墙钟超时。 |
| `runtime.backend` | 仿真选择 `mujoco` 或 `motrixsim`；Unitree 使用真机配方。 |
| `runtime.viewer`、`runtime.realtime` | 仅用于仿真显示和节奏控制；真机始终按实时节奏执行。 |
| `runtime.transport.network_interface` | 真机必填的有线网卡。 |
| `task_settings.preparation` | Go2 准备动作的增益、时序、就绪阈值与按键；不覆盖策略增益。 |

调整真机准备参数前，先检查机械活动空间、支撑和报错中的实测误差。外观合理不代表可以安全接管策略。
不要通过放宽通信超时掩盖状态缺失。

| 现象 | 下一步检查 |
| --- | --- |
| 网卡名称与可用接口不匹配 | 在运行部署的机器或环境内执行 `ip -br addr`。 |
| 无有效状态，或有 CRC／解码错误 | 停留在只读诊断，检查网络与设备设置。 |
| 站立准备超时 | 检查报错中的关节、速度和倾角；调参前排除干涉与异常承重。 |
| 执行期间 `state_timeout` | 按故障停机处理，保留结果与错误，确保支撑，排查接收后再重试。 |

## 7. 其他任务与 Python 示例

G1 WBT 舞蹈使用自身的 artifact 与任务配方，仿真检查示例：

```bash
mxlab deploy task=g1-wbt-dance/sim artifact=artifacts/g1-wbt-dance.deploy runtime.viewer=false
```

artifact 提供 motion 片段，无需速度指令，执行时长不能超过片段长度。播放结束进入阻尼，不保证恢复站立。
不要把 Go2 按键或交接流程套用于 G1。

在仓库根目录运行程序化 Python 示例，不读取 artifact，也不调用部署 CLI：

```bash
python examples/deploy.py --headless --steps 100
python examples/deploy.py --backend motrixsim --headless --steps 100
```

完成第 4 节的真机安全准备后，可显式选择真机：

```bash
python examples/deploy.py --real --network-interface enp5s0
```

示例使用原始 ONNX 和代码中明确声明的机器人与任务参数。替换模型时必须同步核对训练契约，不能仅凭输入输出维度匹配就上真机。
真机路径保留控制权交接和遥控器确认；CLI 部署仍使用前述 artifact 工作流。
