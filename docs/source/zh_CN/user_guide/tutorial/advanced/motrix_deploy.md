# Go2 平地行走：从训练到真机部署

这篇教程用 Go2 平地行走任务演示完整流程：训练策略、导出 artifact、先在 MuJoCo 中检查，最后连接真机运行。
如果已经有训练好的 run，可以直接从“导出 artifact”开始。

## 1. 安装环境

在仓库根目录执行：

```bash
sh install.sh --all
```

这个命令会安装训练、MuJoCo、ONNX Runtime 和 Unitree SDK2 依赖。

## 2. 训练策略

没有现成 run 时，训练 flat-terrain Go2 策略：

```bash
python scripts/train.py task=go2-walk-flat/rslrl.ppo
```

训练结果保存在 `runs/go2-walk-flat/`。

## 3. 导出 artifact

导出最近一次 run：

```bash
python scripts/export_deploy.py env=go2-walk-flat
```

输出目录为 `artifacts/go2-walk-flat.deploy/`。artifact 是部署时唯一需要带走的策略文件，包含模型和运行所需的配置。

## 4. 检查 artifact

```bash
motrix-deploy inspect \
  artifact=artifacts/go2-walk-flat.deploy
```

确认输出中的 `valid` 为 `true`。

## 5. 先跑 Sim2Sim

```bash
motrix-deploy sim2sim \
  --config-name go2_walk_flat_sim2sim \
  artifact=artifacts/go2-walk-flat.deploy
```

会打开 MuJoCo viewer。按 `W/S` 前后移动，`A/D` 横移，`Q/E` 转向；关闭窗口、按 Esc 或 Ctrl-C 退出。

## 6. 运行真机

先确认机器人处于低层/调试模式，网线已连接，并准备好急停。
将 `enp5s0` 换成实际网卡名称：

```bash
motrix-deploy inspect \
  artifact=artifacts/go2-walk-flat.deploy

motrix-deploy sim2real \
  --config-name go2_walk_flat_sim2real \
  artifact=artifacts/go2-walk-flat.deploy \
  backend.network_interface=enp5s0 \
  hardware.confirm=true
```

启动后按遥控器 Start，完成默认姿态过渡后按 A；按住 L1 并移动左右摇杆，可发送移动命令。按 B 进入趴下流程，Select 触发急停。

发送策略前，也可以先查看机器人状态：

```bash
python -m motrix_deploy_unitree.read_lowstate enp5s0
```

## 进阶说明

### 显式覆盖策略 PD 增益

真机父配置 `configs/deploy/sim2real/base.yaml` 当前设置 `backend.kp=50`、`backend.kd=1`，会覆盖 artifact
`TaskSpec.config` 中保存的增益。把二者显式设为 `null` 才会保留 artifact 增益；也可以在命令行传入非负标量统一
覆盖 12 个关节：

```bash
motrix-deploy sim2real \
  artifact=artifacts/go2-walk-flat.deploy \
  backend.network_interface=enp5s0 \
  'backend.kp=[20,25,30,20,25,30,22,27,32,22,27,32]' \
  'backend.kd=[0.3,0.4,0.5,0.3,0.4,0.5,0.35,0.45,0.55,0.35,0.45,0.55]' \
  hardware.confirm=true
```

### 只读检查 LowState 数据

发送任何运动指令前，可先运行只读诊断：

```bash
python -m motrix_deploy_unitree.read_lowstate enp5s0
```

### 发送单关节运动指令

如需绕过策略、调试一个关节的位置运动，可使用随 `motrix_deploy_unitree` 安装的有界运动脚本。默认从当前
`go2-walk-flat` deployment profile 构建控制契约，不需要训练 run、checkpoint、策略或 deployment artifact：

```bash
python -m motrix_deploy_unitree.go2_joint_control \
  enp5s0 \
  FL_thigh_joint \
  0.9 \
  --hardware-confirm
```
