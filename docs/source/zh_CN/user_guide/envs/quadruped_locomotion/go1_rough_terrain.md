# Unitree GO1 固定台阶行走

`go1-stairs-terrain-walk` 使用独立的 `Go1WalkStairsTask` 和 `scene_stairs_terrain.xml` 固定台阶场景。
它不复用 `QuadrupedWalkTask`；通用的 GO1 平地与程序化粗糙地形速度跟踪任务见
[“通用四足速度跟踪环境”](../quadruped_velocity_tracking.md)。

```{video} /_static/videos/go1_stairs_terrain_walk.mp4
:poster: _static/images/poster/go1_stairs_terrain_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

`go1-stairs-terrain-walk` 的 12 维动作直接使用执行器控制范围，随后以 `action_scale=0.05` 缩放为默认姿态
附近的 PD 目标。60 维观察依次包含机体线速度、陀螺仪、局部重力、12 维关节位置偏差、12 维关节速度、
12 维上一动作、3 维速度命令，以及四足各 3 维的接触力。速度命令只采样前向速度 $v_x\in[0.5,1.0]$ m/s，
$v_y$ 和 `yaw_rate` 固定为 0。

重置位置在固定场景的 25 个预设位置间循环，关节恢复模型默认姿态，速度和历史缓冲清零。躯干接触任一地面
几何体或水平线速度平方和超过 $10^8$ 时产生 failure termination；默认 `20 s` 时限产生 truncation。
该任务不随机化物理参数。

```bash
python scripts/view.py env=go1-stairs-terrain-walk
python scripts/train.py task=go1-stairs-terrain-walk/skrl.ppo
python scripts/play.py env=go1-stairs-terrain-walk
```
