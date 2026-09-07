# 环境

MotrixLab Environment 定义可通过 Env Registry 创建的仿真任务，包括场景、观测、动作、奖励、终止和重置逻辑。Environment 与可复用的 Robot 模型以及选择训练算法和参数的 Training Task 相互独立。

使用以下命令可以在不启动训练的情况下预览一个环境：

```bash
uv run scripts/view.py env=<env-id>
```

## 环境主题

-   [基础环境](basic/index.md)
-   [DM Control 环境](dm_control/index.md)
-   [全身动作跟踪](whole_body_tracking/index.md)
-   [通用人形速度跟踪环境](humanoid_velocity_tracking.md)
-   [通用四足速度跟踪环境](quadruped_velocity_tracking.md)
-   [其他四足运动控制环境](quadruped_locomotion/index.md)
-   [操作环境](manipulation/index.md)

<!-- ENV_OVERVIEW_TABLE_START -->

<!-- This table is generated; do not edit this block manually. -->
| 预览图 | Env ID | 描述 | 训练算法 |
| --- | --- | --- | --- |
| <img src="../../_static/images/poster/acrobot.jpg" alt="acrobot" width="240"> | `acrobot` | 摆起并平衡欠驱动双连杆机器人。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/anymal_c_navigation_flat.jpg" alt="anymal_c_navigation_flat" width="240"> | `anymal_c_navigation_flat` | 控制 ANYmal-C 在平地上朝目标位置导航。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/anymalc-walk-flat.jpg" alt="anymalc-walk-flat" width="240"> | `anymalc-walk-flat` | 控制 ANYmal-C 在平地上跟踪移动速度指令。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/anymalc-walk-rough.jpg" alt="anymalc-walk-rough" width="240"> | `anymalc-walk-rough` | 控制 ANYmal-C 在程序化粗糙高度场上跟踪行走指令。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/bounce_ball.jpg" alt="bounce_ball" width="240"> | `bounce_ball` | 控制球拍连续颠起乒乓球。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/cartpole.jpg" alt="cartpole" width="240"> | `cartpole` | 移动小车以保持倒立摆直立。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dex-evt-walk-flat.jpg" alt="dex-evt-walk-flat" width="240"> | `dex-evt-walk-flat` | 控制 Dex-EVT 人形机器人在平地上行走。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/dex-evt-walk-rough.jpg" alt="dex-evt-walk-rough" width="240"> | `dex-evt-walk-rough` | 控制 Dex-EVT 人形机器人在起伏地形上行走。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/dex-evt-wbt-dance.jpg" alt="dex-evt-wbt-dance" width="240"> | `dex-evt-wbt-dance` | 让 Dex-EVT 跟踪内置舞蹈参考动作。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/dm-cheetah.jpg" alt="dm-cheetah" width="240"> | `dm-cheetah` | 驱动平面 Cheetah 尽可能快速地向前奔跑。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-finger-spin.jpg" alt="dm-finger-spin" width="240"> | `dm-finger-spin` | 使用机械手指持续旋转自由物体。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-finger-turn-easy.jpg" alt="dm-finger-turn-easy" width="240"> | `dm-finger-turn-easy` | 将机械手指上的物体转到宽容差目标角度。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-finger-turn-hard.jpg" alt="dm-finger-turn-hard" width="240"> | `dm-finger-turn-hard` | 将机械手指上的物体精确转到目标角度。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-hopper-hop.jpg" alt="dm-hopper-hop" width="240"> | `dm-hopper-hop` | 让单腿 Hopper 保持直立并向前跳跃。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-hopper-stand.jpg" alt="dm-hopper-stand" width="240"> | `dm-hopper-stand` | 让单腿 Hopper 保持直立站立。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-humanoid-run.jpg" alt="dm-humanoid-run" width="240"> | `dm-humanoid-run` | 控制 DM Control 人形机器人高速向前奔跑。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-humanoid-stand.jpg" alt="dm-humanoid-stand" width="240"> | `dm-humanoid-stand` | 控制 DM Control 人形机器人保持站立。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-humanoid-walk.jpg" alt="dm-humanoid-walk" width="240"> | `dm-humanoid-walk` | 控制 DM Control 人形机器人向前行走。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-lqr-2-1.jpg" alt="dm-lqr-2-1" width="240"> | `dm-lqr-2-1` | 稳定具有 2 维状态和 1 维控制的线性系统。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-lqr-6-2.jpg" alt="dm-lqr-6-2" width="240"> | `dm-lqr-6-2` | 稳定具有 6 维状态和 2 维控制的线性系统。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-manipulator-bring-ball.jpg" alt="dm-manipulator-bring-ball" width="240"> | `dm-manipulator-bring-ball` | 控制机械臂把球移动到目标位置。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-escape.jpg" alt="dm-quadruped-escape" width="240"> | `dm-quadruped-escape` | 控制 DM Control 四足机器人逃离碗形区域。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-fetch.jpg" alt="dm-quadruped-fetch" width="240"> | `dm-quadruped-fetch` | 控制 DM Control 四足机器人寻找并取回球。 | `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-run.jpg" alt="dm-quadruped-run" width="240"> | `dm-quadruped-run` | 控制 DM Control 四足机器人高速向前奔跑。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-walk.jpg" alt="dm-quadruped-walk" width="240"> | `dm-quadruped-walk` | 控制 DM Control 四足机器人向前行走。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-reacher.jpg" alt="dm-reacher" width="240"> | `dm-reacher` | 控制双关节机械臂末端到达随机目标。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-runner.jpg" alt="dm-runner" width="240"> | `dm-runner` | 控制平面 Walker 以较高目标速度奔跑。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-stander.jpg" alt="dm-stander" width="240"> | `dm-stander` | 控制平面 Walker 保持直立站立。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-walker.jpg" alt="dm-walker" width="240"> | `dm-walker` | 控制平面 Walker 以目标速度向前行走。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/franka-lift-cube.jpg" alt="franka-lift-cube" width="240"> | `franka-lift-cube` | 控制 Franka 机械臂抓取并抬升立方体。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/franka-open-cabinet.jpg" alt="franka-open-cabinet" width="240"> | `franka-open-cabinet` | 控制 Franka 机械臂抓住把手并打开抽屉。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/g1-29dof-wbt-largebox.jpg" alt="g1-29dof-wbt-largebox" width="240"> | `g1-29dof-wbt-largebox` | 让 Unitree G1 跟踪搬运大箱子的参考动作。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/g1-walk-flat.jpg" alt="g1-walk-flat" width="240"> | `g1-walk-flat` | 控制 Unitree G1 在平地上跟踪行走指令。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/g1-walk-rough.jpg" alt="g1-walk-rough" width="240"> | `g1-walk-rough` | 控制 Unitree G1 在起伏地形上跟踪行走指令。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/g1-wbt-dance.jpg" alt="g1-wbt-dance" width="240"> | `g1-wbt-dance` | 让 Unitree G1 跟踪内置舞蹈参考动作。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/go1-stairs-terrain-walk.jpg" alt="go1-stairs-terrain-walk" width="240"> | `go1-stairs-terrain-walk` | 控制 Unitree Go1 在台阶地形上行走。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go1-walk-flat.jpg" alt="go1-walk-flat" width="240"> | `go1-walk-flat` | 控制 Unitree Go1 在平地上跟踪行走指令。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go1-walk-rough.jpg" alt="go1-walk-rough" width="240"> | `go1-walk-rough` | 控制 Unitree Go1 在程序化粗糙高度场上跟踪行走指令。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go2-walk-flat.jpg" alt="go2-walk-flat" width="240"> | `go2-walk-flat` | 控制 Unitree Go2 在平地上跟踪行走指令。 | `motrix.fastsac`, `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go2-walk-rough.jpg" alt="go2-walk-rough" width="240"> | `go2-walk-rough` | 控制 Unitree Go2 在程序化粗糙高度场上跟踪行走指令。 | `motrix.fastsac`, `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/k1-walk-flat.jpg" alt="k1-walk-flat" width="240"> | `k1-walk-flat` | 控制 Booster K1 在平地上跟踪行走指令。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/k1-walk-rough.jpg" alt="k1-walk-rough" width="240"> | `k1-walk-rough` | 控制 Booster K1 在起伏地形上跟踪行走指令。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/k1-wbt-freekick.jpg" alt="k1-wbt-freekick" width="240"> | `k1-wbt-freekick` | 让 Booster K1 跟踪任意球射门参考动作。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/microduck-walk-flat.jpg" alt="microduck-walk-flat" width="240"> | `microduck-walk-flat` | 控制 Microduck 小型双足机器人在平地上跟踪行走指令。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/microduck-walk-rough.jpg" alt="microduck-walk-rough" width="240"> | `microduck-walk-rough` | 控制 Microduck 小型双足机器人在起伏地形上跟踪行走指令。 | `motrix.fastsac` |
| <img src="../../_static/images/poster/peg-insert.jpg" alt="peg-insert" width="240"> | `peg-insert` | 控制 RM65 抓取插销、对准插座并完成插入。 | `motrix.fastsac`, `skrl.ppo` |
| <img src="../../_static/images/poster/pendulum.jpg" alt="pendulum" width="240"> | `pendulum` | 施加关节力矩使单摆摆起并保持直立。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/point_mass.jpg" alt="point_mass" width="240"> | `point_mass` | 控制二维质点移动到随机目标位置。 | `skrl.ppo` |
| <img src="../../_static/images/poster/rm65-open-cabinet.jpg" alt="rm65-open-cabinet" width="240"> | `rm65-open-cabinet` | 控制 RM65 抓住把手并拉开柜体底部抽屉。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/rm65_insert_peg.jpg" alt="rm65_insert_peg" width="240"> | `rm65_insert_peg` | 控制 RM65 抓取插销、对准插座并完成插入。 | `motrix.fastsac`, `skrl.ppo` |
| <img src="../../_static/images/poster/shadow-hand-repose.jpg" alt="shadow-hand-repose" width="240"> | `shadow-hand-repose` | 控制 Shadow Hand 在手中重定向立方体。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/stewart.jpg" alt="stewart" width="240"> | `stewart` | 控制 Stewart 平台在无外部扰动时保持水平稳定。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/stewart-disturb-xy.jpg" alt="stewart-disturb-xy" width="240"> | `stewart-disturb-xy` | 控制 Stewart 平台在 XY 扰动下恢复并保持水平。 | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/stewart-static.jpg" alt="stewart-static" width="240"> | `stewart-static` | 控制 Stewart 平台在无外部扰动时保持水平稳定。 | `rslrl.ppo`, `skrl.ppo` |

<!-- ENV_OVERVIEW_TABLE_END -->
