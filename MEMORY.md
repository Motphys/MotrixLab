# MotrixLab 协作记忆

> 用途：记录跨会话协作中稳定、可复用的项目上下文。每当完成重要修改、确认新的架构约定、发现环境问题或改变下一步计划时，追加更新；不要把一次性的猜测当成事实。

## 当前状态

- 项目：MotrixLab，基于 MotrixSim 的强化学习框架，采用 UV workspace，Python 3.10。
- 当前工作区：`/home/pc2/lab/MotrixLab`。
- 当前请求：建立并持续维护本记忆文件；尚未指定具体代码修改任务。
- 记录日期：2026-10-09（Asia/Shanghai）。

## 项目结构速记

- `motrix_env_core`：backend 无关的环境框架、配置、registry。
- `motrix_env_motrixsim`：MotrixSim backend、scene compiler、renderer、Torch frontend。
- `motrix_env_mujoco`：编译用 MuJoCo scene backend。
- `motrix_robots`：机器人配置、模型和资产。
- `motrix_envs`：内置环境、任务和运动资产。
- `motrix_rl`：训练控制平面、runner、checkpoint、backend/provider/trainer contracts。
- `motrix_rl_builtin`、`motrix_rl_skrl`、`motrix_rl_rslrl`：内置 FastSAC、SKRL、RSLRL provider 插件。
- `motrix_deploy` 及其插件：artifact、运行时契约、仿真/Unitree 部署后端和任务。
- `scripts/`：训练、查看、评估、ONNX 导出入口；`wiki/`：设计与研究文档；`tests/`：集成测试。

## 必须遵守的约定

- 环境配置使用 `@configclass`，直接工作流继承 `DirectEnvCfg`，Manager 工作流继承 `ManagerBasedEnvCfg`；runtime/result/metadata 使用 `@dataclass`。
- 环境配置先注册（`@registry.envcfg`），再注册环境类（`@registry.env`）；环境类继承受支持的运行时基类。
- core 不得 import 具体 simulator；backend 通过配置字符串和 registry 选择；环境通过 backend-neutral `self.sim` 工作。
- Manager fused-kernel 入口使用 `@dispatch`，不要使用 `@njit`；`@njit` 仅用于独立 helper。
- `observation_space`、`action_space` 用 property；常量预计算；Array 环境不得按 environment 维度写 Python 循环。
- provider 配置必须使用插件自己的 dataclass，并与 `configs/algo_base/*.yaml` 对齐；不要恢复已删除的旧 `@rlcfg` 结构。
- workspace package 的版本号和 MotrixSim 依赖版本必须一致；关键第三方依赖使用精确锁定（`===`）。
- 默认保持 `__init__.py` 精简；只有稳定公共 API 或注册副作用才保留 re-export/import。
- 修改后按风险运行相关 pytest；提交前运行 `prek run --all-files`。
- 避免裸 `uv run` / `uv sync`；先激活 `.venv`，或使用 `uv run --no-sync`。

## 常用验证命令

```bash
source .venv/bin/activate
python -m pytest
prek run --all-files
```

训练入口示例：

```bash
python scripts/train.py task=cartpole/skrl.ppo
python scripts/train.py task=cartpole/rslrl.ppo
python scripts/view.py env=cartpole
python scripts/play.py env=cartpole
```

## 已知工作区信息

- 当前未发现既有 `MEMORY.md`、`*context*` 或 `*notes*` 文件，因此本文件作为仓库级协作记忆入口。
- 本次只创建记忆文件，没有修改源代码或配置。
- `git status` 目前会触发仓库内 Git LFS clean filter 错误（`motrix_robots/.../Left_Ankle_Cross.STL`，提示 read-only filesystem / broken pipe）；后续如需确认变更状态，应优先使用定向的 `git diff -- MEMORY.md`、`git diff --no-ext-diff -- MEMORY.md` 或直接检查文件，避免把该环境问题误判为代码问题。

## RSLRL PPO 代码地图

- 训练入口链路：`scripts/train.py` → `motrix_rl.runner` → RSLRL provider → `motrix_rl_rslrl/torch/train/ppo.py:Trainer.train` → 外部 `rsl_rl.runners.OnPolicyRunner.learn`。
- 仓库内 RSLRL 集成核心是 `motrix_rl_rslrl/src/motrix_rl_rslrl/torch/train/ppo.py`；真正的 PPO 更新（ratio clipping、GAE、value/entropy loss、optimizer step）属于外部 `rsl-rl-lib` 的 `PPO` 实现。
- 算法配置源是 `configs/algo_base/rslrl.ppo.yaml`；类型化配置在 `motrix_rl_rslrl/src/motrix_rl_rslrl/cfg.py`。
- 环境边界适配在 `motrix_rl_rslrl/src/motrix_rl_rslrl/torch/wrap_torch.py` 与 `wrap_np.py`，负责 VecEnv、obs/action/reward/done/time_outs 转换。
- 当前默认关键 PPO 参数：`num_steps_per_env=24`、`num_learning_epochs=5`、`num_mini_batches=4`、`clip_param=0.2`、`gamma=0.99`、`lam=0.95`、`desired_kl=0.008`；默认 actor/critic 为 ELU MLP `[256,128,64]`。
- 原理速记：并行环境采集 on-policy rollout → 用 GAE 计算 advantage/return → 多 epoch、小批次优化 clipped surrogate objective，同时训练 critic、加入 entropy bonus，并按 KL 自适应调学习率。

## 任务的 observation/reward 代码地图

- observation/reward 不在 RSLRL PPO 文件里，而在具体环境任务（`motrix_envs`）中；RSLRL 只从 VecEnv wrapper 接收 `obs/reward/done`。
- 通用 Direct 环境接口由 `motrix_env_core` 提供；具体任务实现 `compute_observation`、`compute_transition` 或 `_compute_reward`。
- CartPole 示例：`motrix_envs/src/motrix_envs/basic/cartpole/cartpole_np.py`。观测为 `[cart_pos, pole_angle, cart_vel, pole_vel]`（`compute_observation`），reward 每步为 `1`，越界/倒下时 `terminated`（`compute_transition`）。空间和 reset 配置在同目录 `cfg.py`。
- 四足行走示例：`motrix_envs/src/motrix_envs/locomotion/quadruped/walk_np.py`。policy observation 由 gyro、gravity、关节位置差、关节速度、上一动作、速度指令、足端相位拼接；value observation 额外包含 local linear velocity。reward 在 `_compute_reward` 中按 `RewardScales` 加权各项，再乘 `ctrl_dt`。
- 四足 reward 的权重/目标/噪声配置在 `motrix_envs/src/motrix_envs/locomotion/quadruped/cfg.py`；Go1 任务的覆盖值在 `motrix_envs/src/motrix_envs/locomotion/quadruped/go1.py`。
- RSLRL 配置的 `obs_groups.actor/critic`（`configs/algo_base/rslrl.ppo.yaml`）决定 runner 从 wrapper 提供的 `policy`/`value` observation 组取哪些输入；它不是 observation 内容的定义处。
- 当前四足后退/左右步态不对称的排查重点：先验证 `Sensor.foot_positions`、四个 contact sensor 与 `robot.foot_contact_geom_names`/关节顺序是否一致；再检查 `trot_pairs=((0,3),(1,2))`、足端相位奖励和默认姿态。仅调 PPO 学习率通常不能修复固定方向偏差。
- 侧向运动与转向要区分：纯 `v_y` 侧移应保持左右镜像，不需要固定的左右前后偏置；有 `ω_z` 转向时，因各腿的横向坐标不同，左右侧出现随 `ω_z` 变化、并随转向方向反号的前后落脚差是正常的。固定方向、与命令无关的偏移通常不是必要步态，而应排查映射/奖励/动力学偏差。
- 运动学判断：对腿/足端相对机体位置 `r_i=(x_i,y_i)`，平面期望速度为 `v_i=(v_x-ω_z*y_i, v_y+ω_z*x_i)`。纯侧移 `v_x=0,ω_z=0` 时各腿期望平移分量相同；纯转向或侧移叠加转向时，左右/前后腿才会因半径不同产生有符号的步幅差，且反向转向应反号。
- Go2 具体核对结果：`UnitreeGo2Robot` 的 key pose 顺序为 FL、FR、RL、RR，默认角度为前腿 `(0.0,0.8,-1.5)`、后腿 `(0.0,1.0,-1.5)`，左右镜像一致；`go2_mjx.xml` 的 position actuator 顺序也为 FL、FR、RL、RR。因此“左右两腿共同后移”不太像 Go2 默认姿态的左右不对称，更应先隔离随机 lateral/yaw command、验证 body-frame 坐标和加入左右/纵向足端对称约束。
- Go2 flat 的 play 配置目前只关闭随机化、观测噪声和 command resampling，不会把初始 command 固定为直行；`commands.velocity` 仍从 `vx∈[-1,1]、vy∈[-0.5,0.5]、wz∈[-0.5,0.5]` 采样。因此播放时若未显式固定 command，左/右足端纵向差可能只是非零侧向/偏航指令的结果。
- 已将 `go2-walk-flat` 的 `for_play()` 固定为 command `[0.5, 0.0, 0.0]`，并关闭站立概率；`go2-walk-rough` 显式保留原有随机 command 范围，避免该修改影响粗糙地形任务。
- 新增 `scripts/play_walk.py` 与 `configs/play_walk.yaml`：复用 metadata/policy 查找，固定四足 command 为 `[forward_velocity, lateral_velocity, yaw_velocity]`，默认直行 `[0.5, 0.0, 0.0]`，默认运行 100 秒后退出；仅支持 RSLRL PPO walk 环境。

## 更新规则

1. 每次完成实质性代码、配置、测试或文档工作后，在“更新日志”追加一条带日期的记录。
2. 记录事实、决策、验证结果和未解决阻塞；不记录敏感凭据，不复制大段源码。
3. 若旧信息失效，修改对应章节并在日志中说明原因；不要只追加互相矛盾的陈述。
4. 新会话开始先读本文件和仓库 `AGENTS.md`，再检查相关文件与当前工作区状态。

## 更新日志

### 2026-10-09

- 创建 `MEMORY.md`，建立项目结构、核心编码约定、验证命令、已知环境问题和持续更新规则。
- 梳理 RSLRL PPO：确认仓库核心是集成 Trainer/配置/VecEnv wrapper，算法更新本体在外部 `rsl-rl-lib`。
- 将 Go2 flat 播放 command 固定为 `[0.5, 0, 0]`，并通过运行时配置检查确认；未运行完整训练。
- 使用现有 `runs/go2-walk-flat/rslrl/torch/ppo/26-10-09_15-03-18-244722/model_999.pt` 在 CPU 上做了 240 步无渲染 rollout，确认 command 始终为 `[0.5,0,0]`；足端 body-frame x 的 `left_mean - right_mean` 约 `+0.0031 m`（左侧略靠前约 3 mm），未复现“左侧始终向后”的偏差。该结果是单 checkpoint/仿真统计，仍需用户视觉复测。
- 新增有界直行验证入口 `scripts/play_walk.py`，并通过 `python3 -m py_compile`、Hydra `--cfg job` 配置解析检查。
- 修复 `scripts/play_walk.py` 中未使用的 `RenderConfig` import；Python 语法检查通过，当前环境未安装可执行的 Ruff。

### 2026-10-10

- `scripts/play_walk.py` 已增加四足落地事件分析：按 FL/FR/RL/RR 的接触上升沿记录 step、时间和 body-frame 足端 `(x, y)`，并在结束时输出 `foot_landings.csv` 与无第三方绘图库依赖的 `foot_landings.svg` 平面曲线图；默认目录为所选 run 下的 `walk_analysis/`，可用 `analysis_dir` 和 `analysis_env_index` 覆盖。
- 使用现有 Go2 RSLRL checkpoint 做 5 秒、单环境运行验证，生成了 `/tmp/motrixlab-foot-analysis/foot_landings.csv` 和 `/tmp/motrixlab-foot-analysis/foot_landings.svg`，四条腿均产生了落地记录。该分析记录的是落地事件位置，不是每个仿真步的完整足端轨迹。
- 将 `play_walk` 的默认 `forward_velocity` 与 `go2-walk-flat.for_play()` 统一为 `0.5`，因此默认 command 为 `[0.5, 0.0, 0.0]`；仍可通过 Hydra 参数显式覆盖。
- 验证：`python3 -m py_compile scripts/play_walk.py`、`git diff --check` 和辅助 CSV/SVG 输出测试通过；尚未运行完整 pytest，当前环境没有可执行 Ruff。
- 默认速度统一后的 5 秒单环境 rollout 再次验证通过：日志显示 command `[0.500, 0.000, 0.000]`，生成 `/tmp/motrixlab-foot-analysis-default5/foot_landings.csv`（6 条事件）和对应 SVG。无显示环境时需使用 `record_video=true` 走 headless renderer；`record_video=false` 会触发现有 renderer 的 DISPLAY/WAYLAND 依赖。
- 足端落地 CSV/SVG 中的 `(x, y)` 来自 `env.feet_pos`，对应 `framepos` 相对于 IMU/body 的机体坐标；因此机器人整体前进时，单腿在该坐标系中的落地点仍会周期性回到近似固定的腿部工作空间，不能用其绝对 x 变化判断世界坐标中的前进距离。若需画真实前进轨迹，应同时记录 `env.sim_data["base_pos"]`（机体世界位置），或将足端 body-frame 坐标用机体姿态变换到世界坐标后再绘图。
- `play_walk.py` 现增加左右偏置诊断输出：`left_right_bias.csv` 记录前/后腿的 `left_minus_right_x`（左脚纵向位置减右脚）与 `left_plus_right_y`（左右镜像后的横向误差），`left_right_bias.png` 用四联图展示 body-frame 落点、各脚 x 分布、左右 y 镜像分布和偏置汇总。5 秒 rollout 的示例指标为前腿 Δx=`+0.0075 m`、后腿 Δx=`+0.0077 m`，横向镜像误差分别 `-0.0004 m`、`-0.0026 m`；这是单 checkpoint/短时仿真结果。
- 在根 workspace 增加精确版本 `matplotlib===3.10.9` 并更新 `uv.lock`；当前 `.venv` 已安装 matplotlib 3.10.9。绘图使用 Agg 后端，适合无显示环境。
