# 基础框架

MotrixLab 将环境实现、仿真后端、训练方法、配置和命令行编排分成独立层。本页自顶向下展开：
先看仿真强化学习的典型流程，再看每个 MotrixLab package 对应该流程的哪部分，最后逐层介绍各部分的能力。

## 仿真强化学习的典型流程

仿真强化学习由**策略（Policy）**与**环境（Environment）**的交互循环驱动，**训练器（Trainer）**
在这个循环之上更新策略：

```{image} /_static/images/tutorial/rl-loop-light.svg
:alt: 仿真强化学习循环：策略输出动作给环境，环境返回观测、奖励与终止标志，训练器收集转移数据并更新策略参数
:class: only-light
```

```{image} /_static/images/tutorial/rl-loop-dark.svg
:alt: 仿真强化学习循环：策略输出动作给环境，环境返回观测、奖励与终止标志，训练器收集转移数据并更新策略参数
:class: only-dark
```

- 策略以环境观测 oₜ 为输入，输出动作 aₜ；
- 环境向量化地并行运行 N 个仿真实例，施加动作并推进物理仿真，产生奖励 rₜ、
  terminated（失败终止）/ truncated（超时截断）与下一观测 oₜ₊₁；
- 下一观测 oₜ₊₁ 回到策略，作为下一步动作的输入；奖励与终止标志**不进入策略网络**，
  它们连同 (oₜ, aₜ) 组成转移数据交给训练器，按所选 RL 算法更新策略参数，如此往复直到收敛。

时序上有一个细节：环境因 aₜ 而产生的观测要到第 t+1 步才成为策略的输入；图中两条交互边统一用
当前周期的 oₜ、aₜ、rₜ 标注。理解了这个循环，MotrixLab 的每个 package 都能在图中找到自己的位置。

## MotrixLab package 与流程的对应

MotrixLab 是一个 UV workspace，由九个 package 组成。按上述流程划分：

| Package             | 流程环节           | 职责                                                             |
| ------------------- | ------------------ | ---------------------------------------------------------------- |
| `motrix_env_core`   | 策略与环境的交互   | backend 无关的环境框架：`EnvCfg`、registry、环境前端与生命周期   |
| `motrix_envs`       | 策略与环境的交互   | 内置环境、机器人模型与任务资产                                   |
| `motrix_rl`         | 策略的训练与更新   | RL 框架集成（provider、trainer）与训练工具                       |
| `configs/`、`scripts/` | 配置与编排      | Hydra 算法基础配置与 Task 配方；train / play / view / export 入口 |
| `motrix_deploy*`    | 策略的部署         | 框架无关 artifact 与运行时契约、MuJoCo 回放与 Unitree 真机后端   |

仿真后端（如 `motrix_env_motrixsim`）通过 `SimBackend` 接口隔离在环境框架之下，使用环境时通常
无需关心它；需要选择或接入仿真后端时，见[编写 DirectEnv 环境](building_envs/direct_env.md)中的
SimBackend 一节。

## 一次训练的完整流程

```bash
python scripts/train.py task=cartpole/skrl.ppo num_envs=1024
```

1. Hydra 组合 `configs/train.yaml`、`configs/algo_base/skrl.ppo.yaml` 和
   `configs/task/cartpole/skrl.ppo.yaml`，`num_envs=1024` 只覆盖本次运行的值。
2. runner 解析 SKRL PPO provider，并从可用的 JAX/Torch 后端中自动选择。
3. trainer 通过环境注册表创建 `cartpole` 环境并开始优化。
4. run metadata、最终 Task 快照、日志和 checkpoint manifest 写入 `runs/cartpole/`。

同一环境可以拥有多份 Task 配方，不需要修改环境实现：

```bash
python scripts/train.py task=cartpole/skrl.ppo
python scripts/train.py task=cartpole/rslrl.ppo
```

## 分层带来的优势

1. **环境复用**：一个注册环境可以由多个 RL 框架训练。
2. **类型化配置**：provider schema 会在训练前拒绝拼写错误或类型不兼容的 YAML/CLI 值。
3. **实验可复现**：每个 run 保存最终 Task 配置和 provider 身份。
4. **多后端**：backend 是配置级选择，环境实现不感知具体仿真器。
5. **易于扩展**：新增环境只需注册环境并添加 Task YAML；新增 RL 集成则添加 provider 与 trainer；
   新增仿真后端则注册新的 SimBackend。
