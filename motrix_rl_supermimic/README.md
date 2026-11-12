# motrix_rl_supermimic

SuperMimic 风格 world-model MBRL 的 MotrixLab provider 插件（开发中）。

## 能力

- **TaskMDP 契约**（`task_mdp.py`）：任务“解析 MDP heads”——observation / reward /
  soft_termination / critic_observation / command。learner 从头到尾只消费该协议；
  它是 learner 与任务之间的稳定 ABI：真实路径（仿真环境）与想象路径（world model
  rollout）共用同一份评价函数。设计纪律：纯函数、无 env/sim 句柄、无隐藏状态、
  设备无关、可 `torch.compile`。
- **MbrlEnvBase**（`mbrl_env.py`）：DirectEnv 生命周期全套实现——action 记录、
  reward/termination/observation 全部委托 TaskMDP，一致性从“约定”变成“结构”。
  子类只写 sim 接线与 reset 分布。

新任务接入 = 实现 TaskMDP + 继承 MbrlEnvBase 补 sim 接线 + 一份 yaml。

任务必须满足“可想象性”前提：状态为低维物理状态元组（无 pixel/点云）；reward 与
termination 是状态元组 + 可微 FK 的解析函数；命令源可由 frame 解析推进。

## 边界

- 依赖方向：`任务包 → motrix_rl_supermimic → motrix_env_core`。本包不 import 任何
  任务包或具体机器人常量即可构建与测试（测试使用玩具 TaskMDP）。
- 想象 rollout（BPTT）发生在 learner 进程内，可微、不落盘；不为想象数据引入跨进程通道。

运行时契约（DirectEnv/ArrayEnv 生命周期口径）见 `motrix_env_core` 文档；
可微 FK 见 `motrix_env_core.math.kinematics`。
