# motrix_rl_supermimic

SuperMimic 风格 world-model MBRL 的独立实现包与 MotrixLab provider（开发中）。

## 安装与集成

本包提供配置、provider、训练器与任务接入接口，不自行注册 framework。
安装 `motrix-rl-builtin` 会同时安装本包，由 builtin 统一注册为 `motrix.supermimic`；
配置类型与算法默认值仍由本包维护。

项目中的 G1 tracking 任务通过统一 CLI 启动：

```bash
mxlab train task=g1-supermimic-tracking/motrix.supermimic
```

## 能力

- **异步训练**：独立 collector 进程运行真实环境，learner 进程训练世界模型与策略；
  通过有界数据通道传递采样数据，并定期回传策略快照。训练入口统一使用此拓扑，
  `algo.collector` 配置采样进程与策略快照频率。
- **TaskMDP 契约**（`task_mdp.py`）：任务“解析 MDP heads”——observation / reward /
  soft_termination / critic_observation / goal。learner 从头到尾只消费该协议；
  它是 learner 与任务之间的稳定 ABI，按 `for_imagined()` 配置独立装配。collector
  直接使用真实环境返回的 observation，不构造想象程序。设计纪律：纯函数、无 env/sim 句柄、无隐藏状态、
  设备无关、可 `torch.compile`。
- **Manager 任务接入**（[mbrl_env.py](src/motrix_rl_supermimic/mbrl_env.py)）：
  `MbrlEnvCfg` 直接扩展 `ManagerBasedEnvCfg`，复用已有 observation/reward 配置，额外声明
  模型状态、预测输入绑定和 soft termination。世界模型 context 的固定仿射缩放由任务配置参数描述。
  框架自动组装数值程序，无需另写任务程序工厂。
  真实环境与 imagination 复用 `BuildContext`、term resolver 和 Torch 数值执行原语，
  但分别拥有自己的 terms；rollout 调度层显式绑定预测输入，同一批输入的数值 heads 共享 `ManagerContext`，
  `ManagerTaskMDP` 只执行已解析的计算，不自行构造上下文。
  collector 与 play 从环境返回的 `obs["goal"]` 获取目标，不访问任务 command 或构造 TaskMDP；
  goal 是普通 observation group，窗口长度由任务的 `MbrlEnvCfg.goal_frames` 唯一定义，训练与评估直接消费；
  reset 与 final observation 由 core 管理，checkpoint 恢复与 play 检查目标窗口一致性。
  任务以具名 `WorldStateTermCfg` 组装物理状态，维度与切片从实际 terms 派生；各 term 的状态表示负责
  特征、预测目标、状态更新和误差。普通向量可使用 `EuclideanState`，机器人姿态/关节动力学显式选择
  `CanonicalRobotState`，不再是框架对所有任务的默认布局。
  `TransitionTermsCfg` 仅提取动力学 context；框架直接读取 action history 与配置选中的
  `ReferenceCommand.frame`，通过 `snapshot_transition()` 导出 transition。
  checkpoint 保存 learner 模型、优化器与随机数状态；恢复后 replay、仿真与任务采样状态重新初始化，
  不保证轨迹级精确续训。
  环境拥有 [TransitionRecorder](src/motrix_rl_supermimic/recorder.py)，在物理推进前保存起点，
  在 simulator read 后、command 更新和 reset 前保存终点；导出快照时合入框架最终 reward/done。
  Transition terms 读取的 simulator queries 由任务显式声明。
  普通 Manager 任务仍需满足可想象状态与共享解析计算的前提。
  任务通过 `ImaginationCfg` 构造具名的 `ImaginationTerm`，实现输入绑定与有效起点判断；镜像增强、回报尺度和 checkpoint 元数据作为数据提供。

新任务继续配置普通 Manager terms；action/command 通过无环境的单一构造入口创建，
learner 独立构造单行 terms，构造期按准确类型直接读取所需信息，不另维护静态 spec 层。
Imagined rollout 的批量状态独立绑定；MBRL 仅补充预测输入和专用能力。示例见
[toy.py](src/motrix_rl_supermimic/toy.py)。第一版仅支持每步 reference frame 前进 1；
预测状态所需的 FK 等派生输入由显式 Context 绑定提供，不自动猜测任意 simulator query。
数值程序见 [manager_task.py](src/motrix_rl_supermimic/manager_task.py)。Imagination 只执行
选择的数值 term，不执行真实环境的 command 推进、reset、recorder 或课程更新。
目标查询也复用配置中的 `goal` observation terms，使用只读参考帧 Context，不依赖预测物理状态，支持整段目标提前合批编码。
Policy observation 的噪声由同一套观测 terms 生成，learner 不额外补噪声；回放历史先生成带噪帧，再重复填充首帧。

任务必须满足“可想象性”前提：状态为低维物理状态元组（无 pixel/点云）；reward 与
termination 是状态元组 + 可微 FK 的解析函数；命令源可由 frame 解析推进。

## 边界

- 依赖方向：`任务包 → motrix_rl_supermimic → motrix_env_core`。本包不 import 任何
  任务包或具体机器人常量即可构建与测试（测试使用玩具 TaskMDP）。
- 想象 rollout（BPTT）发生在 learner 进程内，可微、不落盘；不为想象数据引入跨进程通道。

运行时契约（ManagerEnv/ArrayEnv 生命周期口径）见 `motrix_env_core` 文档；
可微 FK 见 `motrix_env_core.math.kinematics`。
