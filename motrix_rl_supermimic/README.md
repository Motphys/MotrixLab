# Motrix RL SuperMimic

中文 | [English](README.en.md)

`motrix-rl-supermimic` 为 MotrixLab 提供 SuperMimic 风格的模型强化学习训练：从真实仿真采样中学习物理状态的动力学模型，再通过可微的想象轨迹优化策略。它面向参考动作跟踪等任务，目前仍在开发中。

本包提供算法与训练能力，机器人、参考动作、奖励和状态表示由任务包定义。它不是适用于所有 model-based RL 算法的通用世界模型接口，也不包含 Dreamer 式的视觉潜状态学习。

## 安装与使用

本包通过 [`motrix-rl-builtin`](../motrix_rl_builtin/README.md) 集成到统一 CLI，注册为 `motrix.supermimic`。安装 MotrixLab 的 builtin 组件会同时安装本包；单独安装本包不会自动注册 CLI 算法。

安装提供 `g1-supermimic-tracking` 环境及其训练配方的任务包后，即可训练：

```bash
mxlab train task=g1-supermimic-tracking/motrix.supermimic
```

SuperMimic 训练依赖 CUDA。任务包负责环境注册与训练配置，CLI 配置、运行目录与 checkpoint 的通用用法见 [Motrix RL 使用说明](../motrix_rl/README.md)。

训练采用异步采样：collector 运行真实环境，learner 更新世界模型、策略和 critic，并向 collector 发布策略快照。模型与训练设置使用 [`SuperMimicCfg`](src/motrix_rl_supermimic/config.py)，采样进程设置位于 `algo.collector`。

## 接入自己的任务

任务以 [`SuperMimicEnvCfg`](src/motrix_rl_supermimic/env.py) 配置真实环境，复用 Manager 的观测、奖励、终止和 reset terms，并声明世界模型需要预测的物理状态及其动力学条件。真实环境由 `SuperMimicEnv` 负责记录训练转移；任务无需另写 collector。

要在想象轨迹中运行，任务还需要能够从预测状态重新计算观测、奖励和软终止信号。这些计算必须是可微的 Tensor 运算，不能依赖真实 simulator 的句柄或未声明的传感器结果。机器人运动学、坐标系和状态编码由任务自行提供，框架不预设机器人布局。

真实环境与想象任务复用数值项，但不共享可变运行状态。`for_imagined()` 用于选择训练中的想象配置，`for_play()` 用于表达评估差异；需要参考动作上的残差策略时，可通过 `reference_action()` 提供基础动作模块。

一个不依赖物理引擎的最小任务见[测试示例](tests/_supermimic_toy.py)。它用于说明任务组装与验证接口，不随运行时包分发。

## 支持范围与限制

- **显式物理状态**：世界模型预测低维物理状态；不支持从像素或点云学习潜状态动力学。
- **参考动作条件**：目标作为观测组提供，当前参考帧按每步 `+1` 推进；任务必须能判断有效的想象起点。
- **可微任务计算**：想象轨迹在 learner 内计算，不执行真实环境的 command 推进、reset 或课程更新。
- **恢复训练**：checkpoint 保存模型、优化器和随机数状态，但 replay、仿真与任务采样状态会重新初始化，不保证轨迹级精确续训。

环境 step、自动重置和终态观测遵循 [core 环境生命周期](../wiki/design/architecture.md)；随机性与恢复边界见[随机性设计](../wiki/design/randomness-and-seeding.md)。
