# 快速入门：Hello MotrixLab

本教程通过演示一个简单例子 - 加载倒立摆并进行训练，以此来展示 MotrixLab 工作流程：

本教程使用仓库内置的倒立摆任务。请先完成{ref}`在 MotrixLab 仓库中安装 <installation-repository>`，然后激活环境：`source .venv/bin/activate`（Windows PowerShell：`.venv\Scripts\Activate.ps1`）。以下命令均在仓库根目录执行。

安装后通过统一的 `mxlab` 命令训练和评估策略；使用 `mxlab --help` 查看命令，`mxlab --version` 查看版本。

## 环境预览

使用 `mxlab view` 可视化环境，而不执行任何训练，这可以帮助您检测系统的环境依赖是否配置正确：

```bash
mxlab view env=cartpole
```

这将打开一个可视化窗口，显示倒立摆的物理仿真环境，使用随机动作进行演示。

## 训练模型

开始训练倒立摆平衡任务：

```bash
mxlab train task=cartpole/skrl.ppo
```

训练过程会自动：

1. 根据硬件环境自动选择训练后端（JAX 或 PyTorch）
2. 创建训练环境
3. 开始 PPO 算法训练

训练结果会保存在 `runs/cartpole/` 目录下，包含：

-   训练检查点（checkpoint）
-   TensorBoard 日志文件

## 可视化训练过程

如果您想要在训练过程中观察模型的学习过程，可以启用可视化渲染：

```bash
mxlab train task=cartpole/skrl.ppo render=true
```

### 🎮 交互式渲染控制

> **重要提示**：可视化会显著降低训练速度，建议主要用于调试和演示。

在可视化训练过程中，您可以使用**空格键**来动态控制渲染：

-   **开启渲染**：按下空格键开启可视化，观察机器人行为
-   **关闭渲染**：再次按下空格键关闭渲染，提升训练速度
-   **随时切换**：无需重新启动程序，可以在训练过程中随时切换

这种交互式控制让您可以在需要时观察训练效果，在不需要时享受快速训练。这项功能在运行推断时也能生效。

## 查看训练结果

使用 TensorBoard 查看训练进度：

```bash
tensorboard --logdir runs/cartpole
```

## 测试训练好的模型

训练完成后，使用训练好的策略进行测试：

```bash
# 自动寻找最佳策略测试（推荐）
mxlab play env=cartpole

# 手动指定带 metadata 的 run 中的 checkpoint
mxlab play env=cartpole policy=/path/to/run/checkpoints/policy-file
```

> **提示**：系统会自动在 `runs/cartpole/` 下寻找最新 run 的最佳策略。手动指定的 checkpoint 必须属于包含 `metadata.json` 和 `task_config.yaml` 的 run。

## 至此我们完成了整个示例

接下来可以尝试修改参数，观察不同设置下的物理效果，或者尝试其他环境。

## 下一步

-   了解 [基础框架](../tutorial/basic_frame.md)
-   学习 [物理环境配置](../tutorial/building_envs/scene.md)
-   浏览更多 [环境](../envs/index.md)
