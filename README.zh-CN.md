**语言**: [English](README.md) | [简体中文](README.zh-CN.md)

<div align="center">

# MotrixLab

[![GitHub License](https://img.shields.io/github/license/Motphys/MotrixLab)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.10-blue)
[![Release](https://img.shields.io/github/v/release/Motphys/MotrixLab?include_prereleases)](https://github.com/Motphys/MotrixLab/releases)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue)](https://motrixlab.readthedocs.io)

**在仿真中训练机器人策略，并部署到真实硬件。**

<img src="docs/source/_static/images/microduck-walk.gif" alt="使用 MotrixLab 训练的 microduck 机器人在 MotrixRender 中行走" width="720">

_使用 MotrixLab 训练的 microduck 行走策略，由 MotrixRender 实时渲染 — [观看高清视频](https://github.com/user-attachments/assets/4bcf3122-f135-44cb-a966-d2d8e84479da)。_

**📖 用户文档**: [简体中文](https://motrixlab.readthedocs.io/zh-cn/stable/) | [English](https://motrixlab.readthedocs.io/en/stable/)

</div>

## 目录

- [MotrixLab 是什么？](#motrixlab-是什么)
- [主要特性](#主要特性)
- [快速开始](#-快速开始)
- [任务环境](#-任务环境)
- [内置机器人模型](#-内置机器人模型)
- [项目组成](#-项目组成)
- [参与贡献](#-参与贡献)
- [联系方式](#-联系方式)

## MotrixLab 是什么？

**MotrixLab** 是一个开源的机器人强化学习训练框架，构建于高性能的 [MotrixSim](https://github.com/Motphys/motrixsim-docs) 物理引擎之上。环境只需定义一次，即可通过 SKRL、RSLRL 或内置的 FastSAC 使用数千个并行环境实例进行训练，训练得到的策略可部署到 MuJoCo 或 Unitree 硬件 —— 全程只需一个命令行接口。

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/_static/images/architecture-dark.svg">
    <img src="docs/source/_static/images/architecture-light.svg" alt="MotrixLab 架构：环境只需定义一次，即可用 SKRL、RSL-RL 或 FastSAC 在数千个并行 MotrixSim 环境上训练，同一策略产物可部署到 MuJoCo 或 Unitree 硬件" width="720">
  </picture>
</div>

## 主要特性

- **统一接口**: 提供简洁统一的强化学习训练和评估接口
- **多框架支持**: 支持 SKRL（JAX/PyTorch）、RSLRL（PyTorch）和内置 FastSAC 实现
- **丰富环境**: 包含基础控制、运动、操作等多种机器人仿真环境
- **Sim-to-Real 部署**: 同一套策略代码，通过部署 CLI Sim2Sim 到 MuJoCo 仿真、Sim2Real 到真实硬件
- **高精度高性能仿真**: 基于 [MotrixSim](https://motrixsim.readthedocs.io/) 高精度、高性能物理仿真引擎
- **可视化训练**: 支持实时渲染和训练过程可视化

## 🚀 快速开始

### 前置条件

| 要求 | 说明 |
| --- | --- |
| Python **3.10.x** | workspace 锁定 `==3.10.*` |
| [uv](https://docs.astral.sh/uv/) | Python 项目与依赖管理工具 — [安装指南](https://docs.astral.sh/uv/getting-started/installation/) |
| [Git LFS](https://git-lfs.com) | 机器人网格、运动数据与视频由 LFS 管理 |
| 操作系统 | Linux x86_64 或 Windows x86_64；JAX 训练后端仅支持 Linux |

### 1. 克隆仓库

```bash
git clone https://github.com/Motphys/MotrixLab
cd MotrixLab
git lfs pull
```

### 2. 安装依赖

```bash
uv sync --all-packages
```

该命令会安装全部 workspace package，以及内置 FastSAC 所需的默认训练后端 **PyTorch**。SKRL、RSLRL 等第三方训练框架为可选 extras。

### 3. 训练第一个策略

```bash
uv run scripts/train.py task=microduck-walk-flat/motrix.fastsac play=true
```

训练过程中，内置面板会实时显示运行进度、回合统计、吞吐、奖励与系统健康状态：

<p align="center">
  <img src="docs/source/_static/images/train-console.png" alt="microduck-walk-flat motrix.fastsac 任务的 MotrixLab 训练面板" width="720">
</p>

训练会启动数千个并行环境实例；训练结束后会自动加载策略并在查看器中回放。checkpoint 与 TensorBoard 日志保存在 `runs/microduck-walk-flat/` 目录下，通过以下命令查看训练曲线：

```bash
uv run tensorboard --logdir runs/microduck-walk-flat
```

microduck 的训练数分钟内即可完成：平均回报与回合长度通常在约 4,000 次迭代后收敛：

<p align="center">
  <img src="docs/source/_static/images/microduck-training-curves.png" alt="microduck-walk-flat 训练的 TensorBoard 曲线：平均回报与回合长度在约 4,000 次迭代后收敛" width="720">
</p>

### 4. 回放训练好的策略

无需重新训练即可回放最近一次训练得到的策略（例如提前 Ctrl+C 中断训练之后）：

```bash
uv run scripts/play.py env=microduck-walk-flat
```

训练好的 microduck 策略在查看器中的回放效果：

https://github.com/user-attachments/assets/4bcf3122-f135-44cb-a966-d2d8e84479da

## 🌍 任务环境

MotrixLab 内置 50+ 个仿真环境，覆盖基础控制、四足、人形、全身动作跟踪、操作等类别。主要类别如下：

| 预览图 | 类别 | 示例环境 |
| --- | --- | --- |
| <img src="docs/source/_static/images/poster/go2-walk-rough.jpg" alt="go2-walk-rough" width="240"> | 四足速度跟踪 | `go2-walk-flat` · `go2-walk-rough` · `go1-walk-rough` · `anymalc-walk-flat` |
| <img src="docs/source/_static/images/poster/g1-walk-flat.jpg" alt="g1-walk-flat" width="240"> | 人形速度跟踪 | `g1-walk-flat` · `k1-walk-rough` · `dex-evt-walk-flat` · `microduck-walk-flat` |
| <img src="docs/source/_static/images/poster/g1-wbt-dance.jpg" alt="g1-wbt-dance" width="240"> | 全身动作跟踪（WBT） | `g1-wbt-dance` · `k1-wbt-freekick` · `g1-29dof-wbt-largebox` |

```bash
uv run scripts/view.py env=go2-walk-rough
```

完整环境列表与各环境支持的训练算法见[环境总览](https://motrixlab.readthedocs.io/zh-cn/latest/user_guide/envs/index.html)。

## 🤖 内置机器人模型

通过 robot registry 内置 7 个可复用机器人模型，可与任意场景和任务组合：

| 截图 | Registry 名称 | 类型 | 自由度 |
| --- | --- | --- | --- |
| <img src="docs/source/_static/images/robots/anymal_c.png" alt="anymal_c" width="180"> | `anymal_c` | 四足机器人 | 12 |
| <img src="docs/source/_static/images/robots/dex-evt.png" alt="dex-evt" width="180"> | `dex-evt` | 人形机器人 | 23 |
| <img src="docs/source/_static/images/robots/g1-29dof.png" alt="g1-29dof" width="180"> | `g1-29dof` | 人形机器人 | 29 |
| <img src="docs/source/_static/images/robots/go1.png" alt="go1" width="180"> | `go1` | 四足机器人 | 12 |
| <img src="docs/source/_static/images/robots/go2.png" alt="go2" width="180"> | `go2` | 四足机器人 | 12 |
| <img src="docs/source/_static/images/robots/k1.png" alt="k1" width="180"> | `k1` | 人形机器人 | 22 |
| <img src="docs/source/_static/images/robots/microduck.png" alt="microduck" width="180"> | `microduck` | 人形机器人 | 14 |

```bash
uv run scripts/view.py robot=go2
```

机器人配置细节与自定义新模型的方法见[支持的机器人](https://motrixlab.readthedocs.io/zh-cn/latest/user_guide/robots.html)。

## 🏗️ 项目组成

MotrixLab 是一个由九个 package 组成的 [uv](https://docs.astral.sh/uv/) workspace：

| Package | PyPI 名称 | 说明 |
| --- | --- | --- |
| **motrix_deploy** | `motrix-deploy` | 独立于训练框架的 artifact、backend、policy、控制循环、registry 与 CLI |
| **motrix_deploy_mujoco** | `motrix-deploy-mujoco` | MuJoCo 部署 backend plugin |
| **motrix_deploy_unitree** | `motrix-deploy-unitree` | Unitree SDK2 DDS 硬件 backend plugin |
| **motrix_deploy_tasks** | `motrix-deploy-tasks` | 具体的带版本部署任务实现与可执行入口 bootstrap |
| **motrix_env_core** | `motrix-env-core` | 环境基类、配置、registry、场景构建、NumPy runtime 与渲染能力，不包含任何内置任务和机器人资产 |
| **motrix_env_motrixsim** | `motrix-env-motrixsim` | MotrixSim 实时仿真 backend、renderer 与 torch frontend |
| **motrix_env_mujoco** | `motrix-env-mujoco` | 仅负责编译场景的 MuJoCo backend |
| **motrix_envs** | `motrix-envs` | 内置环境、模型、数据及环境到部署 profile 的编译实现 |
| **motrix_rl** | `motrix-rl` | 基于 `motrix-env-core` 的 RL 框架集成，支持 SKRL、RSLRL 和 FastSAC |

## 🤝 参与贡献

欢迎参与贡献！开发环境搭建、分支与提交规范、以及仓库配置的检查工具（`prek`、`ruff`、`dprint`、`mypy`）请参见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📬 联系方式

有问题或建议？欢迎通过以下方式联系我们：

- GitHub Issues: [提交问题](https://github.com/Motphys/MotrixLab/issues)
- Discussions: [加入讨论](https://github.com/Motphys/MotrixLab/discussions)

## 项目规范

- [贡献指南](CONTRIBUTING.md)
- [安全策略](SECURITY.md)
- [行为准则](CODE_OF_CONDUCT.md)
- [支持与提问](SUPPORT.md)
- [第三方声明](THIRD_PARTY_NOTICES.md)
- [更新日志](CHANGELOG.md)
- [引用信息](CITATION.cff)
- [许可证](LICENSE)
