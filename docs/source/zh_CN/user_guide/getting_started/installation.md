# 安装

MotrixLab 支持两种使用方式。请根据你是在仓库中运行现有任务，还是在自己的项目中使用框架，选择对应的安装方式。

| 使用场景 | 安装方式 | 获得的内容 |
| -------- | -------- | ---------- |
| 运行仓库内置任务、修改任务或参与框架开发 | [在 MotrixLab 仓库中安装](#installation-repository) | 仓库任务、配置和资产，以及安装脚本选择的框架与插件 |
| 在自己的项目中编写环境、组织训练配置 | [在自己的项目中安装](#installation-own-project) | `motrixlab` 集成包及其默认组件，由自己的项目管理代码、配置和资产 |

两种方式都通过 `mxlab` 使用训练、评估、可视化、导出与部署命令；Python API 从各组件包导入，不通过 `motrixlab` 重新导出。

## 系统要求

- **Python**：3.10 或 3.11。仓库默认使用 Python 3.10；Unitree 硬件部署插件及包含该插件的完整开发环境仅支持 Python 3.10。Unitree 机器人模型的仿真与训练不受此限制。
- **系统与架构**：Linux x86_64 或 Windows x86_64。具体 GPU 运行能力还取决于驱动和所选的 PyTorch 构建。
- **包管理工具**：仓库安装使用 [uv](https://docs.astral.sh/uv/getting-started/installation/)；自己的项目可以使用 pip 或 uv。
- **Git 与 Git LFS**：仅仓库安装需要，用于获取源码及机器人网格、运动数据等资产。

(installation-repository)=
## 在 MotrixLab 仓库中安装

这种方式适合直接运行文档中的内置任务、修改仓库内的环境与配置，或者参与框架开发。安装脚本负责选择 workspace package 和 GPU 依赖，不需要逐个安装组件。

### 获取源码和资产

先安装 Git LFS，再克隆仓库：

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
git lfs install
git lfs pull
```

安装脚本也会检查并拉取 LFS 资产。缺少 Git LFS 时，请先安装它；不要将仓库中的 LFS 指针文件当作模型或运动数据使用。

### 安装运行环境

Linux：

```bash
sh install.sh
# 使用 Python 3.11，不包含 Unitree 硬件部署插件：
sh install.sh --python 3.11
```

Windows PowerShell：

```powershell
.\install.ps1
# 使用 Python 3.11：
.\install.ps1 --python 3.11
```

若 PowerShell 执行策略阻止脚本运行，可以使用：

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

默认运行环境包含 `motrixlab`、仓库任务和所需插件，以及必需的 SKRL Torch provider 和内置 FastSAC。安装脚本自动探测 NVIDIA / AMD GPU，选择对应的 CUDA / ROCm PyTorch wheel；不能自动探测时回退到 CUDA 配置。ROCm 安装仅支持 Linux。

Linux 安装默认启用 Intel TBB，用于 Numba 并行 kernel。开发工具、文档工具、RSL-RL 和 SKRL JAX 不会在默认运行环境中全部启用，需要时使用下面的参数追加。

### 激活环境并检查安装

Linux：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

检查统一命令入口：

```bash
mxlab --version
mxlab --help
```

随后可按照[第一个 MotrixLab 程序](hello_motrixlab.md)运行内置任务。

```{important}
在这个仓库中，避免裸用 `uv sync` 或 `uv run`。CUDA 与 ROCm 由互斥 extras 选择，裸命令可能切换到默认的 PyPI 依赖分支并替换当前 GPU 环境。

请在激活后的环境中直接执行 `mxlab`，或使用 `uv run --no-sync mxlab ...`。需要调整依赖时，重新执行带有所需参数的安装脚本；脚本每次重新计算依赖集合，不会永久累加之前选择的插件。

这个限制针对 MotrixLab 源码 workspace 的 GPU 配置，不是对所有 uv 项目的通用限制。
```

### 开发环境与可选依赖

参与框架开发、运行完整测试或构建文档时，安装完整开发环境：

```bash
sh install.sh --all
```

`--all` 包含 Unitree 硬件插件，因此只支持 Python 3.10。它启用全部 workspace package、开发与测试依赖，以及支持的训练后端和文档工具；ROCm 配置会跳过 CUDA-only extra，例如 SKRL JAX。

也可以只追加当前需要的能力：

```bash
sh install.sh --rslrl
sh install.sh --skrl-jax
sh install.sh --docs
```

重新安装时，请同时保留仍然需要的其他参数。

| 参数 | 说明 |
| ---- | ---- |
| `--python 3.10` / `--python 3.11` | 选择 Python 版本；默认 3.10，`--all` 要求 3.10 |
| `--gpu cuda` / `--gpu rocm` | 显式选择 GPU 依赖配置，覆盖自动探测；ROCm 仅限 Linux |
| `--gpu hygon` | Linux 海光 DCU 环境；保留外部 DTK 加速栈，需自行提供兼容的 Torch，不与 `--skrl-jax` 组合 |
| `--rslrl` | 添加 RSL-RL provider |
| `--skrl-jax` | 添加 SKRL JAX runtime，仅 Linux |
| `--skrl-torch` | 兼容性参数；SKRL Torch 已是仓库必需依赖 |
| `--tbb` | 显式启用 Intel TBB；Linux 安装脚本默认启用 |
| `--docs` | 添加文档工具链 |
| `--all` | 完整开发环境，仅 Python 3.10 |
| `--help` | 查看脚本的完整参数说明 |

Windows 安装使用 `install.ps1`，可通过其 `--help` 查看支持的选项。

(installation-own-project)=
## 在自己的项目中安装

这种方式不需要克隆 MotrixLab 仓库。将 `motrixlab` 作为项目依赖安装，由自己的项目提供环境实现、任务配置和资产。

### 默认安装范围

`motrixlab` 提供统一的 `mxlab` CLI，并安装同一发布版本的以下组件：

| 组件 | 能力 |
| ---- | ---- |
| `motrix-env-core` | backend 无关的环境框架、配置和 registry |
| `motrix-env-motrixsim` | MotrixSim 环境后端，并依赖安装 `motrixsim` |
| `motrix-rl` | 训练、评估与导出的控制平面 |
| `motrix-rl-builtin` | 内置 FastSAC provider |
| `motrix-deploy` | 部署 artifact 与运行时框架 |

```{note}
`motrixlab` 不包含仓库内置任务、机器人资产、SKRL / RSL-RL provider 或具体部署 backend 插件。因此，安装后不能直接假定仓库中的 `cartpole` 等示例任务已经可用。

如果你希望直接运行这些任务，使用前面的仓库安装方式。自己的项目则需要提供任务配置，并按实际用途安装额外组件。
```

下面的命令用于安装已发布的 PyPI 版本；新版本需在发布完成后才能安装。若当前只有 beta 预发布版本，使用对应示例中的预发布选项。

### 使用 pip

在自己的项目目录创建独立环境：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install motrixlab
```

Windows PowerShell 可使用 `py -3.10 -m venv .venv` 创建环境，然后运行 `.venv\Scripts\Activate.ps1` 激活。也可以选择 Python 3.11。

允许安装 beta 版本时：

```bash
python -m pip install --pre motrixlab
```

在项目的依赖文件中记录选定的版本，避免只修改本地环境而没有记录项目依赖。

### 使用 uv

对于已有 uv 项目，在项目目录中执行下面的步骤。如果尚未创建项目，可以先初始化：

```bash
uv init --python 3.10 my-robot-project
cd my-robot-project
```

在添加依赖前，确认自己项目的 `pyproject.toml` 将 Python 支持范围限制在集成包支持的版本内。例如：

```toml
[project]
requires-python = ">=3.10,<3.12"
```

`uv init --python 3.10` 选择的是解释器，初始化得到的 Python 版本范围仍可能没有上界。uv 会为项目声明的整个 Python 范围解析依赖，因此不能只确认当前解释器是 3.10，而让项目仍声明支持 3.12 及以上版本。

然后添加依赖：

```bash
uv add motrixlab
```

若要允许 beta 版本：

```bash
uv add --prerelease allow motrixlab
```

`uv add` 会更新自己项目的 `pyproject.toml` 和 `uv.lock`。后续使用 `uv sync` 恢复环境，并通过 `uv run` 运行项目命令：

```bash
uv run mxlab --version
uv run mxlab --help
```

### 检查命令与配置 GPU

使用 pip 安装时，在激活的环境中运行；使用 uv 时，在命令前加上 `uv run`：

```bash
mxlab --help
mxlab train --help
mxlab play --help
mxlab view --help
mxlab export-onnx --help
mxlab export-deploy --help
mxlab deploy --help
```

这些命令用于检查 CLI 和配置帮助，不代表已经执行仿真、训练或硬件部署。

与仓库安装脚本不同，`pip install motrixlab` 和 `uv add motrixlab` **不会自动探测 GPU 厂商并选择专用 wheel 源**。默认安装按组件依赖从包索引解析 PyTorch；CUDA / ROCm wheel、驱动及相关环境配置需要由你自己的项目管理。

请根据 [PyTorch 官方安装指引](https://pytorch.org/get-started/locally/)选择与硬件匹配的构建。使用 uv 时，应在自己的项目中声明相应的版本约束和索引来源，不要假定依赖包仓库内的 `tool.uv.sources` 配置会传递到你的项目。仅安装成功或 `mxlab --help` 正常，不代表 GPU runtime 已可用。

安装完成后，先按[编写 DirectEnv 环境](../tutorial/building_envs/direct_env.md)创建并注册环境，再按[Task 配置与命令行参数覆盖](../tutorial/training/task_config.md)组织自己的训练配置。运行 Task 前，需安装它使用的 RL provider：默认集成包包含 FastSAC provider；使用 SKRL PPO 或 RSL-RL PPO 时，还需分别安装 `motrix-rl-skrl` 或 `motrix-rl-rslrl`。
