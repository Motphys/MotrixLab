# 安装环境

本文档将引导您完成 MotrixLab 的安装与配置。请仔细阅读系统要求，并根据您的使用场景选择合适的安装方式。

## 系统要求

-   **Python 版本**：{bdg-danger-line}`3.10.*`

    本项目依赖特定 Python 版本，其他版本暂不受支持：

    | Python 版本 | 支持状态 |
    | :---------: | :------: |
    |    ≤ 3.9    |    ❌    |
    |    3.10     |    ✅    |
    |   ≥ 3.11    |    ❌    |

-   **包管理器**：{bdg-danger-line}`UV`

    本项目采用 UV 作为唯一的包管理工具，以提供快速、可复现的依赖管理环境。UV 的安装方法请参考[官方文档](https://docs.astral.sh/uv/getting-started/installation/)。

-   **系统及架构**：

    -   {bdg-danger-line}`Windows(x86_64)`
    -   {bdg-danger-line}`Linux(x86_64)`

    ```{note}
    不同操作系统对 MotrixLab 各功能模块的支持情况如下：

    | 操作系统 | CPU 仿真 | 交互式查看器 |
    | :------: | :------: | :----------: |
    |  Linux   |    ✅    |      ✅      |
    | Windows  |    ✅    |      ✅      |
    ```

## 安装步骤

### 克隆项目仓库

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
```

### 安装运行环境

在仓库根目录执行以下命令安装运行环境（Windows 上请在 PowerShell 中执行 `install.ps1`；若执行策略受限，改用 `powershell -ExecutionPolicy Bypass -File install.ps1`）：

```bash
sh install.sh
```

该命令会自动探测 GPU 厂商（NVIDIA → CUDA，AMD → ROCm），安装全部 workspace package、对应的
PyTorch wheel、SKRL 训练框架以及内置的 FastSAC 算法。

:::{dropdown} 配置国内镜像源（可选）
:animate: fade-in
:color: warning
:icon: desktop-download
如果您身处中国大陆，建议配置国内镜像源以加速依赖下载。修改项目根目录的 `uv.toml` 文件：

```toml
[[index]]
name = "mirror"
# 请填写您选择的国内镜像源，例如：
# 清华源: "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
url = ""


[[index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"
default = true
```

然后重新执行 `sh install.sh` 即可生效。

:::

### 激活环境

安装完成后，先激活环境再运行命令（Windows PowerShell：`.venv\Scripts\Activate.ps1`）：

```bash
source .venv/bin/activate
```

```{note}
避免裸 `uv sync` / `uv run`：CUDA 与 ROCm wheel 由互斥的 extras 选择，裸命令会解析到默认的 PyPI
分支并重装环境。请从激活后的环境运行命令，或对单条命令使用 `uv run --no-sync`。
```

## 开发环境安装

上述运行环境只包含训练与部署所需的依赖。如果你要参与开发（运行测试、修改代码、构建文档），
执行以下命令安装完整开发环境：

```bash
sh install.sh --all
```

`--all` 会把所有依赖一次性全部启用：开发工具链、测试依赖、全部训练后端
（`--skrl-jax` / `--rslrl`）与文档工具。也可以在运行环境基础上按需追加，
例如 `sh install.sh --docs` 只补文档工具链。

## 安装参数参考

所有模式都会安装全部 workspace package；运行环境在此基础上启用「1 个 GPU extra ＋ 训练后端 extra」，
各参数只负责选择或追加：

| 参数 | 可选值 | 说明 |
| ---- | ------ | ---- |
| `--all` | — | 完整开发环境：把下述所有依赖一次性全部启用（开发工具链、测试依赖、全部训练后端与文档工具） |
| （无参数） | — | 运行环境；自动探测 GPU 厂商（NVIDIA → CUDA，AMD → ROCm，无法探测时回退 CUDA） |
| `--gpu` | `cuda`<br>`rocm` | 指定 torch wheel 来源，覆盖自动探测 |
| `--skrl-jax` | — | SKRL（JAX）训练后端，仅 Linux |
| `--rslrl` | — | RSL-RL（PyTorch）训练后端 |
| `--docs` | — | 追加本地构建文档所需的工具链（Sphinx） |
| `-h`、`--help` | — | 显示帮助 |

完整参数说明见 `sh install.sh --help`。
