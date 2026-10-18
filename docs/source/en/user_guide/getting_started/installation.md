# Installation

MotrixLab supports two ways of working. Choose the installation path that matches whether you want to run existing tasks in the repository or use the framework in your own project.

| Use case | Installation path | What you get |
| -------- | ----------------- | ------------ |
| Run built-in tasks, modify tasks, or contribute to the framework | [Install in the MotrixLab repository](#installation-repository) | Repository tasks, configurations, and assets, plus the framework components and plugins selected by the installer |
| Implement environments and organize training configurations in your own project | [Install in your own project](#installation-own-project) | The `motrixlab` integration package and its default components; your project manages its own code, configurations, and assets |

Both paths provide `mxlab` for training, playback, visualization, export, and deployment. Import Python APIs from their component packages; they are not re-exported through `motrixlab`.

## System requirements

- **Python**: 3.10–3.13. The repository defaults to Python 3.10. The Unitree hardware deployment plugin and the full development environment that includes it require Python 3.10. Simulation and training with Unitree robot models are not subject to this restriction.
- **Operating system and architecture**: Linux x86_64 or Windows x86_64. GPU runtime support also depends on your drivers and the selected PyTorch build.
- **Package manager**: Repository installation uses [uv](https://docs.astral.sh/uv/getting-started/installation/). Your own project can use pip or uv.
- **Git and Git LFS**: Required only for repository installation, to obtain source code and assets such as robot meshes and motion data.

(installation-repository)=
## Install in the MotrixLab repository

Choose this path to run the built-in tasks described in the documentation, modify repository environments and configurations, or contribute to the framework. The installer selects workspace packages and GPU dependencies for you; there is no need to install each component separately.

### Get the source code and assets

Install Git LFS first, then clone the repository:

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
git lfs install
git lfs pull
```

The installer also checks for Git LFS and pulls the assets. If Git LFS is missing, install it first; repository LFS pointer files are not usable model or motion-data files.

### Install the runtime environment

Linux:

```bash
sh install.sh
# Use Python 3.11–3.13 without the Unitree hardware deployment plugin (3.13 shown):
sh install.sh --python 3.13
```

Windows PowerShell:

```powershell
.\install.ps1
# Use Python 3.11:
.\install.ps1 --python 3.11
```

If the PowerShell execution policy blocks the script, use:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

The default runtime environment includes `motrixlab`, repository tasks and the required plugins, the required SKRL Torch provider, and built-in FastSAC. The installer detects NVIDIA / AMD GPUs and selects the corresponding CUDA / ROCm PyTorch wheels. It falls back to the CUDA profile when detection is unavailable. ROCm installation is supported only on Linux.

Linux installation enables Intel TBB by default for Numba parallel kernels. Development tools, documentation tools, RSL-RL, and SKRL JAX are not all enabled in the default runtime environment; add them using the options below when needed.

### Activate the environment and check the installation

Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Check the unified command entry point:

```bash
mxlab --version
mxlab --help
```

Then follow [Your first MotrixLab program](hello_motrixlab.md) to run a built-in task.

```{important}
In this repository, avoid bare `uv sync` or `uv run`. CUDA and ROCm are selected through mutually exclusive extras; a bare command may switch to the default PyPI dependency branch and replace your current GPU environment.

Run `mxlab` directly in the activated environment, or use `uv run --no-sync mxlab ...`. To change dependencies, rerun the installer with the options you need. The installer recomputes the dependency set each time rather than permanently accumulating previously selected plugins.

This restriction applies to GPU configuration in the MotrixLab source workspace, not to every uv project.
```

### Development environment and optional dependencies

To contribute to the framework, run the full test suite, or build the documentation, install the full development environment:

```bash
sh install.sh --all
```

`--all` includes the Unitree hardware plugin and therefore requires Python 3.10. It enables all workspace packages, development and test dependencies, and supported training backends and documentation tools. The ROCm profile skips CUDA-only extras such as SKRL JAX.

You can also add only the capabilities you currently need:

```bash
sh install.sh --rslrl
sh install.sh --skrl-jax
sh install.sh --docs
```

When reinstalling, include any other options you still need in the same invocation.

| Option | Description |
| ------ | ----------- |
| `--python VERSION` | Select Python: `3.10`, `3.11`, `3.12`, or `3.13`; defaults to 3.10, and `--all` requires 3.10 |
| `--gpu cuda` / `--gpu rocm` | Select a GPU dependency profile explicitly, overriding detection; ROCm is Linux-only |
| `--gpu hygon` | Linux Hygon DCU environment; preserve the external DTK accelerator stack and provide compatible Torch yourself; do not combine with `--skrl-jax` |
| `--rslrl` | Add the RSL-RL provider |
| `--skrl-jax` | Add the SKRL JAX runtime, Linux-only |
| `--skrl-torch` | Compatibility option; SKRL Torch is already a required repository dependency |
| `--tbb` | Enable Intel TBB explicitly; already enabled by the Linux installer |
| `--docs` | Add documentation tools |
| `--all` | Full development environment, Python 3.10 only |
| `--help` | Show the full installer option reference |

On Windows, use `install.ps1` and its `--help` option to check the supported options.

(installation-own-project)=
## Install in your own project

This path does not require cloning the MotrixLab repository. Install `motrixlab` as a project dependency and provide your environment implementations, task configurations, and assets in your own project.

### Default installation scope

`motrixlab` provides the unified `mxlab` CLI and installs the following components from the same release:

| Component | Capability |
| --------- | ---------- |
| `motrix-env-core` | Backend-neutral environment framework, configuration, and registry |
| `motrix-env-motrixsim` | MotrixSim environment backend; installs `motrixsim` as a dependency |
| `motrix-rl` | Control plane for training, playback, and export |
| `motrix-rl-builtin` | Built-in FastSAC provider |
| `motrix-deploy` | Deployment artifacts and runtime framework |

```{note}
`motrixlab` does not include repository tasks, robot assets, SKRL / RSL-RL providers, or concrete deployment backend plugins. Installing it does not make repository examples such as `cartpole` available automatically.

Use the repository installation path above if you want to run those tasks directly. In your own project, supply task configurations and install additional components as your use case requires.
```

The commands below install versions published on PyPI. A new version becomes installable only after publication. If only beta releases are available, use the prerelease options shown below.

### Use pip

Create an isolated environment in your project directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install motrixlab
```

On Windows PowerShell, create the environment with `py -3.10 -m venv .venv`, then activate it with `.venv\Scripts\Activate.ps1`. You can also choose Python 3.11–3.13.

To allow beta releases:

```bash
python -m pip install --pre motrixlab
```

Record the selected version in your project's dependency file rather than changing only the local environment without recording the dependency.

### Use uv

For an existing uv project, follow the steps below from the project directory. If you have not created a project yet, initialize one first:

```bash
uv init --python 3.10 my-robot-project
cd my-robot-project
```

Before adding the dependency, make sure your project's `pyproject.toml` restricts Python support to versions supported by the integration package. For example:

```toml
[project]
requires-python = ">=3.10,<3.14"
```

`uv init --python 3.10` selects an interpreter, but the initialized Python requirement may still have no upper bound. uv resolves dependencies for the entire Python range declared by your project. Selecting a Python 3.10 interpreter is not sufficient if the project still declares support for Python 3.14 and later.

Then add the dependency:

```bash
uv add motrixlab
```

To allow beta releases:

```bash
uv add --prerelease allow motrixlab
```

`uv add` updates your project's `pyproject.toml` and `uv.lock`. Use `uv sync` to restore the environment later, and `uv run` to run project commands:

```bash
uv run mxlab --version
uv run mxlab --help
```

### Check commands and configure the GPU

With pip, run commands in the activated environment. With uv, prefix them with `uv run`:

```bash
mxlab --help
mxlab train --help
mxlab play --help
mxlab view --help
mxlab export-onnx --help
mxlab export-deploy --help
mxlab deploy --help
```

These commands check the CLI and configuration help; they do not execute simulation, training, or hardware deployment.

Unlike the repository installer, `pip install motrixlab` and `uv add motrixlab` **do not detect your GPU vendor or select a dedicated wheel index automatically**. Default installation resolves PyTorch from the package index according to component dependencies. Your own project must manage CUDA / ROCm wheels, drivers, and related environment configuration.

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) to select a build for your hardware. With uv, declare the appropriate version constraints and index sources in your own project. Do not assume that `tool.uv.sources` settings in a dependency's repository are inherited by your project. Successful installation or a working `mxlab --help` does not establish that the GPU runtime is available.

After installation, follow [Writing DirectEnv Environments](../tutorial/building_envs/direct_env.md) to create and register an environment, then organize your training configuration with [Task Configuration and CLI Overrides](../tutorial/training/task_config.md). Before running a Task, install its RL provider: the default integration package includes the FastSAC provider; SKRL PPO and RSL-RL PPO additionally require `motrix-rl-skrl` and `motrix-rl-rslrl`, respectively.
