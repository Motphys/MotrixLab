# Installation Environment

This document will guide you through the installation and configuration of MotrixLab. Please read the system requirements carefully and choose the appropriate installation method based on your use case.

## System Requirements

-   **Python Version**: {bdg-danger-line}`3.10.*`

    This project requires a specific Python version, other versions are not currently supported:

    | Python Version | Support Status |
    | :------------: | :------------: |
    |     ≤ 3.9      |       ❌       |
    |      3.10      |       ✅       |
    |     ≥ 3.11     |       ❌       |

-   **Package Manager**: {bdg-danger-line}`UV`

    This project uses UV as the exclusive package management tool to provide fast, reproducible dependency management environment. For UV installation, please refer to the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

-   **System and Architecture**:

    -   {bdg-danger-line}`Windows(x86_64)`
    -   {bdg-danger-line}`Linux(x86_64)`

    ```{note}
    Features supported on each platform:

    | Operating System | CPU Simulation | Interactive Viewer |
    | :--------------: | :------------: | :----------------: |
    |      Linux       |       ✅       |         ✅          |
    |     Windows      |       ✅       |         ✅          |
    ```

## Installation Steps

### Clone Project Repository

```bash
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
```

### Install the Runtime Environment

Install the runtime environment from the repository root (on Windows, run `install.ps1` in PowerShell; if the execution policy blocks it, use `powershell -ExecutionPolicy Bypass -File install.ps1`):

```bash
sh install.sh
```

This auto-detects your GPU vendor (NVIDIA → CUDA, AMD → ROCm) and installs all workspace packages,
the matching PyTorch wheels, the SKRL training framework, and the built-in FastSAC algorithm.

### Activate the Environment

After installing, activate the environment before running commands (Windows PowerShell:
`.venv\Scripts\Activate.ps1`):

```bash
source .venv/bin/activate
```

```{note}
Avoid bare `uv sync` / `uv run`: the CUDA and ROCm wheels are selected by mutually exclusive
extras, so a bare command resolves the default PyPI fork and reinstalls the environment. Run
commands from the activated environment, or pass `--no-sync` to one-off `uv run` calls.
```

## Development Environment

The runtime environment only contains what training and deployment need. To contribute to the
project (run tests, modify code, build the docs), install the full development environment:

```bash
sh install.sh --all
```

`--all` enables everything in one go: the dev toolchain, test dependencies, all training backends
(`--skrl-jax` / `--rslrl`), and the docs tooling. You can also extend the runtime environment
incrementally, e.g. `sh install.sh --docs` adds only the docs toolchain.

## Option Reference

Every mode installs all workspace packages; the runtime environment enables one GPU extra plus the
training backend extra on top of them, and each option only selects or appends to that combination:

| Option | Values | Description |
| ------ | ------ | ----------- |
| `--all` | — | Full development environment: everything below is enabled in one go (dev toolchain, test dependencies, all training backends, and docs tooling) |
| *(none)* | — | Runtime environment; the GPU vendor is auto-detected (NVIDIA → CUDA, AMD → ROCm; CUDA as fallback when detection is impossible) |
| `--gpu` | `cuda`<br>`rocm` | Select the torch wheel flavor explicitly, overriding auto-detection |
| `--skrl-jax` | — | SKRL on JAX backend, Linux only |
| `--rslrl` | — | RSL-RL on PyTorch backend |
| `--docs` | — | Add the toolchain (sphinx) needed to build the documentation locally |
| `-h`, `--help` | — | Show the help message |

Run `sh install.sh --help` for the full option reference.
