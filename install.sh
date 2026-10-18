#!/bin/sh
# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

# Project bootstrap: runtime environment (`sh install.sh`) or full
# development environment (`sh install.sh --all`).
#
# GPU wheel flavor (--gpu): cuda, rocm or hygon. Hygon installs non-accelerator
# dependencies only, preserving the DTK stack through system-site-packages.
# Auto-detection uses bound hycu devices, then nvidia/amdgpu kernel modules;
# CPU-only hosts fall back to cuda.
#
# Built-in FastSAC and SKRL on Torch are always installed. --skrl-jax adds
# the JAX runtime, and --rslrl adds the RSLRL plugin; multiple flags combine.
# Linux installs Intel TBB by default so numba parallel kernels use its threading
# layer (avoids the OpenMP per-region thread-wakeup storm on many-core hosts).
# --tbb remains available to explicitly request TBB on other platforms.

set -e

usage() {
    echo "Usage: sh install.sh [options]"
    echo
    echo "Options:"
    echo "  --all                  Install the full development environment (all packages, groups and extras;"
    echo "                         ROCm hosts skip CUDA-only extras such as skrl-jax)"
    echo "  --docs                 Include the docs toolchain (sphinx) for building the documentation"
    echo "  --gpu cuda|rocm|hygon  GPU profile (default: auto-detected from the kernel driver)"
    echo "                         Hygon skips accelerator packages; install DTK-compatible Torch yourself."
    echo "  --python 3.10|3.11|3.12|3.13  Python version (default: uv's interpreter selection; --all requires 3.10)"
    echo "  --skrl-torch           Training backend: SKRL on PyTorch (default)"
    echo "  --skrl-jax             Training backend: SKRL on JAX (Linux only)"
    echo "  --rslrl                Training backend: RSL-RL on PyTorch"
    echo "  --tbb                  Install Intel TBB for numba parallel kernels (default on Linux)"
    echo "  -h, --help             Show this help"
}

ALL=""
GPU=""
PYTHON=""
SKRL_JAX=""
RSLRL=""
DOCS=""
TBB=""

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            ALL=1
            ;;
        --docs)
            DOCS=1
            ;;
        --gpu)
            [ $# -ge 2 ] || { echo "error: $1 requires a value (cuda, rocm or hygon)" >&2; exit 1; }
            GPU="$2"
            shift
            ;;
        --python)
            [ $# -ge 2 ] || { echo "error: $1 requires a value (3.10, 3.11, 3.12 or 3.13)" >&2; exit 1; }
            case "$2" in
                3.10|3.11|3.12|3.13) PYTHON="$2" ;;
                *) echo "error: invalid --python value: $2 (expected 3.10, 3.11, 3.12 or 3.13)" >&2; exit 1 ;;
            esac
            shift
            ;;
        --skrl-torch)
            # SKRL Torch is already included in the required root dependencies.
            ;;
        --skrl-jax)
            SKRL_JAX=1
            ;;
        --rslrl)
            RSLRL=1
            ;;
        --tbb)
            TBB=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [ -n "$GPU" ] && [ "$GPU" != cuda ] && [ "$GPU" != rocm ] && [ "$GPU" != hygon ]; then
    echo "error: invalid --gpu value: $GPU (expected cuda, rocm or hygon)" >&2
    usage >&2
    exit 1
fi

if [ -n "$ALL" ] && [ -n "$PYTHON" ] && [ "$PYTHON" != 3.10 ]; then
    echo "error: --all includes Unitree, which requires Python 3.10; use --python 3.10 or omit --all" >&2
    exit 1
fi

if [ -z "$GPU" ]; then
    # Inspect bound PCI devices, not the CPU vendor or management-tool presence.
    # sysfs is also visible in DTK containers where lsmod can miss the host driver.
    for device in /sys/bus/pci/drivers/hycu/*:*; do
        [ -e "$device" ] || continue
        GPU=hygon
        break
    done
    if [ "$GPU" = hygon ]; then
        echo "==> Hygon DCU detected; preserving the externally managed DTK accelerator stack."
    elif lsmod 2>/dev/null | grep -q '^nvidia '; then
        GPU=cuda
    elif lsmod 2>/dev/null | grep -q '^amdgpu '; then
        GPU=rocm
    else
        GPU=cuda
    fi
fi

if [ "$GPU" = hygon ] && [ -n "$SKRL_JAX" ]; then
    echo "error: --skrl-jax installs a CUDA JAX runtime and is not supported with --gpu hygon" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "==> uv not found, installing..."
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*)
            # Native Windows (Git Bash): use the official PowerShell installer.
            powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
            ;;
        *)
            if command -v curl >/dev/null 2>&1; then
                curl -LsSf https://astral.sh/uv/install.sh | sh
            elif command -v wget >/dev/null 2>&1; then
                wget -qO- https://astral.sh/uv/install.sh | sh
            else
                echo "error: neither curl nor wget found; install uv manually:" >&2
                echo "       https://docs.astral.sh/uv/getting-started/installation/" >&2
                exit 1
            fi
            ;;
    esac
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "error: uv installation failed; install it manually:" >&2
        echo "       https://docs.astral.sh/uv/getting-started/installation/" >&2
        exit 1
    fi
fi

# Keep uv's interpreter selection unchanged unless explicitly requested. Positional
# arguments preserve the option/value boundary without word splitting or eval.
set --
if [ -n "$PYTHON" ]; then
    set -- --python "$PYTHON"
elif [ -n "$ALL" ]; then
    # Inspect the interpreter without creating an environment or downloading
    # Python. If discovery fails, require an explicit compatible version rather
    # than letting sync download an unchecked interpreter for Unitree.
    if [ -n "${UV_PYTHON:-}" ]; then
        DETECTED_PYTHON=$(uv python find --show-version "$UV_PYTHON" 2>/dev/null || true)
    else
        DETECTED_PYTHON=$(uv python find --show-version 2>/dev/null || true)
    fi
    case "$DETECTED_PYTHON" in
        3.10|3.10.*) ;;
        "")
            echo "error: cannot verify Python for --all (Unitree requires Python 3.10); use --python 3.10" >&2
            exit 1
            ;;
        *)
            echo "error: --all includes Unitree, which requires Python 3.10 (detected $DETECTED_PYTHON); use --python 3.10" >&2
            exit 1
            ;;
    esac
fi

# Robot assets (STL/OBJ meshes, motion data) are stored via Git LFS; pull them
# so training does not hit pointer files. `git lfs pull` is incremental and
# skips objects already present on disk.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if git lfs version >/dev/null 2>&1; then
        git lfs install >/dev/null 2>&1 || true
        echo "==> Pulling Git LFS assets (robot meshes, motion data)..."
        set -x
        git lfs pull
        set +x
    else
        echo "error: git-lfs is not installed, but robot assets are stored via Git LFS" >&2
        echo "       and training will fail without them. Install it with:" >&2
        echo "         Windows (Git Bash): winget install GitHub.git-lfs" >&2
        echo "         Debian/Ubuntu:      sudo apt install git-lfs" >&2
        echo "         Fedora/RHEL:        sudo dnf install git-lfs" >&2
        echo "         macOS:              brew install git-lfs" >&2
        echo "       then run: git lfs install && git lfs pull" >&2
        exit 1
    fi
fi

case "$(uname -s 2>/dev/null)" in
    Linux) IS_WINDOWS=""; TBB=1 ;;
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS="" ;;
esac

if [ "$GPU" = rocm ] && [ -n "$IS_WINDOWS" ]; then
    echo "error: --gpu rocm is not available on Windows; ROCm wheels are Linux-only" >&2
    exit 1
fi

TBB_EXTRA=""
[ -n "$TBB" ] && TBB_EXTRA=" --extra tbb"

if [ "$GPU" = hygon ]; then
    echo "==> Install DTK-compatible Torch yourself in the Python environment used by .venv."
    echo "==> Torch, TorchVision, TorchAudio, Triton and NVIDIA/AMD management packages will not be installed."
    # Use UV_PYTHON or the current Python rather than the workspace's default
    # .python-version when no interpreter was explicitly requested.
    if [ ! -d .venv ]; then
        if [ -z "$PYTHON" ]; then
            set -- --python "${UV_PYTHON:-$(command -v python3)}"
        fi
        uv venv "$@" --system-site-packages .venv
    fi
    # Existing environments must already expose the externally managed DTK stack.
    if ! .venv/bin/python -c 'import sys; assert sys.prefix != sys.base_prefix; assert any(p.startswith(sys.base_prefix + "/lib/") and p.endswith("site-packages") for p in sys.path)'; then
        echo "error: Hygon requires a .venv created with --system-site-packages from your DTK Python" >&2
        exit 1
    fi
    REQUIREMENTS=$(mktemp)
    trap 'rm -f "$REQUIREMENTS"' EXIT HUP INT TERM
    set -- --frozen --no-hashes --no-header --no-annotate
    if [ -n "$ALL" ]; then
        set -- "$@" --all-packages --all-groups --extra rslrl --extra unitree --extra docs
    else
        set -- "$@" --no-default-groups
        [ -n "$RSLRL" ] && set -- "$@" --extra rslrl
        [ -n "$DOCS" ] && set -- "$@" --extra docs
    fi
    [ -n "$TBB" ] && set -- "$@" --extra tbb
    # Prune transitive accelerator dependencies too. --no-deps below prevents
    # editable workspace packages from pulling the skipped stack back in.
    uv export "$@" --prune torch --prune torchvision --prune torchaudio \
        --prune triton --prune triton-rocm --prune amdsmi --prune nvidia-ml-py \
        --output-file "$REQUIREMENTS" >/dev/null
    uv pip install --python .venv/bin/python --no-deps --requirements "$REQUIREMENTS"
    echo "==> Dependencies installed. Before training, install/verify DTK-compatible Torch yourself."
    echo "    Do not use plain uv sync: it would replace the externally managed accelerator stack."
elif [ -n "$ALL" ]; then
    if [ "$GPU" = rocm ]; then
        set -x
        uv sync "$@" --all-packages --all-groups --extra rocm --extra rslrl --extra unitree --extra docs$TBB_EXTRA
    else
        set -x
        uv sync "$@" --all-packages --all-groups --all-extras --no-extra rocm
    fi
else
    EXTRAS=""
    [ -n "$SKRL_JAX" ] && EXTRAS="$EXTRAS --extra skrl-jax"
    [ -n "$RSLRL" ] && EXTRAS="$EXTRAS --extra rslrl"
    [ -n "$DOCS" ] && EXTRAS="$EXTRAS --extra docs"
    EXTRAS="$EXTRAS$TBB_EXTRA"
    set -x
    uv sync "$@" --no-default-groups --extra "$GPU"$EXTRAS
fi
