#!/bin/sh
# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

# Project bootstrap: runtime environment (`sh install.sh`) or full
# development environment (`sh install.sh --all`).
#
# GPU wheel flavor (--gpu): cuda or rocm. When not given explicitly, detect
# from the loaded kernel driver (nvidia -> cuda, amdgpu -> rocm); CPU-only
# hosts fall back to cuda.
#
# Training backends are enabled with --skrl-torch (default), --skrl-jax or
# --rslrl, named after the extras that install them; multiple flags combine.

set -e

usage() {
    echo "Usage: sh install.sh [options]"
    echo
    echo "Options:"
    echo "  --all                  Install the full development environment (all packages, groups and extras;"
    echo "                         ROCm hosts skip CUDA-only extras such as skrl-jax)"
    echo "  --docs                 Include the docs toolchain (sphinx) for building the documentation"
    echo "  --gpu cuda|rocm        GPU wheel flavor (default: auto-detected from the kernel driver)"
    echo "  --skrl-torch           Training backend: SKRL on PyTorch (default)"
    echo "  --skrl-jax             Training backend: SKRL on JAX (Linux only)"
    echo "  --rslrl                Training backend: RSL-RL on PyTorch"
    echo "  -h, --help             Show this help"
}

ALL=""
GPU=""
SKRL_TORCH=""
SKRL_JAX=""
RSLRL=""
DOCS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            ALL=1
            ;;
        --docs)
            DOCS=1
            ;;
        --gpu)
            [ $# -ge 2 ] || { echo "error: $1 requires a value (cuda or rocm)" >&2; exit 1; }
            GPU="$2"
            shift
            ;;
        --skrl-torch)
            SKRL_TORCH=1
            ;;
        --skrl-jax)
            SKRL_JAX=1
            ;;
        --rslrl)
            RSLRL=1
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

if [ -n "$GPU" ] && [ "$GPU" != cuda ] && [ "$GPU" != rocm ]; then
    echo "error: invalid --gpu value: $GPU (expected cuda or rocm)" >&2
    usage >&2
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

if [ -z "$GPU" ]; then
    if lsmod 2>/dev/null | grep -q '^nvidia '; then
        GPU=cuda
    elif lsmod 2>/dev/null | grep -q '^amdgpu '; then
        GPU=rocm
    else
        GPU=cuda
    fi
fi

case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS="" ;;
esac

if [ "$GPU" = rocm ] && [ -n "$IS_WINDOWS" ]; then
    echo "error: --gpu rocm is not available on Windows; ROCm wheels are Linux-only" >&2
    exit 1
fi

if [ -n "$ALL" ]; then
    if [ "$GPU" = rocm ]; then
        set -x
        uv sync --all-packages --all-groups --extra rocm --extra skrl-torch --extra rslrl --extra unitree --extra docs
    else
        set -x
        uv sync --all-packages --all-groups --all-extras --no-extra rocm
    fi
else
    if [ -z "$SKRL_TORCH" ] && [ -z "$SKRL_JAX" ] && [ -z "$RSLRL" ]; then
        SKRL_TORCH=1
    fi
    EXTRAS=""
    [ -n "$SKRL_TORCH" ] && EXTRAS="$EXTRAS --extra skrl-torch"
    [ -n "$SKRL_JAX" ] && EXTRAS="$EXTRAS --extra skrl-jax"
    [ -n "$RSLRL" ] && EXTRAS="$EXTRAS --extra rslrl"
    [ -n "$DOCS" ] && EXTRAS="$EXTRAS --extra docs"
    set -x
    uv sync --all-packages --no-default-groups --extra "$GPU"$EXTRAS
fi
