# syntax=docker/dockerfile:1.7-labs

# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Build from the repository root so the workspace manifests and Git LFS assets
# are part of the context:
#
#   docker build -f docker/Dockerfile --target train-cuda \
#     -t motphys/motrixlab:0.4.0-cuda .
#
# The labs syntax is needed for COPY --parents, which keeps dependency layers
# independent of workspace source changes.

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.4
ARG UNITREE_SDK_IMAGE=no-unitree-sdk

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS python-base

ARG UV_IMAGE
ARG USERNAME=motrix
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    TZ=Asia/Shanghai

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libgomp1 \
        libx11-6 \
        libxext6 \
        tzdata \
    && ln -fs "/usr/share/zoneinfo/${TZ}" /etc/localtime \
    && echo "${TZ}" > /etc/timezone \
    && groupadd --gid "${GID}" "${USERNAME}" \
    && useradd --uid "${UID}" --gid "${GID}" --create-home "${USERNAME}" \
    && mkdir -p /app /workspace/runs \
    && chown -R "${UID}:${GID}" /app /workspace \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

USER ${UID}:${GID}

COPY --chown=${UID}:${GID} --parents motrix*/pyproject.toml ./
COPY --chown=${UID}:${GID} pyproject.toml uv.lock ./

# The locked Unitree dependency is fetched from Git. Keep that requirement out
# of training and simulation deployment images.
FROM python-base AS source-tooling
ARG UID=1000
ARG GID=1000
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER ${UID}:${GID}

# Install only locked third-party dependencies first. Workspace source is not
# copied yet, so normal source edits reuse these heavy layers.
FROM python-base AS train-cuda-deps
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups \
      --extra cuda --extra rslrl --extra docs --extra tbb \
      --no-install-workspace

FROM train-cuda-deps AS train-cuda
ARG MOTRIXLAB_VERSION=0.4.0
ARG UID=1000
ARG GID=1000
LABEL org.opencontainers.image.title="MotrixLab CUDA training runtime" \
      org.opencontainers.image.description="MotrixLab training stack with CUDA Torch, SKRL, RSLRL, and TensorBoard." \
      org.opencontainers.image.source="https://github.com/Motphys/MotrixLab" \
      org.opencontainers.image.version="${MOTRIXLAB_VERSION}"
COPY --chown=${UID}:${GID} --parents motrix* ./
COPY --chown=${UID}:${GID} configs /workspace/configs
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups --no-editable \
      --extra cuda --extra rslrl --extra docs --extra tbb

WORKDIR /workspace
ENTRYPOINT ["/app/.venv/bin/mxlab"]
CMD ["--help"]

FROM train-cuda AS train-cuda-jax
ARG MOTRIXLAB_VERSION=0.4.0
ARG UID=1000
ARG GID=1000
LABEL org.opencontainers.image.title="MotrixLab CUDA training runtime with JAX" \
      org.opencontainers.image.description="MotrixLab CUDA training stack with SKRL Torch/JAX, FastSAC, RSLRL, and TensorBoard." \
      org.opencontainers.image.source="https://github.com/Motphys/MotrixLab" \
      org.opencontainers.image.version="${MOTRIXLAB_VERSION}"
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --project /app --frozen --no-default-groups --no-editable \
      --extra cuda --extra skrl-jax --extra rslrl --extra docs --extra tbb

WORKDIR /workspace
ENTRYPOINT ["/app/.venv/bin/mxlab"]
CMD ["--help"]

FROM python-base AS train-rocm-deps
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups \
      --extra rocm --extra rslrl --extra docs --extra tbb \
      --no-install-workspace

FROM train-rocm-deps AS train-rocm
ARG MOTRIXLAB_VERSION=0.4.0
ARG UID=1000
ARG GID=1000
LABEL org.opencontainers.image.title="MotrixLab ROCm training runtime" \
      org.opencontainers.image.description="MotrixLab training stack with ROCm Torch, SKRL, RSLRL, and TensorBoard." \
      org.opencontainers.image.source="https://github.com/Motphys/MotrixLab" \
      org.opencontainers.image.version="${MOTRIXLAB_VERSION}"
COPY --chown=${UID}:${GID} --parents motrix* ./
COPY --chown=${UID}:${GID} configs /workspace/configs
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups --no-editable \
      --extra rocm --extra rslrl --extra docs --extra tbb

WORKDIR /workspace
ENTRYPOINT ["/app/.venv/bin/mxlab"]
CMD ["--help"]

FROM python-base AS deploy-deps
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups \
      --package motrix-deploy \
      --package motrix-deploy-mujoco \
      --package motrix-deploy-motrixsim \
      --package motrix-deploy-tasks \
      --package motrix-envs \
      --no-install-workspace

FROM deploy-deps AS deploy
ARG MOTRIXLAB_VERSION=0.4.0
ARG UID=1000
ARG GID=1000
LABEL org.opencontainers.image.title="MotrixLab policy deployment runtime" \
      org.opencontainers.image.description="Framework-independent MotrixLab deployment runtime with ONNX Runtime and simulation backends." \
      org.opencontainers.image.source="https://github.com/Motphys/MotrixLab" \
      org.opencontainers.image.version="${MOTRIXLAB_VERSION}"
COPY --chown=${UID}:${GID} --parents motrix* ./
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups --no-editable \
      --package motrix-deploy \
      --package motrix-deploy-mujoco \
      --package motrix-deploy-motrixsim \
      --package motrix-deploy-tasks \
      --package motrix-envs

WORKDIR /workspace
ENTRYPOINT ["/app/.venv/bin/python", "-c", "from motrix_deploy.cli import main; main()"]
CMD ["--help"]

# The default SDK stage is intentionally empty. To produce a hardware-enabled
# image, build an SDK image containing /opt/unitree_robotics and override this
# build argument with that image reference.
FROM python-base AS no-unitree-sdk
USER root
RUN mkdir -p /opt/unitree_robotics

FROM ${UNITREE_SDK_IMAGE} AS unitree-sdk

FROM source-tooling AS unitree-deps
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups \
      --package motrix-deploy \
      --package motrix-deploy-tasks \
      --package motrix-deploy-unitree \
      --extra native \
      --no-install-workspace

FROM unitree-deps AS unitree
ARG MOTRIXLAB_VERSION=0.4.0
ARG UID=1000
ARG GID=1000
LABEL org.opencontainers.image.title="MotrixLab Unitree deployment runtime" \
      org.opencontainers.image.description="Unitree SDK2/DDS policy deployment runtime for MotrixLab artifacts." \
      org.opencontainers.image.source="https://github.com/Motphys/MotrixLab" \
      org.opencontainers.image.version="${MOTRIXLAB_VERSION}"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=unitree-sdk --chown=${UID}:${GID} /opt/unitree_robotics /opt/unitree_robotics
COPY --chown=${UID}:${GID} --parents motrix* ./
USER ${UID}:${GID}
ENV UNITREE_SDK_ROOT=/opt/unitree_robotics
RUN --mount=type=cache,target=/home/motrix/.cache/uv,uid=1000,gid=1000,sharing=locked \
    uv sync --frozen --no-default-groups --no-editable \
      --package motrix-deploy \
      --package motrix-deploy-tasks \
      --package motrix-deploy-unitree \
      --extra native

WORKDIR /workspace
ENTRYPOINT ["/app/.venv/bin/python", "-c", "from motrix_deploy.cli import main; main()"]
CMD ["--help"]

# Keep an unqualified `docker build` on the supported default training profile
# instead of the larger JAX profile.
FROM train-cuda AS default
