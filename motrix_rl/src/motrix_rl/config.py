# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Structured configuration types shared by CLI and runtime code."""

import math
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class LoggingConfig:
    """Framework-neutral training log settings."""

    backend: str = MISSING
    interval: int = MISSING


@dataclass
class CheckpointConfig:
    """Framework-neutral periodic checkpoint settings."""

    interval: int = MISSING


@dataclass
class OnnxParityConfig:
    """Numerical parity settings applied to every exported ONNX policy."""

    seed: int = 1
    samples: int = 32
    atol: float = 1e-4
    rtol: float = 1e-5

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError(f"validation samples must be positive, got {self.samples}")
        if not math.isfinite(self.atol) or not math.isfinite(self.rtol) or self.atol < 0 or self.rtol < 0:
            raise ValueError(f"validation tolerances must be non-negative, got atol={self.atol}, rtol={self.rtol}")


@dataclass
class OnnxExportConfig:
    """Typed Hydra configuration for the ONNX export CLI."""

    run_dir: str = MISSING
    output: str | None = None
    opset: int = 18
    parity: OnnxParityConfig = field(default_factory=OnnxParityConfig)


@dataclass
class DeploymentExportConfig:
    """Typed Hydra configuration for deployment artifact export."""

    env: str | None = None
    run_dir: str | None = None
    output: str | None = None
    validation: OnnxParityConfig = field(
        default_factory=lambda: OnnxParityConfig(atol=1e-5),
    )


@dataclass
class TaskMeta:
    """Method and environment selected by one Hydra task recipe."""

    env: str = MISSING
    rllib: str = MISSING
    algo: str = MISSING
    train_backend: str | None = None


@dataclass
class TaskConfig:
    """A composed task recipe with runtime policy and algorithm config."""

    task: TaskMeta = MISSING
    num_envs: int = MISSING
    play_num_envs: int = MISSING
    seed: int | None = MISSING
    logging: LoggingConfig = MISSING
    checkpoint: CheckpointConfig = MISSING
    # The selected algo base supplies the concrete provider-owned dataclass.
    algo: Any = MISSING


@dataclass
class TrainConfig(TaskConfig):
    """Top-level schema for ``mxlab train``."""

    render: bool = False
    play: bool = False
    sim: str | None = None
    resume: str | None = None


@dataclass
class PlayConfig:
    """Top-level schema for ``mxlab play``."""

    env: str | None = None
    sim: str | None = None
    policy: str | None = None
    num_envs: int | None = None
    seed: int | None = None
    rand_seed: bool = False
    record_video: bool = False
    record_seconds: float = 10.0
    record_width: int = 256
    record_height: int = 256
    # Deprecated for play; only used to filter metadata-backed runs.
    rllib: str | None = None
    rl: Any = field(default_factory=dict)


def register_configs() -> None:
    """Register CLI schemas in Hydra's ConfigStore. Idempotent."""
    cs = ConfigStore.instance()
    cs.store(name="train_schema", node=TrainConfig)
    cs.store(name="play_schema", node=PlayConfig)
    cs.store(name="export_onnx_schema", node=OnnxExportConfig)
    cs.store(name="export_deploy_schema", node=DeploymentExportConfig)


register_configs()
