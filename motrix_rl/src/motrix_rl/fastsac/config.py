# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class FastSacAgentCfg:
    """FastSAC agent hyperparameters (ported from holosoma ``FastSACConfig``)."""

    # optimization
    actor_learning_rate: float = MISSING
    critic_learning_rate: float = MISSING
    alpha_learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_grad_norm: float = MISSING

    # actor / critic networks
    actor_hidden_dim: int = MISSING
    critic_hidden_dim: int = MISSING
    num_q_networks: int = MISSING
    use_layer_norm: bool = MISSING
    use_tanh: bool = MISSING
    log_std_max: float = MISSING
    log_std_min: float = MISSING

    # distributional critic (C51)
    num_atoms: int = MISSING
    v_min: float = MISSING
    v_max: float = MISSING

    # SAC
    gamma: float = MISSING
    tau: float = MISSING
    alpha_init: float = MISSING
    use_autotune: bool = MISSING
    target_entropy_ratio: float = MISSING  # holosoma g1 fast_sac uses 0.0

    # replay / updates
    buffer_size: int = MISSING  # per environment
    num_steps: int = MISSING  # n-step returns
    batch_size: int = MISSING  # global batch size (split across envs)
    learning_starts: int = MISSING
    policy_frequency: int = MISSING
    num_updates: int = MISSING

    # observation normalization
    obs_normalization: bool = MISSING

    # perf knobs (auto-disabled on cpu)
    compile: bool = MISSING
    amp: bool = MISSING
    amp_dtype: str = MISSING


@dataclass
class FastSacAsyncOptionsCfg:
    """Heterogeneous collector/learner trainer configuration."""

    ring_capacity: int = MISSING
    utd_mode: str = MISSING
    weight_publish_interval: int = MISSING
    weight_poll_interval: int = MISSING
    max_ingest_per_iter: int = MISSING
    idle_sleep_s: float = MISSING
    collector_inference_device: str = MISSING
    collector_compile: bool = MISSING
    collector_amp: bool = MISSING
    collector_amp_dtype: str = MISSING
    # CPU affinity pinning for the async learner and collector processes, as a
    # core spec like "0:5,7" (cores 0-5 and 7). Out-of-range and duplicate
    # cores are dropped; an empty effective set or null disables pinning. The
    # env step is CPU-bound while the learner is GPU-bound, so isolating them
    # keeps learner-side drain/ingest work from stealing cores from the
    # collector.
    learner_cpu_cores: str | None = None
    collector_cpu_cores: str | None = None
    # Weight-snapshot transport between the async learner and collector:
    # "auto" picks CUDA-IPC device slots when both sides share one GPU and the
    # actor parameters are large enough (>= weight_ipc_min_bytes) for the
    # device path to pay off, and host shared memory otherwise; "on"/"off"
    # force the device/host path ("on" logs a warning and falls back to the
    # host path when learner and collector are not on the same GPU). Small
    # actors are faster on the host path —
    # the IPC path's stream synchronizations cost more than the sub-millisecond
    # host transfer they avoid.
    weight_ipc: str = "auto"
    weight_ipc_min_bytes: int = 16 * 1024 * 1024
    # Transition-ring transport for the large fields (obs/critic_obs): "auto"
    # picks CUDA-IPC device slots when learner and collector inference share
    # one GPU — the collector already uploads observations for inference, so
    # pushes become D2D copies and the learner drains without crossing the
    # host — and host shared memory otherwise; "on"/"off" force the device/
    # host path. The small scalar fields stay on the host path either way.
    ring_ipc: str = "auto"


@dataclass
class FastSacTrainerCfg:
    # number of environment-interaction iterations (each collects num_envs transitions)
    num_learning_iterations: int = MISSING
    async_options: FastSacAsyncOptionsCfg = field(default_factory=FastSacAsyncOptionsCfg)


@dataclass
class FastSacCfg:
    """Provider-specific FastSAC configuration."""

    # learning device; None -> cuda if available else cpu
    device: str | None = MISSING
    # Neutral policy-variant selector. The ``variant`` mapping is interpreted
    # only by the selected PolicyVariant implementation.
    policy_variant: str = "default"
    variant: dict[str, Any] = field(default_factory=dict)
    agent: FastSacAgentCfg = field(default_factory=FastSacAgentCfg)
    trainer: FastSacTrainerCfg = field(default_factory=FastSacTrainerCfg)

    # Execution topology. False runs the synchronous trainer; True runs the
    # heterogeneous collector/learner trainer.
    asynchronous: bool = MISSING
