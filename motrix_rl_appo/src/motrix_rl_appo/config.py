# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration for the standalone Torch APPO plugin."""

import math
from dataclasses import dataclass, field


@dataclass
class NetworkCfg:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    normalize_observations: bool = True
    init_noise_std: float = 0.5

    def __post_init__(self) -> None:
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.activation not in ("elu", "relu", "tanh"):
            raise ValueError("activation must be elu, relu or tanh")
        if not math.isfinite(self.init_noise_std) or self.init_noise_std <= 0:
            raise ValueError("init_noise_std must be finite and positive")


@dataclass
class AlgorithmCfg:
    learning_rate: float = 1e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    normalize_advantage: bool = True
    rho_clip: float = 1.0
    c_clip: float = 1.0
    pg_rho_clip: float = 1.0
    worker_target_clip: float = 2.0
    target_update_interval: int = 1
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    execution: str = "eager"
    fused_optimizer: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.learning_rate,
            self.max_grad_norm,
            self.rho_clip,
            self.c_clip,
            self.pg_rho_clip,
            self.worker_target_clip,
            self.desired_kl,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("learning rate, gradient norm, importance caps and desired KL must be finite and positive")
        if min(self.num_learning_epochs, self.num_mini_batches, self.target_update_interval) <= 0:
            raise ValueError("epochs, minibatches and target interval must be positive")
        if not 0 <= self.gamma <= 1 or not 0 < self.clip_param < 1:
            raise ValueError("gamma must be in [0, 1] and clip_param in (0, 1)")
        if any(not math.isfinite(value) or value < 0 for value in (self.value_loss_coef, self.entropy_coef)):
            raise ValueError("loss coefficients must be finite and nonnegative")
        if self.execution not in ("eager", "compile", "cuda_graph"):
            raise ValueError("execution must be eager, compile or cuda_graph")
        if self.execution == "cuda_graph" and not self.fused_optimizer:
            raise ValueError("cuda_graph execution requires fused_optimizer=True")
        if self.schedule not in ("fixed", "adaptive"):
            raise ValueError("schedule must be fixed or adaptive")


@dataclass
class AppoCfg:
    actor: NetworkCfg = field(default_factory=NetworkCfg)
    critic: NetworkCfg = field(default_factory=NetworkCfg)
    algorithm: AlgorithmCfg = field(default_factory=AlgorithmCfg)
    rollout_steps: int = 24
    max_iterations: int = 2000
    queue_chunks: int = 2
    device: str | None = None
    collector_device: str | None = None
    transition_transport: str = "auto"
    weight_transport: str = "auto"
    worker_timeout: float = 120.0

    def __post_init__(self) -> None:
        if self.transition_transport not in ("auto", "host", "cuda_ipc"):
            raise ValueError("transition_transport must be auto, host or cuda_ipc")
        if self.weight_transport not in ("auto", "host", "cuda_ipc"):
            raise ValueError("weight_transport must be auto, host or cuda_ipc")
        if min(self.rollout_steps, self.max_iterations, self.queue_chunks) <= 0:
            raise ValueError("rollout_steps, max_iterations and queue_chunks must be positive")
        if not math.isfinite(self.worker_timeout) or self.worker_timeout <= 0:
            raise ValueError("worker_timeout must be finite and positive")
