# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""PPO Agent Configuration for RSLRL

This module provides configuration classes for PPO agents using the
RSLRL (ETH Zurich RL library) framework.

The configuration structure matches rsl_rl's flat format with separate
actor and critic configs at the top level.
"""

from dataclasses import dataclass, field

from omegaconf import MISSING

from motrix_rl.utils import class_to_dict


@dataclass
class RslRlActorCfg:
    """Configuration for the actor network."""

    class_name: str = MISSING
    hidden_dims: list[int] = MISSING
    activation: str = MISSING
    obs_normalization: bool = MISSING
    stochastic: bool = MISSING
    init_noise_std: float = MISSING
    noise_std_type: str = MISSING  # one of: "scalar", "log"
    state_dependent_std: bool = MISSING


@dataclass
class RslRlCriticCfg:
    """Configuration for the critic network."""

    class_name: str = MISSING
    hidden_dims: list[int] = MISSING
    activation: str = MISSING
    obs_normalization: bool = MISSING
    stochastic: bool = MISSING


@dataclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = MISSING
    optimizer: str = MISSING
    learning_rate: float = MISSING
    num_learning_epochs: int = MISSING
    num_mini_batches: int = MISSING
    schedule: str = MISSING
    value_loss_coef: float = MISSING
    clip_param: float = MISSING
    use_clipped_value_loss: bool = MISSING
    desired_kl: float = MISSING
    entropy_coef: float = MISSING
    gamma: float = MISSING
    lam: float = MISSING
    max_grad_norm: float = MISSING
    normalize_advantage_per_mini_batch: bool = MISSING
    rnd_cfg: dict | None = MISSING
    symmetry_cfg: dict | None = MISSING


@dataclass
class RslrlRunnerCfg:
    """Configuration matching rsl_rl's flat structure.

    This configuration provides separate actor and critic configs at the top level,
    matching the structure expected by rsl_rl's OnPolicyRunner.
    """

    # Runner settings
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING

    # Observation groups
    obs_groups: dict[str, list[str]] = MISSING

    # Network configs - TOP LEVEL
    actor: RslRlActorCfg = field(default_factory=RslRlActorCfg)
    critic: RslRlCriticCfg = field(default_factory=RslRlCriticCfg)
    algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)

    def to_dict(self) -> dict:
        """Convert config to dictionary for OnPolicyRunner.

        Returns:
            Dictionary representation matching rsl_rl's expected format.
        """
        return class_to_dict(self)


@dataclass
class RslrlCfg(RslrlRunnerCfg):
    """Provider config completed with runtime logging/checkpoint settings before dispatch."""
