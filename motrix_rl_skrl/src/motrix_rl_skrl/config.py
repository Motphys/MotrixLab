# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SKRL Configuration Classes

This module provides configuration classes for SKRL RL framework integration.
The configuration structure matches configs/algo_base/skrl.ppo.yaml with a hierarchical
design following the same pattern as RslrlConfig.

Configuration Hierarchy:
    SkrlCfg (top-level)
        -> SkrlModelsCfg (models)
            -> SkrlPolicyCfg (policy model)
            -> SkrlValueCfg (value model)
        -> SkrlMemoryCfg (memory)
        -> SkrlAgentCfg (PPO agent)
        -> SkrlTrainerCfg (trainer)
"""

import dataclasses
from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class SkrlPolicyCfg:
    """Configuration for SKRL policy (GaussianMixin) model.

    Corresponds to the policy section in configs/algo_base/skrl.ppo.yaml.
    """

    clip_actions: bool = MISSING
    clip_log_std: bool = MISSING
    initial_log_std: float = MISSING
    min_log_std: float = MISSING
    max_log_std: float = MISSING
    reduction: str = MISSING
    input: str = MISSING
    hiddens: list[int] = MISSING
    hidden_activation: list[str] = MISSING
    output: str = MISSING
    output_activation: str = MISSING
    output_scale: float = MISSING

    def _normalize_activations(self, num_layers: int) -> str | list[str]:
        """Normalize hidden_activation to match num_layers.

        SKRL requires either a single activation string (applied to all layers)
        or a list with length matching the number of layers.

        Args:
            num_layers: Number of hidden layers (len(self.hiddens))

        Returns:
            str or list[str] suitable for SKRL's network format

        Raises:
            ValueError: If activation list length > 1 and doesn't match num_layers
        """
        activations = self.hidden_activation

        # Empty list -> no activations
        if isinstance(activations, list) and len(activations) == 0:
            return [""] * num_layers

        # Single element list -> convert to string (SKRL will replicate)
        if isinstance(activations, list) and len(activations) == 1:
            return activations[0]

        # String -> return as-is (SKRL will replicate)
        if isinstance(activations, str):
            return activations

        # List with matching length -> use as-is
        if isinstance(activations, list) and len(activations) == num_layers:
            return activations

        # List with mismatched length > 1 -> raise error
        if isinstance(activations, list) and len(activations) > 1:
            raise ValueError(
                f"hidden_activation length ({len(activations)}) must match "
                f"the number of hidden layers ({num_layers}), or be a single value "
                f"to apply to all layers. Got hiddens={self.hiddens}, "
                f"hidden_activation={activations}"
            )

        return activations

    def to_network(self) -> tuple[list[dict], str]:
        """Convert configuration to SKRL's network and output format.

        Returns:
            (network, output) tuple where:
            - network: SKRL network definition list of dicts
            - output: SKRL output expression string (e.g., "tanh(ACTIONS)", "ONE")

        Examples:
            Policy with hiddens=[256,128,64], output_activation="tanh", output_scale=1.0:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "tanh(ACTIONS)"

            Value with hiddens=[256,128,64], output_activation="", output_scale=0.5:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "0.5 * ONE"
        """
        # Normalize activations to match hiddens length
        num_layers = len(self.hiddens)
        activations = self._normalize_activations(num_layers)

        # Build network definition
        network = [
            {
                "name": "net",
                "input": "STATES",
                "layers": self.hiddens,
                "activations": activations,
            }
        ]

        # Build output expression
        # Use output field directly (already in correct format)

        # Apply scale if not 1.0
        scale_prefix = f"{self.output_scale} * " if self.output_scale != 1.0 else ""

        # Apply activation if specified
        if self.output_activation:
            output = f"{scale_prefix}{self.output_activation}({self.output})"
        else:
            output = f"{scale_prefix}{self.output}"

        return network, output

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlValueCfg:
    """Configuration for SKRL value (DeterministicMixin) model.

    Corresponds to the value section in configs/algo_base/skrl.ppo.yaml.
    """

    clip_actions: bool = MISSING
    input: str = MISSING
    hiddens: list[int] = MISSING
    hidden_activation: list[str] = MISSING
    output: str = MISSING
    output_activation: str = MISSING
    output_scale: float = MISSING

    def _normalize_activations(self, num_layers: int) -> str | list[str]:
        """Normalize hidden_activation to match num_layers.

        SKRL requires either a single activation string (applied to all layers)
        or a list with length matching the number of layers.

        Args:
            num_layers: Number of hidden layers (len(self.hiddens))

        Returns:
            str or list[str] suitable for SKRL's network format

        Raises:
            ValueError: If activation list length > 1 and doesn't match num_layers
        """
        activations = self.hidden_activation

        # Empty list -> no activations
        if isinstance(activations, list) and len(activations) == 0:
            return [""] * num_layers

        # Single element list -> convert to string (SKRL will replicate)
        if isinstance(activations, list) and len(activations) == 1:
            return activations[0]

        # String -> return as-is (SKRL will replicate)
        if isinstance(activations, str):
            return activations

        # List with matching length -> use as-is
        if isinstance(activations, list) and len(activations) == num_layers:
            return activations

        # List with mismatched length > 1 -> raise error
        if isinstance(activations, list) and len(activations) > 1:
            raise ValueError(
                f"hidden_activation length ({len(activations)}) must match "
                f"the number of hidden layers ({num_layers}), or be a single value "
                f"to apply to all layers. Got hiddens={self.hiddens}, "
                f"hidden_activation={activations}"
            )

        return activations

    def to_network(self) -> tuple[list[dict], str]:
        """Convert configuration to SKRL's network and output format.

        Returns:
            (network, output) tuple where:
            - network: SKRL network definition list of dicts
            - output: SKRL output expression string (e.g., "tanh(ACTIONS)", "ONE")

        Examples:
            Policy with hiddens=[256,128,64], output_activation="tanh", output_scale=1.0:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "tanh(ACTIONS)"

            Value with hiddens=[256,128,64], output_activation="", output_scale=0.5:
                network = [{"name": "net", "input": "STATES", "layers": [256,128,64], "activations": "elu"}]
                output = "0.5 * ONE"
        """
        # Normalize activations to match hiddens length
        num_layers = len(self.hiddens)
        activations = self._normalize_activations(num_layers)

        # Build network definition
        network = [
            {
                "name": "net",
                "input": "STATES",
                "layers": self.hiddens,
                "activations": activations,
            }
        ]

        # Build output expression
        # Use output field directly (already in correct format)

        # Apply scale if not 1.0
        scale_prefix = f"{self.output_scale} * " if self.output_scale != 1.0 else ""

        # Apply activation if specified
        if self.output_activation:
            output = f"{scale_prefix}{self.output_activation}({self.output})"
        else:
            output = f"{scale_prefix}{self.output}"

        return network, output

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlModelsCfg:
    """Configuration for SKRL models section.

    Corresponds to the models section in configs/algo_base/skrl.ppo.yaml.
    """

    separate: bool = MISSING
    policy: SkrlPolicyCfg = field(default_factory=SkrlPolicyCfg)
    value: SkrlValueCfg = field(default_factory=SkrlValueCfg)

    def to_dict(self) -> dict:
        """Convert to dict with nested configs."""
        return {
            "separate": self.separate,
            "policy": self.policy.to_dict(),
            "value": self.value.to_dict(),
        }


@dataclass
class SkrlMemoryCfg:
    """Configuration for SKRL memory.

    Corresponds to the memory section in configs/algo_base/skrl.ppo.yaml.
    """

    class_name: str = MISSING
    memory_size: int = MISSING  # -1: automatically determined

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlAgentCfg:
    """Configuration for SKRL PPO agent.

    Corresponds to the agent section in configs/algo_base/skrl.ppo.yaml.
    Field names match PPO_DEFAULT_CONFIG from SKRL.
    """

    rollouts: int = MISSING
    learning_epochs: int = MISSING
    mini_batches: int = MISSING
    discount_factor: float = MISSING
    lam: float = MISSING
    learning_rate: float = MISSING
    # Optional: tasks disable the scheduler by setting this to None.
    learning_rate_scheduler: str | None = MISSING
    learning_rate_scheduler_kwargs: dict = MISSING
    random_timesteps: int = MISSING
    learning_starts: int = MISSING
    grad_norm_clip: float = MISSING
    ratio_clip: float = MISSING
    value_clip: float = MISSING
    # Kept for compatibility with existing task configs. SKRL 2.1 removed this option.
    clip_predicted_values: bool = MISSING
    entropy_loss_scale: float = MISSING
    value_loss_scale: float = MISSING
    kl_threshold: float = MISSING
    rewards_shaper_scale: float = MISSING
    time_limit_bootstrap: bool = MISSING
    mixed_precision: bool = MISSING

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for SKRL PPO agent.

        Returns:
            Dictionary representation matching SKRL's PPO agent configuration format.
            Maps lam -> gae_lambda for SKRL 2.1 compatibility.

        Note:
            Excludes runtime-owned experiment and preprocessor fields, which are
            added dynamically during training or play.
        """
        # Build base configuration dict
        result = {
            "rollouts": self.rollouts,
            "learning_epochs": self.learning_epochs,
            "mini_batches": self.mini_batches,
            "discount_factor": self.discount_factor,
            "gae_lambda": self.lam,
            "learning_rate": self.learning_rate,
            "learning_rate_scheduler": self.learning_rate_scheduler,
            "learning_rate_scheduler_kwargs": self.learning_rate_scheduler_kwargs,
            "random_timesteps": self.random_timesteps,
            "learning_starts": self.learning_starts,
            "grad_norm_clip": self.grad_norm_clip,
            "ratio_clip": self.ratio_clip,
            "value_clip": self.value_clip,
            "entropy_loss_scale": self.entropy_loss_scale,
            "value_loss_scale": self.value_loss_scale,
            "kl_threshold": self.kl_threshold,
            "rewards_shaper_scale": self.rewards_shaper_scale,
            "time_limit_bootstrap": self.time_limit_bootstrap,
            "mixed_precision": self.mixed_precision,
        }

        return result


@dataclass
class SkrlTrainerCfg:
    """Configuration for SKRL sequential trainer.

    Corresponds to the trainer section in configs/algo_base/skrl.ppo.yaml.
    """

    timesteps: int = MISSING
    """
    The max number of batch env steps to run
    """

    def to_dict(self) -> dict:
        """Convert to dict, mapping class_name to class."""
        from motrix_rl.utils import class_to_dict

        return class_to_dict(self)


@dataclass
class SkrlCfg:
    """Top-level SKRL configuration.

    Contains only provider-specific settings. Environment parallelism and the
    authoritative seed live in the framework-owned task runtime config.
    """

    models: SkrlModelsCfg = field(default_factory=SkrlModelsCfg)
    memory: SkrlMemoryCfg = field(default_factory=SkrlMemoryCfg)
    agent: SkrlAgentCfg = field(default_factory=SkrlAgentCfg)
    trainer: SkrlTrainerCfg = field(default_factory=SkrlTrainerCfg)

    def replace(self, **updates) -> "SkrlCfg":
        """Replace specified fields and return a new instance."""
        return dataclasses.replace(self, **updates)
