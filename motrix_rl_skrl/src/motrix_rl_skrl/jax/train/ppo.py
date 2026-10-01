# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from motrix_rl_skrl.config import SkrlCfg, SkrlMemoryCfg
from motrix_rl_skrl.jax import wrap_env
from motrix_rl_skrl.ppo import SkrlPpoTrainerBase, ppo_memory_size
from skrl.agents.jax.ppo import PPO as BasePPO
from skrl.envs.wrappers.jax import Wrapper
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import RenderConfig


def _instantiate_memory(memory_cfg: SkrlMemoryCfg, memory_size: int, num_envs: int, device) -> Any:
    """Instantiate a SKRL Memory class based on configuration.

    Args:
        memory_cfg: Memory configuration with class_name and settings
        memory_size: Size of the memory buffer
        num_envs: Number of parallel environments
        device: Device to place memory on

    Returns:
        Instantiated SKRL Memory object

    Raises:
        ValueError: If class_name is not supported
    """
    from skrl.memories.jax import RandomMemory

    # Map class_name to actual Memory class
    memory_classes = {
        "RandomMemory": RandomMemory,
    }

    class_name = memory_cfg.class_name
    if class_name not in memory_classes:
        raise ValueError(f"Unsupported memory class_name: {class_name}. Supported: {list(memory_classes.keys())}")

    MemoryClass = memory_classes[class_name]
    return MemoryClass(memory_size=memory_size, num_envs=num_envs, device=device)


class PPO(BasePPO):
    _total_custom_rewards: dict[str, np.ndarray] = {}

    def record_transition(
        self,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ) -> None:
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if "Reward" in infos:
            for key, value in infos["Reward"].items():
                self.tracking_data[f"Reward Instant / {key} (max)"].append(jnp.max(value))
                self.tracking_data[f"Reward Instant / {key} (min)"].append(jnp.min(value))
                self.tracking_data[f"Reward Instant / {key} (mean)"].append(jnp.mean(value))
                if key not in self._total_custom_rewards:
                    self._total_custom_rewards[key] = jnp.zeros_like(value)
                self._total_custom_rewards[key] += value
            done = terminated | truncated
            done = done.reshape(-1)
            if done.any():
                for key in self._total_custom_rewards:
                    self.tracking_data[f"Reward Total/ {key} (mean)"].append(
                        jnp.mean(self._total_custom_rewards[key][done])
                    )
                    self.tracking_data[f"Reward Total/ {key} (min)"].append(
                        jnp.min(self._total_custom_rewards[key][done])
                    )
                    self.tracking_data[f"Reward Total/ {key} (max)"].append(
                        jnp.max(self._total_custom_rewards[key][done])
                    )

                    self._total_custom_rewards[key] = self._total_custom_rewards[key] * (1 - done)

        if "metrics" in infos:
            for key, value in infos["metrics"].items():
                self.tracking_data[f"metrics / {key} (max)"].append(jnp.max(value))
                self.tracking_data[f"metrics / {key} (min)"].append(jnp.min(value))
                self.tracking_data[f"metrics / {key} (mean)"].append(jnp.mean(value))


class Trainer(SkrlPpoTrainerBase):
    train_backend = "jax"
    checkpoint_format = "pickle"
    remove_mixed_precision = True
    scheduler_cls = KLAdaptiveLR
    scaler_cls = RunningStandardScaler
    trainer_cls = SequentialTrainer

    _trainer: SequentialTrainer

    def _make_env(self, num_envs: int, mode: str = "train"):
        return env_registry.make(
            self._env_name,
            num_envs=num_envs,
            mode=mode,
            sim=self._sim,
            seed=self._context.seed,
        )

    def _wrap_env(self, env, render: RenderConfig | None) -> Wrapper:
        return wrap_env(env, render)

    def _set_seed(self, seed: int | None) -> None:
        set_seed(seed)

    def _make_model(self, env: Wrapper, rlcfg: SkrlCfg) -> dict[str, Model]:
        _activation_fn = {
            "elu": nn.elu,
            "relu": nn.relu,
            "tanh": nn.tanh,
            "sigmoid": nn.sigmoid,
            "leaky_relu": nn.leaky_relu,
            "selu": nn.selu,
        }

        policy_cfg = rlcfg.models.policy
        value_cfg = rlcfg.models.value
        separate = rlcfg.models.separate

        def resolve_activations(activation_names: list[str], hiddens: list[int]) -> list:
            if len(activation_names) == 1:
                return [_activation_fn[activation_names[0]]] * len(hiddens)
            if len(activation_names) != len(hiddens):
                raise ValueError(
                    f"hidden_activation length ({len(activation_names)}) must be 1 or "
                    f"match hiddens length ({len(hiddens)})"
                )
            return [_activation_fn[name] for name in activation_names]

        policy_acts = resolve_activations(policy_cfg.hidden_activation, policy_cfg.hiddens)
        value_acts = resolve_activations(value_cfg.hidden_activation, value_cfg.hiddens)

        models = {}

        if separate:

            class Policy(GaussianMixin, Model):
                def __init__(self, observation_space, state_space, action_space, device=None, **kwargs):
                    Model.__init__(
                        self,
                        observation_space=observation_space,
                        state_space=state_space,
                        action_space=action_space,
                        device=device,
                        **kwargs,
                    )
                    GaussianMixin.__init__(
                        self,
                        clip_actions=policy_cfg.clip_actions,
                        clip_log_std=policy_cfg.clip_log_std,
                        min_log_std=policy_cfg.min_log_std,
                        max_log_std=policy_cfg.max_log_std,
                        reduction=policy_cfg.reduction,
                    )

                @nn.compact
                def __call__(self, inputs, role):
                    x = inputs["observations"]
                    for size, act in zip(policy_cfg.hiddens, policy_acts):
                        x = act(nn.Dense(size)(x))
                    x = nn.Dense(self.num_actions)(x)
                    log_std = self.param(
                        "log_std", lambda _: jnp.full(self.num_actions, float(policy_cfg.initial_log_std))
                    )
                    return x, {"log_std": log_std}

            class Value(DeterministicMixin, Model):
                def __init__(self, observation_space, state_space, action_space, device=None, **kwargs):
                    Model.__init__(
                        self,
                        observation_space=observation_space,
                        state_space=state_space,
                        action_space=action_space,
                        device=device,
                        **kwargs,
                    )
                    DeterministicMixin.__init__(self, clip_actions=value_cfg.clip_actions)

                @nn.compact
                def __call__(self, inputs, role):
                    x = inputs["states"]
                    for size, act in zip(value_cfg.hiddens, value_acts):
                        x = act(nn.Dense(size)(x))
                    x = nn.Dense(1)(x)
                    return x, {}

            models["policy"] = Policy(
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                device=env.device,
            )
            models["value"] = Value(
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                device=env.device,
            )

            for role, model in models.items():
                model.init_state_dict(role=role)
        else:
            if env.observation_space != env.state_space:
                raise ValueError("Shared SKRL models require observation_space and state_space to match")
            if value_cfg.clip_actions:
                raise ValueError("JAX shared SKRL value models do not support value.clip_actions=True")

            class Shared(GaussianMixin, Model):
                def __init__(self, observation_space, state_space, action_space, device=None, **kwargs):
                    Model.__init__(
                        self,
                        observation_space=observation_space,
                        state_space=state_space,
                        action_space=action_space,
                        device=device,
                        **kwargs,
                    )
                    GaussianMixin.__init__(
                        self,
                        clip_actions=policy_cfg.clip_actions,
                        clip_log_std=policy_cfg.clip_log_std,
                        min_log_std=policy_cfg.min_log_std,
                        max_log_std=policy_cfg.max_log_std,
                        reduction=policy_cfg.reduction,
                    )

                def act(self, inputs, role="", params=None):
                    if role == "policy":
                        return GaussianMixin.act(self, inputs, role=role, params=params)
                    if role == "value":
                        return self.apply(self.state_dict.params if params is None else params, inputs, role)
                    raise ValueError(f"Unsupported SKRL model role: {role}")

                @nn.compact
                def __call__(self, inputs, role):
                    x = inputs["observations"] if role == "policy" else inputs["states"]
                    for index, (size, act) in enumerate(zip(policy_cfg.hiddens, policy_acts)):
                        x = act(nn.Dense(size, name=f"shared_{index}")(x))
                    mean = nn.Dense(self.num_actions, name="mean_layer")(x)
                    value = nn.Dense(1, name="value_layer")(x)
                    log_std = self.param(
                        "log_std", lambda _: jnp.full(self.num_actions, float(policy_cfg.initial_log_std))
                    )
                    if role == "policy":
                        return mean, {"log_std": log_std}
                    if role == "value":
                        return value, {}
                    raise ValueError(f"Unsupported SKRL model role: {role}")

            models["policy"] = Shared(
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                device=env.device,
            )
            models["policy"].init_state_dict(role="policy")
            models["value"] = models["policy"]

        return models

    def _make_agent(
        self, models: dict[str, Model], env: Wrapper, ppo_cfg: dict[str, Any], memory_cfg: SkrlMemoryCfg
    ) -> PPO:
        memory_size = ppo_memory_size(memory_cfg.memory_size, ppo_cfg)
        memory = _instantiate_memory(memory_cfg, memory_size, env.num_envs, env.device)

        agent = PPO(
            models=models,
            memory=memory,
            cfg=ppo_cfg,
            observation_space=env.observation_space,
            state_space=env.state_space,
            action_space=env.action_space,
            device=env.device,
        )
        return agent


class Player:
    def __init__(self, env_name: str) -> None:
        pass
