# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.nn as nn
from motrix_rl_skrl.config import SkrlCfg, SkrlMemoryCfg
from motrix_rl_skrl.ppo import SkrlPpoTrainerBase, ppo_memory_size
from motrix_rl_skrl.torch import wrap_env
from skrl.agents.torch.ppo import PPO as BasePPO
from skrl.envs.wrappers.torch import Wrapper
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
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
    from skrl.memories.torch import RandomMemory

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
    _total_custom_rewards: dict[str, torch.Tensor] = {}

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
                value = torch.tensor(value, device=self.device)
                self.tracking_data[f"Reward Instant / {key} (max)"].append(torch.max(value).item())
                self.tracking_data[f"Reward Instant / {key} (min)"].append(torch.min(value).item())
                self.tracking_data[f"Reward Instant / {key} (mean)"].append(torch.mean(value).item())
                if key not in self._total_custom_rewards:
                    self._total_custom_rewards[key] = torch.zeros_like(value)
                self._total_custom_rewards[key] += value
            done = terminated | truncated
            done = done.reshape(-1)
            if done.any():
                for key in self._total_custom_rewards:
                    self.tracking_data[f"Reward Total/ {key} (mean)"].append(
                        torch.mean(self._total_custom_rewards[key][done]).item()
                    )
                    self.tracking_data[f"Reward Total/ {key} (min)"].append(
                        torch.min(self._total_custom_rewards[key][done]).item()
                    )
                    self.tracking_data[f"Reward Total/ {key} (max)"].append(
                        torch.max(self._total_custom_rewards[key][done]).item()
                    )

                    self._total_custom_rewards[key] = self._total_custom_rewards[key] * (~done)

        if "metrics" in infos:
            for key, value in infos["metrics"].items():
                tracked_value = torch.tensor(value, device=self.device)
                self.tracking_data[f"metrics / {key} (max)"].append(torch.max(tracked_value).item())
                self.tracking_data[f"metrics / {key} (min)"].append(torch.min(tracked_value).item())
                self.tracking_data[f"metrics / {key} (mean)"].append(torch.mean(tracked_value).item())


class Trainer(SkrlPpoTrainerBase):
    train_backend = "torch"
    checkpoint_format = "pt"
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

    def _play_context(self):
        return torch.no_grad()

    def _make_model(self, env: Wrapper, rlcfg: SkrlCfg) -> dict[str, Model]:
        _activation_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "selu": nn.SELU,
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

        def build_mlp(input_size: int, hidden_sizes: list[int], activations: list) -> nn.Sequential:
            layers = []
            current_size = input_size
            for hidden_size, act in zip(hidden_sizes, activations):
                layers.append(nn.Linear(current_size, hidden_size))
                layers.append(act())
                current_size = hidden_size
            return nn.Sequential(*layers)

        models = {}

        if separate:

            class Policy(GaussianMixin, Model):
                def __init__(self, observation_space, state_space, action_space, device, **kwargs):
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
                    self.net = build_mlp(self.num_observations, policy_cfg.hiddens, policy_acts)
                    self.mean_layer = nn.Linear(policy_cfg.hiddens[-1], self.num_actions)
                    self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), policy_cfg.initial_log_std))

                def compute(self, inputs, role):
                    x = self.net(inputs["observations"])
                    return self.mean_layer(x), {"log_std": self.log_std_parameter}

            class Value(DeterministicMixin, Model):
                def __init__(self, observation_space, state_space, action_space, device, **kwargs):
                    Model.__init__(
                        self,
                        observation_space=observation_space,
                        state_space=state_space,
                        action_space=action_space,
                        device=device,
                        **kwargs,
                    )
                    DeterministicMixin.__init__(self, clip_actions=value_cfg.clip_actions)
                    self.net = build_mlp(self.num_states, value_cfg.hiddens, value_acts)
                    self.value_layer = nn.Linear(value_cfg.hiddens[-1], 1)

                def compute(self, inputs, role):
                    x = self.net(inputs["states"])
                    return self.value_layer(x), {}

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
        else:
            if env.observation_space != env.state_space:
                raise ValueError("Shared SKRL models require observation_space and state_space to match")

            class Shared(GaussianMixin, DeterministicMixin, Model):
                def __init__(self, observation_space, state_space, action_space, device, **kwargs):
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
                    DeterministicMixin.__init__(self, clip_actions=value_cfg.clip_actions)
                    self.net = build_mlp(self.num_observations, policy_cfg.hiddens, policy_acts)
                    self.mean_layer = nn.Linear(policy_cfg.hiddens[-1], self.num_actions)
                    self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), policy_cfg.initial_log_std))
                    self.value_layer = nn.Linear(policy_cfg.hiddens[-1], 1)
                    self._shared_output = None

                def act(self, inputs, role):
                    if role == "policy":
                        return GaussianMixin.act(self, inputs, role=role)
                    elif role == "value":
                        return DeterministicMixin.act(self, inputs, role=role)

                def compute(self, inputs, role):
                    if role == "policy":
                        self._shared_output = self.net(inputs["observations"])
                        return self.mean_layer(self._shared_output), {"log_std": self.log_std_parameter}
                    elif role == "value":
                        shared = (
                            self._shared_output
                            if self._shared_output is not None
                            else self.net(inputs.get("states", inputs["observations"]))
                        )
                        self._shared_output = None
                        return self.value_layer(shared), {}

            models["policy"] = Shared(
                observation_space=env.observation_space,
                state_space=env.state_space,
                action_space=env.action_space,
                device=env.device,
            )
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
