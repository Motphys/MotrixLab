# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._episode_totals: dict[str, torch.Tensor] = {}
        self._tracking_window: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._tracking_dirty = False

    def _track_values(self, tag: str, values: torch.Tensor, done: torch.Tensor | None = None) -> None:
        if done is None:
            stats = torch.stack((values.amin(), values.mean(), values.amax()))
        else:
            count = done.sum()
            lo = values.masked_fill(~done, float("inf")).amin()
            hi = values.masked_fill(~done, -float("inf")).amax()
            mean = values.masked_fill(~done, 0).sum() / count.clamp_min(1)
            stats = torch.where(count > 0, torch.stack((lo, mean, hi)), float("nan"))
        self._tracking_window[tag].append(stats)
        self._tracking_dirty = True

    def _track_episode(self, tag: str, values: torch.Tensor, done: torch.Tensor) -> None:
        if tag not in self._episode_totals:
            self._episode_totals[tag] = torch.zeros_like(values)
        total = self._episode_totals[tag]
        total.add_(values)
        self._track_values(tag, total, done)
        total.masked_fill_(done, 0)

    def _publish_tracking(self) -> None:
        if not self._tracking_dirty:
            return
        summaries = []
        for window in self._tracking_window.values():
            lo, mean, hi = torch.stack(window).unbind(dim=1)
            summaries.append(
                torch.stack(
                    (
                        torch.nan_to_num(lo, nan=float("inf"), posinf=float("inf"), neginf=-float("inf")).amin(),
                        torch.nanmean(mean),
                        torch.nan_to_num(hi, nan=-float("inf"), posinf=float("inf"), neginf=-float("inf")).amax(),
                    )
                )
            )
        for tag, (lo, mean, hi) in zip(self._tracking_window, torch.stack(summaries).cpu().tolist()):
            if mean == mean:
                for stat, value in zip(("min", "mean", "max"), (lo, mean, hi)):
                    self.tracking_data[f"{tag} ({stat})"] = [value]
        self._tracking_dirty = False

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if timestep > 0 and self.checkpoint_interval > 0 and (timestep + 1) % self.checkpoint_interval == 0:
            self._publish_tracking()
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        self._publish_tracking()
        super().write_tracking_data(timestep=timestep, timesteps=timesteps)
        self._tracking_window.clear()

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
        interval = self.write_interval
        device_tracking = rewards.is_cuda and interval > 0
        if device_tracking:
            # The native tracker performs dynamic indexing and host reads each step.
            self.write_interval = 0
        try:
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
        finally:
            self.write_interval = interval
        if interval <= 0:
            return

        done = (terminated | truncated).reshape(-1)
        if device_tracking:
            values = rewards.reshape(-1).float()
            self._track_values("Reward / Instantaneous reward", values)
            self._track_episode("Reward / Total reward", values, done)
            self._track_episode("Episode / Total timesteps", torch.ones_like(values), done)
        for key, value in infos.get("Reward", {}).items():
            value = torch.as_tensor(value, device=self.device).detach().float()
            self._track_values(f"Reward Instant / {key}", value)
            self._track_episode(f"Reward Total/ {key}", value, done)
        for key, value in infos.get("metrics", {}).items():
            mean = torch.as_tensor(value, device=self.device).detach().float().mean()
            self._track_values(f"metrics / {key}", mean)


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
            **({"sim_device": self._context.sim_device} if self._context.sim_device is not None else {}),
            seed=self._context.seed,
            headless=self._render.headless if self._render else False,
            record=self._render.record if self._render else None,
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
