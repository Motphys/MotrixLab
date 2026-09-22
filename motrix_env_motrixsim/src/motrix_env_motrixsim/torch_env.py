# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
from dataclasses import dataclass
from typing import Generic, TypeVar

import motrixsim as mtx
import numpy as np
import torch
from gymnasium import Space, spaces

from motrix_env_core.base import ABEnv, EnvCfg, ObsSpace
from motrix_env_core.perf import Perf, perf_root
from motrix_env_core.sim.backend import (
    RenderConfig,
    SimRenderer,
)
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.renderer import MotrixSimRenderer

# The concrete config type a TorchEnv subclass consumes. Subclasses parametrize
# via ``TorchEnv[MyCfg]`` so ``self.cfg`` is typed as ``MyCfg`` (IDE hint + pyright
# narrowing). Unparametrized subclasses fall back to ``EnvCfg`` (back-compat).
EnvCfgType = TypeVar("EnvCfgType", bound=EnvCfg)


@dataclass
class TorchObs:
    policy: torch.Tensor
    value: torch.Tensor | None = None

    @property
    def value_or_policy(self) -> torch.Tensor:
        return self.policy if self.value is None else self.value


@dataclass
class TorchEnvState:
    data: mtx.SceneData
    obs: TorchObs
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_steps: torch.Tensor
    # Per-term reward breakdown for the latest transition: term name to a
    # ``(num_envs,)`` tensor. Empty when the environment does not decompose
    # its reward.
    reward_terms: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    # Instantaneous diagnostics snapshot: scalar per key, reduced across all
    # environments. Recomputed every step by ``update_state``; resets do not
    # clear or recompute it.
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def done(self) -> torch.Tensor:
        """
        Check if the environment is done.
        """
        return torch.logical_or(self.terminated, self.truncated)

    def replace(self, **updates) -> "TorchEnvState":
        return dataclasses.replace(self, **updates)

    def process_metrics(self) -> dict[str, float]:
        """Return the already-reduced batch-level metric scalars."""
        return dict(self.metrics)

    def validate(self, expected_device: torch.device | str | None = None, num_envs: int | None = None):
        if num_envs is None:
            num_envs = self.data.shape[0]
        assert self.obs.policy.shape[0] == num_envs, self.obs.policy.shape
        if self.obs.value is not None:
            assert self.obs.value.shape[0] == num_envs, self.obs.value.shape
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.episode_steps.shape == (num_envs,), self.episode_steps.shape

        device = self.obs.policy.device
        if expected_device is not None:
            assert device == torch.device(expected_device), device
        assert self.reward.device == device, self.reward.device
        assert self.terminated.device == device, self.terminated.device
        assert self.truncated.device == device, self.truncated.device
        assert self.episode_steps.device == device, self.episode_steps.device
        if self.obs.value is not None:
            assert self.obs.value.device == device, self.obs.value.device


def _as_obs_space(space: Space | ObsSpace) -> ObsSpace:
    if isinstance(space, ObsSpace):
        return space
    assert isinstance(space, spaces.Box)
    return ObsSpace(policy=space)


def _as_obs(obs: TorchObs | torch.Tensor) -> TorchObs:
    if isinstance(obs, TorchObs):
        return obs
    assert isinstance(obs, torch.Tensor)
    return TorchObs(policy=obs)


class TorchEnv(ABEnv, Generic[EnvCfgType]):
    """MotrixSim environment exposing a torch-tensor frontend."""

    _model: mtx.SceneModel
    _cfg: EnvCfgType
    _state: TorchEnvState | None = None
    _render_spacing: float

    def __init__(
        self,
        cfg: EnvCfgType,
        num_envs: int = 1,
        device: torch.device | str = "cpu",
    ):
        self._cfg = cfg
        self._num_envs = num_envs
        self._device = torch.device(device)
        if self._device.type != "cpu":
            raise ValueError(f"TorchEnv currently supports CPU simulation only, got device '{self._device}'.")
        self._device = torch.device("cpu")
        if cfg.scene is None:
            raise ValueError("EnvCfg.scene must be configured")
        self._model = MotrixSimSceneCompiler().compile(cfg.scene, cfg.sim)
        self._render_spacing = cfg.render_spacing
        self.perf = Perf()

    @property
    def model(self) -> mtx.SceneModel:
        """
        Get the scene model
        """
        return self._model

    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        self._require_render_device()
        return MotrixSimRenderer(
            self._model,
            lambda: self._state.data,
            config,
            num_envs=self.num_envs,
            render_spacing=self.render_spacing,
            system_camera=self.cfg.scene.system_camera,
        )

    def _require_render_device(self) -> None:
        if self._device.type != "cpu":
            # Build a CPU render model and synchronize Warp poses here once the
            # Warp simulation backend is integrated.
            raise ValueError(f"Torch rendering currently supports CPU simulation only, got device '{self._device}'.")

    @property
    def state(self) -> TorchEnvState | None:
        """
        Get the current environment state
        """
        return self._state

    @property
    def cfg(self) -> EnvCfgType:
        """
        Get the environment configuration
        """
        return self._cfg

    @property
    def render_spacing(self) -> float:
        """
        Get the render spacing, with which the multi-envs will be rendered seperately in grid
        """
        return self._render_spacing

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> torch.device:
        """Device used by observations, actions, rewards, and episode state tensors."""
        return self._device

    def init_state(self) -> TorchEnvState:
        """
        Create a new environment state
        """
        obs = self._zeros_obs(self.observation_space)
        reward = torch.zeros((self._num_envs,), dtype=torch.float32, device=self._device)
        terminated = torch.ones((self._num_envs,), dtype=torch.bool, device=self._device)
        truncated = torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device)
        episode_steps = torch.zeros((self._num_envs,), dtype=torch.int64, device=self._device)
        data = mtx.SceneData(self._model, batch=[self._num_envs])
        self._state = TorchEnvState(data, obs, reward, terminated, truncated, episode_steps)
        self._reset_done_envs()
        self._state.validate(self._device, self._num_envs)
        return self._state

    @property
    def policy_observation_space(self) -> spaces.Box:
        return _as_obs_space(self.observation_space).policy

    @property
    def value_observation_space(self) -> spaces.Box:
        return _as_obs_space(self.observation_space).value_or_policy

    @property
    def has_value_observation(self) -> bool:
        return _as_obs_space(self.observation_space).value is not None

    def _zeros_box(self, space: spaces.Box) -> torch.Tensor:
        dtype = torch.from_numpy(np.empty((), dtype=space.dtype)).dtype
        return torch.zeros((self._num_envs, *space.shape), dtype=dtype, device=self._device)

    def _zeros_obs(self, space: Space | ObsSpace) -> TorchObs:
        obs_space = _as_obs_space(space)
        value = None if obs_space.value is None else self._zeros_box(obs_space.value)
        return TorchObs(policy=self._zeros_box(obs_space.policy), value=value)

    def _assign_obs(self, dst: TorchObs, mask: torch.Tensor, src: TorchObs | torch.Tensor):
        src = _as_obs(src)
        dst.policy[mask] = src.policy
        if dst.value is not None:
            assert src.value is not None
            dst.value[mask] = src.value
        else:
            assert src.value is None

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        state = self._state
        done = state.done
        assert done.shape == (self._num_envs,)
        if not torch.any(done):
            return

        state.episode_steps[done] = 0
        data = state.data[done.detach().cpu().numpy()]
        obs = self.reset(data)
        self._assign_obs(state.obs, done, obs)

    def _max_episode_steps(self) -> int | None:
        return self._cfg.max_episode_steps

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        assert self._state is not None
        max_episode_steps = self._max_episode_steps()
        if not max_episode_steps:
            return
        self._state.truncated = self._state.episode_steps >= max_episode_steps

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> TorchEnvState:
        """
        Apply the action to the environment

        Args:
            actions (torch.Tensor): The actions to apply
            state (TorchEnvState): The environment state to apply the actions.
        """

    @abc.abstractmethod
    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        """
        Update the environment state after physics step

        Args:
            state (TorchEnvState): The environment state to update
        """

    @abc.abstractmethod
    def reset(
        self,
        data: mtx.SceneData,
    ) -> TorchObs | torch.Tensor:
        """
        Reset the environment for the done envs

        Args:
            data (mtx.SceneData): The scene data to reset

        Returns:
            TorchObs | torch.Tensor: The initial observations after reset
        """
        pass

    def physics_step(self):
        for _ in range(self._cfg.sim_substeps):
            self._model.step(self._state.data)

    def _prev_physics_step(self):
        state = self._state
        assert state is not None
        state.reward.zero_()
        state.terminated.zero_()
        state.truncated.zero_()

    @perf_root("step")
    def step(self, actions: torch.Tensor) -> TorchEnvState:
        """Advance one control step.

        Timing: ``apply_action`` -> ``physics`` -> ``transition`` (state update)
        -> ``reset(env_ids)`` for done rows, mirroring ArrayEnv's scope names so
        the console panel renders the same env_step sub-stage tree.
        """
        if self._state is None:
            self.init_state()
        assert self._state is not None

        self._prev_physics_step()
        with self.perf.scope("apply_action"):
            self._state = self.apply_action(actions, self._state)
            assert self._state is not None, "apply_action must return a valid TorchEnvState"
        with self.perf.scope("physics"):
            self.physics_step()
        with self.perf.scope("transition"):
            self._state = self.update_state(self._state)
        self._state = self._state.replace(obs=_as_obs(self._state.obs))
        self._state.episode_steps += 1
        self._update_truncate()
        with self.perf.scope("reset"):
            self._reset_done_envs()
        return self._state
