# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.config.scene import SceneCfg
from motrix_env_motrixsim.torch_env import TorchEnv, TorchEnvState, TorchObs


def test_registry_infers_torch_frontend_from_runtime_base() -> None:
    class CustomEnvironment(TorchEnv):
        pass

    assert registry._infer_sim_backend(CustomEnvironment) == "torch"


def test_torch_environment_state_matches_numpy_state_contract() -> None:
    policy = torch.zeros((2, 3))
    value = torch.ones((2, 4))
    state = TorchEnvState(
        data=SimpleNamespace(shape=(2,)),
        obs=TorchObs(policy=policy, value=value),
        reward=torch.zeros(2),
        terminated=torch.tensor([False, True]),
        truncated=torch.tensor([True, False]),
        episode_steps=torch.zeros(2, dtype=torch.int64),
    )

    state.validate()
    torch.testing.assert_close(state.obs.value_or_policy, value)
    torch.testing.assert_close(state.done, torch.tensor([True, True]))

    replacement = state.replace(reward=torch.ones(2))
    torch.testing.assert_close(replacement.reward, torch.ones(2))
    torch.testing.assert_close(state.reward, torch.zeros(2))


class _LifecycleTorchEnv(TorchEnv[EnvCfg]):
    def __init__(self, cfg: EnvCfg, num_envs: int = 1, device: torch.device | str = "cpu"):
        super().__init__(cfg, num_envs, device)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._actions = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device)
        self.physics_steps = 0

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def physics_step(self) -> None:
        self.physics_steps += 1

    def apply_action(self, actions: torch.Tensor, state: TorchEnvState) -> TorchEnvState:
        self._actions = actions
        return state

    def update_state(self, state: TorchEnvState) -> TorchEnvState:
        return state.replace(
            obs=self._actions.to(dtype=state.obs.policy.dtype),
            reward=torch.abs(self._actions[:, 0]).to(dtype=torch.float32),
            terminated=self._actions[:, 0] > 0.5,
        )

    def reset(self, data) -> torch.Tensor:
        return torch.full((*data.shape, 1), -1.0, dtype=torch.float64, device=self.device)


def test_torch_environment_rejects_gpu_until_gpu_simulation_is_available() -> None:
    with pytest.raises(ValueError, match="supports CPU simulation only"):
        _LifecycleTorchEnv(EnvCfg(scene=SceneCfg()), device="cuda")


def test_np_simulation_places_torch_environment_lifecycle_on_cpu() -> None:
    cfg = EnvCfg(
        scene=SceneCfg(),
        ctrl_dt=0.01,
        max_episode_seconds=0.02,
    )
    env = _LifecycleTorchEnv(cfg, num_envs=2)
    device = env.device

    initial = env.init_state()
    assert device == torch.device("cpu")
    assert initial.obs.policy.dtype == torch.float64
    assert initial.obs.policy.device == device
    assert initial.reward_terms == {}

    terminated = env.step(torch.tensor([[1.0], [0.25]], dtype=torch.float32, device=device))
    torch.testing.assert_close(terminated.reward, torch.tensor([1.0, 0.25], device=device))
    torch.testing.assert_close(terminated.terminated, torch.tensor([True, False], device=device))
    torch.testing.assert_close(terminated.truncated, torch.tensor([False, False], device=device))
    torch.testing.assert_close(terminated.episode_steps, torch.tensor([0, 1], device=device))
    torch.testing.assert_close(
        terminated.obs.policy[:, 0],
        torch.tensor([-1.0, 0.25], dtype=torch.float64, device=device),
    )

    truncated = env.step(torch.tensor([[0.2], [0.2]], dtype=torch.float32, device=device))
    torch.testing.assert_close(truncated.terminated, torch.tensor([False, False], device=device))
    torch.testing.assert_close(truncated.truncated, torch.tensor([False, True], device=device))
    torch.testing.assert_close(truncated.episode_steps, torch.tensor([1, 0], device=device))
    torch.testing.assert_close(
        truncated.obs.policy[:, 0],
        torch.tensor([0.2, -1.0], dtype=torch.float64, device=device),
    )
    assert env.physics_steps == 2
