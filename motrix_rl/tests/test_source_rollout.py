# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment NumPy policies exchange tensors with source environments."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_rl.deploy import source_rollout


def test_source_rollout_converts_commands_and_readonly_policy_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    command = np.asarray([0.5, -0.2, 0.1], dtype=np.float32)
    policy_action = np.asarray([0.25, -0.5], dtype=np.float32)
    policy_action.setflags(write=False)

    class SourceEnv:
        def __init__(self) -> None:
            self.cfg = SimpleNamespace(noise_config=SimpleNamespace(level=1.0), spawn_xy_range=1.0)
            self._commands = torch.zeros((1, 3))
            self.actions: list[torch.Tensor] = []
            self.state = ArrayEnvState(
                obs=EnvObs({"policy": torch.zeros((1, 4))}),
                reward=torch.zeros(1),
                terminated=torch.zeros(1, dtype=torch.bool),
                truncated=torch.zeros(1, dtype=torch.bool),
                episode_steps=torch.zeros(1, dtype=torch.long),
            )

        def initialize(self) -> ArrayEnvState:
            return self.state

        def step(self, actions: torch.Tensor) -> ArrayEnvState:
            torch.testing.assert_close(self._commands[0], torch.from_numpy(command))
            torch.testing.assert_close(actions, torch.from_numpy(policy_action.copy()).unsqueeze(0))
            self.actions.append(actions.clone())
            # Source tasks may mutate their action state and resample commands.
            actions.zero_()
            self._commands.zero_()
            self.state.obs.policy.add_(1.0)
            return self.state

    class Policy:
        def infer(self, observation: np.ndarray) -> np.ndarray:
            assert isinstance(observation, np.ndarray)
            assert observation.dtype == np.float32
            assert observation.shape == (4,)
            return policy_action

    env = SourceEnv()
    artifact = SimpleNamespace(
        policy_path=Path("policy.onnx"),
        manifest=SimpleNamespace(policy=SimpleNamespace(input=None, output=None)),
    )
    monkeypatch.setattr(source_rollout, "read_artifact", lambda path: artifact)
    monkeypatch.setattr(source_rollout, "OnnxPolicy", lambda *args: Policy())
    monkeypatch.setattr(source_rollout.registry, "make", lambda *args, **kwargs: env)

    result = source_rollout.validate_motrixsim_source_rollout(
        "artifact", env_name="source-walk", steps=2, seed=0, command=command
    )

    assert result.success
    assert result.completed_steps == 2
    assert len(env.actions) == 2
    assert result.reset_observation == [0.0] * 4
    assert result.first_policy_outputs == [[0.25, -0.5]] * 2
    np.testing.assert_array_equal(policy_action, [0.25, -0.5])
