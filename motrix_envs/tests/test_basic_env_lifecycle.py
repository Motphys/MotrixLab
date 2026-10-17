# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core import registry
from motrix_env_core.array.env import NpObs


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize(
    "env_name",
    ["point_mass", "stewart-static", "stewart", "stewart-disturb-xy", "franka-open-cabinet"],
)
def test_basic_environment_step_preserves_numpy_observation_contract(env_name: str, num_envs: int) -> None:
    env = registry.make(env_name, num_envs=num_envs)
    action = np.zeros((num_envs, *env.action_space.shape), dtype=env.action_space.dtype)

    state = None
    for _ in range(10):
        state = env.step(action)

    assert state is not None
    assert isinstance(state.obs, NpObs)
    assert state.obs.policy.shape == (num_envs, *env.observation_space.shape)
    if env_name == "franka-open-cabinet":
        # Episode-scoped gripper state lives on the env instance, not in info.
        assert env._current_gripper_action.shape == (num_envs,)
