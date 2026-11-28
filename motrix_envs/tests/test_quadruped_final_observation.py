# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure task-input tests: no simulator is needed to protect snapshot semantics."""

import pytest
import torch

from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_envs.locomotion.quadruped.go2 import Go2WalkDirectEnvCfg
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


class ObservationTask:
    _build_observation = QuadrupedWalkTask._build_observation
    _capture_final_obs = QuadrupedWalkTask._capture_final_obs
    compute_observation = QuadrupedWalkTask.compute_observation
    _obs_noise = QuadrupedWalkTask._obs_noise
    _policy_obs_dim = QuadrupedWalkTask._policy_obs_dim
    _value_obs_dim = QuadrupedWalkTask._value_obs_dim
    get_dof_pos = QuadrupedWalkTask.get_dof_pos
    get_dof_vel = QuadrupedWalkTask.get_dof_vel
    get_local_linvel = QuadrupedWalkTask.get_local_linvel
    get_gyro = QuadrupedWalkTask.get_gyro
    get_gravity = QuadrupedWalkTask.get_gravity

    def __init__(self, noise: bool) -> None:
        self.cfg = Go2WalkDirectEnvCfg()
        self.cfg.noise_config.level = 1.0 if noise else 0.0
        self._num_action = 2
        self._num_feet = 4
        self.default_angles = torch.zeros(2)
        self.sim_data = {
            "joint_dof_pos": torch.arange(6, dtype=torch.float32).reshape(3, 2),
            "joint_dof_vel": torch.arange(6, dtype=torch.float32).reshape(3, 2) + 10,
            "local_linvel": torch.arange(9, dtype=torch.float32).reshape(3, 3) + 20,
            "gyro": torch.arange(9, dtype=torch.float32).reshape(3, 3) + 30,
            "upvector": torch.arange(9, dtype=torch.float32).reshape(3, 3) + 40,
        }
        self._commands = torch.arange(9, dtype=torch.float32).reshape(3, 3) + 50
        self._current_actions = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 60
        self._feet_phase = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 70
        self._phase = torch.arange(3, dtype=torch.float32)


def empty_state(task: ObservationTask) -> ArrayEnvState:
    return ArrayEnvState(
        EnvObs(
            {
                "policy": torch.full((3, task._policy_obs_dim()), -100.0),
                "value": torch.full((3, task._value_obs_dim()), -200.0),
            }
        ),
        torch.zeros(3),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.long),
    )


@pytest.mark.parametrize("noise", [False, True])
def test_final_observation_is_compact_owned_and_does_not_publish(noise: bool) -> None:
    task = ObservationTask(noise)
    state = empty_state(task)
    before = {name: rows.clone() for name, rows in state.obs.groups.items()}
    inputs = {name: rows.clone() for name, rows in task.sim_data.items()}
    commands, phase, feet_phase = task._commands.clone(), task._phase.clone(), task._feet_phase.clone()
    ids = torch.tensor([2, 0])
    initial_rng = torch.get_rng_state()
    final = task._capture_final_obs(ids)
    final_rng = torch.get_rng_state()
    for name, rows in state.obs.groups.items():
        torch.testing.assert_close(rows, before[name])
    # Replaying the same default-stream samples must reproduce a normal
    # observation at the identical pre-reset state; capture itself advances it.
    torch.set_rng_state(initial_rng)
    task.compute_observation(state)
    assert torch.equal(torch.get_rng_state(), final_rng)
    if noise:
        assert not torch.equal(initial_rng, final_rng)
    for name, rows in final.obs.groups.items():
        torch.testing.assert_close(rows, state.obs.groups[name][ids])
    assert final.obs.policy.shape == (2, task._policy_obs_dim())
    assert final.obs.value.shape == (2, task._value_obs_dim())
    if not noise:
        torch.testing.assert_close(final.obs.value[:, -3:], inputs["local_linvel"][ids])
    for name, rows in task.sim_data.items():
        torch.testing.assert_close(rows, inputs[name])
    torch.testing.assert_close(task._commands, commands)
    torch.testing.assert_close(task._phase, phase)
    torch.testing.assert_close(task._feet_phase, feet_phase)
    saved = {name: rows.clone() for name, rows in final.obs.groups.items()}
    ids.zero_()
    for rows in task.sim_data.values():
        rows.zero_()
    task._commands.zero_()
    task._current_actions.zero_()
    task.compute_observation(state)
    assert final.env_ids.tolist() == [2, 0]
    for name, rows in final.obs.groups.items():
        torch.testing.assert_close(rows, saved[name])
