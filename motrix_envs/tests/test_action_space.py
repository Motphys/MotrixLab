# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motrix_env_core import registry
from motrix_env_core.mdp.action_space import joint_position_action_space
from motrix_env_core.sim.model import ActuatorType


def test_joint_position_action_space_uses_symmetric_position_ranges():
    actuators = [
        SimpleNamespace(name="motor_b", actuator_type=ActuatorType.POSITION, ctrl_range=[-1.0, 3.0]),
        SimpleNamespace(name="motor_a", actuator_type=ActuatorType.POSITION, ctrl_range=[-0.5, 0.75]),
    ]
    space = joint_position_action_space(
        actuators,
        default_angles=torch.tensor([-0.25, 0.5], dtype=torch.float32),
        action_scales=torch.tensor([0.25, 0.5], dtype=torch.float32),
    )

    np.testing.assert_allclose(space.low, [-13.0, -2.0])
    np.testing.assert_allclose(space.high, [13.0, 2.0])


def test_joint_position_action_space_subset_indices_keep_their_own_bounds():
    actuators = [
        SimpleNamespace(name="motor_0", actuator_type=ActuatorType.POSITION, ctrl_range=[-1.0, 1.0]),
        SimpleNamespace(name="motor_1", actuator_type=ActuatorType.POSITION, ctrl_range=[-2.0, 2.0]),
        SimpleNamespace(name="motor_2", actuator_type=ActuatorType.POSITION, ctrl_range=[-4.0, 4.0]),
    ]
    # A subset in explicit order: the space must keep motor_2's wider bound
    # first and motor_0's narrower bound second, without re-indexing rows.
    space = joint_position_action_space(
        actuators,
        default_angles=torch.zeros(2, dtype=torch.float32),
        action_scales=1.0,
        actuator_indices=torch.tensor([2, 0], dtype=torch.int64),
    )

    np.testing.assert_allclose(space.low, [-4.0, -1.0])
    np.testing.assert_allclose(space.high, [4.0, 1.0])


@pytest.mark.parametrize(
    "env_name",
    [
        "anymalc-walk-flat",
        "anymalc-walk-rough",
        "go1-walk-flat",
        "go1-walk-rough",
        "go2-walk-flat",
        "go2-walk-rough",
    ],
)
def test_quadruped_action_space_maps_to_actuator_control_limits(env_name):
    env = registry.make(env_name, num_envs=2, mode="train")
    control_ranges = np.asarray([spec.ctrl_range for spec in env.model.actuators], dtype=np.float32)
    action_scale = env.cfg.control_config.action_scale

    env.initialize()

    state = env.state
    env.apply_action(torch.zeros((2, env._num_action), dtype=torch.float32), state)
    env.sim_data.execute()
    torch.testing.assert_close(
        env.sim_data["actuator_ctrls"], env.default_angles.expand(2, env._num_action), rtol=1e-7, atol=0
    )

    action_low = torch.as_tensor(env.action_space.low)
    action_high = torch.as_tensor(env.action_space.high)
    target_low = env.default_angles + action_scale * action_low
    target_high = env.default_angles + action_scale * action_high
    torch.testing.assert_close(target_low, torch.as_tensor(control_ranges[:, 0]), rtol=1e-7, atol=0)
    torch.testing.assert_close(target_high, torch.as_tensor(control_ranges[:, 1]), rtol=1e-7, atol=0)
