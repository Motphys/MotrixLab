# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment profile compilation for built-in environments."""

from dataclasses import dataclass

import numpy as np
import pytest
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_deploy_tasks.tasks.go2_walk import Go2WalkTaskSpec
from motrix_env_core import registry
from motrix_envs.deploy import build_deployment_profile, registered_profile_compilers
from motrix_envs.deploy.robot import (
    build_robot_model,
    build_robot_spec,
    resolve_joint_actuators_in_canonical_order,
)


@dataclass
class _ActuatorStub:
    target_name: str
    target_type: str = "joint"


@pytest.mark.parametrize(
    ("env_name", "command_scale"),
    [("go2-walk-flat", [1.0, 1.0, 1.0]), ("go2-walk-rough", [0.5, 0.5, 0.5])],
)
def test_go2_walk_environments_compile_shared_deployment_profiles(
    env_name: str,
    command_scale: list[float],
) -> None:
    profile = build_deployment_profile(env_name)
    env_cfg = registry.make_env_config(env_name)

    assert {"go2-walk-flat", "go2-walk-rough"} <= set(registered_profile_compilers())
    assert isinstance(profile.task, Go2WalkTaskSpec)
    assert profile.task.name == "go2_walk/v1"
    env = registry.make(env_name, num_envs=1, mode="train")
    assert profile.observation_size == env.policy_observation_space.shape[0]
    assert profile.action_size == env.action_space.shape[0]
    assert profile.control.period_s == pytest.approx(env_cfg.ctrl_dt)
    assert profile.task.command_lower == env_cfg.commands.velocity.lower.tolist()
    assert profile.task.command_upper == env_cfg.commands.velocity.upper.tolist()
    # Command scale is an artifact/runtime input contract and intentionally differs by deployment environment.
    assert profile.task.command_scale == command_scale


def test_robot_spec_is_built_without_an_environment_config() -> None:
    robot_cfg = UnitreeGo2Robot()
    model = build_robot_model(robot_cfg)
    spec = build_robot_spec(robot_cfg, key_pose_name="default", model=model)

    assert spec.base_link_name == "base"
    assert spec.joint_names[:3] == ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint")
    assert spec.joint_count == 12
    actuators = resolve_joint_actuators_in_canonical_order(model.actuators, spec.joint_names)
    np.testing.assert_array_equal(spec.torque_limit, [actuator.force_range[1] for actuator in actuators])


def test_joint_actuator_resolution_is_canonical_and_strict() -> None:
    joint_b_actuator = _ActuatorStub("joint_b")
    joint_a_actuator = _ActuatorStub("joint_a")
    site_actuator = _ActuatorStub("site", target_type="site")

    assert resolve_joint_actuators_in_canonical_order(
        [joint_b_actuator, site_actuator, joint_a_actuator],
        ["joint_a", "joint_b"],
    ) == [joint_a_actuator, joint_b_actuator]

    with pytest.raises(ValueError, match="Multiple actuators"):
        resolve_joint_actuators_in_canonical_order(
            [_ActuatorStub("joint_a"), _ActuatorStub("joint_a")],
            ["joint_a"],
        )
    with pytest.raises(ValueError, match="No actuator targets canonical joints"):
        resolve_joint_actuators_in_canonical_order([_ActuatorStub("joint_a")], ["joint_a", "joint_b"])
    with pytest.raises(ValueError, match="Canonical joint names must be unique"):
        resolve_joint_actuators_in_canonical_order([_ActuatorStub("joint_a")], ["joint_a", "joint_a"])
