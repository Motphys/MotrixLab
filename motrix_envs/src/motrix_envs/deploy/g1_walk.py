# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile the G1 flat-ground walking environment into its deploy contract."""

import math

import numpy as np
from motrix_robots.unitree import UnitreeG129Dof

from motrix_deploy.artifact import ControlSpec
from motrix_deploy.profile import DeploymentProfile
from motrix_deploy_tasks.tasks.g1_walk import G1WalkTaskSpec
from motrix_env_core import registry
from motrix_env_core.mdp.action import JointPositionActionTerm
from motrix_envs.deploy.robot import build_robot_model, build_robot_spec, read_position_servo_gains
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg

_ACTION_TERM_NAME = "joint_position"
_COMMAND_TERM_NAME = "walk"

# ``WalkCommand._lane_phase`` uses this threshold to pin both phase lanes
# while the planar velocity command is effectively zero. Keep it explicit in
# the exported contract rather than making the deployment task guess it.
_STANDING_COMMAND_THRESHOLD = 0.01

# Deployment uses one deterministic pair of antiphase clocks. Training adds a
# random common phase at reset, but that reset-only augmentation is not an
# operator/runtime input and therefore is not exported.
_GAIT_PHASE_OFFSETS_RAD = (0.0, math.pi)


def build_g1_walk_profile(env_name: str) -> DeploymentProfile:
    """Compile the registered G1 walk task into canonical artifact data.

    The environment is constructed once so the compiler reads the exact
    manager-generated action space, observation size, actuator order, and
    per-joint action scales used by training.  Values written to the artifact
    are always reordered/validated against ``RobotSpec.joint_names``.

    ``G1WalkTaskSpec`` is deliberately implemented in the deployment package,
    not here.  Until that schema declares the fields assembled below this
    function fails before emitting an incomplete artifact.
    """

    if env_name != "g1-walk-flat":
        raise ValueError(f"G1 walk profile supports only 'g1-walk-flat', got {env_name!r}")

    env = registry.make(env_name, num_envs=1, mode="train")
    cfg = env.cfg
    if not isinstance(cfg, HumanoidVelocityTrackingManagerEnvCfg):
        raise TypeError(f"Expected HumanoidVelocityTrackingManagerEnvCfg, got {type(cfg).__name__}")

    robot_cfg = cfg.scene.objs.robot
    if not isinstance(robot_cfg, UnitreeG129Dof):
        raise TypeError(f"Expected UnitreeG129Dof, got {type(robot_cfg).__name__}")

    model = build_robot_model(robot_cfg)
    robot = build_robot_spec(robot_cfg, key_pose_name="default", model=model)
    if robot.joint_count != 29:
        raise ValueError(f"G1 29-DOF deployment requires 29 canonical joints, got {robot.joint_count}")
    kp, kd = read_position_servo_gains(model, robot.joint_names)

    action_term = _single_joint_position_action_term(env)
    action_joint_names = tuple(actuator.target_name for actuator in env.model.actuators)
    if action_joint_names != robot.joint_names:
        raise ValueError(
            "G1 walk policy actuator targets must exactly match canonical robot joint order: "
            f"policy={action_joint_names}, robot={robot.joint_names}"
        )

    action_shape = env.action_space.shape
    observation_shape = env.policy_observation_space.shape
    expected_action_shape = (robot.joint_count,)
    if action_shape != expected_action_shape:
        raise ValueError(f"G1 walk action space must have shape {expected_action_shape}, got {action_shape}")
    if len(observation_shape) != 1 or observation_shape[0] <= 0:
        raise ValueError(f"G1 walk policy observation space must be one non-empty vector, got {observation_shape}")

    action_state = action_term.state
    _assert_vector("action default angles", action_state.default_angles, robot.joint_count)
    _assert_vector("action scales", action_state.action_scales, robot.joint_count, non_negative=True)
    _assert_vector("action lower bounds", env.action_space.low, robot.joint_count)
    _assert_vector("action upper bounds", env.action_space.high, robot.joint_count)
    if not np.allclose(action_state.default_angles, robot.default_joint_position, atol=1e-6, rtol=0.0):
        raise ValueError("G1 walk action default angles do not match RobotSpec.default_joint_position")
    if not np.allclose(action_state.joint_lower, robot.position_lower, atol=1e-6, rtol=0.0):
        raise ValueError("G1 walk action lower joint limits do not match RobotSpec.position_lower")
    if not np.allclose(action_state.joint_upper, robot.position_upper, atol=1e-6, rtol=0.0):
        raise ValueError("G1 walk action upper joint limits do not match RobotSpec.position_upper")

    command_cfg = cfg.commands.walk
    command_bounds = np.asarray(command_cfg.vel_limit, dtype=np.float32)
    if command_bounds.shape != (2, 3) or not np.all(np.isfinite(command_bounds)):
        raise ValueError(f"G1 walk velocity limits must contain finite lower/upper (vx, vy, wz), got {command_bounds}")
    if np.any(command_bounds[0] > 0.0) or np.any(command_bounds[1] < 0.0):
        raise ValueError("G1 walk velocity command range must contain zero on every axis")
    if np.any(command_bounds[0] > command_bounds[1]):
        raise ValueError("G1 walk velocity command lower bounds must not exceed upper bounds")
    if not np.isfinite(command_cfg.gait_period) or command_cfg.gait_period <= 0.0:
        raise ValueError(f"G1 walk gait period must be positive and finite, got {command_cfg.gait_period}")

    policy_obs = cfg.observations.policy
    if policy_obs.command.command_name != _COMMAND_TERM_NAME:
        raise ValueError(
            f"G1 walk policy command observation must read {_COMMAND_TERM_NAME!r}, "
            f"got {policy_obs.command.command_name!r}"
        )
    if policy_obs.actions.action_name != _ACTION_TERM_NAME:
        raise ValueError(
            f"G1 walk previous-action observation must read {_ACTION_TERM_NAME!r}, "
            f"got {policy_obs.actions.action_name!r}"
        )
    if (policy_obs.sin_phase.offset, policy_obs.sin_phase.size) != (0, 2):
        raise ValueError("G1 walk sine-phase observation must select offset=0, size=2")
    if (policy_obs.cos_phase.offset, policy_obs.cos_phase.size) != (2, 2):
        raise ValueError("G1 walk cosine-phase observation must select offset=2, size=2")

    task_values = {
        "action_scale": np.asarray(action_state.action_scales, dtype=np.float32).tolist(),
        "kp": kp.tolist(),
        "kd": kd.tolist(),
        "action_lower": np.asarray(env.action_space.low, dtype=np.float32).tolist(),
        "action_upper": np.asarray(env.action_space.high, dtype=np.float32).tolist(),
        "command_lower": command_bounds[0].tolist(),
        "command_upper": command_bounds[1].tolist(),
        "base_angular_velocity_scale": float(policy_obs.base_ang_vel.scale),
        "joint_position_scale": float(policy_obs.joint_pos.scale),
        "joint_velocity_scale": float(policy_obs.joint_vel.scale),
        "gait_period_s": float(command_cfg.gait_period),
        "gait_phase_offsets_rad": list(_GAIT_PHASE_OFFSETS_RAD),
        "standing_command_threshold": _STANDING_COMMAND_THRESHOLD,
    }
    _require_task_schema_fields(task_values)
    task = G1WalkTaskSpec(**task_values)

    return DeploymentProfile(
        robot=robot,
        task=task,
        control=ControlSpec(period_s=cfg.ctrl_dt, state_timeout_s=0.1),
        observation_size=observation_shape[0],
        action_size=action_shape[0],
    )


def _single_joint_position_action_term(env: object) -> JointPositionActionTerm:
    action_terms = getattr(env, "action_terms", None)
    if not isinstance(action_terms, dict) or set(action_terms) != {_ACTION_TERM_NAME}:
        names = tuple(action_terms) if isinstance(action_terms, dict) else action_terms
        raise ValueError(f"G1 walk deployment requires exactly one {_ACTION_TERM_NAME!r} action term, got {names}")
    action_term = action_terms[_ACTION_TERM_NAME]
    if not isinstance(action_term, JointPositionActionTerm):
        raise TypeError(f"Expected JointPositionActionTerm, got {type(action_term).__name__}")
    return action_term


def _assert_vector(name: str, value: object, size: int, *, non_negative: bool = False) -> None:
    vector = np.asarray(value)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values, got shape {vector.shape}")
    if non_negative and np.any(vector < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")


def _require_task_schema_fields(values: dict[str, object]) -> None:
    """Stop before artifact creation while the reviewed TaskSpec is a shell."""

    available = set(G1WalkTaskSpec.model_fields)
    missing = set(values).difference(available)
    if missing:
        raise NotImplementedError(
            "G1WalkTaskSpec must declare the profile fields before export can continue: " + ", ".join(sorted(missing))
        )


__all__ = ["build_g1_walk_profile"]
