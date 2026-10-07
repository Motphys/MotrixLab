# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile Go2 walking environments into their deployment artifact contract."""

import numpy as np
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_deploy.artifact import ControlSpec, TaskSpec
from motrix_deploy.profile import DeploymentProfile, register_profile_compiler
from motrix_env_core import registry
from motrix_envs.deploy.robot import build_robot_model, build_robot_spec, read_position_servo_gains
from motrix_envs.locomotion.quadruped.cfg import QuadrupedWalkEnvCfg

GO2_TASK_NAME = "go2_walk/v1"
GO2_COMMAND_SCALES = {
    "go2-walk-flat": np.ones(3, dtype=np.float32),
    "go2-walk-rough": np.full(3, 0.5, dtype=np.float32),
}


@register_profile_compiler("go2-walk-flat")
@register_profile_compiler("go2-walk-rough")
def build_go2_walk_profile(env_name: str) -> DeploymentProfile:
    """Compile one registered Go2 walking environment into a deployment profile."""
    env = registry.make(env_name, num_envs=1, mode="train")
    env_cfg = env.cfg
    if not isinstance(env_cfg, QuadrupedWalkEnvCfg):
        raise TypeError(f"Expected QuadrupedWalkEnvCfg, got {type(env_cfg).__name__}")
    robot_cfg = env_cfg.scene.objs.robot
    if not isinstance(robot_cfg, UnitreeGo2Robot):
        raise TypeError(f"Expected UnitreeGo2Robot, got {type(robot_cfg).__name__}")
    if env_cfg.control_config.simulate_action_latency:
        raise ValueError("Go2 deployment v1 requires control_config.simulate_action_latency=false")

    robot = build_robot_spec(
        robot_cfg,
        key_pose_name=env_cfg.key_pose_name,
    )
    kp, kd = read_position_servo_gains(build_robot_model(robot_cfg), robot.joint_names)
    action_shape = env.action_space.shape
    observation_shape = env.policy_observation_space.shape
    if len(action_shape) != 1 or len(observation_shape) != 1:
        raise ValueError("Go2 deployment requires flat policy observation and action spaces")
    action_size = action_shape[0]
    observation_size = observation_shape[0]
    if action_size != robot.joint_count:
        raise ValueError("Go2 deployment requires every model actuator to target one canonical robot joint")
    action_scale = env_cfg.control_config.action_scale
    raw_lower = env.action_space.low
    raw_upper = env.action_space.high
    command_cfg = env_cfg.commands.velocity
    feet_offsets = _feet_phase_offsets(env_cfg)
    task = TaskSpec(
        name=GO2_TASK_NAME,
        observation_size=observation_size,
        action_size=action_size,
        config={
            "action_scale": action_scale,
            "command_lower": command_cfg.lower.tolist(),
            "command_upper": command_cfg.upper.tolist(),
            "command_scale": GO2_COMMAND_SCALES[env_name].tolist(),
            "feet_phase_offsets": feet_offsets.tolist(),
            "gait_frequency_hz": env_cfg.gait_frequency,
            "standing_threshold": env_cfg.commands.velocity.standing_threshold,
            "kp": kp.tolist(),
            "kd": kd.tolist(),
            "raw_clip": [raw_lower.tolist(), raw_upper.tolist()],
        },
    )
    return DeploymentProfile(
        robot=robot,
        task=task,
        control=ControlSpec(period_s=env_cfg.ctrl_dt, state_timeout_s=0.1),
    )


def _feet_phase_offsets(env_cfg: QuadrupedWalkEnvCfg) -> np.ndarray:
    offsets = np.zeros(4, dtype=np.float32)
    seen: set[int] = set()
    for pair_index, pair in enumerate(env_cfg.trot_pairs):
        for foot_index in pair:
            if foot_index in seen or not 0 <= foot_index < 4:
                raise ValueError(f"Go2 trot_pairs must cover each foot exactly once, got {env_cfg.trot_pairs}")
            offsets[foot_index] = np.float32(0.5 * pair_index)
            seen.add(foot_index)
    if seen != set(range(4)):
        raise ValueError(f"Go2 trot_pairs must cover feet [0, 1, 2, 3], got {env_cfg.trot_pairs}")
    return offsets


__all__ = ["build_go2_walk_profile"]
