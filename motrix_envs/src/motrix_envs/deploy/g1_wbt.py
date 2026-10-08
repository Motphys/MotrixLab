# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compile G1 WBT config and one motion clip without constructing an environment."""

import io as io_module

import numpy as np

from motrix_deploy.artifact.schema import ControlSpec
from motrix_deploy.profile import DeploymentProfile, register_profile_compiler
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtMotion, G1WbtTaskSpec
from motrix_env_core import registry
from motrix_envs.deploy.robot import build_robot_model, build_robot_spec, read_position_servo_gains
from motrix_envs.locomotion.wbt.g1.common import G1WbtEnvCfg
from motrix_envs.motion.library import expand_motion_paths
from motrix_envs.motion.loader import MotrixMotion


@register_profile_compiler("g1-wbt-dance")
def build_g1_wbt_profile(env_name: str) -> DeploymentProfile:
    """Embed actor reference channels and hardware-observable safety thresholds."""
    cfg = registry.make_env_config(env_name)
    if not isinstance(cfg, G1WbtEnvCfg):
        raise TypeError(f"Expected G1WbtEnvCfg, got {type(cfg).__name__}")
    motion_cfg = cfg.commands.motion
    paths = expand_motion_paths(motion_cfg.motion_files)
    if len(paths) != 1:
        raise ValueError("G1 WBT deployment v1 requires exactly one motion clip")
    motion = MotrixMotion(paths[0])
    if not np.isclose(motion.fps * cfg.ctrl_dt, 1):
        raise ValueError("Motion fps must match the deployment control period")
    robot_cfg = cfg.scene.objs.robot
    model = build_robot_model(robot_cfg)
    robot = build_robot_spec(robot_cfg, key_pose_name="default", model=model)
    kp, kd = read_position_servo_gains(model, robot.joint_names)
    action = cfg.actions.joint_position
    if not action.action_scales_by_effort_limit_over_p_gain:
        raise ValueError("G1 WBT deployment requires effort/kp action scaling")
    scale = np.zeros(robot.joint_count, dtype=np.float32)
    np.divide(robot.torque_limit, kp, out=scale, where=kp != 0)
    scale *= action.action_scale
    joint_indices = motion.joint_indices(list(robot.joint_names))
    reference_index = motion.body_index(motion_cfg.reference_body_name)
    term = cfg.terminations
    # Standing-hold servo gains for physical preparation, in robot.joint_names order
    # (12 legs, 3 waist, 14 arms). Training gains are dynamic whole-body tracking
    # gains (legs 28-99) and cannot hold a static standing pose; these firmer gains
    # are empirically validated to reach preparation readiness with unchanged safety
    # gates (tilt/error/velocity/timeout) and complete the full deployment sequence.
    preparation_kp = ([350.0, 200.0, 200.0, 300.0, 300.0, 150.0] * 2) + [200.0] * 3 + [40.0] * 14
    preparation_kd = ([5.0, 5.0, 5.0, 10.0, 5.0, 5.0] * 2) + [5.0] * 3 + [3.0] * 14
    joint_pos = motion.joint_pos[:, joint_indices]
    joint_vel = motion.joint_vel[:, joint_indices]
    reference_quaternion = motion.body_quat_w[:, reference_index]
    clip = G1WbtMotion(joint_pos, joint_vel, reference_quaternion)
    # The validated motion clip travels as an artifact NPZ payload; the manifest
    # spec keeps only the payload path and the frame count. The task reloads and
    # revalidates the clip at construction.
    buffer = io_module.BytesIO()
    np.savez(
        buffer,
        joint_pos=clip.joint_pos,
        joint_vel=clip.joint_vel,
        reference_quaternion_xyzw=clip.reference_quaternion_xyzw,
    )
    motion_payload_path = "payloads/motion.npz"
    task = G1WbtTaskSpec(
        period_s=cfg.ctrl_dt,
        action_scale=scale.tolist(),
        kp=kp.tolist(),
        kd=kd.tolist(),
        preparation_kp=preparation_kp,
        preparation_kd=preparation_kd,
        motion_frames=len(clip),
        reference_body_name=motion_cfg.reference_body_name,
        termination_ref_orientation_threshold=term.bad_ref_ori.threshold,
        termination_joint_position_threshold=term.bad_dof_pos.threshold,
        termination_joint_velocity_threshold=term.bad_dof_vel.threshold,
    )
    return DeploymentProfile(
        robot=robot,
        task=task,
        control=ControlSpec(period_s=cfg.ctrl_dt, state_timeout_s=0.1),
        observation_size=154,
        action_size=robot.joint_count,
        payloads={motion_payload_path: buffer.getvalue()},
    )
