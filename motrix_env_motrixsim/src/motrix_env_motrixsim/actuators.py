# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import motrixsim as mtx

from motrix_env_core.config.scene.actuator import ActuatorCfg, MotorActuatorCfg, PositionActuatorCfg
from motrix_env_motrixsim.urdf import _iter_links


def _build_actuator(name: str, cfg: ActuatorCfg) -> Any:
    actuator = mtx.msd.Actuator()
    actuator.name = name
    actuator.target = mtx.msd.ActuatorTarget.joint(cfg.joint_name)
    actuator.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if isinstance(cfg, PositionActuatorCfg):
        param = mtx.msd.PositionParam()
        param.kp = cfg.kp
        param.damping_value = cfg.kv
        param.damping_type = mtx.msd.DampingType.Kv
        actuator.actuator_type = mtx.msd.ActuatorType.position(param)
    elif isinstance(cfg, MotorActuatorCfg):
        actuator.actuator_type = mtx.msd.ActuatorType.motor()
    else:
        raise TypeError(f"MotrixSim does not support actuator config {type(cfg).__name__}")
    if cfg.ctrl_range is not None:
        actuator.ctrlrange = mtx.msd.Range(*cfg.ctrl_range)
    if cfg.force_range is not None:
        actuator.forcerange = mtx.msd.Range(*cfg.force_range)
    return actuator


def apply_actuator_cfgs(world: mtx.msd.World, cfgs: dict[str, ActuatorCfg] | None) -> None:
    """Preserve imported actuation for None, otherwise replace it before attachment.

    Replacement clears keyframe controls, since their old ordering no longer applies.
    MotrixSim MSD keyframes do not expose actuator activation state.
    """
    if cfgs is None:
        return
    target_names = {cfg.joint_name for cfg in cfgs.values()}
    if len(target_names) != len(cfgs):
        raise ValueError("Multiple actuator configs target the same joint")
    if any(not name for name in cfgs):
        raise ValueError("Configured actuator name must not be empty")
    joints = {
        joint.name: joint
        for body in world.hierarchy.bodies
        for link in _iter_links(body.link)
        for joint in link.joints
        if joint.name is not None
    }
    unknown_joints = sorted(target_names.difference(joints))
    if unknown_joints:
        raise ValueError(f"Configured actuator joints do not exist in model: {unknown_joints}")
    replacements = []
    for name, cfg in cfgs.items():
        cfg.validate()
        actuator = _build_actuator(name, cfg)
        if isinstance(cfg, PositionActuatorCfg) and cfg.inherit_joint_range:
            actuator.ctrlrange = joints[cfg.joint_name].pos_limit
        replacements.append(actuator)
    world.actuators.clear()
    world.actuators.extend(replacements)
    for key in world.keyframes:
        key.ctrl = [0.0] * len(replacements)
