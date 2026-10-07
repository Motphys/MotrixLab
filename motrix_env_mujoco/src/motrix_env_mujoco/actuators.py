# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import math

import mujoco as mj

from motrix_env_core.config.scene.actuator import ActuatorCfg, MotorActuatorCfg, PositionActuatorCfg


def apply_actuator_cfgs(spec: mj.MjSpec, cfgs: dict[str, ActuatorCfg] | None) -> None:
    """Replace an imported model's actuators before attachment, or preserve them for ``None``.

    Replacement resets keyframe controls and activations: source actuator ordering and
    dynamics no longer describe the new actuators. Other keyframe state is preserved.
    """
    if cfgs is None:
        return

    if any(not isinstance(name, str) or not name for name in cfgs):
        raise ValueError("Configured actuator names must be non-empty strings")
    target_names = {cfg.joint_name for cfg in cfgs.values()}
    if len(target_names) != len(cfgs):
        raise ValueError("Multiple actuator configs target the same joint")
    joints = {joint.name: joint for joint in spec.joints}
    unknown_joints = sorted(target_names.difference(joints))
    if unknown_joints:
        raise ValueError(f"Configured actuator joints do not exist in model: {unknown_joints}")
    for cfg in cfgs.values():
        if not isinstance(cfg, (PositionActuatorCfg, MotorActuatorCfg)):
            raise TypeError(f"MuJoCo does not support actuator config {type(cfg).__name__}")
        cfg.validate()

    # Materialize the collection before deleting: deletion invalidates the spec iterator.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for key in spec.keys:
        key.ctrl = []
        key.act = []

    for name, cfg in cfgs.items():
        ctrl_range = cfg.ctrl_range
        gain = 1.0
        bias = [0.0] * 10
        bias_type = mj.mjtBias.mjBIAS_NONE
        if isinstance(cfg, PositionActuatorCfg):
            gain = cfg.kp
            bias[:3] = [0.0, -cfg.kp, -cfg.kv]
            bias_type = mj.mjtBias.mjBIAS_AFFINE
            if cfg.inherit_joint_range:
                joint = joints[cfg.joint_name]
                ctrl_range = list(joint.range)
                # Joint hinge limits use compiler angle units; actuator controls always use radians.
                if spec.compiler.degree and joint.type == mj.mjtJoint.mjJNT_HINGE:
                    ctrl_range = [math.radians(value) for value in ctrl_range]
        spec.add_actuator(
            name=name,
            trntype=mj.mjtTrn.mjTRN_JOINT,
            target=cfg.joint_name,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gaintype=mj.mjtGain.mjGAIN_FIXED,
            gainprm=[gain] + [0.0] * 9,
            biastype=bias_type,
            biasprm=bias,
            dyntype=mj.mjtDyn.mjDYN_NONE,
            actdim=0,
            actlimited=mj.mjtLimited.mjLIMITED_FALSE,
            ctrllimited=mj.mjtLimited.mjLIMITED_TRUE if ctrl_range is not None else mj.mjtLimited.mjLIMITED_FALSE,
            ctrlrange=ctrl_range if ctrl_range is not None else [0.0, 0.0],
            forcelimited=mj.mjtLimited.mjLIMITED_TRUE if cfg.force_range is not None else mj.mjtLimited.mjLIMITED_FALSE,
            forcerange=cfg.force_range if cfg.force_range is not None else [0.0, 0.0],
        )
