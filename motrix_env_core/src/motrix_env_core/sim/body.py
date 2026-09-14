# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Body-model assembly: the cross-backend contract for ``SimModel.bodies``.

Backends extract per-body facts from their engine model (names, limits,
actuator metadata, one forward-kinematics evaluation) and hand them to
:func:`assemble_body_model`, which owns the cross-backend contract: key-pose
permutation into the body joint order, existence/coverage validation, and the
baked init snapshot. Consumers read the result through :class:`BodyModel`
(defined with the other model-surface types in ``sim.backend``) on
``SimModel.bodies``.

Cross-backend invariants:

- ``bodies`` keys are ``SceneObjsCfg`` field names, identical for the same
  ``SceneCfg`` regardless of backend;
- ordering inside one ``BodyModel`` is the backend's body joint-DOF order and
  may differ across backends — consumers must resolve joints by name, never
  by hard-coded index;
- quaternions are ``(x, y, z, w)`` float32; link poses are world-frame values
  evaluated with the body at its default placement.
"""

from __future__ import annotations

import numpy as np

from motrix_env_core.config.scene.base import RobotCfg
from motrix_env_core.sim.model import ActuatorSpec, BodyModel

__all__ = ["assemble_body_model", "resolved_key_pose"]


def resolved_key_pose(
    name: str,
    robot_cfg: RobotCfg,
    pose_name: str,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    """Permutate one declared key pose into the body joint order.

    Every declared joint must exist on the body and every body joint must be
    covered by the declaration; violations raise at compile time.

    Args:
        name: ``SceneObjsCfg`` field name of the body, for error messages.
        robot_cfg: The robot config declaring the key poses.
        pose_name: The key pose to permutate.
        joint_names: The body's named joints in joint-DOF order.

    Returns:
        ``(len(joint_names),)`` float32 joint angles in body order.

    Raises:
        ValueError: If the pose is undeclared, a declared joint does not exist
            on the body, or a body joint is not covered by the declaration.
    """
    joint_index = {joint: index for index, joint in enumerate(joint_names)}
    resolved = [robot_cfg.resolve_name(joint) for joint in robot_cfg.key_pose.joint_names]
    unknown = sorted(set(resolved).difference(joint_index))
    if unknown:
        raise ValueError(f"RobotCfg for body {name!r} declares key-pose joints unknown to the body: {unknown}.")
    missing = sorted(set(joint_names).difference(resolved))
    if missing:
        raise ValueError(f"RobotCfg for body {name!r} key poses must cover every joint; missing: {missing}.")
    try:
        values = robot_cfg.key_pose.poses[pose_name]
    except KeyError as error:
        raise ValueError(
            f"RobotCfg for body {name!r} has no key pose {pose_name!r}; "
            f"available poses: {sorted(robot_cfg.key_pose.poses)}."
        ) from error
    pose = np.zeros((len(joint_names),), dtype=np.float32)
    for joint, value in zip(resolved, values, strict=True):
        pose[joint_index[joint]] = value
    return pose


def assemble_body_model(
    *,
    name: str,
    base_link_name: str,
    link_names: tuple[str, ...],
    joint_names: tuple[str, ...],
    joint_pos_limits: tuple[np.ndarray, np.ndarray] | None,
    scene_actuators: tuple[ActuatorSpec, ...],
    init_base_position: np.ndarray,
    init_base_quat: np.ndarray,
    init_link_positions: np.ndarray,
    init_link_quats: np.ndarray,
    robot_cfg: RobotCfg | None = None,
) -> BodyModel:
    """Assemble a :class:`BodyModel` from backend-extracted engine facts.

    Permutates each declared key pose from ``KeyPoseCfg.joint_names`` order
    into the body joint order — every declared joint must exist on the body
    and every body joint must be covered — and bakes the init snapshot from
    the configured ``init_key_pose``. The body-scoped actuator view is
    derived here once from the scene-wide specs, so backends never implement
    per-body actuator filtering themselves. Pass ``robot_cfg=None`` for prop
    bodies.

    Args:
        name: ``SceneObjsCfg`` field name of the body.
        base_link_name: Resolved engine name of the attach root link.
        link_names: All links of the body; alignment axis of the FK arrays.
        joint_names: The body's named joints in joint-DOF order.
        joint_pos_limits: ``(lower, upper)`` aligned to ``joint_names``, or
            ``None`` when the body declares no joints.
        scene_actuators: The full-scene actuator specs in engine model order
            (the same tuple the backend reports as ``SimModel.actuators``).
        init_base_position: ``(3,)`` world-frame position of the default
            base placement.
        init_base_quat: ``(4,)`` xyzw orientation of the default base
            placement.
        init_link_positions: ``(num_links, 3)`` FK positions at the init pose.
        init_link_quats: ``(num_links, 4)`` xyzw FK rotations at the init pose.
        robot_cfg: The scene's ``RobotCfg`` when the body declares key poses.

    Returns:
        The assembled body model with a baked init snapshot.

    Raises:
        ValueError: If array shapes disagree with the name axes, or the
            key-pose declaration does not exactly cover the body joints.
    """
    init_base_position = np.asarray(init_base_position, dtype=np.float32).reshape(-1)
    init_base_quat = np.asarray(init_base_quat, dtype=np.float32).reshape(-1)
    if init_base_position.shape != (3,):
        raise ValueError(f"Body {name!r} init_base_position must have shape (3,), got {init_base_position.shape}.")
    if init_base_quat.shape != (4,):
        raise ValueError(f"Body {name!r} init_base_quat must have shape (4,), got {init_base_quat.shape}.")
    positions = np.ascontiguousarray(init_link_positions, dtype=np.float32)
    quats = np.ascontiguousarray(init_link_quats, dtype=np.float32)
    if positions.shape != (len(link_names), 3):
        raise ValueError(
            f"Body {name!r} init_link_positions must have shape ({len(link_names)}, 3), got {positions.shape}."
        )
    if quats.shape != (len(link_names), 4):
        raise ValueError(f"Body {name!r} init_link_quats must have shape ({len(link_names)}, 4), got {quats.shape}.")
    if joint_pos_limits is not None:
        lower, upper = (np.asarray(limits, dtype=np.float32).reshape(-1) for limits in joint_pos_limits)
        if lower.shape != (len(joint_names),) or upper.shape != (len(joint_names),):
            raise ValueError(
                f"Body {name!r} joint_pos_limits must align with {len(joint_names)} joints, "
                f"got shapes {lower.shape} and {upper.shape}."
            )
        limits: tuple[np.ndarray, np.ndarray] | None = (lower, upper)
    else:
        limits = None

    joint_names = tuple(joint_names)
    # Body-scoped view of the scene actuators: specs whose target is one of
    # this body's joints, keeping the scene-wide engine model order.
    body_actuators = tuple(spec for spec in scene_actuators if spec.target_name in set(joint_names))
    if robot_cfg is None or not robot_cfg.key_pose.poses:
        # Bodies without key poses get zero init joint angles.
        init_joint_pos = np.zeros((len(joint_names),), dtype=np.float32)
    else:
        init_joint_pos = resolved_key_pose(name, robot_cfg, robot_cfg.init_key_pose, joint_names)
    return BodyModel(
        name=name,
        base_link_name=base_link_name,
        link_names=tuple(link_names),
        joint_names=joint_names,
        joint_pos_limits=limits,
        actuators=body_actuators,
        init_base_position=init_base_position,
        init_base_quat=init_base_quat,
        init_joint_pos=init_joint_pos,
        init_link_positions=positions,
        init_link_quats=quats,
    )
