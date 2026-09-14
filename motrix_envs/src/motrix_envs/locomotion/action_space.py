# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Action-space helpers shared by locomotion environments."""

import gymnasium as gym
import numpy as np

from motrix_env_core.sim.model import ActuatorType


def symmetric_residual_action_space(
    control_ranges: np.ndarray,
    default_values: np.ndarray,
    action_scales: float | np.ndarray,
) -> gym.spaces.Box:
    """Build zero-centered residual-action bounds for affine controls.

    Bounds are divided by the per-actuator scale applied by the environment so
    the normalized action range maps exactly onto the control limits. Zero
    scales (inert actuators) get a zero bound instead of an infinite one.
    """
    lower, upper = control_ranges
    scales = np.asarray(action_scales, dtype=np.float32)
    boundaries = np.maximum(np.abs(lower - default_values), np.abs(upper - default_values)) / scales
    boundaries = np.where(scales == 0.0, 0.0, boundaries)
    return gym.spaces.Box(-boundaries, boundaries, dtype=np.float32)


def asymmetric_residual_action_space(
    control_ranges: np.ndarray,
    default_values: np.ndarray,
    action_scales: float | np.ndarray,
) -> gym.spaces.Box:
    """Build non-symmetric residual-action bounds for affine controls.

    Unlike :func:`symmetric_residual_action_space`, each dimension keeps its
    own lower/upper residual extent, so the action is not constrained to be
    zero-centered. This is useful when the default value sits far from the
    midpoint of the control range (e.g. heavily asymmetric joint limits).
    """
    lower, upper = control_ranges
    scales = np.asarray(action_scales, dtype=np.float32)
    low = (lower - default_values) / scales
    high = (upper - default_values) / scales
    return gym.spaces.Box(
        np.where(scales == 0.0, 0.0, low),
        np.where(scales == 0.0, 0.0, high),
        dtype=np.float32,
    )


def joint_position_action_space(
    actuators,
    default_angles: np.ndarray,
    action_scales: float | np.ndarray,
    actuator_indices: np.ndarray | None = None,
) -> gym.spaces.Box:
    """Build symmetric bounds for joint-position actions.

    Each actuator's bound is the furthest distance from its default angle to
    either position-control limit, divided by the per-actuator scale applied
    by the environment. Pass a scalar for uniform scales. Position actuators
    must declare ``ctrl_range`` directly or inherit it from their target joint
    during model construction.
    """
    if actuator_indices is None:
        actuator_indices = np.arange(len(actuators), dtype=np.int64)
    control_ranges = []
    for actuator_index in actuator_indices:
        actuator = actuators[int(actuator_index)]
        if actuator.actuator_type is not ActuatorType.POSITION:
            raise ValueError(f"actuator {actuator.name!r} must be a position actuator, got {actuator.actuator_type!r}")
        if actuator.ctrl_range is None:
            raise ValueError(f"position actuator {actuator.name!r} must define or inherit ctrl_range")
        control_ranges.append(actuator.ctrl_range)
    return joint_position_action_space_from_ctrl_ranges(
        np.asarray(control_ranges, dtype=np.float32),
        default_angles,
        action_scales,
        # The loop above already selected rows in ``actuator_indices`` order;
        # passing the indices again would re-index (and permute) the rows.
        None,
    )


def joint_position_action_space_from_ctrl_ranges(
    ctrl_ranges: np.ndarray,
    default_angles: np.ndarray,
    action_scales: float | np.ndarray,
    actuator_indices: np.ndarray | None = None,
) -> gym.spaces.Box:
    """Build symmetric joint-position action bounds from ``(num_actuators, 2)`` ctrl ranges."""
    if actuator_indices is None:
        actuator_indices = np.arange(ctrl_ranges.shape[0], dtype=np.int64)
    default_angles = np.asarray(default_angles, dtype=np.float32)
    expected_shape = (actuator_indices.size,)
    if default_angles.shape != expected_shape:
        raise ValueError(f"default_angles must have shape {expected_shape}, got {default_angles.shape}")
    control_ranges = np.asarray(ctrl_ranges, dtype=np.float32)[actuator_indices].T
    if control_ranges.shape != (2, actuator_indices.size):
        raise ValueError(
            f"actuator control ranges must have shape (2, {actuator_indices.size}), got {control_ranges.shape}"
        )
    return symmetric_residual_action_space(control_ranges, default_angles, action_scales)
