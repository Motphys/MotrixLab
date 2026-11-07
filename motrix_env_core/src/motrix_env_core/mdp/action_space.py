# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Action-space construction helpers for manager and direct environments.

Computations run on Torch tensors; the boundary to ``gym.spaces.Box`` (NumPy
currency) is the final construction step.
"""

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.sim.model import ActuatorType


def symmetric_residual_action_space(
    control_ranges: torch.Tensor,
    default_values: torch.Tensor,
    action_scales: float | torch.Tensor,
) -> gym.spaces.Box:
    """Build zero-centered residual-action bounds for affine controls.

    Bounds are divided by the per-actuator scale applied by the environment so
    the normalized action range maps exactly onto the control limits. Zero
    scales (inert actuators) get a zero bound instead of an infinite one.

    Args:
        control_ranges: ``(2, A)`` lower/upper position-control limits.
        default_values: Default (center) values, ``(A,)``.
        action_scales: Per-actuator scales ``(A,)`` or one scalar for all.

    Returns:
        Symmetric ``Box`` with bounds ``[-limit, +limit]`` per actuator.
    """
    lower, upper = control_ranges
    scales = torch.as_tensor(action_scales, dtype=torch.float32)
    boundaries = torch.maximum(torch.abs(lower - default_values), torch.abs(upper - default_values)) / scales
    boundaries = torch.where(scales == 0.0, torch.zeros_like(boundaries), boundaries)
    return gym.spaces.Box(-boundaries.numpy(), boundaries.numpy(), dtype=np.float32)


def asymmetric_residual_action_space(
    control_ranges: torch.Tensor,
    default_values: torch.Tensor,
    action_scales: float | torch.Tensor,
) -> gym.spaces.Box:
    """Build non-symmetric residual-action bounds for affine controls.

    Unlike :func:`symmetric_residual_action_space`, each dimension keeps its
    own lower/upper residual extent, so the action is not constrained to be
    zero-centered. This is useful when the default value sits far from the
    midpoint of the control range (e.g. heavily asymmetric joint limits).

    Args:
        control_ranges: ``(2, A)`` lower/upper position-control limits.
        default_values: Default (center) values, ``(A,)``.
        action_scales: Per-actuator scales ``(A,)`` or one scalar for all.

    Returns:
        Asymmetric ``Box`` with independent per-actuator low/high bounds.
    """
    lower, upper = control_ranges
    scales = torch.as_tensor(action_scales, dtype=torch.float32)
    low = (lower - default_values) / scales
    high = (upper - default_values) / scales
    zero = torch.zeros_like(low)
    return gym.spaces.Box(
        torch.where(scales == 0.0, zero, low).numpy(),
        torch.where(scales == 0.0, zero, high).numpy(),
        dtype=np.float32,
    )


def joint_position_action_space(
    actuators: tuple,
    default_angles: torch.Tensor,
    action_scales: float | torch.Tensor,
    actuator_indices: torch.Tensor | None = None,
) -> gym.spaces.Box:
    """Build symmetric bounds for joint-position actions.

    Each actuator's bound is the furthest distance from its default angle to
    either position-control limit, divided by the per-actuator scale applied by
    the environment. Pass a scalar for uniform scales. Position actuators
    must declare ``ctrl_range`` directly or inherit it from their target joint
    during model construction.

    Args:
        actuators: Actuator specs in canonical model order.
        default_angles: Default angles for the selected actuators.
        action_scales: Per-actuator scales or one scalar for all.
        actuator_indices: Indices of the actuators to include; ``None``
            selects every actuator.

    Returns:
        Symmetric ``Box`` with bounds ``[-limit, +limit]`` per actuator.
    """
    if actuator_indices is None:
        actuator_indices = torch.arange(len(actuators), dtype=torch.int64)
    control_ranges = []
    for actuator_index in actuator_indices.tolist():
        actuator = actuators[actuator_index]
        if actuator.actuator_type is not ActuatorType.POSITION:
            raise ValueError(f"actuator {actuator.name!r} must be a position actuator, got {actuator.actuator_type!r}")
        if actuator.ctrl_range is None:
            raise ValueError(f"position actuator {actuator.name!r} must define or inherit ctrl_range")
        control_ranges.append(actuator.ctrl_range)
    return joint_position_action_space_from_ctrl_ranges(
        torch.tensor(control_ranges, dtype=torch.float32),
        default_angles,
        action_scales,
        # The loop above already selected rows in ``actuator_indices`` order;
        # passing the indices again would re-index (and permute) the rows.
        None,
    )


def joint_position_action_space_from_ctrl_ranges(
    ctrl_ranges: torch.Tensor,
    default_angles: torch.Tensor,
    action_scales: float | torch.Tensor,
    actuator_indices: torch.Tensor | None = None,
) -> gym.spaces.Box:
    """Build symmetric joint-position action bounds from ctrl ranges.

    Args:
        ctrl_ranges: ``(num_actuators, 2)`` lower/upper control limits.
        default_angles: Default angles, ``(num_actuators,)`` (after any
            subsetting by ``actuator_indices``).
        action_scales: Per-actuator scales or one scalar for all.
        actuator_indices: Indices of the actuators to include; ``None``
            selects every row.

    Returns:
        Symmetric ``Box`` with bounds ``[-limit, +limit]`` per actuator.
    """
    if actuator_indices is None:
        actuator_indices = torch.arange(ctrl_ranges.shape[0], dtype=torch.int64)
    expected_shape = (actuator_indices.numel(),)
    if default_angles.shape != expected_shape:
        raise ValueError(f"default_angles must have shape {expected_shape}, got {tuple(default_angles.shape)}")
    control_ranges = ctrl_ranges[actuator_indices].T
    if control_ranges.shape != (2, actuator_indices.numel()):
        raise ValueError(
            f"actuator control ranges must have shape (2, {actuator_indices.numel()}), got {tuple(control_ranges.shape)}"
        )
    return symmetric_residual_action_space(control_ranges, default_angles, action_scales)
