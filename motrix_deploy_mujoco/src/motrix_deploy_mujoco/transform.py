# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment-specific MuJoCo specification transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from motrix_deploy.contracts import RobotSpec
from motrix_deploy.errors import ValidationError

if TYPE_CHECKING:
    import mujoco


def convert_actuators_to_motors(
    model_spec: mujoco.MjSpec,
    source_model: mujoco.MjModel,
    actuator_indices: NDArray[np.int64],
    robot: RobotSpec,
) -> list[str]:
    """Convert selected, prevalidated position servos to torque motors."""
    import mujoco as mj

    actuator_names: list[str] = []
    for index in actuator_indices:
        name = mj.mj_id2name(source_model, mj.mjtObj.mjOBJ_ACTUATOR, index)
        if name is None:
            raise ValidationError(f"backend.actuators.{index}", "a named actuator", "unnamed")
        actuator_names.append(name)

    for name, torque_limit in zip(actuator_names, robot.torque_limit, strict=True):
        actuator = model_spec.actuator(name)
        if actuator is None:
            raise ValidationError(f"backend.actuators.{name}", "an actuator in MjSpec", "missing")
        actuator.set_to_motor()
        actuator.ctrllimited = mj.mjtLimited.mjLIMITED_TRUE
        actuator.ctrlrange = [-torque_limit, torque_limit]
        actuator.forcelimited = mj.mjtLimited.mjLIMITED_TRUE
        actuator.forcerange = [-torque_limit, torque_limit]
    return actuator_names


__all__ = ["convert_actuators_to_motors"]
