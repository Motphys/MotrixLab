# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Send a bounded single-joint position command to a physical Unitree Go2."""

import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from motrix_deploy_unitree import UnitreeGo2DirectInterface


def _build_robot(
    artifact: str | Path,
    network_interface: str,
    *,
    kp: float,
    kd: float,
) -> UnitreeGo2DirectInterface:
    return UnitreeGo2DirectInterface.from_artifact(
        artifact,
        network_interface=network_interface,
        backend_options={"kp": kp, "kd": kd},
    )


def _validate_duration(name: str, value: float, *, allow_zero: bool) -> None:
    minimum_valid = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not minimum_valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} finite number, got {value!r}")


def _send_trajectory(
    robot: UnitreeGo2DirectInterface,
    start: np.ndarray,
    target: np.ndarray,
    duration_s: float,
) -> None:
    steps = max(1, math.ceil(duration_s / robot.control_period_s))
    next_deadline = time.monotonic()
    for step in range(1, steps + 1):
        robot.read_data()
        alpha = np.float32(step / steps)
        position = ((np.float32(1.0) - alpha) * start + alpha * target).astype(np.float32)
        robot.send_joint_command(position)

        next_deadline += robot.control_period_s
        remaining_s = next_deadline - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)


def _hold_position(robot: UnitreeGo2DirectInterface, position: np.ndarray, duration_s: float) -> None:
    steps = math.ceil(duration_s / robot.control_period_s)
    next_deadline = time.monotonic()
    for _ in range(steps):
        robot.read_data()
        robot.send_joint_command(position)

        next_deadline += robot.control_period_s
        remaining_s = next_deadline - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)


def run_joint_control(
    *,
    artifact: str | Path,
    network_interface: str,
    joint_name: str,
    target_position: float,
    kp: float,
    kd: float,
    move_duration: float,
    hold_duration: float,
    return_duration: float,
) -> None:
    _validate_duration("--move-duration", move_duration, allow_zero=False)
    _validate_duration("--hold-duration", hold_duration, allow_zero=True)
    _validate_duration("--return-duration", return_duration, allow_zero=True)
    if not math.isfinite(target_position):
        raise ValueError(f"target_position must be finite, got {target_position!r}")

    robot = _build_robot(artifact, network_interface, kp=kp, kd=kd)
    joint_names: Sequence[str] = robot.robot.joint_names
    if joint_name not in joint_names:
        raise ValueError(f"unknown joint {joint_name!r}; expected one of {list(joint_names)}")

    joint_index = joint_names.index(joint_name)
    lower = float(robot.robot.position_lower[joint_index])
    upper = float(robot.robot.position_upper[joint_index])
    if not lower <= target_position <= upper:
        raise ValueError(f"target {target_position:.6f} rad for {joint_name} is outside [{lower:.6f}, {upper:.6f}]")

    default_position = np.array(robot.robot.default_joint_position, dtype=np.float32, copy=True)
    command_position = np.array(default_position, copy=True)
    command_position[joint_index] = np.float32(target_position)

    print(f"Joint: {joint_name}")
    print(f"Control contract: artifact {artifact}")
    print(f"Default: {default_position[joint_index]:.6f} rad")
    print(f"Target: {command_position[joint_index]:.6f} rad")
    print("Press Start to begin the default-pose transition, A to enable commands, or Select for emergency stop.")

    try:
        with robot:
            robot.enable_command_output()
            _send_trajectory(robot, default_position, command_position, move_duration)
            _hold_position(robot, command_position, hold_duration)
            if return_duration > 0.0:
                _send_trajectory(robot, command_position, default_position, return_duration)
                _hold_position(robot, default_position, robot.control_period_s)
    except KeyboardInterrupt:
        print("Interrupted; sending damping stop.")
