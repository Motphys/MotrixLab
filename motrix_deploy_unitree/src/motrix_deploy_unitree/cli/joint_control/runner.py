# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Assemble an artifact-free joint controller with the production hardware runtime."""

import json
import math

from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.cli.joint_control.controller import Go2JointController
from motrix_deploy_unitree.config.control import load_control_config
from motrix_deploy_unitree.config.go2 import (
    Go2JointSettings,
    Go2StartupLimits,
    UnitreeGo2HardwareCfg,
)
from motrix_deploy_unitree.config.runtime import UnitreeTransportConfig
from motrix_deploy_unitree.go2.robot import UnitreeGo2Robot


def run_joint_control(
    *,
    network_interface: str,
    joint_name: str,
    kp: float,
    kd: float,
    target_position: float | None = None,
    offset: float | None = None,
    hold_pose: str = "default",
    control_period_s: float = 0.02,
    state_timeout_s: float = 0.1,
    move_duration: float = 2.0,
    hold_duration: float = 1.0,
    return_duration: float = 2.0,
    startup_max_speed_rad_s: float = 0.5,
    startup_timeout_s: float = 10.0,
    startup_gain_ramp_s: float = 2.0,
) -> None:
    if (target_position is None) == (offset is None):
        raise ValueError("specify exactly one of target_position and offset")
    if not math.isfinite(move_duration) or move_duration <= 0:
        raise ValueError("move_duration must be finite and positive")
    if not math.isfinite(hold_duration) or hold_duration < 0:
        raise ValueError("hold_duration must be finite and non-negative")
    if not math.isfinite(return_duration) or return_duration < 0:
        raise ValueError("return_duration must be finite and non-negative")
    settings = Go2JointSettings(
        kp=kp,
        kd=kd,
        move_duration_s=move_duration,
        hold_duration_s=hold_duration,
        return_duration_s=return_duration,
    )
    startup_limits = Go2StartupLimits(
        max_speed_rad_s=startup_max_speed_rad_s,
        timeout_s=startup_timeout_s,
        gain_ramp_s=startup_gain_ramp_s,
        kp=kp,
        kd=kd,
    )
    hardware = UnitreeGo2HardwareCfg()
    spec = hardware.to_robot_spec(hold_pose)
    robot = UnitreeGo2Robot(
        UnitreeTransportConfig(network_interface=network_interface),
        control=load_control_config("go2"),
        spec=spec,
        control_period_s=control_period_s,
        state_timeout_s=state_timeout_s,
    )
    if joint_name not in spec.joint_names:
        raise ValueError(f"unknown joint {joint_name!r}; expected one of {list(spec.joint_names)}")
    index = spec.joint_names.index(joint_name)
    if offset is not None:
        target_position = float(spec.default_joint_position[index]) + offset
    assert target_position is not None
    controller = Go2JointController(
        spec,
        joint_name,
        target_position,
        settings,
        gamepad=robot.get_gamepad_device(),
        startup_limits=startup_limits,
    )
    print("Measured startup and command targets must stay within model and hardware position bounds.")
    print(
        f"Startup speed <= {startup_max_speed_rad_s:g} rad/s; timeout {startup_timeout_s:g}s; "
        f"gain ramp {startup_gain_ramp_s:g}s."
    )
    print("First startup target is measured pose with zero gains; B during startup cancels directly to damping.")
    print(f"Joint: {joint_name}; hold pose: {hold_pose}")
    print(f"Joint order: {', '.join(spec.joint_names)}")
    print(f"Full-body hold targets (rad): {spec.default_joint_position.tolist()}")
    print(f"Reference: {spec.default_joint_position[index]:.6f} rad; target: {target_position:.6f} rad")
    print(f"Control period: {control_period_s:g}s; state timeout: {state_timeout_s:g}s")
    print(f"All-joint diagnostic gains: kp={kp:g}, kd={kd:g}")
    print("WARNING: All 12 joints will be commanded. Model limits do not certify bench safety.")
    print("Run release-control first. Opening checks ownership; it does not release official control.")
    print("Start: transition to hold pose; A: move; B: return and stop; Select: emergency damping.")
    try:
        confirmed = input("Confirm support, clearance, gains and independent stop; type 'yes' to connect: ").strip()
    except (KeyboardInterrupt, EOFError):
        confirmed = ""
    if confirmed != "yes":
        print("Cancelled; no hardware connection was opened.")
        return
    runtime = HardwareRuntime(robot, realtime=True)
    runtime.bind_session(
        ControlSession(
            robot=robot,
            controller=controller,
            period_s=control_period_s,
            state_timeout_s=state_timeout_s,
        )
    )
    with runtime:
        result = runtime.run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.success:
        raise RuntimeError(f"joint diagnostic stopped: {result.exit_reason}: {result.error or ''}")
