# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Explicit release of the built-in Go2 motion controller."""

from collections.abc import Callable, Mapping

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


def check_motion_mode(client) -> str:
    status, mode = client.CheckMode()
    if status != 0:
        raise RuntimeError(f"MotionSwitcher CheckMode failed with code {status}")
    if not isinstance(mode, Mapping) or not isinstance(mode.get("name"), str):
        raise RuntimeError(f"MotionSwitcher returned invalid mode: {mode!r}")
    return mode["name"]


def require_official_control_released(timeout_s: float) -> None:
    """Query ownership only; never change robot mode or service state."""
    client = MotionSwitcherClient()
    client.SetTimeout(timeout_s)
    client.Init()
    mode = check_motion_mode(client)
    if mode:
        raise RuntimeError(
            f"Official motion controller is still active ({mode!r}). "
            "Release it explicitly with release-control before low-level control."
        )


def release_official_control(timeout_s: float, *, confirm_stand_down: Callable[[], bool]) -> bool:
    """Request stand-down/release and disable sport_mode; caller confirms safety."""
    client = MotionSwitcherClient()
    client.SetTimeout(timeout_s)
    client.Init()
    if check_motion_mode(client):
        sport = SportClient()
        sport.SetTimeout(timeout_s)
        sport.Init()
        status = sport.StandDown()
        if status != 0:
            raise RuntimeError(f"SportClient StandDown failed with code {status}")
        if not confirm_stand_down():
            return False
        status, _ = client.ReleaseMode()
        if status != 0:
            raise RuntimeError(f"MotionSwitcher ReleaseMode failed with code {status}")
        if check_motion_mode(client):
            raise RuntimeError("MotionSwitcher mode remains active after release")
    robot_state = RobotStateClient()
    robot_state.SetTimeout(timeout_s)
    robot_state.Init()
    status = robot_state.ServiceSwitch("sport_mode", False)
    if status != 0:
        raise RuntimeError(f"RobotState ServiceSwitch(sport_mode, false) failed with code {status}")
    if check_motion_mode(client):
        raise RuntimeError("MotionSwitcher mode is active after disabling sport_mode")
    return True
