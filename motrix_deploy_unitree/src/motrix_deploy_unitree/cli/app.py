# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command-line diagnostics for Unitree deployment backends."""

import argparse
import math
from collections.abc import Sequence

from motrix_deploy_unitree.cli.joint_control.runner import run_joint_control
from motrix_deploy_unitree.cli.read_lowstate import run_read_lowstate
from motrix_deploy_unitree.cli.release_official_control import run_release_official_control


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _non_negative_float(value: str) -> float:
    result = _finite_float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_float(value: str) -> float:
    result = _finite_float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unitree hardware diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)

    lowstate = commands.add_parser(
        "read-lowstate",
        help="Read rt/lowstate without creating a LowCmd publisher",
    )
    lowstate.add_argument("network_interface", help="Ethernet interface connected to the robot, for example enp5s0")
    lowstate.add_argument("--robot", choices=("go2", "g1-29dof"), default="go2")
    lowstate.add_argument("--domain-id", type=_non_negative_int, default=0, help="DDS domain id (default: 0)")
    lowstate.add_argument("--topic", default="rt/lowstate", help="LowState DDS topic (default: rt/lowstate)")
    lowstate.add_argument("--queue-depth", type=_positive_int, default=10, help="Subscriber queue depth (default: 10)")
    lowstate.add_argument(
        "--duration-s",
        type=_non_negative_float,
        default=10.0,
        help="Test duration; 0 runs until Ctrl+C (default: 10 seconds)",
    )
    lowstate.add_argument(
        "--print-interval-s",
        type=_positive_float,
        default=0.5,
        help="Minimum interval between printed samples (default: 0.5 seconds)",
    )
    lowstate.add_argument("--no-validate-crc", action="store_true", help="Print frames without checking CRC")

    release = commands.add_parser("release-control", help="Explicitly stand down and release official motion control")
    release.add_argument("network_interface")
    release.add_argument("--domain-id", type=_non_negative_int, default=0)
    release.add_argument("--timeout-s", type=_positive_float, default=5.0)

    joint = commands.add_parser("joint-control", help="Send a bounded single-joint position command")
    joint.add_argument("network_interface", help="Ethernet interface connected to Go2, for example enp5s0")
    joint.add_argument("joint_name", help="Canonical joint name, for example FL_thigh_joint")
    target = joint.add_mutually_exclusive_group(required=True)
    target.add_argument("--target", type=_finite_float, help="Absolute target position in radians")
    target.add_argument(
        "--offset", type=_finite_float, help="Offset from the selected hold pose, not measured state (rad)"
    )
    joint.add_argument("--hold-pose", default="default", help="Named hardware hold pose (default: default)")
    joint.add_argument("--control-period-s", type=_positive_float, default=0.02)
    joint.add_argument("--state-timeout-s", type=_positive_float, default=0.1)
    joint.add_argument("--kp", type=_non_negative_float, required=True, help="Explicit backend servo safety stiffness")
    joint.add_argument("--kd", type=_non_negative_float, required=True, help="Explicit backend servo safety damping")
    joint.add_argument("--move-duration", type=_positive_float, default=2.0)
    joint.add_argument("--hold-duration", type=_non_negative_float, default=1.0)
    joint.add_argument("--return-duration", type=_non_negative_float, default=2.0)
    joint.add_argument("--startup-max-speed-rad-s", type=_positive_float, default=0.5)
    joint.add_argument("--startup-timeout-s", type=_positive_float, default=10.0)
    joint.add_argument("--startup-gain-ramp-s", type=_positive_float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and dispatch Unitree diagnostic commands."""
    args = _parser().parse_args(argv)
    if args.command == "read-lowstate":
        return run_read_lowstate(
            network_interface=args.network_interface,
            domain_id=args.domain_id,
            topic=args.topic,
            queue_depth=args.queue_depth,
            duration_s=args.duration_s,
            print_interval_s=args.print_interval_s,
            validate_crc=not args.no_validate_crc,
            robot=args.robot,
        )

    if args.command == "release-control":
        return run_release_official_control(
            network_interface=args.network_interface, domain_id=args.domain_id, timeout_s=args.timeout_s
        )

    run_joint_control(
        network_interface=args.network_interface,
        joint_name=args.joint_name,
        target_position=args.target,
        offset=args.offset,
        hold_pose=args.hold_pose,
        control_period_s=args.control_period_s,
        state_timeout_s=args.state_timeout_s,
        kp=args.kp,
        kd=args.kd,
        move_duration=args.move_duration,
        hold_duration=args.hold_duration,
        return_duration=args.return_duration,
        startup_max_speed_rad_s=args.startup_max_speed_rad_s,
        startup_timeout_s=args.startup_timeout_s,
        startup_gain_ramp_s=args.startup_gain_ramp_s,
    )
    return 0
