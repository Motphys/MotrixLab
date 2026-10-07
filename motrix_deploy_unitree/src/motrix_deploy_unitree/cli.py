# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Command-line diagnostics for the Unitree Go2 deployment backend."""

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

from motrix_deploy_unitree.go2_joint_control import run_joint_control
from motrix_deploy_unitree.read_lowstate import run_read_lowstate


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
    parser = argparse.ArgumentParser(description="Unitree Go2 hardware diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)

    lowstate = commands.add_parser(
        "read-lowstate",
        help="Read rt/lowstate without creating a LowCmd publisher",
    )
    lowstate.add_argument("network_interface", help="Ethernet interface connected to Go2, for example enp5s0")
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

    joint = commands.add_parser("joint-control", help="Send a bounded single-joint position command")
    joint.add_argument("network_interface", help="Ethernet interface connected to Go2, for example enp5s0")
    joint.add_argument("joint_name", help="Canonical joint name, for example FL_thigh_joint")
    joint.add_argument("target_position", type=_finite_float, help="Absolute target position in radians")
    joint.add_argument("--artifact", type=Path, required=True, help="Frozen Go2 deployment artifact")
    joint.add_argument("--move-duration", type=_positive_float, default=2.0)
    joint.add_argument("--hold-duration", type=_non_negative_float, default=1.0)
    joint.add_argument("--return-duration", type=_non_negative_float, default=2.0)
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
        )

    run_joint_control(
        artifact=args.artifact,
        network_interface=args.network_interface,
        joint_name=args.joint_name,
        target_position=args.target_position,
        move_duration=args.move_duration,
        hold_duration=args.hold_duration,
        return_duration=args.return_duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
