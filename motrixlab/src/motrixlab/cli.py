# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unified command entry point; component packages own command behavior."""

import argparse
import sys
from importlib import import_module
from importlib.metadata import version

_COMMANDS = {
    "train": "motrix_rl.commands.train",
    "play": "motrix_rl.commands.play",
    "view": "motrixlab.commands.view",
    "export-onnx": "motrix_rl.commands.export_onnx",
    "export-deploy": "motrix_rl.commands.export_deploy",
    "deploy": "motrix_deploy.cli",
}


def main() -> int | None:
    parser = argparse.ArgumentParser(prog="mxlab", description="MotrixLab training and deployment tools.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('motrixlab')}")
    parser.add_argument("command", choices=tuple(_COMMANDS), help="command to run; use COMMAND --help for details")
    if sys.argv[1:2] and sys.argv[1] in _COMMANDS:
        command = sys.argv[1]
    else:
        parser.parse_args(sys.argv[1:] or ["--help"])
        return None

    original_argv = sys.argv
    sys.argv = [f"mxlab {command}", *original_argv[2:]]
    try:
        return import_module(_COMMANDS[command]).main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
