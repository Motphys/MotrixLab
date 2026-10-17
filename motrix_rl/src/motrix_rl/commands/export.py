# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Export training results as ONNX policies or deployment artifacts."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=("onnx", "deploy"), help="artifact format to export")
    if sys.argv[1:2] in (["onnx"], ["deploy"]):
        format_name = sys.argv[1]
    else:
        parser.parse_args(sys.argv[1:] or ["--help"])
        return

    if format_name == "onnx":
        from motrix_rl.commands.export_onnx import main as export_main
    else:
        from motrix_rl.commands.export_deploy import main as export_main

    original_argv = sys.argv
    sys.argv = [f"{original_argv[0]} {format_name}", *original_argv[2:]]
    try:
        export_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
