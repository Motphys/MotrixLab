# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Build a native MotrixLab SONIC mmap store from paired motrixlab NPZ directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from motrixlab_sonic_packer import pack_motrixlab_npz_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("robot_root", type=Path, help="Directory containing robot_filtered NPZ clips")
    parser.add_argument("smpl_root", type=Path, help="Directory containing smpl_filtered NPZ clips")
    parser.add_argument("output", type=Path, help="New MotrixLab packed-store directory")
    args = parser.parse_args()
    result = pack_motrixlab_npz_store(args.robot_root, args.smpl_root, args.output)
    print(f"Wrote MotrixLab SONIC packed store: {result}")


if __name__ == "__main__":
    main()
