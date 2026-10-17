# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Compatibility wrapper; prefer ``mxexport deploy``."""

from motrix_rl.commands.export_deploy import main

if __name__ == "__main__":
    main()
