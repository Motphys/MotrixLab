# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl import frameworks
from motrix_rl_builtin.fastsac.framework import MotrixFramework


def register() -> None:
    frameworks.register_framework(MotrixFramework())
