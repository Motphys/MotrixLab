# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl import frameworks
from motrix_rl_skrl.framework import SkrlFramework


def register() -> None:
    frameworks.register_framework(SkrlFramework())
