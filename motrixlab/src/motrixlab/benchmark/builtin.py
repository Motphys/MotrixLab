# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Registration of built-in named benchmarks."""

from .env import register as register_env_step
from .physics import register as register_physics_step

__all__ = ["register_env_step", "register_physics_step"]
