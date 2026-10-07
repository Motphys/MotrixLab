# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core import registry
from motrix_envs import core

_registered = False


def register() -> None:
    """Register all built-in MotrixLab environments exactly once."""
    global _registered
    if _registered:
        return
    _registered = True
    try:
        from motrix_envs import basic, locomotion, manipulation  # noqa: F401
    except Exception:
        _registered = False
        raise


register()

__all__ = ["core", "register", "registry"]
