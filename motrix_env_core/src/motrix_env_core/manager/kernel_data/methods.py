# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from motrix_env_core.manager.dispatch import _DISPATCH_MARKER


def dispatch_methods(logical_type: type) -> dict[str, Callable[..., Any]]:
    """Resolve marked instance methods with normal Python MRO shadowing."""
    methods: dict[str, Callable[..., Any]] = {}
    seen: set[str] = set()
    for base in logical_type.__mro__:
        for name, value in vars(base).items():
            if name in seen:
                continue
            seen.add(name)
            if inspect.isfunction(value) and getattr(value, _DISPATCH_MARKER, False):
                methods[name] = value
    return methods


__all__ = ["dispatch_methods"]
