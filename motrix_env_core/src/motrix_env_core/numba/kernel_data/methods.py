# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from numba import types
from numba.extending import overload_method

from motrix_env_core.numba.manager.dispatch import _DISPATCH_MARKER

_METHOD_OVERLOADS: dict[str, dict[type, Callable[..., Any]]] = {}


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


def register_proxy_method(proxy: type, name: str, method: Callable[..., Any]) -> None:
    """Register one marked KernelData method for a lowered proxy type."""
    methods = _METHOD_OVERLOADS.get(name)
    if methods is not None:
        methods[proxy] = method
        return
    methods = {proxy: method}
    _METHOD_OVERLOADS[name] = methods

    def getter(self: Any, *args: Any, **kwargs: Any) -> Any:
        return methods.get(self.instance_class)

    # Register one getter per name: competing attribute templates for the same
    # tuple method can hide later proxies. Homogeneous records (including
    # single-field records) use NamedUniTuple.
    for tuple_type in (types.NamedTuple, types.NamedUniTuple):
        overload_method(tuple_type, name, strict=False, inline="always")(getter)


__all__ = ["dispatch_methods", "register_proxy_method"]
