# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from motrix_env_core.numba.kernel_data import canonicalize_kernel_data, is_kernel_data
from motrix_env_core.sim.read import SimDataQuery


def canonicalize_term_args(args: tuple[Any, ...], *, context: str) -> tuple[Any, ...]:
    """Validate and canonicalize positional Numba-compatible term arguments.

    Accepted leaves are scalars (including strings for compile-time Map-name
    lookups), scalar tuples, ndarrays, simulator data queries, and
    ``@kernel_data`` carriers.
    """
    values: list[Any] = []
    for index, value in enumerate(args):
        if is_kernel_data(value):
            values.append(canonicalize_kernel_data(value, context=f"{context} args[{index}]"))
        elif isinstance(value, tuple) and all(isinstance(item, (bool, int, float, np.generic)) for item in value):
            values.append(value)
        elif isinstance(value, (bool, int, float, str, np.generic)):
            values.append(value)
        elif isinstance(value, np.ndarray):
            values.append(value)
        elif isinstance(value, SimDataQuery):
            values.append(value)
        else:
            raise TypeError(
                f"{context} args[{index}] must be a scalar, a scalar tuple, an ndarray, "
                f"or a @kernel_data value; got {type(value).__name__}."
            )
    return tuple(values)


@dataclass(frozen=True, slots=True, init=False)
class BaseTerm:
    """A dispatch function and its static positional arguments."""

    dispatch: Callable[..., Any]
    args: tuple[Any, ...]

    def __init__(self, dispatch: Callable[..., Any], *args: Any) -> None:
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "args", args)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not inspect.isfunction(self.dispatch):
            raise TypeError(f"Term dispatch must be a Python function, got {type(self.dispatch).__name__}.")
        if not getattr(self.dispatch, "__motrix_manager_dispatch__", False):
            raise TypeError(f"Term dispatch {self.dispatch.__qualname__!r} must be decorated with @dispatch.")
        if not isinstance(self.args, tuple):
            raise TypeError("Term invocation args must be a tuple.")


__all__ = [
    "BaseTerm",
    "canonicalize_term_args",
]
