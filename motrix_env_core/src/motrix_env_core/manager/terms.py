# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from motrix_env_core.manager.kernel_data import is_kernel_data, validate_kernel_data
from motrix_env_core.sim.read import SimDataQuery

# Static term-argument leaves accepted as-is; anything else is a cfg-authoring
# error caught at term construction.
_STATIC_LEAVES = (bool, int, float, str, np.generic, np.ndarray, torch.Tensor, SimDataQuery)


def canonicalize_term_args(args: tuple[Any, ...], *, context: str) -> tuple[Any, ...]:
    """Validate and canonicalize positional term arguments.

    Accepted leaves are scalars (including strings for compile-time Map-name
    lookups), scalar tuples, ndarrays, tensors, simulator data queries, and
    kernel-data dataclass carriers (which are normalized in place).
    """
    values: list[Any] = []
    for index, value in enumerate(args):
        if is_kernel_data(value):
            values.append(validate_kernel_data(value, context=f"{context} args[{index}]"))
        elif isinstance(value, _STATIC_LEAVES):
            values.append(value)
        elif isinstance(value, tuple) and all(isinstance(item, (bool, int, float, np.generic)) for item in value):
            values.append(value)
        else:
            raise TypeError(
                f"{context} args[{index}] must be a scalar, a scalar tuple, an ndarray, a tensor, "
                f"a kernel-data dataclass, or a simulator data query; got {type(value).__name__}."
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
        if not isinstance(self.args, tuple):
            raise TypeError("Term invocation args must be a tuple.")


__all__ = [
    "BaseTerm",
    "canonicalize_term_args",
]
