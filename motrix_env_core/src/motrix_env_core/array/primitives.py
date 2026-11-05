# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral primitive namespace for shared MDP functions.

Public functions dispatch eagerly to native NumPy or Torch arrays.
"""

from __future__ import annotations

import numpy as np
from array_api_compat import is_torch_array

from motrix_env_core.array_api import Array, NativeArray, unwrap_array


def sum(x: Array | NativeArray, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False) -> NativeArray:
    x = unwrap_array(x)
    if isinstance(x, np.ndarray):
        return np.sum(x, axis=axis, keepdims=keepdims)
    if is_torch_array(x):
        import torch

        return torch.sum(x, dim=axis, keepdim=keepdims)
    raise TypeError(f"Unsupported array type {type(x).__name__}.")


def exp(x: Array | NativeArray) -> NativeArray:
    x = unwrap_array(x)
    if isinstance(x, np.ndarray):
        return np.exp(x)
    if is_torch_array(x):
        import torch

        return torch.exp(x)
    raise TypeError(f"Unsupported array type {type(x).__name__}.")


def where(condition: Array | NativeArray, x: Array | NativeArray, y: Array | NativeArray) -> NativeArray:
    condition = unwrap_array(condition)
    x = unwrap_array(x)
    y = unwrap_array(y)
    if isinstance(condition, np.ndarray):
        return np.where(condition, x, y)
    if is_torch_array(condition):
        import torch

        return torch.where(condition, x, y)
    raise TypeError(f"Unsupported array type {type(condition).__name__}.")


def clip(x: Array | NativeArray, low: float | int, high: float | int) -> NativeArray:
    x = unwrap_array(x)
    if isinstance(x, np.ndarray):
        return np.clip(x, low, high)
    if is_torch_array(x):
        import torch

        return torch.clamp(x, min=low, max=high)
    raise TypeError(f"Unsupported array type {type(x).__name__}.")


def zeros_like(x: Array | NativeArray) -> NativeArray:
    x = unwrap_array(x)
    if isinstance(x, np.ndarray):
        return np.zeros_like(x)
    if is_torch_array(x):
        import torch

        return torch.zeros_like(x)
    raise TypeError(f"Unsupported array type {type(x).__name__}.")


__all__ = ["clip", "exp", "sum", "where", "zeros_like"]
