# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral array values and namespace helpers.

``Array`` is a host-side value object. Compiled Manager regions must receive its
``native`` storage instead of the wrapper.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, TypeGuard, Union, overload

import numpy as np
import numpy.typing as npt
from array_api_compat import array_namespace as _array_namespace
from array_api_compat import is_numpy_array, is_torch_array

NativeArray = Union[npt.NDArray[Any], "torch.Tensor"]  # noqa: F821


class Array:
    """A small host-side wrapper around a NumPy array or Torch tensor."""

    __slots__ = ("_value",)

    def __init__(self, value: NativeArray | Array) -> None:
        if isinstance(value, Array):
            value = value._value
        if not is_array(value):
            raise TypeError(f"Array only supports NumPy arrays and Torch tensors, got {type(value).__name__}.")
        self._value = value

    @property
    def native(self) -> NativeArray:
        """Return the backing native array without copying."""
        return self._value

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._value.shape)

    @property
    def dtype(self) -> Any:
        return self._value.dtype

    @property
    def device(self) -> Any:
        return getattr(self._value, "device", None)

    def numpy(self) -> np.ndarray:
        if isinstance(self._value, np.ndarray):
            return self._value
        return self._value.detach().cpu().numpy()

    def torch(self, *, device: Any = None):
        import torch

        if isinstance(self._value, torch.Tensor):
            return self._value if device is None else self._value.to(device)
        return torch.as_tensor(self._value, device=device)

    def __array_namespace__(self, *, api_version: str | None = None) -> ModuleType:
        del api_version
        return array_namespace(self._value)

    def __getitem__(self, index: Any) -> Array:
        return Array(self._value[index])

    def _binary(self, other: Any, operation) -> Array:
        other = unwrap_array(other)
        return Array(operation(self._value, other))

    def __add__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: left + right)

    def __radd__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: right + left)

    def __sub__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: left - right)

    def __rsub__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: right - left)

    def __mul__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: left * right)

    def __rmul__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: right * left)

    def __truediv__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: left / right)

    def __rtruediv__(self, other: Any) -> Array:
        return self._binary(other, lambda left, right: right / left)

    def __neg__(self) -> Array:
        return Array(-self._value)

    def __repr__(self) -> str:
        return f"Array({self._value!r})"


@overload
def unwrap_array(value: Array) -> NativeArray: ...


@overload
def unwrap_array(value: NativeArray) -> NativeArray: ...


def unwrap_array(value: Array | NativeArray) -> NativeArray:
    return value.native if isinstance(value, Array) else value


def array_namespace(*arrays: Array | NativeArray) -> ModuleType:
    """Return the NumPy or Torch Array API namespace shared by ``arrays``."""
    native = tuple(unwrap_array(array) for array in arrays)
    for array in native:
        if not is_array(array):
            raise TypeError(
                f"array_namespace only supports NumPy arrays and Torch tensors, got {type(array).__name__}."
            )
    return _array_namespace(*native)


def is_array(value: object) -> TypeGuard[NativeArray | Array]:
    """Return whether ``value`` is a supported array or host wrapper."""
    if isinstance(value, Array):
        return True
    return (is_numpy_array(value) and isinstance(value, np.ndarray)) or is_torch_array(value)


__all__ = ["Array", "NativeArray", "array_namespace", "is_array", "unwrap_array"]
