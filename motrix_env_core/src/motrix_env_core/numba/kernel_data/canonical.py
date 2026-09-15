# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar, cast

import numpy as np

from motrix_env_core.numba.kernel_data.lowering import KernelDataLowering
from motrix_env_core.numba.kernel_data.tree import (
    flatten_kernel_data,
    is_kernel_data,
    unflatten_kernel_data,
)

KernelDataType = TypeVar("KernelDataType")


def canonicalize_kernel_data(value: KernelDataType, *, context: str) -> KernelDataType:
    """Copy one KernelData value into environment-owned backing arrays."""
    if not is_kernel_data(value):
        raise TypeError(f"{context} must return a concrete @kernel_data, got {type(value).__name__}.")
    leaves, tree_def = flatten_kernel_data(value)
    KernelDataLowering().lower(tree_def, context=context)
    copied = tuple(leaf.copy() if isinstance(leaf, np.ndarray) else leaf for leaf in leaves)
    return cast(KernelDataType, unflatten_kernel_data(tree_def, copied))


__all__ = ["canonicalize_kernel_data"]
