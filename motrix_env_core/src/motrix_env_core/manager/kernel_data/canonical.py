# Copyright Motphys Technology Co., Ltd. 2025, 2026

from typing import TypeVar, cast

import numpy as np
import torch

from motrix_env_core.manager.kernel_data.tree import (
    flatten_kernel_data,
    is_kernel_data,
    iter_tree_leaves,
    leaf_value_type,
    unflatten_kernel_data,
)

KernelDataType = TypeVar("KernelDataType")


def canonicalize_kernel_data(value: KernelDataType, *, context: str) -> KernelDataType:
    """Copy one kernel-data value into environment-owned backing arrays."""
    if not is_kernel_data(value):
        raise TypeError(f"{context} must return a concrete kernel-data dataclass, got {type(value).__name__}.")
    leaves, tree_def = flatten_kernel_data(value)
    for leaf_def in iter_tree_leaves(tree_def):
        if leaf_value_type(leaf_def.annotation) is None:
            raise TypeError(
                f"{context} KernelData {'.'.join(leaf_def.path)} expected a tensor, an ndarray, or a numeric scalar."
            )
    copied = tuple(
        leaf.clone() if isinstance(leaf, torch.Tensor) else leaf.copy() if isinstance(leaf, np.ndarray) else leaf
        for leaf in leaves
    )
    return cast(KernelDataType, unflatten_kernel_data(tree_def, copied))


__all__ = ["canonicalize_kernel_data"]
