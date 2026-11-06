# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar, cast

from motrix_env_core.manager.kernel_data.tree import (
    clone_tree,
    is_kernel_data,
    iter_leaves,
    leaf_value_type,
    tree_def_for,
)

KernelDataType = TypeVar("KernelDataType")


def canonicalize_kernel_data(value: KernelDataType, *, context: str) -> KernelDataType:
    """Copy one kernel-data value into environment-owned backing arrays."""
    if not is_kernel_data(value):
        raise TypeError(f"{context} must return a concrete kernel-data dataclass, got {type(value).__name__}.")
    tree_def = tree_def_for(value)
    for leaf_def, _leaf_value in iter_leaves(value, tree_def):
        if leaf_value_type(leaf_def.annotation) is None:
            raise TypeError(
                f"{context} KernelData {'.'.join(leaf_def.path)} expected a tensor, an ndarray, or a numeric scalar."
            )
    return cast(KernelDataType, clone_tree(value, tree_def))


__all__ = ["canonicalize_kernel_data"]
