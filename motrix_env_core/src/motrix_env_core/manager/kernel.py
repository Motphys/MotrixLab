# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar, cast

import numpy as np

from motrix_env_core.manager.kernel_data.resolved import ResolvedLeaf, ResolvedTree

T = TypeVar("T")


class KernelOutputs(NamedTuple):
    """Preallocated arrays written by one manager task kernel."""

    policy_obs: np.ndarray
    value_obs: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray


@dataclass(frozen=True)
class ManagerWarmupResult:
    """Compilation and first-execution diagnostics produced by :meth:`ManagerEnv.warmup`."""

    compile_seconds: float
    signatures: tuple[str, ...]
    threading_layer: str | None
    num_threads: int
    term_compile_seconds: float = 0.0
    kernel_compile_seconds: float = 0.0
    first_execution_seconds: float = 0.0


def validate_kernel_context(
    trees: Any,
    reward_weights: np.ndarray,
    buffers: Any,
    outputs: KernelOutputs,
    num_envs: int,
) -> None:
    """Validate the numeric kernel contract once when a context is bound.

    ``trees`` is the resolved program input: one :class:`ResolvedTree` per
    prepared kernel-data source. Leaf values are checked for dtype and type
    sanity without any batch-dimension assumptions.
    """
    if not isinstance(trees, tuple):
        raise TypeError(f"inputs must be a tuple of resolved trees, got {type(trees).__name__}.")
    for tree in trees:
        for path, leaf in tree.leaves.items():
            _validate_value(leaf.value, f"inputs.{'.'.join(path)}", num_envs=num_envs, batched=False, writable=False)
    _validate_value(reward_weights, "reward_weights", num_envs=num_envs, batched=False, writable=False)
    _validate_value(buffers, "buffers", num_envs=num_envs, batched=True, writable=True)
    _validate_outputs(outputs, num_envs)


def clone_resolved_trees(trees: tuple[ResolvedTree, ...]) -> tuple[ResolvedTree, ...]:
    """Clone every leaf of the resolved trees for isolated warmup execution.

    Leaves are cloned with :func:`clone_kernel_value` semantics (dtype,
    strides, and read-only flag preserved); the host root objects stay shared
    exactly as in real execution, where namespace fallback reads immutable
    config attributes from the original records.
    """
    return tuple(
        ResolvedTree(
            tree.value,
            {path: ResolvedLeaf(clone_kernel_value(leaf.value)) for path, leaf in tree.leaves.items()},
        )
        for tree in trees
    )


def clone_kernel_value(value: T) -> T:
    """Recursively clone arrays while preserving task-specific tuple types."""
    if isinstance(value, np.ndarray):
        # Warmup must use the same Numba array type as real execution: copy()
        # normalizes strided views to C layout and drops the readonly flag.
        lower = (
            sum(min(0, (size - 1) * stride) for size, stride in zip(value.shape, value.strides)) if value.size else 0
        )
        upper = (
            sum(max(0, (size - 1) * stride) for size, stride in zip(value.shape, value.strides)) if value.size else 0
        )
        storage = np.empty(max(value.itemsize, upper - lower + value.itemsize), dtype=np.uint8)
        copied = np.ndarray(value.shape, dtype=value.dtype, buffer=storage, offset=-lower, strides=value.strides)
        copied[...] = value
        copied.flags.writeable = value.flags.writeable
        return cast(T, copied)
    if isinstance(value, tuple):
        items = [clone_kernel_value(item) for item in value]
        if hasattr(type(value), "_fields"):
            return cast(T, type(value)(*items))
        return cast(T, tuple(items))
    return value


def _validate_value(value: Any, path: str, *, num_envs: int, batched: bool, writable: bool) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{path} must not use object dtype, got {value.dtype}.")
        if batched and (value.ndim == 0 or value.shape[0] != num_envs):
            raise ValueError(f"{path} must have batch dimension {num_envs}, got shape {value.shape}.")
        if writable and not value.flags.writeable:
            raise ValueError(f"{path} must be writable.")
        if writable and not value.flags.c_contiguous:
            raise ValueError(f"{path} must be C-contiguous, got shape {value.shape} and strides {value.strides}.")
        return

    if isinstance(value, tuple):
        field_names = getattr(type(value), "_fields", None)
        for index, item in enumerate(value):
            field = field_names[index] if field_names is not None else str(index)
            _validate_value(
                item,
                f"{path}.{field}",
                num_envs=num_envs,
                batched=batched,
                writable=writable,
            )
        return

    if isinstance(value, (bool, int, float, np.generic)):
        return

    raise TypeError(
        f"{path} must contain only ndarrays, numeric scalars, and fixed tuples; got {type(value).__name__}."
    )


def _validate_outputs(outputs: KernelOutputs, num_envs: int) -> None:
    _validate_value(outputs, "outputs", num_envs=num_envs, batched=True, writable=True)
    if outputs.reward.shape != (num_envs,):
        raise ValueError(f"outputs.reward must have shape ({num_envs},), got {outputs.reward.shape}.")
    if outputs.terminated.shape != (num_envs,):
        raise ValueError(f"outputs.terminated must have shape ({num_envs},), got {outputs.terminated.shape}.")
    if outputs.terminated.dtype != np.bool_:
        raise TypeError(f"outputs.terminated must use bool dtype, got {outputs.terminated.dtype}.")
