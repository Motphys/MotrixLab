# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple, TypeVar

import numpy as np
import torch

T = TypeVar("T")


class KernelOutputs(NamedTuple):
    """Preallocated tensors written by one manager task kernel."""

    policy_obs: torch.Tensor
    value_obs: torch.Tensor | None
    reward: torch.Tensor
    terminated: torch.Tensor


def _validate_value(value: Any, path: str, *, num_envs: int, batched: bool, writable: bool) -> None:
    if isinstance(value, torch.Tensor):
        if batched and (value.dim() == 0 or value.shape[0] != num_envs):
            raise ValueError(f"{path} must have batch dimension {num_envs}, got shape {tuple(value.shape)}.")
        return

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
        f"{path} must contain only tensors, ndarrays, numeric scalars, and fixed tuples; got {type(value).__name__}."
    )


def _validate_outputs(outputs: KernelOutputs, num_envs: int) -> None:
    _validate_value(outputs, "outputs", num_envs=num_envs, batched=True, writable=True)
    if tuple(outputs.reward.shape) != (num_envs,):
        raise ValueError(f"outputs.reward must have shape ({num_envs},), got {tuple(outputs.reward.shape)}.")
    if tuple(outputs.terminated.shape) != (num_envs,):
        raise ValueError(f"outputs.terminated must have shape ({num_envs},), got {tuple(outputs.terminated.shape)}.")
    if outputs.terminated.dtype is not torch.bool:
        raise TypeError(f"outputs.terminated must use bool dtype, got {outputs.terminated.dtype}.")


def validate_kernel_context(
    trees: Any,
    reward_weights: torch.Tensor,
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
