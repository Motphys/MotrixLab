# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Resolved tree values consumed by manager executors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from motrix_env_core.manager.kernel_data.lowering import KernelDataScope


@dataclass(frozen=True)
class ResolvedLeaf:
    """One resolved leaf of a kernel-data tree.

    Attributes:
        value: Live leaf value reference — an ndarray held and refreshed in
            place by its owning runtime object, or an immutable scalar.
        scope: Whether the leaf is an ``(N, ...)`` batched array or shared.
    """

    value: Any
    scope: KernelDataScope


@dataclass(frozen=True)
class ResolvedTree:
    """Path-keyed resolution of one kernel-data root value.

    ``value`` keeps the host root object (used as the attribute fallback for
    executor namespace views); ``leaves`` maps every leaf path to its live
    value reference and scope. Array leaves are rewritten in place by their
    owning runtime objects, so the tree itself never needs rebuilding.
    """

    value: Any
    leaves: Mapping[tuple[str, ...], ResolvedLeaf]


__all__ = ["ResolvedLeaf", "ResolvedTree"]
