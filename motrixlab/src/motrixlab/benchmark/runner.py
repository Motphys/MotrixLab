# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Perf scope profiling for benchmark cases, without CLI dependencies."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motrix_env_core.perf import Perf, PerfNode


@dataclass(frozen=True, slots=True)
class ProfileResult:
    elapsed_seconds: float
    root: PerfNode


def _profile_calls(
    perf: Perf,
    function: Callable[[], object],
    count: int,
    root_name: str,
    *,
    synchronize: Callable[[], object] | None = None,
) -> ProfileResult:
    was_enabled = perf.enabled
    perf.reset()
    if not was_enabled:
        perf.enable(synchronize=synchronize)
    started = time.perf_counter()
    try:
        for _ in range(count):
            function()
    finally:
        elapsed_seconds = time.perf_counter() - started
        if not was_enabled:
            perf.disable()
    roots = {root.name: root for root in perf.snapshot()}
    try:
        root = roots[root_name]
    except KeyError as error:
        raise RuntimeError(f"Perf did not record expected root scope {root_name!r}.") from error
    return ProfileResult(elapsed_seconds=elapsed_seconds, root=root)


def print_profile_tree(root: PerfNode) -> None:
    """Print the aggregate Perf scope tree for a measured benchmark case."""
    print("Perf values are aggregate means over production calls, not per-call medians.")
    print(f"{'scope':<38} | {'mean ms':>10} | {'self ms':>10} | {'share':>7} | {'calls/root':>10}")

    if root.count <= 0:
        print("(no recorded calls)")
        return

    def print_node(node: PerfNode, depth: int) -> None:
        mean_ms = node.total_ns / root.count / 1e6
        self_ms = node.self_ns / root.count / 1e6
        share = 0.0 if root.total_ns <= 0 else node.total_ns / root.total_ns * 100
        calls_per_root = node.count / root.count
        print(
            f"{'  ' * depth}{node.name:<38} | {mean_ms:>10.4f} | {self_ms:>10.4f} | "
            f"{share:>6.2f}% | {calls_per_root:>10.2f}"
        )
        for child in node.children:
            print_node(child, depth + 1)

    print_node(root, 0)
