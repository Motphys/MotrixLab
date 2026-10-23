# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Single-threaded collect -> learn -> publish -> observer scheduling.

Callers own resources (use context managers/try-finally). Unlike AsyncExecution,
callbacks may share mutable state: no collection overlaps learning or on_step.
"""

from collections.abc import Callable
from typing import Any

from motrix_rl.execution.contracts import Cancellation, Parameters, Progress, Sample


class SyncExecution:
    def run(
        self,
        collect: Callable[[Parameters, Cancellation], Any],
        learn: Callable[[Sample, Cancellation], Parameters | None],
        *,
        initial_parameters: Parameters,
        max_iterations: int,
        cancellation: Cancellation | None = None,
        on_step: Callable[[Progress], None] | None = None,
    ) -> Progress:
        """Run exactly max_iterations handoffs unless cancelled or a callback fails.

        Returning None from learn means no publication (warmup/rejected data).
        Algorithms decide how many optimizer updates, if any, each handoff uses.
        on_step is the safe checkpoint/logging boundary after parameter publish.
        """
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        cancellation = cancellation if cancellation is not None else Cancellation()
        parameters = initial_parameters
        progress = Progress(policy_version=parameters.version)
        cancellation.raise_if_cancelled()
        for sequence in range(max_iterations):
            cancellation.raise_if_cancelled()
            sample = Sample(collect(parameters, cancellation), parameters.version, sequence)
            cancellation.raise_if_cancelled()
            published = learn(sample, cancellation)
            cancellation.raise_if_cancelled()
            if published is not None:
                if published.version <= parameters.version:
                    raise ValueError("Published parameter versions must increase")
                parameters = published
            progress = Progress(
                iterations=sequence + 1,
                samples=sequence + 1,
                updates=progress.updates + (published is not None),
                policy_version=parameters.version,
            )
            if on_step is not None:
                on_step(progress)
            cancellation.raise_if_cancelled()
        return progress
