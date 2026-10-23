# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Execution metadata, independent of algorithm payloads and run persistence.

Providers retain TrainerContext and own serialization, logging and checkpoints.
An executor's on_step callback runs at a learner boundary, never during an update.
Payload ownership transfers to the learner; parameter snapshots must be immutable
(or detached copies), including tensors nested inside them.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any


class ExecutionCancelled(Exception):
    """Cooperative cancellation; adapters must not mark a cancelled run successful."""


class Cancellation:
    """Thread-safe stop request shared with algorithm callbacks."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ExecutionCancelled("Training execution cancelled")


@dataclass(frozen=True)
class Parameters:
    """Algorithm-owned snapshot and publication version.

    Frozen metadata does not freeze ``value`` recursively. The adapter must
    detach/copy mutable model state before publishing; execution never clones
    arbitrary payloads or assumes they are Torch tensors.
    """

    value: Any
    version: int = 0


@dataclass(frozen=True)
class Sample:
    """Transferred algorithm data tagged with the policy used to collect it."""

    value: Any
    policy_version: int
    sequence: int


@dataclass(frozen=True)
class Progress:
    """Executor counters, not environment steps or algorithm optimizer steps.

    iterations counts completed collect/learn handoffs; updates counts published
    snapshots. Providers map these to their own run/checkpoint/logging axes.
    """

    iterations: int = 0
    samples: int = 0
    updates: int = 0
    policy_version: int = 0
