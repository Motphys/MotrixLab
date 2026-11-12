# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Transition schema for the async transport, derived from the task spec.

Replay fields (from :mod:`motrix_rl_supermimic.replay`) plus the policy
observation and flattened goal, which the learner needs for normalizer
statistics. Curriculum payloads are ordinary schema fields when a task
declares them — no extra mechanism.
"""

from __future__ import annotations

from typing import Protocol

import torch

from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl_supermimic.replay import transition_fields
from motrix_rl_supermimic.task_mdp import StateSpec, optional_task_hook


class TaskMDPLike(Protocol):
    """Minimal structural view of a TaskMDP for schema derivation."""

    spec: StateSpec

__all__ = ["transition_schema"]


def transition_schema(spec: StateSpec, goal_frames: int, task_mdp: TaskMDPLike | None = None) -> TransitionSchema:
    fields = [
        TransitionField(name, (shape,) if isinstance(shape, int) else (), dtype)
        for name, (shape, dtype) in transition_fields(spec).items()
    ]
    fields.append(TransitionField("obs", (spec.observation_dim,), torch.float32))
    fields.append(TransitionField("goal", (spec.command_dim * goal_frames,), torch.float32))
    # Task-declared transport buffers (e.g. curriculum state) ride in the
    # schema so collector and learner share one definition.
    transport_hook = optional_task_hook(task_mdp, "transport_fields") if task_mdp is not None else None
    if transport_hook is not None:
        for name, buffer in transport_hook().items():
            fields.append(TransitionField(name, tuple(buffer.shape), buffer.dtype))
    return TransitionSchema(tuple(fields))
