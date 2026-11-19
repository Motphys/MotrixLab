# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Algorithm transition schema shared by replay and collector transport."""

from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl_supermimic.replay import transition_fields
from motrix_rl_supermimic.task_mdp import StateSpec

__all__ = ["transition_schema"]


def transition_schema(spec: StateSpec, *, action_dim: int) -> TransitionSchema:
    """Transport state fields plus the assembled Manager action width."""
    return TransitionSchema(
        tuple(
            TransitionField(name, shape, dtype)
            for name, (shape, dtype) in transition_fields(spec, action_dim=action_dim).items()
        )
    )
