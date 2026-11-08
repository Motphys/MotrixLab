# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC's declaration of its transition field set.

The async FastSAC trainer consumes six fields per env-step batch. The
``next_obs``/``next_critic_obs`` tensors are deliberately absent: with
auto-reset envs the observation returned by step ``t`` is exactly the stored
observation of step ``t+1`` (at episode ends it is the reset observation), so
the consumer's replay buffer derives them from the successive slot — see the
final-observation store for the terminating-step exception.
"""

from __future__ import annotations

import torch

from motrix_rl.transport.schema import TransitionField, TransitionSchema


def fastsac_transition_schema(obs_dim: int, critic_obs_dim: int, act_dim: int) -> TransitionSchema:
    """Build the FastSAC transition schema for one observation/action geometry."""
    return TransitionSchema(
        fields=(
            TransitionField("obs", (obs_dim,), torch.float32),
            TransitionField("critic_obs", (critic_obs_dim,), torch.float32),
            TransitionField("actions", (act_dim,), torch.float32),
            TransitionField("rewards", (), torch.float32),
            TransitionField("dones", (), torch.int64),
            TransitionField("truncations", (), torch.int64),
        )
    )
