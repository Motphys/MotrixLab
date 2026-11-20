# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned replay buffers, populated by configured numerical terms."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motrix_env_core.array.env import ArrayEnvState


@dataclass(frozen=True, slots=True)
class TransitionRecorder:
    """Saved endpoints survive auto-reset; consumers snapshot before the next step."""

    s: torch.Tensor
    s1: torch.Tensor
    frame: torch.Tensor
    action: torch.Tensor
    previous: torch.Tensor
    context: torch.Tensor
    episode_step: torch.Tensor

    @classmethod
    def create(
        cls,
        num_envs: int,
        device: torch.device | str = "cpu",
        *,
        state_dim: int,
        context_dim: int,
        action_dim: int,
    ) -> TransitionRecorder:
        return cls(
            s=torch.zeros(num_envs, state_dim, device=device),
            s1=torch.zeros(num_envs, state_dim, device=device),
            frame=torch.zeros(num_envs, dtype=torch.long, device=device),
            action=torch.zeros(num_envs, action_dim, device=device),
            previous=torch.zeros(num_envs, action_dim, device=device),
            context=torch.zeros(num_envs, context_dim, device=device),
            episode_step=torch.zeros(num_envs, dtype=torch.int32, device=device),
        )

    def snapshot(self, result: ArrayEnvState) -> dict[str, torch.Tensor]:
        """Owned replay tensors including final framework reward/done outputs."""
        return {
            "s": self.s.detach().clone(),
            "s1": self.s1.detach().clone(),
            "f": self.frame.clone(),
            "a": self.action.detach().clone(),
            "a_prev": self.previous.detach().clone(),
            "xi": self.context.detach().clone(),
            "ep_step": self.episode_step.clone(),
            "r": result.reward.detach().clone(),
            "done": (result.terminated | result.truncated).clone(),
        }
