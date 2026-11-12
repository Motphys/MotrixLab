# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Fixed-size replay of simulator transitions laid out as (time, env).

Fields come from the task's ``StateSpec``; extra scalar bookkeeping
(frame index, episode step, done) is part of the provider contract.
"""

from __future__ import annotations

import torch

from motrix_rl_supermimic.task_mdp import StateSpec

__all__ = ["ReplayBuffer", "transition_fields"]


def transition_fields(spec: StateSpec) -> dict[str, tuple[int, torch.dtype]]:
    n_j = (spec.state_dim - 13) // 2
    return {
        "s": (spec.state_dim, torch.float32),
        "s1": (spec.state_dim, torch.float32),
        "a": (spec.action_dim, torch.float32),
        "a_prev": (spec.action_dim, torch.float32),
        "xi": (spec.context_dim, torch.float32),
        "r": ((), torch.float32),
        "f": ((), torch.long),
        "ep_step": ((), torch.int32),
        "done": ((), torch.bool),
    }


class ReplayBuffer:
    def __init__(self, spec: StateSpec, T: int, N: int, device: str | torch.device = "cpu"):
        if T <= 0 or N <= 0:
            raise ValueError("Replay capacity and environment count must be positive")
        self.spec, self.T, self.N, self.device = spec, T, N, device
        self.data = {}
        for k, (shape, dt) in transition_fields(spec).items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    @torch.no_grad()
    def add(self, tr: dict):
        for k in self.data:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        if B <= 0 or k <= 0 or k > self.size:
            raise ValueError("Need B > 0 and 1 <= k <= the stored replay size")
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int):
        """Start points for imagination plus the indices of their observation history (oldest first)."""
        if B <= 0 or hist <= 0 or self.size == 0:
            raise ValueError("Need a nonempty replay and positive batch/history sizes")
        age = torch.randint(0, self.size, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        return start, hist_d
