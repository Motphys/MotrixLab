# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Fixed-size replay of simulator transitions laid out as (time, env).

State/context widths come from their assembled terms and actions from the task space; scalar bookkeeping
(frame index, episode step, done) is part of the provider contract.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl_supermimic.transitions import Transitions

__all__ = ["ReplayBuffer", "transition_schema"]


def transition_schema(*, state_dim: int, context_dim: int, action_dim: int) -> TransitionSchema:
    """Replay and transport fields derived from the assembled task dimensions."""
    return TransitionSchema(
        (
            TransitionField("s", (state_dim,), torch.float32),
            TransitionField("s1", (state_dim,), torch.float32),
            TransitionField("a", (action_dim,), torch.float32),
            TransitionField("a_prev", (action_dim,), torch.float32),
            TransitionField("xi", (context_dim,), torch.float32),
            TransitionField("r", (), torch.float32),
            TransitionField("f", (), torch.long),
            TransitionField("ep_step", (), torch.int32),
            TransitionField("done", (), torch.bool),
        )
    )


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str | torch.device = "cpu",
        *,
        state_dim: int,
        context_dim: int,
        action_dim: int,
    ) -> None:
        if T <= 0 or N <= 0:
            raise ValueError("Replay capacity and environment count must be positive")
        self.T, self.N, self.device = T, N, device
        self.data: dict[str, torch.Tensor] = {}
        schema = transition_schema(state_dim=state_dim, context_dim=context_dim, action_dim=action_dim)
        for field in schema.fields:
            self.data[field.name] = torch.zeros((T,) + schema.slot_shape(field, N), dtype=field.dtype, device=device)
        self.ptr, self.size = 0, 0

    @torch.no_grad()
    def add(self, transitions: Transitions) -> None:
        for name, value in transitions.to_dict().items():
            self.data[name][self.ptr] = value
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int) -> dict[str, torch.Tensor]:
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

    def sample_starts(
        self,
        B: int,
        hist: int,
        *,
        valid_frames: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Select valid starts before gathering their oldest-first observation histories.

        ``valid_frames`` restricts candidate reference frames to those supporting
        the task's imagination horizon. History continuity and this predicate
        must both hold; sampling with replacement returns exactly ``B`` starts.
        """
        if B <= 0 or hist <= 0 or self.size == 0:
            raise ValueError("Need a nonempty replay and positive batch/history sizes")
        selected_ages, selected_envs = [], []
        count = 0
        # Keep retry kernels full-sized rather than producing small GPU tail
        # batches. This is the only rejection loop, so a predicate uses 2B, not 4B.
        candidates = B if valid_frames is None else 2 * B
        for _ in range(100):
            age = torch.randint(0, self.size, (candidates,), device=self.device)
            env = torch.randint(0, self.N, (candidates,), device=self.device)
            rows = self._row(age)
            steps = self.data["ep_step"][rows, env].long()
            required = steps.clamp(max=hist - 1)
            valid = age >= required
            # Only inspect sampled candidates; adjacent physical rows alone do
            # not establish logical continuity after a collector discontinuity.
            for offset in range(1, hist):
                previous = self._row((age - offset).clamp(min=0))
                continuous = (self.data["ep_step"][previous, env] == steps - offset) & ~self.data["done"][previous, env]
                valid &= (required < offset) | continuous
            if valid_frames is not None:
                valid &= valid_frames(self.data["f"][rows, env])
            ids = valid.nonzero().flatten()[: B - count]
            selected_ages.append(age[ids])
            selected_envs.append(env[ids])
            count += ids.numel()
            if count == B:
                break
        if count != B:
            raise RuntimeError(
                f"Replay found {count}/{B} valid imagination starts after 100 attempts; "
                "require a complete observation history prefix and valid future frames"
            )
        age, env = torch.cat(selected_ages), torch.cat(selected_envs)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        positions = torch.arange(hist, device=self.device)[None].expand(B, -1)
        index = torch.maximum(positions, (hist - 1 - ep_step).clamp(min=0)[:, None])
        back = hist - 1 - index
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f")}
        hist_d["index"] = index
        return start, hist_d
