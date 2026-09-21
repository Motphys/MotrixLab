# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Replay buffer and observation normalizer for FastSAC.

Ported from holosoma's ``fast_sac_utils`` with asymmetric observations: each
transition stores both the actor observation and the (privileged) critic
observation.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleReplayBuffer(nn.Module):
    """Per-environment circular replay buffer with n-step returns and
    asymmetric (actor / critic) observations.

    Observations are stored once per time step in a ``(n_env, buffer_size + 1,
    dim)`` sequence: slot ``t % (buffer_size + 1)`` holds the observation of
    step ``t`` and doubles as the next-observation of step ``t - 1``. With
    auto-reset envs the observation returned by step ``t`` is exactly the
    observation stored for step ``t + 1`` (at episode ends it is the reset
    observation), the invariant already relied upon by the async transition
    ring, so a dedicated ``next_observations`` copy would be pure redundancy
    and doubles GPU memory for high-dimensional observation tasks.
    """

    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        super().__init__()
        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        # Observation rings use one extra slot so the next-observation of the
        # newest transition never aliases its own observation slot.
        cap = buffer_size + 1
        self._cap = cap
        z = lambda d: torch.zeros((n_env, cap, d), device=device, dtype=torch.float)  # noqa: E731
        self.observations = z(n_obs)
        self.actions = z(n_act)
        self.rewards = torch.zeros((n_env, cap), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.critic_observations = z(n_critic_obs)
        self.ptr = 0

    @property
    def num_stored(self) -> int:
        return min(self.ptr, self.buffer_size)

    def extend(self, obs, critic_obs, actions, rewards, dones, truncations, next_obs, next_critic_obs) -> None:
        """Append one transition per environment.

        ``next_obs``/``next_critic_obs`` must equal the ``obs``/``critic_obs``
        of the following step (auto-reset semantics); the buffer stores them at
        slot ``ptr + 1`` and the next call's data lands there, so per step only
        one observation write per field is performed.
        """
        ptr = self.ptr
        slot = ptr % self._cap
        if ptr == 0:
            self.observations[:, 0] = obs
            self.critic_observations[:, 0] = critic_obs
        next_slot = (ptr + 1) % self._cap
        self.observations[:, next_slot] = next_obs
        self.critic_observations[:, next_slot] = next_critic_obs
        self.actions[:, slot] = actions
        self.rewards[:, slot] = rewards
        self.dones[:, slot] = dones
        self.truncations[:, slot] = truncations
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict:
        if self.ptr == 0:
            raise RuntimeError("cannot sample from an empty replay buffer")
        n_env, no, na, nco = self.n_env, self.n_obs, self.n_act, self.n_critic_obs
        flat = n_env * batch_size

        if self.n_steps == 1:
            idx = torch.randint(0, self.num_stored, (n_env, batch_size), device=self.device)
            # Sampled ring positions are expressed as absolute time steps over
            # the contiguous valid window, then folded into the ring. The next
            # observation of step ``t`` is the observation stored at ``t + 1``.
            t = self.ptr - self.num_stored + idx
            ring = t % self._cap
            next_ring = (t + 1) % self._cap
            oi = ring.unsqueeze(-1).expand(-1, -1, no)
            ni = next_ring.unsqueeze(-1).expand(-1, -1, no)
            ai = ring.unsqueeze(-1).expand(-1, -1, na)
            ci = ring.unsqueeze(-1).expand(-1, -1, nco)
            nci = next_ring.unsqueeze(-1).expand(-1, -1, nco)
            out = {
                "obs": torch.gather(self.observations, 1, oi).reshape(flat, no),
                "next_obs": torch.gather(self.observations, 1, ni).reshape(flat, no),
                "critic_obs": torch.gather(self.critic_observations, 1, ci).reshape(flat, nco),
                "next_critic_obs": torch.gather(self.critic_observations, 1, nci).reshape(flat, nco),
                "actions": torch.gather(self.actions, 1, ai).reshape(flat, na),
                "rewards": torch.gather(self.rewards, 1, ring).reshape(flat),
                "dones": torch.gather(self.dones, 1, ring).reshape(flat),
                "truncations": torch.gather(self.truncations, 1, ring).reshape(flat),
            }
            out["effective_n_steps"] = torch.ones_like(out["dones"])
            return out

        # n-step (>1)
        if self.ptr < self.buffer_size:
            max_start = max(1, self.ptr - self.n_steps + 1)
            idx = torch.randint(0, max_start, (n_env, batch_size), device=self.device)
            t = idx
        else:
            idx = torch.randint(0, self.buffer_size, (n_env, batch_size), device=self.device)
            t = self.ptr - self.buffer_size + idx
        ring = t % self._cap

        oi = ring.unsqueeze(-1).expand(-1, -1, no)
        ai = ring.unsqueeze(-1).expand(-1, -1, na)
        ci = ring.unsqueeze(-1).expand(-1, -1, nco)
        obs = torch.gather(self.observations, 1, oi).reshape(flat, no)
        actions = torch.gather(self.actions, 1, ai).reshape(flat, na)
        critic_obs = torch.gather(self.critic_observations, 1, ci).reshape(flat, nco)

        offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
        all_idx = (t.unsqueeze(-1) + offsets) % self._cap
        all_rew = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_idx)
        all_done = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_idx)
        all_trunc = torch.gather(self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_idx)

        # A window must not read past the newest stored transition; folded ring
        # positions beyond it hold stale data from a full cycle earlier.
        max_off = (self.ptr - 1 - t).clamp(min=0)
        in_win = offsets <= max_off.unsqueeze(-1)
        all_rew = all_rew * in_win
        all_done = all_done * in_win
        all_trunc = all_trunc * in_win

        done_shift = torch.cat([torch.zeros_like(all_done[:, :, :1]), all_done[:, :, :-1]], dim=2)
        done_mask = torch.cumprod(1.0 - done_shift, dim=2) * in_win
        # Keep the same integer dtype as the 1-step branch's ones_like(dones).
        eff_n = done_mask.sum(2).long()
        discounts = torch.pow(self.gamma, torch.arange(self.n_steps, device=self.device))
        n_step_rew = (all_rew * done_mask * discounts.view(1, 1, -1)).sum(dim=2)

        first_done = torch.argmax((all_done > 0).float(), dim=2)
        first_trunc = torch.argmax((all_trunc > 0).float(), dim=2)
        first_done = torch.where(all_done.sum(2) == 0, self.n_steps - 1, first_done)
        first_trunc = torch.where(all_trunc.sum(2) == 0, self.n_steps - 1, first_trunc)
        final = torch.minimum(torch.minimum(first_done, first_trunc), max_off)
        # The next observation of the final transition lives one slot ahead.
        final_next = (t + final + 1) % self._cap
        final_ring = (t + final) % self._cap

        next_obs = self.observations.gather(1, final_next.unsqueeze(-1).expand(-1, -1, no)).reshape(flat, no)
        next_critic_obs = self.critic_observations.gather(1, final_next.unsqueeze(-1).expand(-1, -1, nco)).reshape(
            flat, nco
        )
        dones = self.dones.gather(1, final_ring).reshape(flat)
        truncations = self.truncations.gather(1, final_ring).reshape(flat)

        return {
            "obs": obs,
            "next_obs": next_obs,
            "critic_obs": critic_obs,
            "next_critic_obs": next_critic_obs,
            "actions": actions,
            "rewards": n_step_rew.reshape(flat),
            "dones": dones,
            "truncations": truncations,
            "effective_n_steps": eff_n.reshape(flat),
        }


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        big_m2 = m_a + m_b + delta2.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(big_m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)
