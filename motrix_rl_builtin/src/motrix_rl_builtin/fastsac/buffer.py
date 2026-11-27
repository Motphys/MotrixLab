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
        final_obs_bootstrap: bool = False,
        policy_history: int = 1,
        value_history: int = 1,
        shared_observations: bool = False,
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
        self.policy_history = policy_history
        self.value_history = value_history
        self._history = max(policy_history, value_history)
        self._shared_observations = shared_observations
        self._episode_start = torch.zeros(n_env, device=device, dtype=torch.long)

        # All fields share the same capacity so one absolute-time indexing
        # scheme (t % cap) covers them; the extra slot over buffer_size keeps
        # the newest transition's next-observation slot distinct from its own.
        # The scalar fields' extra slot is negligible and NOT independently
        # shrinkable — a different modulus would break the shared indexing.
        cap = buffer_size + 1
        self._cap = cap
        z = lambda d: torch.zeros((n_env, cap, d), device=device, dtype=torch.float)  # noqa: E731
        self.observations = z(n_obs)
        self.actions = z(n_act)
        self.rewards = torch.zeros((n_env, cap), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.truncations = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.critic_observations = self.observations if shared_observations else z(n_critic_obs)
        self.episode_starts = torch.zeros((n_env, cap), device=device, dtype=torch.long)
        self.ptr = 0

        # Final rows are allocated only for live truncated transitions. Capacity
        # grows geometrically, bounded by the replay's transition capacity; no
        # assumption about time between truncations is required. Allocation and
        # ownership bookkeeping are eager ingest work, outside compiled updates.
        self._final_obs_table = None
        self._final_critic_table = None
        self._final_slot = None
        self._final_capacity = 0
        self._final_free: list[int] = []
        self._final_owners: list[list[int]] = [[] for _ in range(cap)]
        if final_obs_bootstrap:
            self._enable_final_storage()

    def _enable_final_storage(self) -> None:
        self._final_slot = torch.full((self.n_env, self._cap), -1, device=self.device, dtype=torch.int32)
        self._grow_final_storage(self.n_env)

    def _grow_final_storage(self, needed: int) -> None:
        old = self._final_capacity
        capacity = min(self.n_env * self._cap, max(self.n_env, old * 2, old + needed))
        # Tables may be referenced by work on the compute and ingest streams.
        # Growth is rare and eager; finish those uses before replacing backing.
        if self.observations.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        policy = torch.empty((capacity, self.n_obs), device=self.device)
        critic = policy if self._shared_observations else torch.empty((capacity, self.n_critic_obs), device=self.device)
        if old:
            policy[:old].copy_(self._final_obs_table)
            critic[:old].copy_(self._final_critic_table)
        else:
            # Samples without a final row still gather index zero before masking.
            policy[0].zero_()
            critic[0].zero_()
        self._final_obs_table = policy
        self._final_critic_table = critic
        self._final_free.extend(range(capacity - 1, old - 1, -1))
        self._final_capacity = capacity

    @property
    def num_stored(self) -> int:
        # A transition is only complete once the following step's observation
        # has been written (its next observation), so ingestion runs one step
        # ahead of what is sampleable.
        return min(max(self.ptr - 1, 0), self.buffer_size)

    def extend(self, obs, critic_obs, actions, rewards, dones, truncations) -> None:
        """Append one transition per environment.

        Only the observation of step ``ptr`` is written. Its next-observation
        slot (``ptr + 1``) is filled by the following call — with auto-reset
        envs the observation returned by step ``t`` is exactly the observation
        of step ``t + 1`` (at episode ends it is the reset observation), the
        invariant already relied upon by the async transition ring.
        """
        slot = self.ptr % self._cap
        self._invalidate_final_slot(slot, 1)
        self.observations[:, slot] = obs
        if not self._shared_observations:
            self.critic_observations[:, slot] = critic_obs
        self.actions[:, slot] = actions
        self.rewards[:, slot] = rewards
        self.dones[:, slot] = dones
        self.truncations[:, slot] = truncations
        self.episode_starts[:, slot] = self._episode_start
        self._episode_start = torch.where((dones | truncations).bool(), self.ptr + 1, self._episode_start)
        self.ptr += 1

    def extend_batch(self, obs, critic_obs, actions, rewards, dones, truncations, *, final_payloads=None) -> None:
        """Append ``k`` transitions per environment from slot-major tensors.

        Inputs are shaped ``(k, n_env, dim)`` for obs / critic_obs / actions
        and ``(k, n_env)`` for the scalar fields — the contiguous multi-slot
        layout produced by the async transition ring's :meth:`read_span`.
        Writes go directly into the strided buffer slices (one copy per field
        per contiguous chunk instead of one per slot) on the caller's current
        stream, so an async ingest can issue non-blocking pinned H2D copies on
        a side stream.
        """
        k = obs.shape[0]
        if k > self._cap:
            # Preserve only the newest replay-capacity transitions, but keep
            # absolute time progression identical to successive extends.
            skip = k - self._cap
            for offset in range(skip):
                self._episode_start = torch.where(
                    (dones[offset].bool() | truncations[offset].bool()).to(self.device),
                    self.ptr + offset + 1,
                    self._episode_start,
                )
            self.ptr += skip
            self.extend_batch(
                *(v[skip:] for v in (obs, critic_obs, actions, rewards, dones, truncations)),
                final_payloads=None if final_payloads is None else final_payloads[skip:],
            )
            return
        start = self.ptr % self._cap
        self._invalidate_final_slot(start, k)
        n = min(k, self._cap - start)
        self._write_span(start, (t[:n] for t in (obs, critic_obs, actions, rewards, dones, truncations)))
        if k > n:
            self._write_span(0, (t[n:] for t in (obs, critic_obs, actions, rewards, dones, truncations)))
        # Track real episode starts, including a prefix whose first frame has
        # since been overwritten. An overwrite must never become repeat-fill.
        for offset in range(k):
            self.episode_starts[:, (self.ptr + offset) % self._cap] = self._episode_start
            self._episode_start = torch.where(
                (
                    self.dones[:, (self.ptr + offset) % self._cap]
                    | self.truncations[:, (self.ptr + offset) % self._cap]
                ).bool(),
                self.ptr + offset + 1,
                self._episode_start,
            )
        self.ptr += k
        if final_payloads is not None:
            for offset, payload in enumerate(final_payloads):
                if payload is not None:
                    self.write_final(payload.env_ids, self.ptr - k + offset, payload.obs, payload.critic_obs)

    def _invalidate_final_slot(self, start: int, count: int) -> None:
        """Clear the final-obs grid for slots about to be overwritten.

        The transitions being written carry no terminal observation (yet — a
        ``write_final`` for exactly these slots may follow the ingest); stale
        pointers from a full ring cycle earlier must not survive."""
        if self._final_slot is None:
            return
        for offset in range(count):
            owners = self._final_owners[(start + offset) % self._cap]
            self._final_free.extend(owners)
            owners.clear()
        n = min(count, self._cap - start)
        self._final_slot[:, start : start + n] = -1
        if count > n:
            self._final_slot[:, : count - n] = -1

    @torch.no_grad()
    def write_final(self, env_ids: torch.Tensor, t_abs: int, obs_rows: torch.Tensor, critic_rows: torch.Tensor) -> None:
        """Attach owned final rows to an already-ingested transition.

        Host allocation tracks live references, not episode frequency. Device
        tables are read by eager sampling, outside compiled gradient updates.
        """
        if not env_ids.numel():
            return
        slot = t_abs % self._cap
        owners = self._final_owners[slot]
        device_ids = env_ids.to(device=self.device, dtype=torch.long)
        # Normal ingest supplies every final for a slot together. A subsequent
        # write to the same slot reuses rows already assigned to those envs.
        new_ids = device_ids
        if owners:
            new_ids = device_ids[self._final_slot[device_ids, slot] < 0]
        count = new_ids.numel()
        if len(self._final_free) < count:
            self._grow_final_storage(count - len(self._final_free))
        rows = [self._final_free.pop() for _ in range(count)]
        owners.extend(rows)
        allocated = torch.tensor(rows, device=self.device, dtype=torch.int32)
        self._final_slot[new_ids, slot] = allocated
        row_ids = self._final_slot[device_ids, slot].long()
        self._final_obs_table[row_ids] = obs_rows.to(self.device, torch.float)
        self._final_critic_table[row_ids] = critic_rows.to(self.device, torch.float)

    def _write_span(self, slot: int, fields) -> None:
        obs, critic_obs, actions, rewards, dones, truncations = fields
        n = obs.shape[0]
        end = slot + n
        self.observations[:, slot:end].copy_(obs.permute(1, 0, 2), non_blocking=True)
        if not self._shared_observations:
            self.critic_observations[:, slot:end].copy_(critic_obs.permute(1, 0, 2), non_blocking=True)
        self.actions[:, slot:end].copy_(actions.permute(1, 0, 2), non_blocking=True)
        self.rewards[:, slot:end].copy_(rewards.permute(1, 0), non_blocking=True)
        self.dones[:, slot:end].copy_(dones.permute(1, 0), non_blocking=True)
        self.truncations[:, slot:end].copy_(truncations.permute(1, 0), non_blocking=True)

    def _window(self, frames: torch.Tensor, times: torch.Tensor, length: int) -> torch.Tensor:
        starts = self.episode_starts.gather(1, times % self._cap)
        offsets = torch.arange(1 - length, 1, device=self.device)
        indices = torch.maximum(times.unsqueeze(-1) + offsets, starts.unsqueeze(-1)) % self._cap
        lanes = torch.arange(self.n_env, device=self.device)[:, None, None]
        return frames[lanes, indices].flatten(2)

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict:
        if self.num_stored == 0:
            raise RuntimeError("cannot sample from a replay buffer with no complete transitions")
        # Keep the default single-frame, one-step workload on a compact gather
        # path. History metadata and the generic n-step state machine are only
        # needed for the replay modes that actually use them.
        if self._history == 1 and self.n_steps == 1:
            idx = torch.randint(0, self.num_stored, (self.n_env, batch_size), device=self.device)
            t = self.ptr - 1 - self.num_stored + idx
            ring = t % self._cap
            next_ring = (t + 1) % self._cap
            lanes = torch.arange(self.n_env, device=self.device)[:, None]
            obs = self.observations[lanes, ring]
            next_obs = self.observations[lanes, next_ring]
            truncations = self.truncations[lanes, ring]
            have_finals = len(self._final_free) < self._final_capacity
            if have_finals:
                final_ids = self._final_slot[lanes, ring].long()
                has_final = truncations.bool() & (final_ids >= 0)
                final_ids.clamp_min_(0)
                next_obs = torch.where(has_final[..., None], self._final_obs_table[final_ids], next_obs)
            if self._shared_observations:
                critic, next_critic = obs, next_obs
            else:
                critic = self.critic_observations[lanes, ring]
                next_critic = self.critic_observations[lanes, next_ring]
                if have_finals:
                    next_critic = torch.where(has_final[..., None], self._final_critic_table[final_ids], next_critic)
            flat = self.n_env * batch_size
            return {
                "obs": obs.reshape(flat, self.n_obs),
                "next_obs": next_obs.reshape(flat, self.n_obs),
                "critic_obs": critic.reshape(flat, self.n_critic_obs),
                "next_critic_obs": next_critic.reshape(flat, self.n_critic_obs),
                "actions": self.actions[lanes, ring].reshape(flat, self.n_act),
                "rewards": self.rewards[lanes, ring].reshape(flat),
                "dones": self.dones[lanes, ring].reshape(flat),
                "truncations": truncations.reshape(flat),
                "effective_n_steps": torch.ones(flat, device=self.device, dtype=self.dones.dtype),
            }

        oldest = max(0, self.ptr - self._cap)
        first = torch.full((self.n_env,), oldest, device=self.device, dtype=torch.long)
        # Only the overwritten prefix can be invalid. Inspect at most H-1
        # positions, not the potentially million-step replay capacity.
        for offset in range(min(self._history - 1, self.num_stored)):
            current = oldest + offset
            starts = self.episode_starts[:, current % self._cap]
            missing = torch.maximum(torch.full_like(starts, current - self._history + 1), starts) < oldest
            first = torch.where(missing, current + 1, first)
        available = self.ptr - 1 - first
        if (available <= 0).any():
            raise RuntimeError("cannot sample replay: no complete observation history on an environment lane")
        if self._history == 1:
            count = self.num_stored
            if self.n_steps > 1 and self.ptr - 1 < self.buffer_size:
                count = max(1, self.ptr - self.n_steps)
            t = oldest + torch.randint(0, count, (self.n_env, batch_size), device=self.device)
        else:
            t = first[:, None] + (torch.rand((self.n_env, batch_size), device=self.device) * available[:, None]).long()
        ring = t % self._cap
        lanes = torch.arange(self.n_env, device=self.device)[:, None]
        rewards = torch.zeros_like(t, dtype=torch.float)
        active = torch.ones_like(t, dtype=torch.bool)
        effective = torch.zeros_like(t)
        endpoint = t.clone()
        # Horizon is independent of history length. Both termination and time
        # limit stop rewards; the learner decides whether truncation bootstraps.
        for offset in range(self.n_steps):
            current = t + offset
            take = active & (current < self.ptr - 1)
            slot = current % self._cap
            rewards += self.gamma**offset * self.rewards[lanes, slot] * take
            effective += take.long()
            endpoint = torch.where(take, current, endpoint)
            active = take & ~(self.dones[lanes, slot].bool() | self.truncations[lanes, slot].bool())
        end_ring = endpoint % self._cap
        next_time = endpoint + 1
        obs = self._window(self.observations, t, self.policy_history)
        critic = self._window(self.critic_observations, t, self.value_history)
        next_obs = self._window(self.observations, next_time, self.policy_history)
        next_critic = self._window(self.critic_observations, next_time, self.value_history)
        truncations = self.truncations[lanes, end_ring]
        if self._final_slot is not None:
            final_ids = self._final_slot[lanes, end_ring].long()
            has_final = truncations.bool() & (final_ids >= 0)
            final_policy = self._final_obs_table[final_ids.clamp_min(0)]
            final_value = self._final_critic_table[final_ids.clamp_min(0)]
            # Preview append uses the endpoint's pre-reset history, never the
            # reset observation at endpoint+1. Only single final frames persist.
            endpoint_policy = self._window(self.observations, endpoint, self.policy_history)
            endpoint_value = self._window(self.critic_observations, endpoint, self.value_history)
            policy_preview = torch.cat((endpoint_policy[..., self.n_obs :], final_policy), dim=-1)
            value_preview = torch.cat((endpoint_value[..., self.n_critic_obs :], final_value), dim=-1)
            next_obs = torch.where(has_final[..., None], policy_preview, next_obs)
            next_critic = torch.where(has_final[..., None], value_preview, next_critic)
        flat = self.n_env * batch_size
        return {
            "obs": obs.reshape(flat, -1),
            "next_obs": next_obs.reshape(flat, -1),
            "critic_obs": critic.reshape(flat, -1),
            "next_critic_obs": next_critic.reshape(flat, -1),
            "actions": self.actions[lanes, ring].reshape(flat, self.n_act),
            "rewards": rewards.reshape(flat),
            "dones": self.dones[lanes, end_ring].reshape(flat),
            "truncations": truncations.reshape(flat),
            "effective_n_steps": effective.reshape(flat),
        }


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values.

    Also maintains optional LOCAL accumulators (``_local_*``) for the
    multi-learner cross-rank merge: the public ``_mean``/``_var``/``count``
    may be overwritten by a merge (they then hold GLOBAL stats), so they
    must never feed the next merge — that would double-count all history on
    every sync (count doubles per merge until float32 overflows). The
    accumulators stay disabled (``local_enabled=False``) until a learner
    turns them on; single-learner runs never pay the mirrored-update cost.
    Local accumulators are plain attributes, deliberately NOT registered
    buffers — they are per-process bookkeeping, not checkpoint state.
    """

    def __init__(self, shape, device, eps=1e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.local_enabled = False
        # MPS has no float64; the local accumulators exist only for the
        # multi-learner merge (a CUDA-only path), so keep them on the CPU.
        acc_device = device if torch.device(device).type != "mps" else torch.device("cpu")
        self._local_mean = torch.zeros(shape, dtype=torch.float64, device=acc_device)
        self._local_var = torch.ones(shape, dtype=torch.float64, device=acc_device)
        self._local_count = 0
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
        # Mirror into the LOCAL accumulators: the per-rank cumulative stats
        # that cross-rank merges consume. The public _mean/_var/_count may be
        # overwritten by a merge (they then hold GLOBAL stats), so they must
        # never feed the next merge — that would double-count all history on
        # every sync (count doubles per merge until float32 overflows).
        if self.local_enabled:
            b_mean = batch_mean.squeeze(0)
            b_var = batch_var.squeeze(0)
            l_count = self._local_count + batch_size
            delta = b_mean - self._local_mean
            self._local_mean += delta * (batch_size / l_count)
            delta2 = b_mean - self._local_mean
            l_m2 = (
                self._local_var * self._local_count
                + b_var * batch_size
                + delta2.pow(2) * (self._local_count * batch_size / l_count)
            )
            self._local_var.copy_(l_m2 / l_count)
            self._local_count = l_count

    def seed_local_accumulators(self) -> None:
        """Enable local accumulation, seeded from the current public stats.

        Used on the first cross-rank merge (including after resume): seeding
        from the loaded global stats means the first merge does not discard
        pre-resume history (the shared history is counted once per rank, an
        acceptable one-off bias that washes out as new data arrives).
        """
        self.local_enabled = True
        self._local_mean = self._mean.detach().clone().squeeze(0).double()
        self._local_var = self._var.detach().clone().squeeze(0).double()
        self._local_count = int(self.count)

    def local_sufficient_stats_flat(self) -> torch.Tensor:
        """Pack the LOCAL accumulators as ``[count, sum, sumsq]`` (float64).

        Single-tensor layout so a cross-rank merge needs exactly one
        collective. ``sumsq`` is reconstructed from var via
        ``E[x^2] = var + mean^2`` — SUM is additive while mean/var are not.
        """
        count = torch.tensor(float(self._local_count), dtype=torch.float64, device=self._mean.device)
        total = count * self._local_mean
        sumsq = self._local_var * count + total * total / count.clamp_min(1.0)
        return torch.cat([count.reshape(1), total, sumsq])

    def apply_global_sufficient_stats(self, flat: torch.Tensor) -> None:
        """Restore merged ``[count, sum, sumsq]`` into the public buffers.

        Computes the global mean/var in float64 (``sumsq`` is a sum of
        same-scale squares; float32 would catastrophically cancel), clamps
        the sample count to ``until`` to match the update freeze, and writes
        ``_mean``/``_var``/``_std``/``count``. The local accumulators are
        untouched: they keep tracking rank-local samples only.
        """
        d = (flat.numel() - 1) // 2
        n, s, q = flat[0], flat[1 : 1 + d], flat[1 + d :]
        if n <= 0:
            return
        mean = (s / n).float()
        var = ((q - s * s / n) / n).clamp_min_(0.0).float()
        if self.until is not None:
            n = torch.minimum(n, torch.tensor(float(self.until), dtype=n.dtype, device=n.device))
        self._mean.copy_(mean.unsqueeze(0))
        self._var.copy_(var.unsqueeze(0))
        self._std.copy_(var.sqrt().unsqueeze(0))
        self.count.copy_(n.long())
