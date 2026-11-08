# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC's sparse per-slot final-observation store.

With auto-reset envs the observation returned by the terminating step is NOT
the reset observation that the following slot stores, yet the replay buffer
needs it to bootstrap the n-step return. This store keeps those sparse rows
in host shared memory; the generic ring treats it as an opaque
``SlotPayloadStore`` and governs row reclamation with the transition cursors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motrix_rl.transport.common import _shared


@dataclass
class FinalObsPayload:
    """Owned compact CPU observation rows belonging to one transition slot."""

    env_ids: torch.Tensor
    obs: torch.Tensor
    critic_obs: torch.Tensor


class FinalObsStorage:
    """Shared sparse rows whose lifetime is governed by the transition cursors.

    Only the producer manages row allocation. Every slot records an absolute
    row end, including slots without a payload. Observing an acknowledged slot
    releases its rows; no independently published or consumed cursor exists.
    """

    def __init__(self, capacity: int, num_envs: int, obs_dim: int, critic_obs_dim: int):
        self.slot_capacity = capacity
        self.capacity = 3 * num_envs
        self.env_ids = _shared((self.capacity,), torch.int64)
        self.obs = _shared((self.capacity, obs_dim), torch.float32)
        self.critic_obs = _shared((self.capacity, critic_obs_dim), torch.float32)
        self.starts = _shared((capacity,), torch.int64)
        self.counts = _shared((capacity,), torch.int64)
        self.ends = _shared((capacity,), torch.int64)
        self._row_write = 0
        self._row_read = 0
        self._reclaimed_slots = 0

    def can_write(self, payload: FinalObsPayload | None, acknowledged: int) -> bool:
        n = 0 if payload is None else len(payload.env_ids)
        if n > self.capacity:
            raise ValueError("final payload exceeds the sparse row store capacity")
        return self.can_write_rows(n, acknowledged)

    def can_write_rows(self, count: int, acknowledged: int) -> bool:
        """Reclaim completed slots and check space for a prospective row count."""
        # Reclaim before the producer can overwrite any slot metadata. The
        # main ring's capacity guarantees these ends still belong to the
        # acknowledged generation. IPC callers pass event-proven cursors.
        while self._reclaimed_slots < acknowledged:
            slot = self._reclaimed_slots % self.slot_capacity
            self._row_read = int(self.ends[slot])
            self._reclaimed_slots += 1
        return self._row_write - self._row_read + count <= self.capacity

    def write(self, index: int, payload: FinalObsPayload | None) -> None:
        """Write payload before the owning transition slot is published."""
        slot = index % self.slot_capacity
        n = 0 if payload is None else len(payload.env_ids)
        start = self._row_write
        if n:
            rows = (start + torch.arange(n, dtype=torch.int64)) % self.capacity
            self.env_ids[rows] = torch.as_tensor(payload.env_ids, dtype=torch.int64)
            self.obs[rows] = torch.as_tensor(payload.obs, dtype=torch.float32)
            self.critic_obs[rows] = torch.as_tensor(payload.critic_obs, dtype=torch.float32)
        self.starts[slot] = start
        self.counts[slot] = n
        self.ends[slot] = start + n
        self._row_write = start + n

    def read(self, index: int, count: int) -> list[FinalObsPayload | None]:
        """Copy all payload rows before the caller acknowledges their slots."""
        result = []
        for offset in range(count):
            slot = (index + offset) % self.slot_capacity
            n = int(self.counts[slot])
            if not n:
                result.append(None)
                continue
            rows = (int(self.starts[slot]) + torch.arange(n, dtype=torch.int64)) % self.capacity
            # Advanced indexing produces owned snapshots, including on wrap.
            result.append(FinalObsPayload(self.env_ids[rows], self.obs[rows], self.critic_obs[rows]))
        return result
