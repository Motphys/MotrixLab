# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Learner: owns a full ``FastSacAgent`` and drives training off the shared ring.

Unlike the sync trainer it does NOT step the env. It drains raw transitions from
:class:`~motrix_rl.fastsac.async_impl.shm.SharedTransitionRing` into the agent's GPU
replay buffer, runs gradient updates governed by ``utd_mode`` (§6 of the
design), and periodically publishes actor weights + obs-normalizer stats to the
collector via its :class:`~motrix_rl.fastsac.async_impl.shm.WeightSender` endpoint.

The update math is reused unchanged from the sync agent: this module delegates
the per-step gradient work to ``agent.update(n)`` and only owns the
async-specific orchestration (drain, UTD-ratio governance, weight publishing).
"""

from __future__ import annotations

import time

import torch

from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing
from motrix_rl.fastsac.async_impl.shm.weight_channel import WeightSender
from motrix_rl.fastsac.config import FastSacCfg


class Learner:
    def __init__(
        self,
        agent: FastSacAgent,
        cfg: FastSacCfg,
        ring: SharedTransitionRing,
        weights: WeightSender,
        control: Control,
    ):
        self.agent = agent
        self.cfg = cfg
        self.async_options = cfg.trainer.async_options
        self.ring = ring
        self.weights = weights
        self.control = control
        self._learning_starts = agent.cfg.learning_starts
        self._last_publish_ms = 0.0
        # Device-transport ingest bookkeeping: one CUDA event per ingested slot
        # whose replay-buffer copy is still in flight (see _retire_device_reads).
        self._device_read_events: list = []

        # keep normalizers/actor in train mode: the learner is the update side.
        self.agent.set_train_mode()

    # The persistent gradient-update counter lives on the agent so both sync
    # and async trainers share the same policy-frequency gating source-of-truth.
    # Exposed here as a read-only property for the worker's UTD / publish panel.
    @property
    def update_idx(self) -> int:
        return self.agent.update_idx

    # ------------------------------------------------------------------ ingest
    def drain(self) -> int:
        """Move up to ``max_ingest_per_iter`` ring slots into the replay buffer.

        Returns the number of slots ingested. Read cursor advances only after the
        slot data has left the ring (synchronously on the host transport; after
        the enqueued device copies are *proven complete* on the device transport
        — see ``_retire_device_reads``), so the collector cannot clobber an
        in-flight slot. The replay buffer derives each transition's
        ``next_obs`` from the following slot's stored observation, so no
        successor peek is needed and a slot is ingested as soon as it is
        committed.
        """
        if self.ring.device_slots_bound:
            return self._drain_device()
        device = self.agent.device
        ingested = 0
        for _ in range(max(self.async_options.max_ingest_per_iter, 1)):
            if not self.ring.has_next():
                break
            slot = self.ring.read_slot()
            assert slot is not None  # has_next implies a readable slot
            obs, critic_obs, actions, rewards, dones, truncations = slot
            self.agent.rb.extend(
                obs.to(device),
                critic_obs.to(device),
                actions.to(device),
                rewards.to(device),
                dones.to(device),
                truncations.to(device),
            )
            self.ring.commit_read()
            ingested += 1
        return ingested

    def _retire_device_reads(self) -> None:
        """Advance the read cursor for device-ingested slots whose copies landed.

        The replay-buffer writes are asynchronous D2D copies enqueued on the
        learner's compute stream; committing the slot before they retire would
        let the collector wrap around and overwrite memory the GPU has not
        read yet. Each drain records one event per slot; polling them here
        keeps the cursor lag bounded by the in-flight window (the copies sit
        ahead of the update kernels on the same stream, so they complete
        within one learner loop).
        """
        while self._device_read_events and self._device_read_events[0].query():
            self._device_read_events.pop(0)
            self.ring.commit_read()

    def _drain_device(self) -> int:
        """Device-transport ingest: ring slots alias CUDA-IPC device memory.

        obs/critic_obs go straight into the replay buffer as D2D copies on the
        current stream (ordered before the update's replay sampling for free —
        same stream); only the small scalar fields cross from host shared
        memory. Nothing blocks the CPU.
        """
        self._retire_device_reads()
        device = self.agent.device
        ingested = 0
        for _ in range(max(self.async_options.max_ingest_per_iter, 1)):
            if not self.ring.has_next():
                break
            slot = self.ring.read_slot()
            assert slot is not None  # has_next implies a readable slot
            obs, critic_obs, actions, rewards, dones, truncations = slot
            self.agent.rb.extend(
                obs,
                critic_obs,
                actions.to(device),
                rewards.to(device),
                dones.to(device),
                truncations.to(device),
            )
            event = torch.cuda.Event()
            event.record()
            self._device_read_events.append(event)
            ingested += 1
        return ingested

    # ------------------------------------------------------------------ update
    def _ready(self) -> bool:
        return self.control.collector_steps >= self._learning_starts and self.agent.rb.num_stored > 0

    def _num_updates_for(self, ingested: int) -> int:
        """Decide how many gradient updates to run this iteration."""
        base = self.agent.cfg.num_updates
        mode = self.async_options.utd_mode
        if mode == "strict":
            # exactly num_updates per ingested env-step batch -> matches sync UTD.
            return ingested * base
        # learner_bound: run a full batch of updates whenever ready.
        # In the two-process path the learner loops continuously; here per-call.
        return base

    def maybe_train(self, ingested: int) -> dict | None:
        """Run ratio-governed updates. Returns last metrics dict or ``None``."""
        if not self._ready():
            return None
        n = self._num_updates_for(ingested)
        # Delegate the per-step work to the agent; this module no longer keeps
        # its own update-loop / update_idx / _last_actor — the agent's
        # counter is the single source of truth for policy-frequency gating.
        metrics = self.agent.update(n)
        self._last_publish_ms = 0.0
        if metrics is not None:
            started = time.perf_counter()
            self.publish_if_due()
            self._last_publish_ms = (time.perf_counter() - started) * 1000.0
        return metrics

    # ------------------------------------------------------------------ publish
    def publish_weights(self) -> None:
        self.weights.publish(self.agent.actor, self.agent.obs_normalizer)

    def publish_if_due(self) -> None:
        if self.agent.update_idx % max(self.async_options.weight_publish_interval, 1) == 0:
            self.publish_weights()
