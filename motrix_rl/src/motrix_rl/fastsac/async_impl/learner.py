# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Learner: owns a full ``FastSacAgent`` and drives training off the shared ring.

Unlike the sync trainer it does NOT step the env. It drains raw transitions from
:class:`~motrix_rl.fastsac.async_impl.shm.SharedTransitionRing` into the agent's GPU
replay buffer, runs gradient updates governed by ``utd_mode`` (§6 of the
design), and periodically publishes actor weights + obs-normalizer stats to the
collector via :class:`~motrix_rl.fastsac.async_impl.shm.WeightSnapshot`.

The update math is reused unchanged from the sync agent: this module delegates
the per-step gradient work to ``agent.update(n)`` and only owns the
async-specific orchestration (drain, UTD-ratio governance, weight publishing).
"""

from __future__ import annotations

import time

from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing, WeightSnapshot
from motrix_rl.fastsac.config import FastSacCfg


class Learner:
    def __init__(
        self,
        agent: FastSacAgent,
        cfg: FastSacCfg,
        ring: SharedTransitionRing,
        weights: WeightSnapshot,
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
        GPU copy, so the collector cannot clobber an in-flight slot. Each slot's
        ``next_obs``/``next_critic_obs`` come from the successor slot
        (``ring.peek_next``), so a slot is only ingested once its successor is
        committed; the last slot of a drain wave waits for the next one.
        """
        device = self.agent.device
        ingested = 0
        for _ in range(max(self.async_options.max_ingest_per_iter, 1)):
            if not self.ring.has_next():
                break
            slot = self.ring.read_slot()
            assert slot is not None  # has_next implies a readable slot
            obs, critic_obs, actions, rewards, dones, truncations = slot
            next_obs, next_critic_obs = self.ring.peek_next()
            self.agent.rb.extend(
                obs.to(device),
                critic_obs.to(device),
                actions.to(device),
                rewards.to(device),
                dones.to(device),
                truncations.to(device),
                next_obs.to(device),
                next_critic_obs.to(device),
            )
            self.ring.commit_read()
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
