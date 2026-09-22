# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Learner: owns a full ``FastSacAgent`` and drives training off the shared ring.

Unlike the sync trainer it does NOT step the env. It drains raw transitions from
the shared transition ring (host fields or CUDA-IPC device fields, see
``transport/ring.py`` / ``transport/ipc_ring.py``) into the agent's GPU replay buffer, runs
gradient updates governed by ``utd_mode`` (§6 of the design), and periodically
publishes actor weights + obs-normalizer stats to the collector via its
:class:`~motrix_rl.fastsac.async_impl.transport.WeightSender` endpoint.

The update math is reused unchanged from the sync agent: this module delegates
the per-step gradient work to ``agent.update(n)`` and only owns the
async-specific orchestration (drain, UTD-ratio governance, weight publishing).
"""

from __future__ import annotations

import time

import torch

from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.transport import Control, IpcTransitionRing, SharedTransitionRing
from motrix_rl.fastsac.async_impl.transport.weight_channel import WeightSender
from motrix_rl.fastsac.config import FastSacCfg


class Learner:
    def __init__(
        self,
        agent: FastSacAgent,
        cfg: FastSacCfg,
        ring: SharedTransitionRing | IpcTransitionRing,
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

        # Host-ring async-ingest plumbing (CUDA only): contiguous runs are
        # staged through pinned buffers and moved to the GPU with non-blocking
        # H2D copies on a dedicated stream, so ingestion overlaps gradient
        # updates. The CUDA-IPC device ring needs none of this — its slots are
        # already on the device and ingest is a same-stream D2D copy.
        self._device_ring = isinstance(ring, IpcTransitionRing)
        self._copy_stream = None
        self._copy_event = None
        self._pending_copy = False
        self._staging = None
        if agent.device.type == "cuda" and not self._device_ring:
            self._copy_stream = torch.cuda.Stream(device=agent.device)
            self._copy_event = torch.cuda.Event()
            rb = agent.rb
            chunk = max(self.async_options.max_ingest_per_iter, 1)
            pin = lambda *shape: torch.empty(*shape, pin_memory=True)  # noqa: E731
            self._staging = (
                pin(chunk, rb.n_env, rb.n_obs),
                pin(chunk, rb.n_env, rb.n_critic_obs),
                pin(chunk, rb.n_env, rb.n_act),
                pin(chunk, rb.n_env),
                torch.empty(chunk, rb.n_env, dtype=torch.int64, pin_memory=True),
                torch.empty(chunk, rb.n_env, dtype=torch.int64, pin_memory=True),
            )

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

        Returns the number of slots ingested. Slots are consumed in contiguous
        runs (``ring.read_span()``):

        * CUDA-IPC device ring: the strided device views go straight into the
          replay buffer with D2D copies on the current stream (ordered before
          any subsequent sample by stream order); the read cursor is released
          behind an event once those copies complete.
        * host ring on CUDA: each run is memcpy'd into pinned staging and moved
          to the GPU as one non-blocking H2D copy per field on the copy stream
          (overlapping the next gradient update), then the read cursor
          advances — the producer cannot clobber in-flight data because the
          ring slot was already fully copied to staging.
        * host ring on CPU: the views copy directly into the buffer.

        The replay buffer derives each transition's ``next_obs`` from the
        following slot's stored observation, so no successor peek is needed and
        a slot is ingested as soon as it is committed.
        """
        budget = max(self.async_options.max_ingest_per_iter, 1)
        if self._staging is not None:
            # Staging may still be the source of the previous drain's in-flight
            # H2D copies; wait before overwriting it. Any pending copy was
            # already joined by the intervening maybe_train(), so this is
            # normally a no-op sync.
            self._copy_stream.synchronize()
        ingested = 0
        while ingested < budget and self.ring.has_next():
            k, views = self.ring.read_span()
            k = min(k, budget - ingested)
            if k < views[0].shape[0]:
                views = tuple(v[:k] for v in views)
            if self._staging is not None:
                for stage, view in zip(self._staging, views):
                    stage[:k].copy_(view)  # ring -> pinned (plain CPU memcpy)
                with torch.cuda.stream(self._copy_stream):
                    self.agent.rb.extend_batch(*(stage[:k] for stage in self._staging))
                self._pending_copy = True
            else:
                # CUDA-IPC device ring or CPU host ring: consume the views
                # directly (D2D strided copy, or plain CPU copy).
                self.agent.rb.extend_batch(*views)
            self.ring.commit_reads(k)
            ingested += k
        if self._pending_copy:
            self._copy_event.record(self._copy_stream)
        return ingested

    # ------------------------------------------------------------------ update
    def wait_ingest(self) -> None:
        """Block until every issued ingest copy has landed in the buffer."""
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        elif self._device_ring:
            torch.cuda.synchronize(self.agent.device)
        self._pending_copy = False

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
        if self._pending_copy:
            # Sampling reads slots the copy stream may still be filling; make
            # the compute stream wait for the in-flight H2D ingest copies.
            # (The device-ring path needs no event: its D2D copies are on the
            # same stream as sampling.)
            self._copy_event.wait()
            self._pending_copy = False
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
