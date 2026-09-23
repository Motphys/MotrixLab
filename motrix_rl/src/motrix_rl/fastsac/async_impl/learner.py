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
from dataclasses import dataclass

import torch
import torch.distributed as dist

from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.transport import Control, IpcTransitionRing, SharedTransitionRing
from motrix_rl.fastsac.async_impl.transport.weight_channel import WeightSender
from motrix_rl.fastsac.buffer import EmpiricalNormalization
from motrix_rl.fastsac.config import FastSacCfg


@dataclass
class CollectorEndpoint:
    """The learner-side endpoints for one collector.

    Both halves belong to collector ``i`` of this rank: the ring the
    collector's transitions arrive on, and the weight channel they are
    published back to. Index alignment is the ownership relation — a
    misordered pair silently cross-wires two collectors.
    """

    ring: SharedTransitionRing | IpcTransitionRing
    weight_sender: WeightSender


class GenerationAssembler:
    """Cross-collector generation assembly: shards in, complete generations out.

    Shards are indexed by their ABSOLUTE generation number (the ring slot's
    ordinal since process start), so collectors delivering at different rates
    land on the same timeline. A generation is complete once every ring's
    shard has arrived; :meth:`pop_ready` pops complete generations in strict
    ordinal order, and incomplete ones wait here indefinitely without
    blocking the other rings (their own rings backpressure them instead).
    """

    def __init__(self, num_rings: int):
        self.num_rings = num_rings
        self.next_gen = 0
        self._slots: dict[int, list[tuple | None]] = {}

    def add(self, ring_id: int, gen: int, part: tuple) -> None:
        """Place one ring's shard at its absolute generation slot."""
        slots = self._slots.setdefault(gen, [None] * self.num_rings)
        slots[ring_id] = part

    def pop_ready(self) -> list[list[tuple]]:
        """Pop consecutive complete generations starting at the cursor."""
        ready: list[list[tuple]] = []
        while (slots := self._slots.get(self.next_gen)) is not None and None not in slots:
            ready.append(slots)
            del self._slots[self.next_gen]
            self.next_gen += 1
        return ready

    def stats(self) -> tuple[int, int, int, int]:
        """``(next_gen, oldest pending gen, newest pending gen, pending count)``."""
        return (
            self.next_gen,
            min(self._slots) if self._slots else -1,
            max(self._slots) if self._slots else -1,
            len(self._slots),
        )


class DrainStaging:
    """Pinned-staging H2D pipeline for host-ring shards on a CUDA learner.

    Exists only when the learner runs on CUDA and owns at least one host
    ring. Host-ring views are memcpy'd into pinned buffers and moved to the
    GPU with one non-blocking H2D copy per field on a dedicated stream, so
    ingest overlaps gradient updates. IPC shards need none of this — their
    slots are already on the device (see _drain_generations' host pull-down).

    Lifecycle discipline:

    * :meth:`acquire` — wait before overwriting buffers that may still back
      the previous drain's in-flight copies;
    * :meth:`stage` — host views -> pinned -> async H2D ``rb.extend_batch``;
    * :meth:`mark_in_flight` — record the event at the end of a drain, so
      later sampling waits for the copies;
    * :meth:`wait` — make the compute stream wait for in-flight copies
      before sampling;
    * :meth:`finish` — block until everything landed (end of ingesting).
    """

    def __init__(self, rb, chunk: int, device: torch.device):
        self.rb = rb
        self.stream = torch.cuda.Stream(device=device)
        self.event = torch.cuda.Event()
        self.pending = False
        pin = lambda *shape: torch.empty(*shape, pin_memory=True)  # noqa: E731
        self.buffers = (
            pin(chunk, rb.n_env, rb.n_obs),
            pin(chunk, rb.n_env, rb.n_critic_obs),
            pin(chunk, rb.n_env, rb.n_act),
            pin(chunk, rb.n_env),
            torch.empty(chunk, rb.n_env, dtype=torch.int64, pin_memory=True),
            torch.empty(chunk, rb.n_env, dtype=torch.int64, pin_memory=True),
        )

    def acquire(self) -> None:
        """Join in-flight copies before the buffers get overwritten."""
        self.stream.synchronize()

    def stage(self, views) -> None:
        """Copy one host run into the pinned buffers and issue the H2D ingest."""
        k = views[0].shape[0]
        for stage, view in zip(self.buffers, views):
            stage[:k].copy_(view)  # ring -> pinned (plain CPU memcpy)
        with torch.cuda.stream(self.stream):
            self.rb.extend_batch(*(stage[:k] for stage in self.buffers))
        self.pending = True

    def mark_in_flight(self) -> None:
        """Record the event that later ``wait`` calls synchronize on."""
        if self.pending:
            self.event.record(self.stream)

    def wait(self) -> None:
        """Make the compute stream wait for in-flight copies, then clear."""
        if self.pending:
            self.event.wait()
        self.pending = False

    def finish(self) -> None:
        """Block until every issued ingest copy has landed in the buffer."""
        self.stream.synchronize()
        self.pending = False


class Learner:
    """Drains N independent SPSC rings (one per collector) and broadcasts weights.

    Each collector owns its own ring and its own weight-channel sender; the
    single-writer invariants of both primitives are preserved unchanged. The
    learner round-robins the rings (per-ring ingest bound so a fast collector's
    full ring never starves the others) and publishes the current actor snapshot
    to every collector in turn — each publish is independent and lock-free, so
    one slow reader never blocks the others.

    Ring slots carry ``num_envs / num_collectors`` transitions each. The replay
    buffer (and its n-step time adjacency per env) is built around full
    ``num_envs`` batches, so shards are merged by generation: the k-th slot of
    every collector assembles into the k-th full batch (env ids map to
    contiguous shard blocks). A slow collector's generation stays pending on
    the CPU slot views until complete — its own ring fills up and backpressures
    only that collector; read cursors are committed only after the merged batch
    reached the GPU, preserving the ring's no-clobber guarantee.
    """

    def __init__(
        self,
        agent: FastSacAgent,
        cfg: FastSacCfg,
        collectors: list[CollectorEndpoint],
        control: Control,
        ddp_rank: int | None = None,
    ):
        self.agent = agent
        self.cfg = cfg
        self.async_options = cfg.trainer.async_options
        self.collectors = collectors
        self.rings = [collector.ring for collector in collectors]
        self.weights = [collector.weight_sender for collector in collectors]
        self.control = control
        self._pending = GenerationAssembler(num_rings=len(self.rings))
        # Ring transport per LEARNER RANK is all-or-nothing (topology), so
        # each rank gets its own ingest path: a PURE-HOST rank on CUDA runs
        # the staged async H2D pipeline (DrainStaging); pure-IPC and MIXED
        # ranks lift host shards at assembly and consume everything D2D
        # (see _drain_rings), so staging would be dead weight there.
        has_host = any(not isinstance(ring, IpcTransitionRing) for ring in self.rings)
        has_ipc = any(isinstance(ring, IpcTransitionRing) for ring in self.rings)
        self._staging: DrainStaging | None = None
        if agent.device.type == "cuda" and has_host and not has_ipc:
            self._staging = DrainStaging(agent.rb, max(self.async_options.max_ingest_per_iter, 1), agent.device)
        # This rank owns and publishes to its own collectors only: the slice
        # must cover exactly this rank's share of the global collector count
        # (the total lives in the shared control block).
        per_learner = control.num_collectors // agent.world_size
        if len(collectors) != per_learner:
            raise ValueError(
                f"{len(collectors)} collector endpoints must cover this rank's slice of "
                f"{per_learner} collectors ({control.num_collectors} total)"
            )
        # DDP replica id (multi-learner); None keeps the single-learner path.
        self.ddp_rank = ddp_rank
        self._learning_starts = agent.cfg.learning_starts
        self._last_publish_ms = 0.0
        # global-progress basis consumed by the last lockstep update call
        # (multi-learner path only); the worker re-bases it on resume.
        self._last_train_gstep = 0

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
        """Move up to ``max_ingest_per_iter`` slots per ring into the replay buffer.

        Returns the number of full ``num_envs`` batches ingested. Slots are
        assembled into generations (one shard per collector) and complete
        generations are merged, moved to the GPU and written with one
        :meth:`extend_batch` per field per flush; a single ring is the
        ``num_rings == 1`` special case (its shard alone is the generation,
        no merge needed). Read cursors advance only after the GPU copy, so a
        collector cannot clobber an in-flight slot; a full or slow ring only
        blocks itself.

        Per-ring transport notes:

        * CUDA-IPC device ring: the strided device views go straight into the
          replay buffer with D2D copies on the current stream (ordered before
          any subsequent sample by stream order).
        * host ring on a pure-host CUDA rank: each shard is memcpy'd into
          pinned staging and moved to the GPU as one non-blocking H2D copy per
          field on the copy stream (overlapping the next gradient update).
        * host ring in a MIXED rank: the shard is lifted to the ingest device
          at assembly, so the merge never leaves the GPU.
        * host ring on CPU: the views copy directly into the buffer.

        The replay buffer derives each transition's ``next_obs`` from the
        following slot's stored observation, so no successor peek is needed
        and a slot is ingested as soon as it is committed.
        """
        return self._drain_rings(max(self.async_options.max_ingest_per_iter, 1))

    def _drain_rings(self, budget: int) -> int:
        for ring_id, ring in enumerate(self.rings):
            base = ring.read_idx
            avail = min(budget, ring.size())
            if avail <= 0:
                continue
            count, views = ring.read_span()
            # PURE-HOST rank (self._staging): keep the host views — the staged
            # pipeline moves them with async H2D on the copy stream,
            # overlapping the next gradient update. MIXED rank: lift host
            # shards to the device at assembly so the merge is single-device
            # D2D; IPC shards are already there. Host ring on CPU consumes
            # directly into the buffer.
            if self._staging is None and not isinstance(ring, IpcTransitionRing):
                views = tuple(view.to(self.agent.device) for view in views)
            count = min(count, avail)
            for offset in range(count):
                self._pending.add(ring_id, base + offset, tuple(v[offset] for v in views))
        return self._flush_complete_generations()

    def _stall_diag(self, tag: str) -> None:
        import logging
        import os

        if not os.environ.get("MOTRIX_STALL_DEBUG"):
            return
        next_gen, oldest, newest, count = self._pending.stats()
        logging.getLogger(__name__).error(
            "STALL[%s] next_gen=%d pending=[%s..%s] n_pending=%d rings=[%s]",
            tag,
            next_gen,
            oldest,
            newest,
            count,
            [(r.read_idx, r.write_idx) for r in self.rings],
        )

    def _flush_complete_generations(self) -> int:
        generations = self._pending.pop_ready()
        if not generations:
            self._stall_diag("flush-empty")
            return 0
        if self._staging is not None:
            # staging may still back the previous drain's in-flight H2D copies
            self._staging.acquire()
        # env blocks concatenate in collector order -> env id mapping is
        # stable across batches, so per-env trajectories stay contiguous in
        # the buffer's time dimension. A single shard skips the cat.
        runs = [
            parts[0] if len(parts) == 1 else tuple(torch.cat(field) for field in zip(*parts)) for parts in generations
        ]
        # runs arrive env-major per generation (n_env, dim); the buffer write
        # path is slot-major (k, n_env, dim), so stack generations first.
        stacked = tuple(torch.stack(column, dim=0) for column in zip(*runs))
        ingested = self._ingest_runs((stacked,))
        for ring in self.rings:
            ring.commit_reads(ingested)
        if self._staging is not None:
            self._staging.mark_in_flight()
        return ingested

    def _ingest_runs(self, runs) -> int:
        """Ingest slot-major device runs ``(k, n_env, dim)``; caller commits reads.

        Every shard was lifted to the ingest device before assembly, so each
        ``extend_batch`` is a same-device strided copy. Returns the number of
        RING SLOTS consumed — each run's leading dim IS the slot count;
        returning ``len(runs)`` instead desynced the read cursor from
        ``_next_gen`` by G-1 per flush (learner stalls permanently once the
        gap exceeds the pending window).
        """
        if self._staging is not None:
            # pure-host rank: host run -> pinned buffers -> async H2D extend
            self._staging.stage(runs[0])
            return runs[0][0].shape[0]
        k = 0
        for views in runs:
            self.agent.rb.extend_batch(*views)
            k += views[0].shape[0]
        return k

    # ------------------------------------------------------------------ update
    def wait_ingest(self) -> None:
        """Block until every issued ingest copy has landed in the buffer."""
        if self._staging is not None:
            self._staging.finish()

    def maybe_train(self, gstep: int) -> dict | None:
        """Run ratio-governed updates when the global progress basis advanced.

        ``gstep`` is the trainer's full-batch-equivalent progress
        (``control.collector_steps // num_collectors``) — identical on every
        DDP rank, and the single learner's own position when
        ``num_collectors == 1``. One decision formula serves both topologies:

        * single learner: decides locally; an empty replay buffer merely
          defers the update (the cumulative delta covers it next call);
        * multi-learner: rank 0 decides, the count is broadcast, and every
          rank waits for data rather than skipping — skipping would desync
          the DDP collectives.
        """
        is_src = self.ddp_rank in (None, 0)
        n = 0
        if is_src:
            warmup_met = self.control.collector_steps >= self._learning_starts * self.control.num_collectors
            if gstep > self._last_train_gstep and warmup_met:
                delta = gstep - self._last_train_gstep
                base = self.agent.cfg.num_updates
                n = base * delta if self.async_options.utd_mode == "strict" else base
        if self.ddp_rank is not None:
            count_tensor = torch.tensor([n], device=self.agent.device)
            dist.broadcast(count_tensor, src=0)
            n = int(count_tensor[0])
            # Global readiness does not guarantee THIS rank's shard has data
            # yet; wait (bounded by the shared stop flag) rather than skipping
            # the update — skipping would desynchronize the DDP collectives.
            while self.agent.rb.num_stored == 0 and not self.control.stop:
                self.drain()
                time.sleep(self.async_options.idle_sleep_s)
            if self.control.stop and self.agent.rb.num_stored == 0:
                n = 0
        elif self.agent.rb.num_stored == 0:
            # Single learner: an empty buffer merely DEFERS the update — the
            # cumulative delta grows, so nothing is lost on the next call.
            n = 0
        if n <= 0:
            return None
        # Sampling reads replay-buffer slots the copy stream may still be
        # filling (staged async H2D ingest); make the compute stream wait for
        # the in-flight copies first, mirroring the pre-staging event wait.
        if self._staging is not None:
            self._staging.wait()
        self._last_train_gstep = gstep
        # Publish cadence per gradient step: chunk strict-mode ``n`` back to
        # ``num_updates``-sized updates (measured 15x more publishes than one
        # huge update call — collectors act on much fresher weights).
        base = self.agent.cfg.num_updates
        metrics = None
        remaining = n
        while remaining > 0 and not self.control.stop:
            metrics = self.agent.update(min(base, remaining))
            remaining -= base
            self._last_publish_ms = 0.0
            if metrics is not None:
                started = time.perf_counter()
                self.publish_if_due()
                self._last_publish_ms = (time.perf_counter() - started) * 1000.0
        return metrics

    # ------------------------------------------------------------------ publish
    def _sync_normalizer_stats(self) -> None:
        """All-reduce every normalizer's (count, sum, sumsq) across DDP ranks.

        Each rank's EmpiricalNormalization only sees its own collector shard,
        so per-rank statistics diverge; collectors act with rank-0's published
        stats while rank-1 trains on inputs normalized differently — the
        averaged gradients then pull the networks toward two different input
        scalings (observed as a large multi-learner convergence gap early in
        training). Merging sufficient statistics (n, Sum, SumSq) with one
        SUM all-reduce per normalizer makes every rank hold identical GLOBAL
        stats at each weight publish.

        Sync cadence is a deliberate design point (survey of IsaacLab /
        mjlab / rsl_rl / holosoma, see MotrixLab#75): rsl_rl syncs nothing
        (per-rank stats diverge forever); holosoma embeds an all_reduce in
        every update() — exact per-step identity, but it requires lockstep
        stepping with equal batch shapes and doubles collectives in the hot
        learner loop. We merge at the weight-publish cadence instead: drift
        between publishes is bounded by ``weight_publish_interval`` steps of
        rank-local batches layered on fresh global stats, which captures
        essentially all of the convergence benefit at 1/Nth the collectives.
        If tighter consistency is ever needed, raising the sync frequency is
        a call-site change — per-update merging stays structurally possible
        because the local accumulators make the merge exact and idempotent
        at any cadence.
        """
        for norm in (self.agent.obs_normalizer, self.agent.critic_obs_normalizer):
            if not isinstance(norm, EmpiricalNormalization):
                continue  # nn.Identity when obs_normalization is off
            if not norm.local_enabled:
                norm.seed_local_accumulators()
            flat = norm.local_sufficient_stats_flat()
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            norm.apply_global_sufficient_stats(flat)

    def publish_weights(self) -> None:
        """Broadcast the current actor snapshot to every collector's snapshot."""
        if self.agent.world_size > 1:
            self._sync_normalizer_stats()
        for weights in self.weights:
            weights.publish(self.agent.actor, self.agent.obs_normalizer)

    def publish_if_due(self) -> None:
        if self.agent.update_idx % max(self.async_options.weight_publish_interval, 1) == 0:
            self.publish_weights()
