# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the CUDA-IPC transition ring.

Single-process owner + receiver pairs share one CUDA tensor (the in-process
stand-in for the cross-process IPC mapping), so every protocol property under
test — fused-slot layout, event-ordered cursor publishing, lazy commit,
backpressure, replay-buffer equivalence with the host ring — behaves exactly
as it does between the two worker processes.

GPU-less environments skip the ring tests; the host ring keeps its own tests
in ``test_fastsac_buffer.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from motrix_rl.fastsac.async_impl.transport import (
    IpcTransitionRing,
    RingCursors,
    SharedTransitionRing,
)
from motrix_rl.fastsac.buffer import SimpleReplayBuffer

CAPACITY, N_ENV, OBS, CRI, ACT = 5, 3, 4, 6, 2
FEAT = OBS + CRI + ACT + 3

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="IPC ring requires a GPU")


def _batch(t: int, seed: int = 0):
    """Deterministic per-step field tensors (the collector's CPU outputs)."""
    g = torch.Generator().manual_seed(seed * 1000 + t)
    return (
        torch.randn(N_ENV, OBS, generator=g),
        torch.randn(N_ENV, CRI, generator=g),
        torch.randn(N_ENV, ACT, generator=g),
        torch.randn(N_ENV, generator=g),
        torch.randint(0, 2, (N_ENV,), generator=g, dtype=torch.long),
        torch.randint(0, 2, (N_ENV,), generator=g, dtype=torch.long),
    )


def _owner_receiver():
    cursors = RingCursors()
    slots = torch.zeros(CAPACITY, N_ENV, FEAT, dtype=torch.float32, device="cuda")
    owner = IpcTransitionRing(cursors, slots, CAPACITY, N_ENV, OBS, CRI, ACT)
    receiver = IpcTransitionRing(cursors, slots, CAPACITY, N_ENV, OBS, CRI, ACT)
    return owner, receiver


@cuda_only
def test_ipc_ring_rejects_bad_slots():
    cursors = RingCursors()
    with pytest.raises(ValueError, match="CUDA"):
        IpcTransitionRing(cursors, torch.zeros(CAPACITY, N_ENV, FEAT), CAPACITY, N_ENV, OBS, CRI, ACT)
    with pytest.raises(ValueError, match="shape"):
        IpcTransitionRing(
            cursors, torch.zeros(CAPACITY, N_ENV, FEAT + 1, device="cuda"), CAPACITY, N_ENV, OBS, CRI, ACT
        )


@cuda_only
def test_ipc_ring_roundtrip_matches_pushed_data():
    """Every field survives the fused H2D roundtrip, in FIFO order, across wraps."""
    owner, receiver = _owner_receiver()
    pushed = [_batch(t, seed=7) for t in range(CAPACITY * 3)]
    expected = 0

    def drain_and_verify() -> None:
        nonlocal expected
        # Emulate the cross-process roles: only the producer's own calls flush
        # its publish events (the collector polls is_full every step), so nudge
        # the owner side before the receiver looks.
        owner.size()
        while receiver.has_next():
            k, views = receiver.read_span()
            for i in range(k):
                f = pushed[expected + i]
                torch.testing.assert_close(views[0][i], f[0].cuda())
                torch.testing.assert_close(views[1][i], f[1].cuda())
                torch.testing.assert_close(views[2][i], f[2].cuda())
                torch.testing.assert_close(views[3][i], f[3].cuda())
                torch.testing.assert_close(views[4][i].long().cpu(), f[4])
                torch.testing.assert_close(views[5][i].long().cpu(), f[5])
            receiver.commit_reads(k)
            expected += k

    for fields in pushed:
        while not owner.push(*fields):
            drain_and_verify()  # free slots (the learner's role) and retry
        drain_and_verify()
    torch.cuda.synchronize()  # land the last push's event before the final drain
    drain_and_verify()
    assert expected == len(pushed)


@cuda_only
def test_ipc_ring_cursor_publishes_only_after_event():
    """A pushed slot is invisible until its event completes; size() flushes."""
    owner, receiver = _owner_receiver()
    fields = _batch(0, seed=1)
    assert owner.push(*fields)
    torch.cuda.synchronize()  # land the H2D; the cursor must catch up lazily
    assert owner.size() == 1
    assert receiver.has_next()


@cuda_only
def test_ipc_ring_backpressure_bounds_in_flight():
    owner, receiver = _owner_receiver()
    for t in range(CAPACITY):
        assert owner.push(*_batch(t, seed=3))
    assert owner.is_full()
    assert not owner.push(*_batch(CAPACITY, seed=3))
    # consumer frees everything -> producer can push again (across the wrap)
    while receiver.has_next():
        k, _ = receiver.read_span()
        receiver.commit_reads(k)
    assert not owner.is_full()
    assert owner.push(*_batch(CAPACITY, seed=3))


@cuda_only
def test_ipc_ring_back_to_back_pushes_do_not_collide():
    """Regression: pushes issued while the cursor lags (events in flight) must
    still target distinct slots and publish to distinct cursor values."""
    owner, receiver = _owner_receiver()
    pushed = [_batch(t, seed=9) for t in range(CAPACITY)]
    for fields in pushed:  # NO producer-side flush between pushes
        assert owner.push(*fields)
    torch.cuda.synchronize()
    owner.size()
    expected = 0
    while receiver.has_next():
        k, views = receiver.read_span()
        for i in range(k):
            torch.testing.assert_close(views[0][i], pushed[expected + i][0].cuda())
        receiver.commit_reads(k)
        expected += k
    assert expected == CAPACITY


@cuda_only
def test_ipc_and_host_ring_produce_identical_replay_buffers():
    """The learner's ingest path is transport-equivalent at the buffer level."""
    total = CAPACITY * 4
    host = SharedTransitionRing(total, N_ENV, OBS, CRI, ACT)
    rb_host = SimpleReplayBuffer(N_ENV, 2 * total, OBS, ACT, CRI, device="cpu")
    rb_ipc = SimpleReplayBuffer(N_ENV, 2 * total, OBS, ACT, CRI, device="cuda")
    owner, receiver = _owner_receiver()
    batches = [_batch(t, seed=11) for t in range(total)]
    for fields in batches:
        assert host.push(*fields)
        while not owner.push(*fields):
            while receiver.has_next():
                k, views = receiver.read_span()
                rb_ipc.extend_batch(*views)
                receiver.commit_reads(k)
    torch.cuda.synchronize()  # land the last push's event before the final drains
    owner.size()  # producer-side flush, as the collector's polling would do
    while host.has_next():
        _k, views = host.read_span()
        rb_host.extend_batch(*views)
        host.commit_reads(_k)
    while receiver.has_next():
        k, views = receiver.read_span()
        rb_ipc.extend_batch(*views)
        receiver.commit_reads(k)
    assert rb_ipc.ptr == rb_host.ptr == total
    for name in ("observations", "critic_observations", "actions", "rewards", "dones", "truncations"):
        assert torch.equal(getattr(rb_ipc, name).cpu(), getattr(rb_host, name)), name


def test_use_ipc_transition_ring_gating():
    from motrix_rl.fastsac.async_impl.worker import use_ipc_transition_ring

    def opts(mode):
        return SimpleNamespace(transition_ipc=mode)

    cpu, gpu, gpu0, gpu1 = (
        torch.device("cpu"),
        torch.device("cuda"),
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    )
    assert use_ipc_transition_ring(opts("off"), gpu0, gpu0) is False
    assert use_ipc_transition_ring(opts("auto"), gpu0, gpu0) is True
    # unspecified index means the default current device (cuda:0), NOT a
    # wildcard: it matches cuda:0 but never an explicit other GPU
    assert use_ipc_transition_ring(opts("auto"), gpu, gpu0) is True
    assert use_ipc_transition_ring(opts("auto"), gpu, gpu1) is False
    assert use_ipc_transition_ring(opts("auto"), cpu, gpu0) is False
    assert use_ipc_transition_ring(opts("auto"), gpu0, gpu1) is False
    assert use_ipc_transition_ring(opts("on"), gpu0, gpu1) is False  # warns + falls back
    assert use_ipc_transition_ring(opts("on"), cpu, gpu0) is False
    assert use_ipc_transition_ring(opts(True), gpu0, gpu0) is True  # YAML boolean form
    with pytest.raises(ValueError, match="transition_ipc"):
        use_ipc_transition_ring(opts("nope"), gpu0, gpu0)


@cuda_only
def test_learner_drains_ipc_ring_end_to_end():
    """Learner.drain on the IPC ring fills the GPU replay buffer correctly.

    Covers the full consumer path — strided fused views, float->int64 done
    conversion, rb wrap — interleaved with publish/commit laziness, mirroring
    the earlier batched-drain smoke test for the host ring.
    """
    from motrix_rl.fastsac.agent import FastSacAgent
    from motrix_rl.fastsac.async_impl.learner import Learner

    agent = FastSacAgent(
        obs_dim=OBS,
        critic_obs_dim=CRI,
        act_dim=ACT,
        num_envs=N_ENV,
        cfg=SimpleNamespace(
            actor_hidden_dim=16,
            critic_hidden_dim=16,
            num_q_networks=2,
            actor_learning_rate=1e-3,
            critic_learning_rate=1e-3,
            alpha_learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=0.0,
            use_layer_norm=False,
            use_tanh=True,
            log_std_max=0.0,
            log_std_min=-5.0,
            num_atoms=5,
            v_min=-20.0,
            v_max=20.0,
            gamma=0.97,
            tau=0.125,
            alpha_init=0.001,
            use_autotune=True,
            target_entropy_ratio=0.0,
            buffer_size=32,
            num_steps=1,
            batch_size=4,
            learning_starts=1,
            policy_frequency=4,
            num_updates=1,
            obs_normalization=False,
            compile=False,
            amp=False,
            amp_dtype="bf16",
            device=None,
        ),
        device=torch.device("cuda"),
    )
    owner, receiver = _owner_receiver()
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            async_options=SimpleNamespace(max_ingest_per_iter=CAPACITY, utd_mode="strict", weight_publish_interval=1)
        )
    )
    learner = Learner(agent, cfg, receiver, SimpleNamespace(publish=lambda *a, **k: None), SimpleNamespace())
    assert learner._staging is None and learner._device_ring

    total = 40  # > rb cap 33 -> exercises the rb wrap
    batches = [_batch(t, seed=5) for t in range(total)]
    for t, fields in enumerate(batches):
        while not owner.push(*fields):
            owner.size()
            learner.drain()
    owner.size()
    torch.cuda.synchronize()  # land the last publish event before the final drain
    owner.size()
    while receiver.has_next():
        learner.drain()
    learner.wait_ingest()
    torch.cuda.synchronize()
    assert agent.rb.ptr == total
    cap = 33
    for t in range(total - cap + 1, total):  # surviving slots after the wrap
        s = t % cap
        torch.testing.assert_close(agent.rb.observations[:, s].cpu(), batches[t][0])
        torch.testing.assert_close(agent.rb.critic_observations[:, s].cpu(), batches[t][1])
        torch.testing.assert_close(agent.rb.actions[:, s].cpu(), batches[t][2])
        torch.testing.assert_close(agent.rb.rewards[:, s].cpu(), batches[t][3])
        torch.testing.assert_close(agent.rb.dones[:, s].cpu(), batches[t][4])
        torch.testing.assert_close(agent.rb.truncations[:, s].cpu(), batches[t][5])
