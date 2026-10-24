# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Snapshot consistency, independent auxiliary layouts and parameter binding."""

import pytest
import torch
from torch import nn

from motrix_rl.transport.snapshot import (
    GpuIpcSnapshotReceiver,
    GpuIpcSnapshotSender,
    HostSnapshotReceiver,
    HostSnapshotSender,
    SnapshotTensorSpec,
    TensorSnapshotShared,
    bind_flat_params,
    flatten_params,
    load_flat_params,
    snapshot_receiver_for,
)


def _state():
    # Deliberately not observation-normalizer state: sequence inference metadata.
    return {
        "temperature": torch.tensor([0.75], dtype=torch.float64),
        "action_mask": torch.tensor([[True, False], [False, True]]),
        "generation": torch.tensor([2**53 + 7], dtype=torch.int64),
    }


def _channel():
    auxiliary = _state()
    shared = TensorSnapshotShared(
        {
            name: SnapshotTensorSpec(tuple(value.shape), value.dtype, host_scalar=name == "generation")
            for name, value in auxiliary.items()
        }
    )
    actor = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    sender = HostSnapshotSender(shared, flatten_params(actor).numel())
    receiver = snapshot_receiver_for(shared, sender.params)
    target = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    targets = {name: torch.zeros_like(value) for name, value in auxiliary.items()}
    return shared, actor, auxiliary, sender, receiver, target, targets


@pytest.mark.parametrize("bound", [False, True])
def test_snapshot_preserves_named_auxiliary_dtypes_and_reuses_staging(bound):
    _, actor, auxiliary, sender, receiver, target, targets = _channel()
    flat = bind_flat_params(target) if bound else None
    sender.publish(actor, auxiliary)
    assert receiver.loaded_version == 0
    assert receiver.lag == 1
    version, wait, snapshot_s, load_s = receiver.maybe_load(target, targets, flat)
    assert version == 1
    assert wait == 0
    assert snapshot_s >= 0 and load_s >= 0
    torch.testing.assert_close(flatten_params(target), flatten_params(actor))
    for name in auxiliary:
        torch.testing.assert_close(targets[name], auxiliary[name], rtol=0, atol=0)
    assert targets["generation"].item() == 2**53 + 7
    pointers = {name: value.data_ptr() for name, value in receiver._aux_staging.items()}
    param_pointer = receiver._param_staging.data_ptr()
    with torch.no_grad():
        next(actor.parameters()).add_(1)
    auxiliary["generation"].add_(1)
    sender.publish(actor, auxiliary)
    assert receiver.maybe_load(target, targets, flat)[0] == 2
    assert receiver.lag == 0
    assert pointers == {name: value.data_ptr() for name, value in receiver._aux_staging.items()}
    assert param_pointer == receiver._param_staging.data_ptr()
    torch.testing.assert_close(flatten_params(target), flatten_params(actor))
    assert targets["generation"].item() == 2**53 + 8
    if flat is not None:
        assert flat.untyped_storage().data_ptr() == next(target.parameters()).untyped_storage().data_ptr()


def test_snapshot_retries_when_writer_reuses_slot_during_read():
    shared, actor, auxiliary, sender, receiver, target, targets = _channel()
    sender.publish(actor, auxiliary)
    snapshot = receiver._snapshot
    attempts = []

    def racing_snapshot(slot, flat_params, module):
        snapshot(slot, flat_params, module)
        attempts.append(slot)
        if len(attempts) == 1:
            # Publish twice: the writer reuses the slot we just copied. Params
            # and auxiliary state from this attempt must not be loaded together.
            with torch.no_grad():
                for parameter in actor.parameters():
                    parameter.fill_(4)
            auxiliary["generation"].fill_(99)
            auxiliary["temperature"].fill_(3)
            sender.publish(actor, auxiliary)
            sender.publish(actor, auxiliary)

    receiver._snapshot = racing_snapshot
    assert receiver.maybe_load(target, targets)[0] == 3
    assert len(attempts) == 2
    assert int(shared.seq[0]) == 6
    torch.testing.assert_close(flatten_params(target), flatten_params(actor))
    for name in auxiliary:
        torch.testing.assert_close(targets[name], auxiliary[name], rtol=0, atol=0)


def test_snapshot_poll_skips_writer_without_mutating_inference_state():
    shared, actor, auxiliary, sender, receiver, target, targets = _channel()
    sender.publish(actor, auxiliary)
    receiver.maybe_load(target, targets)
    before = flatten_params(target).clone()
    shared.seq[0] = 3
    assert receiver.maybe_load(target, targets) == (1, 0.0, 0.0, 0.0)
    torch.testing.assert_close(flatten_params(target), before)
    assert targets["generation"].item() == 2**53 + 7


def test_flat_parameters_reject_empty_actor_and_wrong_length_before_loading():
    with pytest.raises(ValueError, match="must have parameters"):
        flatten_params(nn.Identity())
    actor = nn.Linear(2, 1)
    before = flatten_params(actor).clone()
    with pytest.raises(ValueError, match="length"):
        load_flat_params(actor, torch.ones(before.numel() - 1))
    torch.testing.assert_close(flatten_params(actor), before)


def test_snapshot_without_auxiliary_state():
    shared = TensorSnapshotShared({})
    actor, target = nn.Linear(2, 1), nn.Linear(2, 1)
    sender = HostSnapshotSender(shared, flatten_params(actor).numel())
    receiver = HostSnapshotReceiver(shared, sender.params)
    assert receiver.maybe_load(target, {}) == (0, 0.0, 0.0, 0.0)
    sender.publish(actor, {})
    assert receiver.maybe_load(target, {})[0] == 1
    assert receiver.maybe_load(target, {}) == (1, 0.0, 0.0, 0.0)
    torch.testing.assert_close(flatten_params(target), flatten_params(actor))


@pytest.mark.parametrize("invalid", ["missing", "extra", "shape", "dtype", "length", "rank", "integer"])
def test_rejected_publish_keeps_completed_snapshot_and_seqlock(invalid):
    shared, actor, auxiliary, sender, receiver, target, targets = _channel()
    sender.publish(actor, auxiliary)
    original = flatten_params(actor).clone()
    flat = original + 10
    auxiliary = dict(auxiliary)
    if invalid == "missing":
        auxiliary.pop("temperature")
    elif invalid == "extra":
        auxiliary["extra"] = torch.zeros(1)
    elif invalid == "shape":
        auxiliary["temperature"] = torch.zeros(2, dtype=torch.float64)
    elif invalid == "dtype":
        auxiliary["temperature"] = torch.zeros(1, dtype=torch.float32)
    elif invalid == "length":
        flat = flat[:-1]
    elif invalid == "rank":
        flat = flat.unsqueeze(0)
    else:
        flat = flat.long()
    with pytest.raises(ValueError):
        sender.publish_flat(flat, auxiliary)
    assert int(shared.seq[0]) == 2
    assert receiver.maybe_load(target, targets)[0] == 1
    torch.testing.assert_close(flatten_params(target), original)
    assert targets["generation"].item() == 2**53 + 7


def test_rejected_load_does_not_mix_new_params_with_old_auxiliary():
    _, actor, auxiliary, sender, receiver, target, targets = _channel()
    sender.publish(actor, auxiliary)
    receiver.maybe_load(target, targets)
    before = flatten_params(target).clone()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(10)
    sender.publish(actor, auxiliary)
    with pytest.raises(ValueError, match="auxiliary schema mismatch"):
        receiver.maybe_load(target, {})
    assert receiver.loaded_version == 1
    torch.testing.assert_close(flatten_params(target), before)
    assert receiver.maybe_load(target, targets)[0] == 2
    torch.testing.assert_close(flatten_params(target), flatten_params(actor))


@pytest.mark.parametrize("writer_odd", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=[
                pytest.mark.integration,
                pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA snapshot path requires a GPU"),
            ],
        ),
    ],
)
def test_ipc_rejected_snapshot_never_mutates_bound_actor(monkeypatch, writer_odd, device):
    # CPU tensors plus a no-op stream exercise IPC's staging/commit logic without
    # requiring a GPU; real CUDA ordering is covered by the integration test.
    from types import SimpleNamespace

    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: None))
    shared = TensorSnapshotShared({})
    actor, target = nn.Linear(2, 1).to(device), nn.Linear(2, 1).to(device)
    sender = (
        HostSnapshotSender(shared, flatten_params(actor).numel())
        if device == "cpu"
        else GpuIpcSnapshotSender(shared, flatten_params(actor).numel(), torch.device(device))
    )
    receiver = GpuIpcSnapshotReceiver(shared, sender.params)
    flat = bind_flat_params(target)
    before = flat.clone()
    sender.publish(actor, {})
    snapshot = receiver._snapshot
    attempts = []

    def racing_snapshot(slot, flat_params, module):
        snapshot(slot, flat_params, module)
        torch.testing.assert_close(flat, before)
        attempts.append(slot)
        if len(attempts) == 1:
            if writer_odd:
                shared.seq[0] = 3
            else:
                with torch.no_grad():
                    for parameter in actor.parameters():
                        parameter.fill_(9)
                sender.publish(actor, {})
                sender.publish(actor, {})

    receiver._snapshot = racing_snapshot
    version = receiver.maybe_load(target, {}, flat)[0]
    if writer_odd:
        assert version == receiver.loaded_version == 0
        torch.testing.assert_close(flat, before)
        shared.seq[0] = 2
        receiver._snapshot = snapshot
        assert receiver.maybe_load(target, {}, flat)[0] == 1
    else:
        assert version == 3
        assert len(attempts) == 2
    torch.testing.assert_close(flat, flatten_params(actor))
    assert receiver._param_staging.data_ptr() != flat.data_ptr()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA snapshot path requires a GPU")
@pytest.mark.parametrize("ipc", [False, True])
def test_cuda_snapshot_paths_preserve_parameters_and_auxiliary_state(ipc):
    shared, actor, auxiliary, _, _, target, targets = _channel()
    actor, target = actor.cuda(), target.cuda()
    auxiliary = {name: value.cuda() for name, value in auxiliary.items()}
    targets = {name: value.cuda() for name, value in targets.items()}
    sender = (
        GpuIpcSnapshotSender(shared, flatten_params(actor).numel(), torch.device("cuda:0"))
        if ipc
        else HostSnapshotSender(shared, flatten_params(actor).numel())
    )
    receiver = snapshot_receiver_for(shared, sender.params)
    flat = bind_flat_params(target)
    sender.publish(actor, auxiliary)
    assert receiver.maybe_load(target, targets, flat)[0] == 1
    torch.cuda.synchronize()
    torch.testing.assert_close(flatten_params(target), flatten_params(actor))
    for name in auxiliary:
        torch.testing.assert_close(targets[name], auxiliary[name], rtol=0, atol=0)
