# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Snapshot consistency, independent auxiliary layouts and parameter binding."""

import pytest
import torch
from torch import nn

from motrix_rl.transport.snapshot import (
    GpuIpcSnapshotSender,
    HostSnapshotReceiver,
    HostSnapshotSender,
    SnapshotTensorSpec,
    TensorSnapshotShared,
    bind_flat_params,
    flatten_params,
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
