# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch
from motrix_rl_appo.collector import publish_chunk
from motrix_rl_appo.config import AppoCfg, NetworkCfg
from motrix_rl_appo.networks import build_actor, build_critic
from motrix_rl_appo.transport import (
    CollectorMetadata,
    chunk_schema,
    inference_spec,
    resolve_transport,
    take_chunk,
)

from motrix_rl.transport.common import bind_flat_params
from motrix_rl.transport.ring import HostTransitionRing
from motrix_rl.transport.weight_channel import HostWeightSender, WeightChannelShared, weight_receiver_for


def test_owned_chunk_survives_ring_wrap_and_consumes_one() -> None:
    dims = CollectorMetadata(2, 3, 1)
    schema = chunk_schema(4, dims)
    ring = HostTransitionRing(schema, 2, 3)
    payload = {field.name: torch.ones(schema.slot_shape(field, 3), dtype=field.dtype) for field in schema.fields}
    assert ring.push(payload)
    assert ring.push(payload)
    batch = take_chunk(ring, torch.device("cpu"))
    assert ring.size() == 1
    assert batch["obs"].shape == (4, 3, 2)
    assert batch["last_critic_obs"].shape == (3, 3)
    for tensor in payload.values():
        tensor.zero_()
    assert ring.push(payload)
    assert batch["dones"].dtype == torch.bool
    assert batch["dones"].all()
    assert all(value.is_contiguous() for value in batch.values())
    assert torch.all(batch["obs"] == 1)
    assert torch.all(batch["last_critic_obs"] == 1)
    take_chunk(ring, torch.device("cpu"))
    assert torch.all(take_chunk(ring, torch.device("cpu"))["obs"] == 0)


@pytest.mark.parametrize(
    ("learner", "collector", "expected"),
    [
        ("cpu", "cpu", "host"),
        ("cuda:0", "cpu", "host"),
        ("cuda:0", "cuda:1", "host"),
        ("cuda", "cuda:0", "cuda_ipc"),
        ("cuda:1", "cuda:1", "cuda_ipc"),
    ],
)
def test_transport_device_selection(monkeypatch, learner: str, collector: str, expected: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    devices = (torch.device(learner), torch.device(collector))
    assert resolve_transport("auto", *devices) == expected
    assert resolve_transport("host", *devices) == "host"
    if expected == "host":
        with pytest.raises(ValueError, match="same available CUDA"):
            resolve_transport("cuda_ipc", *devices)
    else:
        assert resolve_transport("cuda_ipc", *devices) == expected


def test_unavailable_cuda_and_invalid_transport(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    devices = (torch.device("cuda:0"), torch.device("cuda:0"))
    assert resolve_transport("auto", *devices) == "host"
    with pytest.raises(ValueError, match="same available CUDA"):
        resolve_transport("cuda_ipc", *devices)
    with pytest.raises(ValueError, match="transport"):
        AppoCfg(transition_transport="invalid")


def test_flattened_float_views_restore_schema_and_bool() -> None:
    schema = chunk_schema(3, CollectorMetadata(2, 4, 1))

    class FlatRing:
        num_envs = 2
        committed = 0

        def __init__(self) -> None:
            self.schema = schema
            self.views = {
                field.name: torch.arange(2 * field.per_env_numel, dtype=torch.float32).reshape(1, 2, -1)
                for field in schema.fields
            }
            self.views["dones"] = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]])

        def read_span(self) -> tuple[int, dict[str, torch.Tensor]]:
            return 2, self.views

        def commit_reads(self, count: int) -> None:
            self.committed += count
            # Simulate immediate producer reuse at acknowledgement.
            for view in self.views.values():
                view.zero_()

    ring = FlatRing()
    batch = take_chunk(ring, torch.device("cpu"))
    assert ring.committed == 1
    assert batch["obs"].shape == (3, 2, 2)
    assert batch["last_critic_obs"].shape == (2, 4)
    assert batch["actions"].shape == (3, 2, 1)
    torch.testing.assert_close(batch["dones"], torch.tensor([[False, True], [True, False], [False, True]]))
    torch.testing.assert_close(batch["obs"], torch.arange(12.0).reshape(2, 3, 2).transpose(0, 1))
    assert all(value.is_contiguous() for value in batch.values())


class DelayedPublicationRing:
    """Model a push whose write event needs endpoint polling to publish."""

    def __init__(self, polls: int = 3) -> None:
        self.polls = polls
        self.write_issued_count = 0
        self.completed = 0

    def push(self, chunk: dict[str, torch.Tensor]) -> bool:
        self.write_issued_count += 1
        return True

    @property
    def write_completed_count(self) -> int:
        self.polls -= 1
        if self.polls <= 0:
            self.completed = self.write_issued_count
        return self.completed


def test_last_chunk_is_published_without_another_push() -> None:
    ring = DelayedPublicationRing()
    assert publish_chunk(ring, {}, threading.Event(), 1)
    assert ring.completed == ring.write_issued_count == 1


def test_publication_wait_obeys_stop_and_timeout() -> None:
    stopped = threading.Event()
    stopped.set()
    ring = DelayedPublicationRing(polls=100000)
    assert not publish_chunk(ring, {}, stopped, 1)
    assert ring.write_issued_count == 0

    class StopDuringPublication(DelayedPublicationRing):
        @property
        def write_completed_count(self) -> int:
            stopped.set()
            return 0

    stopped.clear()
    pending = StopDuringPublication()
    assert not publish_chunk(pending, {}, stopped, 1)
    assert pending.write_issued_count == 1
    with pytest.raises(TimeoutError, match="write did not complete"):
        publish_chunk(ring, {}, threading.Event(), 0.001)
    full = HostTransitionRing(chunk_schema(2, CollectorMetadata(1, 1, 1)), 1, 1)
    payload = {
        field.name: torch.zeros(full.schema.slot_shape(field, 1), dtype=field.dtype) for field in full.schema.fields
    }
    assert publish_chunk(full, payload, threading.Event(), 1)
    with pytest.raises(TimeoutError, match="remained full"):
        publish_chunk(full, payload, threading.Event(), 0.001)


@pytest.mark.parametrize("normalize", [False, True])
def test_real_host_weights_transport_actor_critic_and_normalizers(normalize: bool) -> None:
    dims = CollectorMetadata(2, 3, 1)
    cfg = NetworkCfg(hidden_dims=[4], normalize_observations=normalize)
    actor, critic = build_actor(cfg, 2, 1), build_critic(cfg, 3)
    dst_actor, dst_critic = build_actor(cfg, 2, 1), build_critic(cfg, 3)
    if normalize:
        actor.obs_normalizer.update(torch.tensor([[2.0, 4.0], [4.0, 8.0]]))
        critic.obs_normalizer.update(torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
    modules = {"actor": actor, "critic": critic}
    norms = {"actor_norm": actor.obs_normalizer, "critic_norm": critic.obs_normalizer}
    shared = WeightChannelShared(inference_spec(actor, critic, dims))
    sender = HostWeightSender(shared)
    sender.publish(modules, norms)
    receiver = weight_receiver_for(shared, sender.params)
    destinations = {"actor": dst_actor, "critic": dst_critic}
    # Exercise actual contiguous parameter binding, not a fake transport API.
    flat = {name: bind_flat_params(module) for name, module in destinations.items()}
    receiver.maybe_load(
        destinations, {"actor_norm": dst_actor.obs_normalizer, "critic_norm": dst_critic.obs_normalizer}, flat
    )
    assert receiver.loaded_version == 1
    for source, destination in ((actor, dst_actor), (critic, dst_critic)):
        for key, tensor in source.state_dict().items():
            torch.testing.assert_close(destination.state_dict()[key], tensor)
