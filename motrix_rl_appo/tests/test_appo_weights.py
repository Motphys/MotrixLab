# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APPO weight routing and spawned bundle snapshots; CUDA opt-in: APPO_TEST_CUDA=1."""

import multiprocessing as mp
import os
import traceback
from multiprocessing.connection import Connection

import pytest
import torch
from hydra.core.config_store import ConfigStore
from motrix_rl_appo.config import AppoCfg, NetworkCfg
from motrix_rl_appo.networks import build_actor, build_critic
from motrix_rl_appo.transport import CollectorMetadata, inference_spec, resolve_transport

from motrix_rl.transport.common import bind_flat_params
from motrix_rl.transport.weight_channel import (
    GpuIpcWeightReceiver,
    GpuIpcWeightSender,
    HostWeightReceiver,
    HostWeightSender,
    WeightChannelShared,
    weight_receiver_for,
)


@pytest.mark.parametrize(
    ("learner", "collector", "available", "expected"),
    [
        ("cpu", "cpu", False, "host"),
        ("cuda:0", "cpu", True, "host"),
        ("cuda:0", "cuda:1", True, "host"),
        ("cuda:0", "cuda:0", False, "host"),
        ("cuda", "cuda:0", True, "cuda_ipc"),
        ("cuda:1", "cuda:1", True, "cuda_ipc"),
        ("cuda:2", "cuda:2", True, "host"),
    ],
)
def test_weight_transport_resolution(monkeypatch, learner: str, collector: str, available: bool, expected: str) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    devices = torch.device(learner), torch.device(collector)
    # Weight routing uses the same topology rule, not the transition choice.
    cfg = AppoCfg(transition_transport="host", weight_transport="auto")
    assert resolve_transport(cfg.weight_transport, *devices) == expected
    assert resolve_transport("host", *devices) == "host"
    if expected == "cuda_ipc":
        assert resolve_transport("cuda_ipc", *devices) == expected
    else:
        with pytest.raises(ValueError, match="same available CUDA"):
            resolve_transport("cuda_ipc", *devices)


def test_weight_transport_config_validation() -> None:
    for mode in ("auto", "host", "cuda_ipc"):
        assert AppoCfg(weight_transport=mode).weight_transport == mode
    with pytest.raises(ValueError, match="weight_transport"):
        AppoCfg(weight_transport="invalid")
    with pytest.raises(ValueError, match="transport"):
        resolve_transport("invalid", torch.device("cpu"), torch.device("cpu"))


def test_builtin_appo_config_exposes_weight_transport() -> None:
    from motrix_rl.plugins import load_plugins

    load_plugins()
    node = ConfigStore.instance().load("algo_base/motrix.appo.yaml").node
    assert node.weight_transport == AppoCfg().weight_transport
    node.weight_transport = "host"
    cfg = AppoCfg(**{key: node[key] for key in ("transition_transport", "weight_transport")})
    assert cfg.weight_transport == "host"


def _weight_reader(pipe: Connection) -> None:
    try:
        shared, params = pipe.recv()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        cfg = NetworkCfg(hidden_dims=[4])
        actor = build_actor(cfg, 2, 1).to(device).eval()
        critic = build_critic(cfg, 3).to(device).eval()
        modules = {"actor": actor, "critic": critic}
        norms = {"actor_norm": actor.obs_normalizer, "critic_norm": critic.obs_normalizer}
        flat = {name: bind_flat_params(module) for name, module in modules.items()}
        pointers = {name: value.data_ptr() for name, value in flat.items()}
        for name, module in modules.items():
            for parameter in module.parameters():
                assert parameter.untyped_storage().data_ptr() == flat[name].untyped_storage().data_ptr()
        receiver = weight_receiver_for(shared, params)
        expected_type = GpuIpcWeightReceiver if params[0].is_cuda else HostWeightReceiver
        assert isinstance(receiver, expected_type)
        if params[0].is_cuda:
            with pytest.raises(ValueError, match="bound flat_params"):
                receiver.maybe_load(modules, norms)
        pipe.send("ready")
        while True:
            command = pipe.recv()
            if command is None:
                break
            version, expected = command
            assert receiver.maybe_load(modules, norms, flat)[0] == version
            assert receiver.loaded_version == version
            for name, module in modules.items():
                assert flat[name].data_ptr() == pointers[name]
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value.cpu(), expected[name][key])
            # No new publication leaves the loaded snapshot/version unchanged.
            assert receiver.maybe_load(modules, norms, flat)[0] == version
            pipe.send(version)
        del receiver, params
        pipe.send("released")
    except BaseException:
        pipe.send(traceback.format_exc())
        raise
    finally:
        pipe.close()


def _receive(pipe: Connection) -> str | int:
    assert pipe.poll(60), "spawned weight receiver did not respond"
    return pipe.recv()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("APPO_TEST_CUDA") != "1", reason="set APPO_TEST_CUDA=1 for real CUDA IPC")
@pytest.mark.parametrize("transport", ["host", "cuda_ipc"])
def test_cuda_spawn_actor_critic_normalizers_flat_binding_and_owner_lifetime(transport: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime required")
    device = torch.device("cuda:0")
    cfg = NetworkCfg(hidden_dims=[4])
    actor, critic = build_actor(cfg, 2, 1).to(device), build_critic(cfg, 3).to(device)
    modules = {"actor": actor, "critic": critic}
    norms = {"actor_norm": actor.obs_normalizer, "critic_norm": critic.obs_normalizer}
    spec = inference_spec(actor, critic, CollectorMetadata(2, 3, 1))
    shared = WeightChannelShared(spec)
    sender = GpuIpcWeightSender(shared, device) if transport == "cuda_ipc" else HostWeightSender(shared)
    sender.publish(modules, norms)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_weight_reader, args=(child,))
    process.start()
    child.close()
    try:
        parent.send((shared, sender.params))
        assert _receive(parent) == "ready"
        # Multiple wraps cover both slots; distinct module/stat values catch mixed bundles.
        for version in range(2, 8):
            with torch.no_grad():
                for index, module in enumerate(modules.values(), start=1):
                    for parameter in module.parameters():
                        parameter.fill_(version * index / 10)
                actor.obs_normalizer.update(torch.tensor([[version, version + 1.0]], device=device))
                critic.obs_normalizer.update(torch.tensor([[version + 2.0, version, -version]], device=device))
            sender.publish(modules, norms)
            expected = {
                name: {key: value.cpu().clone() for key, value in module.state_dict().items()}
                for name, module in modules.items()
            }
            parent.send((version, expected))
            assert _receive(parent) == version
            assert process.is_alive()
            assert sender.version == version
        parent.send(None)
        assert _receive(parent) == "released"
        process.join(60)
        assert process.exitcode == 0
        # Exporter remains strongly owned until the importing process has exited.
        assert len(sender.params) == 2
        assert all(value.device.type == ("cuda" if transport == "cuda_ipc" else "cpu") for value in sender.params)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        process.close()
