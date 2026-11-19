# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Collector snapshots preserve policy bundles and reset boundaries."""

from threading import Event
from types import SimpleNamespace

import pytest
import torch
from _helpers import make_toy
from motrix_rl_supermimic.config import AgentCfg
from motrix_rl_supermimic.networks import NormStatsView, build_policy
from motrix_rl_supermimic.randomness import derive_seed
from motrix_rl_supermimic.toy import ToyEnvCfg, _flat_state
from motrix_rl_supermimic.toy import register as register_toy
from motrix_rl_supermimic.transition import transition_schema
from motrix_rl_supermimic.transport import HostRingConnection

from motrix_rl.transport.inference_bundle import InferenceBundleSpec, ModuleSection, NormalizerSection
from motrix_rl.transport.ring import HostTransitionRing
from motrix_rl.transport.weight_channel import (
    GpuIpcWeightReceiver,
    GpuIpcWeightSender,
    HostWeightReceiver,
    HostWeightSender,
    WeightChannelShared,
)
from motrix_rl_supermimic import collector


@pytest.mark.parametrize(
    ("policy_device", "weight_ipc"),
    [
        ("cpu", False),
        pytest.param(
            "cuda:0",
            False,
            marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")],
        ),
        pytest.param(
            "cuda:0",
            True,
            marks=[pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")],
        ),
    ],
)
def test_policy_bundle_and_transition_snapshots(monkeypatch, policy_device, weight_ipc):
    register_toy()
    cfg = ToyEnvCfg(goal_frames=2)
    manager = make_toy(num_envs=2, cfg=cfg)
    env = manager
    agent = AgentCfg(hidden=16, z_dim=8)
    spec = env.state_spec
    goal_dim = env.observation_spec["goal"].space.shape[0]
    obs_dim = env.observation_spec.policy.frame_space.shape[0]
    action_dim = env.action_space.shape[0]
    live = build_policy(
        agent,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        device="cpu",
        observation_spec=env.observation_spec.policy,
    )
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.zero_()
        live.dec[-1].bias.fill_(0.25)
        for norm in (live.obs_norm, live.goal_norm):
            norm.mean.fill_(2.0)
            norm.var.fill_(3.0)
            norm.count.fill_(11.0)
    modules = {"policy": live}
    normalizers = {}
    sections = []
    for name, policy in modules.items():
        for field, dim in (("obs_norm", obs_dim), ("goal_norm", goal_dim)):
            key = field
            normalizers[key] = NormStatsView(getattr(policy, field))
            sections.append(NormalizerSection(key, dim))
    shared = WeightChannelShared(
        InferenceBundleSpec(
            modules=tuple(
                ModuleSection(name, sum(p.numel() for p in policy.parameters())) for name, policy in modules.items()
            ),
            normalizers=tuple(sections),
        )
    )
    live.to(policy_device)
    sender = GpuIpcWeightSender(shared, torch.device(policy_device)) if weight_ipc else HostWeightSender(shared)
    sender.publish(modules, normalizers)
    stop = Event()
    actions, errors = [], []
    snapshots = []
    original_snapshot = env.snapshot_transition

    def snapshot_transition(result):
        snapshots.append(result)
        return original_snapshot(result)

    monkeypatch.setattr(env, "snapshot_transition", snapshot_transition)
    initializations = []
    original_init = manager.init_state

    def init_state():
        initializations.append(True)
        # Extra environment/default-stream draws cannot perturb exploration.
        torch.rand(37)
        result = original_init()
        expected_states.append(_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity")))
        return result

    monkeypatch.setattr(manager, "init_state", init_state)
    transported = []
    original_step = manager.step
    expected_states = []

    def forbidden_build(*args, **kwargs):
        pytest.fail("Collector must not construct an imagined TaskMDP")

    import motrix_rl_supermimic.mbrl_env as _mbrl_env

    monkeypatch.setattr(_mbrl_env, "assemble_task", forbidden_build)

    loaded = []
    receiver_for = collector.weight_receiver_for

    def capture_receiver(shared, params):
        receiver = receiver_for(shared, params)
        assert isinstance(receiver, GpuIpcWeightReceiver if weight_ipc else HostWeightReceiver)
        maybe_load = receiver.maybe_load

        def load(modules, normalizers, flat_params):
            result = maybe_load(modules, normalizers, flat_params)
            loaded.append(modules)
            return result

        monkeypatch.setattr(receiver, "maybe_load", load)
        return receiver

    monkeypatch.setattr(collector, "weight_receiver_for", capture_receiver)

    def step(action):
        actions.append(action.clone())
        return original_step(action)

    parent_ring = HostTransitionRing(transition_schema(spec, action_dim=action_dim), 4, 2)
    connection = HostRingConnection(parent_ring.schema, 4, 2, parent_ring.cursors, parent_ring.data)
    handshake = []
    metadata = []

    class Startup:
        def send(self, message):
            assert message[0] == spec
            metadata.append(message)
            handshake.append("metadata")

        def recv(self):
            assert handshake == ["metadata"]
            handshake.append("connection")
            return connection

        def close(self):
            if handshake[-1] != "closed":
                handshake.append("closed")

    connect = collector.connect_producer

    def connect_producer(received):
        assert received is connection
        producer = connect(received)
        push = producer.push

        def capture(row):
            transported.append({name: value.clone() for name, value in row.items()})
            result = push(row)
            if len(transported) == 2:
                stop.set()
            return result

        producer.push = capture
        return producer

    monkeypatch.setattr(collector, "connect_producer", connect_producer)
    monkeypatch.setattr(manager, "step", step)

    def make(**kwargs):
        environment_seed = derive_seed(0, "environment", worker_id=0)
        assert kwargs["seed"] == environment_seed
        expected_rng = torch.Generator(device="cpu").manual_seed(environment_seed)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng.get_state())
        return manager

    monkeypatch.setattr(collector.env_registry.EnvBuildSpec, "make", lambda self, **kwargs: make(**kwargs))
    args = (
        "supermimic-toy",
        None,
        2,
        0,
        0.1,
        20.0,
        1,
        agent,
        policy_device,
        shared,
        sender.params,
        stop,
        SimpleNamespace(send=errors.append),
        Startup(),
    )
    collector.collector_process(*args)
    assert handshake == ["metadata", "connection", "closed"]
    assert parent_ring.write_completed_count == 2
    assert initializations == [True]
    assert errors == []
    assert len(actions) == 2
    assert len(snapshots) == 2
    exploration = torch.Generator(device="cpu").manual_seed(derive_seed(0, "exploration", worker_id=0))
    for action in actions:
        expected = 0.25 + 0.1 * torch.randn(action.shape, generator=exploration)
        torch.testing.assert_close(action, expected)
    assert metadata[0][0] == spec
    torch.testing.assert_close(transported[0]["s"], expected_states[0])
    torch.testing.assert_close(transported[0]["a"], actions[0])
    snapshot = transported[0]["s1"].clone()
    env.step(torch.zeros(2, action_dim))
    torch.testing.assert_close(transported[0]["s1"], snapshot)
    policy = loaded[-1]["policy"]
    expected = live
    for actual, wanted in zip(policy.parameters(), expected.parameters()):
        torch.testing.assert_close(actual, wanted)
    for field in ("obs_norm", "goal_norm"):
        actual, wanted = getattr(policy, field), getattr(expected, field)
        torch.testing.assert_close(actual.mean, wanted.mean)
        torch.testing.assert_close(actual.var, wanted.var)
        torch.testing.assert_close(actual.count, wanted.count)
    with torch.inference_mode():
        output = policy(
            torch.zeros(2, env.observation_spec.policy.history_length, obs_dim, device=policy_device),
            torch.zeros(2, goal_dim, device=policy_device),
        )
    torch.testing.assert_close(output.cpu(), torch.full((2, action_dim), 0.25))


def test_collector_preserves_block_and_healthy_episode_on_numerical_failure(monkeypatch):
    """Engine termination resets only the failed lane through the normal lifecycle."""
    register_toy()
    cfg = ToyEnvCfg(goal_frames=2)
    manager = make_toy(num_envs=2, cfg=cfg)
    env = manager
    manager.init_state()
    agent = AgentCfg(hidden=16, z_dim=8)
    spec = env.state_spec
    goal_dim = env.observation_spec["goal"].space.shape[0]
    obs_dim = env.observation_spec.policy.frame_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = build_policy(
        agent,
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        action_dim=action_dim,
        device="cpu",
        observation_spec=env.observation_spec.policy,
    )
    shared = WeightChannelShared(
        InferenceBundleSpec(
            modules=(ModuleSection("policy", sum(p.numel() for p in policy.parameters())),),
            normalizers=(
                NormalizerSection("obs_norm", obs_dim),
                NormalizerSection("goal_norm", goal_dim),
            ),
        )
    )
    sender = HostWeightSender(shared)
    sender.publish(
        {"policy": policy},
        {"obs_norm": NormStatsView(policy.obs_norm), "goal_norm": NormStatsView(policy.goal_norm)},
    )
    stop = Event()
    pushes, errors, reinits = [], [], []
    original_init = manager.init_state
    original_sim_step = manager.sim.step
    failed = []

    def sim_step(substeps, result=None):
        original_sim_step(substeps, result)
        if result is not None:
            if not failed:
                failed.append(True)
                result[1] = 7  # first block: one row fails numerically
            else:
                result.zero_()  # backend storage is overwritten every step

    parent_ring = HostTransitionRing(transition_schema(spec, action_dim=action_dim), 4, 2)
    connection = HostRingConnection(parent_ring.schema, 4, 2, parent_ring.cursors, parent_ring.data)
    handshake = []

    class Startup:
        def send(self, message):
            assert message[0] == spec
            handshake.append("metadata")

        def recv(self):
            assert handshake == ["metadata"]
            handshake.append("connection")
            return connection

        def close(self):
            if handshake[-1] != "closed":
                handshake.append("closed")

    connect = collector.connect_producer

    def connect_producer(received):
        assert received is connection
        producer = connect(received)
        push = producer.push

        def capture(row):
            assert manager.state.episode_steps[0] == len(pushes) + 1
            pushes.append(row)
            result = push(row)
            if len(pushes) == 2:
                stop.set()
            return result

        producer.push = capture
        return producer

    monkeypatch.setattr(collector, "connect_producer", connect_producer)
    monkeypatch.setattr(manager, "init_state", lambda: (reinits.append(True), original_init())[1])
    monkeypatch.setattr(manager.sim, "step", sim_step)
    monkeypatch.setattr(collector.env_registry.EnvBuildSpec, "make", lambda self, **kw: manager)
    collector.collector_process(
        "supermimic-toy",
        None,
        2,
        0,
        0.0,
        20.0,
        2,  # both transitions are retained even when one lane terminates
        agent,
        "cpu",
        shared,
        sender.params,
        stop,
        SimpleNamespace(send=errors.append),
        Startup(),
    )
    assert errors == []
    assert handshake == ["metadata", "connection", "closed"]
    assert parent_ring.write_completed_count == 2
    assert len(pushes) == 2
    assert pushes[0]["done"].tolist() == [False, True]
    assert pushes[1]["done"].tolist() == [False, False]
    assert pushes[0]["ep_step"].tolist() == [0, 0]
    assert pushes[1]["ep_step"].tolist() == [1, 0]
    assert env.state.episode_steps.tolist() == [2, 1]
    # Startup initializes once; the failed lane's ordinary auto-reset is enough.
    assert len(reinits) == 1
