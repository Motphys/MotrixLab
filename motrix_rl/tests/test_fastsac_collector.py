# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from types import SimpleNamespace

import pytest
import torch

from motrix_env_core.perf import Perf
from motrix_rl.fastsac.async_impl.collector import Collector, resolve_collector_inference_device
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing
from motrix_rl.fastsac.async_impl.shm.weight_channel import HostWeightReceiver, HostWeightSender, WeightChannelShared
from motrix_rl.fastsac.buffer import EmpiricalNormalization
from motrix_rl.fastsac.networks import Actor

_NUM_ENVS = 8
_OBS_DIM = 6
_CRITIC_OBS_DIM = 9
_ACT_DIM = 3


class _CpuEnv:
    """Wrapper stand-in: the collector reads the original env via ``.env``."""

    def __init__(self):
        self.num_envs = _NUM_ENVS
        self.last_info = {}
        self.last_actions = None

    @property
    def env(self):
        return self

    def reset(self):
        return torch.zeros(_NUM_ENVS, _OBS_DIM), torch.zeros(_NUM_ENVS, _CRITIC_OBS_DIM)

    def step(self, actions):
        assert actions.device.type == "cpu"
        self.last_actions = actions.clone()
        return (
            torch.ones(_NUM_ENVS, _OBS_DIM),
            torch.ones(_NUM_ENVS, _CRITIC_OBS_DIM),
            torch.ones(_NUM_ENVS),
            torch.zeros(_NUM_ENVS, dtype=torch.bool),
            torch.zeros(_NUM_ENVS, dtype=torch.bool),
        )


class _PerfEnv(_CpuEnv):
    """Env exposing the core Perf contract the collector's panel hooks into."""

    def __init__(self):
        super().__init__()
        self.perf = Perf()

    def step(self, actions):
        with self.perf.scope("step"):
            with self.perf.scope("apply_action"):
                pass
            with self.perf.scope("physics"):
                with self.perf.scope("read"):
                    result = super().step(actions)
        return result


def _cfg(device: str, *, compile: bool = False, amp: bool = False):
    return SimpleNamespace(
        agent=SimpleNamespace(
            actor_hidden_dim=32,
            log_std_max=0.0,
            log_std_min=-5.0,
            use_tanh=True,
            use_layer_norm=True,
            obs_normalization=True,
            learning_starts=0,
        ),
        trainer=SimpleNamespace(
            async_options=SimpleNamespace(
                collector_inference_device=device,
                collector_compile=compile,
                collector_amp=amp,
                collector_amp_dtype="fp16",
                weight_poll_interval=1,
            )
        ),
    )


def _source_policy():
    actor = Actor(
        n_obs=_OBS_DIM,
        n_act=_ACT_DIM,
        hidden_dim=32,
        log_std_max=0.0,
        log_std_min=-5.0,
        use_tanh=True,
        use_layer_norm=True,
        action_scale=torch.tensor([1.0, 2.0, 3.0]),
        action_bias=torch.tensor([0.0, 0.5, -0.5]),
        device="cpu",
    )
    torch.manual_seed(11)
    with torch.no_grad():
        for param in actor.parameters():
            param.uniform_(-0.1, 0.1)
    normalizer = EmpiricalNormalization(_OBS_DIM, device="cpu")
    normalizer._mean.fill_(0.25)
    normalizer._std.fill_(1.5)
    normalizer._var.fill_(2.25)
    normalizer.count.fill_(123)
    return actor, normalizer


def _collector(device: str, *, compile: bool = False, amp: bool = False):
    env = _CpuEnv()
    cfg = _cfg(device, compile=compile, amp=amp)
    ring = SharedTransitionRing(2, _NUM_ENVS, _OBS_DIM, _CRITIC_OBS_DIM, _ACT_DIM)
    source_actor, source_normalizer = _source_policy()
    shared = WeightChannelShared(_OBS_DIM)
    weight_tx = HostWeightSender(shared, sum(p.numel() for p in source_actor.parameters()))
    weight_rx = HostWeightReceiver(shared, weight_tx.params)
    weight_tx.publish(source_actor, source_normalizer)
    collector = Collector(
        env,
        cfg,
        _OBS_DIM,
        _CRITIC_OBS_DIM,
        _ACT_DIM,
        source_actor.action_scale,
        source_actor.action_bias,
        ring,
        weight_rx,
        Control(),
    )
    collector.reset()
    collector.sync_weights()
    return collector, source_actor, source_normalizer, weight_tx


def test_weight_publish_prepares_params_before_opening_seqlock_write(monkeypatch) -> None:
    actor, normalizer = _source_policy()
    shared = WeightChannelShared(_OBS_DIM)
    weight_tx = HostWeightSender(shared, sum(param.numel() for param in actor.parameters()))
    weight_rx = HostWeightReceiver(shared, weight_tx.params)
    weight_channel_module = sys.modules[WeightChannelShared.__module__]
    original_flatten = weight_channel_module.flatten_params
    observed_sequences = []

    def observe_flatten(module):
        observed_sequences.append(int(weight_tx.shared.seq[0]))
        return original_flatten(module)

    monkeypatch.setattr(weight_channel_module, "flatten_params", observe_flatten)
    weight_tx.publish(actor, normalizer)
    weight_tx.publish(actor, normalizer)

    assert observed_sequences == [0, 2]
    assert weight_tx.version == 2
    assert weight_rx.version == 2
    assert weight_rx.lag == 2


def test_seqlock_read_is_nonblocking_during_publish() -> None:
    """A mid-publish (odd seq) poll returns the current version immediately.

    Contract: the collector never busy-waits for the learner's publish window
    (it spans the learner's in-flight gradient kernels); weights are
    eventually consistent and the next poll picks up the new version.
    """
    actor, normalizer = _source_policy()
    shared = WeightChannelShared(_OBS_DIM)
    weight_tx = HostWeightSender(shared, sum(param.numel() for param in actor.parameters()))
    weight_rx = HostWeightReceiver(shared, weight_tx.params)
    weight_tx.publish(actor, normalizer)

    shared.seq[0] = 3  # simulate a publish in progress (odd)
    start = time.perf_counter()
    version, wait_writer_s, _, _ = weight_rx.maybe_load(actor, normalizer)
    elapsed = time.perf_counter() - start

    assert version == 0  # keeps the receiver's current version (nothing loaded yet)
    assert wait_writer_s == 0.0
    assert elapsed < 1.0  # returned immediately, did not spin until seq closes

    shared.seq[0] = 4  # publish completed
    version, _, _, _ = weight_rx.maybe_load(actor, normalizer)
    assert version == 2  # picked up on the next poll


def test_collector_explicit_cpu_placement_and_timing() -> None:
    collector, source_actor, source_normalizer, _ = _collector("cpu")

    assert collector.device.type == "cpu"
    assert collector.obs.device.type == "cpu"
    assert all(param.device.type == "cpu" for param in collector.actor.parameters())
    assert not collector.weights._param_staging.is_pinned()
    assert all(not buffer.is_pinned() for buffer in collector.weights._norm_staging)
    torch.testing.assert_close(
        torch.cat([p.detach().flatten() for p in collector.actor.parameters()]),
        torch.cat([p.detach().flatten() for p in source_actor.parameters()]),
    )
    torch.testing.assert_close(collector.obs_normalizer._mean, source_normalizer._mean)

    assert collector.step_once()
    stats = collector.snapshot_stats()
    assert "collect" in stats["timing_ms"]
    assert "sample_actions" in stats["timing_ms"]
    assert "sync" in stats["timing_ms"]
    assert "sync.wait_writer" in stats["timing_ms"]
    assert "sync.host_snapshot" in stats["timing_ms"]
    assert "sync.actor_load" in stats["timing_ms"]
    assert collector.ring.obs.device.type == "cpu"
    assert collector.ring.critic_obs.device.type == "cpu"


def test_collector_reports_env_step_substage_timing() -> None:
    env = _PerfEnv()
    cfg = _cfg("cpu")
    ring = SharedTransitionRing(2, _NUM_ENVS, _OBS_DIM, _CRITIC_OBS_DIM, _ACT_DIM)
    source_actor, source_normalizer = _source_policy()
    shared = WeightChannelShared(_OBS_DIM)
    weight_tx = HostWeightSender(shared, sum(p.numel() for p in source_actor.parameters()))
    weight_rx = HostWeightReceiver(shared, weight_tx.params)
    weight_tx.publish(source_actor, source_normalizer)
    collector = Collector(
        env,
        cfg,
        _OBS_DIM,
        _CRITIC_OBS_DIM,
        _ACT_DIM,
        source_actor.action_scale,
        source_actor.action_bias,
        ring,
        weight_rx,
        Control(),
    )
    collector.reset()
    collector.sync_weights()

    assert collector.step_once()
    stats = collector.snapshot_stats()

    assert "env_step" in stats["timing_ms"]
    assert "env_step.apply_action" in stats["timing_ms"]
    assert "env_step.physics" in stats["timing_ms"]
    # nested sub-stages arrive as dotted paths for the panel's tree rebuild
    assert "env_step.physics.read" in stats["timing_ms"]
    assert stats["timing_ms"]["env_step.apply_action"] >= 0.0
    # sub-stage aggregation is windowed like the other timings
    assert env.perf.snapshot() == ()


def test_collector_cuda_request_fails_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requested CUDA, but CUDA is unavailable"):
        resolve_collector_inference_device("cuda")


def test_collector_rejects_non_cpu_cuda_device() -> None:
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        resolve_collector_inference_device("meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device-index test requires a GPU")
def test_collector_rejects_unavailable_cuda_index() -> None:
    invalid_device = f"cuda:{torch.cuda.device_count()}"
    with pytest.raises(RuntimeError, match="but only"):
        resolve_collector_inference_device(invalid_device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA collector test requires a GPU")
def test_cuda_collector_uses_flat_weight_copy_and_reuses_staging() -> None:
    collector, source_actor, source_normalizer, weight_tx = _collector("cuda")

    assert collector.device.type == "cuda"
    assert collector.obs.device.type == "cpu"
    assert all(param.device.type == "cuda" for param in collector.actor.parameters())
    assert collector.obs_normalizer._mean.device.type == "cuda"
    assert collector.ring.obs.device.type == "cpu"
    assert collector.ring.critic_obs.device.type == "cpu"
    assert (
        collector._flat_params.untyped_storage().data_ptr()
        == next(collector.actor.parameters()).untyped_storage().data_ptr()
    )
    torch.testing.assert_close(
        collector._flat_params.cpu(), torch.cat([p.detach().flatten() for p in source_actor.parameters()])
    )
    torch.testing.assert_close(collector.obs_normalizer._mean.cpu(), source_normalizer._mean)

    with torch.no_grad():
        source_actor.fc_mu.bias.add_(0.25)
        source_normalizer._mean.add_(0.5)
    weight_tx.publish(source_actor, source_normalizer)
    assert collector.policy_lag == 1
    collector.sync_weights()
    assert collector.policy_lag == 0
    torch.testing.assert_close(
        collector._flat_params.cpu(), torch.cat([p.detach().flatten() for p in source_actor.parameters()])
    )
    torch.testing.assert_close(collector.obs_normalizer._mean.cpu(), source_normalizer._mean)

    pointers = (
        collector._obs_host.data_ptr(),
        collector._obs_device.data_ptr(),
        collector._actions_host.data_ptr(),
        collector.weights._param_staging.data_ptr(),
        *(buffer.data_ptr() for buffer in collector.weights._norm_staging),
        collector._flat_params.data_ptr(),
    )
    first = collector._infer(torch.randn(_NUM_ENVS, _OBS_DIM))
    second = collector._infer(torch.randn(_NUM_ENVS, _OBS_DIM))
    assert first.data_ptr() == second.data_ptr() == collector._actions_host.data_ptr()
    assert pointers == (
        collector._obs_host.data_ptr(),
        collector._obs_device.data_ptr(),
        collector._actions_host.data_ptr(),
        collector.weights._param_staging.data_ptr(),
        *(buffer.data_ptr() for buffer in collector.weights._norm_staging),
        collector._flat_params.data_ptr(),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA collector test requires a GPU")
def test_cuda_fp32_deterministic_matches_cpu() -> None:
    cpu_collector, _, _, _ = _collector("cpu")
    cuda_collector, _, _, _ = _collector("cuda")
    obs = torch.randn(_NUM_ENVS, _OBS_DIM)

    cpu_actions = cpu_collector._policy.deterministic(obs)
    cuda_actions = cuda_collector._policy.deterministic(obs.to(cuda_collector.device)).cpu()

    torch.testing.assert_close(cuda_actions, cpu_actions, rtol=2e-4, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA collector test requires a GPU")
def test_compiled_cuda_stochastic_actions_change_and_stay_in_bounds() -> None:
    collector, _, _, _ = _collector("cuda", compile=True)
    collector.warmup_inference()
    obs = torch.randn(_NUM_ENVS, _OBS_DIM)

    first = collector._infer(obs).clone()
    second = collector._infer(obs).clone()

    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert not torch.equal(first, second)
    low = collector._action_bias_cpu - collector._action_scale_cpu
    high = collector._action_bias_cpu + collector._action_scale_cpu
    assert torch.all(first >= low)
    assert torch.all(first <= high)
    assert torch.all(second >= low)
    assert torch.all(second <= high)
