# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the schema-driven seqlock inference-weight channel.

These tests deliberately use a NON-FastSAC payload — two inference modules
and two normalizers — to pin the public contract: the bundle
layout follows the declared :class:`~motrix_rl.transport.inference_bundle.InferenceBundleSpec`,
not any particular trainer's section names. A regression here is what lets a
new asynchronous backend adopt the channel without touching it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from motrix_rl.transport.inference_bundle import (
    InferenceBundleSpec,
    ModuleSection,
    NormalizerSection,
)
from motrix_rl.transport.weight_channel import (
    HostWeightReceiver,
    HostWeightSender,
    WeightChannelShared,
)


class _Module(nn.Module):
    def __init__(self, size: int, fill: float) -> None:
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(fill)


class _Norm(nn.Module):
    """Minimal EmpiricalNormalization-shaped stat holder."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self._mean = torch.zeros(1, dim)
        self._std = torch.ones(1, dim)
        self._var = torch.ones(1, dim)
        self.count = torch.tensor(0)

    def fill(self, mean: float, std: float, count: int) -> None:
        self._mean.fill_(mean)
        self._std.fill_(std)
        self._var.fill_(std**2)
        self.count.fill_(count)


SPEC = InferenceBundleSpec(
    modules=(ModuleSection("policy", 2 * 2), ModuleSection("critic_head", 3 * 3)),
    normalizers=(NormalizerSection("obs", 4), NormalizerSection("command", 6)),
)


def _endpoint():
    shared = WeightChannelShared(SPEC)
    return shared, HostWeightSender(shared), HostWeightReceiver(shared, None)  # params injected below


def test_spec_layouts_and_validation():
    assert SPEC.params_numel == 4 + 9
    assert SPEC.normalizer_dim == 10
    assert SPEC.module_layout == {"policy": (0, 4), "critic_head": (4, 13)}
    assert SPEC.normalizer_layout == {"obs": (0, 4), "command": (4, 10)}
    with pytest.raises(ValueError, match="unique"):
        InferenceBundleSpec(modules=(ModuleSection("a", 2),), normalizers=(NormalizerSection("a", 2),))
    with pytest.raises(ValueError, match="at least one module"):
        InferenceBundleSpec(modules=())
    with pytest.raises(ValueError, match="dim"):
        InferenceBundleSpec(modules=(ModuleSection("policy", 4),), normalizers=(NormalizerSection("obs", 0),))


def _publish(sender, policy, head, obs, command):
    sender.publish({"policy": policy, "critic_head": head}, {"obs": obs, "command": command})


def test_multi_section_roundtrip_updates_every_consumer_module():
    shared = WeightChannelShared(SPEC)
    sender = HostWeightSender(shared)
    receiver = HostWeightReceiver(shared, sender.params)

    policy, head = _Module(2, 1.0), _Module(3, 2.0)
    obs, command = _Norm(4), _Norm(6)
    obs.fill(0.5, 2.0, 11)
    command.fill(-1.0, 0.5, 22)
    _publish(sender, policy, head, obs, command)

    rx_policy, rx_head = _Module(2, 9.0), _Module(3, 9.0)
    rx_obs, rx_command = _Norm(4), _Norm(6)
    version, *_ = receiver.maybe_load(
        {"policy": rx_policy, "critic_head": rx_head}, {"obs": rx_obs, "command": rx_command}
    )
    assert version == 1
    torch.testing.assert_close(rx_policy.linear.weight, policy.linear.weight)
    torch.testing.assert_close(rx_head.linear.weight, head.linear.weight)
    torch.testing.assert_close(rx_obs._mean, obs._mean)
    torch.testing.assert_close(rx_command._std, command._std)
    assert int(rx_obs.count) == 11
    assert int(rx_command.count) == 22


def test_versions_advance_and_lag_semantics_hold():
    shared = WeightChannelShared(SPEC)
    sender = HostWeightSender(shared)
    receiver = HostWeightReceiver(shared, sender.params)
    policy, head = _Module(2, 1.0), _Module(3, 1.0)
    obs, command = _Norm(4), _Norm(6)
    modules = {"policy": _Module(2, 9.0), "critic_head": _Module(3, 9.0)}
    normals = {"obs": _Norm(4), "command": _Norm(6)}

    assert receiver.maybe_load(modules, normals)[0] == 0  # nothing published
    for tag in range(1, 4):
        with torch.no_grad():
            policy.linear.weight.fill_(float(tag))
        _publish(sender, policy, head, obs, command)
    assert receiver.version == 3
    assert receiver.lag == 3
    version, *_ = receiver.maybe_load(modules, normals)
    assert version == 3
    assert receiver.lag == 0
    # a poll with no new publish returns the loaded version unchanged
    assert receiver.maybe_load(modules, normals)[0] == 3
    torch.testing.assert_close(modules["policy"].linear.weight, torch.full((2, 2), 3.0))


def test_publish_rejects_sections_that_do_not_match_the_spec():
    shared = WeightChannelShared(SPEC)
    sender = HostWeightSender(shared)
    policy, head = _Module(2, 1.0), _Module(3, 1.0)
    obs, command = _Norm(4), _Norm(6)
    _publish(sender, policy, head, obs, command)  # the matching bundle publishes fine
    with pytest.raises(ValueError, match="modules"):
        sender.publish({"policy": policy}, {"obs": obs, "command": command})
    with pytest.raises(ValueError, match="normalizers"):
        sender.publish({"policy": policy, "critic_head": head}, {"obs": obs})


def test_maybe_load_rejects_flat_params_that_do_not_match_the_spec():
    """flat_params is the CUDA-IPC path's actual write target, so a partial or
    extra-key mapping must fail fast instead of silently leaving a module's
    weights stale (IPC) or degrading to per-parameter copies (host)."""
    shared = WeightChannelShared(SPEC)
    sender = HostWeightSender(shared)
    receiver = HostWeightReceiver(shared, sender.params)
    modules = {"policy": _Module(2, 9.0), "critic_head": _Module(3, 9.0)}
    normals = {"obs": _Norm(4), "command": _Norm(6)}
    with pytest.raises(ValueError, match="flat_params"):
        receiver.maybe_load(modules, normals, flat_params={"policy": torch.empty(4)})
    with pytest.raises(ValueError, match="flat_params"):
        receiver.maybe_load(
            modules, normals, flat_params={"policy": torch.empty(4), "critic_head": torch.empty(9), "extra": None}
        )


def test_statless_normalizer_sections_roundtrip_symmetrically():
    """Identity-style normalizers (no NORM_KEYS) skip the stat path on both ends.

    Stat presence is a config-driven property agreed by both endpoints (both
    processes build the same normalizer type), mirroring the sync-path
    convention carried over from the FastSAC channel.
    """

    class _Statless(nn.Module):
        pass

    spec = InferenceBundleSpec(
        modules=(ModuleSection("policy", 4),),
        normalizers=(NormalizerSection("obs", 3),),
    )
    shared = WeightChannelShared(spec)
    sender = HostWeightSender(shared)
    receiver = HostWeightReceiver(shared, sender.params)
    sender.publish({"policy": _Module(2, 1.0)}, {"obs": _Statless()})
    version, *_ = receiver.maybe_load({"policy": _Module(2, 9.0)}, {"obs": _Statless()})
    assert version == 1
