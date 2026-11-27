# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from motrix_rl_builtin.fastsac.async_impl.topology import resolve_fastsac_topology
from motrix_rl_builtin.fastsac.config import FastSacAsyncOptionsCfg


@pytest.mark.parametrize(
    ("sim_device", "expected_ipc"),
    [("cpu", (True, True)), ("cuda", (True, True)), ("cuda:0", (True, False)), ("cuda:1", (False, True))],
)
def test_fastsac_transition_transport_respects_simulation_placement(
    sim_device, expected_ipc, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    topology = resolve_fastsac_topology(
        num_envs=8,
        num_collectors=2,
        num_learners=2,
        learner_device_specs=["cuda:0", "cuda:1"],
        collector_inference_device="cuda",
        default_device=torch.device("cuda:0"),
        async_options=FastSacAsyncOptionsCfg(),
        actor_param_numel=0,
        sim_device=sim_device,
    )
    assert tuple(channel.use_cuda_ipc for channel in topology.transition_channels) == expected_ipc
