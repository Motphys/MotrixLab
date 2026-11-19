# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Transport and learner use frame storage but final network dimensions."""

import gymnasium as gym
import numpy as np
import torch
from motrix_rl_builtin.fastsac.agent import FastSacAgent
from motrix_rl_builtin.fastsac.async_impl.transport import HostTransitionRing, fastsac_transition_schema
from motrix_rl_builtin.fastsac.buffer import SimpleReplayBuffer
from test_fastsac_learner import _tiny_agent_cfg

from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec


def test_history_agent_updates_final_width_normalizers_from_sampled_windows():
    spec = ObservationSpec(
        {
            "policy": ObservationGroupSpec(gym.spaces.Box(-np.inf, np.inf, (2,)), 3),
            "value": ObservationGroupSpec(gym.spaces.Box(-np.inf, np.inf, (3,)), 2),
        }
    )
    agent = FastSacAgent(
        6, 6, 1, 2, _tiny_agent_cfg(compile=False, amp=False), torch.device("cpu"), observation_spec=spec
    )
    for t in range(6):
        agent.rb.extend(
            torch.full((2, 2), float(t)),
            torch.full((2, 3), float(t)),
            torch.zeros(2, 1),
            torch.ones(2),
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
        )
    assert agent.rb.observations.shape[-1] == 2
    assert agent.rb.critic_observations.shape[-1] == 3
    agent.set_train_mode()
    agent.update(1)
    assert agent.obs_normalizer._mean.shape[-1] == 6
    assert agent.critic_obs_normalizer._mean.shape[-1] == 6
    assert agent.obs_normalizer.count > 0
    assert agent.act(torch.ones(2, 6)).shape == (2, 1)


def test_host_transport_batch_ingest_matches_sync_history_after_wrap():
    sync = SimpleReplayBuffer(2, 5, 2, 1, 3, device="cpu", policy_history=3, value_history=2, n_steps=3)
    asynchronous = SimpleReplayBuffer(2, 5, 2, 1, 3, device="cpu", policy_history=3, value_history=2, n_steps=3)
    ring = HostTransitionRing(fastsac_transition_schema(2, 3, 1), 3, 2)
    for t in range(18):
        fields = dict(
            obs=torch.full((2, 2), float(t)),
            critic_obs=torch.full((2, 3), float(t + 100)),
            actions=torch.zeros(2, 1),
            rewards=torch.ones(2),
            dones=torch.tensor([t % 4 == 2, False]).long(),
            truncations=torch.tensor([False, t % 5 == 3]).long(),
        )
        sync.extend(*fields.values())
        assert ring.push(fields)
        if t % 3 == 2:
            count, span = ring.read_span()
            asynchronous.extend_batch(*span.values())
            ring.commit_reads(count)
    torch.manual_seed(19)
    expected = sync.sample(32)
    torch.manual_seed(19)
    actual = asynchronous.sample(32)
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name])
