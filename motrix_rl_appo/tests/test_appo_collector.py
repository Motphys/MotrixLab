# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import threading

import gymnasium as gym
import numpy as np
import pytest
import torch
from motrix_rl_appo.collector import ChunkCollector, publish_chunk
from motrix_rl_appo.config import AppoCfg
from motrix_rl_appo.transport import CollectorMetadata, chunk_schema, take_chunk

from motrix_env_core.array.env import ArrayEnvState, EnvObs, FinalObs
from motrix_rl.transport.ring import HostTransitionRing


class FakeActor:
    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return obs * 10, obs[:, 0] * -2


class FakeCritic:
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, 0]


class FakeEnv:
    num_envs = 3
    action_space = gym.spaces.Box(-1000, 1000, (1,), dtype=np.float32)

    def __init__(self, missing_final: bool = False) -> None:
        self.obs = torch.tensor([[1.0], [2.0], [3.0]])
        self.value = self.obs + 10
        self.missing_final = missing_final
        self.actions: list[torch.Tensor] = []

    def initialize(self) -> ArrayEnvState:
        return self.state(torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))

    def state(self, terminated: torch.Tensor, truncated: torch.Tensor, final: FinalObs | None = None) -> ArrayEnvState:
        return ArrayEnvState(
            EnvObs({"policy": self.obs, "value": self.value}),
            torch.ones(3),
            terminated,
            truncated,
            torch.zeros(3, dtype=torch.long),
            final_obs=final,
        )

    def step(self, actions: torch.Tensor) -> ArrayEnvState:
        self.actions.append(actions.clone())
        # Deliberately overwrite all borrowed buffers, exposing late obs copies.
        self.obs.add_(1)
        self.value.add_(1)
        terminated = torch.tensor([False, True, False])
        truncated = torch.tensor([False, True, True])
        # Compact ids are deliberately not sorted; value != policy.
        final = FinalObs(
            torch.tensor([2, 1]),
            EnvObs({"policy": torch.tensor([[99.0], [98.0]]), "value": torch.tensor([[7.0], [9.0]])}),
        )
        return self.state(terminated, truncated, None if self.missing_final else final)


def test_chunk_alignment_timeout_and_endpoint() -> None:
    cfg = AppoCfg(rollout_steps=2)
    cfg.algorithm.gamma = 0.5
    env = FakeEnv()
    collector = ChunkCollector(env, FakeActor(), FakeCritic(), cfg, torch.device("cpu"))
    chunk, returns, lengths, _ = collector.collect()
    torch.testing.assert_close(chunk["obs"][:, :, 0], torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
    torch.testing.assert_close(chunk["critic_obs"], chunk["obs"] + 10)
    torch.testing.assert_close(chunk["actions"], chunk["obs"] * 10)
    torch.testing.assert_close(chunk["behaviour_logp"], -2 * chunk["obs"][:, :, 0])
    torch.testing.assert_close(chunk["rewards"], torch.tensor([[1.0, 1.0], [1.0, 1.0], [4.5, 4.5]]))
    torch.testing.assert_close(chunk["last_critic_obs"][:, 0], torch.tensor([13.0, 14.0, 15.0]))
    assert chunk["dones"].tolist() == [[False, False], [True, True], [True, True]]
    assert returns == [1.0, 1.0, 1.0, 1.0]
    assert lengths == [1, 1, 1, 1]
    assert torch.equal(env.actions[0], chunk["actions"][:, 0])
    # Storage survives further env writes, including endpoint.
    env.obs.add_(100)
    env.value.add_(100)
    assert chunk["last_critic_obs"][0, 0] == 13


def test_borrowed_carrier_reuse_only_after_publication() -> None:
    cfg = AppoCfg(rollout_steps=2)
    collector = ChunkCollector(FakeEnv(), FakeActor(), FakeCritic(), cfg, torch.device("cpu"))
    ring = HostTransitionRing(chunk_schema(2, CollectorMetadata(1, 1, 1)), 2, 3)
    first, *_ = collector.collect()
    pointers = {name: value.data_ptr() for name, value in first.items()}
    expected = {name: value.clone() for name, value in first.items()}
    assert publish_chunk(ring, first, threading.Event(), 1)
    second, *_ = collector.collect()
    assert {name: value.data_ptr() for name, value in second.items()} == pointers
    assert publish_chunk(ring, second, threading.Event(), 1)
    owned = take_chunk(ring, torch.device("cpu"))
    for name, value in expected.items():
        if name != "last_critic_obs":
            value = value.transpose(0, 1).contiguous()
        torch.testing.assert_close(owned[name], value)
    next_chunk = take_chunk(ring, torch.device("cpu"))
    torch.testing.assert_close(next_chunk["obs"], owned["obs"] + 2)
    assert all(value.device.type == "cpu" for value in second.values())


def test_timeout_without_final_obs_uses_owned_pre_action_value_and_warns_once(caplog) -> None:
    cfg = AppoCfg(rollout_steps=2)
    cfg.algorithm.gamma = 0.5
    collector = ChunkCollector(FakeEnv(missing_final=True), FakeActor(), FakeCritic(), cfg, torch.device("cpu"))
    chunk, returns, _, _ = collector.collect()
    # Row2 is timeout-only: start values13,14, not overwritten next values14,15.
    # Row1 is both terminated and truncated and must never bootstrap.
    torch.testing.assert_close(chunk["rewards"], torch.tensor([[1.0, 1.0], [1.0, 1.0], [7.5, 8.0]]))
    assert returns == [1.0, 1.0, 1.0, 1.0]
    collector.collect()
    assert sum("pre-action V(s_t)" in record.message for record in caplog.records) == 1


def test_partial_final_observations_remain_a_contract_error() -> None:
    class IncompleteFinalEnv(FakeEnv):
        def step(self, actions: torch.Tensor) -> ArrayEnvState:
            state = super().step(actions)
            final = FinalObs(torch.tensor([1]), EnvObs({"policy": torch.zeros(1, 1), "value": torch.zeros(1, 1)}))
            return self.state(state.terminated, state.truncated, final)

    collector = ChunkCollector(
        IncompleteFinalEnv(), FakeActor(), FakeCritic(), AppoCfg(rollout_steps=1), torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match="do not cover"):
        collector.collect()
