# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Single-frame replay preserves online history and episode boundaries."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from motrix_rl_builtin.fastsac.buffer import SimpleReplayBuffer
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap, FastSacStepResult

from motrix_env_core.array.env import EnvObs, FinalObs
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec


@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize("bootstrap", [False, True])
def test_history_replay_wrap_repeat_fill_and_final_endpoint(n_steps, bootstrap):
    rb = SimpleReplayBuffer(
        2,
        8,
        1,
        1,
        1,
        n_steps=n_steps,
        gamma=0.5,
        device="cpu",
        policy_history=4,
        value_history=2,
        final_obs_bootstrap=bootstrap,
    )
    starts = [0, 0]
    windows = {}
    for t in range(23):
        frame = torch.tensor([[float(t)], [float(100 + t)]])
        trunc = torch.tensor([int(t % 5 == 3), int(t % 7 == 4)])
        for lane in range(2):
            windows[lane, t] = (
                [float(lane * 100 + max(starts[lane], k)) for k in range(t - 3, t + 1)],
                [float(lane * 100 + max(starts[lane], k)) for k in range(t - 1, t + 1)],
            )
        rb.extend(frame, frame, torch.full((2, 1), float(t)), torch.ones(2), torch.zeros(2, dtype=torch.long), trunc)
        if bootstrap and trunc.any():
            ids = trunc.nonzero().flatten()
            rb.write_final(ids, t, frame[ids] + 0.5, frame[ids] + 0.5)
        for lane in range(2):
            if trunc[lane]:
                starts[lane] = t + 1
    assert rb.observations.shape == (2, 9, 1)
    assert rb._final_obs_table is None or rb._final_obs_table.shape[1] == 1
    batch = rb.sample(256)
    for row in range(512):
        lane = row // 256
        t = int(batch["actions"][row, 0])
        torch.testing.assert_close(batch["obs"][row], torch.tensor(windows[lane, t][0]))
        torch.testing.assert_close(batch["critic_obs"][row], torch.tensor(windows[lane, t][1]))
        effective = int(batch["effective_n_steps"][row])
        endpoint = t + effective - 1
        assert batch["rewards"][row] == pytest.approx(sum(0.5**k for k in range(effective)))
        if bootstrap and batch["truncations"][row]:
            expected = windows[lane, endpoint][0][1:] + [lane * 100 + endpoint + 0.5]
        else:
            expected = windows[lane, endpoint + 1][0]
        torch.testing.assert_close(batch["next_obs"][row], torch.tensor(expected))


@pytest.mark.parametrize("with_value", [False, True])
def test_owned_frame_staging_survives_borrowed_environment_mutation(with_value: bool) -> None:
    groups = {"policy": ObservationGroupSpec(gym.spaces.Box(-np.inf, np.inf, (1,)), 3)}
    if with_value:
        groups["value"] = ObservationGroupSpec(gym.spaces.Box(-np.inf, np.inf, (2,)), 2)

    class Env:
        num_envs = 1
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        observation_spec = ObservationSpec(groups)
        stack = torch.tensor([[3.0, 5.0, 7.0]])
        value_stack = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    env = Env()
    adapter = FastSacEnvWrap(env, torch.device("cpu"))
    saved = adapter.stage_frames(env.stack, env.value_stack)
    env.stack.fill_(9)
    env.value_stack.fill_(9)
    assert saved.policy.item() == 7
    if with_value:
        torch.testing.assert_close(saved.value, torch.tensor([[3.0, 4.0]]))
    else:
        assert saved.value is None
    final = FinalObs(torch.tensor([0]), EnvObs({"policy": torch.tensor([[5.0, 7.0, 8.0]])}))
    result = FastSacStepResult(
        env.stack,
        env.stack,
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        final,
        {},
    )
    staged_final = adapter.final_frames(result)
    assert staged_final.obs.policy.item() == 8
    final.obs.policy.fill_(99)
    assert staged_final.obs.policy.item() == 8


def test_shared_value_history_does_not_duplicate_replay_storage():
    rb = SimpleReplayBuffer(1, 4, 2, 1, 2, device="cpu", policy_history=3, value_history=3, shared_observations=True)
    assert rb.observations is rb.critic_observations
