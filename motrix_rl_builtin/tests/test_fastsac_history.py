# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Single-frame replay preserves online history and episode boundaries."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from motrix_rl_builtin.fastsac.async_impl.transport import HostTransitionRing, fastsac_transition_schema
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
@pytest.mark.parametrize("destination", ["replay", "ring"])
def test_frame_views_are_copied_before_environment_buffer_reuse(with_value: bool, destination: str) -> None:
    groups = {"policy": ObservationGroupSpec(gym.spaces.Box(-np.inf, np.inf, (1,)), 3)}
    if with_value:
        groups["value"] = ObservationGroupSpec(gym.spaces.Box(-np.inf, np.inf, (2,)), 2)

    class Env:
        device = torch.device("cpu")
        num_envs = 1
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        observation_spec = ObservationSpec(groups)
        stack = torch.tensor([[3.0, 5.0, 7.0]])
        value_stack = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    env = Env()
    adapter = FastSacEnvWrap(env, torch.device("cpu"))
    saved = adapter.stage_frames(env.stack, env.value_stack)
    assert saved.policy.untyped_storage().data_ptr() == env.stack.untyped_storage().data_ptr()
    if with_value:
        torch.testing.assert_close(saved.value, torch.tensor([[3.0, 4.0]]))
        assert saved.value.untyped_storage().data_ptr() == env.value_stack.untyped_storage().data_ptr()
    else:
        assert saved.value is None

    fields = dict(
        obs=saved.policy,
        critic_obs=saved.value_or_policy,
        actions=torch.zeros(1, 1),
        rewards=torch.ones(1),
        dones=torch.zeros(1, dtype=torch.long),
        truncations=torch.zeros(1, dtype=torch.long),
    )
    value_dim = 2 if with_value else 1
    if destination == "replay":
        storage = SimpleReplayBuffer(1, 4, 1, 1, value_dim, device="cpu", final_obs_bootstrap=True)
        storage.extend(*fields.values())
        stored_policy = storage.observations[:, 0]
        stored_value = storage.critic_observations[:, 0]
    else:
        storage = HostTransitionRing(fastsac_transition_schema(1, value_dim, 1), 4, 1)
        assert storage.push(fields)
        _, span = storage.read_span()
        stored_policy = span["obs"][0]
        stored_value = span["critic_obs"][0]

    # The next buffer reuse may mutate the borrowed views, not the committed transition.
    env.stack.fill_(9)
    env.value_stack.fill_(9)
    assert saved.policy.item() == 9
    assert stored_policy.item() == 7
    torch.testing.assert_close(stored_value, torch.tensor([[3.0, 4.0]]) if with_value else torch.tensor([[7.0]]))
    final_groups = {"policy": torch.tensor([[5.0, 7.0, 8.0]])}
    if with_value:
        final_groups["value"] = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    final = FinalObs(torch.tensor([0]), EnvObs(final_groups))
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
    assert staged_final.obs.policy.untyped_storage().data_ptr() == final.obs.policy.untyped_storage().data_ptr()
    if destination == "replay":
        storage.write_final(torch.tensor([0]), 0, staged_final.obs.policy, staged_final.obs.value_or_policy)
        final.obs.policy.fill_(99)
        row = storage._final_slot[0, 0].long()
        assert storage._final_obs_table[row].item() == 8


def test_shared_value_history_does_not_duplicate_replay_storage():
    rb = SimpleReplayBuffer(1, 4, 2, 1, 2, device="cpu", policy_history=3, value_history=3, shared_observations=True)
    assert rb.observations is rb.critic_observations
