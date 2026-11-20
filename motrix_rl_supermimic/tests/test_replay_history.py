# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Imagination histories must never reach across a reset episode boundary."""

import pytest
import torch
from _helpers import make_toy
from motrix_rl_supermimic.replay import ReplayBuffer, transition_schema
from motrix_rl_supermimic.toy import register as register_toy


def test_transition_schema_matches_replay_storage() -> None:
    replay = ReplayBuffer(4, 2, state_dim=2, context_dim=0, action_dim=1)
    schema = transition_schema(state_dim=2, context_dim=0, action_dim=1)
    for field in schema.fields:
        if field.name in replay.data:
            stored = replay.data[field.name]
            assert stored.shape == (replay.T,) + schema.slot_shape(field, replay.N)
            assert stored.dtype == field.dtype


def test_first_post_reset_transition_repeats_new_episode_history(monkeypatch):
    register_toy()
    env = make_toy(num_envs=1)
    env.init_state()
    replay = ReplayBuffer(
        8, 1, state_dim=env.world_state.size, context_dim=env.world_context.size, action_dim=env.action_space.shape[0]
    )
    # Ensure the preceding episode has a distinct state and observation history.
    env.sim.position.fill_(2.5)
    env.sim_data.execute()
    env.step(torch.full((1, 1), 0.4))
    terminal = env.snapshot_transition(env.state)
    assert terminal["done"].item()
    replay.add(terminal)

    env.step(torch.full((1, 1), 0.2))
    fresh = env.snapshot_transition(env.state)
    assert fresh["ep_step"].item() == 0
    replay.add(fresh)

    def sample_latest(low, high, size, *, device=None):
        value = 1 if high == replay.size else 0
        return torch.full(size, value, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", sample_latest)
    start, history = replay.sample_starts(1, hist=4)
    for field in ("s", "a_prev", "xi", "f"):
        torch.testing.assert_close(start[field], fresh[field])
        torch.testing.assert_close(history[field], fresh[field][:, None].expand_as(history[field]))
    assert not torch.equal(history["s"][:, 0], terminal["s"])


def _row(step: int) -> dict[str, torch.Tensor]:
    from motrix_rl_supermimic.replay import transition_fields

    row = {
        name: torch.zeros((1,) + shape, dtype=dtype)
        for name, (shape, dtype) in transition_fields(state_dim=2, context_dim=0, action_dim=1).items()
    }
    row["ep_step"].fill_(step)
    row["f"].fill_(step)
    return row


def test_replay_overwrite_does_not_fabricate_episode_prefix() -> None:
    replay = ReplayBuffer(3, 1, state_dim=2, context_dim=0, action_dim=1)
    for step in range(6):
        replay.add(_row(step))
    start, history = replay.sample_starts(32, hist=3)
    assert start["f"].tolist() == [5] * 32
    torch.testing.assert_close(history["f"], torch.tensor([[3, 4, 5]]).expand(32, -1))
    with pytest.raises(RuntimeError, match="complete observation history prefix"):
        replay.sample_starts(1, hist=4)


def test_replay_discontinuity_is_not_repeat_padding() -> None:
    replay = ReplayBuffer(3, 1, state_dim=2, context_dim=0, action_dim=1)
    for step in (4, 8, 9):
        replay.add(_row(step))
    with pytest.raises(RuntimeError, match="complete observation history prefix"):
        replay.sample_starts(1, hist=3)


def test_future_validity_and_history_continuity_are_selected_together() -> None:
    replay = ReplayBuffer(6, 1, state_dim=2, context_dim=0, action_dim=1)
    for step in (0, 1, 2, 7, 8, 9):
        replay.add(_row(step))
    # Frame 7 passes the future predicate but lacks its episode prefix; only
    # frames 0/1/2 can satisfy both constraints with a three-frame history.
    start, history = replay.sample_starts(64, hist=3, valid_frames=lambda frames: frames < 8)
    assert start["f"].shape == (64,)
    assert bool((start["f"] <= 2).all())
    torch.testing.assert_close(history["f"][:, -1], start["f"])
    for row, frame in zip(history["f"], start["f"], strict=True):
        expected = (frame - torch.arange(2, -1, -1)).clamp(min=0)
        torch.testing.assert_close(row, expected)


def test_no_future_valid_starts_fail_without_accepting_invalid_frames() -> None:
    replay = ReplayBuffer(4, 1, state_dim=2, context_dim=0, action_dim=1)
    for step in range(4):
        replay.add(_row(step))
    with pytest.raises(RuntimeError, match="0/8 valid imagination starts.*100 attempts"):
        replay.sample_starts(8, hist=2, valid_frames=lambda frames: frames < 0)


@pytest.mark.parametrize("batch,hist", [(0, 1), (1, 0), (1, 2)])
def test_empty_or_invalid_start_request_is_rejected(batch: int, hist: int) -> None:
    replay = ReplayBuffer(4, 1, state_dim=2, context_dim=0, action_dim=1)
    with pytest.raises(ValueError, match="nonempty replay and positive batch/history"):
        replay.sample_starts(batch, hist)


def test_no_predicate_samples_requested_candidate_count(monkeypatch) -> None:
    replay = ReplayBuffer(4, 1, state_dim=2, context_dim=0, action_dim=1)
    replay.add(_row(0))
    sizes = []
    randint = torch.randint

    def record(low, high, size, *, device=None):
        sizes.append(size)
        return randint(low, high, size, device=device)

    monkeypatch.setattr(torch, "randint", record)
    start, history = replay.sample_starts(5, hist=3)
    assert sizes == [(5,), (5,)]
    assert start["s"].shape[0] == history["s"].shape[0] == 5


def test_learner_samples_starts_with_bound_imagination_horizon(tmp_path) -> None:
    from _helpers import make_toy_task
    from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg, TrainerCfg
    from motrix_rl_supermimic.learner import Learner
    from motrix_rl_supermimic.toy import ToyEnvCfg, ToyImaginationCfg, ToyImaginationTerm

    from motrix_env_core.config import configclass
    from motrix_env_core.manager.context import BuildContext

    class ClipImagination(ToyImaginationTerm):
        def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
            return frame + horizon < 5

    @configclass
    class ClipImaginationCfg(ToyImaginationCfg):
        def __call__(self, ctx: BuildContext) -> ClipImagination:
            return ClipImagination()

    task = make_toy_task(cfg=ToyEnvCfg(imagination=ClipImaginationCfg()))
    cfg = SuperMimicCfg(agent=AgentCfg(hidden=8), trainer=TrainerCfg(pi_batch=16, horizon=3))
    learner = Learner(cfg, task, 1, 0.02, "cpu", tmp_path)
    for step in range(5):
        learner.buffer.add(_row(step))
    start, history = learner.starts()
    assert start["f"].shape == (16,)
    assert bool((start["f"] < 2).all())
    assert history["f"].shape == (16, task.observation_spec.policy.history_length)


def test_episode_padding_reuses_same_noise_realization_and_preserves_gradients(tmp_path) -> None:
    from _helpers import make_toy_task
    from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg
    from motrix_rl_supermimic.learner import Learner

    from motrix_env_core.observations import append_observation

    learner = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8)), make_toy_task(), 1, 0.02, "cpu", tmp_path)
    replay = ReplayBuffer(4, 1, state_dim=2, context_dim=0, action_dim=1)
    replay.add(_row(0))
    _, history = replay.sample_starts(2, hist=4)
    noisy = learner.observation_history(history)
    torch.testing.assert_close(noisy, noisy[:, -1:].expand_as(noisy))
    initial = noisy.clone().requires_grad_()
    first = torch.randn(2, learner.mdp.observation_dim, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    imagined = append_observation(append_observation(initial, first), second)
    imagined.sum().backward()
    torch.testing.assert_close(first.grad, torch.ones_like(first))
    torch.testing.assert_close(second.grad, torch.ones_like(second))
    assert bool((initial.grad[:, 2:] == 1).all())
