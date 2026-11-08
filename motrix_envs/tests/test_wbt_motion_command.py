# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch contract tests for the WBT motion command state machine.

These protect the behavioral contracts of ``WbtMotionCommand`` after the
Torch executor migration: start-frame sampling (uniform and adaptive-CDF
search-left binning, clamping, start-at-zero draws), the ``advance``
wrap/hold/sequential-clip branches, and the reset pass's adaptive-CDF
rebuild plus reset-lane-only RNG consumption.
"""

import torch

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.rand import RandValue, draw_uniform, initialize_rand_states
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand, _sample_motion_step
from motrix_envs.motion import WbtMotionClip


def _clip(num_frames: int) -> WbtMotionClip:
    """Minimal numeric clip corpus: only the frame axis and clip ends matter."""
    return WbtMotionClip(
        joint_pos=torch.zeros((num_frames, 1), dtype=torch.float32),
        joint_vel=torch.zeros((num_frames, 1), dtype=torch.float32),
        tracked_bodies_pos_w=torch.zeros((num_frames, 1, 3), dtype=torch.float32),
        tracked_bodies_quat_w=torch.zeros((num_frames, 1, 4), dtype=torch.float32),
        tracked_bodies_lin_vel_w=torch.zeros((num_frames, 1, 3), dtype=torch.float32),
        tracked_bodies_ang_vel_w=torch.zeros((num_frames, 1, 3), dtype=torch.float32),
        root_body_pos_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        root_body_quat_w=torch.zeros((num_frames, 4), dtype=torch.float32),
        root_body_lin_vel_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        root_body_ang_vel_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        reference_body_pos_w=torch.zeros((num_frames, 3), dtype=torch.float32),
        reference_body_quat_w=torch.zeros((num_frames, 4), dtype=torch.float32),
        frame_clip_end=torch.full((num_frames,), num_frames - 1, dtype=torch.int64),
        extensions=Map({}),
    )


def _command(
    num_envs: int,
    num_frames: int,
    *,
    num_bins: int = 0,
    hold_at_clip_end: bool = False,
    sequential_clips: bool = False,
    start_zero_prob: float = 0.0,
    steps: torch.Tensor | None = None,
    seed: int = 7,
) -> tuple[WbtMotionCommand, torch.Tensor]:
    state = initialize_rand_states(num_envs, seed)
    command = WbtMotionCommand(
        clip=_clip(num_frames),
        reference_index=0,
        target_body_position_relative=torch.empty((num_envs, 1, 3), dtype=torch.float32),
        target_body_orientation_relative=torch.empty((num_envs, 1, 4), dtype=torch.float32),
        adaptive_bin_failed_count=torch.zeros((num_bins,), dtype=torch.float32),
        adaptive_current_bin_failed_count=torch.zeros((num_bins,), dtype=torch.float32),
        sampling_cdf=(
            torch.ones((num_bins,), dtype=torch.float32) if num_bins else torch.empty((0,), dtype=torch.float32)
        ),
        start_at_timestep_zero_prob=start_zero_prob,
        hold_at_clip_end=hold_at_clip_end,
        sequential_clips=sequential_clips,
        uniform_ratio=0.1,
        alpha=0.001,
        kernel_size=1,
        kernel_lambda=0.8,
        steps=steps if steps is not None else torch.zeros((num_envs, 1), dtype=torch.int64),
        clip_ended=torch.zeros((num_envs, 1), dtype=torch.bool),
        command=torch.empty((num_envs, 2), dtype=torch.float32),
        adaptive_entropy=torch.zeros((), dtype=torch.float32),
        adaptive_top1_prob=torch.zeros((), dtype=torch.float32),
        adaptive_top1_bin=torch.zeros((), dtype=torch.float32),
        adaptive_failure_mass=torch.zeros((), dtype=torch.float32),
    )
    return command, state


def _ctx(
    state: torch.Tensor,
    *,
    reset_ids: torch.Tensor | None = None,
    reset_episode: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
) -> ManagerContext:
    num_envs = state.shape[0]
    return ManagerContext(
        env_id=torch.arange(num_envs, dtype=torch.int64),
        actions=Map({}),
        commands=Map({}),
        metrics=Map({}),
        rand=RandValue(state),
        sim=Map({}),
        dt=0.02,
        sim_reset_requested=torch.zeros((num_envs, 1), dtype=torch.bool),
        reset_ids=torch.zeros(0, dtype=torch.int64) if reset_ids is None else reset_ids,
        # Compact reset-pass flags: one entry per lane in ``reset_ids``.
        reset_episode=torch.ones(0, dtype=torch.bool) if reset_episode is None else reset_episode,
        terminated=torch.zeros(num_envs, dtype=torch.bool) if terminated is None else terminated,
    )


# -- _sample_motion_step ------------------------------------------------------


def test_sample_motion_step_uniform_clamps_and_maps():
    num_frames = 10
    zero = torch.tensor(0.0)
    # unit = (draw + 1) / 2 in [0, 1); uniform step = unit * (T - 1), clamped
    # to [0, T - 2].
    draws = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [0.9998, 0.0]])
    steps = _sample_motion_step(draws, torch.empty(0), num_frames, zero)
    torch.testing.assert_close(steps, torch.tensor([0, 4, 8], dtype=torch.int64))


def test_sample_motion_step_adaptive_search_left():
    num_frames = 100
    zero = torch.tensor(0.0)
    cdf = torch.tensor([0.25, 0.5, 0.75, 1.0])
    # unit == 0.5 lands in bin 1 (search-left: equality selects the equal
    # bin); unit == 0.25 lands in bin 0. Within-bin phase comes from the
    # second draw: step = (bin + bin_unit) / 4 * 100, bin_unit = 0.5 here.
    draws = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    steps = _sample_motion_step(draws, cdf, num_frames, zero)
    torch.testing.assert_close(steps, torch.tensor([37, 12], dtype=torch.int64))


def test_sample_motion_step_start_at_zero():
    num_frames = 10
    draws = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.9]])
    # zero_unit = (draw + 1) / 2; prob 1.0 sends every lane to frame zero,
    # prob 0.0 sends none (zero_unit is always in [0, 1)).
    steps = _sample_motion_step(draws, torch.empty(0), num_frames, torch.tensor(1.0))
    torch.testing.assert_close(steps, torch.tensor([0, 0], dtype=torch.int64))
    steps = _sample_motion_step(draws, torch.empty(0), num_frames, torch.tensor(0.0))
    torch.testing.assert_close(steps, torch.tensor([4, 4], dtype=torch.int64))


# -- advance ------------------------------------------------------------------


def test_advance_wraps_ended_lanes_with_sampled_frames():
    command, state = _command(2, 5, steps=torch.tensor([[0], [4]]))
    ctx = _ctx(state)
    draws, _ = draw_uniform(state.clone(), 2)
    expected_wrap = _sample_motion_step(
        draws, command.sampling_cdf, 5, torch.tensor(command.start_at_timestep_zero_prob)
    )
    command.advance(ctx)
    assert command.steps[0, 0].item() == 1
    assert command.steps[1, 0].item() == expected_wrap[1].item()
    assert not command.clip_ended[0, 0].item()
    assert command.clip_ended[1, 0].item()
    assert not ctx.sim_reset_requested[0, 0].item()
    assert ctx.sim_reset_requested[1, 0].item()


def test_advance_wrap_consumes_only_ended_lane_rng():
    command, state = _command(2, 5, steps=torch.tensor([[0], [4]]))
    ctx = _ctx(state)
    before = state.clone()
    _, next_state = draw_uniform(state.clone(), 2)
    command.advance(ctx)
    torch.testing.assert_close(ctx.rand.state[0], before[0])
    torch.testing.assert_close(ctx.rand.state[1], next_state[1])


def test_advance_hold_at_clip_end_freezes_lane():
    command, state = _command(2, 5, hold_at_clip_end=True, steps=torch.tensor([[2], [4]]))
    ctx = _ctx(state)
    command.advance(ctx)
    torch.testing.assert_close(command.steps[:, 0], torch.tensor([3, 4]))
    assert not command.clip_ended[0, 0].item()
    assert command.clip_ended[1, 0].item()
    assert not ctx.sim_reset_requested.any()


def test_advance_sequential_wraps_corpus_end():
    command, state = _command(2, 5, sequential_clips=True, steps=torch.tensor([[3], [4]]))
    ctx = _ctx(state)
    command.advance(ctx)
    torch.testing.assert_close(command.steps[:, 0], torch.tensor([4, 0]))
    assert not ctx.sim_reset_requested[0, 0].item()
    assert ctx.sim_reset_requested[1, 0].item()


def test_advance_sequential_hold_at_corpus_end():
    command, state = _command(1, 5, sequential_clips=True, hold_at_clip_end=True, steps=torch.tensor([[4]]))
    ctx = _ctx(state)
    command.advance(ctx)
    torch.testing.assert_close(command.steps[:, 0], torch.tensor([4]))
    assert command.clip_ended[0, 0].item()
    assert not ctx.sim_reset_requested.any()


# -- reset --------------------------------------------------------------------


def test_reset_sequential_keeps_current_frames():
    steps = torch.tensor([[2], [3]])
    command, state = _command(2, 5, num_bins=2, sequential_clips=True, steps=steps)
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 1]),
        reset_episode=torch.tensor([True, True]),
    )
    command.reset(ctx)
    torch.testing.assert_close(command.steps, steps)


def test_reset_rebuilds_adaptive_cdf_from_failures():
    # Lane 0 terminated inside bin 1 (steps=4 of 8 frames, 2 bins);
    # lane 1 reset without terminating contributes no failure mass.
    command, state = _command(2, 8, num_bins=2, steps=torch.tensor([[4], [0]]))
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 1]),
        reset_episode=torch.tensor([True, True]),
        terminated=torch.tensor([True, False]),
    )
    command.reset(ctx)
    assert command.adaptive_current_bin_failed_count.tolist() == [0.0, 1.0]
    cdf = command.sampling_cdf
    assert cdf[-1].item() == 1.0
    assert torch.all(cdf[:-1] <= cdf[1:])
    # With zero failure history and uniform_ratio mass spread evenly, the
    # rebuilt distribution stays uniform over the reachable bins.
    torch.testing.assert_close(cdf, torch.tensor([0.5, 1.0]))


def test_reset_consumes_only_reset_lane_rng():
    command, state = _command(4, 5, steps=torch.tensor([[0], [1], [2], [3]]))
    before = state.clone()
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([1, 3]),
        reset_episode=torch.tensor([True, True]),
    )
    _, next_state = draw_uniform(state.clone(), 2)
    command.reset(ctx)
    torch.testing.assert_close(ctx.rand.state[0], before[0])
    torch.testing.assert_close(ctx.rand.state[2], before[2])
    torch.testing.assert_close(ctx.rand.state[1], next_state[1])
    torch.testing.assert_close(ctx.rand.state[3], next_state[3])


def test_reset_resamples_steps_in_valid_range():
    command, state = _command(3, 6, start_zero_prob=0.5, steps=torch.tensor([[5], [5], [5]]))
    ctx = _ctx(
        state,
        reset_ids=torch.tensor([0, 2]),
        reset_episode=torch.tensor([True, True]),
    )
    command.reset(ctx)
    steps = command.steps[:, 0]
    assert steps[1].item() == 5
    for lane in (0, 2):
        assert 0 <= steps[lane].item() <= 4
