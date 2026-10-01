# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pickle
from dataclasses import fields, replace
from types import SimpleNamespace

import numba
import numpy as np
import pytest

import motrix_envs  # noqa: E402, F401
from motrix_env_core import registry  # noqa: E402
from motrix_env_core.manager import (  # noqa: E402
    ManagerContext,
    ManagerEnv,
    ManagerResetCfg,
)
from motrix_env_core.mdp.action import (  # noqa: E402
    JointPositionActionCfg,
    JointPositionActionState,
)
from motrix_env_core.mdp.observations import (  # noqa: E402
    BodyAngularVelocityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.state import RandValue  # noqa: E402
from motrix_env_core.numba.manager.commands import ResetContext  # noqa: E402
from motrix_env_core.numba.manager.compiler.compiler import TermScalarBuffer  # noqa: E402
from motrix_env_core.sim import BatchLinkPositionQuery  # noqa: E402
from motrix_envs.locomotion.wbt.cfg import (  # noqa: E402
    ActionsCfg,
    RewardsCfg,
    WbtEnvCfg,
)
from motrix_envs.locomotion.wbt.dex_evt import DexEvtWbtEnvCfg  # noqa: E402
from motrix_envs.locomotion.wbt.g1.common import MOTION_DIR as _G1_MOTION_DIR  # noqa: E402
from motrix_envs.locomotion.wbt.g1.common import G1WbtEnvCfg  # noqa: E402
from motrix_envs.locomotion.wbt.k1 import K1WbtEnvCfg  # noqa: E402
from motrix_envs.locomotion.wbt.mdp.command import (  # noqa: E402
    WbtMotionCommand,
    WbtMotionCommandCfg,
)
from motrix_envs.locomotion.wbt.mdp.observations import (  # noqa: E402
    DofPosRelObsCfg,
    DofVelObsCfg,
    MotionReferenceOrientationObsCfg,
)
from motrix_envs.locomotion.wbt.mdp.reset import (  # noqa: E402
    BodyDofPosResetCfg,
    BodyLinVelResetCfg,
    BodyPosResetCfg,
    BodyRotResetCfg,
    BodyRotVelResetCfg,
)
from motrix_envs.motion import MotrixMotion, WbtMotionClip  # noqa: E402


def _motion_command(env: ManagerEnv) -> WbtMotionCommand:
    command = env.command_terms["motion"]
    assert isinstance(command, WbtMotionCommand)
    return command


def _deterministic_manager_cfg(*, hold_at_clip_end: bool = False) -> WbtEnvCfg:
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    motion = cfg.commands.motion
    assert isinstance(motion, WbtMotionCommandCfg)
    policy = cfg.observations.policy
    dof_pos = policy.dof_pos
    dof_vel = policy.dof_vel
    base_ang_vel = policy.base_ang_vel
    motion_ref_ori = policy.motion_ref_ori_b
    assert isinstance(dof_pos, DofPosRelObsCfg)
    assert isinstance(dof_vel, DofVelObsCfg)
    assert isinstance(base_ang_vel, BodyAngularVelocityObsCfg)
    assert isinstance(motion_ref_ori, MotionReferenceOrientationObsCfg)
    return replace(
        cfg,
        observations=replace(
            cfg.observations,
            policy=replace(
                policy,
                dof_pos=replace(dof_pos, noise=UniformNoiseCfg()),
                dof_vel=replace(dof_vel, noise=UniformNoiseCfg()),
                base_ang_vel=replace(base_ang_vel, noise=UniformNoiseCfg()),
                motion_ref_ori_b=replace(motion_ref_ori, noise=UniformNoiseCfg()),
            ),
        ),
        commands=replace(
            cfg.commands,
            motion=replace(
                motion,
                hold_at_clip_end=hold_at_clip_end,
                # Pin the head-frame start so wrap/buffer mechanics stay
                # deterministic regardless of the play start distribution.
                start_at_timestep_zero_prob=1.0,
                sequential_clips=False,
            ),
        ),
    )


def _assert_float_array_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, equal_nan=True)


def _make_numba_env(cfg: WbtEnvCfg, *, num_envs: int, seed: int = 1) -> ManagerEnv:
    return ManagerEnv(cfg, num_envs=num_envs, seed=seed)


def _motion_ref_ori_term(env: ManagerEnv):
    return next(entry.term for entry in env.observation_groups["policy"].terms if entry.name == "motion_ref_ori_b")


def _policy_observation_noise(env: ManagerEnv, state, name: str) -> np.ndarray:
    policy_layout = next(term for term in env.manager_layout.observations["policy"].terms if term.name == name)
    value_layout = next(term for term in env.manager_layout.observations["value"].terms if term.name == name)
    assert state.obs.value is not None
    return state.obs.policy[:, policy_layout.output_slice] - state.obs.value[:, value_layout.output_slice]


def _motion_ref_ori_noise(env: ManagerEnv, state) -> np.ndarray:
    return _policy_observation_noise(env, state, "motion_ref_ori_b")


def _motion_ref_ori_noise_amplitude(env: ManagerEnv) -> float:
    term = env.cfg.observations.policy.motion_ref_ori_b
    assert isinstance(term, MotionReferenceOrientationObsCfg)
    return term.noise.amplitude


def _execute_task_kernel(env: ManagerEnv, state) -> None:
    env._refresh_sim_reads()
    env._execute_task_kernel(env._kernel_inputs)


def test_wbt_reward_terms_are_typed_manager_group() -> None:
    cfg = _deterministic_manager_cfg()
    rewards = cfg.rewards
    assert isinstance(rewards, RewardsCfg)
    assert [field.name for field in fields(rewards)] == [
        "motion_global_ref_position_error_exp",
        "motion_global_ref_orientation_error_exp",
        "motion_relative_body_position_error_exp",
        "motion_relative_body_orientation_error_exp",
        "motion_global_body_lin_vel",
        "motion_global_body_ang_vel",
        "action_rate_l2",
        "limits_dof_pos",
        "undesired_contacts",
    ]
    assert not hasattr(cfg, "reward_config")
    assert rewards.motion_global_ref_position_error_exp.weight == pytest.approx(1.0)
    assert rewards.motion_global_ref_position_error_exp.sigma == pytest.approx(0.3)
    assert rewards.motion_relative_body_orientation_error_exp.weight == pytest.approx(1.0)
    assert rewards.motion_relative_body_orientation_error_exp.sigma == pytest.approx(0.4)
    assert rewards.action_rate_l2.weight == pytest.approx(-0.5)
    assert rewards.limits_dof_pos.soft_limit == pytest.approx(0.9)
    assert rewards.undesired_contacts.threshold == pytest.approx(1.0)


def test_numba_wbt_read_plan_reuses_preallocated_arrays() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=4)
    state = env.init_state()
    assert env._compiled_manager_program is not None
    assert isinstance(env._rand, RandValue)
    motion_joint_entries = [
        entry for group in env.observation_groups.values() for entry in group.terms if entry.name == "motion_joint"
    ]
    assert len(motion_joint_entries) == 2
    motion = _motion_command(env)
    assert all(entry.size == motion.command.shape[1] for entry in motion_joint_entries)
    first = env._kernel_inputs
    env._refresh_sim_reads()
    second = env._kernel_inputs

    assert len(first) == len(env.manager_layout.inputs)
    sources = env._compiled_manager_program.read_plan.sources
    # One source per kernel-data term plus one runtime scalar buffer per float
    # term argument. Scalar args must stay runtime-buffered (not baked into the
    # plan/source): the deterministic config has float term args, so require at
    # least one scalar-buffer source, and the manager context source either way.
    scalar_buffer_sources = [source for source in sources if source.value_type is TermScalarBuffer]
    assert scalar_buffer_sources
    assert any(source.value_type is ManagerContext for source in sources)
    context_source = next(source for source in sources if source.value_type is ManagerContext)
    arrays = tuple(value for value in context_source.values if isinstance(value, np.ndarray))
    assert any(not array.flags.writeable for array in arrays)
    assert motion.steps.shape == (env.num_envs, 1)
    assert motion.clip.joint_pos.shape[0] == motion.clip.tracked_bodies_pos_w.shape[0]
    assert env.sim_data["robot_dof_pos"].shape[0] == env.num_envs
    assert first is second
    for first_value, second_value in zip(first, second, strict=True):
        assert first_value is second_value or np.shares_memory(first_value, second_value) or first_value.size == 0

    action = env.action_terms["joint_position"].state
    actions = np.full((env.num_envs, *env.action_space.shape), 0.25, dtype=np.float32)
    env.apply_action(actions, state)
    partial = env._compiled_manager_program.read_plan.read(
        env,
        np.asarray([1, 3], dtype=np.int64),
    )
    assert partial is first
    assert any(value is action.action_queue for value in first)
    np.testing.assert_array_equal(action.action_queue[:, action.action_ptr[0]], actions)

    state.terminated[:] = [False, True, False, True]
    env._reset_done_envs()
    env._refresh_sim_reads()
    assert env._kernel_inputs is first
    np.testing.assert_array_equal(action.action_queue[[0, 2], action.action_ptr[0]], 0.25)
    np.testing.assert_array_equal(action.action_queue[[1, 3], action.action_ptr[0]], 0.0)


def test_numba_wbt_step_preserves_previous_actor_and_critic_observations() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=2)
    initial = env.init_state()
    assert initial.obs.value is not None
    initial_policy = initial.obs.policy
    initial_value = initial.obs.value
    initial_policy_snapshot = initial_policy.copy()
    initial_value_snapshot = initial_value.copy()

    first = env.step(np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32))

    np.testing.assert_array_equal(initial_policy, initial_policy_snapshot)
    np.testing.assert_array_equal(initial_value, initial_value_snapshot)
    assert not np.shares_memory(initial_policy, first.obs.policy)
    assert first.obs.value is not None
    assert not np.shares_memory(initial_value, first.obs.value)


def test_numba_wbt_observation_noise_bounds_and_determinism() -> None:
    """Observation noise is uniform(-amp, amp), zero for zero amplitudes, and
    reproducible for a fresh environment with the same seed and episode step."""
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    env = _make_numba_env(cfg, num_envs=8)
    state = env.init_state()
    env.warmup()
    term = _motion_ref_ori_term(env)
    amplitude = _motion_ref_ori_noise_amplitude(env)
    assert term.args == (np.float32(amplitude),)
    env._refresh_sim_reads()
    rng_states = env._rand.state
    assert rng_states.shape == (env.num_envs, 1)
    assert rng_states.dtype == np.uint64
    _execute_task_kernel(env, state)

    noise = _motion_ref_ori_noise(env, state)
    assert noise.shape == (env.num_envs, 6)
    assert noise.dtype == np.float32
    assert np.all(noise >= -amplitude)
    assert np.all(noise <= amplitude)
    assert np.any(noise != 0.0)

    # Fresh environment with the same construction sequence reproduces noise.
    cfg2 = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg2, WbtEnvCfg)
    env2 = _make_numba_env(cfg2, num_envs=8)
    state2 = env2.init_state()
    env2.warmup()
    _execute_task_kernel(env2, state2)
    np.testing.assert_array_equal(_motion_ref_ori_noise(env2, state2), noise)


def test_numba_wbt_observation_noise_is_seeded_and_thread_count_independent() -> None:
    base_cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(base_cfg, WbtEnvCfg)
    original_threads = numba.get_num_threads()
    parallel_threads = min(4, original_threads)
    environments = [_make_numba_env(base_cfg, num_envs=32, seed=seed) for seed in (123, 123, 124)]
    states = [env.init_state() for env in environments]
    for env in environments:
        env.warmup()

    try:
        numba.set_num_threads(1)
        _execute_task_kernel(environments[0], states[0])

        numba.set_num_threads(parallel_threads)
        _execute_task_kernel(environments[1], states[1])
        _execute_task_kernel(environments[2], states[2])
        serial = _motion_ref_ori_noise(environments[0], states[0]).copy()
        parallel = _motion_ref_ori_noise(environments[1], states[1]).copy()
        different_seed = _motion_ref_ori_noise(environments[2], states[2]).copy()

        states[0].terminated.fill(True)
        states[1].terminated.fill(True)
        numba.set_num_threads(1)
        environments[0]._reset_done_envs()
        serial_reset = _motion_ref_ori_noise(environments[0], states[0]).copy()
        numba.set_num_threads(parallel_threads)
        environments[1]._reset_done_envs()
        parallel_reset = _motion_ref_ori_noise(environments[1], states[1]).copy()
    finally:
        numba.set_num_threads(original_threads)

    np.testing.assert_array_equal(parallel, serial)
    np.testing.assert_array_equal(parallel_reset, serial_reset)
    assert np.any(different_seed != serial)


def test_numba_wbt_observation_noise_sequence_advances_during_reset() -> None:
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    env = _make_numba_env(cfg, num_envs=4)
    state = env.init_state()
    first_episode = _motion_ref_ori_noise(env, state).copy()
    rng_states = env._rand.state
    states_before_reset = rng_states.copy()

    state.terminated.fill(True)
    env._refresh_sim_reads()
    env._reset_done_envs()
    env.compute_observation(state)

    assert np.any(rng_states != states_before_reset)
    assert np.any(_motion_ref_ori_noise(env, state) != first_episode)


def test_numba_wbt_stateful_noise_has_uniform_distribution() -> None:
    num_envs = 4096
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    env = _make_numba_env(cfg, num_envs=num_envs, seed=20260809)
    state = env.init_state()
    env.warmup()
    _execute_task_kernel(env, state)

    noise = _motion_ref_ori_noise(env, state)
    normalized = noise.reshape(-1) / _motion_ref_ori_noise_amplitude(env)
    assert np.all(normalized >= -1.0)
    assert np.all(normalized < 1.0)
    assert abs(float(normalized.mean())) < 0.01
    assert float(normalized.var()) == pytest.approx(1.0 / 3.0, abs=0.01)


@pytest.mark.parametrize("hold_at_clip_end", [False, True])
def test_numba_wbt_clip_end_behavior(hold_at_clip_end: bool) -> None:
    env = _make_numba_env(_deterministic_manager_cfg(hold_at_clip_end=hold_at_clip_end), num_envs=2)
    state = env.init_state()
    motion = _motion_command(env)
    motion.steps.fill(motion.clip.joint_pos.shape[0] - 1)

    env.compute_transition(state)
    env.compute_observation(state)

    np.testing.assert_array_equal(motion.clip_ended, True)
    expected_step = motion.clip.joint_pos.shape[0] - 1 if hold_at_clip_end else 0
    np.testing.assert_array_equal(motion.steps[:, 0], expected_step)


def test_numba_wbt_clip_wrap_rematerializes_sim_only() -> None:
    # The single dance file keeps the wrap frame adjacent to the episode-start
    # pose, so forcing a wrap there cannot diverge the physics; the corpus
    # directory's final frame belongs to a different clip and would.
    env = _make_numba_env(_single_file_cfg(start_at_timestep_zero_prob=1.0), num_envs=2, seed=11)
    env.init_state()
    motion = _motion_command(env)
    action = env.action_terms["joint_position"].state
    assert isinstance(action, JointPositionActionState)
    num_frames = motion.clip.joint_pos.shape[0]
    # 0.25 is exactly representable in float32, so equality checks stay exact.
    actions = np.full((env.num_envs, *env.action_space.shape), 0.25, dtype=np.float32)

    # Prime persistent action state and the episode counter.
    env.step(actions)
    np.testing.assert_array_equal(action.action_queue[:, action.action_ptr[0]], 0.25)
    episode_steps_before_wrap = env.state.episode_steps.copy()

    # Force every lane to wrap the clip on the next transition.
    motion.steps[:, 0] = num_frames - 1
    state = env.step(actions)

    # The wrap lanes were rematerialized: frames resampled within range.
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    assert np.all(motion.steps[:, 0] <= num_frames - 2)
    # Rematerialization is not an episode boundary.
    np.testing.assert_array_equal(state.terminated, False)
    np.testing.assert_array_equal(state.truncated, False)
    np.testing.assert_array_equal(state.episode_steps, episode_steps_before_wrap + 1)
    # Sim-only: the persistent action state (which an action reset would
    # zero) keeps the raw policy actions of this step.
    np.testing.assert_array_equal(action.action_queue[:, action.action_ptr[0]], 0.25)
    np.testing.assert_array_equal(
        action.action_queue[:, (action.action_ptr[0] - 1) % action.action_queue.shape[1]], 0.25
    )

    # The request flag is cleared before the next physics step: a follow-up
    # step neither rematerializes the lane nor rewinds its frame.
    steps_after_wrap = motion.steps[:, 0].copy()
    next_state = env.step(actions)
    np.testing.assert_array_equal(env._sim_reset_requested[:, 0], False)
    np.testing.assert_array_equal(motion.steps[:, 0], steps_after_wrap + 1)
    np.testing.assert_array_equal(next_state.episode_steps, episode_steps_before_wrap + 2)


def test_numba_wbt_adaptive_sampler_state_is_manager_owned() -> None:
    cfg = _deterministic_manager_cfg()
    command_cfg = cfg.commands.motion
    assert isinstance(command_cfg, WbtMotionCommandCfg)
    command_cfg = replace(
        command_cfg,
        alpha=1.0,
        kernel_size=1,
    )
    command_cfg = replace(command_cfg, adaptive_sampling_enabled=True)
    cfg = replace(
        cfg,
        commands=replace(cfg.commands, motion=command_cfg),
    )
    env = _make_numba_env(cfg, num_envs=3)
    state = env.init_state()
    motion = _motion_command(env)
    expected_num_bins = motion.clip.joint_pos.shape[0] // round(1.0 / cfg.ctrl_dt) + 1
    assert motion.adaptive_bin_failed_count.size == expected_num_bins

    motion.steps[:, 0] = [0, motion.clip.joint_pos.shape[0] // 2, motion.clip.joint_pos.shape[0] - 1]
    motion.adaptive_bin_failed_count.fill(0.0)
    motion.adaptive_current_bin_failed_count.fill(0.0)
    state.terminated[:] = [False, True, True]
    from motrix_env_core.numba.manager.commands import ResetContext

    motion.reset(
        ResetContext(
            env_ids=np.arange(env.num_envs, dtype=np.int64),
            terminated=state.terminated,
            metrics=state.metrics,
        )
    )
    motion.on_transition()

    assert np.sum(motion.adaptive_bin_failed_count) == pytest.approx(2.0)
    np.testing.assert_array_equal(motion.adaptive_current_bin_failed_count, 0.0)
    assert {metric.name for metric in env.manager_layout.metrics} >= {"motion_step", "clip_ended"}
    assert isinstance(state.metrics["adaptive_sampling_entropy"], float)
    assert isinstance(state.metrics["adaptive_sampling_top1_prob"], float)
    assert isinstance(state.metrics["adaptive_sampling_top1_bin"], float)
    assert isinstance(state.metrics["adaptive_failure_mass"], float)


def test_numba_wbt_masked_reset_preserves_bound_buffer_identity() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=3)
    state = env.init_state()
    buffers = env._kernel_buffers
    assert buffers is not None
    env._refresh_sim_reads()
    action_value = env.action_terms["joint_position"].state
    motion = _motion_command(env)
    identities = {
        "action_queue": id(action_value.action_queue),
        "action_ptr": id(action_value.action_ptr),
        "reward_terms": id(buffers[0]),
        "termination_masks": id(buffers[2]),
        "target_body_position_relative": id(motion.target_body_position_relative),
        "sim_inputs": tuple(id(env.sim_data[key]) for key in env.sim_data.keys),
    }
    env.apply_action(np.ones_like(action_value.action_queue[:, 0]), state)
    motion.steps[:, 0] = [1, 2, 3]
    robot_dof_pos = env.sim_data["robot_dof_pos"]
    non_reset_dof_pos = robot_dof_pos[1].copy()
    non_reset_policy = state.obs.policy[1].copy()
    assert state.obs.value is not None
    non_reset_value = state.obs.value[1].copy()
    state.reward[:] = [1.0, 2.0, 3.0]
    state.terminated[:] = [True, False, True]
    reward = state.reward.copy()
    terminated = state.terminated.copy()

    env._reset_done_envs()

    assert env._task_program is not None
    assert env._task_program.reset_kernel.nopython_signatures
    assert id(action_value.action_queue) == identities["action_queue"]
    assert id(action_value.action_ptr) == identities["action_ptr"]
    assert id(buffers[0]) == identities["reward_terms"]
    assert id(buffers[2]) == identities["termination_masks"]
    assert id(motion.target_body_position_relative) == identities["target_body_position_relative"]
    assert tuple(id(env.sim_data[key]) for key in env.sim_data.keys) == identities["sim_inputs"]
    ptr = int(action_value.action_ptr[0])
    previous_ptr = (ptr - 1) % action_value.action_queue.shape[1]
    np.testing.assert_array_equal(action_value.action_queue[[0, 2], ptr], 0.0)
    np.testing.assert_array_equal(action_value.action_queue[1, ptr], 1.0)
    np.testing.assert_array_equal(action_value.action_queue[[0, 2], previous_ptr], 0.0)
    np.testing.assert_array_equal(motion.steps[1, 0], 2)
    np.testing.assert_allclose(robot_dof_pos[[0, 2]], motion.clip.joint_pos[motion.steps[[0, 2], 0]])
    np.testing.assert_array_equal(robot_dof_pos[1], non_reset_dof_pos)
    np.testing.assert_array_equal(state.obs.policy[1], non_reset_policy)
    np.testing.assert_array_equal(state.obs.value[1], non_reset_value)
    np.testing.assert_array_equal(state.reward, reward)
    np.testing.assert_array_equal(state.terminated, terminated)


def test_numba_wbt_manager_rolls_and_resets_action_buffers() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=2)
    state = env.init_state()
    term = env.action_terms["joint_position"]
    assert term is env._action_terms["joint_position"]
    value = term.state

    actions = np.ones((env.num_envs, *env.action_space.shape), dtype=np.float32)
    env.apply_action(actions, state)
    ptr = int(value.action_ptr[0])
    previous_ptr = (ptr - 1) % value.action_queue.shape[1]
    np.testing.assert_array_equal(value.action_queue[:, ptr], 1.0)
    np.testing.assert_array_equal(value.action_queue[:, previous_ptr], 0.0)

    actions.fill(2.0)
    env.apply_action(actions, state)
    ptr = int(value.action_ptr[0])
    previous_ptr = (ptr - 1) % value.action_queue.shape[1]
    np.testing.assert_array_equal(value.action_queue[:, ptr], 2.0)
    np.testing.assert_array_equal(value.action_queue[:, previous_ptr], 1.0)

    state.terminated[:] = True
    ptr_before_reset = value.action_ptr.copy()
    env._reset_done_envs()
    np.testing.assert_array_equal(value.action_queue, 0.0)
    np.testing.assert_array_equal(value.action_ptr, ptr_before_reset)


def test_numba_wbt_action_owns_shared_writable_model_data() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=1)
    state = SimpleNamespace()
    actions = np.full((1, *env.action_space.shape), 0.25, dtype=np.float32)
    value = env.action_terms["joint_position"].state

    env.apply_action(actions, state)

    ptr = int(value.action_ptr[0])
    previous_ptr = (ptr - 1) % value.action_queue.shape[1]
    np.testing.assert_array_equal(value.action_queue[:, ptr], actions)
    np.testing.assert_array_equal(value.action_queue[:, previous_ptr], 0.0)
    assert value.action_queue.flags.writeable
    assert value.action_ptr.flags.writeable
    assert all(
        array.flags.writeable
        for array in (
            value.default_angles,
            value.joint_lower,
            value.joint_upper,
            value.action_scales,
        )
    )
    assert not hasattr(value, "kps")
    np.testing.assert_allclose(
        env._action_writes.buffer("joint_position"),
        actions * value.action_scales + value.default_angles,
    )


def test_manager_rejects_overlapping_wbt_action_routes() -> None:
    cfg = _deterministic_manager_cfg()
    action = cfg.actions.joint_position
    cfg = replace(cfg, actions={"first": action, "second": action})

    with pytest.raises(ValueError, match="both control actuator"):
        _make_numba_env(cfg, num_envs=1)


def test_numba_wbt_registry_uses_generic_manager_env() -> None:
    manager_cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    env = registry.make("g1-wbt-dance", mode="play", num_envs=1)

    assert isinstance(manager_cfg, WbtEnvCfg)
    assert not hasattr(manager_cfg, "manager")
    assert "motion_files" not in {field.name for field in fields(manager_cfg)}
    assert not hasattr(manager_cfg, "tracked_body_names")
    assert not hasattr(manager_cfg, "reference_body_name")
    assert [term_field.name for term_field in fields(manager_cfg.commands)] == ["motion"]
    motion_command_cfg = manager_cfg.commands.motion
    assert isinstance(motion_command_cfg, WbtMotionCommandCfg)
    assert not hasattr(motion_command_cfg, "adaptive_timestep_sampler")
    assert motion_command_cfg.uniform_ratio == pytest.approx(0.1)
    assert motion_command_cfg.alpha == pytest.approx(0.001)
    assert motion_command_cfg.kernel_size == 1
    assert motion_command_cfg.kernel_lambda == pytest.approx(0.8)
    action_cfg = manager_cfg.actions.joint_position
    assert isinstance(action_cfg, JointPositionActionCfg)
    assert not hasattr(manager_cfg, "values")
    assert motion_command_cfg.motion_files == (str(_G1_MOTION_DIR / "dance"),)
    tracked_body_pos = env.sim_data.query("tracked_body_pos")
    assert isinstance(tracked_body_pos, BatchLinkPositionQuery)
    assert motion_command_cfg.tracked_body_names == tracked_body_pos.links
    assert motion_command_cfg.reference_body_name in motion_command_cfg.tracked_body_names
    assert not motion_command_cfg.adaptive_sampling_enabled
    assert not hasattr(action_cfg, "robot")
    assert not hasattr(motion_command_cfg, "robot")
    assert not hasattr(env, "value_manager")
    assert isinstance(manager_cfg.actions, ActionsCfg)
    assert isinstance(manager_cfg.actions.joint_position, JointPositionActionCfg)
    assert isinstance(manager_cfg.sim_reset, ManagerResetCfg)
    assert isinstance(manager_cfg.sim_reset.body_pos, BodyPosResetCfg)
    assert isinstance(manager_cfg.sim_reset.body_rot, BodyRotResetCfg)
    assert isinstance(manager_cfg.sim_reset.body_lin_vel, BodyLinVelResetCfg)
    assert isinstance(manager_cfg.sim_reset.body_rot_vel, BodyRotVelResetCfg)
    assert isinstance(manager_cfg.sim_reset.body_dof_pos, BodyDofPosResetCfg)
    assert not hasattr(manager_cfg, "reset_noise")
    assert manager_cfg.sim_reset.body_pos.noise == (0.05, 0.05, 0.01)
    assert type(env) is ManagerEnv
    assert not hasattr(env, "command_manager")
    assert isinstance(_motion_command(env), WbtMotionCommand)


def test_wbt_manager_play_disables_each_reset_term_noise() -> None:
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")

    assert all(term.noise_scale == 0.0 for term in cfg.sim_reset.to_dict().values())


def test_wbt_manager_play_enables_sequential_playback() -> None:
    """Play walks the motion source as one ordered sequence: start at the
    corpus head, cross clip boundaries in order, and at the corpus end loop
    back to the head — hold_at_clip_end is cleared so playback loops the
    corpus instead of freezing on the final frame."""
    for name in ("g1-29dof-wbt-largebox", "g1-wbt-dance"):
        cfg = registry.make_env_config(name, mode="play")
        assert isinstance(cfg, WbtEnvCfg)
        assert cfg.commands.motion.sequential_clips
        assert not cfg.commands.motion.hold_at_clip_end
        assert not cfg.commands.motion.adaptive_sampling_enabled


def test_single_clip_motion_dir_reproduces_file_clip(tmp_path) -> None:
    """A corpus directory holding one clip assembles bit-for-bit like loading
    that clip through the single-file path (the library's N=1 special case)."""
    corpus_dir = tmp_path / "dance"
    corpus_dir.mkdir()
    (corpus_dir / _DANCE_MOTION.name).write_bytes(_DANCE_MOTION.read_bytes())

    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    motion_cfg = cfg.commands.motion
    assert isinstance(motion_cfg, WbtMotionCommandCfg)
    dir_cfg = replace(
        cfg,
        commands=replace(cfg.commands, motion=replace(motion_cfg, motion_files=(str(corpus_dir),))),
    )

    env = _make_numba_env(dir_cfg, num_envs=2)
    motion = _motion_command(env)
    direct = WbtMotionClip.create(
        MotrixMotion(_DANCE_MOTION),
        list(motion_cfg.joint_names),
        motion_cfg.tracked_body_names,
        motion_cfg.reference_body_name,
        cfg.scene.objs.robot.base_link_name,
        motion_cfg.extension_channels,
    )
    for field in _CLIP_FRAME_FIELDS:
        np.testing.assert_array_equal(getattr(motion.clip, field), getattr(direct, field))
    np.testing.assert_array_equal(motion.clip.frame_clip_end, motion.clip.joint_pos.shape[0] - 1)


def test_wbt_robot_config_subclasses_isolate_nested_overrides() -> None:
    g1 = registry.make_env_config("g1-wbt-dance")
    another_g1 = registry.make_env_config("g1-wbt-dance")
    k1 = registry.make_env_config("k1-wbt-freekick")
    dex_evt = registry.make_env_config("dex-evt-wbt-dance")

    assert isinstance(g1, G1WbtEnvCfg)
    assert isinstance(another_g1, G1WbtEnvCfg)
    assert isinstance(k1, K1WbtEnvCfg)
    assert isinstance(dex_evt, DexEvtWbtEnvCfg)
    assert g1.commands.motion.joint_names
    assert g1.commands.motion.tracked_body_names
    assert g1.commands.motion.reference_body_name == "torso_link"
    assert k1.commands.motion.reference_body_name == "Trunk"
    assert dex_evt.commands.motion.reference_body_name == "waist_pitch_link"
    assert g1.queries.model["actuator_kp"] == k1.queries.model["actuator_kp"]
    assert g1.queries.model["actuator_kp"] == dex_evt.queries.model["actuator_kp"]
    assert g1.queries.model["robot_joint_position_limits"].body == "pelvis"
    assert k1.queries.model["robot_joint_position_limits"].body == "Trunk"
    assert dex_evt.queries.model["robot_joint_position_limits"].body == "pelvis"
    assert g1.queries is not another_g1.queries
    assert g1.queries.data is not another_g1.queries.data
    assert g1.queries.model is not another_g1.queries.model
    assert g1.rewards is not another_g1.rewards
    assert g1.commands.motion is not another_g1.commands.motion


def test_wbt_queries_use_authoritative_motion_orders() -> None:
    cfg = registry.make_env_config("g1-wbt-dance")
    assert isinstance(cfg, G1WbtEnvCfg)
    joint_names = cfg.commands.motion.joint_names
    tracked_body_names = cfg.commands.motion.tracked_body_names
    assert cfg.queries.model["robot_joint_position_limits"].body == cfg.scene.objs.robot.resolved_base_link_name
    assert cfg.queries.data["undesired_contact_forces"].body == cfg.scene.objs.robot.resolved_base_link_name
    assert cfg.queries.data["robot_dof_pos"].joints == joint_names
    assert cfg.queries.data["robot_dof_vel"].joints == joint_names
    assert cfg.queries.data["tracked_body_pos"].links == tracked_body_names
    assert cfg.queries.data["tracked_body_quat"].links == tracked_body_names
    assert cfg.queries.data["tracked_body_linear_velocity"].links == tracked_body_names
    assert cfg.queries.data["tracked_body_angular_velocity"].links == tracked_body_names


def test_wbt_build_spec_is_spawn_pickle_safe() -> None:
    spec = registry.resolve("g1-wbt-dance")

    pickle.dumps(spec)
    assert spec.env_cfg.queries.data["robot_dof_pos"].joints == spec.env_cfg.commands.motion.joint_names
    assert spec.env_cfg.queries.data["tracked_body_pos"].links == spec.env_cfg.commands.motion.tracked_body_names


@pytest.mark.parametrize(
    "env_name",
    ["dex-evt-wbt-dance", "g1-29dof-wbt-largebox", "g1-wbt-dance", "k1-wbt-freekick"],
)
def test_numba_wbt_manager_builds_for_all_wbt_presets(env_name: str) -> None:
    env = registry.make(env_name, mode="play", num_envs=2)
    assert isinstance(env, ManagerEnv)
    assert env.cfg.queries.data["robot_dof_pos"].joints == env.cfg.commands.motion.joint_names
    assert env.sim_data.query("robot_dof_pos").joints == env.cfg.commands.motion.joint_names
    env.init_state()
    action = env.action_terms["joint_position"].state
    assert isinstance(action, JointPositionActionState)
    joint_lower, joint_upper = env.model.others["robot_joint_position_limits"]
    np.testing.assert_array_equal(action.joint_lower, joint_lower)
    np.testing.assert_array_equal(action.joint_upper, joint_upper)
    assert action.joint_lower.shape == env.sim_data["robot_dof_pos"].shape[1:]

    policy_terms = env.manager_layout.observations["policy"].terms
    value_terms = env.manager_layout.observations["value"].terms
    assert [term.name for term in policy_terms] == [
        "motion_joint",
        "motion_ref_ori_b",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "actions",
    ]
    assert [term.name for term in value_terms] == [
        "motion_joint",
        "motion_ref_pos_b",
        "motion_ref_ori_b",
        "robot_body_pos_b",
        "robot_body_ori_b",
        "base_lin_vel",
        "base_ang_vel",
        "dof_pos",
        "dof_vel",
        "actions",
    ]
    observation_space = env.observation_space
    assert observation_space.value is not None
    for terms, width in (
        (policy_terms, observation_space.policy.shape[0]),
        (value_terms, observation_space.value.shape[0]),
    ):
        assert terms[0].output_slice.start == 0
        assert all(left.output_slice.stop == right.output_slice.start for left, right in zip(terms, terms[1:]))
        assert terms[-1].output_slice.stop == width

    assert [term.name for term in env.manager_layout.rewards] == [field.name for field in fields(env.cfg.rewards)]
    assert [term.name for term in env.manager_layout.terminations] == [
        "bad_ref_z",
        "bad_ref_ori",
        "bad_body_z",
        "bad_dof_pos",
        "bad_dof_vel",
    ]


def test_g1_wbt_dance_mgr_uses_manager_environment() -> None:
    spec = registry.resolve("g1-wbt-dance")

    assert spec.env_cls is ManagerEnv
    assert isinstance(spec.env_cfg, WbtEnvCfg)


# ---------------------------------------------------------------------------
# Multi-clip corpus (MotionLibrary) behavior
# ---------------------------------------------------------------------------

_DANCE_MOTION = _G1_MOTION_DIR / "dance" / "dance1_subject2.npz"
_DANCE_SPLIT = 500


def _split_dance_corpus(tmp_path, *, split=_DANCE_SPLIT, extension_channels=()):
    """Split the bundled G1 dance motion into two schema v1 files.

    Slicing preserves every per-frame field, so the concatenated corpus
    reproduces the original clip bit-for-bit on the global frame axis.
    """
    with np.load(_DANCE_MOTION, allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files}
    total = int(np.asarray(fields["num_frames"]).reshape(-1)[0])
    rng = np.random.default_rng(7)
    paths = []
    bounds = (0, split, total)
    for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        part = {"num_frames": np.int32(stop - start)}
        for name, value in fields.items():
            if name == "num_frames":
                continue
            # Per-frame arrays (including ext_ channels) carry the frame axis;
            # names and scalars are corpus-wide metadata.
            if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
                part[name] = value[start:stop]
            else:
                part[name] = value
        for channel in extension_channels:
            part[f"ext_{channel}"] = rng.standard_normal((stop - start, 3)).astype(np.float32)
        path = tmp_path / f"part{index}.npz"
        np.savez(path, **part)
        paths.append(str(path))
    return tuple(paths)


def _multi_clip_cfg(
    tmp_path,
    *,
    hold_at_clip_end: bool = False,
    sequential_clips: bool = False,
    start_at_timestep_zero_prob: float | None = None,
    adaptive_sampling_enabled: bool | None = None,
    alpha: float | None = None,
    split: int = _DANCE_SPLIT,
    extension_channels: tuple[str, ...] = (),
    motion_files: tuple[str, ...] | None = None,
) -> WbtEnvCfg:
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    motion = cfg.commands.motion
    assert isinstance(motion, WbtMotionCommandCfg)
    replacements = {
        "motion_files": motion_files
        if motion_files is not None
        else _split_dance_corpus(tmp_path, split=split, extension_channels=extension_channels),
        "extension_channels": tuple(extension_channels),
        "hold_at_clip_end": hold_at_clip_end,
        "sequential_clips": sequential_clips,
    }
    if start_at_timestep_zero_prob is not None:
        replacements["start_at_timestep_zero_prob"] = start_at_timestep_zero_prob
    if adaptive_sampling_enabled is not None:
        replacements["adaptive_sampling_enabled"] = adaptive_sampling_enabled
    if alpha is not None:
        replacements["alpha"] = alpha
    return replace(cfg, commands=replace(cfg.commands, motion=replace(motion, **replacements)))


def _single_file_cfg(*, start_at_timestep_zero_prob: float | None = None) -> WbtEnvCfg:
    """A play config pinned to the bundled dance file.

    The g1-wbt-dance corpus directory holds however many clips are currently
    dropped into it, so tests that need one exact clip load the file directly
    instead of going through the registry preset.
    """
    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    assert isinstance(cfg, WbtEnvCfg)
    motion = cfg.commands.motion
    assert isinstance(motion, WbtMotionCommandCfg)
    replacements: dict[str, object] = {
        "motion_files": (str(_DANCE_MOTION),),
        "hold_at_clip_end": False,
        "sequential_clips": False,
    }
    if start_at_timestep_zero_prob is not None:
        replacements["start_at_timestep_zero_prob"] = start_at_timestep_zero_prob
    return replace(cfg, commands=replace(cfg.commands, motion=replace(motion, **replacements)))


_CLIP_FRAME_FIELDS = (
    "joint_pos",
    "joint_vel",
    "tracked_bodies_pos_w",
    "tracked_bodies_quat_w",
    "tracked_bodies_lin_vel_w",
    "tracked_bodies_ang_vel_w",
    "root_body_pos_w",
    "root_body_quat_w",
    "root_body_lin_vel_w",
    "root_body_ang_vel_w",
    "reference_body_pos_w",
    "reference_body_quat_w",
)


def test_numba_wbt_multi_clip_corpus_reproduces_single_clip_arrays(tmp_path) -> None:
    multi = _make_numba_env(_multi_clip_cfg(tmp_path), num_envs=2)
    single = _make_numba_env(_single_file_cfg(), num_envs=2)
    clip = _motion_command(multi).clip
    single_clip = _motion_command(single).clip

    for field in _CLIP_FRAME_FIELDS:
        np.testing.assert_array_equal(getattr(clip, field), getattr(single_clip, field))
    last = single_clip.joint_pos.shape[0] - 1
    np.testing.assert_array_equal(clip.frame_clip_end[:_DANCE_SPLIT], _DANCE_SPLIT - 1)
    np.testing.assert_array_equal(clip.frame_clip_end[_DANCE_SPLIT:], last)


def test_numba_wbt_multi_clip_wrap_rematerializes_sim_only(tmp_path) -> None:
    # Split after two frames: the first clip's final frame is adjacent to the
    # episode-start pose, so forcing a wrap there cannot diverge the physics.
    # The head-frame start keeps the priming step on that same pose.
    cfg = _multi_clip_cfg(tmp_path, split=2, start_at_timestep_zero_prob=1.0)
    env = _make_numba_env(cfg, num_envs=2, seed=11)
    env.init_state()
    motion = _motion_command(env)
    action = env.action_terms["joint_position"].state
    assert isinstance(action, JointPositionActionState)
    total = motion.clip.joint_pos.shape[0]
    actions = np.full((env.num_envs, *env.action_space.shape), 0.25, dtype=np.float32)

    env.step(actions)
    episode_steps_before_wrap = env.state.episode_steps.copy()

    # A lane inside the second clip advances normally: no wrap, no rematerialize.
    motion.steps[:, 0] = 2
    state = env.step(actions)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], False)
    np.testing.assert_array_equal(motion.steps[:, 0], 3)
    np.testing.assert_array_equal(state.terminated, False)
    np.testing.assert_array_equal(env._sim_reset_requested[:, 0], False)

    # A lane on the first clip's final frame wraps at the clip boundary, not at
    # the corpus end: the frame is resampled over the whole corpus and the reset
    # pipeline rematerializes the lane without ending the episode.
    motion.steps[:, 0] = 1
    state = env.step(actions)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    assert np.all(motion.steps[:, 0] <= total - 2)
    np.testing.assert_array_equal(state.terminated, False)
    np.testing.assert_array_equal(state.truncated, False)
    np.testing.assert_array_equal(state.episode_steps, episode_steps_before_wrap + 2)
    np.testing.assert_array_equal(action.action_queue[:, action.action_ptr[0]], 0.25)
    np.testing.assert_array_equal(
        action.action_queue[:, (action.action_ptr[0] - 1) % action.action_queue.shape[1]], 0.25
    )

    steps_after_wrap = motion.steps[:, 0].copy()
    env.step(actions)
    np.testing.assert_array_equal(env._sim_reset_requested[:, 0], False)
    np.testing.assert_array_equal(motion.steps[:, 0], steps_after_wrap + 1)


def test_numba_wbt_multi_clip_hold_clamps_to_own_clip_end(tmp_path) -> None:
    env = _make_numba_env(_multi_clip_cfg(tmp_path, hold_at_clip_end=True), num_envs=2)
    state = env.init_state()
    motion = _motion_command(env)
    total = motion.clip.joint_pos.shape[0]

    # Hold on the first clip keeps the lane on that clip's final frame, not on
    # the corpus final frame.
    motion.steps.fill(_DANCE_SPLIT - 1)
    env.compute_transition(state)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    np.testing.assert_array_equal(motion.steps[:, 0], _DANCE_SPLIT - 1)

    # The corpus-final clip still holds on the global final frame.
    motion.steps.fill(total - 1)
    env.compute_transition(state)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    np.testing.assert_array_equal(motion.steps[:, 0], total - 1)


def test_numba_wbt_multi_clip_sequential_crosses_boundaries_and_loops(tmp_path) -> None:
    # Split after two frames so boundary crossings stay adjacent to the
    # episode-start pose and cannot diverge the physics (same trick as the
    # wrap test above).
    cfg = _multi_clip_cfg(tmp_path, sequential_clips=True, split=2)
    env = _make_numba_env(cfg, num_envs=2, seed=11)
    env.init_state()
    motion = _motion_command(env)
    action = env.action_terms["joint_position"].state
    assert isinstance(action, JointPositionActionState)
    total = motion.clip.joint_pos.shape[0]
    actions = np.full((env.num_envs, *env.action_space.shape), 0.25, dtype=np.float32)

    # The initial reset keeps the initialized corpus-head frame.
    np.testing.assert_array_equal(motion.steps[:, 0], 0)

    env.step(actions)
    episode_steps_before_boundary = env.state.episode_steps.copy()

    # A lane inside the second clip advances normally: no boundary crossed.
    motion.steps[:, 0] = 2
    state = env.step(actions)
    np.testing.assert_array_equal(motion.steps[:, 0], 3)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], False)
    np.testing.assert_array_equal(env._sim_reset_requested[:, 0], False)

    # The first clip's final frame crosses into the second clip's head frame
    # instead of resampling, rematerializing the lane there.
    motion.steps[:, 0] = 1
    state = env.step(actions)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    np.testing.assert_array_equal(motion.steps[:, 0], 2)
    np.testing.assert_array_equal(state.terminated, False)
    np.testing.assert_array_equal(state.truncated, False)
    np.testing.assert_array_equal(state.episode_steps, episode_steps_before_boundary + 2)
    # Rematerialization is sim-only: the persistent action state keeps the
    # raw policy actions of this step.
    np.testing.assert_array_equal(action.action_queue[:, action.action_ptr[0]], 0.25)
    np.testing.assert_array_equal(
        action.action_queue[:, (action.action_ptr[0] - 1) % action.action_queue.shape[1]], 0.25
    )

    # The corpus-final frame loops back to the corpus head frame.
    motion.steps[:, 0] = total - 1
    state = env.step(actions)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    np.testing.assert_array_equal(motion.steps[:, 0], 0)
    np.testing.assert_array_equal(state.terminated, False)


def test_numba_wbt_multi_clip_sequential_reset_keeps_frame(tmp_path) -> None:
    env = _make_numba_env(_multi_clip_cfg(tmp_path, sequential_clips=True), num_envs=3)
    state = env.init_state()
    motion = _motion_command(env)
    frames = [_DANCE_SPLIT - 1, _DANCE_SPLIT, motion.clip.joint_pos.shape[0] - 1]

    # Episode resets keep the lane on its current frame: the robot is
    # rematerialized onto that reference frame and the sequence continues.
    motion.steps[:, 0] = frames
    env._refresh_sim_reads()
    state.terminated[:] = True
    env._reset_done_envs()

    np.testing.assert_array_equal(motion.steps[:, 0], frames)
    np.testing.assert_allclose(env.sim_data["robot_dof_pos"], motion.clip.joint_pos[motion.steps[:, 0]])


def test_numba_wbt_multi_clip_sequential_hold_stops_at_corpus_end(tmp_path) -> None:
    env = _make_numba_env(_multi_clip_cfg(tmp_path, sequential_clips=True, hold_at_clip_end=True, split=2), num_envs=2)
    state = env.init_state()
    motion = _motion_command(env)
    total = motion.clip.joint_pos.shape[0]

    # Intermediate boundaries still cross into the next clip's head frame.
    motion.steps[:, 0] = 1
    env.compute_transition(state)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    np.testing.assert_array_equal(motion.steps[:, 0], 2)

    # The corpus end holds on the final frame instead of looping back.
    env._sim_reset_requested.fill(False)
    motion.steps[:, 0] = total - 1
    env.compute_transition(state)
    np.testing.assert_array_equal(motion.clip_ended[:, 0], True)
    np.testing.assert_array_equal(motion.steps[:, 0], total - 1)
    np.testing.assert_array_equal(env._sim_reset_requested[:, 0], False)


def test_numba_wbt_multi_clip_sampling_sequence_matches_single_clip(tmp_path) -> None:
    """Same-seed start-frame draws over the split corpus match the original file."""

    def build(cfg: WbtEnvCfg) -> list[np.ndarray]:
        env = _make_numba_env(cfg, num_envs=4, seed=123)
        motion = _motion_command(env)
        state = env.init_state()
        env.warmup()
        sequence = [motion.steps[:, 0].copy()]
        for _ in range(3):
            state.terminated.fill(True)
            env._refresh_sim_reads()
            env._reset_done_envs()
            sequence.append(motion.steps[:, 0].copy())
        return sequence

    single_cfg = _single_file_cfg(start_at_timestep_zero_prob=0.0)
    multi_cfg = _multi_clip_cfg(tmp_path, start_at_timestep_zero_prob=0.0)

    for multi_draws, single_draws in zip(
        build(multi_cfg),
        build(single_cfg),
        strict=True,
    ):
        np.testing.assert_array_equal(multi_draws, single_draws)


def test_numba_wbt_multi_clip_extension_channels_reach_kernel_inputs(tmp_path) -> None:
    cfg = _multi_clip_cfg(tmp_path, extension_channels=("aux_reference",))
    env = _make_numba_env(cfg, num_envs=2)
    env.init_state()
    motion = _motion_command(env)
    channel = motion.clip.extensions["aux_reference"].data
    total = motion.clip.joint_pos.shape[0]
    assert channel.shape == (total, 3)
    assert channel.dtype == np.float32

    env._refresh_sim_reads()
    arrays = (value for value in env._kernel_inputs if isinstance(value, np.ndarray))
    assert any(np.shares_memory(value, channel) for value in arrays)


def test_numba_wbt_multi_clip_adaptive_bins_span_whole_corpus(tmp_path) -> None:
    # alpha=1.0 folds only this update's failure counts into the histogram.
    cfg = _multi_clip_cfg(tmp_path, adaptive_sampling_enabled=True, alpha=1.0)
    env = _make_numba_env(cfg, num_envs=3)
    state = env.init_state()
    motion = _motion_command(env)
    total = motion.clip.joint_pos.shape[0]
    env_fps = round(1.0 / cfg.ctrl_dt)
    expected_num_bins = total // env_fps + 1
    assert motion.adaptive_bin_failed_count.size == expected_num_bins

    # Failures on frames of both clips fold into global bins without going out
    # of range, and the rebuilt CDF stays normalized.
    motion.steps[:, 0] = [_DANCE_SPLIT - 1, _DANCE_SPLIT, total - 1]
    motion.adaptive_bin_failed_count.fill(0.0)
    motion.adaptive_current_bin_failed_count.fill(0.0)
    state.terminated[:] = [True, True, True]
    motion.reset(
        ResetContext(
            env_ids=np.arange(env.num_envs, dtype=np.int64),
            terminated=state.terminated,
            metrics=state.metrics,
        )
    )
    motion.on_transition()
    assert np.sum(motion.adaptive_bin_failed_count) == pytest.approx(3.0)
    assert np.all(motion.sampling_cdf[:-1] <= motion.sampling_cdf[1:] + 1e-6)
    assert motion.sampling_cdf[-1] == pytest.approx(1.0)


def test_numba_wbt_multi_clip_cfg_survives_pickle_round_trip(tmp_path) -> None:
    """Async collectors receive the env spec via pickle; the assembled corpus
    must survive the round-trip unchanged."""
    cfg = _multi_clip_cfg(tmp_path)
    restored = pickle.loads(pickle.dumps(cfg))
    assert isinstance(restored, WbtEnvCfg)
    env = _make_numba_env(restored, num_envs=2)
    motion_term = _motion_command(env)
    assert motion_term.clip.frame_clip_end[_DANCE_SPLIT - 1] == _DANCE_SPLIT - 1
    assert motion_term.clip.frame_clip_end[-1] == 999 - 1
