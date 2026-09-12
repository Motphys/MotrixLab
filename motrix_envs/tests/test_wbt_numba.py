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
from motrix_env_core.mdp.observations import (  # noqa: E402
    RobotBaseAngularVelocityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.state import RandValue  # noqa: E402
from motrix_env_core.sim import BatchLinkPositionQuery  # noqa: E402
from motrix_envs.locomotion.wbt.cfg import (  # noqa: E402
    ActionsCfg,
    RewardsCfg,
    WbtEnvCfg,
)
from motrix_envs.locomotion.wbt.dex_evt import DexEvtWbtEnvCfg  # noqa: E402
from motrix_envs.locomotion.wbt.g1 import G1WbtEnvCfg  # noqa: E402
from motrix_envs.locomotion.wbt.k1 import K1WbtEnvCfg  # noqa: E402
from motrix_envs.locomotion.wbt.mdp.action import (  # noqa: E402
    WbtJointPositionAction,
    WbtJointPositionActionCfg,
)
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
    assert isinstance(base_ang_vel, RobotBaseAngularVelocityObsCfg)
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
    assert all(entry.size == motion.command_buffer.shape[1] for entry in motion_joint_entries)
    first = env._kernel_inputs
    env._refresh_sim_reads()
    second = env._kernel_inputs

    assert len(first) == len(env.manager_layout.inputs)
    sources = env._compiled_manager_program.read_plan.sources
    # One source per kernel-data term plus one runtime scalar buffer per float
    # term argument; the manager context source must be present either way.
    context_source = next(source for source in sources if source.value_type is ManagerContext)
    arrays = tuple(value for value in context_source.values if isinstance(value, np.ndarray))
    assert any(not array.flags.writeable for array in arrays)
    assert motion.steps.shape == (env.num_envs, 1)
    assert motion.clip.joint_pos.shape[0] == motion.clip.tracked_bodies_pos_w.shape[0]
    assert env.sim_data["robot_dof_pos"].shape[0] == env.num_envs
    assert first is second
    for first_value, second_value in zip(first, second, strict=True):
        assert first_value is second_value or np.shares_memory(first_value, second_value) or first_value.size == 0

    action = env.action_terms["joint_position"]
    actions = np.full((env.num_envs, *env.action_space.shape), 0.25, dtype=np.float32)
    env.apply_action(actions, state)
    partial = env._compiled_manager_program.read_plan.read(
        env,
        np.asarray([1, 3], dtype=np.int64),
    )
    assert partial is first
    assert any(value is action.current for value in first)
    np.testing.assert_array_equal(action.current, actions)

    state.terminated[:] = [False, True, False, True]
    env._reset_done_envs()
    env._refresh_sim_reads()
    assert env._kernel_inputs is first
    np.testing.assert_array_equal(action.current[[0, 2]], 0.25)
    np.testing.assert_array_equal(action.current[[1, 3]], 0.0)


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
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=2, seed=11)
    env.init_state()
    motion = _motion_command(env)
    action = env.action_terms["joint_position"]
    assert isinstance(action, WbtJointPositionAction)
    num_frames = motion.clip.joint_pos.shape[0]
    # 0.25 is exactly representable in float32, so equality checks stay exact.
    actions = np.full((env.num_envs, *env.action_space.shape), 0.25, dtype=np.float32)

    # Prime persistent action state and the episode counter.
    env.step(actions)
    np.testing.assert_array_equal(action.current, 0.25)
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
    # zero) keeps the processed values of this step.
    np.testing.assert_array_equal(action.current, 0.25)
    np.testing.assert_array_equal(action.previous, 0.25)

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
        ctrl_dt=0.04,
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
    action_value = env.action_terms["joint_position"]
    motion = _motion_command(env)
    identities = {
        "current_actions": id(action_value.current),
        "last_actions": id(action_value.previous),
        "reward_terms": id(buffers[0]),
        "termination_masks": id(buffers[2]),
        "target_body_position_relative": id(motion.target_body_position_relative),
        "sim_inputs": tuple(id(env.sim_data[key]) for key in env.sim_data.keys),
    }
    env.apply_action(np.ones_like(action_value.current), state)
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
    assert id(action_value.current) == identities["current_actions"]
    assert id(action_value.previous) == identities["last_actions"]
    assert id(buffers[0]) == identities["reward_terms"]
    assert id(buffers[2]) == identities["termination_masks"]
    assert id(motion.target_body_position_relative) == identities["target_body_position_relative"]
    assert tuple(id(env.sim_data[key]) for key in env.sim_data.keys) == identities["sim_inputs"]
    np.testing.assert_array_equal(action_value.current[[0, 2]], 0.0)
    np.testing.assert_array_equal(action_value.current[1], 1.0)
    np.testing.assert_array_equal(action_value.previous[[0, 2]], 0.0)
    np.testing.assert_array_equal(motion.steps[1, 0], 2)
    np.testing.assert_allclose(robot_dof_pos[[0, 2]], motion.clip.joint_pos[motion.steps[[0, 2], 0]])
    np.testing.assert_array_equal(robot_dof_pos[1], non_reset_dof_pos)
    np.testing.assert_array_equal(state.obs.policy[1], non_reset_policy)
    np.testing.assert_array_equal(state.obs.value[1], non_reset_value)
    np.testing.assert_array_equal(state.reward, reward)
    np.testing.assert_array_equal(state.terminated, terminated)


def test_numba_wbt_action_term_owns_rolls_and_resets_action_buffers() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=2)
    state = env.init_state()
    value = env.action_terms["joint_position"]
    assert value is env._action_terms["joint_position"]

    env.apply_action(np.ones_like(value.current), state)
    np.testing.assert_array_equal(value.current, 1.0)
    np.testing.assert_array_equal(value.previous, 0.0)

    env.apply_action(np.full_like(value.current, 2.0), state)
    np.testing.assert_array_equal(value.current, 2.0)
    np.testing.assert_array_equal(value.previous, 1.0)

    state.terminated[:] = True
    env._reset_done_envs()
    np.testing.assert_array_equal(value.current, 0.0)
    np.testing.assert_array_equal(value.previous, 0.0)


def test_numba_wbt_action_owns_shared_writable_model_data() -> None:
    env = _make_numba_env(_deterministic_manager_cfg(), num_envs=1)
    state = SimpleNamespace()
    actions = np.full((1, *env.action_space.shape), 0.25, dtype=np.float32)
    value = env.action_terms["joint_position"]

    env.apply_action(actions, state)

    np.testing.assert_array_equal(value.current, actions)
    np.testing.assert_array_equal(value.previous, 0.0)
    assert value.current.flags.writeable
    assert value.previous.flags.writeable
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
    assert "motion_file" not in {field.name for field in fields(manager_cfg)}
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
    assert isinstance(action_cfg, WbtJointPositionActionCfg)
    assert not hasattr(manager_cfg, "values")
    assert motion_command_cfg.motion_file
    tracked_body_pos = env.sim_data.query("tracked_body_pos")
    assert isinstance(tracked_body_pos, BatchLinkPositionQuery)
    assert motion_command_cfg.tracked_body_names == tracked_body_pos.links
    assert motion_command_cfg.reference_body_name in motion_command_cfg.tracked_body_names
    assert not motion_command_cfg.adaptive_sampling_enabled
    assert not hasattr(action_cfg, "robot")
    assert not hasattr(motion_command_cfg, "robot")
    assert not hasattr(env, "value_manager")
    assert isinstance(manager_cfg.actions, ActionsCfg)
    assert isinstance(manager_cfg.actions.joint_position, WbtJointPositionActionCfg)
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
    action = env.action_terms["joint_position"]
    assert isinstance(action, WbtJointPositionAction)
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
