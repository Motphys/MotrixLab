# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    HFieldTerrainCfg,
    KeyPoseCfg,
    MjcfFileCfg,
    ProceduralHFieldAssetCfg,
    SceneCfg,
    SceneObjsCfg,
)
from motrix_env_core.config.scene.base import RobotCfg
from motrix_env_core.sim import BodyAngularVelocityWrite
from motrix_env_core.sim.model import ActuatorType
from motrix_env_motrixsim.runtime import MotrixSimBackend
from motrix_envs.locomotion.quadruped.cfg import RewardScales
from motrix_envs.locomotion.quadruped.velocity_command import RandomPlanarVelocityBinding
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask, _sim_data_queries


def _read_param_overrides(env):
    """Read the per-env kp/damping/friction state through native motrixsim getters.

    The framework override API is write-only (nominal bases come from declared
    model queries), so verification reads the backend objects directly.
    """
    model, data = env.sim._model, env.sim._data
    kp = np.column_stack([np.asarray(a.get_kp_override(data)).reshape(-1) for a in model.actuators])
    damping = np.column_stack([np.asarray(a.get_kd_override(data)).reshape(-1) for a in model.actuators])
    friction = np.asarray(model.get_geom(env.cfg.ground_geom_name).get_friction_override(data), dtype=np.float32)
    return None, kp, damping, friction


def _read_mass_overrides(env):
    link = env.sim._model.get_link(env._base_link_name)
    data = env.sim._data
    return (
        np.asarray(link.get_mass_override(data), dtype=np.float32).reshape(-1),
        np.asarray(link.get_center_of_mass_override(data), dtype=np.float32).reshape(-1, 3),
    )


def _nominal_params(env):
    """Nominal model values resolved through the env's declared model queries."""
    return (
        np.broadcast_to(env.model.others["actuator_kp"], (env.num_envs, len(env.model.actuators))).copy(),
        np.broadcast_to(env.model.others["actuator_kd"], (env.num_envs, len(env.model.actuators))).copy(),
    )


def _nominal_masses(env):
    return (
        np.full((env.num_envs,), env.model.others["base_mass"], dtype=np.float32),
        np.broadcast_to(env.model.others["base_com"], (env.num_envs, 3)).copy(),
    )


@pytest.fixture
def quadruped_env():
    return registry.make("go1-walk-flat", num_envs=1, mode="train")


def _all(value) -> bool:
    return bool(np.asarray(value).all())


def _any(value) -> bool:
    return bool(np.asarray(value).any())


def _swing_reward(quadruped_env, contacts: np.ndarray) -> np.ndarray:
    quadruped_env._feet_phase[:] = 0.75
    quadruped_env.feet_contact.copy_(torch.as_tensor(contacts))
    return quadruped_env._reward_swing_feet_z(None, None, None)


def test_quadruped_reward_scales_are_structured_and_not_shared():
    go1 = registry.make_env_config("go1-walk-flat").reward_config.scales
    go2 = registry.make_env_config("go2-walk-flat").reward_config.scales

    assert isinstance(go1, RewardScales)
    assert isinstance(go2, RewardScales)
    assert go1 is not go2


def test_go2_uses_locomotion_first_reward_set():
    env = registry.make("go2-walk-flat", num_envs=1, mode="train")
    env.init_state()
    env.step(torch.zeros((env.num_envs, *env.action_space.shape), dtype=torch.float32))

    assert "similar_to_default" in env.state.reward_terms
    assert "dof_acc" in env.state.reward_terms
    assert "hip_position" in env.state.reward_terms
    assert "base_com_position" in env.state.reward_terms

    baseline = registry.make("go1-walk-flat", num_envs=1, mode="train")
    baseline.init_state()
    baseline.step(np.zeros((baseline.num_envs, *baseline.action_space.shape), dtype=np.float32))

    assert "similar_to_default" in baseline.state.reward_terms
    assert "dof_acc" not in baseline.state.reward_terms
    assert "hip_position" not in baseline.state.reward_terms


def test_go2_velocity_command_range_matches_supported_envelope():
    velocity_cfg = registry.make_env_config("go2-walk-flat").commands.velocity

    np.testing.assert_allclose(velocity_cfg.lower, np.array([-1.0, -0.5, -1.0], dtype=np.float32))
    np.testing.assert_allclose(velocity_cfg.upper, np.array([1.0, 0.5, 1.0], dtype=np.float32))


def test_go2_posture_constraints_use_tighter_tolerances_and_more_standing():
    env_cfg = registry.make_env_config("go2-walk-flat")

    assert env_cfg.commands.velocity.standing_probability == pytest.approx(0.50)
    assert env_cfg.commands.velocity.hard_case_probability == pytest.approx(0.20)
    assert len(env_cfg.commands.velocity.hard_cases) == 24
    hard_cases = {tuple(case) for case in env_cfg.commands.velocity.hard_cases}
    assert all((-case[0], case[1], case[2]) in hard_cases for case in hard_cases)
    assert env_cfg.reward_config.feet_spacing_tolerance == pytest.approx(0.03)
    assert env_cfg.reward_config.scales.flat_orientation == pytest.approx(-4.0)
    # Raised from -0.3 to damp rapid roll/pitch oscillation without changing
    # the flat-body target or any tracking/gait reward.
    assert env_cfg.reward_config.scales.ang_vel_xy == pytest.approx(-0.5)
    assert env_cfg.reward_config.scales.dof_acc == pytest.approx(-2e-5)
    assert env_cfg.reward_config.scales.base_com_position == pytest.approx(-0.5)
    assert env_cfg.reward_config.base_com_position_tolerance == pytest.approx(0.02)
    assert env_cfg.reward_config.hip_position_tolerance == pytest.approx(0.08)
    assert env_cfg.reward_config.base_height_target == pytest.approx(0.305)
    # Raised from -100 to reduce base-height oscillation without changing the
    # target or any gait/tracking reward.
    assert env_cfg.reward_config.scales.base_height == pytest.approx(-300.0)
    assert env_cfg.reward_config.swing_fraction == pytest.approx(0.45)
    assert env_cfg.reward_config.swing_feet_height_sigma == pytest.approx(0.05)
    assert env_cfg.randomization.sliding_friction_range == (0.6, 1.4)
    # Raised from 5.0 in the low-speed dragging intervention while keeping
    # target height and all other gait rewards unchanged.
    assert env_cfg.reward_config.scales.swing_feet_z == pytest.approx(6.0)
    assert env_cfg.reward_config.scales.swing_contact == pytest.approx(-1.5)
    assert env_cfg.reward_config.scales.touchdown_velocity == pytest.approx(-0.8)
    assert env_cfg.reward_config.touchdown_speed_sigma == pytest.approx(0.35)
    assert env_cfg.reward_config.tracking_lin_vel_sigma == pytest.approx(0.35)
    assert env_cfg.reward_config.tracking_lin_vel_forward_sigma == pytest.approx(0.16)
    assert env_cfg.reward_config.tracking_lin_vel_lateral_sigma == pytest.approx(0.25)
    assert env_cfg.reward_config.swing_contact_forward_scale == pytest.approx(1.5)


def test_tracking_lin_vel_reward_applies_directional_tracking_sigmas():
    env = registry.make("go2-walk-flat", num_envs=3, mode="train")
    env.init_state()
    env._commands.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]))
    linvel = torch.zeros((env.num_envs, 3), dtype=torch.float32)

    reward = env._reward_tracking_lin_vel(linvel, None, None)

    np.testing.assert_allclose(
        reward,
        np.array(
            [
                np.exp(-1.0 / 0.16),
                np.exp(-1.0 / 0.35),
                np.exp(-0.25 / 0.25),
            ],
            dtype=np.float32,
        ),
    )


def test_go2_hip_position_reward_only_penalizes_excess_abduction():
    env = registry.make("go2-walk-flat", num_envs=1, mode="train")
    env.init_state()

    assert env._hip_indices.numel() == 4
    dof_pos = env.default_angles.reshape(1, -1).clone()
    within_tolerance = env._reward_hip_position(None, None, dof_pos)
    dof_pos[0, env._hip_indices[0]] += env.cfg.reward_config.hip_position_tolerance
    at_tolerance = env._reward_hip_position(None, None, dof_pos)
    dof_pos[0, env._hip_indices[0]] += env.cfg.reward_config.hip_position_sigma
    excessive = env._reward_hip_position(None, None, dof_pos)

    np.testing.assert_allclose(within_tolerance, 0.0)
    np.testing.assert_allclose(at_tolerance, 0.0)
    assert excessive[0] > 0.0


def test_go2_dof_acc_reward_uses_worst_joint_acceleration(quadruped_env):
    env = quadruped_env
    env._last_joint_vel[:] = 0.0
    joint_vel = torch.zeros_like(env._last_joint_vel)
    joint_vel[0, 0] = env.cfg.ctrl_dt
    joint_vel[0, 1] = 2.0 * env.cfg.ctrl_dt
    env.get_dof_vel = lambda: joint_vel

    penalty = env._reward_dof_acc(None, None, None)

    np.testing.assert_allclose(penalty, [4.0])


def test_go2_velocity_commands_remain_constant_before_resampling_interval():
    torch.manual_seed(7)
    env = registry.make("go2-walk-flat", num_envs=64, mode="train")
    env.init_state()
    commands = env._commands.clone()
    velocity_cfg = env.cfg.commands.velocity

    assert commands.shape == (64, 3)
    assert np.unique(commands, axis=0).shape[0] > 1
    assert _all(commands.numpy() >= velocity_cfg.lower)
    assert _all(commands.numpy() <= velocity_cfg.upper)

    env.step(torch.zeros((env.num_envs, *env.action_space.shape), dtype=torch.float32))

    np.testing.assert_array_equal(env._commands, commands)


def test_go2_hard_case_curriculum_interpolates_by_training_iteration():
    env = registry.make("go2-walk-flat", num_envs=1, mode="train")
    binding = env._velocity_command_binding
    assert binding.hard_case_probability == pytest.approx(0.0)
    env._set_hard_case_probability(600)
    assert binding.hard_case_probability == pytest.approx(0.05)
    env._set_hard_case_probability(1000)
    assert binding.hard_case_probability == pytest.approx(0.15)
    env._set_hard_case_probability(1500)
    assert binding.hard_case_probability == pytest.approx(0.20)


def test_go2_play_disables_hard_case_curriculum():
    env = registry.make("go2-walk-flat", num_envs=1, mode="play")
    assert env.cfg.commands.velocity.hard_case_curriculum == ()
    assert env._velocity_command_binding.hard_case_probability == pytest.approx(0.0)


def test_go2_randomization_is_enabled_for_training_and_disabled_for_play():
    flat_cfg = registry.make_env_config("go2-walk-flat")
    rough_cfg = registry.make_env_config("go2-walk-rough")
    flat_play = registry.make("go2-walk-flat", num_envs=2, mode="play")
    rough_play = registry.make("go2-walk-rough", num_envs=2, mode="play")

    assert flat_cfg.randomization.enabled
    assert rough_cfg.randomization.enabled
    assert flat_cfg.commands.velocity.resampling_seconds_range is not None
    assert rough_cfg.commands.velocity.resampling_seconds_range is None
    for play in (flat_play, rough_play):
        assert not play.cfg.randomization.enabled
        assert play.cfg.commands.velocity.resampling_seconds_range is None
        assert play.cfg.noise_config.level == 0.0


@pytest.mark.parametrize("env_name", ["go2-walk-flat", "go2-walk-rough"])
def test_go2_uses_position_actuators_for_pd_randomization(env_name: str):
    env = registry.make(env_name, num_envs=1, mode="train")

    assert all(spec.actuator_type is ActuatorType.POSITION for spec in env.model.actuators)


@pytest.mark.parametrize("env_name", ["go2-walk-flat", "go2-walk-rough"])
def test_go2_reset_randomization_stays_within_configured_ranges(env_name: str):
    env = registry.make(env_name, num_envs=32, mode="train")
    env.init_state()
    randomization = env.cfg.randomization
    joint_pos_diff = env.get_dof_pos().numpy() - env.default_angles.numpy()
    dof_vel = env.sim_data["dof_vel"].numpy()

    assert _all(np.abs(joint_pos_diff) <= randomization.joint_pos_noise + 1e-6)
    assert _any(np.abs(joint_pos_diff) > 0.0)
    assert _all(np.abs(dof_vel[:, -env._num_action :]) <= randomization.joint_vel_noise + 1e-6)
    assert _any(np.abs(dof_vel[:, -env._num_action :]) > 0.0)
    assert _all(np.abs(dof_vel[:, :3]) <= np.asarray(randomization.base_lin_vel_noise, dtype=np.float32) + 1e-6)
    assert _all(np.abs(dof_vel[:, 3:6]) <= np.asarray(randomization.base_ang_vel_noise, dtype=np.float32) + 1e-6)
    assert np.unique(joint_pos_diff, axis=0).shape[0] > 1
    command_interval = env.cfg.commands.velocity.resampling_seconds_range
    if command_interval is not None:
        assert _all(env._command_resampling_time >= command_interval[0])
        assert _all(env._command_resampling_time <= command_interval[1])


def test_go2_pd_and_friction_randomization_is_per_env_and_episode_constant():
    env = registry.make("go2-walk-flat", num_envs=16, mode="train")
    env.init_state()
    randomization = env.cfg.randomization
    _, kp, damping, friction = _read_param_overrides(env)
    nominal_kp, nominal_damping = _nominal_params(env)
    kp_scale = kp / nominal_kp
    damping_scale = damping / nominal_damping

    assert _all(kp_scale >= randomization.kp_scale_range[0])
    assert _all(kp_scale <= randomization.kp_scale_range[1])
    assert _all(damping_scale >= randomization.damping_scale_range[0])
    assert _all(damping_scale <= randomization.damping_scale_range[1])
    assert np.unique(kp, axis=0).shape[0] > 1
    assert np.unique(damping, axis=0).shape[0] > 1
    assert _all(friction[:, 0] >= randomization.sliding_friction_range[0])
    assert _all(friction[:, 0] <= randomization.sliding_friction_range[1])
    np.testing.assert_array_equal(friction[:, 1:], np.broadcast_to(friction[:1, 1:], friction[:, 1:].shape))

    env.step(np.zeros((env.num_envs, env._num_action), dtype=np.float32))

    _, kp_after, damping_after, friction_after = _read_param_overrides(env)
    np.testing.assert_array_equal(kp_after, kp)
    np.testing.assert_array_equal(damping_after, damping)
    np.testing.assert_array_equal(friction_after, friction)


def test_go2_base_mass_and_com_randomization_is_per_env_and_episode_constant():
    env = registry.make("go2-walk-flat", num_envs=16, mode="train")
    env.init_state()
    randomization = env.cfg.randomization

    mass, center_of_mass = _read_mass_overrides(env)
    nominal_mass, nominal_center_of_mass = _nominal_masses(env)
    mass_scale = mass / nominal_mass
    center_of_mass_offset = center_of_mass - nominal_center_of_mass
    com_noise = np.asarray(randomization.base_com_offset_noise, dtype=np.float32)

    assert _all(mass_scale >= randomization.base_mass_scale_range[0])
    assert _all(mass_scale <= randomization.base_mass_scale_range[1])
    assert _all(np.abs(center_of_mass_offset) <= com_noise + 1e-7)
    assert np.unique(mass_scale).size > 1
    assert np.unique(center_of_mass_offset, axis=0).shape[0] > 1

    env.step(np.zeros((env.num_envs, env._num_action), dtype=np.float32))

    mass_after, com_after = _read_mass_overrides(env)
    np.testing.assert_array_equal(mass_after, mass)
    np.testing.assert_array_equal(com_after, center_of_mass)


def test_go2_action_delay_selects_current_or_previous_action_per_env():
    env = registry.make("go2-walk-flat", num_envs=2, mode="train")
    state = env.init_state()
    previous = np.full((2, env._num_action), -0.2, dtype=np.float32)
    current = np.full((2, env._num_action), 0.3, dtype=np.float32)
    env._current_actions.copy_(torch.as_tensor(previous))
    env._action_delay_steps.copy_(torch.tensor((0, 1)))

    env.apply_action(torch.as_tensor(current), state)
    env.sim_data.execute()

    expected_actions = np.stack([current[0], previous[1]])
    expected_ctrls = expected_actions * env.cfg.control_config.action_scale + env.default_angles.numpy()
    np.testing.assert_allclose(env.sim_data["actuator_ctrls"], expected_ctrls)


def test_go2_command_resampling_only_updates_due_environments(monkeypatch: pytest.MonkeyPatch):
    env = registry.make("go2-walk-flat", num_envs=3, mode="train")
    env.init_state()
    commands = np.array([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.0, -0.1]], dtype=np.float32)
    replacement = np.array([[0.4, -0.2, 0.3], [0.5, 0.2, -0.3]], dtype=np.float32)
    env._commands.copy_(torch.as_tensor(commands))
    env._command_resampling_time.copy_(torch.tensor((0.0, 1.0, 0.0)))
    monkeypatch.setattr(env, "resample_commands", lambda num_envs: torch.as_tensor(replacement[:num_envs]))

    env._update_commands()

    np.testing.assert_array_equal(env._commands[[0, 2]], replacement)
    np.testing.assert_array_equal(env._commands[[1]], commands[[1]])
    assert env._command_resampling_time[1] == pytest.approx(1.0 - env.cfg.ctrl_dt)
    assert _all(env._command_resampling_time[[0, 2]] > 0.0)


def test_random_planar_velocity_binding_is_seeded_vectorized_and_task_specific():
    first = RandomPlanarVelocityBinding(
        torch.tensor([-0.5, -0.4, -1.0]),
        torch.tensor([1.0, 0.4, 1.0]),
        rng=torch.Generator().manual_seed(7),
        standing_probability=0.0,
    ).read_command(batch_size=8)
    repeated = RandomPlanarVelocityBinding(
        torch.tensor([-0.5, -0.4, -1.0]),
        torch.tensor([1.0, 0.4, 1.0]),
        rng=torch.Generator().manual_seed(7),
        standing_probability=0.0,
    ).read_command(batch_size=8)

    assert first.values.shape == (8, 3)
    assert np.unique(first.values, axis=0).shape[0] > 1
    np.testing.assert_array_equal(first.values, repeated.values)
    assert _all(first.values.numpy() >= [-0.5, -0.4, -1.0])
    assert _all(first.values.numpy() <= [1.0, 0.4, 1.0])


def test_random_planar_velocity_binding_can_sample_a_standing_batch():
    binding = RandomPlanarVelocityBinding(
        torch.full((3,), -1.0),
        torch.full((3,), 1.0),
        rng=torch.Generator().manual_seed(3),
        standing_probability=1.0,
    )

    np.testing.assert_array_equal(binding.read_command(batch_size=5).values, np.zeros((5, 3), dtype=np.float32))


def test_go2_partial_reset_only_resamples_finished_environments(monkeypatch: pytest.MonkeyPatch):
    env = registry.make("go2-walk-flat", num_envs=3, mode="train")
    state = env.init_state()
    commands = np.array(
        [
            [0.1, 0.0, 0.0],
            [0.2, 0.1, 0.0],
            [0.3, 0.0, -0.1],
        ],
        dtype=np.float32,
    )
    env._commands.copy_(torch.as_tensor(commands))
    action_delay_steps = env._action_delay_steps.clone()
    _, kp, damping, friction = _read_param_overrides(env)
    mass, center_of_mass = _read_mass_overrides(env)
    state.terminated.copy_(torch.tensor((False, True, False)))
    replacement = np.array([[0.4, -0.2, 0.3]], dtype=np.float32)

    def resample_commands(num_envs):
        return torch.as_tensor(np.broadcast_to(replacement, (num_envs, 3)))

    monkeypatch.setattr(env, "resample_commands", resample_commands)

    env._reset_done_envs()

    np.testing.assert_array_equal(env._commands[[0, 2]], commands[[0, 2]])
    np.testing.assert_array_equal(env._commands[[1]], replacement)
    np.testing.assert_array_equal(env._action_delay_steps[[0, 2]], action_delay_steps[[0, 2]])
    _, kp_after, damping_after, friction_after = _read_param_overrides(env)
    mass_after, com_after = _read_mass_overrides(env)
    np.testing.assert_array_equal(kp_after[[0, 2]], kp[[0, 2]])
    np.testing.assert_array_equal(damping_after[[0, 2]], damping[[0, 2]])
    np.testing.assert_array_equal(friction_after[[0, 2]], friction[[0, 2]])
    np.testing.assert_array_equal(mass_after[[0, 2]], mass[[0, 2]])
    np.testing.assert_array_equal(com_after[[0, 2]], center_of_mass[[0, 2]])


def test_go2_play_reset_preserves_nominal_joint_state_and_runtime_parameters():
    env = registry.make("go2-walk-flat", num_envs=4, mode="play")
    env.init_state()

    np.testing.assert_array_equal(
        env.get_dof_pos(),
        np.broadcast_to(env.default_angles, (4, env._num_action)),
    )
    np.testing.assert_array_equal(env.sim_data["dof_vel"], np.zeros_like(env.sim_data["dof_vel"]))
    assert {
        "action_delay_steps",
        "kp_scale",
        "damping_scale",
        "sliding_friction",
        "base_mass_scale",
        "base_com_offset",
    }.isdisjoint(env.state.reward_terms)
    mass, center_of_mass = _read_mass_overrides(env)
    nominal_mass, nominal_center_of_mass = _nominal_masses(env)
    np.testing.assert_array_equal(mass, nominal_mass)
    np.testing.assert_array_equal(center_of_mass, nominal_center_of_mass)


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (lambda cfg: setattr(cfg.randomization, "joint_pos_noise", -0.1), "joint_pos_noise"),
        (lambda cfg: setattr(cfg.randomization, "action_delay_steps", (0, 2)), "action_delay_steps"),
        (lambda cfg: setattr(cfg.randomization, "base_mass_scale_range", (1.1, 0.9)), "base_mass_scale_range"),
        (
            lambda cfg: setattr(cfg.randomization, "base_com_offset_noise", (0.01, -0.01, 0.005)),
            "base_com_offset_noise",
        ),
        (
            lambda cfg: setattr(cfg.commands.velocity, "resampling_seconds_range", (10.0, 5.0)),
            "resampling_seconds_range",
        ),
        (
            lambda cfg: setattr(cfg.control_config, "simulate_action_latency", True),
            "random action delay",
        ),
    ],
)
def test_quadruped_randomization_config_rejects_invalid_ranges(update, message):
    cfg = registry.make_env_config("go2-walk-flat")
    update(cfg)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_gait_schedule_and_stand_transition_are_continuous():
    env = registry.make("go2-walk-flat", num_envs=2, mode="train")
    env.init_state()
    cfg = env.cfg
    env._commands.copy_(torch.tensor([0.8, 0.0, 0.0]))
    env._gait_schedule_alpha[:] = 0.0

    env._advance_phase()

    max_alpha_step = cfg.ctrl_dt / cfg.gait_schedule_smoothing_time_s
    assert env._gait_schedule_alpha[0] == pytest.approx(max_alpha_step)
    assert env._gait_frequency[0] < cfg.gait_schedule_frequency_high
    assert env._swing_fraction[0] < cfg.gait_schedule_swing_fraction_high

    env._commands.zero_()
    env._phase[:] = 0.5
    env._feet_phase[:] = 0.5
    env._advance_phase()

    # Stand and walk share phase zero.  Advancing a hidden clock while the
    # observed phase is masked would make startup jump to an arbitrary swing.
    np.testing.assert_array_equal(env._feet_phase, np.zeros((2, env._num_feet), dtype=np.float32))
    np.testing.assert_array_equal(env._phase, np.zeros(2, dtype=np.float32))

    env._commands.copy_(torch.tensor([0.0, 0.0, 1.0]))
    env._advance_phase()
    assert env._gait_frequency[0] > cfg.gait_schedule_frequency_low

    env._commands.copy_(torch.tensor([0.0, 0.5, 0.0]))
    env._advance_phase()
    assert env._gait_frequency[0] > cfg.gait_schedule_frequency_low


@pytest.mark.parametrize(
    ("flat_env", "rough_env"),
    [
        ("anymalc-walk-flat", "anymalc-walk-rough"),
        ("go1-walk-flat", "go1-walk-rough"),
        ("go2-walk-flat", "go2-walk-rough"),
    ],
)
def test_quadruped_rough_walk_uses_procedural_hfield_and_shared_runtime_config(flat_env, rough_env):
    flat_cfg = registry.make_env_config(flat_env)
    rough_cfg = registry.make_env_config(rough_env)

    assert isinstance(rough_cfg.scene.assets.terrain, ProceduralHFieldAssetCfg)
    assert isinstance(rough_cfg.scene.objs.floor, HFieldTerrainCfg)
    assert rough_cfg.control_config == flat_cfg.control_config
    assert rough_cfg.spawn_xy_range == flat_cfg.spawn_xy_range > 0.0
    assert rough_cfg.render_spacing == flat_cfg.render_spacing == 0.0


@pytest.mark.parametrize("env_name", ["anymalc-walk-rough", "go1-walk-rough", "go2-walk-rough"])
def test_quadruped_rough_walk_reset_and_base_height_are_terrain_relative(env_name):
    env = registry.make(env_name, num_envs=8, mode="train")
    env.init_state()
    base_pos = env.sim_data["base_pos"]
    env_ids = torch.arange(env.num_envs)
    ground_height = env.sim.sample_terrain_height(env.cfg.ground_geom_name, env_ids, base_pos[:, None, :2])[:, 0]
    relative_height = (base_pos[:, 2] - ground_height).numpy()
    height_scale = env.cfg.scene.assets.terrain.generator.height_scale

    assert ground_height.shape == (env.num_envs,)
    assert _all(np.abs(base_pos[:, :2]) <= env.cfg.spawn_xy_range)
    assert _all(relative_height >= env.cfg.initial_base_position[2] - 1e-6)
    assert _all(relative_height <= env.cfg.initial_base_position[2] + height_scale + 1e-6)
    expected = np.square(relative_height - env.cfg.reward_config.base_height_target)
    np.testing.assert_allclose(env._reward_base_height(None, None, None), expected)

    next_state = env.step(torch.zeros((env.num_envs, *env.action_space.shape), dtype=torch.float32))
    assert next_state.reward.shape == (env.num_envs,)
    assert bool(torch.isfinite(next_state.reward).all())


def test_foot_position_sensor_names_belong_to_walk_task_config():
    go1 = registry.make_env_config("go1-walk-flat")
    anymalc = registry.make_env_config("anymalc-walk-flat")

    assert go1.sensor.foot_positions == ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
    assert anymalc.sensor.foot_positions == go1.sensor.foot_positions


@pytest.mark.parametrize("robot_name", ["go1", "go2", "anymalc"])
def test_walk_task_gyro_config_compiles_for_each_robot(robot_name):
    cfg = registry.make_env_config(f"{robot_name}-walk-flat")
    rough_cfg = registry.make_env_config(f"{robot_name}-walk-rough")
    # Terrain variants use the same model-local gyro wiring; compile each robot only once.
    assert _sim_data_queries(rough_cfg)["gyro"] == _sim_data_queries(cfg)["gyro"]
    robot = cfg.scene.objs.robot
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    reads = backend.compile_reads({"gyro": _sim_data_queries(cfg)["gyro"]})
    reset = backend.write_compiler.compile(
        {"angular_velocity": BodyAngularVelocityWrite((robot.resolved_base_link_name,))}, reset=True
    )
    velocity = np.asarray([0.4, -0.5, 0.6], dtype=np.float32)
    reset.buffer("angular_velocity")[0, 0] = torch.tensor(velocity)
    reset.execute(torch.tensor([0]))
    reads.execute()
    np.testing.assert_allclose(reads["gyro"][0], velocity, atol=1e-6)


def test_walk_task_gyro_config_is_instance_local():
    first = registry.make_env_config("anymalc-walk-flat")
    second = registry.make_env_config("anymalc-walk-flat")
    unitree = registry.make_env_config("go2-walk-flat")
    second_sensor = second.sensor.gyro
    unitree_sensor = unitree.sensor.gyro
    first.sensor.gyro = "custom_gyro"
    assert second.sensor.gyro == second_sensor
    assert unitree.sensor.gyro == unitree_sensor
    assert _sim_data_queries(first)["gyro"].sensors == (first.scene.objs.robot.resolve_name("custom_gyro"),)


@pytest.mark.parametrize(("prefix", "suffix"), [("robot_", None), (None, "_instance"), ("robot_", "_instance")])
def test_walk_gyro_query_reads_renamed_model_sensor_with_scene_affixes(tmp_path, prefix, suffix):
    # A real imported gyro with a custom name catches accidental hard-coded sensor lookup.
    model_path = tmp_path / "robot.xml"
    model_path.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/>'
        '<geom type="sphere" size="0.1" mass="1"/><site name="imu"/></body></worldbody>'
        '<sensor><gyro name="custom_gyro" site="imu"/></sensor></mujoco>'
    )
    robot = RobotCfg(
        model=MjcfFileCfg(file=model_path), base_link_name="base", prefix=prefix, suffix=suffix, key_pose=KeyPoseCfg()
    )

    @configclass
    class RobotObjsCfg(SceneObjsCfg):
        robot: RobotCfg

    cfg = registry.make_env_config("go2-walk-flat")
    cfg.scene = SceneCfg(objs=RobotObjsCfg(robot=robot))
    cfg.sensor.gyro = "custom_gyro"
    queries = _sim_data_queries(cfg)
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    reads = backend.compile_reads({"gyro": queries["gyro"]})
    reset = backend.write_compiler.compile(
        {"angular_velocity": BodyAngularVelocityWrite((robot.resolved_base_link_name,))}, reset=True
    )
    velocity = np.asarray([[0.4, -0.5, 0.6], [-0.7, 0.8, -0.9]], dtype=np.float32)
    reset.buffer("angular_velocity")[:, 0] = torch.as_tensor(velocity)
    reset.execute(torch.arange(2))
    reads.execute()
    np.testing.assert_allclose(reads["gyro"], velocity, atol=1e-6)


def test_walk_task_selects_named_robot_key_pose():
    cfg = registry.make_env_config("go1-walk-flat")
    joint_count = len(cfg.scene.objs.robot.key_pose.joint_names)
    cfg.scene.objs.robot.key_pose.poses["zero"] = [0.0] * joint_count
    cfg.key_pose_name = "zero"

    env = QuadrupedWalkTask(cfg, num_envs=1)

    np.testing.assert_array_equal(env.default_angles, np.zeros((joint_count,), dtype=np.float32))


def test_swing_height_reward_requires_feet_to_leave_ground(quadruped_env):
    reward_cfg = quadruped_env.cfg.reward_config
    quadruped_env.feet_pos[:, :, 2] = (
        reward_cfg.target_foot_height - reward_cfg.base_height_target - reward_cfg.foot_height_reference_offset
    )

    reward = _swing_reward(
        quadruped_env,
        contacts=np.ones((1, quadruped_env._num_feet), dtype=bool),
    )

    np.testing.assert_array_equal(reward, np.zeros((1,), dtype=np.float32))


def test_swing_height_reward_prefers_target_height(quadruped_env):
    reward_cfg = quadruped_env.cfg.reward_config
    target_z = reward_cfg.target_foot_height - reward_cfg.base_height_target - reward_cfg.foot_height_reference_offset
    no_contacts = np.zeros((1, quadruped_env._num_feet), dtype=bool)

    quadruped_env.feet_pos[:, :, 2] = target_z
    target_reward = _swing_reward(quadruped_env, no_contacts)

    quadruped_env.feet_pos[:, :, 2] = target_z - 2.0 * reward_cfg.swing_feet_height_sigma
    off_target_reward = _swing_reward(quadruped_env, no_contacts)

    np.testing.assert_allclose(target_reward, np.ones((1,), dtype=np.float32))
    assert _all(target_reward > off_target_reward)
    assert _all(off_target_reward > 0.0)


def test_go2_swing_height_reward_uses_lift_plateau_and_slow_place_profile():
    env = registry.make("go2-walk-flat", num_envs=1, mode="train")
    env.init_state()
    env.cfg.reward_config.swing_feet_z_sinusoidal = True
    env._swing_fraction[:] = env.cfg.reward_config.swing_fraction
    no_contacts = np.zeros((1, env._num_feet), dtype=bool)
    ground_z = -env.cfg.reward_config.base_height_target - env.cfg.reward_config.foot_height_reference_offset
    peak_z = ground_z + env.cfg.reward_config.target_foot_height
    rewards = {}

    rise_fraction = env.cfg.reward_config.swing_rise_fraction
    plateau_fraction = env.cfg.reward_config.swing_plateau_fraction
    swing_start = 1.0 - env.cfg.reward_config.swing_fraction
    swing_duration = env.cfg.reward_config.swing_fraction
    plateau_phase = swing_start + swing_duration * (rise_fraction + 0.5 * plateau_fraction)
    for phase, foot_z in ((swing_start, ground_z), (plateau_phase, peak_z), (1.0, ground_z)):
        env._feet_phase[:] = phase
        env.feet_contact.copy_(torch.as_tensor(no_contacts))
        env.feet_pos[:, :, 2] = foot_z
        rewards[phase] = env._reward_swing_feet_z(None, None, None)[0]

    assert env.cfg.reward_config.swing_feet_z_sinusoidal is True
    np.testing.assert_allclose(rewards[swing_start], 1.0)
    np.testing.assert_allclose(rewards[plateau_phase], 1.0)
    np.testing.assert_allclose(rewards[1.0], 1.0)

    env.feet_pos[:, :, 2] = peak_z
    end_off_profile = env._reward_swing_feet_z(None, None, None)[0]
    assert end_off_profile < rewards[1.0]


def test_go2_touchdown_velocity_penalty_only_applies_to_low_descending_feet():
    env = registry.make("go2-walk-flat", num_envs=1, mode="train")
    env.init_state()
    reward_cfg = env.cfg.reward_config
    sensor_ground_z = -reward_cfg.base_height_target - reward_cfg.foot_height_reference_offset

    env._last_feet_pos[:, :, 2] = sensor_ground_z
    env.feet_pos[:, :, 2] = sensor_ground_z - 0.01
    low_fast_touchdown = env._reward_touchdown_velocity(None, None, None)

    env.feet_pos[:, :, 2] = sensor_ground_z - 0.001
    low_slow_touchdown = env._reward_touchdown_velocity(None, None, None)

    env._last_feet_pos[:, :, 2] = sensor_ground_z + reward_cfg.target_foot_height
    env.feet_pos[:, :, 2] = sensor_ground_z + reward_cfg.target_foot_height - 0.01
    high_start_touchdown = env._reward_touchdown_velocity(None, None, None)

    env._last_feet_pos[:, :, 2] = sensor_ground_z
    env.feet_pos[:, :, 2] = sensor_ground_z + 0.01
    low_liftoff = env._reward_touchdown_velocity(None, None, None)

    assert low_slow_touchdown[0] > 0.0
    assert low_fast_touchdown[0] > low_slow_touchdown[0]
    np.testing.assert_allclose(high_start_touchdown, np.zeros((1,), dtype=np.float32))
    np.testing.assert_allclose(low_liftoff, np.zeros((1,), dtype=np.float32))


def test_swing_contact_penalty_detects_dragging_feet(quadruped_env):
    feet_phase = np.full((1, quadruped_env._num_feet), 0.75, dtype=np.float32)
    no_contacts = np.zeros((1, quadruped_env._num_feet), dtype=bool)
    all_contacts = np.ones((1, quadruped_env._num_feet), dtype=bool)

    quadruped_env._feet_phase.copy_(torch.as_tensor(feet_phase))
    quadruped_env._last_feet_contact.copy_(torch.as_tensor(no_contacts))
    quadruped_env.feet_contact.copy_(torch.as_tensor(no_contacts))
    no_drag = quadruped_env._reward_swing_contact(None, None, None)
    quadruped_env.feet_contact.copy_(torch.as_tensor(all_contacts))
    all_drag = quadruped_env._reward_swing_contact(None, None, None)

    np.testing.assert_array_equal(no_drag, np.zeros((1,), dtype=np.float32))
    progress = (
        0.75 - (1.0 - quadruped_env.cfg.reward_config.swing_fraction)
    ) / quadruped_env.cfg.reward_config.swing_fraction
    expected = 2.0 * np.square(np.sin(np.pi * progress))
    np.testing.assert_allclose(all_drag, np.asarray([expected], dtype=np.float32))

    quadruped_env._feet_phase[:] = 1.0 - quadruped_env.cfg.reward_config.swing_fraction
    np.testing.assert_allclose(
        quadruped_env._reward_swing_contact(None, None, None),
        np.zeros((1,), dtype=np.float32),
    )

    # Consecutive stance contact is penalized by horizontal foot motion, while
    # a first contact tick is excluded as a touchdown transition.
    quadruped_env._commands.copy_(torch.tensor([[0.5, 0.0, 0.0]]))
    quadruped_env._feet_phase[:] = 0.0
    quadruped_env._last_feet_contact.copy_(torch.as_tensor(all_contacts))
    quadruped_env.feet_contact.copy_(torch.as_tensor(all_contacts))
    quadruped_env._last_feet_world_pos.copy_(quadruped_env.feet_world_pos)
    quadruped_env.feet_world_pos[:, :, 0] += 0.001
    stance_motion = quadruped_env._reward_swing_contact(None, None, None)
    np.testing.assert_allclose(stance_motion, np.asarray([0.3], dtype=np.float32), rtol=1e-5)

    quadruped_env._last_feet_contact.copy_(torch.as_tensor(no_contacts))
    np.testing.assert_allclose(
        quadruped_env._reward_swing_contact(None, None, None),
        np.zeros((1,), dtype=np.float32),
    )


def test_swing_contact_penalty_scales_with_forward_command(quadruped_env):
    quadruped_env.cfg.reward_config.swing_contact_forward_scale = 1.5
    quadruped_env._feet_phase[:] = 0.75
    quadruped_env.feet_contact.fill_(True)
    quadruped_env._last_feet_contact.fill_(False)

    commands = (
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, -1.0),
    )
    penalties = []
    for command in commands:
        quadruped_env._commands.copy_(torch.tensor([command]))
        penalties.append(quadruped_env._reward_swing_contact(None, None, None)[0])

    reverse, half_forward, full_forward, lateral, yaw = penalties
    nominal = full_forward / 1.5
    np.testing.assert_allclose(reverse, nominal)
    np.testing.assert_allclose(half_forward, 1.25 * nominal)
    np.testing.assert_allclose(lateral, nominal)
    np.testing.assert_allclose(yaw, nominal)
    assert full_forward > half_forward > nominal


def test_static_stance_anchors_contact_points_without_affecting_motion(quadruped_env):
    all_contacts = np.ones((1, quadruped_env._num_feet), dtype=bool)
    quadruped_env._commands.zero_()
    quadruped_env._feet_phase[:] = 0.0
    quadruped_env._last_feet_contact.copy_(torch.as_tensor(all_contacts))
    quadruped_env.feet_contact.copy_(torch.as_tensor(all_contacts))
    quadruped_env._static_anchor_valid.fill_(False)
    quadruped_env._last_feet_world_pos.copy_(quadruped_env.feet_world_pos)

    first = quadruped_env._reward_swing_contact(None, None, None)
    np.testing.assert_allclose(first, np.zeros((1,), dtype=np.float32))
    assert _all(quadruped_env._static_anchor_valid)

    # Move all feet by 6 mm without a one-step velocity contribution.  The
    # static anchor still sees the accumulated position error and penalizes it.
    quadruped_env.feet_world_pos[:, :, 0] += 0.006
    quadruped_env._last_feet_world_pos.copy_(quadruped_env.feet_world_pos)
    anchored = quadruped_env._reward_swing_contact(None, None, None)
    np.testing.assert_allclose(anchored, np.asarray([0.8], dtype=np.float32), rtol=1e-5)

    # The bounded barrier keeps increasing for large accumulated drift rather
    # than becoming exactly flat after a finite clipping threshold.
    quadruped_env.feet_world_pos[:, :, 0] += 0.054
    quadruped_env._last_feet_world_pos.copy_(quadruped_env.feet_world_pos)
    larger_drift = quadruped_env._reward_swing_contact(None, None, None)
    assert larger_drift[0] > anchored[0]
    assert larger_drift[0] < 4.0

    # A moving command releases the anchor and leaves the walking penalty path
    # active, so the position latch cannot constrain normal gait motion.
    quadruped_env._commands.copy_(torch.tensor([[0.5, 0.0, 0.0]]))
    quadruped_env._reward_swing_contact(None, None, None)
    assert not _any(quadruped_env._static_anchor_valid)


def test_feet_spacing_reward_is_symmetric_for_splay_and_pigeon_toes(quadruped_env):
    quadruped_env._default_feet_width = torch.tensor([0.3, 0.3])
    quadruped_env.feet_pos[:, :, :] = 0.0
    quadruped_env.feet_pos[:, 0, 1] = 0.15
    quadruped_env.feet_pos[:, 1, 1] = -0.15
    quadruped_env.feet_pos[:, 2, 1] = 0.15
    quadruped_env.feet_pos[:, 3, 1] = -0.15
    nominal = quadruped_env._reward_feet_spacing(None, None, None)

    quadruped_env.feet_pos[:, :, :] = 0.0
    quadruped_env.feet_pos[:, 0, 1] = 0.20
    quadruped_env.feet_pos[:, 1, 1] = -0.20
    quadruped_env.feet_pos[:, 2, 1] = 0.20
    quadruped_env.feet_pos[:, 3, 1] = -0.20
    splay = quadruped_env._reward_feet_spacing(None, None, None)

    quadruped_env.feet_pos[:, :, :] = 0.0
    quadruped_env.feet_pos[:, 0, 1] = 0.10
    quadruped_env.feet_pos[:, 1, 1] = -0.10
    quadruped_env.feet_pos[:, 2, 1] = 0.10
    quadruped_env.feet_pos[:, 3, 1] = -0.10
    pigeon = quadruped_env._reward_feet_spacing(None, None, None)

    np.testing.assert_allclose(nominal, np.ones((1,), dtype=np.float32))
    np.testing.assert_allclose(splay, pigeon, rtol=0.0, atol=1e-8)
    assert _all(splay < nominal)


def test_feet_spacing_reward_covers_front_and_rear_pairs(quadruped_env):
    quadruped_env._default_feet_width = torch.tensor([0.3, 0.3])
    quadruped_env.feet_pos[:, :, :] = 0.0
    quadruped_env.feet_pos[:, 0, 1] = 0.20
    quadruped_env.feet_pos[:, 1, 1] = -0.20
    quadruped_env.feet_pos[:, 2, 1] = 0.15
    quadruped_env.feet_pos[:, 3, 1] = -0.15
    front_bad = quadruped_env._reward_feet_spacing(None, None, None)

    quadruped_env.feet_pos[:, 0, 1] = 0.15
    quadruped_env.feet_pos[:, 1, 1] = -0.15
    quadruped_env.feet_pos[:, 2, 1] = 0.20
    quadruped_env.feet_pos[:, 3, 1] = -0.20
    rear_bad = quadruped_env._reward_feet_spacing(None, None, None)

    np.testing.assert_allclose(front_bad, rear_bad)
    assert _all(front_bad < 1.0)


def test_feet_spacing_reward_does_not_cancel_front_and_rear_errors(quadruped_env):
    quadruped_env._default_feet_width = torch.tensor([0.3, 0.3])
    quadruped_env.feet_pos[:, :, :] = 0.0
    quadruped_env.feet_pos[:, 0, 1] = 0.20
    quadruped_env.feet_pos[:, 1, 1] = -0.20
    quadruped_env.feet_pos[:, 2, 1] = 0.15
    quadruped_env.feet_pos[:, 3, 1] = -0.15
    one_pair_bad = quadruped_env._reward_feet_spacing(None, None, None)

    # The signed front/rear errors have opposite directions, but both pairs are
    # outside tolerance. Their penalties must not cancel before shaping.
    quadruped_env.feet_pos[:, 0, 1] = 0.20
    quadruped_env.feet_pos[:, 1, 1] = -0.20
    quadruped_env.feet_pos[:, 2, 1] = 0.10
    quadruped_env.feet_pos[:, 3, 1] = -0.10
    opposite_signed_bad = quadruped_env._reward_feet_spacing(None, None, None)

    assert _all(one_pair_bad < 1.0)
    assert _all(opposite_signed_bad < one_pair_bad)


def test_posture_constraints_are_full_for_sagittal_and_soft_for_lateral_yaw():
    env = registry.make("go2-walk-flat", num_envs=6, mode="train")
    env.init_state()
    env._commands.copy_(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
                [0.0, 0.3, 0.0],
                [0.0, 0.0, 0.3],
                [0.0, 0.8, 0.8],
            ]
        )
    )
    scales = env._posture_constraint_scale()

    np.testing.assert_allclose(scales[:3], 1.0)
    np.testing.assert_allclose(scales[3:5], 0.70, rtol=1e-6)
    np.testing.assert_allclose(scales[5], env._POSTURE_GATE_FLOOR)


@pytest.mark.parametrize("robot_name", ["go1", "go2", "anymalc"])
def test_walk_state_sensor_queries_resolve_model_and_scene_names(robot_name):
    cfg = registry.make_env_config(f"{robot_name}-walk-flat")
    robot = cfg.scene.objs.robot
    robot.prefix = "robot_"
    robot.suffix = "_instance"
    # Scene sensor targets are already assembled names; update their references
    # when changing instance affixes after scene construction.
    for _, sensor in cfg.scene.iter_sensors():
        if hasattr(sensor, "geom2"):
            sensor.geom2 = robot.resolve_name(sensor.geom2)
        else:
            sensor.object_name = robot.resolve_name(sensor.object_name)
            if sensor.ref_object_name is not None:
                sensor.ref_object_name = robot.resolve_name(sensor.ref_object_name)
    queries = _sim_data_queries(cfg)
    names = ("gyro", "local_linvel", "upvector", "FL_pos", "FR_pos", "RL_pos", "RR_pos")
    scene_names = {name for name, _ in cfg.scene.iter_sensors()}
    for name in names:
        sensor_name = cfg.sensor.gyro if name == "gyro" else name
        expected = sensor_name if sensor_name in scene_names else robot.resolve_name(sensor_name)
        assert queries[name].sensors == (expected,)
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    reads = backend.compile_reads({name: queries[name] for name in names})
    reads.execute()
    for name in names:
        assert reads[name].shape == (1, 3)
        assert np.isfinite(reads[name]).all()
