# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_env_core.config.scene import HFieldTerrainCfg, ProceduralHFieldAssetCfg
from motrix_env_core.sim.model import ActuatorType
from motrix_envs.locomotion.quadruped.cfg import RewardScales
from motrix_envs.locomotion.quadruped.velocity_command import RandomPlanarVelocityBinding
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


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


def _swing_reward(quadruped_env, contacts: np.ndarray) -> np.ndarray:
    info = {
        "feet_phase": np.full((1, quadruped_env._num_feet), 0.75, dtype=np.float32),
        "contacts": contacts,
    }
    return quadruped_env._reward_swing_feet_z(info, None, None, None)


def test_quadruped_reward_scales_are_structured_and_not_shared():
    go1 = registry.make_env_config("go1-walk-flat").reward_config.scales
    go2 = registry.make_env_config("go2-walk-flat").reward_config.scales

    assert isinstance(go1, RewardScales)
    assert isinstance(go2, RewardScales)
    assert go1 is not go2


def test_go2_velocity_commands_remain_constant_before_resampling_interval():
    np.random.seed(7)
    env = registry.make("go2-walk-flat", num_envs=64, mode="train")
    state = env.init_state()
    commands = state.info["commands"].copy()
    velocity_cfg = env.cfg.commands.velocity

    assert commands.shape == (64, 3)
    assert np.unique(commands, axis=0).shape[0] > 1
    assert np.all(commands >= velocity_cfg.lower)
    assert np.all(commands <= velocity_cfg.upper)

    env.step(np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32))

    np.testing.assert_array_equal(state.info["commands"], commands)


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
    state = env.init_state()
    randomization = env.cfg.randomization
    joint_pos_diff = env.get_dof_pos() - env.default_angles
    dof_vel = env.sim_data["dof_vel"]

    assert np.all(np.abs(joint_pos_diff) <= randomization.joint_pos_noise + 1e-6)
    assert np.any(np.abs(joint_pos_diff) > 0.0)
    assert np.all(np.abs(dof_vel[:, -env._num_action :]) <= randomization.joint_vel_noise + 1e-6)
    assert np.any(np.abs(dof_vel[:, -env._num_action :]) > 0.0)
    assert np.all(np.abs(dof_vel[:, :3]) <= np.asarray(randomization.base_lin_vel_noise, dtype=np.float32) + 1e-6)
    assert np.all(np.abs(dof_vel[:, 3:6]) <= np.asarray(randomization.base_ang_vel_noise, dtype=np.float32) + 1e-6)
    assert np.unique(joint_pos_diff, axis=0).shape[0] > 1
    command_interval = env.cfg.commands.velocity.resampling_seconds_range
    if command_interval is not None:
        assert np.all(state.info["command_resampling_time"] >= command_interval[0])
        assert np.all(state.info["command_resampling_time"] <= command_interval[1])


def test_go2_pd_and_friction_randomization_is_per_env_and_episode_constant():
    env = registry.make("go2-walk-flat", num_envs=16, mode="train")
    env.init_state()
    randomization = env.cfg.randomization
    _, kp, damping, friction = _read_param_overrides(env)
    nominal_kp, nominal_damping = _nominal_params(env)
    kp_scale = kp / nominal_kp
    damping_scale = damping / nominal_damping

    assert np.all(kp_scale >= randomization.kp_scale_range[0])
    assert np.all(kp_scale <= randomization.kp_scale_range[1])
    assert np.all(damping_scale >= randomization.damping_scale_range[0])
    assert np.all(damping_scale <= randomization.damping_scale_range[1])
    assert np.unique(kp, axis=0).shape[0] > 1
    assert np.unique(damping, axis=0).shape[0] > 1
    assert np.all(friction[:, 0] >= randomization.sliding_friction_range[0])
    assert np.all(friction[:, 0] <= randomization.sliding_friction_range[1])
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

    assert np.all(mass_scale >= randomization.base_mass_scale_range[0])
    assert np.all(mass_scale <= randomization.base_mass_scale_range[1])
    assert np.all(np.abs(center_of_mass_offset) <= com_noise + 1e-7)
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
    state.info["current_actions"] = previous
    state.info["action_delay_steps"][:] = (0, 1)

    env.apply_action(current, state)
    env.sim_data.execute()

    expected_actions = np.stack([current[0], previous[1]])
    expected_ctrls = expected_actions * env.cfg.control_config.action_scale + env.default_angles
    np.testing.assert_allclose(env.sim_data["actuator_ctrls"], expected_ctrls)


def test_go2_command_resampling_only_updates_due_environments(monkeypatch: pytest.MonkeyPatch):
    env = registry.make("go2-walk-flat", num_envs=3, mode="train")
    state = env.init_state()
    commands = np.array([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0], [0.3, 0.0, -0.1]], dtype=np.float32)
    replacement = np.array([[0.4, -0.2, 0.3], [0.5, 0.2, -0.3]], dtype=np.float32)
    state.info["commands"][:] = commands
    state.info["command_resampling_time"][:] = (0.0, 1.0, 0.0)
    monkeypatch.setattr(env, "resample_commands", lambda num_envs: replacement[:num_envs].copy())

    env._update_commands(state.info)

    np.testing.assert_array_equal(state.info["commands"][[0, 2]], replacement)
    np.testing.assert_array_equal(state.info["commands"][[1]], commands[[1]])
    assert state.info["command_resampling_time"][1] == pytest.approx(1.0 - env.cfg.ctrl_dt)
    assert np.all(state.info["command_resampling_time"][[0, 2]] > 0.0)


def test_random_planar_velocity_binding_is_seeded_vectorized_and_task_specific():
    first = RandomPlanarVelocityBinding(
        np.array([-0.5, -0.4, -1.0], dtype=np.float32),
        np.array([1.0, 0.4, 1.0], dtype=np.float32),
        rng=np.random.default_rng(7),
        standing_probability=0.0,
    ).read_command(batch_size=8)
    repeated = RandomPlanarVelocityBinding(
        np.array([-0.5, -0.4, -1.0], dtype=np.float32),
        np.array([1.0, 0.4, 1.0], dtype=np.float32),
        rng=np.random.default_rng(7),
        standing_probability=0.0,
    ).read_command(batch_size=8)

    assert first.values.shape == (8, 3)
    assert np.unique(first.values, axis=0).shape[0] > 1
    np.testing.assert_array_equal(first.values, repeated.values)
    assert np.all(first.values >= [-0.5, -0.4, -1.0])
    assert np.all(first.values <= [1.0, 0.4, 1.0])


def test_random_planar_velocity_binding_can_sample_a_standing_batch():
    binding = RandomPlanarVelocityBinding(
        np.full(3, -1.0, dtype=np.float32),
        np.full(3, 1.0, dtype=np.float32),
        rng=np.random.default_rng(3),
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
    state.info["commands"][:] = commands
    action_delay_steps = state.info["action_delay_steps"].copy()
    _, kp, damping, friction = _read_param_overrides(env)
    mass, center_of_mass = _read_mass_overrides(env)
    state.terminated[:] = (False, True, False)
    replacement = np.array([[0.4, -0.2, 0.3]], dtype=np.float32)
    monkeypatch.setattr(env, "resample_commands", lambda num_envs: np.broadcast_to(replacement, (num_envs, 3)).copy())

    env._reset_done_envs()

    np.testing.assert_array_equal(state.info["commands"][[0, 2]], commands[[0, 2]])
    np.testing.assert_array_equal(state.info["commands"][[1]], replacement)
    np.testing.assert_array_equal(state.info["action_delay_steps"][[0, 2]], action_delay_steps[[0, 2]])
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
    }.isdisjoint(env.state.info)
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


def test_zero_velocity_command_freezes_gait_phase():
    env = registry.make("go2-walk-flat", num_envs=2, mode="train")
    info = {
        "commands": np.zeros((2, 3), dtype=np.float32),
        "phase": np.full(2, 0.5, dtype=np.float32),
        "feet_phase": np.full((2, env._num_feet), 0.5, dtype=np.float32),
    }

    env._advance_phase(info)

    np.testing.assert_array_equal(info["phase"], np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(info["feet_phase"], np.zeros((2, env._num_feet), dtype=np.float32))


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
    # Regression: SceneCfg uses full widths, while both native backends consume half-extents.
    assert rough_cfg.scene.assets.terrain.size == (64.0, 64.0)
    assert rough_cfg.control_config == flat_cfg.control_config
    assert rough_cfg.spawn_xy_range == flat_cfg.spawn_xy_range > 0.0
    assert rough_cfg.render_spacing == flat_cfg.render_spacing == 0.0


@pytest.mark.parametrize("env_name", ["anymalc-walk-rough", "go1-walk-rough", "go2-walk-rough"])
def test_quadruped_rough_walk_reset_and_base_height_are_terrain_relative(env_name):
    env = registry.make(env_name, num_envs=8, mode="train")
    env.init_state()
    base_pos = env.sim_data["base_pos"]
    env_ids = np.arange(env.num_envs, dtype=np.int64)
    ground_height = env.sim.sample_terrain_height(env.cfg.ground_geom_name, env_ids, base_pos[:, None, :2])[:, 0]
    relative_height = base_pos[:, 2] - ground_height
    height_scale = env.cfg.scene.assets.terrain.generator.height_scale

    assert ground_height.shape == (env.num_envs,)
    assert np.all(np.abs(base_pos[:, :2]) <= env.cfg.spawn_xy_range)
    assert np.all(relative_height >= env.cfg.initial_base_position[2] - 1e-6)
    assert np.all(relative_height <= env.cfg.initial_base_position[2] + height_scale + 1e-6)
    expected = np.square(relative_height - env.cfg.reward_config.base_height_target)
    np.testing.assert_allclose(env._reward_base_height(None, None, None, None), expected)

    next_state = env.step(np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32))
    assert next_state.reward.shape == (env.num_envs,)
    assert np.all(np.isfinite(next_state.reward))


def test_foot_position_sensor_names_belong_to_walk_task_config():
    go1 = registry.make_env_config("go1-walk-flat")
    anymalc = registry.make_env_config("anymalc-walk-flat")

    assert go1.sensor.foot_positions == ("FL_pos", "FR_pos", "RL_pos", "RR_pos")
    assert anymalc.sensor.foot_positions == go1.sensor.foot_positions


def test_walk_task_selects_named_robot_key_pose():
    cfg = registry.make_env_config("go1-walk-flat")
    joint_count = len(cfg.scene.objs.robot.key_pose.joint_names)
    cfg.scene.objs.robot.key_pose.poses["zero"] = [0.0] * joint_count
    cfg.key_pose_name = "zero"

    env = QuadrupedWalkTask(cfg, num_envs=1)

    np.testing.assert_array_equal(env.default_angles, np.zeros((joint_count,), dtype=np.float32))


def test_swing_height_reward_requires_feet_to_leave_ground(quadruped_env):
    reward_cfg = quadruped_env.cfg.reward_config
    quadruped_env.feet_pos[:, :, 2] = reward_cfg.target_foot_height - reward_cfg.base_height_target

    reward = _swing_reward(
        quadruped_env,
        contacts=np.ones((1, quadruped_env._num_feet), dtype=bool),
    )

    np.testing.assert_array_equal(reward, np.zeros((1,), dtype=np.float32))


def test_swing_height_reward_prefers_target_height(quadruped_env):
    reward_cfg = quadruped_env.cfg.reward_config
    target_z = reward_cfg.target_foot_height - reward_cfg.base_height_target
    no_contacts = np.zeros((1, quadruped_env._num_feet), dtype=bool)

    quadruped_env.feet_pos[:, :, 2] = target_z
    target_reward = _swing_reward(quadruped_env, no_contacts)

    quadruped_env.feet_pos[:, :, 2] = target_z - 2.0 * reward_cfg.swing_feet_height_sigma
    off_target_reward = _swing_reward(quadruped_env, no_contacts)

    np.testing.assert_allclose(target_reward, np.ones((1,), dtype=np.float32))
    assert np.all(target_reward > off_target_reward)
    assert np.all(off_target_reward > 0.0)


def test_swing_contact_penalty_detects_dragging_feet(quadruped_env):
    feet_phase = np.full((1, quadruped_env._num_feet), 0.75, dtype=np.float32)
    no_contacts = np.zeros((1, quadruped_env._num_feet), dtype=bool)
    all_contacts = np.ones((1, quadruped_env._num_feet), dtype=bool)

    no_drag = quadruped_env._reward_swing_contact(
        {"feet_phase": feet_phase, "contacts": no_contacts},
        None,
        None,
        None,
    )
    all_drag = quadruped_env._reward_swing_contact(
        {"feet_phase": feet_phase, "contacts": all_contacts},
        None,
        None,
        None,
    )

    np.testing.assert_array_equal(no_drag, np.zeros((1,), dtype=np.float32))
    np.testing.assert_array_equal(all_drag, np.ones((1,), dtype=np.float32))
