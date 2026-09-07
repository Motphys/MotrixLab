# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core import registry
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingEnvCfg, HumanoidWalkSceneCfg
from motrix_envs.locomotion.humanoid.dex_evt import (
    make_dex_evt_walk_flat_cfg,
    make_dex_evt_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.g1 import (
    G129dofWalkTask,
    make_g129dof_walk_flat_cfg,
    make_g129dof_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.k1 import make_k1_walk_flat_cfg, make_k1_walk_rough_cfg
from motrix_envs.locomotion.humanoid.microduck import (
    make_microduck_walk_flat_cfg,
    make_microduck_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.walk_np import HumanoidVelocityTrackingEnv


@pytest.mark.parametrize(
    ("env_name", "action_dim", "policy_dim", "value_dim"),
    [
        ("g1-walk-flat", 29, 100, 103),
        ("g1-walk-rough", 29, 100, 103),
        ("dex-evt-walk-flat", 23, 82, 85),
        ("dex-evt-walk-rough", 23, 82, 85),
        ("k1-walk-flat", 22, 79, 82),
        ("k1-walk-rough", 22, 79, 82),
        ("microduck-walk-flat", 14, 55, 58),
        ("microduck-walk-rough", 14, 55, 58),
    ],
)
def test_walk_presets_use_shared_humanoid_env(env_name, action_dim, policy_dim, value_dim):
    env = registry.make(env_name, num_envs=2)
    state = env.init_state()

    assert type(env) is HumanoidVelocityTrackingEnv
    assert isinstance(env.cfg, HumanoidVelocityTrackingEnvCfg)
    assert env.action_space.shape == (action_dim,)
    assert state.obs.policy.shape == (2, policy_dim)
    assert state.obs.value.shape == (2, value_dim)

    joint_names = env._joint_names
    robot = env.cfg.scene.objs.robot
    assert env._base_link_name == robot.resolved_base_link_name
    assert tuple(env._feet_link_names) == robot.resolved_foot_link_names
    assert env.sim_data["foot_pos"].shape == (2, 2, 3)
    default_pose = dict(zip(robot.key_pose.joint_names, robot.key_pose.poses["default"], strict=True))
    expected_defaults = np.asarray([default_pose[name] for name in joint_names])
    expected_pose_weights = np.asarray([env.cfg.reward_config.pose_weights[name] for name in joint_names])
    np.testing.assert_allclose(env.default_joint_angles, expected_defaults)
    np.testing.assert_allclose(env.pose_weights, expected_pose_weights)


def test_dex_evt_walk_uses_shared_reward_scales_and_named_contact_termination():
    walk_cfg = make_dex_evt_walk_flat_cfg()
    g1_cfg = make_g129dof_walk_flat_cfg()
    env = HumanoidVelocityTrackingEnv(walk_cfg, num_envs=2)
    env.init_state()
    quantities = env._state_quantities(slice(None), np.arange(2, dtype=np.int64))

    assert walk_cfg.reward_config.scales == g1_cfg.reward_config.scales
    assert env._num_termination_pairs == 14
    np.testing.assert_allclose(quantities.foot_clearance, 0.0, atol=1e-3)


def test_k1_walk_uses_shared_reward_scales_and_named_contact_termination():
    walk_cfg = make_k1_walk_flat_cfg()
    g1_cfg = make_g129dof_walk_flat_cfg()
    env = HumanoidVelocityTrackingEnv(walk_cfg, num_envs=2)
    env.init_state()
    quantities = env._state_quantities(slice(None), np.arange(2, dtype=np.int64))

    assert walk_cfg.reward_config.scales == g1_cfg.reward_config.scales
    assert env._num_termination_pairs == 18
    np.testing.assert_allclose(quantities.foot_clearance, 0.00716, atol=2e-3)


def test_microduck_walk_uses_named_contact_termination():
    walk_cfg = make_microduck_walk_flat_cfg()
    env = HumanoidVelocityTrackingEnv(walk_cfg, num_envs=2)
    env.init_state()
    quantities = env._state_quantities(slice(None), np.arange(2, dtype=np.int64))

    assert env._num_termination_pairs == 1
    np.testing.assert_allclose(quantities.foot_clearance, 0.0, atol=4e-3)
    # Microduck ankle-link frames are ~90 degrees away from world-aligned, so
    # the foot-orientation penalty must be measured against the default-pose
    # reference: it is zero at the default stance for any link orientation.
    np.testing.assert_allclose(env._r_feet_ori(quantities), 0.0, atol=1e-3)


@pytest.mark.parametrize(
    ("make_flat_cfg", "make_rough_cfg"),
    [
        (make_g129dof_walk_flat_cfg, make_g129dof_walk_rough_cfg),
        (make_dex_evt_walk_flat_cfg, make_dex_evt_walk_rough_cfg),
        (make_k1_walk_flat_cfg, make_k1_walk_rough_cfg),
        (make_microduck_walk_flat_cfg, make_microduck_walk_rough_cfg),
    ],
)
def test_walk_rough_only_overrides_scene_spawn_range_and_render_spacing(make_flat_cfg, make_rough_cfg):
    flat_cfg = make_flat_cfg()
    rough_cfg = make_rough_cfg()

    assert rough_cfg.control_config == flat_cfg.control_config
    assert rough_cfg.reward_config == flat_cfg.reward_config
    assert rough_cfg.commands == flat_cfg.commands
    assert rough_cfg.normalization == flat_cfg.normalization
    assert rough_cfg.gait == flat_cfg.gait
    assert rough_cfg.curriculum == flat_cfg.curriculum
    assert rough_cfg.asset == flat_cfg.asset
    assert rough_cfg.sim == flat_cfg.sim
    assert flat_cfg.spawn_xy_range == 0.0
    assert rough_cfg.spawn_xy_range == 4.0
    assert flat_cfg.render_spacing > 0.0
    assert rough_cfg.render_spacing == 0.0
    assert rough_cfg.scene.assets.terrain.size == (32.0, 32.0)


@pytest.mark.parametrize(
    ("make_cfg", "camera_distance", "camera_lookat"),
    [
        (make_g129dof_walk_flat_cfg, 6.0, None),
        (make_g129dof_walk_rough_cfg, 6.0, None),
        (make_dex_evt_walk_flat_cfg, 6.0, None),
        (make_dex_evt_walk_rough_cfg, 6.0, None),
        (make_k1_walk_flat_cfg, 6.0, None),
        (make_k1_walk_rough_cfg, 6.0, None),
        (make_microduck_walk_flat_cfg, 0.35, (0.0, 0.0, 0.12)),
        (make_microduck_walk_rough_cfg, 0.35, (0.0, 0.0, 0.12)),
    ],
)
def test_walk_presets_use_shared_system_camera(make_cfg, camera_distance, camera_lookat):
    scene = make_cfg().scene

    assert isinstance(scene, HumanoidWalkSceneCfg)
    assert scene.system_camera.lookat == camera_lookat
    assert scene.system_camera.distance == pytest.approx(camera_distance)
    assert scene.system_camera.elevation == pytest.approx(-20.0)
    assert scene.system_camera.azimuth == pytest.approx(180.0)


def test_g1_walk_legacy_class_name_is_preserved():
    assert G129dofWalkTask is HumanoidVelocityTrackingEnv


def test_humanoid_walk_rejects_incomplete_joint_preset():
    cfg = make_g129dof_walk_flat_cfg()
    cfg.scene.objs.robot.key_pose.joint_names.pop(0)
    cfg.scene.objs.robot.key_pose.poses["default"].pop(0)

    with pytest.raises(KeyError, match="robot key pose 'default' must match robot joints exactly"):
        HumanoidVelocityTrackingEnv(cfg)
