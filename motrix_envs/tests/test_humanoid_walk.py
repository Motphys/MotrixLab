# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contract tests for the manager-based humanoid walk presets."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from motrix_env_core import registry
from motrix_env_core.config.scene import ProceduralHFieldAssetCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.random import RandomSource
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg
from motrix_envs.locomotion.humanoid.dex_evt import (
    make_dex_evt_walk_flat_cfg,
    make_dex_evt_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.g1 import (
    make_g129dof_walk_flat_cfg,
    make_g129dof_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.k1 import (
    make_k1_walk_flat_cfg,
    make_k1_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.microduck import (
    make_microduck_walk_flat_cfg,
    make_microduck_walk_rough_cfg,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommandCfg


def test_walk_command_resamples_and_resets_only_selected_rows():
    cfg = WalkCommandCfg(stand_prob=0.0, resampling_time=0.04, gait_period_randomization_width=0.1)
    build = BuildContext(None, {}, {}, None, num_envs=3, seed=0)
    command = cfg(build)
    command.command.fill_(7)
    command.steps[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    torch.manual_seed(0)
    rand = RandomSource()
    command.advance(SimpleNamespace(rand=rand))
    torch.testing.assert_close(command.command[1:], torch.full((2, 3), 7.0))
    assert bool((command.command[0] >= command.vel_limit_low).all())
    assert bool((command.command[0] < command.vel_limit_high).all())
    before_command = command.command.clone()
    before_offsets = command.phase_offset.clone()
    before_steps = command.phase_step.clone()
    command.reset(SimpleNamespace(rand=rand, dt=cfg.ctrl_dt), torch.tensor([1]), torch.tensor([False]))
    torch.testing.assert_close(command.command[[0, 2]], before_command[[0, 2]])
    torch.testing.assert_close(command.phase_offset[[0, 2]], before_offsets[[0, 2]])
    torch.testing.assert_close(command.phase_step[[0, 2]], before_steps[[0, 2]])
    assert bool((command.command[1] >= command.vel_limit_low).all())
    assert bool((command.command[1] < command.vel_limit_high).all())
    assert command.steps[1, 0] == 0


@pytest.mark.parametrize(
    ("env_name", "action_dim", "policy_dim", "value_dim"),
    [
        ("g1-walk-flat", 29, 100, 103),
        ("dex-evt-walk-flat", 23, 82, 85),
        ("k1-walk-flat", 22, 79, 82),
        ("microduck-walk-flat", 14, 55, 58),
    ],
)
def test_walk_presets_use_shared_manager_env(env_name, action_dim, policy_dim, value_dim):
    env = registry.make(env_name, num_envs=2)
    env.initialize()
    state = env.state

    assert type(env) is ManagerEnv
    assert isinstance(env.cfg, HumanoidVelocityTrackingManagerEnvCfg)
    assert env.action_space.shape == (action_dim,)
    assert state.obs.policy.shape == (2, policy_dim)
    assert state.obs.value.shape == (2, value_dim)


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

    assert rough_cfg.rewards == flat_cfg.rewards
    assert rough_cfg.commands == flat_cfg.commands
    assert rough_cfg.actions == flat_cfg.actions
    assert rough_cfg.terminations == flat_cfg.terminations
    assert replace(rough_cfg.sim, max_contacts=None, max_contact_constraints=None) == replace(
        flat_cfg.sim, max_contacts=None, max_contact_constraints=None
    )
    assert rough_cfg.sim.max_contacts >= flat_cfg.sim.max_contacts
    assert rough_cfg.sim.max_contact_constraints >= flat_cfg.sim.max_contact_constraints
    assert flat_cfg.sim_reset.humanoid_state.spawn_xy_range == 0.0
    assert rough_cfg.sim_reset.humanoid_state.spawn_xy_range > flat_cfg.sim_reset.humanoid_state.spawn_xy_range
    assert flat_cfg.render_spacing > 0.0
    assert rough_cfg.render_spacing == 0.0
    assert isinstance(rough_cfg.scene.assets.terrain, ProceduralHFieldAssetCfg)


def test_humanoid_walk_rejects_incomplete_joint_preset():
    cfg = make_g129dof_walk_flat_cfg()
    cfg.scene.objs.robot.key_pose.joint_names.pop(0)
    cfg.scene.objs.robot.key_pose.poses["default"].pop(0)

    # The backend rejects incomplete key poses at model compile time.
    with pytest.raises(ValueError, match="must cover every joint"):
        ManagerEnv(cfg, num_envs=2)
