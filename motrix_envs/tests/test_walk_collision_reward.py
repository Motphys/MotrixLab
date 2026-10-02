# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from motrix_env_core.sim import GeomPairCollidingQuery
from motrix_envs.locomotion.humanoid.cfg import WalkRewardsCfg
from motrix_envs.locomotion.humanoid.g1 import make_g129dof_walk_flat_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    PenaltyCollisionRewardCfg,
    penalty_collision_reward,
)


@pytest.mark.parametrize("colliding", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("scale", [0.0, 0.25, 2.0])
def test_collision_reward_uses_any_pair_and_command_penalty_scale(colliding, scale):
    ctx = SimpleNamespace(
        commands={"walk": SimpleNamespace(penalty_scale=np.asarray([scale], dtype=np.float32))}
    )
    value = penalty_collision_reward(ctx, np.asarray(colliding, dtype=np.bool_))
    assert value == pytest.approx(float(any(colliding)) * scale)


def test_collision_reward_config_defaults_are_optional_and_unresolved():
    assert WalkRewardsCfg().penalty_collision is None
    cfg = PenaltyCollisionRewardCfg(weight=-1.0)
    assert cfg.ground_geom == ""
    assert cfg.termination_geoms == ()
    assert cfg.command_name == "walk"


def test_collision_reward_config_declares_exact_pairs():

    cfg = PenaltyCollisionRewardCfg(
        ground_geom="terrain", termination_geoms=("body_a", "body_b"), weight=-1.0
    )
    term = cfg(None)
    assert term.dispatch is penalty_collision_reward
    (query,) = term.args
    assert isinstance(query, GeomPairCollidingQuery)
    assert query.pairs == (("body_a", "terrain"), ("body_b", "terrain"))


@pytest.mark.parametrize("ground_geom,termination_geoms", [("", ("body",)), ("terrain", ())])
def test_collision_reward_requires_resolved_geometry_at_build(ground_geom, termination_geoms):
    cfg = PenaltyCollisionRewardCfg(ground_geom=ground_geom, termination_geoms=termination_geoms, weight=-1.0)
    with pytest.raises(ValueError, match="requires non-empty termination_geoms and ground_geom"):
        cfg(None)


def test_humanoid_config_resolves_collision_reward_from_termination_config():
    base = make_g129dof_walk_flat_cfg()
    cfg = replace(base, rewards=replace(base.rewards, penalty_collision=PenaltyCollisionRewardCfg(weight=-1.0)))
    reward = cfg.rewards.penalty_collision
    assert reward.ground_geom == cfg.terminations.colliding.ground_geom
    assert reward.termination_geoms == cfg.terminations.colliding.termination_geoms
    assert base.rewards.penalty_collision is None
    assert cfg.terminations.colliding == base.terminations.colliding
    query = reward(None).args[0]
    assert query.pairs == tuple((name, reward.ground_geom) for name in cfg.terminations.colliding.termination_geoms)


def test_humanoid_config_preserves_explicit_collision_reward_geometry():
    base = make_g129dof_walk_flat_cfg()
    reward = PenaltyCollisionRewardCfg(ground_geom="other_ground", termination_geoms=("other_body",), weight=-1.0)
    cfg = replace(base, rewards=replace(base.rewards, penalty_collision=reward))
    assert cfg.rewards.penalty_collision.ground_geom == "other_ground"
    assert cfg.rewards.penalty_collision.termination_geoms == ("other_body",)
    assert cfg.terminations.colliding == base.terminations.colliding
