# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Slope-only G1 experiment using the mixed task's unchanged slope geometry."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg
from motrix_envs.locomotion.humanoid.g1_stairs import _make_mixed_terrain, make_g129dof_walk_mixed_cfg


@registry.envcfg("g1-walk-slope")
def make_g129dof_walk_slope_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Train normal/inverted slopes with the original mixed-equivalent curriculum.

    Columns 0/1 are normal/inverted pyramids. Rows retain the mixed terrain's
    difficulty gradient and per-lane curriculum. No adaptive
    column reassignment or privileged actor linear velocity is used.
    """
    cfg = make_g129dof_walk_mixed_cfg()
    return replace(
        cfg,
        scene=replace(
            cfg.scene,
            assets=replace(cfg.scene.assets, terrain=_make_mixed_terrain(("slope", "slope_inv"))),
        ),
        commands=replace(
            cfg.commands,
            walk=replace(
                cfg.commands.walk,
                terrain_sampling_enabled=False,
                terrain_balanced_columns=True,
            ),
        ),
        observations=replace(
            cfg.observations,
            policy=replace(cfg.observations.policy, base_lin_vel=None),
        ),
        sim_reset=replace(
            cfg.sim_reset,
            humanoid_state=replace(
                cfg.sim_reset.humanoid_state,
                spawn_tiles=(cfg.sim_reset.humanoid_state.spawn_tiles[0], 2),
            ),
        ),
    )


registry.env("g1-walk-slope")(ManagerEnv)
