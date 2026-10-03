# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 pyramid-stairs velocity-tracking task (mjlab stairs parity).

Reproduces the mjlab ``STAIRS_TERRAINS_CFG`` training in the MotrixLab
FastSAC stack: a flat/easy/moderate/challenging pyramid-stairs tile grid,
a yaw-aligned height-scan observation sampled from the static height field,
heading-random spawns bound to fixed tiles, and the shared walk reward and
termination scaffold.
"""

import math
from dataclasses import replace

import numpy as np

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import (
    FlatTerrainGeneratorCfg,
    HFieldTerrainCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    PyramidSlopeTerrainGeneratorCfg,
    StairsTerrainGeneratorCfg,
    SystemCameraCfg,
    WaveTerrainGeneratorCfg,
    grid_terrain,
)
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.terminations import BadDofVelocityTerminationCfg, CollidingTerminationCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.cfg import (
    HumanoidVelocityTrackingManagerEnvCfg,
    WalkCommandsCfg,
    WalkObservationsCfg,
    WalkTerminationsCfg,
)
from motrix_envs.locomotion.humanoid.g1 import (
    _G1_TERMINATION_GEOMS,
    _make_g1_rewards,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommandCfg
from motrix_envs.robot import UnitreeG129Dof

# mjlab STAIRS_TERRAINS_CFG proportions and step parameters. Per-tile ring
# count follows mjlab's (size - 2*border - platform) / (2 * step_width) with
# the border folded into the composite margin: rings + 1 sets step_count so
# the outermost ring lands at level 0, flush with the tile rim.
_STAIRS_TIERS = (
    # (name, proportion, step_height range, step width, platform width, step_count)
    ("easy", 0.35, (0.02, 0.05), 0.40, 3.0, 7),
    ("moderate", 0.25, (0.05, 0.08), 0.35, 2.5, 8),
    ("challenging", 0.15, (0.08, 0.10), 0.30, 2.0, 9),
)
_FLAT_PROPORTION = 0.25
_ROWS, _COLS = 10, 8
_TILE = 8.0
_BORDER = 20.0


def _make_stairs_terrain() -> ProceduralHFieldAssetCfg:
    """Build the mjlab-style flat/easy/moderate/challenging stairs mixture."""
    height_scale = 1.0
    rim_level = 0.15  # Flat tiles and stair rims share this normalized level.
    rng = np.random.default_rng(8)
    bounds = []
    cumulative = _FLAT_PROPORTION
    for _name, proportion, h_range, width, platform, step_count in _STAIRS_TIERS:
        cumulative += proportion
        bounds.append((cumulative, h_range, width, platform, step_count))
    cells = []
    for _ in range(_ROWS):
        row = []
        for _ in range(_COLS):
            pick = rng.random()
            if pick < _FLAT_PROPORTION:
                row.append(FlatTerrainGeneratorCfg(height=rim_level, height_scale=height_scale))
                continue
            for bound, h_range, width, platform, step_count in bounds:
                if pick < bound:
                    row.append(
                        StairsTerrainGeneratorCfg(
                            axis="radial",
                            # mjlab pyramid stairs: the central platform is the
                            # top and steps descend outward to the rim.
                            profile="descending",
                            step_count=step_count,
                            step_width=width,
                            step_height=float(rng.uniform(*h_range)),
                            platform_width=platform,
                            base_level=rim_level,
                            height_scale=height_scale,
                        )
                    )
                    break
        cells.append(row)
    size = (_COLS * _TILE + 2.0 * _BORDER, _ROWS * _TILE + 2.0 * _BORDER)
    shape = (int(size[1] / 0.1), int(size[0] / 0.1))
    return ProceduralHFieldAssetCfg(
        generator=grid_terrain(cells, size=size, border=(_BORDER, _BORDER)),
        size=size,
        shape=shape,
    )


@registry.envcfg("g1-walk-stairs")
def make_g129dof_walk_stairs_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track G1 walking commands over a pyramid-stairs tile grid."""
    robot = UnitreeG129Dof()
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            assets=humanoid_cfg.TerrainSceneAssetsCfg(terrain=_make_stairs_terrain()),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground"),
                robot=robot,
            ),
        ),
        rewards=_make_g1_rewards(robot),
        terminations=WalkTerminationsCfg(
            colliding=CollidingTerminationCfg(
                termination_geoms=_G1_TERMINATION_GEOMS,
                ground_geom="floor",
            ),
            bad_dof_velocity=BadDofVelocityTerminationCfg(threshold=100.0),
        ),
        # mjlab-style command space: yaw-rate range matches the velocity task
        # defaults and heading-random spawns face arbitrary stair directions.
        commands=WalkCommandsCfg(
            walk=WalkCommandCfg(
                vel_limit=((-1.0, -1.0, -0.5), (1.0, 1.0, 0.5)),
                stand_prob=0.1,
                resampling_time=6.0,
                gait_period_randomization_width=0.2,
            )
        ),
        observations=WalkObservationsCfg(
            policy=replace(
                WalkObservationsCfg.PolicyCfg(),
                height_scan=humanoid_cfg.HeightScanObsCfg(),
            ),
            value=replace(
                WalkObservationsCfg.ValueCfg(),
                height_scan=humanoid_cfg.HeightScanObsCfg(noise=0.0),
            ),
        ),
        sim_reset=humanoid_cfg.WalkResetCfg(
            humanoid_state=humanoid_cfg.WalkStateResetCfg(
                tile_spawn=True,
                spawn_tiles=(_ROWS, _COLS),
                spawn_border=(_BORDER, _BORDER),
                spawn_yaw_range=math.pi,
            )
        ),
        sim=SimCfg(dt=0.005, solver_iterations=8, solver_tolerance=1e-4),
        render_spacing=0.0,
    )


registry.env("g1-walk-stairs")(ManagerEnv)


# ---------------------------------------------------------------------------
# g1-walk-mixed: mjlab ROUGH_TERRAINS_CFG parity.
# ---------------------------------------------------------------------------

# Column layout mirrors mjlab's proportions exactly: flat .2 / pyramid stairs
# .2 / inverted stairs .2 / slope .1 / inverted slope .1 / random rough .1 /
# wave .1 across a 10-row x 10-column grid. Difficulty, which mjlab
# interpolates along curriculum rows, is sampled per tile instead.
_MIX_COLS = (
    "flat",
    "flat",
    "stairs",
    "stairs",
    "stairs_inv",
    "stairs_inv",
    "slope",
    "slope_inv",
    "rough",
    "wave",
)
_MIX_ROWS = 10
_MIX_TILE = 8.0
_MIX_BORDER = 20.0
_SLOPE_HEIGHT_SCALE = 3.2  # slope peak = slope * (8/2 - platform/2) <= 3.0 m.


def _make_mixed_terrain() -> ProceduralHFieldAssetCfg:
    """Build the mjlab ROUGH_TERRAINS_CFG-style seven-type mixture."""
    rng = np.random.default_rng(8)

    def stairs(kind: str) -> StairsTerrainGeneratorCfg:
        difficulty = rng.uniform(0.25, 1.0)
        return StairsTerrainGeneratorCfg(
            axis="radial",
            # descending: central platform is the top; ascending: central pit.
            profile="descending" if kind == "stairs" else "ascending",
            step_count=9,
            step_width=0.30,
            step_height=0.10 * difficulty,
            platform_width=3.0,
            base_level=0.1,
            height_scale=1.0,
        )

    def slope(inverted: bool) -> PyramidSlopeTerrainGeneratorCfg:
        return PyramidSlopeTerrainGeneratorCfg(
            slope=1.0 * rng.uniform(0.25, 1.0),
            inverted=inverted,
            platform_width=2.0,
            height_scale=_SLOPE_HEIGHT_SCALE,
        )

    cells = []
    for _ in range(_MIX_ROWS):
        row = []
        for kind in _MIX_COLS:
            if kind == "flat":
                row.append(FlatTerrainGeneratorCfg(height=0.0, height_scale=1.0))
            elif kind in ("stairs", "stairs_inv"):
                row.append(stairs(kind))
            elif kind == "slope":
                row.append(slope(False))
            elif kind == "slope_inv":
                row.append(slope(True))
            elif kind == "rough":
                row.append(
                    NoiseTerrainGeneratorCfg(
                        seed=int(rng.integers(1 << 30)),
                        height_range=(0.02, 0.10),
                        height_scale=0.3,
                    )
                )
            else:  # wave
                row.append(
                    WaveTerrainGeneratorCfg(
                        amplitude=0.2 * rng.uniform(0.25, 1.0),
                        num_waves=4,
                        height_scale=0.5,
                    )
                )
        cells.append(row)
    size = (len(_MIX_COLS) * _MIX_TILE + 2.0 * _MIX_BORDER, _MIX_ROWS * _MIX_TILE + 2.0 * _MIX_BORDER)
    shape = (int(size[1] / 0.1), int(size[0] / 0.1))
    return ProceduralHFieldAssetCfg(
        generator=grid_terrain(cells, size=size, border=(_MIX_BORDER, _MIX_BORDER)),
        size=size,
        shape=shape,
    )


@registry.envcfg("g1-walk-mixed")
def make_g129dof_walk_mixed_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track G1 walking commands over the mjlab rough terrain mixture."""
    robot = UnitreeG129Dof()
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            assets=humanoid_cfg.TerrainSceneAssetsCfg(terrain=_make_mixed_terrain()),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground"),
                robot=robot,
            ),
        ),
        rewards=_make_g1_rewards(robot),
        terminations=WalkTerminationsCfg(
            colliding=CollidingTerminationCfg(
                termination_geoms=_G1_TERMINATION_GEOMS,
                ground_geom="floor",
            ),
            bad_dof_velocity=BadDofVelocityTerminationCfg(threshold=100.0),
        ),
        commands=WalkCommandsCfg(
            walk=WalkCommandCfg(
                vel_limit=((-1.0, -1.0, -0.5), (1.0, 1.0, 0.5)),
                stand_prob=0.1,
                resampling_time=6.0,
                gait_period_randomization_width=0.2,
            )
        ),
        observations=WalkObservationsCfg(
            policy=replace(
                WalkObservationsCfg.PolicyCfg(),
                height_scan=humanoid_cfg.HeightScanObsCfg(),
            ),
            value=replace(
                WalkObservationsCfg.ValueCfg(),
                height_scan=humanoid_cfg.HeightScanObsCfg(noise=0.0),
            ),
        ),
        sim_reset=humanoid_cfg.WalkResetCfg(
            humanoid_state=humanoid_cfg.WalkStateResetCfg(
                tile_spawn=True,
                spawn_tiles=(_MIX_ROWS, len(_MIX_COLS)),
                spawn_border=(_MIX_BORDER, _MIX_BORDER),
                spawn_yaw_range=math.pi,
            )
        ),
        sim=SimCfg(dt=0.005, solver_iterations=8, solver_tolerance=1e-4),
        render_spacing=0.0,
    )


registry.env("g1-walk-mixed")(ManagerEnv)
