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
# Global tile datum: every tile rim sits at 3.0 m so inverted-pit rims stay
# flush with flat/stairs neighbors (mjlab keeps its rim at z=0 and digs pits
# below ground via per-geom hfield offsets; a single shared non-negative
# composite hfield instead shifts the whole map up by the max pit depth).
_DATUM = 3.0
_FIELD_HEIGHT_SCALE = 6.0
_DATUM_LEVEL = _DATUM / _FIELD_HEIGHT_SCALE


def _make_mixed_terrain(
    columns: tuple[str, ...] = _MIX_COLS,
    slope_scale: float = 1.0,
    slope_platform_width: float = 2.0,
) -> ProceduralHFieldAssetCfg:
    """Build the mjlab ROUGH_TERRAINS_CFG-style seven-type mixture.

    Rows are a difficulty gradient -- row ``r`` scales every generator's
    amplitude by ``(r + 1) / rows`` -- so the terrain curriculum can move
    lanes between easy and hard rows on episode ends, mirroring mjlab's
    difficulty-interpolated terrain levels. ``slope_scale`` multiplies the
    slope generators' rise-per-meter so derived tasks can cap the hardest
    pitch below the mixed-task default of 1.0 (45 degrees).
    ``slope_platform_width`` resizes the slopes' flat central pad.
    """
    rng = np.random.default_rng(8)

    def stairs(kind: str, difficulty: float) -> StairsTerrainGeneratorCfg:
        return StairsTerrainGeneratorCfg(
            axis="radial",
            # descending: central platform is the top; ascending: central pit.
            profile="descending" if kind == "stairs" else "ascending",
            step_count=9,
            step_width=0.30,
            step_height=0.10 * difficulty,
            platform_width=3.0,
            base_level=_DATUM_LEVEL,
            height_scale=_FIELD_HEIGHT_SCALE,
        )

    def slope(inverted: bool, difficulty: float) -> PyramidSlopeTerrainGeneratorCfg:
        # Pit depth equals the mound peak, so an inverted rim stays flush with
        # the shared datum by shifting the shape up by (datum - peak); a
        # normal mound simply rises from the datum.
        peak = slope_scale * difficulty * (_MIX_TILE / 2.0 - 1.0)
        base = _DATUM_LEVEL if not inverted else (_DATUM - peak) / _FIELD_HEIGHT_SCALE
        return PyramidSlopeTerrainGeneratorCfg(
            slope=slope_scale * difficulty,
            inverted=inverted,
            platform_width=slope_platform_width,
            base_level=base,
            height_scale=_FIELD_HEIGHT_SCALE,
        )

    cells = []
    for row_index in range(_MIX_ROWS):
        difficulty = (row_index + 1) / _MIX_ROWS
        row = []
        for kind in columns:
            if kind == "flat":
                row.append(FlatTerrainGeneratorCfg(height=_DATUM_LEVEL, height_scale=_FIELD_HEIGHT_SCALE))
            elif kind in ("stairs", "stairs_inv"):
                row.append(stairs(kind, difficulty))
            elif kind == "slope":
                row.append(slope(False, difficulty))
            elif kind == "slope_inv":
                row.append(slope(True, difficulty))
            elif kind == "rough":
                row.append(
                    NoiseTerrainGeneratorCfg(
                        seed=int(rng.integers(1 << 30)),
                        height_range=(0.02, 0.10),
                        height_scale=_FIELD_HEIGHT_SCALE,
                    )
                )
            else:  # wave
                row.append(
                    WaveTerrainGeneratorCfg(
                        amplitude=0.2 * difficulty,
                        num_waves=4,
                        height_scale=_FIELD_HEIGHT_SCALE,
                    )
                )
        cells.append(row)
    size = (len(columns) * _MIX_TILE + 2.0 * _MIX_BORDER, _MIX_ROWS * _MIX_TILE + 2.0 * _MIX_BORDER)
    shape = (int(size[1] / 0.1), int(size[0] / 0.1))
    return ProceduralHFieldAssetCfg(
        generator=grid_terrain(
            cells,
            size=size,
            border=(_MIX_BORDER, _MIX_BORDER),
            base=FlatTerrainGeneratorCfg(height=_DATUM_LEVEL, height_scale=_FIELD_HEIGHT_SCALE),
        ),
        size=size,
        shape=shape,
    )


@registry.envcfg("g1-walk-mixed")
def make_g129dof_walk_mixed_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track G1 walking commands over the mjlab rough terrain mixture."""
    robot = UnitreeG129Dof()
    rewards = _make_g1_rewards(robot)
    # A world-horizontal foot target conflicts with conforming to slopes.
    # Leave the torso upright term intact; mjlab G1 has no foot-orientation term.
    rewards.penalty_feet_ori.weight = 0.0
    # Translation must improve value beyond the in-place gait solution.
    rewards.tracking_lin_vel.weight = 12.0
    # Test free contact timing rather than rewarding an in-place height clock.
    rewards.feet_phase.weight = 0.0
    # Permit faster corrective ankle/leg actions on slopes; retain smoothing.
    rewards.penalty_action_rate.weight = -0.5
    # Keep torso/arm regularization, but allow feet to conform to terrain.
    for joint in rewards.pose.pose_weights:
        if "ankle_pitch" in joint or "ankle_roll" in joint:
            rewards.pose.pose_weights[joint] = 1.0
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            assets=humanoid_cfg.TerrainSceneAssetsCfg(terrain=_make_mixed_terrain()),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground"),
                robot=robot,
            ),
        ),
        rewards=rewards,
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
                resampling_time_range=(3.0, 8.0),
                heading_command=True,
                heading_prob=0.3,
                heading_control_stiffness=0.5,
                gait_period_randomization_width=0.2,
                terrain_tile_length=_MIX_TILE,
                terrain_progress_promotion=True,
                terrain_require_survival_for_promotion=True,
                terrain_sampling_enabled=False,
                terrain_min_level=1,
                terrain_failure_demote_ratio=0.25,
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
                terrain_curriculum=True,
            )
        ),
        sim=SimCfg(dt=0.01, solver_iterations=8, solver_tolerance=1e-4),
        render_spacing=0.0,
    )


registry.env("g1-walk-mixed")(ManagerEnv)
