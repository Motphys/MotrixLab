# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the humanoid velocity-tracking task."""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.sim import LinkPositionQuery, LinkQuaternionQuery
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommand


@dispatch
def gait_phase_obs(ctx: ManagerContext, out: np.ndarray, offset: np.int64, size: np.int64) -> None:
    walk: WalkCommand = ctx.commands["walk"]
    for index in range(size):
        out[index] = walk.sin_cos[offset + index]


@configclass(kw_only=True)
class GaitPhaseObsCfg(ObservationTermCfg):
    """``sin``/``cos`` slice of the gait-phase clock.

    The command term's ``sin_cos`` lane layout is
    ``[sin_l, sin_r, cos_l, cos_r]``: offset 0 gives the two sin values and
    offset 2 the two cos values, matching the direct env's obs order.
    """

    offset: int = 0
    size: int = 2

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        return ObsTerm(self.size, gait_phase_obs, np.int64(self.offset), np.int64(self.size))


@dispatch
def height_scan_obs(
    ctx: ManagerContext,
    out: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    offsets: np.ndarray,
    scale: np.float32,
    noise_amplitude: np.float32,
    heightfield: HeightFieldGrid,
) -> None:
    """Terrain heights under a yaw-aligned sample grid around the base.

    Each sample is the ground height relative to the base, scaled by
    ``scale`` and perturbed by uniform noise applied before scaling, matching
    the joint-velocity observation's noise ordering.
    """
    # Yaw of the [x, y, z, w] base orientation rotates the sample pattern.
    x, y, z, w = base_quat
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    bx, by, bz = base_pos[0], base_pos[1], base_pos[2]
    for index in range(offsets.shape[0]):
        ox = offsets[index, 0]
        oy = offsets[index, 1]
        wx = bx + ox * cos_yaw - oy * sin_yaw
        wy = by + ox * sin_yaw + oy * cos_yaw
        height = heightfield_lookup(heightfield, wx, wy) - bz
        if noise_amplitude > 0.0:
            height += ctx.rand.uniform_range(-noise_amplitude, noise_amplitude)
        out[index] = height * scale


@configclass(kw_only=True)
class HeightScanObsCfg(ObservationTermCfg):
    """Terrain height scan around one body's base link.

    Samples the static height field on a rectangular grid centered on the
    base link, yaw-aligned so the pattern rotates with the heading. The
    output is relative ground height scaled to roughly [-1, 1].
    """

    body: str = "robot"
    # Sample grid extents (forward, lateral) in meters, centered on the base.
    pattern_size: tuple[float, float] = (1.6, 1.0)
    # Grid spacing in meters along both pattern axes.
    pattern_resolution: float = 0.1
    # Output scaling applied after the relative height and noise.
    scale: float = 0.2
    noise: float = 0.1

    def __post_init__(self) -> None:
        if any(s <= 0.0 for s in self.pattern_size):
            raise ValueError(f"HeightScanObsCfg.pattern_size must be positive, got {self.pattern_size!r}")
        if self.pattern_resolution <= 0.0:
            raise ValueError(f"HeightScanObsCfg.pattern_resolution must be positive, got {self.pattern_resolution!r}")

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid

        ground_geom = ctx.cfg.ground_heightfield_geom
        if ground_geom is None:
            raise ValueError("HeightScanObsCfg requires a height-field ground geom.")
        body = ctx.model.bodies[self.body]
        xs = np.arange(-self.pattern_size[0] / 2.0, self.pattern_size[0] / 2.0 + 1e-9, self.pattern_resolution)
        ys = np.arange(-self.pattern_size[1] / 2.0, self.pattern_size[1] / 2.0 + 1e-9, self.pattern_resolution)
        offsets = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
        return ObsTerm(
            offsets.shape[0],
            height_scan_obs,
            LinkPositionQuery(link=body.base_link_name),
            LinkQuaternionQuery(link=body.base_link_name),
            np.ascontiguousarray(offsets),
            np.float32(self.scale),
            np.float32(self.noise),
            ground_height_grid(ctx, ground_geom),
        )
