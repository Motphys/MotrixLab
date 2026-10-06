# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Slope-only G1 experiment using the mixed task's slope geometry."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import TrackingAngVelZRewardCfg, TrackingLinVelXyRewardCfg
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingManagerEnvCfg
from motrix_envs.locomotion.humanoid.g1_stairs import _make_mixed_terrain, make_g129dof_walk_mixed_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    FeetPhaseRewardCfg,
    PenaltyAngVelZMismatchRewardCfg,
    PenaltyFeetOriRewardCfg,
)

# Hardest row pitch in rise-per-meter: 0.35 (~19 degrees) is a steep but
# masterable ramp for position-controlled bipedal walking (holosoma caps
# its G1 terrain at 0.3). Higher caps (the mixed 1.0 = 45 degrees, 0.5, or
# 0.4) sit at or beyond the capability cliff: the terrain curriculum
# plateaus a few rows down, the top row never earns training exposure, and
# its termination rate cannot be driven below ~20%.
_SLOPE_MAX_GRADIENT = 0.35
# Central spawn pad width: 1.2 m leaves the G1 a full stabilizing step
# before the crest (0.8 m was too tight -- robots hit the flat-to-slope
# transition with unbalanced momentum and that zone hosted every fall),
# while still too small to host paced camping loops like the mixed 2.0 m.
_SLOPE_PLATFORM_WIDTH = 1.2


@registry.envcfg("g1-walk-slope")
def make_g129dof_walk_slope_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Train normal/inverted slope descent and ascent with a per-lane curriculum.

    Columns 0/1 are normal (spawn on the peak, descend) and inverted (spawn
    in the pit, ascend) pyramids. Rows keep the mixed terrain's difficulty
    gradient scaled so the hardest row pitches at ``_SLOPE_MAX_GRADIENT``.
    No adaptive column reassignment or privileged actor linear velocity.

    The mixed-task defaults make camping on the flat spawn pad the rational
    strategy: rewards are dt-scaled, so alive alone (10 -> 0.2/step)
    out-earns best-case tracking (12 * 0.02), and the disabled gait term
    never bootstraps stepping. The rebalance here -- tracking 30/3, gait
    phase re-enabled, a yaw-rate mismatch penalty -- makes walking the
    income-maximizing behavior; sustained rarely-turning commands and a
    trimmed lateral range keep hard-row falls off the fall-dominated
    regimes; the relaxed promotion thresholds let the curriculum expose the
    top rows instead of wedging just above the start level.
    """
    cfg = make_g129dof_walk_mixed_cfg()
    return replace(
        cfg,
        scene=replace(
            cfg.scene,
            assets=replace(
                cfg.scene.assets,
                terrain=_make_mixed_terrain(
                    ("slope", "slope_inv"),
                    slope_scale=_SLOPE_MAX_GRADIENT,
                    slope_platform_width=_SLOPE_PLATFORM_WIDTH,
                ),
            ),
        ),
        rewards=replace(
            cfg.rewards,
            tracking_lin_vel=TrackingLinVelXyRewardCfg(command_name="walk", sigma=0.25, weight=30.0),
            tracking_ang_vel=TrackingAngVelZRewardCfg(command_name="walk", sigma=0.25, weight=3.0),
            feet_phase=FeetPhaseRewardCfg(
                sole_l_site="left_foot_contact_point",
                sole_r_site="right_foot_contact_point",
                weight=5.0,
            ),
            # Direct anti-spin term: rhythmic stepping while spinning on the
            # spawn platform earns alive + gait income at zero risk; the
            # exponential tracking term has no gradient that far off-command.
            penalty_ang_vel_z_mismatch=PenaltyAngVelZMismatchRewardCfg(weight=-3.0),
            # Keep the soles flat on the terrain: without this term nothing
            # constrains foot attitude (the mixed task disables it, and the
            # gait-phase term only tracks one center point), so the policy
            # walks heel-first with the toes pitched up. The terrain-relative
            # variant judges flatness against the local slope normal instead
            # of world horizontal, which is the right reference on ramps.
            # Measured stance attitude error stayed ~40 deg at weight -2 and
            # the soles flattened but falls exploded at -8 without gating;
            # settled-stance gating (landing-only exemption) runs at -4 --
            # hotter weights exploit the exemption by micro-bouncing.
            penalty_feet_ori=PenaltyFeetOriRewardCfg(
                terrain_relative=True,
                weight=-4.0,
                swing_weight_scale=0.1,
                sole_l_site="left_foot_contact_point",
                sole_r_site="right_foot_contact_point",
                # Deadband experiments both failed and stay off: a global
                # 12 deg band slackened every direction to 24-32 deg stance
                # pitch, and an 8 deg downhill-only band still degraded the
                # median to ~25 deg without stabilizing descent braking.
            ),
        ),
        commands=replace(
            cfg.commands,
            walk=replace(
                cfg.commands.walk,
                terrain_sampling_enabled=False,
                terrain_balanced_columns=True,
                # The mixed default (0.7) demands near-perfect command-aligned
                # progress, which pins the curriculum a row above the start:
                # half-speed walkers never promote and never see harder rows.
                terrain_move_up_ratio=0.55,
                # Credit a 3 m crossing even when the episode fails
                # afterwards: demanding a surviving half-tile (4 m) at every
                # row wedges the curriculum at the top, where pit spawns that
                # camp never promote and never practice the hardest climb;
                # early failures still demote.
                terrain_require_survival_for_promotion=False,
                terrain_promote_distance_ratio=0.375,
                # Trim linear commands from the mixed +-1.0 to +-0.6 m/s:
                # near-limit descents on steep ramps are fall-dominated
                # (braking margins shrink with pitch), and an all-slope task
                # has no flat tile to absorb top speed.
                vel_limit=((-0.6, -0.25, -0.5), (0.6, 0.25, 0.5)),
                # Sustained, rarely-turning commands: every pivot executed on
                # the 1.2 m spawn pad risks a fall, and a slope traversal
                # wants a held heading, not frequent reversals. Lateral
                # commands are trimmed hard: side-slope walking on a 0.35
                # ramp is far past reliable tracking and hosted many falls.
                heading_prob=0.1,
                resampling_time_range=(6.0, 10.0),
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
                # Keep spawns well inside the platform's flat pad so the
                # first steps plant on level ground, not on the crest.
                tile_xy_offset_range=0.15,
            ),
        ),
    )


registry.env("g1-walk-slope")(ManagerEnv)
