# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator reset terms for the humanoid velocity-tracking task."""

import math

import numpy as np
from numba import njit

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ResetTerm,
    ResetTermCfg,
    kernel_data,
)
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.numba.kernel_data.map import Map
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    JointPositionWrite,
    JointVelocityWrite,
)


@kernel_data
class WalkResetParams:
    """Reset parameters, including in-kernel rough-terrain spawn sampling.

    When ``spawn_range > 0``, each lane samples its world xy from the same
    uniform range as the direct env and lifts the base above the highest
    terrain height in a +/-0.15 m 9-point grid around the spawn point
    (bilinear lookups on the static height-field grid). Otherwise the base
    spawns at the model's default pose.
    """

    default_joint_angles: np.ndarray
    init_pose: np.ndarray
    heightfield: HeightFieldGrid
    spawn_range: np.float32


@njit(inline="always")
def _spawn_ground_height(grid: HeightFieldGrid, x: float, y: float) -> float:
    """Highest terrain height over the +/-0.15 m 9-point spawn grid."""
    best = -math.inf
    for i in range(3):
        for j in range(3):
            best = max(best, heightfield_lookup(grid, x + (i - 1) * 0.15, y + (j - 1) * 0.15))
    return best


@dispatch
def reset_walk_state(ctx: ManagerContext, sim_writes: Map[np.ndarray], params: WalkResetParams) -> None:
    pose = params.init_pose
    x, y, z = pose[0], pose[1], pose[2]
    if params.spawn_range > 0.0:
        rand = ctx.rand
        x = rand.uniform_range(-params.spawn_range, params.spawn_range)
        y = rand.uniform_range(-params.spawn_range, params.spawn_range)
        z += _spawn_ground_height(params.heightfield, x, y)
    position = sim_writes["position"]
    position[0, 0] = x
    position[0, 1] = y
    position[0, 2] = z
    sim_writes["rotation"][0] = pose[3:]
    sim_writes["linear_velocity"][0, :] = 0.0
    sim_writes["angular_velocity"][0, :] = 0.0
    sim_writes["joints_position"][:] = params.default_joint_angles
    sim_writes["joints_velocity"][:] = 0.0


@configclass(kw_only=True)
class WalkStateResetCfg(ResetTermCfg):
    """Reset the floating base to the sampled spawn pose, joints to default.

    ``spawn_xy_range > 0`` samples each lane's world xy uniformly and lifts
    the base above the terrain; ``ground_geom`` names the floor geom used
    for flat-ground height lookups.
    """

    spawn_xy_range: float = 0.0
    ground_geom: str = ""

    def __call__(self, ctx) -> ResetTerm:
        from motrix_env_core.sim.model import ActuatorType

        if not self.ground_geom:
            raise ValueError("WalkStateResetCfg requires ground_geom.")
        cfg = ctx.cfg
        robot = cfg.scene.objs.robot
        base_link = robot.resolved_base_link_name
        body = ctx.model.bodies["robot"]
        joint_names = body.joint_names
        if not joint_names or len(set(joint_names)) != len(joint_names):
            raise ValueError("humanoid walk requires unique actuator target joints")
        # The dof_pos query (and every consumer aligned to it) uses the
        # key-pose declaration order; require it to match the body's
        # joint-DOF order so per-joint arrays stay aligned.
        key_pose_names = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        if joint_names != key_pose_names:
            raise ValueError(
                "robot key_pose joint order must match the body's joint order: "
                f"key_pose={key_pose_names}, body={joint_names}"
            )
        for actuator in body.actuators:
            if actuator.actuator_type is not ActuatorType.POSITION:
                raise TypeError(f"humanoid walk actuator {actuator.name!r} must be a position actuator")

        from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid

        return ResetTerm(
            reset_walk_state,
            WalkResetParams(
                default_joint_angles=body.init_joint_pos,
                init_pose=np.concatenate([body.init_base_position, body.init_base_quat]).astype(np.float32),
                heightfield=ground_height_grid(ctx, self.ground_geom),
                spawn_range=np.float32(self.spawn_xy_range),
            ),
            writes={
                "position": BodyPositionWrite((base_link,)),
                "rotation": BodyRotationWrite((base_link,)),
                "linear_velocity": BodyLinearVelocityWrite((base_link,)),
                "angular_velocity": BodyAngularVelocityWrite((base_link,)),
                "joints_position": JointPositionWrite(joint_names),
                "joints_velocity": JointVelocityWrite(joint_names),
            },
        )
