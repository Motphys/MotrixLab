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
    SharedArray,
    kernel_data,
)
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.numba.kernel_data.map import Map
from motrix_env_core.numba.manager.context import BuildContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomFrictionWrite,
    JointPositionWrite,
    JointVelocityWrite,
    LinkComWrite,
    LinkMassWrite,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.randomization import WalkRandomizationCfg


@kernel_data
class WalkResetParams:
    """Reset parameters, including in-kernel rough-terrain spawn sampling.

    When ``spawn_range > 0``, each lane samples its world xy from the same
    uniform range as the direct env and lifts the base above the highest
    terrain height in a +/-0.15 m 9-point grid around the spawn point
    (bilinear lookups on the static height-field grid). Otherwise the base
    spawns at the model's default pose. Enabled randomization items are
    resampled per lane at each reset and the backend keeps the written
    overrides between resets; when ``randomization_enabled`` is false the
    sampling block is skipped and its arrays are empty.

    Attributes:
        default_joint_angles: Nominal joint angles the reset writes, ``(A,)``.
        init_pose: Default base pose ``(x, y, z, qw, qx, qy, qz)`` the spawn
            sampling starts from.
        heightfield: Static terrain grid for spawn-height lookups.
        spawn_range: Half-width of the uniform world-xy spawn sampling; 0
            spawns at the default pose.
        randomization_enabled: Whether the dynamics-randomization fields are
            active.
        kp_default: Nominal actuator kp, ``(A,)``.
        damping_default: Nominal actuator damping, ``(A,)``.
        link_mass_default: Nominal per-link masses, ``(L,)`` in body order.
        base_link_index: Index of the base link within the body order.
        base_mass_default: Nominal base link mass in kg.
        base_com_default: Nominal base center of mass, ``(3,)``.
        friction_default: Nominal ground friction parameters, ``(3,)``.
        kp_range: ``(min, max)`` kp scale; a degenerate pair disables the item.
        damping_range: ``(min, max)`` damping scale.
        friction_range: ``(min, max)`` friction scale.
        mass_scale_range: ``(min, max)`` non-base link mass scale.
        base_mass_off_range: ``(min, max)`` additive base mass offset in kg.
        com_noise: Per-axis uniform offset width ``(x, y, z)`` in m for the
            base center of mass.
    """

    default_joint_angles: SharedArray
    init_pose: SharedArray
    heightfield: HeightFieldGrid
    spawn_range: np.float32
    randomization_enabled: bool
    kp_default: SharedArray
    damping_default: SharedArray
    link_mass_default: SharedArray
    base_link_index: np.int64
    base_mass_default: np.float32
    base_com_default: SharedArray
    friction_default: SharedArray
    kp_range: np.ndarray
    damping_range: np.ndarray
    friction_range: np.ndarray
    mass_scale_range: np.ndarray
    base_mass_off_range: np.ndarray
    com_noise: np.ndarray


@njit(inline="always")
def _sample_randomized_dynamics(ctx: ManagerContext, sim_writes: Map[np.ndarray], params: WalkResetParams) -> None:
    """Sample one lane's randomized dynamics into the reset write buffers.

    Row-scoped buffers: ``(N, A)``-style writes arrive as 1-D rows, while
    ``(N, 1, 3)``-style writes keep their leading lane dimension.
    """
    rand = ctx.rand
    if params.kp_range[1] > params.kp_range[0]:
        kp = sim_writes["kp"]
        for i in range(kp.shape[0]):
            kp[i] = params.kp_default[i] * rand.uniform_range(params.kp_range[0], params.kp_range[1])
    if params.damping_range[1] > params.damping_range[0]:
        damping = sim_writes["damping"]
        for i in range(damping.shape[0]):
            damping[i] = params.damping_default[i] * rand.uniform_range(
                params.damping_range[0], params.damping_range[1]
            )
    if params.friction_range[1] > params.friction_range[0]:
        ratio = rand.uniform_range(params.friction_range[0], params.friction_range[1])
        friction = sim_writes["friction"]
        for i in range(3):
            friction[0, i] = params.friction_default[i] * ratio
    mass_randomized = params.mass_scale_range[1] > params.mass_scale_range[0]
    base_mass_randomized = params.base_mass_off_range[1] > params.base_mass_off_range[0]
    if mass_randomized or base_mass_randomized:
        mass = sim_writes["mass"]
        for i in range(mass.shape[0]):
            if i == params.base_link_index:
                mass[i] = params.base_mass_default + rand.uniform_range(
                    params.base_mass_off_range[0], params.base_mass_off_range[1]
                )
            else:
                mass[i] = params.link_mass_default[i] * rand.uniform_range(
                    params.mass_scale_range[0], params.mass_scale_range[1]
                )
    if params.com_noise[0] > 0.0 or params.com_noise[1] > 0.0 or params.com_noise[2] > 0.0:
        com = sim_writes["com"]
        com[0, 0] = params.base_com_default[0] + rand.uniform_range(-params.com_noise[0], params.com_noise[0])
        com[0, 1] = params.base_com_default[1] + rand.uniform_range(-params.com_noise[1], params.com_noise[1])
        com[0, 2] = params.base_com_default[2] + rand.uniform_range(-params.com_noise[2], params.com_noise[2])


@njit(inline="always")
def _spawn_ground_height(grid: HeightFieldGrid, x: float, y: float) -> float:
    """Highest terrain height over the +/-0.15 m 9-point spawn grid."""
    best = -math.inf
    for i in range(3):
        for j in range(3):
            best = max(best, heightfield_lookup(grid, x + (i - 1) * 0.15, y + (j - 1) * 0.15))
    return best


@njit(inline="always")
def _write_spawn_state(
    ctx: ManagerContext, sim_writes: Map[np.ndarray], params: WalkResetParams, x: float, y: float, z: float
) -> None:
    # Homogeneous float32 tuple: z picks up float64 from the terrain-height add.
    sim_writes["position"][0, :3] = np.float32(x), np.float32(y), np.float32(z)
    sim_writes["rotation"][0] = params.init_pose[3:]
    sim_writes["linear_velocity"][0, :] = 0.0
    sim_writes["angular_velocity"][0, :] = 0.0
    sim_writes["joints_position"][:] = params.default_joint_angles
    sim_writes["joints_velocity"][:] = 0.0


@dispatch
def reset_walk_state(ctx: ManagerContext, sim_writes: Map[np.ndarray], params: WalkResetParams) -> None:
    """Spawn-state-only reset for configs without dynamics randomization."""
    pose = params.init_pose
    x, y, z = pose[0], pose[1], pose[2]
    if params.spawn_range > 0.0:
        rand = ctx.rand
        x = rand.uniform_range(-params.spawn_range, params.spawn_range)
        y = rand.uniform_range(-params.spawn_range, params.spawn_range)
        z += _spawn_ground_height(params.heightfield, x, y)
    _write_spawn_state(ctx, sim_writes, params, x, y, z)


@dispatch
def reset_walk_state_randomized(ctx: ManagerContext, sim_writes: Map[np.ndarray], params: WalkResetParams) -> None:
    """Reset plus in-kernel dynamics randomization (kp/damping/friction/mass/com)."""
    pose = params.init_pose
    x, y, z = pose[0], pose[1], pose[2]
    if params.spawn_range > 0.0:
        rand = ctx.rand
        x = rand.uniform_range(-params.spawn_range, params.spawn_range)
        y = rand.uniform_range(-params.spawn_range, params.spawn_range)
        z += _spawn_ground_height(params.heightfield, x, y)
    _sample_randomized_dynamics(ctx, sim_writes, params)
    _write_spawn_state(ctx, sim_writes, params, x, y, z)


@configclass(kw_only=True)
class WalkStateResetCfg(ResetTermCfg):
    """Reset the floating base to the sampled spawn pose, joints to default.

    ``spawn_xy_range > 0`` samples each lane's world xy uniformly and lifts
    the base above the terrain; ``ground_geom`` names the floor geom used
    for flat-ground height lookups. ``randomization`` enables reset-time
    dynamics randomization (kp/damping/friction/mass/com); the nominal values
    it needs are read from the model queries that the env config assembles
    when ``randomization.enabled`` is set.
    """

    spawn_xy_range: float = 0.0
    ground_geom: str = ""
    randomization: WalkRandomizationCfg = WalkRandomizationCfg()

    def __call__(self, ctx: BuildContext) -> ResetTerm:
        from motrix_env_core.sim.model import ActuatorType
        from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid

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

        writes = {
            "position": BodyPositionWrite((base_link,)),
            "rotation": BodyRotationWrite((base_link,)),
            "linear_velocity": BodyLinearVelocityWrite((base_link,)),
            "angular_velocity": BodyAngularVelocityWrite((base_link,)),
            "joints_position": JointPositionWrite(joint_names),
            "joints_velocity": JointVelocityWrite(joint_names),
        }
        params = dict(
            default_joint_angles=body.init_joint_pos,
            init_pose=np.concatenate([body.init_base_position, body.init_base_quat]).astype(np.float32),
            heightfield=ground_height_grid(ctx, self.ground_geom),
            spawn_range=np.float32(self.spawn_xy_range),
        )

        randomization = self.randomization
        if randomization.enabled:
            actuator_names = tuple(spec.name for spec in ctx.model.actuators)
            link_names = body.link_names
            base_index = link_names.index(body.base_link_name)
            # Keep the write-map schema stable for Numba: Map keys are part of
            # the compiled type even when a particular range is degenerate.
            writes.update(
                kp=ActuatorKpWrite(actuator_names),
                damping=ActuatorDampingWrite(actuator_names),
                friction=GeomFrictionWrite((self.ground_geom,)),
                mass=LinkMassWrite(link_names),
                com=LinkComWrite((base_link,)),
            )
            others = ctx.model.others
            link_masses = np.asarray(others["randomize_link_masses"], dtype=np.float32)
            params.update(
                randomization_enabled=True,
                kp_default=np.asarray(others["actuator_kp"], dtype=np.float32),
                damping_default=np.asarray(others["randomize_actuator_kd"], dtype=np.float32),
                link_mass_default=link_masses,
                base_link_index=np.int64(base_index),
                base_mass_default=np.float32(link_masses[base_index]),
                base_com_default=np.asarray(others["randomize_base_com"], dtype=np.float32),
                friction_default=np.asarray(others["randomize_friction_default"], dtype=np.float32),
                kp_range=np.asarray(randomization.kp_scale_range, dtype=np.float32),
                damping_range=np.asarray(randomization.damping_scale_range, dtype=np.float32),
                friction_range=np.asarray(
                    (1.0, 1.0)
                    if randomization.sliding_friction_range is None
                    else randomization.sliding_friction_range,
                    dtype=np.float32,
                ),
                mass_scale_range=np.asarray(randomization.link_mass_scale_range, dtype=np.float32),
                base_mass_off_range=np.asarray(randomization.base_mass_offset_range, dtype=np.float32),
                com_noise=np.asarray(randomization.base_com_offset_noise, dtype=np.float32),
            )
        else:
            params.update(
                randomization_enabled=False,
                kp_default=np.zeros(0, dtype=np.float32),
                damping_default=np.zeros(0, dtype=np.float32),
                link_mass_default=np.zeros(0, dtype=np.float32),
                base_link_index=np.int64(0),
                base_mass_default=np.float32(0.0),
                base_com_default=np.zeros(3, dtype=np.float32),
                friction_default=np.zeros(3, dtype=np.float32),
                kp_range=np.ones(2, dtype=np.float32),
                damping_range=np.ones(2, dtype=np.float32),
                friction_range=np.ones(2, dtype=np.float32),
                mass_scale_range=np.ones(2, dtype=np.float32),
                base_mass_off_range=np.zeros(2, dtype=np.float32),
                com_noise=np.zeros(3, dtype=np.float32),
            )

        kernel = reset_walk_state_randomized if randomization.enabled else reset_walk_state
        return ResetTerm(
            kernel,
            WalkResetParams(**params),
            writes=writes,
        )
