# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Measured pelvis IMU + named waist FK matches independent simulator truth."""

from dataclasses import replace

import numpy as np
import pytest
from motrix_robots.unitree import UnitreeG129Dof

from motrix_deploy.contracts import RobotState
from motrix_deploy.runtime.config import (
    SensorBindings,
    SimulationRuntimeConfig,
)
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtMotion, G1WbtPolicyProcessor
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.config.sim import SimCfg
from motrix_envs.deploy.g1_wbt import build_g1_wbt_profile


@configclass
class _FlatG1Objs(SceneObjsCfg):
    floor: FlatTerrainCfg = FlatTerrainCfg()


@pytest.fixture(scope="module")
def profile():
    return build_g1_wbt_profile("g1-wbt-dance")


def _profile_motion(profile):
    """Reload the compiled motion clip from the profile's NPZ payload."""
    import io

    payload = next(iter(profile.payloads.values()))
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return G1WbtMotion(arrays["joint_pos"], arrays["joint_vel"], arrays["reference_quaternion_xyzw"])


def _random_quaternion(rng):
    quaternion = rng.normal(size=4)
    return (quaternion / np.linalg.norm(quaternion)).astype(np.float32)


def _rotation(quaternion):
    """Independent Rodrigues construction, insensitive to quaternion sign."""
    vector = np.asarray(quaternion[:3], dtype=np.float64)
    scalar = float(quaternion[3])
    x, y, z = vector
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    norm_squared = np.dot(vector, vector) + scalar**2
    return np.eye(3) + (2 / norm_squared) * (scalar * skew + skew @ skew)


def _waist_rotation(yaw, roll, pitch):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return rz @ rx @ ry


def _state(robot, quaternion, waist):
    joints = robot.default_joint_position.copy()
    for axis, angle in zip(("yaw", "roll", "pitch"), waist):
        joints[robot.joint_names.index(f"waist_{axis}_joint")] = angle
    return RobotState(
        sample_time_ns=0,
        receive_time_ns=0,
        joint_position=joints,
        joint_velocity=np.zeros(robot.joint_count, np.float32),
        base_orientation_xyzw=quaternion,
        base_angular_velocity=np.zeros(3, np.float32),
        base_linear_acceleration=np.zeros(3, np.float32),
    )


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize(
    "waist",
    [(0, 0, 0), (0.7, 0, 0), (0, -0.3, 0), (0, 0, 0.4), (0.7, -0.3, 0.4), (-0.8, 0.2, -0.3)],
)
def test_measured_fk_axis_order_base_rotation_and_named_joint_mapping(profile, permuted, waist):
    robot = profile.robot
    if permuted:
        order = np.random.default_rng(13).permutation(robot.joint_count)
        robot = replace(
            robot,
            joint_names=tuple(robot.joint_names[index] for index in order),
            default_joint_position=robot.default_joint_position[order],
            position_lower=robot.position_lower[order],
            position_upper=robot.position_upper[order],
            torque_limit=robot.torque_limit[order],
        )
    task = G1WbtPolicyProcessor(profile.task, robot, _profile_motion(profile))
    base = _random_quaternion(np.random.default_rng(19))
    state = _state(robot, base, waist)
    expected = _rotation(base) @ _waist_rotation(*waist)
    np.testing.assert_allclose(_rotation(task.torso_orientation(state)), expected, atol=1e-6, rtol=0)
    state.base_orientation_xyzw = -base
    np.testing.assert_allclose(_rotation(task.torso_orientation(state)), expected, atol=1e-6, rtol=0)


def test_actor_orientation_uses_measured_state(profile):
    task = G1WbtPolicyProcessor(profile.task, profile.robot, motion := _profile_motion(profile))
    state = _state(profile.robot, _random_quaternion(np.random.default_rng(37)), (0.8, -0.25, 0.3))
    context = ControlContext(0, 0.0, None, dt_s=0.02)
    observation = task.build_observation(state, context)
    expected = (_rotation(task.torso_orientation(state)).T @ _rotation(motion.reference_quaternion_xyzw[0]))[
        :2
    ].reshape(-1)
    np.testing.assert_allclose(observation[58:64], expected, atol=1e-6, rtol=0)


def _set_fk_pose(runtime, backend, orientation, joints):
    """Test-only pose injection through simulator APIs, not deployment contracts."""
    robot = runtime.scene.objs.robot
    if backend == "mujoco":
        model, data = runtime.model, runtime.data
        root = model.body(robot.resolved_base_link_name).id
        address = model.jnt_qposadr[model.body_jntadr[root]]
        data.qpos[address : address + 3] = [0.2, -0.1, 2.0]
        data.qpos[address + 3 : address + 7] = orientation[[3, 0, 1, 2]]
        for name, position in zip(runtime.robot.spec.joint_names, joints):
            data.qpos[model.jnt_qposadr[model.joint(robot.resolve_name(name)).id]] = position
        runtime.mj.mj_forward(model, data)
    else:
        import motrixsim as mtx

        program = runtime.model.compile_write(
            {
                "position": mtx.write.BodyPosition((robot.resolved_base_link_name,)),
                "orientation": mtx.write.BodyRotation((robot.resolved_base_link_name,)),
                "joints": mtx.write.BodyJointPosition(
                    tuple(robot.resolve_name(name) for name in runtime.robot.spec.joint_names)
                ),
            },
            forward_kinematic=True,
        ).allocate(runtime.data)
        program["position"][:] = [0.2, -0.1, 2.0]
        program["orientation"][:] = orientation
        program["joints"][:] = joints
        program.execute(runtime.data)


@pytest.mark.parametrize("backend,solver_iterations", [("mujoco", 100), ("motrixsim", 3)])
def test_measured_fk_matches_randomized_simulator_world_orientation(profile, backend, solver_iterations):
    pytest.importorskip(backend)
    config = SimulationRuntimeConfig(
        scene=SceneCfg(objs=_FlatG1Objs(robot=UnitreeG129Dof())),
        physics=SimCfg(dt=0.005, solver_iterations=solver_iterations),
        sensor_bindings=SensorBindings(base_angular_velocity="base_gyro", base_linear_acceleration="base_accel"),
    )
    runtime = create_simulation_runtime(backend, config)
    task = G1WbtPolicyProcessor(profile.task, profile.robot, motion := _profile_motion(profile))
    rng = np.random.default_rng(2026)
    try:
        profile.robot.validate_compatible(runtime.robot.spec)
        runtime.open()
        for case in range(64):
            joints = rng.uniform(profile.robot.position_lower, profile.robot.position_upper).astype(np.float32)
            # All three waist angles are independently sampled, not yaw-only probes.
            _set_fk_pose(runtime, backend, _random_quaternion(rng), joints)
            state = runtime.robot.read_state(0.1)
            np.testing.assert_allclose(state.joint_position, joints, atol=1e-6, rtol=0)
            # Direct simulator query is a test oracle, not public deployment state.
            name = runtime.scene.objs.robot.resolve_name("torso_link")
            if backend == "mujoco":
                truth = runtime.data.xmat[runtime.model.body(name).id].reshape(3, 3).copy()
            else:
                import motrixsim as mtx

                query = runtime.model.compile_query({"truth": mtx.query.LinkRotation((name,))}).allocate(runtime.data)
                query.execute(runtime.data)
                truth = _rotation(query["truth"].reshape(4))
            np.testing.assert_allclose(
                _rotation(task.torso_orientation(state)), truth, atol=1e-5, rtol=0, err_msg=f"{backend} case {case}"
            )
            # Neither orientation nor position body queries are needed by actor I/O.
            observation = task.build_observation(state, ControlContext(0, 0.0, None, dt_s=0.02))
            expected = (truth.T @ _rotation(motion.reference_quaternion_xyzw[0]))[:2].reshape(-1)
            np.testing.assert_allclose(observation[58:64], expected, atol=1e-5, rtol=0)
    finally:
        runtime.close()
