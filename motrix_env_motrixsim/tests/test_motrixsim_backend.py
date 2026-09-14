# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim backend: scene compilation, SimCfg plumbing, and reset behavior."""

import motrixsim as mtx
import numpy as np
import pytest

from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import SceneCfg, SceneCompiler
from motrix_env_core.sim import (
    ActuatorKdQuery,
    ActuatorKpQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionLimitsQuery,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionLimitsQuery,
    DofPositionQuery,
    DofPositionWrite,
    DofVelocityQuery,
    GeomSpecsQuery,
    JointPositionQuery,
    JointPositionWrite,
    JointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SitePositionQuery,
)
from motrix_env_core.sim.backend import SimModel
from motrix_env_core.sim.registry import create_sim_backend, list_sim_backends
from motrix_env_core.sim.write import BodyJointVelocityWrite, DofVelocityWrite, JointVelocityWrite, MocapPoseWrite
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.runtime import MotrixSimBackend


def _make_backend(scene: SceneCfg, sim: SimCfg, *, num_envs: int) -> MotrixSimBackend:
    """Construction compiles the scene inside the backend."""
    return MotrixSimBackend(scene, sim, num_envs)


def _resolve_core(scene: SceneCfg, sim: SimCfg) -> SimModel:
    return MotrixSimBackend(scene, sim, 1).model_query_compiler.compile({})


def test_scene_compiler_is_an_abstract_backend_boundary():
    with pytest.raises(TypeError):
        SceneCompiler()

    assert isinstance(MotrixSimSceneCompiler(), SceneCompiler)
    assert MotrixSimSceneCompiler().compile(SceneCfg(), SimCfg()) is not None


def test_motrixsim_backend_is_registered():
    # Test modules register their own fake backends into the same registry.
    assert set(list_sim_backends()) >= {"motrixsim"}

    with pytest.raises(ValueError, match="Unknown sim backend 'unknown'"):
        create_sim_backend("unknown")


def test_sim_cfg_configures_msd_world_before_build():
    cfg = SimCfg(
        dt=0.005,
        solver_iterations=3,
        solver_tolerance=1e-4,
        gravity=(0.0, 0.0, -3.0),
    )

    model = MotrixSimSceneCompiler().compile(SceneCfg(), cfg)
    options = model.options

    assert options.timestep == pytest.approx(0.005)
    assert options.max_iterations == 3
    assert options.solver_tolerance == pytest.approx(1e-4)
    assert options.gravity.tolist() == pytest.approx([0.0, 0.0, -3.0])


class _ResettableSceneData:
    def __init__(self) -> None:
        self.reset_models = []

    def reset(self, model) -> None:
        self.reset_models.append(model)


def test_motrixsim_runtime_reset_restores_default_state():
    backend = _make_backend(SceneCfg(), SimCfg(), num_envs=2)
    model = _resolve_core(SceneCfg(), SimCfg())

    assert backend.num_dof_pos == 0
    assert backend.num_dof_vel == 0
    assert backend.num_actuators == 0
    assert model.actuators == ()

    # A scene without DOF resets to backend defaults without any value channels.
    backend.write_compiler.compile(
        {"state_position": DofPositionWrite(), "state_velocity": DofVelocityWrite()}, reset=True
    ).execute(np.asarray([0, 1], dtype=np.int64))


def test_body_joint_position_limits_follow_body_joint_dof_order():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    body_name = cfg.scene.objs.robot.resolved_base_link_name
    model = backend.model_query_compiler.compile({"limits": BodyJointPositionLimitsQuery(body=body_name)})
    lower, upper = model.others["limits"]
    body = backend._model.get_body(body_name)

    expected = np.concatenate(
        [np.asarray(joint.range, dtype=np.float32).reshape(-1, 2) for joint in body.joints],
        axis=0,
    )
    np.testing.assert_array_equal(lower, expected[:, 0])
    np.testing.assert_array_equal(upper, expected[:, 1])
    assert lower.shape == (body.num_joint_dof_pos,)


def test_dof_position_limits_follow_global_dof_position_order():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-humanoid-walk", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    model = backend.model_query_compiler.compile({"limits": DofPositionLimitsQuery()})
    lower, upper = model.others["limits"]

    assert lower.shape == upper.shape == (backend.num_dof_pos,)
    assert np.all(np.isneginf(lower[:7]))
    assert np.all(np.isposinf(upper[:7]))
    for joint in backend._model.joints:
        limits = np.asarray(joint.range, dtype=np.float32).reshape(-1, 2)
        position_slice = slice(joint.dof_pos_index, joint.dof_pos_index + joint.num_dof_pos)
        np.testing.assert_array_equal(lower[position_slice], limits[:, 0])
        np.testing.assert_array_equal(upper[position_slice], limits[:, 1])


def test_geom_specs_include_only_declared_names_in_order():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-turn-easy", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    model = backend.model_query_compiler.compile({"geoms": GeomSpecsQuery(names=("target_geom", "cap1"))})

    assert tuple(model.others["geoms"]) == ("target_geom", "cap1")
    assert len(model.others["geoms"]["target_geom"].local_pose) == 7
    assert model.others["geoms"]["cap1"].size

    with pytest.raises(KeyError, match="Unknown geom 'missing'"):
        backend.model_query_compiler.compile({"geoms": GeomSpecsQuery(names=("missing",))})


def test_actuator_params_support_declared_names_or_full_model_order():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("go2-walk-flat", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    all_names = tuple(actuator.name for actuator in backend._model.actuators)
    selected_names = (all_names[-1], all_names[0])
    model = backend.model_query_compiler.compile(
        {
            "all_kp": ActuatorKpQuery(),
            "selected_kp": ActuatorKpQuery(names=selected_names),
            "selected_kd": ActuatorKdQuery(names=selected_names),
        }
    )

    np.testing.assert_array_equal(
        model.others["all_kp"],
        np.asarray([actuator.kp for actuator in backend._model.actuators], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        model.others["selected_kp"],
        np.asarray([backend._model.actuators[-1].kp, backend._model.actuators[0].kp], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        model.others["selected_kd"],
        np.asarray([backend._model.actuators[-1].kd, backend._model.actuators[0].kd], dtype=np.float32),
    )

    with pytest.raises(KeyError, match="Unknown actuator 'missing'"):
        backend.model_query_compiler.compile({"kp": ActuatorKpQuery(names=("missing",))})


def test_named_joint_queries_follow_declared_order():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    program = backend.compile_reads(
        {
            "all_pos": DofPositionQuery(),
            "all_vel": DofVelocityQuery(),
            "joint_pos": JointPositionQuery(joints=("hinge", "proximal", "distal")),
            "joint_vel": JointVelocityQuery(joints=("hinge", "proximal", "distal")),
        }
    )
    program.execute()

    position_indices = [backend._model.get_joint(name).dof_pos_index for name in ("hinge", "proximal", "distal")]
    velocity_indices = [backend._model.get_joint(name).dof_vel_index for name in ("hinge", "proximal", "distal")]
    np.testing.assert_array_equal(program["joint_pos"], program["all_pos"][:, position_indices])
    np.testing.assert_array_equal(program["joint_vel"], program["all_vel"][:, velocity_indices])


def test_named_joint_reset_follows_declared_order_and_partial_selection():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    joints = ("hinge", "proximal", "distal")
    reset = backend.write_compiler.compile(
        {"joints_position": JointPositionWrite(joints), "joints_velocity": JointVelocityWrite(joints)}, reset=True
    )
    read = backend.compile_reads(
        {
            "position": JointPositionQuery(joints=joints),
            "velocity": JointVelocityQuery(joints=joints),
        }
    )
    reset.buffer("joints_position")[1] = [0.1, 0.2, 0.3]
    reset.buffer("joints_velocity")[1] = [0.4, 0.5, 0.6]

    reset.execute(np.asarray([1], dtype=np.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(read["velocity"][1], [0.4, 0.5, 0.6])
    assert not np.allclose(read["position"][0], read["position"][1])


def test_full_dof_reset_preserves_unsorted_environment_id_values():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 3)
    reset = backend.write_compiler.compile(
        {"state_position": DofPositionWrite(), "state_velocity": DofVelocityWrite()}, reset=True
    )
    read = backend.compile_reads({"position": DofPositionQuery()})
    position = reset.buffer("state_position")
    position[2] = [0.2, 0.3, 0.4]
    position[0] = [-0.2, -0.3, -0.4]

    reset.execute(np.asarray([2, 0], dtype=np.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][2], [0.2, 0.3, 0.4])
    np.testing.assert_allclose(read["position"][0], [-0.2, -0.3, -0.4])


def test_body_dof_reset_matches_body_query_layout():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    body = cfg.scene.objs.robot.resolved_base_link_name
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    reset = backend.write_compiler.compile(
        {"body_position": BodyJointPositionWrite(body), "body_velocity": BodyJointVelocityWrite(body)}, reset=True
    )
    read = backend.compile_reads(
        {
            "position": BodyJointPositionQuery(body=body),
            "velocity": BodyJointVelocityQuery(body=body),
        }
    )
    position = reset.buffer("body_position")
    velocity = reset.buffer("body_velocity")
    position[0] = np.linspace(-0.2, 0.2, position.shape[1], dtype=np.float32)
    velocity[0] = np.linspace(0.3, -0.3, velocity.shape[1], dtype=np.float32)

    reset.execute(np.asarray([0], dtype=np.int64))
    read.execute(np.asarray([0], dtype=np.int64))

    np.testing.assert_allclose(read["position"][0], position[0])
    np.testing.assert_allclose(read["velocity"][0], velocity[0])


def test_reset_compilation_rejects_conflicting_targets():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    with pytest.raises(ValueError, match="conflict"):
        backend.write_compiler.compile(
            {
                "all_position": DofPositionWrite(),
                "all_velocity": DofVelocityWrite(),
                "joint_position": JointPositionWrite(("hinge",)),
                "joint_velocity": JointVelocityWrite(("hinge",)),
            },
            reset=True,
        )


def test_body_state_reset_is_visible_to_link_queries():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-humanoid-walk", mode="play")
    body = "torso"
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    reset = backend.write_compiler.compile(
        {
            "body_position": BodyPositionWrite((body,)),
            "body_rotation": BodyRotationWrite((body,)),
            "body_linear_velocity": BodyLinearVelocityWrite((body,)),
            "body_angular_velocity": BodyAngularVelocityWrite((body,)),
        },
        reset=True,
    )
    read = backend.compile_reads(
        {
            "position": LinkPositionQuery(link=body),
            "rotation": LinkQuaternionQuery(link=body),
            "linear_velocity": LinkLinearVelocityQuery(link=body),
            "angular_velocity": LinkAngularVelocityQuery(link=body),
        }
    )
    assert reset.buffer("body_position").shape == (1, 1, 3)
    assert reset.buffer("body_rotation").shape == (1, 1, 4)
    reset.buffer("body_position")[0, 0] = [1.0, 2.0, 3.0]
    reset.buffer("body_rotation")[0, 0] = [0.0, 0.0, 0.0, 1.0]
    reset.buffer("body_linear_velocity")[0, 0] = [0.1, 0.2, 0.3]
    reset.buffer("body_angular_velocity")[0, 0] = [0.4, 0.5, 0.6]

    reset.execute(np.asarray([0], dtype=np.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(read["rotation"][0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(read["linear_velocity"][0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(read["angular_velocity"][0], [0.4, 0.5, 0.6])


def test_mixed_reset_program_applies_native_and_numpy_writes_together():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-humanoid-walk", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    body = "torso"
    reset = backend.write_compiler.compile(
        {
            "base_position": BodyPositionWrite((body,)),
            "base_rotation": BodyRotationWrite((body,)),
            "joint_position": JointPositionWrite(("abdomen_z",)),
            "joint_velocity": JointVelocityWrite(("abdomen_z",)),
        },
        reset=True,
    )
    read = backend.compile_reads(
        {
            "position": LinkPositionQuery(link=body),
            "joint_position": JointPositionQuery(joints=("abdomen_z",)),
            "joint_velocity": JointVelocityQuery(joints=("abdomen_z",)),
        }
    )
    assert reset.buffer("base_position").shape == (2, 1, 3)
    reset.buffer("base_position")[1, 0] = [0.5, 0.0, 1.0]
    reset.buffer("joint_position")[1] = [0.25]
    reset.buffer("joint_velocity")[1] = [-0.75]

    reset.execute(np.asarray([1], dtype=np.int64))
    read.execute()

    # Native reset applies first, then the declared writes land on top.
    np.testing.assert_allclose(read["position"][1], [0.5, 0.0, 1.0])
    np.testing.assert_allclose(read["joint_position"][1], [0.25])
    np.testing.assert_allclose(read["joint_velocity"][1], [-0.75])
    # The unselected environment stays at the model initial state.
    assert not np.allclose(read["joint_position"][0], [0.25])


def test_dof_channel_writes_without_reset_land_in_sim_state():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    writes = backend.write_compiler.compile(
        {"joint_position": JointPositionWrite(("hinge",)), "joint_velocity": JointVelocityWrite(("hinge",))}
    )
    read = backend.compile_reads(
        {
            "joint_position": JointPositionQuery(joints=("hinge",)),
            "joint_velocity": JointVelocityQuery(joints=("hinge",)),
        }
    )
    writes.buffer("joint_position")[0] = [0.3]
    writes.buffer("joint_velocity")[0] = [-0.6]

    writes.execute(np.asarray([0], dtype=np.int64))
    read.execute(np.asarray([0], dtype=np.int64))

    np.testing.assert_allclose(read["joint_position"][0], [0.3])
    np.testing.assert_allclose(read["joint_velocity"][0], [-0.6])
    np.testing.assert_array_equal(read["joint_velocity"][1], [0.0])


def test_mocap_pose_writes_keep_the_flat_pose_layout_and_reach_the_sim():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-manipulator-bring-ball", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    writes = backend.write_compiler.compile({"target": MocapPoseWrite(("target_ball",))})
    read = backend.compile_reads({"pos": SitePositionQuery(site="target_ball")})

    buffer = writes.buffer("target")
    assert buffer.shape == (2, 1, 7)
    buffer[1, 0] = [0.11, 0.22, 0.33, 0.0, 0.0, 0.0, 1.0]

    writes.execute(np.asarray([1], dtype=np.int64))
    read.execute(np.asarray([1], dtype=np.int64))

    np.testing.assert_allclose(read["pos"][1], [0.11, 0.22, 0.33])
    # The other environment keeps its own mocap pose.
    np.testing.assert_allclose(read["pos"][0], [0.0, 0.0, 0.0])


def test_named_joint_queries_reject_unknown_joints():
    import motrix_envs  # noqa: F401
    from motrix_env_core import registry

    finger_cfg = registry.make_env_config("dm-finger-spin", mode="play")
    finger = MotrixSimBackend(finger_cfg.scene, finger_cfg.sim, 1)
    with pytest.raises(KeyError, match="unknown joint"):
        finger.compile_reads({"joint_pos": JointPositionQuery(joints=("missing",))})


def test_motrixsim_reset_program_validates_environment_ids():
    backend = _make_backend(SceneCfg(), SimCfg(), num_envs=2)
    program = backend.write_compiler.compile(
        {"state_position": DofPositionWrite(), "state_velocity": DofVelocityWrite()}, reset=True
    )

    with pytest.raises(TypeError, match="int64 ndarray"):
        program.execute(np.asarray([0], dtype=np.int32))
    with pytest.raises(IndexError, match="out of range"):
        program.execute(np.asarray([2], dtype=np.int64))
    with pytest.raises(ValueError, match="duplicates"):
        program.execute(np.asarray([0, 0], dtype=np.int64))


def test_sim_cfg_preserves_unspecified_solver_options():
    world = mtx.msd.World()
    world.simulate_option.constraint_solver_iterations = 7
    world.simulate_option.constraint_solver_tolerance = 2e-4

    MotrixSimSceneCompiler().configure_world(world, SimCfg(dt=0.01))
    options = world.build().options

    assert options.max_iterations == 7
    assert options.solver_tolerance == pytest.approx(2e-4)
