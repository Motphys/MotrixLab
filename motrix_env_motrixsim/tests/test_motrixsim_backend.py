# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim backend: scene compilation, SimCfg plumbing, and reset behavior."""

import motrixsim as mtx
import numpy as np
import pytest
import torch

from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import SceneCfg, SceneCompiler
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    ActuatorKdQuery,
    ActuatorKpQuery,
    BatchLinkNetContactForceQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionLimitsQuery,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyLinkNetContactForceQuery,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionLimitsQuery,
    DofPositionQuery,
    DofVelocityQuery,
    GeomSpecsQuery,
    JointPositionQuery,
    JointPositionWrite,
    JointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkNetContactForceQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.model import KinematicTreeQuery, SimModel
from motrix_env_core.sim.registry import create_sim_backend, list_sim_backends
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite, JointQuaternionWrite, JointVelocityWrite
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_env_motrixsim.runtime import MotrixSimBackend


def _make_backend(scene: SceneCfg, sim: SimCfg, *, num_envs: int) -> MotrixSimBackend:
    """Construction compiles the scene inside the backend."""
    return MotrixSimBackend(scene, sim, num_envs)


def _resolve_core(scene: SceneCfg, sim: SimCfg) -> SimModel:
    return MotrixSimBackend(scene, sim, 1).model_compiler.compile(scene, {})


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


def test_motrixsim_factory_defaults_to_cpu_and_rejects_other_device_types():
    assert create_sim_backend() is MotrixSimBackend
    with pytest.raises(ValueError, match="does not support simulation on meta"):
        create_sim_backend("motrixsim", "meta")


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
    program = backend.write_compiler.compile({"ctrl": CtrlTargetsWrite()}, reset=True)
    program.execute(torch.tensor([0, 1], dtype=torch.int64))


def test_body_joint_position_limits_follow_body_joint_dof_order():
    from motrix_env_core import registry

    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    body_name = cfg.scene.objs.robot.resolved_base_link_name
    model = backend.model_compiler.compile(cfg.scene, {"limits": BodyJointPositionLimitsQuery(body=body_name)})
    lower, upper = model.others["limits"]
    body = backend._model.get_body(body_name)

    expected = np.concatenate(
        [np.asarray(joint.range, dtype=np.float32).reshape(-1, 2) for joint in body.joints],
        axis=0,
    )
    np.testing.assert_array_equal(lower, expected[:, 0])
    np.testing.assert_array_equal(upper, expected[:, 1])
    assert lower.shape == (body.num_joint_dof_pos,)


def test_bodies_bake_init_snapshot():
    from motrix_env_core import registry

    cfg = registry.make_env_config("g1-wbt-dance", mode="play")
    robot = cfg.scene.objs.robot
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    model = backend.model_compiler.compile(cfg.scene, {})

    body = model.bodies["robot"]
    engine_body = backend._model.get_body(robot.resolved_base_link_name)
    assert body.base_link_name == robot.resolved_base_link_name
    assert body.joint_names == tuple(joint.name for joint in engine_body.joints)
    assert body.link_names == tuple(link.name for link in engine_body.links)
    # Single-robot scene: the body scope covers the full actuator set.
    assert body.actuators == model.actuators

    # The init joint angles are the designated key pose permuted into the
    # engine body joint order.
    by_joint = dict(zip(robot.key_pose.joint_names, robot.key_pose.poses[robot.init_key_pose]))
    np.testing.assert_allclose(body.init_joint_pos, [by_joint[name] for name in body.joint_names])
    assert len(body.joint_pos_limits[0]) == len(body.joint_names)

    # FK snapshot: link-aligned arrays and unit quaternions at the default placement.
    assert body.init_link_positions.shape == (len(body.link_names), 3)
    assert body.init_link_quats.shape == (len(body.link_names), 4)
    np.testing.assert_allclose(np.linalg.norm(body.init_link_quats, axis=-1), 1.0, rtol=1e-5)
    assert body.init_base_position.shape == (3,)
    assert body.init_base_quat.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(body.init_base_quat), 1.0, rtol=1e-5)


def test_bodies_stay_empty_for_scene_without_body_objs():
    model = _resolve_core(SceneCfg(), SimCfg())

    assert model.bodies == {}


def test_dof_position_limits_follow_global_dof_position_order():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-humanoid-walk", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    model = backend.model_compiler.compile(cfg.scene, {"limits": DofPositionLimitsQuery()})
    lower, upper = model.others["limits"]

    assert lower.shape == upper.shape == (backend.num_dof_pos,)
    assert bool(torch.isneginf(lower[:7]).all())
    assert bool(torch.isposinf(upper[:7]).all())
    for joint in backend._model.joints:
        limits = np.asarray(joint.range, dtype=np.float32).reshape(-1, 2)
        position_slice = slice(joint.dof_pos_index, joint.dof_pos_index + joint.num_dof_pos)
        np.testing.assert_array_equal(lower[position_slice], limits[:, 0])
        np.testing.assert_array_equal(upper[position_slice], limits[:, 1])


def test_geom_specs_include_only_declared_names_in_order():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-turn-easy", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    model = backend.model_compiler.compile(cfg.scene, {"geoms": GeomSpecsQuery(names=("target_geom", "cap1"))})

    assert tuple(model.others["geoms"]) == ("target_geom", "cap1")
    assert len(model.others["geoms"]["target_geom"].local_pose) == 7
    assert model.others["geoms"]["cap1"].size

    with pytest.raises(KeyError, match="Unknown geom 'missing'"):
        backend.model_compiler.compile(cfg.scene, {"geoms": GeomSpecsQuery(names=("missing",))})


def test_actuator_params_support_declared_names_or_full_model_order():
    from motrix_env_core import registry

    cfg = registry.make_env_config("go2-walk-flat", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    all_names = tuple(actuator.name for actuator in backend._model.actuators)
    selected_names = (all_names[-1], all_names[0])
    model = backend.model_compiler.compile(
        cfg.scene,
        {
            "all_kp": ActuatorKpQuery(),
            "selected_kp": ActuatorKpQuery(names=selected_names),
            "selected_kd": ActuatorKdQuery(names=selected_names),
        },
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
        backend.model_compiler.compile(cfg.scene, {"kp": ActuatorKpQuery(names=("missing",))})


def test_named_joint_queries_follow_declared_order():
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
    reset.buffer("joints_position")[1] = torch.tensor([0.1, 0.2, 0.3])
    reset.buffer("joints_velocity")[1] = torch.tensor([0.4, 0.5, 0.6])

    reset.execute(torch.tensor([1], dtype=torch.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][1], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(read["velocity"][1], [0.4, 0.5, 0.6])
    assert not np.allclose(read["position"][0], read["position"][1])


def test_full_dof_reset_preserves_unsorted_environment_id_values():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 3)
    reset = backend.write_compiler.compile(
        {"joint_position": JointPositionWrite(("hinge",)), "joint_velocity": JointVelocityWrite(("hinge",))},
        reset=True,
    )
    read = backend.compile_reads({"position": DofPositionQuery()})
    hinge_dof = backend._model.get_joint("hinge").dof_pos_index
    position = reset.buffer("joint_position")
    position[2] = torch.tensor([0.4])
    position[0] = torch.tensor([-0.4])

    reset.execute(torch.tensor([2, 0], dtype=torch.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][2, hinge_dof], [0.4])
    np.testing.assert_allclose(read["position"][0, hinge_dof], [-0.4])


def test_body_dof_reset_matches_body_query_layout():
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
    position[0] = torch.linspace(-0.2, 0.2, position.shape[1], dtype=torch.float32)
    velocity[0] = torch.linspace(0.3, -0.3, velocity.shape[1], dtype=torch.float32)

    reset.execute(torch.tensor([0], dtype=torch.int64))
    read.execute(torch.tensor([0], dtype=torch.int64))

    np.testing.assert_allclose(read["position"][0], position[0])
    np.testing.assert_allclose(read["velocity"][0], velocity[0])


def test_reset_compilation_rejects_conflicting_targets():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    with pytest.raises(ValueError, match="conflict"):
        backend.write_compiler.compile(
            {
                "joint_position": JointPositionWrite(("hinge",)),
                "joint_position_again": JointPositionWrite(("hinge",)),
            },
            reset=True,
        )


def test_body_state_reset_is_visible_to_link_queries():
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
    reset.buffer("body_position")[0, 0] = torch.tensor([1.0, 2.0, 3.0])
    reset.buffer("body_rotation")[0, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    reset.buffer("body_linear_velocity")[0, 0] = torch.tensor([0.1, 0.2, 0.3])
    reset.buffer("body_angular_velocity")[0, 0] = torch.tensor([0.4, 0.5, 0.6])

    reset.execute(torch.tensor([0], dtype=torch.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(read["rotation"][0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(read["linear_velocity"][0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(read["angular_velocity"][0], [0.4, 0.5, 0.6])


def test_named_joint_queries_reject_unknown_joints():
    from motrix_env_core import registry

    finger_cfg = registry.make_env_config("dm-finger-spin", mode="play")
    finger = MotrixSimBackend(finger_cfg.scene, finger_cfg.sim, 1)
    with pytest.raises(KeyError, match="unknown joint"):
        finger.compile_reads({"joint_pos": JointPositionQuery(joints=("missing",))})


def test_motrixsim_reset_program_validates_environment_ids():
    backend = _make_backend(SceneCfg(), SimCfg(), num_envs=2)
    program = backend.write_compiler.compile({"ctrl": CtrlTargetsWrite()}, reset=True)

    with pytest.raises(TypeError, match="integer tensor"):
        program.execute(np.asarray([0], dtype=np.int32))
    with pytest.raises(IndexError, match="out of range"):
        program.execute(torch.tensor([2], dtype=torch.int64))
    with pytest.raises(ValueError, match="duplicates"):
        program.execute(torch.tensor([0, 0], dtype=torch.int64))


def test_sim_cfg_preserves_unspecified_solver_options():
    world = mtx.msd.World()
    world.simulate_option.constraint_solver_iterations = 7
    world.simulate_option.constraint_solver_tolerance = 2e-4

    MotrixSimSceneCompiler().configure_world(world, SimCfg(dt=0.01))
    options = world.build().options

    assert options.max_iterations == 7
    assert options.solver_tolerance == pytest.approx(2e-4)


def test_ctrl_targets_write_routes_named_actuators_through_native_plan():
    from motrix_env_core import registry

    cfg = registry.make_env_config("go2-walk-flat", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 3)
    actuator_names = tuple(actuator.name for actuator in backend._model.actuators)
    reordered = (actuator_names[-1], actuator_names[0])
    ctrl = backend.write_compiler.compile({"ctrl": CtrlTargetsWrite(reordered)})

    assert ctrl.buffer("ctrl").shape == (3, 2)
    ctrl.buffer("ctrl")[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ctrl.execute(torch.tensor([1], dtype=torch.int64))

    read = backend.compile_reads({"ctrl_values": ActuatorCtrlQuery()})
    read.execute()
    # The declared order defines the buffer layout: column 0 goes to the last
    # declared actuator, column 1 to the first. Full model order is read back.
    expected = np.zeros(len(actuator_names), dtype=np.float32)
    expected[[len(actuator_names) - 1, 0]] = [3.0, 4.0]
    np.testing.assert_allclose(read["ctrl_values"][1], expected)
    np.testing.assert_allclose(read["ctrl_values"][0], np.zeros(len(actuator_names), dtype=np.float32))


def test_partial_env_ids_leave_other_rows_untouched_with_native_writes():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-humanoid-walk", mode="play")
    body = "torso"
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    program = backend.write_compiler.compile(
        {
            "body_position": BodyPositionWrite((body,)),
            "body_rotation": BodyRotationWrite((body,)),
        }
    )
    read = backend.compile_reads({"position": LinkPositionQuery(link=body), "rotation": LinkQuaternionQuery(link=body)})
    program.buffer("body_position")[1, 0] = torch.tensor([1.0, 2.0, 3.0])
    program.buffer("body_rotation")[1, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])

    program.execute(torch.tensor([1], dtype=torch.int64))
    read.execute()

    np.testing.assert_allclose(read["position"][1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(read["rotation"][1], [0.0, 0.0, 0.0, 1.0])
    assert not np.allclose(read["position"][0], [1.0, 2.0, 3.0])


def test_ctrl_targets_reject_duplicated_actuators_with_stable_message():
    from motrix_env_core import registry

    cfg = registry.make_env_config("go2-walk-flat", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 1)
    name = backend._model.actuators[0].name
    with pytest.raises(ValueError, match="both target actuator"):
        backend.write_compiler.compile(
            {
                "first": CtrlTargetsWrite((name,)),
                "second": CtrlTargetsWrite((name,)),
            }
        )


def test_mixed_dof_and_native_writes_execute_in_one_program():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-humanoid-walk", mode="play")
    body = "torso"
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    program = backend.write_compiler.compile(
        {
            "joint_position": BodyJointPositionWrite(body),
            "base_position": BodyPositionWrite((body,)),
        }
    )
    read = backend.compile_reads(
        {"joint_pos": BodyJointPositionQuery(body=body), "position": LinkPositionQuery(link=body)}
    )
    program.buffer("joint_position")[0] = 0.0
    program.buffer("base_position")[0, 0] = torch.tensor([0.5, -0.5, 1.5])

    program.execute(torch.tensor([0], dtype=torch.int64))
    read.execute(torch.tensor([0], dtype=torch.int64))

    # Legacy DOF-channel scatter and the native body write land in the same
    # step, and forward kinematics still refreshes link poses once.
    np.testing.assert_allclose(read["position"][0], [0.5, -0.5, 1.5])
    np.testing.assert_allclose(read["joint_pos"][0], program.buffer("joint_position")[0])


def test_read_views_alias_arena_and_stay_conventionally_read_only():
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    backend = MotrixSimBackend(cfg.scene, cfg.sim, 2)
    program = backend.compile_reads({"all_pos": DofPositionQuery()})
    program.execute()

    tensor = program.view("all_pos")
    arena = program._read.program.buffer
    # Zero-copy contract: the public tensor aliases the private arena storage
    # (the read program rewrites it in place), never a copy.
    arena_start = arena.__array_interface__["data"][0]
    arena_end = arena_start + arena.nbytes
    assert arena_start <= tensor.data_ptr() < arena_end
    # The backing numpy views stay non-writeable; torch cannot carry that
    # flag, so the read-only contract is documented on ``view``.
    assert not program._views["all_pos"].flags.writeable


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_live_control_views_on_alternating_caller_streams():
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry

    cfg = registry.make_env_config("cartpole")
    cfg.sim.device = "cuda:0"
    device = torch.device(cfg.sim.device)
    backend = create_sim_backend("motrixsim", device)(cfg.scene, cfg.sim, 3)
    reads = backend.compile_reads({"ctrl": ActuatorCtrlQuery(), "alias": ActuatorCtrlQuery()})
    writes = backend.compile_action_writes({"ctrl": CtrlTargetsWrite()})
    controls = reads["ctrl"]
    assert controls is reads["alias"]
    assert controls.data_ptr() == writes.buffer("ctrl").data_ptr()
    assert reads.arena_bytes == 0
    with pytest.raises(RuntimeError, match="full-batch"):
        writes.execute(torch.tensor([1], device=device))
    with pytest.raises(RuntimeError, match="full-batch"):
        writes.execute_masked(torch.tensor([False, True, False], device=device))
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    caller = torch.cuda.current_stream(device)
    for index, stream in enumerate(streams, 1):
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            writes.buffer("ctrl").fill_(index)
            writes.execute()
            reads.execute(torch.tensor([1], device=device))
            torch.testing.assert_close(controls, torch.full_like(controls, index))
        caller.wait_stream(stream)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("masked", [False, True])
def test_cuda_control_staging_preserves_unselected_worlds(masked):
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry

    cfg = registry.make_env_config("cartpole")
    cfg.sim.device = "cuda:0"
    backend = create_sim_backend("motrixsim", cfg.sim.device)(cfg.scene, cfg.sim, 3)
    controls = backend.compile_reads({"ctrl": ActuatorCtrlQuery()})["ctrl"]
    writes = backend.compile_writes({"ctrl": CtrlTargetsWrite()})
    initial = controls.clone()
    writes.buffer("ctrl").copy_(initial + 1)
    torch.testing.assert_close(controls, initial)

    if masked:
        writes.execute_masked(torch.tensor([False, True, False], device=backend.device))
    else:
        writes.execute(torch.tensor([1], device=backend.device))

    initial[1] += 1
    torch.testing.assert_close(controls, initial)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_contact_reports_for_offset_load(tmp_path):
    pytest.importorskip("motrixsim.gpu")
    scene = tmp_path / "offset_contact.xml"
    scene.write_text(
        """<mujoco><worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="load" pos="0 0 .4">
          <joint name="height" type="slide" axis="0 0 1"/>
          <inertial pos="0 0 0" mass="1" diaginertia=".01 .01 .01"/>
          <geom name="sphere" pos=".1 0 0" type="sphere" size=".15"/>
          <site name="bottom" pos=".1 0 -.15" type="box" size=".03 .03 .03"/>
          <site name="top" pos=".1 0 .15" size=".03"/>
        </body></worldbody>
        <sensor>
          <touch name="pressure" site="bottom"/><touch name="miss" site="top"/>
          <contact name="up" geom1="floor" geom2="sphere" data="normal" num="1"/>
          <contact name="down" geom1="sphere" geom2="floor" data="normal" num="1"/>
        </sensor></mujoco>"""
    )
    device = torch.device("cuda:0")
    backend = create_sim_backend("motrixsim", device)(
        SceneCfg(file=scene), SimCfg(dt=0.005, gravity=(0, 0, -9.81), device=str(device)), 2
    )
    reads = backend.compile_reads(
        {
            "wrench": BodyLinkNetContactForceQuery(body="load"),
            "link_wrench": LinkNetContactForceQuery(link="load"),
            "batch_wrench": BatchLinkNetContactForceQuery(links=("load",)),
            "reports": SensorValuesQuery(sensors=("pressure", "miss", "up", "down")),
        }
    )
    backend.step(180)
    reads.execute()

    # The normal balances the 1 kg load, 0.1 m from its center of mass.
    expected = torch.tensor([0, 0, 9.81, 0, -0.981, 0], device=device)
    torch.testing.assert_close(reads["wrench"][:, 0], expected.expand(2, -1), atol=0.01, rtol=0.001)
    reports = torch.tensor([9.81, 0, 0, 0, 1, 0, 0, -1], device=device)
    torch.testing.assert_close(reads["reports"], reports.expand(2, -1), atol=0.01, rtol=0.001)
    torch.testing.assert_close(reads["link_wrench"], reads["wrench"][:, 0])
    torch.testing.assert_close(reads["batch_wrench"], reads["wrench"])

    untouched = reads["link_wrench"][1].clone()
    mask = torch.tensor([True, False], device=device)
    backend.reset_masked(mask)
    reads.execute_masked(mask)
    torch.testing.assert_close(reads["link_wrench"][0], torch.zeros_like(untouched))
    torch.testing.assert_close(reads["link_wrench"][1], untouched)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_compact_writes_route_rows_in_caller_order():
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry

    cfg = registry.make_env_config("dm-finger-spin", mode="play")
    cfg.sim.device = "cuda:0"
    device = torch.device(cfg.sim.device)
    backend = create_sim_backend("motrixsim", device)(cfg.scene, cfg.sim, 4)
    joints = ("distal", "hinge")
    program = backend.compile_writes({"position": JointPositionWrite(joints)}, reset=True)
    read = backend.compile_reads({"position": JointPositionQuery(joints=joints)})
    read.execute()
    expected = read["position"].clone()
    ids = torch.tensor([3, 0], device=device)
    compact = torch.tensor([[0.2, -0.1], [-0.4, 0.3]], device=device)
    staging = program.buffer("position")
    staging[:2].copy_(compact)
    program.execute_compact(ids, (staging,))
    read.execute()
    expected[ids] = compact
    torch.testing.assert_close(read["position"], expected)
    with pytest.raises(ValueError, match="conflict"):
        backend.compile_writes({"a": JointPositionWrite(joints), "b": JointPositionWrite(("hinge",))})


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_actuator_gains_skip_invalid_values_and_restore_selected_worlds():
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry
    from motrix_env_core.sim.write import ActuatorDampingWrite, ActuatorKpWrite

    cfg = registry.make_env_config("go2-walk-flat")
    cfg.sim.device = "cuda:0"
    device = torch.device(cfg.sim.device)
    backend = create_sim_backend("motrixsim", device)(cfg.scene, cfg.sim, 3)
    actuators = backend.model.actuators[:2][::-1]
    names = tuple(a.name for a in actuators)
    gains = backend.compile_writes({"kp": ActuatorKpWrite(names), "kd": ActuatorDampingWrite(names)})
    reset = backend.compile_writes({}, reset=True)
    rows = torch.tensor([2, 0], device=device)
    for key in ("kp", "kd"):
        gains.buffer(key)[:2] = gains.buffer(key).new_tensor([[3.5, float("nan")], [float("inf"), 0]])
    with backend.physics_stream():
        defaults = [(a.get_kp_override(backend.data), a.get_damping_override(backend.data)) for a in actuators]
    gains.execute_compact(rows, (gains.buffer("kp"), gains.buffer("kd")))
    for restored in (False, True):
        if restored:
            reset.execute(rows[:1])
        with backend.physics_stream():
            for column, actuator in enumerate(actuators):
                for initial, getter in zip(defaults[column], (actuator.get_kp_override, actuator.get_damping_override)):
                    expected = initial.copy()
                    if column == 0 and not restored:
                        expected[2] = 3.5
                    elif column == 1:
                        expected[0] = 0
                    np.testing.assert_array_equal(getter(backend.data), expected)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_reset_commits_final_parameter_values_once(monkeypatch):
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry
    from motrix_env_core.sim.write import LinkMassWrite

    cfg = registry.make_env_config("cartpole")
    cfg.sim.device = "cuda:0"
    backend = create_sim_backend("motrixsim", cfg.sim.device)(cfg.scene, cfg.sim, 3)
    updates = backend.compile_writes({"mass": LinkMassWrite(("cart", "pole"))})
    reset = backend.compile_writes({"pole": LinkMassWrite(("pole",))}, reset=True)
    get_cart_mass = backend.model.get_link("cart").get_mass_override
    get_pole_mass = backend.model.get_link("pole").get_mass_override
    with backend.physics_stream():
        cart_default = get_cart_mass(backend.data).copy()
        pole_default = get_pole_mass(backend.data).copy()
    updates.buffer("mass").copy_(updates.buffer("mass").new_tensor([[2, 5], [3, 6], [4, 7]]))
    updates.execute()
    reset.buffer("pole")[:, 0] = reset.buffer("pole").new_tensor([float("nan"), 8, 9])

    commits = 0
    original_commit = backend.overrides.commit_link_inertia

    def record(warp_mask):
        nonlocal commits
        commits += 1
        original_commit(warp_mask)

    monkeypatch.setattr(backend.overrides, "commit_link_inertia", record)
    reset.execute_masked(torch.tensor([True, False, True], device=backend.device))

    assert commits == 1
    cart_default[1] = 3
    pole_default[1:] = [6, 9]
    with backend.physics_stream():
        np.testing.assert_allclose(get_cart_mass(backend.data), cart_default)
        np.testing.assert_allclose(get_pole_mass(backend.data), pole_default)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_parameter_overrides_match_sdk_setters_for_welded_links_and_colliders(tmp_path):
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core.sim.write import (
        ActuatorDampingWrite,
        ActuatorKpWrite,
        GeomFrictionWrite,
        LinkComWrite,
        LinkMassWrite,
    )

    scene = tmp_path / "welded_overrides.xml"
    scene.write_text(
        """<mujoco>
        <worldbody>
          <geom name="floor" type="plane" size="5 5 0.1" friction="0.8 0.01 0.001"/>
          <body name="base" pos="0 0 0.35">
            <freejoint name="root"/>
            <inertial pos="0.02 -0.03 0.04" quat="0.9238795 0.3826834 0 0" mass="2.5" diaginertia="0.1 0.2 0.3"/>
            <geom name="g1" type="sphere" size="0.1" pos="0.1 0 -0.2" friction="0.6 0.01 0.001"/>
            <body name="welded_child" pos="0.15 0.2 -0.1" quat="0.6 0.8 0 0">
              <inertial pos="-0.01 0.05 0.02" quat="0.8 0.6 0 0" mass="1.2" diaginertia="0.02 0.03 0.04"/>
              <geom name="g2" type="box" size="0.05 0.05 0.05" friction="0.7 0.02 0.002"/>
            </body>
            <body name="arm" pos="0 0.2 0">
              <joint name="j1" type="hinge" axis="0 1 0"/>
              <inertial pos="0 0.1 0" mass="0.8" diaginertia="0.01 0.015 0.02"/>
              <geom name="g3" type="capsule" size="0.03 0.1"/>
            </body>
          </body>
        </worldbody>
        <actuator>
          <position name="act_root" joint="root" kp="12"/>
          <position name="act_j1" joint="j1" kp="20"/>
        </actuator></mujoco>"""
    )
    sim_cfg = SimCfg(dt=0.005, device="cuda:0")
    scene_cfg = SceneCfg(file=scene)
    backend = create_sim_backend("motrixsim", sim_cfg.device)(scene_cfg, sim_cfg, 4)
    reference = create_sim_backend("motrixsim", sim_cfg.device)(scene_cfg, sim_cfg, 4)

    writes = backend.compile_writes(
        {
            "mass": LinkMassWrite(("base", "welded_child", "arm")),
            "com": LinkComWrite(("base", "welded_child")),
            "friction": GeomFrictionWrite(("g1", "g2")),
            "kp": ActuatorKpWrite(("act_root", "act_j1")),
            "kd": ActuatorDampingWrite(("act_root", "act_j1")),
        }
    )
    device = backend.device
    masses = torch.tensor([[3.1, 1.7, 0.9], [2.2, float("nan"), 1.1], [4.0, 0.8, 0.6], [2.5, 2.1, 1.3]], device=device)
    coms = torch.tensor(
        [
            [[0.05, -0.01, 0.02], [0.03, -0.02, 0.08]],
            [[0.02, -0.03, 0.04], [-0.04, 0.09, -0.03]],
            [[float("nan"), 0.0, 0.0], [0.08, 0.01, -0.05]],
            [[-0.03, 0.04, 0.01], [0.02, -0.06, 0.07]],
        ],
        device=device,
    )
    frictions = torch.tensor(
        [
            [[1.1, 0.01, 0.001], [0.4, 0.02, 0.002]],
            [[0.3, 0.01, 0.001], [float("nan"), 0.02, 0.002]],
            [[0.9, 0.01, 0.001], [1.3, 0.02, 0.002]],
            [[0.5, 0.01, 0.001], [0.9, 0.02, 0.002]],
        ],
        device=device,
    )
    kps = torch.tensor([[15.0, 25.0], [10.0, float("nan")], [18.0, 30.0], [12.0, 22.0]], device=device)
    kds = torch.tensor([[1.5, 2.5], [float("nan"), 1.0], [2.0, 3.0], [0.5, 1.2]], device=device)

    writes.buffer("mass").copy_(masses)
    writes.buffer("com").copy_(coms)
    writes.buffer("friction").copy_(frictions)
    writes.buffer("kp").copy_(kps)
    writes.buffer("kd").copy_(kds)
    writes.execute_masked(torch.tensor([True, True, True, False], device=device))

    with reference.physics_stream():
        for col, name in enumerate(("base", "welded_child", "arm")):
            vals = masses[:, col].cpu().numpy()
            vals[3] = np.nan
            reference.model.get_link(name).set_mass_override(reference.data, vals)
        for col, name in enumerate(("base", "welded_child")):
            vals = coms[:, col].cpu().numpy()
            vals[3] = np.nan
            reference.model.get_link(name).set_center_of_mass_override(reference.data, vals)
        for col, name in enumerate(("g1", "g2")):
            vals = frictions[:, col, 0].cpu().numpy()
            vals[3] = np.nan
            reference.model.get_geom(name).set_friction_override(reference.data, vals)
        for col, name in enumerate(("act_root", "act_j1")):
            kp_vals = kps[:, col].cpu().numpy()
            kd_vals = kds[:, col].cpu().numpy()
            kp_vals[3] = np.nan
            kd_vals[3] = np.nan
            reference.model.get_actuator(name).set_kp_override(reference.data, kp_vals)
            reference.model.get_actuator(name).set_damping_override(reference.data, kd_vals)

    backend.control.fill_(0.35)
    reference.control.fill_(0.35)
    backend.step(20)
    reference.step(20)
    torch.testing.assert_close(backend.position, reference.position, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(backend.velocity, reference.velocity, atol=1e-5, rtol=1e-5)

    # Masked reset of world 0 restores every overridden domain to nominal defaults.
    backend.reset_masked(torch.tensor([True, False, False, False], device=device))
    with backend.physics_stream(), reference.physics_stream():
        for name in ("base", "welded_child", "arm"):
            assert np.isclose(
                backend.model.get_link(name).get_mass_override(backend.data)[0],
                reference.model.get_link(name).get_mass_override(reference.data)[3],
            )
        for name in ("g1", "g2"):
            assert np.isclose(
                backend.model.get_geom(name).get_friction_override(backend.data)[0],
                reference.model.get_geom(name).get_friction_override(reference.data)[3],
            )


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_ball_joint_write_normalizes_each_quaternion(tmp_path):
    pytest.importorskip("motrixsim.gpu")
    scene = tmp_path / "ball.xml"
    scene.write_text(
        '<mujoco><worldbody><body name="link"><joint type="ball" name="ball"/>'
        '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    device = torch.device("cuda:0")
    backend = create_sim_backend("motrixsim", device)(SceneCfg(file=scene), SimCfg(device=str(device)), 2)
    writes = backend.compile_writes({"ball": JointQuaternionWrite(("ball",))})
    reads = backend.compile_reads({"position": DofPositionQuery()})
    writes.buffer("ball")[:, 0] = torch.tensor([[0, 3, 0, 4], [0, 0, 0, 2]], device=device)
    writes.execute()
    reads.execute()
    torch.testing.assert_close(reads["position"], torch.tensor([[0, 0.6, 0, 0.8], [0, 0, 0, 1]], device=device))


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
@pytest.mark.parametrize("name", ["go2-walk-flat", "go2-walk-rough"])
def test_cuda_metadata_and_terrain_follow_simulation_device(name):
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry

    cfg = registry.make_env_config(name)
    cpu = create_sim_backend("motrixsim")(cfg.scene, cfg.sim, 1)
    cfg.sim.device = "cuda:0"
    device = torch.device(cfg.sim.device)
    backend = create_sim_backend("motrixsim", device)(cfg.scene, cfg.sim, 1)
    body = cfg.scene.objs.robot.resolved_base_link_name
    model = backend.compile_model({"tree": KinematicTreeQuery(body), "host": KinematicTreeQuery(body, "cpu")})
    assert (
        model.init_dof_pos.device
        == model.bodies["robot"].init_joint_pos.device
        == model.others["tree"].axes.device
        == device
    )
    assert model.others["host"].axes.device.type == "cpu"
    ids = torch.tensor([0])
    xy = torch.tensor([[[0.0, 0.0], [0.25, 0.25]]])
    expected = cpu.sample_terrain_height(cfg.ground_geom_name, ids, xy)
    actual = backend.sample_terrain_height(cfg.ground_geom_name, ids, xy)
    assert actual.device == device
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-3, rtol=1e-3)
    with pytest.raises(KeyError, match="absent"):
        backend.sample_terrain_height("absent", ids, xy)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA")
def test_cuda_manager_keeps_captured_read_inputs_alive():
    pytest.importorskip("motrixsim.gpu")
    from motrix_env_core import registry

    env = registry.make("g1-wbt-dance", num_envs=16, seed=7, sim_device="cuda:0")
    actions = torch.zeros((env.num_envs, *env.action_space.shape), device=env.device)
    env.initialize()
    for _ in range(4):
        state = env.step(actions)
        torch.testing.assert_close(state.reward, torch.nan_to_num(state.reward))
        torch.testing.assert_close(state.obs.policy, torch.nan_to_num(state.obs.policy))
    assert (env.engine_error_codes == 0).all()
