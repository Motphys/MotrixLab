# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    ContactReportField,
    ContactSensorCfg,
    ContactSensorReduce,
    FlatTerrainCfg,
    FrameObjectKind,
    FrameRefKind,
    FrameSensorCfg,
    FrameSensorType,
    HFieldTerrainCfg,
    MjcfFileCfg,
    NoiseTerrainGeneratorCfg,
    PositionActuatorCfg,
    ProceduralHFieldAssetCfg,
    RobotCfg,
    SceneAssetsCfg,
    SceneCfg,
    SceneObjsCfg,
    SceneSensorsCfg,
    SiteCfg,
    UrdfFileCfg,
)
from motrix_env_core.sim.registry import list_sim_backends


def test_mujoco_backend_is_registered_and_loaded_lazily():
    mj = pytest.importorskip("mujoco")
    from motrix_env_mujoco.backend import MuJoCoSimBackend
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    assert "mujoco" in list_sim_backends()
    assert isinstance(MuJoCoSceneCompiler().compile(SceneCfg(), SimCfg()), mj.MjModel)

    # Compile-only backend: construction compiles the scene, behavior gaps loudly.
    backend = MuJoCoSimBackend(SceneCfg(), SimCfg(), 1)
    with pytest.raises(NotImplementedError, match="no live simulation"):
        backend.compile_reads({})
    with pytest.raises(NotImplementedError):
        backend.step(1)


def test_mujoco_compiler_create_spec_supports_precompile_transforms():
    mj = pytest.importorskip("mujoco")
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    compiler = MuJoCoSceneCompiler()
    spec = compiler.create_spec(
        SceneCfg(),
        SimCfg(dt=0.005, solver_iterations=7, solver_tolerance=1e-5, gravity=(0.0, 0.0, -3.0)),
    )
    spec.worldbody.add_geom(
        name="injected_floor",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
    )
    model = spec.compile()

    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "injected_floor") >= 0
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.opt.iterations == 7
    assert model.opt.tolerance == pytest.approx(1e-5)
    assert model.opt.gravity == pytest.approx([0.0, 0.0, -3.0])

    with pytest.raises(ValueError, match="sim.dt must be positive"):
        compiler.create_spec(SceneCfg(), SimCfg(dt=0.0))


def test_core_import_and_registry_work_without_optional_mujoco_dependency():
    source_root = Path(__file__).parents[1] / "src"
    script = """
import builtins
import importlib

real_import = builtins.__import__

def import_without_mujoco(name, *args, **kwargs):
    if name == "mujoco" or name.startswith("mujoco."):
        raise ModuleNotFoundError("blocked optional dependency", name="mujoco")
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_mujoco

import motrix_env_core
from motrix_env_core.sim.registry import create_sim_backend, list_sim_backends

assert "motrixsim" in list_sim_backends()
assert "mujoco" in list_sim_backends()
try:
    create_sim_backend("mujoco")
except ModuleNotFoundError as exc:
    assert "motrix-env-mujoco" in str(exc)
else:
    raise AssertionError("MuJoCo compiler unexpectedly loaded")

try:
    importlib.import_module("motrix_env_mujoco.compiler")
except ModuleNotFoundError as exc:
    assert "motrix-env-mujoco" in str(exc)
else:
    raise AssertionError("MuJoCo backend module unexpectedly loaded")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source_root), env.get("PYTHONPATH"))))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr


def test_mujoco_compiler_lowers_scene_objects_sensors_and_sim_options(tmp_path):
    mj = pytest.importorskip("mujoco")
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    robot_file = tmp_path / "robot.xml"
    robot_file.write_text(
        """
<mujoco model="test_robot">
  <asset>
    <material name="hidden" rgba="0 0 0 0"/>
  </asset>
  <worldbody>
    <body name="base">
      <freejoint/>
      <geom name="foot" type="sphere" size="0.1" mass="1" material="hidden"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    @configclass
    class TestObjsCfg(SceneObjsCfg):
        floor: FlatTerrainCfg = FlatTerrainCfg(height=0.25)
        robot: RobotCfg = RobotCfg(
            model=MjcfFileCfg(file=robot_file),
            base_link_name="base",
            translation=(1.0, 2.0, 3.0),
            prefix="robot_",
        )

    @configclass
    class TestSensorsCfg(SceneSensorsCfg):
        contacts: ContactSensorCfg = ContactSensorCfg(
            geom1="floor",
            geom2="robot_foot",
            num=2,
            data=[ContactReportField.found, ContactReportField.force, ContactReportField.dist],
            reduce=ContactSensorReduce.maxforce,
        )
        robot_pos: FrameSensorCfg = FrameSensorCfg(
            object_type=FrameObjectKind.link,
            object_name="robot_base",
            sensor_type=FrameSensorType.framepos,
        )

    model = MuJoCoSceneCompiler().compile(
        SceneCfg(objs=TestObjsCfg(), sensors=TestSensorsCfg()),
        SimCfg(dt=0.005, solver_iterations=7, solver_tolerance=1e-5, gravity=(0.0, 0.0, -3.0)),
    )

    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "robot_base")
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    contact_sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "contacts")
    frame_sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "robot_pos")

    assert body_id >= 0
    assert model.body_pos[body_id] == pytest.approx([1.0, 2.0, 3.0])
    assert floor_id >= 0
    assert model.geom_pos[floor_id] == pytest.approx([0.0, 0.0, 0.25])
    assert model.sensor_type[contact_sensor_id] == mj.mjtSensor.mjSENS_CONTACT
    assert model.sensor_intprm[contact_sensor_id].tolist() == [11, 2, 2]
    assert model.sensor_type[frame_sensor_id] == mj.mjtSensor.mjSENS_FRAMEPOS
    assert model.opt.timestep == pytest.approx(0.005)
    assert model.opt.iterations == 7
    assert model.opt.tolerance == pytest.approx(1e-5)
    assert model.opt.gravity == pytest.approx([0.0, 0.0, -3.0])


def test_mujoco_compiler_lowers_local_site_linear_velocity_to_velocimeter(tmp_path):
    mj = pytest.importorskip("mujoco")
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    model_file = tmp_path / "floating_body.xml"
    model_file.write_text(
        """
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint/>
      <site name="imu"/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )

    @configclass
    class LocalFrameSensorsCfg(SceneSensorsCfg):
        local_velocity: FrameSensorCfg = FrameSensorCfg(
            object_type=FrameObjectKind.site,
            object_name="imu",
            sensor_type=FrameSensorType.framelinvel,
            ref_kind=FrameRefKind.local,
        )

    model = MuJoCoSceneCompiler().compile(
        SceneCfg(file=model_file, sensors=LocalFrameSensorsCfg()),
        SimCfg(),
    )
    data = mj.MjData(model)
    data.qpos[3:7] = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]
    data.qvel[:3] = [1.0, 0.0, 0.0]
    mj.mj_forward(model, data)

    sensor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SENSOR, "local_velocity")
    assert model.sensor_type[sensor_id] == mj.mjtSensor.mjSENS_VELOCIMETER
    assert data.sensordata[model.sensor_adr[sensor_id] : model.sensor_adr[sensor_id] + 3] == pytest.approx(
        [0.0, -1.0, 0.0],
        abs=1e-7,
    )


def test_mujoco_compiler_rejects_unsupported_local_frame_sensor_semantics():
    pytest.importorskip("mujoco")
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    @configclass
    class LocalFrameSensorsCfg(SceneSensorsCfg):
        local_position: FrameSensorCfg = FrameSensorCfg(
            object_type=FrameObjectKind.link,
            object_name="base",
            sensor_type=FrameSensorType.framepos,
            ref_kind=FrameRefKind.local,
        )

    with pytest.raises(ValueError, match="does not support local-frame semantics.*local_position"):
        MuJoCoSceneCompiler().compile(SceneCfg(sensors=LocalFrameSensorsCfg()), SimCfg())


def test_mujoco_compiler_builds_procedural_hfield():
    pytest.importorskip("mujoco")
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
            generator=NoiseTerrainGeneratorCfg(seed=7, height_scale=0.2),
            size=(8.0, 4.0),
            shape=(5, 7),
        )

    @configclass
    class TerrainObjsCfg(SceneObjsCfg):
        floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain")

    model = MuJoCoSceneCompiler().compile(
        SceneCfg(assets=TerrainAssetsCfg(), objs=TerrainObjsCfg()),
        SimCfg(),
    )

    assert model.nhfield == 1
    assert model.hfield_nrow[0] == 5
    assert model.hfield_ncol[0] == 7
    assert model.hfield_size[0] == pytest.approx([4.0, 2.0, 0.2, np.finfo(np.float32).eps])
    assert np.all(np.isfinite(model.hfield_data))
    assert np.min(model.hfield_data) >= 0.0
    assert np.max(model.hfield_data) <= 1.0


def test_scene_compilers_use_matching_hfield_half_extents():
    from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
            generator=NoiseTerrainGeneratorCfg(seed=7, height_scale=0.2),
            size=(8.0, 4.0),
            shape=(5, 7),
        )

    @configclass
    class TerrainObjsCfg(SceneObjsCfg):
        floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain")

    scene = SceneCfg(assets=TerrainAssetsCfg(), objs=TerrainObjsCfg())
    motrixsim_model = MotrixSimSceneCompiler().compile(scene, SimCfg())
    mujoco_model = MuJoCoSceneCompiler().compile(scene, SimCfg())
    motrixsim_hfield = motrixsim_model.get_hfield("terrain")

    assert motrixsim_hfield is not None
    assert motrixsim_hfield.bound[[0, 1, 3, 4]] == pytest.approx([-4.0, -2.0, 4.0, 2.0])
    assert mujoco_model.hfield_size[0, :2] == pytest.approx([4.0, 2.0])


def test_mujoco_compiler_applies_urdf_augmentations(tmp_path):
    mj = pytest.importorskip("mujoco")
    from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

    urdf_file = tmp_path / "robot.urdf"
    urdf_file.write_text(
        """
<robot name="test_robot">
  <link name="base">
    <inertial>
      <mass value="1"/><origin xyz="0 0 0"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="base_collision"><geometry><sphere radius="0.1"/></geometry></collision>
  </link>
  <link name="tip">
    <inertial>
      <mass value="1"/><origin xyz="0 0 0"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision name="tip_collision"><geometry><sphere radius="0.1"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="tip"/><origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="10"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    model_cfg = UrdfFileCfg(
        file=urdf_file,
        sites=[SiteCfg(name="probe", parent_link_name="tip", position=(0.0, 0.0, 0.1))],
    )
    actuators = {
        "hinge": PositionActuatorCfg(
            joint_name="hinge",
            kp=20.0,
            kv=2.0,
            inherit_joint_range=True,
            force_range=(-5.0, 5.0),
        )
    }

    @configclass
    class RobotObjsCfg(SceneObjsCfg):
        robot: RobotCfg = RobotCfg(model=model_cfg, base_link_name="base", actuators=actuators)

    model = MuJoCoSceneCompiler().compile(SceneCfg(objs=RobotObjsCfg()), SimCfg())
    actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "hinge")

    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "probe") >= 0
    assert actuator_id >= 0
    assert model.actuator_ctrlrange[actuator_id] == pytest.approx([-1.0, 1.0])
    assert model.actuator_forcerange[actuator_id] == pytest.approx([-5.0, 5.0])
    assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(20.0)
    assert model.actuator_biasprm[actuator_id, :3] == pytest.approx([0.0, -20.0, -2.0])
