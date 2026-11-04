# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core.config import SimCfg, configclass
from motrix_env_core.config.scene.base import SceneCfg, SceneSensorsCfg
from motrix_env_core.config.scene.sensor import (
    OrthographicCameraCfg,
    PinholeCameraCfg,
    RaycastCameraSensorCfg,
)
from motrix_env_core.sim.read import DofPositionQuery, RaycastCameraImageQuery, SensorValuesQuery
from motrix_env_core.sim.write import JointPositionWrite, JointVelocityWrite
from motrix_env_motrixsim.runtime import MotrixSimBackend

pytestmark = pytest.mark.integration


def camera_backend(tmp_path, *, moving=False, foreground=False, **overrides):
    path = tmp_path / "camera.xml"
    mount = (
        '<body name="rig"><joint name="slide" type="slide" axis="0 0 1"/>'
        '<geom type="sphere" size=".05" mass="1"/><site name="optical"/></body>'
        if moving
        else '<site name="optical"/>'
    )
    obstruction = '<geom type="box" size="10 10 .01" pos="0 0 .2"/>' if foreground else ""
    path.write_text(
        '<mujoco><option gravity="0 0 0"/><worldbody>'
        '<geom type="box" size="10 10 .1" pos="0 0 5"/>' + obstruction + mount + "</worldbody></mujoco>"
    )

    @configclass
    class Sensors(SceneSensorsCfg):
        camera: RaycastCameraSensorCfg = RaycastCameraSensorCfg(
            **dict(
                site="optical",
                width=4,
                height=3,
                projection=PinholeCameraCfg(fx=4, fy=4, cx=2, cy=1.5),
                far=10,
                outputs=("depth", "distance", "point", "normal"),
            )
            | overrides
        )

    return MotrixSimBackend(SceneCfg(file=path, sensors=Sensors()), SimCfg(dt=0.01), 2)


def test_image_fields_match_geometry_and_native_flat_layout(tmp_path):
    backend = camera_backend(tmp_path)
    queries = {
        field: RaycastCameraImageQuery(sensor="camera", field=field)
        for field in ("depth", "distance", "point", "normal")
    }
    queries["alias"] = queries["depth"]
    queries["raw"] = SensorValuesQuery(sensors=("camera",))
    program = backend.compile_reads(queries)
    backend.step(1)
    program.execute()
    depth = program["depth"]
    points = program["point"]
    normals = program["normal"]
    assert depth.shape == (2, 3, 4)
    assert points.shape == normals.shape == (2, 3, 4, 3)
    np.testing.assert_allclose(depth, 4.9, atol=1e-5)
    np.testing.assert_allclose(points[..., 2], depth, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(points, axis=-1), program["distance"], atol=1e-5)
    np.testing.assert_allclose(normals[..., 2], -1)
    flat = program["raw"].reshape(2, 3, 4, 8)
    np.testing.assert_array_equal(depth, flat[..., 0])
    np.testing.assert_array_equal(program["distance"], flat[..., 1])
    np.testing.assert_array_equal(points, flat[..., 2:5])
    np.testing.assert_array_equal(normals, flat[..., 5:8])
    # Disjoint interleaved channels share an allocation, not overlapping bytes.
    assert depth.base is points.base
    assert program["alias"] is depth
    assert not depth.flags.writeable
    before = depth.copy()
    program.execute(np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(depth, before)
    # Camera field layout is independent of declaration order.
    reverse = backend.compile_reads(
        {
            "normal": RaycastCameraImageQuery(sensor="camera", field="normal"),
            "depth": RaycastCameraImageQuery(sensor="camera"),
        }
    )
    reverse.execute()
    np.testing.assert_array_equal(reverse["normal"], normals)
    np.testing.assert_array_equal(reverse["depth"], depth)
    assert reverse.arena_bytes == 2 * 3 * 4 * 8 * np.dtype(np.float32).itemsize


def test_reset_invalidates_cached_image_but_intra_episode_fk_respects_update_period(tmp_path):
    # MotrixSim contract: reset leaves no valid frame, so the next FK write
    # resamples; without reset the cached image is kept until update_period
    # elapses on a physics step.
    backend = camera_backend(tmp_path, moving=True, update_period=100.0)
    reads = backend.compile_reads({"depth": RaycastCameraImageQuery(sensor="camera")})
    reset = backend.compile_writes({"q": JointPositionWrite(joints=("slide",))}, reset=True, forward_kinematics=True)
    fk = backend.compile_writes({"q": JointPositionWrite(joints=("slide",))}, reset=False, forward_kinematics=True)

    backend.step(1)
    reads.execute()
    np.testing.assert_allclose(reads["depth"], 4.9, atol=1e-5)

    fk.buffer("q")[:] = [[1.0], [2.0]]
    fk.execute()
    reads.execute()
    np.testing.assert_allclose(reads["depth"], 4.9, atol=1e-5)

    reset.buffer("q")[:] = [[1.0], [2.0]]
    reset.execute()
    reads.execute()
    np.testing.assert_allclose(reads["depth"][0], 3.9, atol=1e-5)
    np.testing.assert_allclose(reads["depth"][1], 2.9, atol=1e-5)


def test_partial_reset_resamples_only_selected_rows(tmp_path):
    backend = camera_backend(tmp_path, moving=True, update_period=100.0)
    reads = backend.compile_reads({"depth": RaycastCameraImageQuery(sensor="camera")})
    reset = backend.compile_writes({"q": JointPositionWrite(joints=("slide",))}, reset=True, forward_kinematics=True)
    backend.step(1)
    reads.execute()
    reset.buffer("q")[:] = [[0.0], [2.0]]
    reset.execute(np.array([1], dtype=np.int64))
    reads.execute()
    np.testing.assert_allclose(reads["depth"][0], 4.9, atol=1e-5)
    np.testing.assert_allclose(reads["depth"][1], 2.9, atol=1e-5)


def test_non_integer_update_period_resamples_on_step_at_elapsed_times(tmp_path):
    backend = camera_backend(tmp_path, moving=True, update_period=0.015)  # 1.5 * dt
    reads = backend.compile_reads({"depth": RaycastCameraImageQuery(sensor="camera")})
    vel = backend.compile_writes({"v": JointVelocityWrite(joints=("slide",))})
    backend.step(1)
    vel.buffer("v")[:] = [[1.0], [1.0]]
    vel.execute()
    backend.step(1)
    backend.step(2)  # resamples at t=0.015 and t=0.03; sampling precedes integration, so z=0.02 at t=0.03
    reads.execute()
    np.testing.assert_allclose(reads["depth"], 4.9 - 0.02, atol=1e-5)


def test_partial_read_updates_only_selected_row_with_interleaved_fields(tmp_path):
    backend = camera_backend(tmp_path, moving=True)
    reads = backend.compile_reads(
        {
            "depth": RaycastCameraImageQuery(sensor="camera"),
            "normal": RaycastCameraImageQuery(sensor="camera", field="normal"),
            "q": DofPositionQuery(),
        }
    )
    backend.step(1)
    reads.execute()
    before = reads["depth"].copy()
    writes = backend.compile_writes({"q": JointPositionWrite(joints=("slide",))})
    writes.buffer("q")[:] = [[1.0], [2.0]]
    writes.execute()
    backend.step(1)
    reads.execute(np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(reads["depth"][0], before[0])
    np.testing.assert_allclose(reads["depth"][1], 2.9, atol=1e-5)
    np.testing.assert_array_equal(reads["q"], [[0], [2]])
    np.testing.assert_allclose(reads["normal"][..., 2], -1)
    reads.execute()
    np.testing.assert_allclose(reads["depth"][0], 3.9, atol=1e-5)


def test_orthographic_grid_has_parallel_rays_and_metric_spacing(tmp_path):
    backend = camera_backend(tmp_path, projection=OrthographicCameraCfg(pixel_size_x=0.2, pixel_size_y=0.3))
    reads = backend.compile_reads(
        {
            field: RaycastCameraImageQuery(sensor="camera", field=field)
            for field in ("depth", "distance", "point", "normal")
        }
    )
    backend.step(1)
    reads.execute()
    np.testing.assert_allclose(reads["distance"], reads["depth"])
    np.testing.assert_allclose(reads["depth"], 4.9, atol=1e-5)
    points = reads["point"]
    np.testing.assert_allclose(np.diff(points[..., 0], axis=2), 0.2, atol=1e-5)
    np.testing.assert_allclose(np.diff(points[..., 1], axis=1), 0.3, atol=1e-5)


def test_near_surface_occludes_far_surface_instead_of_seeing_through(tmp_path):
    backend = camera_backend(tmp_path, foreground=True, near=0.5)
    reads = backend.compile_reads(
        {
            "depth": RaycastCameraImageQuery(sensor="camera"),
            "point": RaycastCameraImageQuery(sensor="camera", field="point"),
        }
    )
    backend.step(1)
    reads.execute()
    np.testing.assert_array_equal(reads["depth"], -1)
    np.testing.assert_array_equal(reads["point"], 0)


def test_disabled_field_is_rejected_and_miss_is_not_normalized(tmp_path):
    backend = camera_backend(tmp_path, far=1, outputs=("depth",))
    with pytest.raises(ValueError, match="does not enable"):
        backend.compile_reads({"points": RaycastCameraImageQuery(sensor="camera", field="point")})
    with pytest.raises(KeyError, match="raycast camera sensor"):
        backend.compile_reads({"unknown": RaycastCameraImageQuery(sensor="missing")})
    program = backend.compile_reads({"depth": RaycastCameraImageQuery(sensor="camera")})
    backend.step(1)
    program.execute()
    np.testing.assert_array_equal(program["depth"], -1)
