# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import BodyCfg, SceneCfg, SceneObjsCfg, SceneSensorsCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg
from motrix_env_core.config.scene.sensor import (
    OrthographicCameraCfg,
    PinholeCameraCfg,
    RaycastCameraSensorCfg,
)
from motrix_env_core.config.scene.site import SiteCfg
from motrix_env_core.config.scene.urdf import UrdfFileCfg

mtx = pytest.importorskip("motrixsim")

from motrix_env_motrixsim.compiler import build_scene_world  # noqa: E402


def make_scene(tmp_path, source="mjcf", **camera_kwargs):
    if source == "mjcf":
        path = tmp_path / "model.xml"
        path.write_text(
            '<mujoco><worldbody><body name="base"><geom type="sphere" size=".1" mass="1"/>'
            '<site name="old"/></body></worldbody></mujoco>'
        )
        model = MjcfFileCfg(file=path)
    else:
        path = tmp_path / "model.urdf"
        path.write_text(
            '<robot name="tiny"><link name="base"><inertial><mass value="1"/>'
            '<inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>'
            "</inertial></link></robot>"
        )
        model = UrdfFileCfg(file=path, sites=[SiteCfg(name="old", parent_link_name="base")])

    @configclass
    class Objects(SceneObjsCfg):
        # Deliberately before body: mounting must resolve the complete scene.
        optical: SiteCfg = SiteCfg(parent_link_name="base", position=(0, 0, 0.2))
        body: BodyCfg = BodyCfg(model=model, base_link_name="base", prefix="p_", suffix="_s")

    @configclass
    class Sensors(SceneSensorsCfg):
        camera: RaycastCameraSensorCfg = RaycastCameraSensorCfg(
            **dict(
                site="p_optical_s",
                width=4,
                height=2,
                projection=PinholeCameraCfg(fx=2, fy=3, cx=2, cy=1),
                far=12,
                near=0.1,
                update_period=0.02,
                outputs=("depth", "distance", "point", "normal"),
                exclude="parent_subtree",
                include_geom_groups=[0, 255],
            )
            | camera_kwargs
        )

    return SceneCfg(objs=Objects(), sensors=Sensors())


@pytest.mark.parametrize("source", ["mjcf", "urdf"])
@pytest.mark.parametrize(
    "projection", [PinholeCameraCfg(fx=2, fy=3, cx=2, cy=1), OrthographicCameraCfg(pixel_size_x=0.1, pixel_size_y=0.2)]
)
def test_camera_maps_native_projection_outputs_filters_and_link_site(tmp_path, source, projection):
    scene = make_scene(tmp_path, source, projection=projection)
    world = build_scene_world(scene)
    sensor = world.sensors.raycast_camera[0]
    assert (sensor.name, sensor.site, sensor.width, sensor.height) == ("camera", "p_optical_s", 4, 2)
    assert sensor.near == pytest.approx(0.1)
    assert sensor.far == 12
    assert sensor.update_period == pytest.approx(0.02)
    assert sensor.exclude == mtx.msd.LidarExclude.ParentSubtree
    np.testing.assert_array_equal(sensor.include_geom_groups, [0, 255])
    assert all(getattr(sensor.returns, field) for field in ("depth", "distance", "point", "normal"))
    if isinstance(projection, PinholeCameraCfg):
        assert sensor.projection.variant == "pinhole"
        np.testing.assert_allclose(sensor.projection.value["intrinsics"], [2, 3, 2, 1])
    else:
        assert sensor.projection.variant == "orthographic"
        np.testing.assert_allclose(sensor.projection.value["spacing"], [0.1, 0.2])
    model = world.build()
    assert model.get_site("p_optical_s") is not None
    assert model.get_site("p_old_s") is not None


def test_only_requested_outputs_are_enabled(tmp_path):
    sensor = build_scene_world(make_scene(tmp_path, outputs=("distance", "normal"))).sensors.raycast_camera[0]
    assert not sensor.returns.depth
    assert sensor.returns.distance
    assert not sensor.returns.point
    assert sensor.returns.normal


@pytest.mark.parametrize("groups", [None, []])
def test_filter_none_and_empty_are_distinct(tmp_path, groups):
    sensor = build_scene_world(make_scene(tmp_path, include_geom_groups=groups)).sensors.raycast_camera[0]
    if groups is None:
        assert sensor.include_geom_groups is None
    else:
        assert len(sensor.include_geom_groups) == 0


def test_link_site_rejects_unknown_parent_and_existing_site(tmp_path):
    scene = make_scene(tmp_path)
    scene.objs.optical.parent_link_name = "missing"
    with pytest.raises(ValueError, match="parent link"):
        build_scene_world(scene)
    scene.objs.optical.parent_link_name = "p_base_s"
    scene.objs.optical = None

    @configclass
    class Objects(SceneObjsCfg):
        body: BodyCfg = scene.objs.body
        old: SiteCfg = SiteCfg(parent_link_name="base")

    scene.objs = Objects()
    with pytest.raises(ValueError, match="already exist"):
        build_scene_world(scene)


def test_two_bodies_resolve_site_parent_by_owning_body(tmp_path):
    base_model = _write_urdf(tmp_path, "l.urdf", "base")
    tool_model = _write_urdf(tmp_path, "r.urdf", "tool")

    @configclass
    class Objects(SceneObjsCfg):
        optical: SiteCfg = SiteCfg(parent_link_name="base")
        tool_site: SiteCfg = SiteCfg(parent_link_name="tool")
        left: BodyCfg = BodyCfg(model=UrdfFileCfg(file=base_model), base_link_name="base", prefix="a_")
        right: BodyCfg = BodyCfg(model=UrdfFileCfg(file=tool_model), base_link_name="tool", prefix="b_")

    world = build_scene_world(SceneCfg(objs=Objects()))
    mounted = {site.name: link.name for link in _all_links(world) for site in link.sites if site.name}
    assert mounted["a_optical"] == "a_base"
    assert mounted["b_tool_site"] == "b_tool"


def test_two_bodies_with_same_link_name_require_decorated_parent(tmp_path):
    model = _write_urdf(tmp_path, "l.urdf", "base")

    @configclass
    class Objects(SceneObjsCfg):
        optical: SiteCfg = SiteCfg(parent_link_name="base")
        left: BodyCfg = BodyCfg(model=UrdfFileCfg(file=model), base_link_name="base", prefix="a_")
        right: BodyCfg = BodyCfg(model=UrdfFileCfg(file=model), base_link_name="base", prefix="b_")

    objs = Objects()
    with pytest.raises(ValueError, match="multiple attached bodies"):
        build_scene_world(SceneCfg(objs=objs))

    objs.optical.parent_link_name = "b_base"
    world = build_scene_world(SceneCfg(objs=objs))
    mounted = {site.name: link.name for link in _all_links(world) for site in link.sites if site.name}
    assert mounted["optical"] == "b_base"


def test_mjcf_model_level_sites_are_applied_and_decorated(tmp_path):
    path = tmp_path / "model.xml"
    path.write_text(
        '<mujoco><worldbody><body name="base"><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )

    @configclass
    class Objects(SceneObjsCfg):
        body: BodyCfg = BodyCfg(
            model=MjcfFileCfg(file=path, sites=[SiteCfg(name="marker", parent_link_name="base")]),
            base_link_name="base",
            prefix="p_",
            suffix="_s",
        )

    model = build_scene_world(SceneCfg(objs=Objects())).build()
    assert model.get_site("p_marker_s") is not None


def _write_urdf(tmp_path, filename, link):
    path = tmp_path / filename
    path.write_text(
        f'<robot name="tiny"><link name="{link}"><inertial><mass value="1"/>'
        f'<inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/></inertial></link></robot>'
    )
    return path


def _all_links(world):
    from motrix_env_motrixsim.urdf import _iter_links

    return [link for body in world.hierarchy.bodies for link in _iter_links(body.link)]
