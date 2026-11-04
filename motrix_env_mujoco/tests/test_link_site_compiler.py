# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import BodyCfg, SceneCfg, SceneObjsCfg, SceneSensorsCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg
from motrix_env_core.config.scene.sensor import PinholeCameraCfg, RaycastCameraSensorCfg
from motrix_env_core.config.scene.site import SiteCfg
from motrix_env_core.config.scene.urdf import UrdfFileCfg

mj = pytest.importorskip("mujoco")

from motrix_env_mujoco.compiler import build_mujoco_spec  # noqa: E402


@pytest.fixture(params=["mjcf", "urdf"])
def scene(request, tmp_path):
    if request.param == "mjcf":
        path = tmp_path / "tiny.xml"
        path.write_text(
            '<mujoco><worldbody><body name="base"><geom type="sphere" size=".1" mass="1"/>'
            '<site name="old"/></body></worldbody></mujoco>'
        )
        model = MjcfFileCfg(file=path)
    else:
        path = tmp_path / "tiny.urdf"
        path.write_text(
            '<robot name="tiny"><link name="base"><inertial><mass value="1"/>'
            '<inertia ixx=".01" iyy=".01" izz=".01" ixy="0" ixz="0" iyz="0"/>'
            "</inertial></link></robot>"
        )
        model = UrdfFileCfg(file=path, sites=[SiteCfg(name="old", parent_link_name="base")])

    @configclass
    class Objects(SceneObjsCfg):
        optical: SiteCfg = SiteCfg(parent_link_name="base", position=(0.1, 0.2, 0.3), rotation=(1, 0, 0, 0))
        body: BodyCfg = BodyCfg(model=model, base_link_name="base", prefix="p_")

    return SceneCfg(objs=Objects())


def test_two_bodies_resolve_site_parent_by_owning_body(tmp_path):
    @configclass
    class Objects(SceneObjsCfg):
        optical: SiteCfg = SiteCfg(parent_link_name="base")
        second: SiteCfg = SiteCfg(parent_link_name="tool")
        left: BodyCfg = BodyCfg(
            model=MjcfFileCfg(file=_write_link_model(tmp_path, "l.xml", "base")), base_link_name="base", prefix="a_"
        )
        right: BodyCfg = BodyCfg(
            model=MjcfFileCfg(file=_write_link_model(tmp_path, "r.xml", "tool")), base_link_name="tool", prefix="b_"
        )

    spec = build_mujoco_spec(SceneCfg(objs=Objects()))
    assert spec.site("a_optical").parent.name == "a_base"
    assert spec.site("b_second").parent.name == "b_tool"


def test_two_bodies_with_same_link_name_require_decorated_parent(tmp_path):
    @configclass
    class Objects(SceneObjsCfg):
        optical: SiteCfg = SiteCfg(parent_link_name="base")
        left: BodyCfg = BodyCfg(
            model=MjcfFileCfg(file=_write_link_model(tmp_path, "l.xml", "base")), base_link_name="base", prefix="a_"
        )
        right: BodyCfg = BodyCfg(
            model=MjcfFileCfg(file=_write_link_model(tmp_path, "r.xml", "base")), base_link_name="base", prefix="b_"
        )

    objs = Objects()
    with pytest.raises(ValueError, match="multiple attached bodies"):
        build_mujoco_spec(SceneCfg(objs=objs))

    objs.optical.parent_link_name = "b_base"
    spec = build_mujoco_spec(SceneCfg(objs=objs))
    assert spec.site("optical").parent.name == "b_base"


def _write_link_model(tmp_path, filename, link):
    path = tmp_path / filename
    path.write_text(
        f'<mujoco><worldbody><body name="{link}"><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    )
    return path


def test_mounts_before_attach_with_decorated_name_and_xyzw_rotation(scene):
    spec = build_mujoco_spec(scene)
    site = spec.site("p_optical")
    assert site.parent.name == "p_base"
    np.testing.assert_allclose(site.pos, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(site.quat, [0, 1, 0, 0])
    model = spec.compile()
    assert model.site("p_optical").id >= 0
    assert model.site("p_old").id >= 0


def test_site_rejects_unknown_parent(scene):
    scene.objs.optical.parent_link_name = "missing"
    with pytest.raises(ValueError, match="parent link"):
        build_mujoco_spec(scene)


def test_site_does_not_replace_imported_site(scene):
    @configclass
    class Objects(SceneObjsCfg):
        old: SiteCfg = SiteCfg(parent_link_name="base")
        body: BodyCfg = scene.objs.body

    scene.objs = Objects()
    with pytest.raises(ValueError, match="already exist"):
        build_mujoco_spec(scene)


def test_mjcf_model_level_sites_are_applied_and_decorated(tmp_path):
    @configclass
    class Objects(SceneObjsCfg):
        body: BodyCfg = BodyCfg(
            model=MjcfFileCfg(
                file=_write_link_model(tmp_path, "m.xml", "base"),
                sites=[SiteCfg(name="marker", parent_link_name="base")],
            ),
            base_link_name="base",
            prefix="p_",
        )

    spec = build_mujoco_spec(SceneCfg(objs=Objects()))
    assert spec.site("p_marker").parent.name == "p_base"


def test_raycast_camera_is_explicitly_unsupported(scene):
    @configclass
    class Sensors(SceneSensorsCfg):
        camera: RaycastCameraSensorCfg = RaycastCameraSensorCfg(
            site="optical",
            width=4,
            height=2,
            far=10,
            projection=PinholeCameraCfg.from_fov(width=4, height=2, horizontal_fov=90),
        )

    scene.sensors = Sensors()
    with pytest.raises(NotImplementedError, match="MuJoCo does not support RaycastCameraSensorCfg"):
        build_mujoco_spec(scene)
