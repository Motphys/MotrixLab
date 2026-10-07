# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot model references form a self-contained package asset closure."""

import xml.etree.ElementTree as ET
from pathlib import Path

from motrix_robots.anymal import AnymalC
from motrix_robots.booster import BoosterK1
from motrix_robots.dex_evt import DexEvt
from motrix_robots.microduck import Microduck
from motrix_robots.unitree import UnitreeG129Dof, UnitreeGo1Robot, UnitreeGo2Robot


def test_builtin_models_resolve_from_package_assets() -> None:
    for robot_type in (AnymalC, BoosterK1, DexEvt, Microduck, UnitreeG129Dof, UnitreeGo1Robot, UnitreeGo2Robot):
        robot = robot_type()
        robot.validate("robot")
        assert Path(robot.model.file).is_file()
        assert len(robot.key_pose.poses[robot.init_key_pose]) == len(robot.key_pose.joint_names)


def test_bundled_mjcf_file_references_resolve() -> None:
    asset_root = Path(UnitreeGo2Robot().model.file).parents[1]
    for model in asset_root.rglob("*.xml"):
        root = ET.parse(model).getroot()
        if root.tag != "mujoco":
            continue
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        texturedir = compiler.get("texturedir", "") if compiler is not None else ""
        for element in root.iter():
            file = element.get("file")
            if file is None:
                continue
            directory = meshdir if element.tag == "mesh" else texturedir if element.tag == "texture" else ""
            assert (model.parent / directory / file).is_file(), (model, element.tag, file)


def test_bundled_urdf_mesh_references_resolve() -> None:
    asset_root = Path(UnitreeGo2Robot().model.file).parents[1]
    for model in asset_root.rglob("*.urdf"):
        root = ET.parse(model).getroot()
        for mesh in root.iter("mesh"):
            assert (model.parent / mesh.attrib["filename"]).is_file(), (model, mesh.attrib["filename"])
