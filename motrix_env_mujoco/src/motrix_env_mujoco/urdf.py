# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import mujoco as mj

from motrix_env_core.config.scene.urdf import UrdfFileCfg


def _mujoco_quat(rotation: tuple[float, float, float, float]) -> list[float]:
    """Convert the config's xyzw quaternion to MuJoCo's wxyz order."""
    x, y, z, w = rotation
    return [w, x, y, z]


def apply_urdf_cfgs(spec: mj.MjSpec, cfg: UrdfFileCfg) -> None:
    """Apply declarative URDF augmentations to a MuJoCo specification."""
    cfg_by_geom = {geom.geom_name: geom for geom in cfg.geoms}
    if len(cfg_by_geom) != len(cfg.geoms):
        raise ValueError("Multiple geom configs target the same geom")

    cfg_by_joint = {joint.joint_name: joint for joint in cfg.joints}
    if len(cfg_by_joint) != len(cfg.joints):
        raise ValueError("Multiple joint configs target the same joint")

    cfg_by_site = {site.name: site for site in cfg.sites}
    if len(cfg_by_site) != len(cfg.sites):
        raise ValueError("Duplicate configured site name")

    existing_geom_names = {geom.name for geom in spec.geoms}
    unknown_geoms = sorted(set(cfg_by_geom).difference(existing_geom_names))
    if unknown_geoms:
        raise ValueError(f"Configured geoms do not exist in model: {unknown_geoms}")

    existing_joint_names = {joint.name for joint in spec.joints}
    unknown_joints = sorted(set(cfg_by_joint).difference(existing_joint_names))
    if unknown_joints:
        raise ValueError(f"Configured joints do not exist in model: {unknown_joints}")

    existing_site_names = {site.name for site in spec.sites}
    duplicate_site_names = sorted(existing_site_names.intersection(cfg_by_site))
    if duplicate_site_names:
        raise ValueError(f"Configured site names already exist in model: {duplicate_site_names}")

    existing_body_names = {body.name for body in spec.bodies}
    parent_link_names = {site.parent_link_name for site in cfg.sites}
    unknown_site_links = sorted(parent_link_names.difference(existing_body_names))
    if unknown_site_links:
        raise ValueError(f"Configured site parent links do not exist in model: {unknown_site_links}")

    for name, geom_cfg in cfg_by_geom.items():
        geom = spec.geom(name)
        assert geom is not None
        if geom_cfg.friction is not None:
            geom.friction = geom_cfg.friction
        if geom_cfg.condim is not None:
            geom.condim = geom_cfg.condim
        if geom_cfg.priority is not None:
            geom.priority = geom_cfg.priority

    for name, joint_cfg in cfg_by_joint.items():
        joint = spec.joint(name)
        assert joint is not None
        if joint_cfg.armature is not None:
            joint.armature = joint_cfg.armature
        if joint_cfg.friction_loss is not None:
            joint.frictionloss = joint_cfg.friction_loss

    for site_cfg in cfg.sites:
        parent = spec.body(site_cfg.parent_link_name)
        assert parent is not None
        parent.add_site(
            name=site_cfg.name,
            pos=site_cfg.position,
            quat=_mujoco_quat(site_cfg.rotation),
        )
