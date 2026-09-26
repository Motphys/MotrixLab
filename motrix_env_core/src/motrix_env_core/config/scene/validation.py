# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config.scene._utils import resolve_path
from motrix_env_core.config.scene.asset import (
    HFieldAssetCfg,
    MaterialCfg,
    ProceduralHFieldAssetCfg,
    SkyboxCfg,
    TextureCfg,
)
from motrix_env_core.config.scene.base import BodyCfg, SceneAssetCfg, SceneCfg, SceneVisualCfg
from motrix_env_core.config.scene.geometry import GeomCfg, HFieldTerrainCfg


def validate_scene_cfg(scene: SceneCfg) -> None:
    """Validate backend-independent scene structure and references."""
    if scene.file is not None:
        path = resolve_path(scene.file)
        if not path.exists():
            raise FileNotFoundError(f"Scene file does not exist: {path}")

    scene.system_camera.validate()

    assets: dict[str, SceneAssetCfg] = {}
    skybox_names: list[str] = []
    for name, asset in scene.iter_assets():
        if name in assets:
            raise ValueError(f"SceneCfg asset names must be unique, got duplicate {name!r}")
        asset.validate(name)
        assets[name] = asset
        if isinstance(asset, SkyboxCfg):
            skybox_names.append(name)

    if len(skybox_names) > 1:
        raise ValueError(f"SceneCfg supports at most one SkyboxCfg, got {skybox_names!r}")

    if not isinstance(scene.visual, SceneVisualCfg):
        raise TypeError(f"SceneCfg.visual must contain SceneVisualCfg, got {type(scene.visual).__name__}")
    scene.visual.validate()

    for name, asset in assets.items():
        if isinstance(asset, MaterialCfg) and asset.texture is not None:
            texture = assets.get(asset.texture)
            if not isinstance(texture, TextureCfg):
                raise ValueError(f"Material asset {name!r} must reference a TextureCfg, got {asset.texture!r}")

    names: set[str] = set()
    body_names: set[str] = set()
    for name, obj in scene.iter_objs():
        obj.validate(name)
        if name in names:
            raise ValueError(f"SceneCfg object names must be unique, got duplicate {name!r}")
        names.add(name)
        if isinstance(obj, BodyCfg):
            body_names.add(name)

        if isinstance(obj, GeomCfg) and obj.material is not None:
            material = assets.get(obj.material)
            if not isinstance(material, MaterialCfg):
                raise ValueError(f"Geometry {name!r} must reference a MaterialCfg, got {obj.material!r}")

        if isinstance(obj, HFieldTerrainCfg):
            hfield = assets.get(obj.hfield)
            if not isinstance(hfield, (HFieldAssetCfg, ProceduralHFieldAssetCfg)):
                raise ValueError(
                    f"HField terrain {name!r} must reference an HFieldAssetCfg or ProceduralHFieldAssetCfg, "
                    f"got {obj.hfield!r}"
                )

    follow = scene.system_camera.follow
    if follow is not None and follow not in body_names:
        raise ValueError(
            f"scene.system_camera.follow must name a body object in the scene, got {follow!r}; "
            f"available body objects: {sorted(body_names)}."
        )

    sensor_names: set[str] = set()
    for name, sensor in scene.iter_sensors():
        sensor.validate(name)
        if name in sensor_names:
            raise ValueError(f"SceneCfg sensor names must be unique, got duplicate {name!r}")
        sensor_names.add(name)
