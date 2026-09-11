# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import fields
from pathlib import Path

from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec3, Vec4, optional_vec, resolve_path


@configclass(kw_only=True)
class SceneAssetCfg:
    """Base config for a named scene asset."""

    def validate(self, name: str) -> None:
        if not name:
            raise ValueError("Scene asset name must not be empty")


@configclass
class SceneAssetsCfg:
    """A field-based registry whose field names are scene asset names."""

    def items(self) -> Iterator[tuple[str, SceneAssetCfg]]:
        for cfg_field in fields(self):
            asset = getattr(self, cfg_field.name)
            if asset is None:
                continue
            if not isinstance(asset, SceneAssetCfg):
                raise TypeError(
                    f"SceneAssetsCfg field {cfg_field.name!r} must contain SceneAssetCfg or None, "
                    f"got {type(asset).__name__}"
                )
            yield cfg_field.name, asset

    def keys(self) -> Iterator[str]:
        for name, _ in self.items():
            yield name

    def values(self) -> Iterator[SceneAssetCfg]:
        for _, asset in self.items():
            yield asset

    def __getitem__(self, name: str) -> SceneAssetCfg:
        for asset_name, asset in self.items():
            if asset_name == name:
                return asset
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.items())


@configclass(kw_only=True)
class SceneObjCfg:
    """Base config for a named object in a generated scene."""

    def validate(self, name: str) -> None:
        if not name:
            raise ValueError("Scene object name must not be empty")


@configclass
class SceneObjsCfg:
    """A field-based registry whose field names are ordered scene object names.

    The optional ``robot`` slot names the scene's primary robot so that
    robot-scoped configs can resolve it without a hard-coded name. Scenes
    without a robot leave it as ``None``.
    """

    robot: RobotCfg | None = None

    def items(self) -> Iterator[tuple[str, SceneObjCfg]]:
        for cfg_field in fields(self):
            obj = getattr(self, cfg_field.name)
            if obj == MISSING:
                raise ValueError(f"SceneObjsCfg field {cfg_field.name!r} is mandatory and must be provided")
            if obj is None:
                continue
            if not isinstance(obj, SceneObjCfg):
                raise TypeError(
                    f"SceneObjsCfg field {cfg_field.name!r} must contain SceneObjCfg or None, got {type(obj).__name__}"
                )
            yield cfg_field.name, obj

    def keys(self) -> Iterator[str]:
        for name, _ in self.items():
            yield name

    def values(self) -> Iterator[SceneObjCfg]:
        for _, obj in self.items():
            yield obj

    def __getitem__(self, name: str) -> SceneObjCfg:
        for obj_name, obj in self.items():
            if obj_name == name:
                return obj
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.items())


@configclass(kw_only=True)
class SceneSensorCfg:
    """Base config for a named sensor in a generated scene."""

    def validate(self, name: str) -> None:
        if not name:
            raise ValueError("Scene sensor name must not be empty")


@configclass
class SceneSensorsCfg:
    """A field-based registry whose field names are scene sensor names."""

    def items(self) -> Iterator[tuple[str, SceneSensorCfg]]:
        for cfg_field in fields(self):
            sensor = getattr(self, cfg_field.name)
            if sensor == MISSING:
                raise ValueError(f"SceneSensorsCfg field {cfg_field.name!r} is mandatory and must be provided")
            if sensor is None:
                continue
            if not isinstance(sensor, SceneSensorCfg):
                raise TypeError(
                    f"SceneSensorsCfg field {cfg_field.name!r} must contain SceneSensorCfg or None, "
                    f"got {type(sensor).__name__}"
                )
            yield cfg_field.name, sensor

    def keys(self) -> Iterator[str]:
        for name, _ in self.items():
            yield name

    def values(self) -> Iterator[SceneSensorCfg]:
        for _, sensor in self.items():
            yield sensor

    def __getitem__(self, name: str) -> SceneSensorCfg:
        for sensor_name, sensor in self.items():
            if sensor_name == name:
                return sensor
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __len__(self) -> int:
        return sum(1 for _ in self.items())


@configclass
class SceneVisualCfg:
    """Optional scene-level visual settings applied by the selected backend."""

    ambient_light_color: Vec3 | None = None
    ambient_light_brightness: float | None = None
    head_light_color: Vec3 | None = None
    head_light_luminous_power: float | None = None
    haze: Vec4 | None = None
    tone_mapping: str | None = None

    def validate(self) -> None:
        optional_vec("SceneVisualCfg.ambient_light_color", self.ambient_light_color, 3)
        optional_vec("SceneVisualCfg.head_light_color", self.head_light_color, 3)
        optional_vec("SceneVisualCfg.haze", self.haze, 4)
        if self.ambient_light_brightness is not None and self.ambient_light_brightness < 0.0:
            raise ValueError(
                f"SceneVisualCfg.ambient_light_brightness must be non-negative, got {self.ambient_light_brightness}"
            )
        if self.head_light_luminous_power is not None and self.head_light_luminous_power < 0.0:
            raise ValueError(
                f"SceneVisualCfg.head_light_luminous_power must be non-negative, got {self.head_light_luminous_power}"
            )
        if self.tone_mapping not in (None, "none", "aces"):
            raise ValueError(f"SceneVisualCfg.tone_mapping must be None, 'none', or 'aces', got {self.tone_mapping!r}")


@configclass
class SystemCameraCfg:
    """System camera settings used by interactive viewing and video recording."""

    lookat: Vec3 | None = None
    distance: float = 2.0
    elevation: float = -20.0
    azimuth: float = 90.0

    def validate(self) -> None:
        optional_vec("scene.system_camera.lookat", self.lookat, 3)
        if self.distance <= 0.0:
            raise ValueError(f"scene.system_camera.distance must be positive, got {self.distance}")


@configclass
class SceneCfg:
    """Optional base file plus assets, visual settings, objects, and sensors."""

    file: str | Path | None = None
    assets: SceneAssetsCfg = SceneAssetsCfg()
    visual: SceneVisualCfg = SceneVisualCfg()
    system_camera: SystemCameraCfg = SystemCameraCfg()
    objs: SceneObjsCfg = SceneObjsCfg()
    sensors: SceneSensorsCfg = SceneSensorsCfg()

    def iter_assets(self) -> Iterator[tuple[str, SceneAssetCfg]]:
        yield from self.assets.items()

    def iter_objs(self) -> Iterator[tuple[str, SceneObjCfg]]:
        yield from self.objs.items()

    def iter_sensors(self) -> Iterator[tuple[str, SceneSensorCfg]]:
        yield from self.sensors.items()


@configclass(kw_only=True)
class ModelFileCfg:
    """Base config for a model loaded from a file by a scene compiler."""

    file: str | Path

    def validate(self) -> None:
        path = resolve_path(self.file)
        if not path.exists():
            raise FileNotFoundError(f"Model file does not exist: {path}")


@configclass
class KeyPoseCfg:
    """Named robot joint poses sharing one explicit joint order."""

    joint_names: list[str] = []
    poses: dict[str, list[float]] = {}

    def validate(self) -> None:
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("KeyPoseCfg.joint_names must be unique")
        if any(not name for name in self.joint_names):
            raise ValueError("KeyPoseCfg.joint_names must not contain empty names")
        if self.poses and not self.joint_names:
            raise ValueError("KeyPoseCfg.joint_names must not be empty when poses are configured")
        for name, positions in self.poses.items():
            if not name:
                raise ValueError("KeyPoseCfg pose names must not be empty")
            if len(positions) != len(self.joint_names):
                raise ValueError(
                    f"KeyPoseCfg pose {name!r} must contain {len(self.joint_names)} joint positions, "
                    f"got {len(positions)}"
                )
            if any(not math.isfinite(position) for position in positions):
                raise ValueError(f"KeyPoseCfg pose {name!r} joint positions must be finite")


@configclass(kw_only=True)
class BodyCfg(SceneObjCfg):
    """A rigid-body model file attached to the scene as a free body.

    Holds the backend-neutral contract for attaching an external MJCF/URDF
    model (a free-floating body tree): which file, which link is the attach
    root, and optional placement/name decoration. Robots extend this with
    key-pose and actuation semantics; prop objects (balls, boxes, ...) use
    it directly.
    """

    model: ModelFileCfg
    base_link_name: str
    translation: Vec3 | None = None
    rotation: Vec4 | None = None
    prefix: str | None = None
    suffix: str | None = None

    def validate(self, name: str) -> None:
        super().validate(name)
        if not isinstance(self.model, ModelFileCfg):
            raise TypeError(f"BodyCfg.model must contain ModelFileCfg, got {type(self.model).__name__}")
        self.model.validate()
        if not self.base_link_name:
            raise ValueError("BodyCfg.base_link_name must not be empty")
        optional_vec("BodyCfg.translation", self.translation, 3)
        optional_vec("BodyCfg.rotation", self.rotation, 4)

    def resolve_name(self, name: str) -> str:
        return f"{self.prefix or ''}{name}{self.suffix or ''}"

    @property
    def resolved_base_link_name(self) -> str:
        return self.resolve_name(self.base_link_name)


@configclass(kw_only=True)
class RobotCfg(BodyCfg):
    """Base config for a robot instance in a generated scene."""

    key_pose: KeyPoseCfg = KeyPoseCfg()

    def validate(self, name: str) -> None:
        super().validate(name)
        if not isinstance(self.key_pose, KeyPoseCfg):
            raise TypeError(f"RobotCfg.key_pose must contain KeyPoseCfg, got {type(self.key_pose).__name__}")
        self.key_pose.validate()
