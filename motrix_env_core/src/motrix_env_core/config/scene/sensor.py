# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import math
from enum import Enum
from typing import Literal, TypeAlias

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import SceneSensorCfg

RaycastCameraOutput: TypeAlias = Literal[
    "depth",
    "distance",
    "point",
    "normal",
]
RaycastCameraExclude: TypeAlias = Literal["none", "parent_body", "parent_subtree"]
_RAYCAST_CAMERA_OUTPUTS: tuple[RaycastCameraOutput, ...] = (
    "depth",
    "distance",
    "point",
    "normal",
)


@configclass(kw_only=True)
class PinholeCameraCfg:
    """Pixel intrinsics for +X right, +Y down, +Z forward optical coordinates.

    Rays pass through pixel centers ``(u + 0.5, v + 0.5)``. Principal points
    outside the image are supported; only focal lengths must be positive.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.fx, self.fy, self.cx, self.cy)):
            raise ValueError("PinholeCameraCfg intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("PinholeCameraCfg focal lengths must be positive")

    @classmethod
    def from_fov(
        cls, *, width: int, height: int, horizontal_fov: float, vertical_fov: float | None = None
    ) -> "PinholeCameraCfg":
        """Create centered intrinsics from FOV angles in degrees.

        Omitting vertical FOV uses square pixels (``fy == fx``), deriving the
        vertical field of view from the image aspect ratio.
        """
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            raise ValueError("Camera width and height must be positive integers")
        for angle in (horizontal_fov, vertical_fov):
            if angle is not None and (not math.isfinite(angle) or not 0 < angle < 180):
                raise ValueError("Camera FOV must be finite and strictly between 0 and 180 degrees")
        fx = width / (2 * math.tan(math.radians(horizontal_fov) / 2))
        fy = fx if vertical_fov is None else height / (2 * math.tan(math.radians(vertical_fov) / 2))
        return cls(fx=fx, fy=fy, cx=width / 2, cy=height / 2)


@configclass(kw_only=True)
class OrthographicCameraCfg:
    """Spacing in meters between parallel +Z rays on a centered pixel grid."""

    pixel_size_x: float
    pixel_size_y: float

    def validate(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (self.pixel_size_x, self.pixel_size_y)):
            raise ValueError("OrthographicCameraCfg pixel sizes must be finite and positive")


@configclass(kw_only=True)
class RaycastCameraSensorCfg(SceneSensorCfg):
    """Physics raycast camera mounted on a named site in the assembled scene.

    Near/far are inclusive optical-axis depths, not distances along oblique rays.
    Filtering occurs before closest-hit selection; range rejection occurs after it,
    so a surface closer than near still occludes farther surfaces. A zero-distance
    nearest hit invalidates the pixel. ``update_period=0`` samples every physics
    step; other values are seconds. Group IDs are collision geometry groups (0..255),
    not collision masks: None includes all groups and [] includes none.
    """

    site: str
    width: int
    height: int
    projection: PinholeCameraCfg | OrthographicCameraCfg
    far: float
    near: float = 0.0
    update_period: float = 0.0
    exclude: RaycastCameraExclude = "parent_body"
    include_geom_groups: list[int] | None = None
    outputs: tuple[RaycastCameraOutput, ...] = ("depth",)

    def validate(self, name: str) -> None:
        super().validate(name)
        if not self.site:
            raise ValueError("RaycastCameraSensorCfg.site must not be empty")
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0:
            raise ValueError("RaycastCameraSensorCfg width and height must be positive integers")
        if not isinstance(self.projection, (PinholeCameraCfg, OrthographicCameraCfg)):
            raise TypeError("RaycastCameraSensorCfg.projection must be PinholeCameraCfg or OrthographicCameraCfg")
        self.projection.validate()
        if not math.isfinite(self.near) or not math.isfinite(self.far) or not 0 <= self.near < self.far:
            raise ValueError("RaycastCameraSensorCfg requires finite 0 <= near < far")
        if not math.isfinite(self.update_period) or self.update_period < 0:
            raise ValueError("RaycastCameraSensorCfg.update_period must be finite and non-negative")
        if self.exclude not in {"none", "parent_body", "parent_subtree"}:
            raise ValueError("RaycastCameraSensorCfg.exclude must be one of: none, parent_body, parent_subtree")
        if self.include_geom_groups is not None and any(
            type(group) is not int or not 0 <= group <= 255 for group in self.include_geom_groups
        ):
            raise ValueError("RaycastCameraSensorCfg.include_geom_groups must contain integers in 0..255")
        if not self.outputs or any(field not in _RAYCAST_CAMERA_OUTPUTS for field in self.outputs):
            raise ValueError(f"RaycastCameraSensorCfg.outputs must contain only {_RAYCAST_CAMERA_OUTPUTS}")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("RaycastCameraSensorCfg.outputs must be unique")
        if not {"depth", "distance"}.intersection(self.outputs):
            raise ValueError("RaycastCameraSensorCfg.outputs requires at least one distance field")


class ContactReportField(str, Enum):
    """Fields reported by contact sensors."""

    found = "found"
    force = "force"
    torque = "torque"
    dist = "dist"
    pos = "pos"
    normal = "normal"
    tangent = "tangent"


class ContactSensorReduce(str, Enum):
    """Reduction modes supported by contact sensors."""

    none = "none"
    mindist = "mindist"
    maxforce = "maxforce"
    netforce = "netforce"


@configclass(kw_only=True)
class ContactSensorCfg(SceneSensorCfg):
    """Contact sensor matching a pair of named geometries."""

    geom1: str
    geom2: str
    num: int = 1
    data: list[ContactReportField] = [ContactReportField.found]
    reduce: ContactSensorReduce = ContactSensorReduce.none

    def validate(self, name: str) -> None:
        super().validate(name)
        if not self.geom1 or not self.geom2:
            raise ValueError("ContactSensorCfg.geom1 and geom2 must not be empty")
        if self.num <= 0:
            raise ValueError(f"ContactSensorCfg.num must be positive, got {self.num}")
        if not self.data:
            raise ValueError("ContactSensorCfg.data must contain at least one report field")
        unsupported_fields = [field for field in self.data if not isinstance(field, ContactReportField)]
        if unsupported_fields:
            raise ValueError(f"ContactSensorCfg.data contains unsupported fields: {unsupported_fields}")


class FrameObjectKind(str, Enum):
    """Kinds of named scene objects supported by frame sensors."""

    site = "site"
    geom = "geom"
    link = "link"
    link_inertia = "link_inertia"


class FrameRefKind(str, Enum):
    """Reference frames supported by frame sensors."""

    local = "local"
    world = "world"
    object = "object"


class FrameSensorType(str, Enum):
    """Frame quantities supported by frame sensors."""

    framepos = "framepos"
    framequat = "framequat"
    framelinvel = "framelinvel"
    frameangvel = "frameangvel"
    framelinacc = "framelinacc"
    xaxis = "xaxis"
    yaxis = "yaxis"
    zaxis = "zaxis"


@configclass(kw_only=True)
class FrameSensorCfg(SceneSensorCfg):
    """Configuration for sampling a frame quantity from a named scene object.

    The target object must already exist in the assembled scene. The sampled value is expressed in
    the reference frame selected by ``ref_kind``.

    Attributes:
        object_type: Kind of target object: ``site``, ``geom``, ``link``, or ``link_inertia``.
        object_name: Name of the target object whose frame is sampled.
        sensor_type: Quantity to sample. ``framepos`` is position; ``framequat`` is orientation;
            ``framelinvel`` and ``frameangvel`` are linear and angular velocity; ``framelinacc`` is
            linear acceleration; and ``xaxis``, ``yaxis``, and ``zaxis`` are the target frame's axis
            directions.
        ref_kind: Reference frame used to express the sampled value. ``world`` uses the world frame,
            ``local`` uses the target object's local frame, and ``object`` uses another named object's
            frame.
        ref_object_type: Kind of reference object. Required only when ``ref_kind`` is ``object``.
        ref_object_name: Name of the reference object. Required only when ``ref_kind`` is ``object``.
    """

    object_type: FrameObjectKind
    object_name: str
    sensor_type: FrameSensorType
    ref_kind: FrameRefKind = FrameRefKind.world
    ref_object_type: FrameObjectKind | None = None
    ref_object_name: str | None = None

    def validate(self, name: str) -> None:
        super().validate(name)
        if not self.object_name:
            raise ValueError("FrameSensorCfg.object_name must not be empty")
        if self.ref_kind is FrameRefKind.object:
            if self.ref_object_type is None or not self.ref_object_name:
                raise ValueError(
                    "FrameSensorCfg.ref_kind='object' requires ref_object_type and a non-empty ref_object_name"
                )
        elif self.ref_object_type is not None or self.ref_object_name is not None:
            raise ValueError(
                f"FrameSensorCfg.ref_kind={self.ref_kind!r} takes no reference object; "
                "leave ref_object_type and ref_object_name unset"
            )
