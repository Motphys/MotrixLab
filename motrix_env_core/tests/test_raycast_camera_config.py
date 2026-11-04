# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from motrix_env_core.config.scene.sensor import (
    OrthographicCameraCfg,
    PinholeCameraCfg,
    RaycastCameraSensorCfg,
)


def camera(**kwargs):
    return RaycastCameraSensorCfg(
        **dict(
            site="optical",
            width=4,
            height=2,
            projection=PinholeCameraCfg.from_fov(width=4, height=2, horizontal_fov=90),
            far=10,
        )
        | kwargs
    )


def test_fov_uses_pixel_centers_and_square_pixels():
    p = PinholeCameraCfg.from_fov(width=640, height=480, horizontal_fov=90)
    assert p.fx == pytest.approx(320)
    assert p.fy == p.fx
    assert (p.cx, p.cy) == (320, 240)
    p = PinholeCameraCfg.from_fov(width=640, height=480, horizontal_fov=90, vertical_fov=60)
    assert p.fy == pytest.approx(240 / math.tan(math.pi / 6))


@pytest.mark.parametrize("fov", [0, 180, -1, math.inf, math.nan])
def test_fov_rejects_degenerate_angles(fov):
    with pytest.raises(ValueError, match="FOV"):
        PinholeCameraCfg.from_fov(width=4, height=2, horizontal_fov=fov)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"site": ""},
        {"width": 0},
        {"height": 1.5},
        {"width": True},
        {"near": -1},
        {"near": 10},
        {"far": math.inf},
        {"near": math.nan},
        {"update_period": -1},
        {"update_period": math.nan},
        {"include_geom_groups": [-1]},
        {"include_geom_groups": [256]},
        {"include_geom_groups": [True]},
        {"outputs": []},
        {"outputs": ["normal"]},
        {"outputs": ["distance"] * 2},
        {"projection": OrthographicCameraCfg(pixel_size_x=0, pixel_size_y=1)},
        {"projection": PinholeCameraCfg(fx=1, fy=-1, cx=0, cy=0)},
        {"projection": PinholeCameraCfg(fx=1, fy=1, cx=math.nan, cy=0)},
    ],
)
def test_camera_rejects_invalid_boundary_values(kwargs):
    with pytest.raises(ValueError):
        camera(**kwargs).validate("camera")


@pytest.mark.parametrize("groups", [None, [], [0, 255]])
def test_camera_valid_outputs_and_filters(groups):
    cfg = camera(include_geom_groups=groups, outputs=("depth", "distance", "point", "normal"))
    cfg.validate("camera")
    cfg.projection = OrthographicCameraCfg(pixel_size_x=0.1, pixel_size_y=0.2)
    cfg.validate("camera")


def test_camera_mutable_defaults_are_independent():
    first, second = camera(), camera()
    first.outputs = (*first.outputs, "point")
    assert second.outputs == ("depth",)
