# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields
from pathlib import Path

import gymnasium as gym
import motrixsim as mtx
import numpy as np
import pytest
from omegaconf import MISSING, OmegaConf

from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    BodyCfg,
    CompositeTerrainGeneratorCfg,
    ContactReportField,
    ContactSensorCfg,
    ContactSensorReduce,
    DiscreteObstaclesTerrainGeneratorCfg,
    FlatTerrainCfg,
    FlatTerrainGeneratorCfg,
    FrameObjectKind,
    FrameRefKind,
    FrameSensorCfg,
    FrameSensorType,
    GeomCfg,
    HFieldAssetCfg,
    HFieldTerrainCfg,
    LightCfg,
    MaterialCfg,
    MjcfFileCfg,
    ModelFileCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    PyramidSlopeTerrainGeneratorCfg,
    QuantizedTerrainGeneratorCfg,
    RobotCfg,
    SceneAssetCfg,
    SceneAssetsCfg,
    SceneCfg,
    SceneObjCfg,
    SceneObjsCfg,
    SceneSensorCfg,
    SceneSensorsCfg,
    SceneVisualCfg,
    SkyboxCfg,
    StairsTerrainGeneratorCfg,
    SystemCameraCfg,
    TerrainGeneratorCfg,
    TerrainRegionCfg,
    TextureCfg,
    grid_terrain,
    validate_scene_cfg,
)
from motrix_env_core.direct.env import DirectEnv, DirectEnvCfg
from motrix_env_motrixsim.compiler import build_scene_model, build_scene_world
from motrix_envs.config.scene import StandardSceneAssetsCfg, StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid.dex_evt import make_dex_evt_walk_flat_cfg, make_dex_evt_walk_rough_cfg
from motrix_envs.locomotion.humanoid.g1 import make_g129dof_walk_flat_cfg, make_g129dof_walk_rough_cfg
from motrix_envs.locomotion.humanoid.k1 import make_k1_walk_flat_cfg, make_k1_walk_rough_cfg

_CARTPOLE_XML = Path(__file__).parents[1] / "src" / "motrix_envs" / "basic" / "cartpole" / "cartpole.xml"
_GROUND_TEXTURE = Path(__file__).parents[1] / "src" / "motrix_envs" / "common" / "motphys-ground.png"
_HFIELD_FILE = (
    Path(__file__).parents[1]
    / "src"
    / "motrix_envs"
    / "locomotion"
    / "go1"
    / "xmls"
    / "assets"
    / "heightmap_stairs.hfield"
)


def test_scene_cfg_loads_base_model_file():
    scene = SceneCfg(file=_CARTPOLE_XML)

    model = build_scene_model(scene)

    assert "cart" in model.body_names
    assert "slider" in model.joint_names
    assert "slide" in model.actuator_names


def test_scene_cfg_rejects_missing_base_model_file(tmp_path):
    missing_file = tmp_path / "missing.xml"

    with pytest.raises(FileNotFoundError, match="Scene file does not exist"):
        validate_scene_cfg(SceneCfg(file=missing_file))


def test_scene_cfg_builds_flat_terrain_and_prefixed_robot():
    @configclass
    class FlatRobotSceneObjsCfg(SceneObjsCfg):
        scene_floor: FlatTerrainCfg = FlatTerrainCfg()
        cartpole: RobotCfg = RobotCfg(
            model=MjcfFileCfg(file=_CARTPOLE_XML),
            base_link_name="cart",
            prefix="robot0_",
            translation=(1.0, 0.0, 0.0),
        )

    scene = SceneCfg(objs=FlatRobotSceneObjsCfg())

    model = build_scene_model(scene)

    assert "scene_floor" in model.geom_names
    assert "robot0_cart" in model.body_names
    assert "robot0_slider" in model.joint_names


def test_flat_terrain_uses_infinite_plane_at_configured_height():
    scene = StandardSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                height=0.25,
            ),
            robot=RobotCfg(
                model=MjcfFileCfg(file=_CARTPOLE_XML),
                base_link_name="cart",
                prefix="robot_",
            ),
        )
    )
    world = build_scene_world(scene)

    geom = world.hierarchy.geoms[0]
    assert geom.shape == mtx.msd.ShapeType.InfinitePlane
    assert geom.position == pytest.approx([0.0, 0.0, 0.25])
    assert geom.visual.material == "mat_ground"

    texture = world.assets.textures["tex_ground"]
    assert texture.name == "tex_ground"
    assert texture.source.variant == "file"
    assert texture.source.value == _GROUND_TEXTURE.resolve()
    assert texture.type_ == mtx.msd.TextureType.D2
    assert texture.color_space == mtx.msd.ColorSpace.Srgb
    assert texture.gen_mipmaps is True

    material = world.assets.materials["mat_ground"]
    assert material.name == "mat_ground"
    assert material.texture_name == "tex_ground"
    assert material.tex_repeat == pytest.approx([0.4, 0.4])
    assert material.tex_uniform is True
    assert material.metallic == pytest.approx(0.0)
    assert material.roughness == pytest.approx(0.0)


def test_standard_scene_builds_gradient_skybox_and_visual_environment():
    scene = StandardSceneCfg(
        objs=StandardSceneObjsCfg(
            robot=RobotCfg(
                model=MjcfFileCfg(file=_CARTPOLE_XML),
                base_link_name="cart",
                prefix="robot_",
            )
        )
    )
    world = build_scene_world(scene)

    assert world.assets.skybox == "skybox"
    skybox = world.assets.textures["skybox"]
    assert skybox.name == "skybox"
    assert skybox.type_ == mtx.msd.TextureType.Skybox
    assert skybox.source.variant == "gradient"
    assert skybox.source.value.rgb1 == pytest.approx([0.4, 0.4, 0.4])
    assert skybox.source.value.rgb2 == pytest.approx([0.0, 0.0, 0.0])
    assert skybox.source.value.width == pytest.approx(512.0)
    assert skybox.source.value.height == pytest.approx(3072.0)
    assert skybox.source.value.markrgb == pytest.approx([1.0, 1.0, 1.0])
    assert skybox.color_space == mtx.msd.ColorSpace.Srgb
    assert skybox.gen_mipmaps is True
    assert world.visual.ambient_light.color == pytest.approx([0.3, 0.3, 0.3])
    assert world.visual.ambient_light.brightness == pytest.approx(1_000.0)
    assert world.visual.head_light.color == pytest.approx([0.6, 0.6, 0.6])
    assert world.visual.head_light.luminous_power == pytest.approx(1_000.0)
    assert world.visual.haze == pytest.approx([0.1, 0.1, 0.1, 1.0])
    assert world.visual.tonemapping.method == mtx.msd.ToneMappingMethod.None_

    sun = next(light for light in world.hierarchy.lights if light.name == "sun")
    assert sun.name == "sun"
    assert sun.type_.variant == "directional"
    assert sun.type_.value.illuminance == pytest.approx(10_000.0)
    assert sun.position == pytest.approx([0.0, 0.0, 1.5])
    assert sun.direction == pytest.approx([-1.0, -1.0, -1.0])
    assert sun.color == pytest.approx([0.7, 0.7, 0.7])
    assert sun.cast_shadows is True


def test_standard_scene_requires_robot():
    with pytest.raises(ValueError, match="SceneObjsCfg field 'robot' is mandatory and must be provided"):
        validate_scene_cfg(StandardSceneCfg())


def test_blank_scene_preserves_msd_visual_defaults():
    default_world = mtx.msd.World()
    world = build_scene_world(SceneCfg())

    assert world.assets.skybox is None
    assert world.visual.haze is None
    assert world.visual.ambient_light.color == pytest.approx(default_world.visual.ambient_light.color)
    assert world.visual.ambient_light.brightness == pytest.approx(default_world.visual.ambient_light.brightness)
    assert world.visual.head_light.color == pytest.approx(default_world.visual.head_light.color)
    assert world.visual.head_light.luminous_power == pytest.approx(default_world.visual.head_light.luminous_power)
    assert world.visual.tonemapping.method == default_world.visual.tonemapping.method


def test_scene_visual_cfg_applies_explicit_settings():
    scene = SceneCfg(
        visual=SceneVisualCfg(
            ambient_light_color=(0.1, 0.2, 0.3),
            ambient_light_brightness=750.0,
            head_light_color=(0.4, 0.5, 0.6),
            head_light_luminous_power=1_250.0,
            haze=(0.2, 0.3, 0.4, 0.5),
            tone_mapping="aces",
        )
    )

    world = build_scene_world(scene)

    assert world.visual.ambient_light.color == pytest.approx([0.1, 0.2, 0.3])
    assert world.visual.ambient_light.brightness == pytest.approx(750.0)
    assert world.visual.head_light.color == pytest.approx([0.4, 0.5, 0.6])
    assert world.visual.head_light.luminous_power == pytest.approx(1_250.0)
    assert world.visual.haze == pytest.approx([0.2, 0.3, 0.4, 0.5])
    assert world.visual.tonemapping.method == mtx.msd.ToneMappingMethod.Aces


def test_scene_cfg_builds_directional_light():
    @configclass
    class LightSceneObjsCfg(SceneObjsCfg):
        sun: LightCfg = LightCfg(
            position=(1.0, 2.0, 3.0),
            direction=(0.0, 1.0, -1.0),
            color=(0.8, 0.7, 0.6),
            illuminance=12_000.0,
            cast_shadows=False,
        )

    scene = SceneCfg(objs=LightSceneObjsCfg())

    light = build_scene_world(scene).hierarchy.lights[0]

    assert light.name == "sun"
    assert light.type_.variant == "directional"
    assert light.type_.value.illuminance == pytest.approx(12_000.0)
    assert light.position == pytest.approx([1.0, 2.0, 3.0])
    assert light.direction == pytest.approx([0.0, 1.0, -1.0])
    assert light.color == pytest.approx([0.8, 0.7, 0.6])
    assert light.cast_shadows is False


def test_scene_cfg_builds_named_contact_sensors_after_objects():
    @configclass
    class ContactSensorsCfg(SceneSensorsCfg):
        cart_floor: ContactSensorCfg = ContactSensorCfg(
            geom1="floor",
            geom2="robot_cart",
            num=2,
            data=[ContactReportField.found, ContactReportField.dist],
            reduce=ContactSensorReduce.mindist,
        )

    scene = StandardSceneCfg(
        objs=StandardSceneObjsCfg(
            robot=RobotCfg(
                model=MjcfFileCfg(file=_CARTPOLE_XML),
                base_link_name="cart",
                prefix="robot_",
            )
        ),
        sensors=ContactSensorsCfg(),
    )

    world = build_scene_world(scene)
    sensor = world.sensors.contact[0]

    assert [name for name, _ in scene.iter_sensors()] == ["cart_floor"]
    assert sensor.name == "cart_floor"
    assert sensor.match_.variant == "geom_pair"
    assert sensor.match_.value == ("floor", "robot_cart")
    assert sensor.max_num == 2
    assert sensor.report.found is True
    assert sensor.report.dist is True
    assert sensor.report.force is False
    assert sensor.reduce == mtx.msd.ContactSensorReduce.MinDist
    assert build_scene_model(scene).num_sensors > 0


def test_contact_sensor_reduce_supports_lowercase_omegaconf_values():
    cfg = OmegaConf.structured(
        ContactSensorCfg(
            geom1="floor",
            geom2="foot",
            reduce=ContactSensorReduce.mindist,
        )
    )

    OmegaConf.update(cfg, "data", ["force", "dist"])
    OmegaConf.update(cfg, "reduce", "maxforce")
    typed_cfg = OmegaConf.to_object(cfg)

    assert typed_cfg.data == [ContactReportField.force, ContactReportField.dist]
    assert typed_cfg.reduce is ContactSensorReduce.maxforce


def test_scene_cfg_builds_named_frame_sensors_after_objects():
    @configclass
    class FrameSensorsCfg(SceneSensorsCfg):
        floor_pos: FrameSensorCfg = FrameSensorCfg(
            object_type=FrameObjectKind.geom,
            object_name="floor",
            sensor_type=FrameSensorType.framepos,
        )
        cart_linvel: FrameSensorCfg = FrameSensorCfg(
            object_type=FrameObjectKind.geom,
            object_name="robot_cart",
            sensor_type=FrameSensorType.framelinvel,
            ref_kind=FrameRefKind.local,
        )
        cart_up: FrameSensorCfg = FrameSensorCfg(
            object_type=FrameObjectKind.geom,
            object_name="robot_cart",
            sensor_type=FrameSensorType.zaxis,
            ref_kind=FrameRefKind.object,
            ref_object_type=FrameObjectKind.geom,
            ref_object_name="floor",
        )

    scene = StandardSceneCfg(
        objs=StandardSceneObjsCfg(
            robot=RobotCfg(
                model=MjcfFileCfg(file=_CARTPOLE_XML),
                base_link_name="cart",
                prefix="robot_",
            )
        ),
        sensors=FrameSensorsCfg(),
    )

    world = build_scene_world(scene)
    floor_pos, cart_linvel, cart_up = world.sensors.frame

    assert [name for name, _ in scene.iter_sensors()] == ["floor_pos", "cart_linvel", "cart_up"]
    assert floor_pos.name == "floor_pos"
    assert floor_pos.object_type.variant == "geom"
    assert floor_pos.object_type.value == "floor"
    assert floor_pos.sensor_type == mtx.msd.FrameSensorType.FramePos
    assert floor_pos.ref_frame.variant == "world"
    assert cart_linvel.sensor_type == mtx.msd.FrameSensorType.FrameLinVel
    assert cart_linvel.ref_frame.variant == "local"
    assert cart_up.sensor_type == mtx.msd.FrameSensorType.ZAxis
    assert cart_up.ref_frame.variant == "object"
    assert cart_up.ref_frame.value.variant == "geom"
    assert cart_up.ref_frame.value.value == "floor"
    assert build_scene_model(scene).num_sensors > 0


def test_frame_sensor_enums_support_lowercase_omegaconf_values():
    cfg = OmegaConf.structured(
        FrameSensorCfg(
            object_type=FrameObjectKind.geom,
            object_name="foot",
            sensor_type=FrameSensorType.framepos,
            ref_kind=FrameRefKind.object,
            ref_object_type=FrameObjectKind.site,
            ref_object_name="base",
        )
    )

    OmegaConf.update(cfg, "object_type", "link")
    OmegaConf.update(cfg, "sensor_type", "framequat")
    OmegaConf.update(cfg, "ref_kind", "object")
    OmegaConf.update(cfg, "ref_object_type", "link_inertia")
    typed_cfg = OmegaConf.to_object(cfg)

    assert typed_cfg.object_type is FrameObjectKind.link
    assert typed_cfg.sensor_type is FrameSensorType.framequat
    assert typed_cfg.ref_kind is FrameRefKind.object
    assert typed_cfg.ref_object_type is FrameObjectKind.link_inertia


def test_light_cfg_rejects_negative_illuminance():
    @configclass
    class InvalidLightSceneObjsCfg(SceneObjsCfg):
        sun: LightCfg = LightCfg(illuminance=-1.0)

    scene = SceneCfg(objs=InvalidLightSceneObjsCfg())

    with pytest.raises(ValueError, match="illuminance must be non-negative"):
        validate_scene_cfg(scene)


def test_scene_cfg_builds_hfield_terrain():
    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain_hfield: HFieldAssetCfg = HFieldAssetCfg(file=_HFIELD_FILE)

    @configclass
    class TerrainSceneObjsCfg(SceneObjsCfg):
        hfield_floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain_hfield")

    scene = SceneCfg(
        assets=TerrainAssetsCfg(),
        objs=TerrainSceneObjsCfg(),
    )

    model = build_scene_model(scene)
    hfield = model.get_hfield("terrain_hfield")

    assert model.num_hfields == 1
    assert hfield is not None
    assert hfield.bound[[0, 1, 3, 4]] == pytest.approx([-8.0, -8.0, 8.0, 8.0])
    assert "hfield_floor" in model.geom_names


def test_scene_cfg_builds_reproducible_procedural_hfield_terrain():
    generator = NoiseTerrainGeneratorCfg(seed=42, height_scale=0.15)

    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain_hfield: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
            generator=generator,
            size=(8.0, 6.0),
            shape=(5, 4),
        )

    @configclass
    class TerrainSceneObjsCfg(SceneObjsCfg):
        floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain_hfield")

    scene = SceneCfg(assets=TerrainAssetsCfg(), objs=TerrainSceneObjsCfg())
    world = build_scene_world(scene)
    hfield = world.assets.hfields["terrain_hfield"]
    heights = hfield.source_type.value["hfield"]

    assert hfield.source_type.variant == "buffer"
    assert hfield.source_type.value["name"] == "terrain_hfield"
    assert hfield.nrow == 5
    assert hfield.ncol == 4
    assert hfield.size == pytest.approx([4.0, 3.0])
    assert hfield.height_scale == pytest.approx(0.15)
    assert heights.dtype == np.float32
    assert heights == pytest.approx(generator.generate((8.0, 6.0), (5, 4)).reshape(-1))
    assert np.all(heights >= 0.0)
    assert np.all(heights <= 1.0)
    model = build_scene_model(scene)
    compiled_hfield = model.get_hfield("terrain_hfield")
    assert model.num_hfields == 1
    assert compiled_hfield is not None
    assert compiled_hfield.bound[[0, 1, 3, 4]] == pytest.approx([-4.0, -3.0, 4.0, 3.0])


def test_terrain_generator_cfg_is_abstract():
    with pytest.raises(TypeError, match="abstract class TerrainGeneratorCfg"):
        TerrainGeneratorCfg()


def test_procedural_hfield_supports_hydra_overrides():
    cfg = OmegaConf.structured(
        ProceduralHFieldAssetCfg(
            generator=NoiseTerrainGeneratorCfg(),
        )
    )

    OmegaConf.update(cfg, "generator.seed", 7)
    OmegaConf.update(cfg, "generator.height_scale", 0.2)
    OmegaConf.update(cfg, "generator.flip_y", True)
    OmegaConf.update(cfg, "size", [4.0, 3.0])
    OmegaConf.update(cfg, "shape", [8, 6])
    asset = OmegaConf.to_object(cfg)

    assert isinstance(asset, ProceduralHFieldAssetCfg)
    assert isinstance(asset.generator, NoiseTerrainGeneratorCfg)
    assert asset.generator.seed == 7
    assert asset.generator.height_scale == pytest.approx(0.2)
    assert asset.generator.flip_y is True

    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain: ProceduralHFieldAssetCfg = asset

    world = build_scene_world(SceneCfg(assets=TerrainAssetsCfg()))
    assert world.assets.hfields["terrain"].nrow == 8
    assert world.assets.hfields["terrain"].ncol == 6


@pytest.mark.parametrize(
    ("asset", "match"),
    [
        (
            ProceduralHFieldAssetCfg(generator=NoiseTerrainGeneratorCfg(), size=(0.0, 1.0)),
            "size must be finite and positive",
        ),
        (
            ProceduralHFieldAssetCfg(generator=NoiseTerrainGeneratorCfg(), shape=(1, 4)),
            "shape must contain two dimensions of at least 2",
        ),
        (
            ProceduralHFieldAssetCfg(generator=NoiseTerrainGeneratorCfg(height_scale=-0.1)),
            "height_scale must be finite and positive",
        ),
        (
            # Generators divide by height_scale when normalizing, so zero is rejected.
            ProceduralHFieldAssetCfg(generator=NoiseTerrainGeneratorCfg(height_scale=0.0)),
            "height_scale must be finite and positive",
        ),
    ],
)
def test_procedural_hfield_rejects_invalid_configuration(asset, match):
    with pytest.raises(ValueError, match=match):
        asset.validate("terrain")


@pytest.mark.parametrize(
    ("heights", "match"),
    [
        ([[0.0, 0.0]], "returned shape"),
        ([[0.0, float("nan")], [0.0, 0.0]], "returned non-finite heights"),
    ],
)
def test_procedural_hfield_rejects_invalid_generator_output(heights, match):
    @configclass
    class FixedTerrainGeneratorCfg(TerrainGeneratorCfg):
        heights: list[list[float]]

        def generate(self, size: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
            del size, shape
            return np.asarray(self.heights, dtype=np.float32)

    asset = ProceduralHFieldAssetCfg(
        generator=FixedTerrainGeneratorCfg(heights=heights),
        shape=(2, 2),
    )

    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain: ProceduralHFieldAssetCfg = asset

    with pytest.raises(ValueError, match=match):
        build_scene_world(SceneCfg(assets=TerrainAssetsCfg()))


def test_stairs_terrain_runs_along_configured_axis():
    # axis semantics contract: "x" varies along shape[0] rows, "y" along shape[1] columns.
    generator = StairsTerrainGeneratorCfg(
        axis="x",
        profile="ascending",
        step_count=4,
        step_height=0.05,
        height_scale=0.15,
    )

    heights = generator.generate((8.0, 8.0), (8, 8))

    assert np.all(heights == heights[:, :1])
    row_levels = heights[:, 0]
    assert sorted(np.unique(row_levels)) == pytest.approx([0.0, 1.0 / 3, 2.0 / 3, 1.0])
    assert np.all(np.diff(row_levels) >= 0.0)

    transposed = StairsTerrainGeneratorCfg(
        axis="y",
        profile="ascending",
        step_count=4,
        step_height=0.05,
        height_scale=0.15,
    ).generate((8.0, 8.0), (8, 8))
    assert np.all(transposed == transposed[0:1, :])
    assert np.all(transposed[:, 0] == heights[0, :])


def test_stairs_terrain_profiles():
    pyramid = StairsTerrainGeneratorCfg(
        profile="pyramid",
        step_count=8,
        step_height=0.1,
        height_scale=0.7,
    ).generate((8.0, 8.0), (8, 8))[:, 0]
    assert np.all(np.diff(pyramid[:4]) > 0.0)
    assert np.all(np.diff(pyramid[4:]) < 0.0)
    assert pyramid[0] == pytest.approx(0.0)
    assert pyramid.max() == pyramid[3]

    pit = StairsTerrainGeneratorCfg(
        profile="inverted_pyramid",
        step_count=8,
        step_height=0.1,
        height_scale=0.7,
    ).generate((8.0, 8.0), (8, 8))[:, 0]
    # The central pit floor is the lowest point and the edges are the highest.
    assert pit[3:5].min() == pit.min()
    assert pit[0] == pit.max()
    assert np.all(pit >= 0.0)
    assert np.all(pit <= 1.0)


@pytest.mark.parametrize(
    "generator",
    [
        StairsTerrainGeneratorCfg(step_count=4, step_height=0.2, height_scale=0.5),
        StairsTerrainGeneratorCfg(step_count=3),
        DiscreteObstaclesTerrainGeneratorCfg(height=0.1, height_scale=0.1),
        DiscreteObstaclesTerrainGeneratorCfg(size_min=0.3, size_max=0.2),
        QuantizedTerrainGeneratorCfg(source=FlatTerrainGeneratorCfg(), levels=1),
        TerrainRegionCfg(generator=FlatTerrainGeneratorCfg(), size=(1.5, 0.5)),
        TerrainRegionCfg(generator=FlatTerrainGeneratorCfg(), blend=0.9),
    ],
)
def test_terrain_generators_reject_invalid_configuration(generator):
    with pytest.raises(ValueError):
        generator.validate()


def test_stairs_terrain_lays_radial_rings():
    pit = StairsTerrainGeneratorCfg(
        axis="radial",
        profile="ascending",
        step_count=8,
        step_height=0.1,
        height_scale=0.7,
    ).generate((8.0, 8.0), (16, 16))

    # Concentric rings: the center cell is the lowest, the edges are the highest.
    assert pit[7, 7] == pit.min()
    assert pit[0, 0] == pit.max() == pit[0, 15] == pit[15, 0]
    # Symmetric under both axis flips.
    assert np.all(pit == pit[::-1, :])
    assert np.all(pit == pit[:, ::-1])
    # Ring levels are multiples of the normalized step height.
    assert sorted(np.unique(pit)) == pytest.approx([index * 0.1 / 0.7 for index in range(8)])

    peak = StairsTerrainGeneratorCfg(
        axis="radial",
        profile="descending",
        step_count=8,
        step_height=0.1,
        height_scale=0.7,
    ).generate((8.0, 8.0), (16, 16))
    assert peak[7, 7] == peak.max()
    assert np.all(peak + pit == pit.max())


def test_quantized_terrain_snaps_heights_to_levels():
    generator = QuantizedTerrainGeneratorCfg(
        source=NoiseTerrainGeneratorCfg(seed=11),
        levels=5,
    )

    heights = generator.generate((8.0, 8.0), (16, 16))

    allowed = {round(index / 4, 6) for index in range(5)}
    snapped = {round(float(value), 6) for value in np.unique(heights)}
    assert snapped <= allowed
    assert len(snapped) > 1
    assert np.all(heights >= 0.0)
    assert np.all(heights <= 1.0)


def test_discrete_obstacles_terrain_is_reproducible():
    generator = DiscreteObstaclesTerrainGeneratorCfg(seed=3, count=6, height=0.05, height_scale=0.2)

    heights = generator.generate((8.0, 8.0), (32, 32))
    replay = generator.generate((8.0, 8.0), (32, 32))

    assert np.all(heights == replay)
    # Flat base plus a single bump level at the signed obstacle height.
    assert sorted(np.unique(heights)) == pytest.approx([0.5, 0.5 + 0.05 / 0.2])
    assert np.any(heights > 0.5)


def test_stairs_terrain_step_width_and_platform():
    generator = StairsTerrainGeneratorCfg(
        axis="radial",
        profile="descending",
        step_count=8,
        step_height=0.1,
        step_width=0.5,
        platform_width=2.0,
        height_scale=1.0,
    )

    heights = generator.generate((8.0, 8.0), (64, 64))

    # The central platform is flat at the top step level.
    assert np.all(heights[28:36, 28:36] == heights[32, 32])
    assert np.isclose(heights[32, 32], 7 * 0.1 / 1.0)
    # Rings descend outward with the configured tread width: mid-edge and corner
    # cells sit strictly below the platform, and rings repeat every ~step_width.
    assert heights[32, 4] < heights[32, 32]
    assert heights[2, 2] < heights[32, 4]
    assert np.all(heights >= 0.0) and np.all(heights <= 1.0)


def test_stairs_terrain_platform_rejects_negative_width():
    generator = StairsTerrainGeneratorCfg(step_count=4, step_height=0.05, height_scale=0.5, platform_width=-1.0)

    with pytest.raises(ValueError, match="platform_width"):
        generator.validate()


def test_discrete_obstacles_terrain_choice_mode_and_platform():
    generator = DiscreteObstaclesTerrainGeneratorCfg(
        seed=2,
        count=20,
        size_min=0.05,
        size_max=0.1,
        height=0.2,
        height_scale=0.5,
        height_mode="choice",
        platform_width=2.0,
    )

    heights = generator.generate((8.0, 8.0), (80, 80))

    # Choice mode draws from +/- full and +/- half of the obstacle height.
    magnitude = 0.2 / 0.5
    unique = sorted(np.unique(np.round(heights, 4)).tolist())
    assert unique == pytest.approx(
        sorted([0.5, 0.5 + magnitude, 0.5 + magnitude / 2, 0.5 - magnitude / 2, 0.5 - magnitude])
    )
    assert np.any(heights > 0.5 + magnitude / 4)
    assert np.any(heights < 0.5 - magnitude / 4)
    # The central platform stays clear at base level.
    assert np.all(heights[38:42, 38:42] == 0.5)


def test_discrete_obstacles_terrain_rejects_unknown_height_mode():
    generator = DiscreteObstaclesTerrainGeneratorCfg(height_mode="random")

    with pytest.raises(ValueError, match="height_mode"):
        generator.validate()


def test_pyramid_slope_terrain_rises_to_center():
    generator = PyramidSlopeTerrainGeneratorCfg(slope=0.25, height_scale=1.0)

    heights = generator.generate((8.0, 8.0), (64, 64))

    # The peak sits at the center: slope * half extent = 0.25 * 4 m = 1 m.
    # Cell centers sample slightly inside the boundary, so the max is just below 1.
    assert heights.max() == pytest.approx(1.0, abs=0.02)
    assert heights[32, 32] == heights.max()
    assert heights[0, 0] == heights.min()
    # Smooth monotonic rise from each edge midpoint to the center, then fall.
    assert np.all(np.diff(heights[:32, 32]) >= -1e-6)
    assert np.all(np.diff(heights[32:, 32]) <= 1e-6)
    assert np.all(np.diff(heights[32, :32]) >= -1e-6)
    assert np.all(np.diff(heights[32, 32:]) <= 1e-6)


def test_pyramid_slope_terrain_inverted_dips_at_center():
    generator = PyramidSlopeTerrainGeneratorCfg(slope=0.25, inverted=True, height_scale=1.0)

    heights = generator.generate((8.0, 8.0), (64, 64))

    assert heights[32, 32] == heights.min() == pytest.approx(0.0)
    # Corner cells sample 0.0625 m inside the boundary, so the rim tops out just below 1.
    assert heights.max() == pytest.approx(1.0, abs=0.04)


def test_pyramid_slope_terrain_rejects_peak_above_height_scale():
    generator = PyramidSlopeTerrainGeneratorCfg(slope=0.5, height_scale=0.5)

    with pytest.raises(ValueError, match="height_scale"):
        generator.generate((8.0, 8.0), (16, 16))


def test_noise_terrain_downsampled_scale_smooths_speckle():
    def jaggedness(heights: np.ndarray) -> float:
        return float(np.abs(np.diff(heights, axis=0)).mean() + np.abs(np.diff(heights, axis=1)).mean())

    rough = NoiseTerrainGeneratorCfg(seed=3, height_scale=0.1)
    smooth = NoiseTerrainGeneratorCfg(seed=3, height_scale=0.1, downsampled_scale=0.4)

    rough_heights = rough.generate((8.0, 8.0), (80, 80))
    smooth_heights = smooth.generate((8.0, 8.0), (80, 80))

    assert jaggedness(smooth_heights) < jaggedness(rough_heights)
    assert np.all(smooth_heights >= 0.0) and np.all(smooth_heights <= 1.0)
    assert np.array_equal(smooth_heights, smooth.generate((8.0, 8.0), (80, 80)))


def test_noise_terrain_rejects_invalid_downsampled_scale():
    generator = NoiseTerrainGeneratorCfg(downsampled_scale=-0.1)

    with pytest.raises(ValueError, match="downsampled_scale"):
        generator.validate()


def test_composite_terrain_conserves_physical_heights_and_stays_normalized():
    composite = CompositeTerrainGeneratorCfg(
        base=FlatTerrainGeneratorCfg(height=0.25, height_scale=0.4),
        height_scale=0.4,
        regions=(
            TerrainRegionCfg(
                generator=StairsTerrainGeneratorCfg(
                    axis="x",
                    profile="ascending",
                    step_count=4,
                    step_height=0.05,
                    height_scale=0.15,
                ),
                center=(0.5, 0.75),
                size=(1.0, 0.5),
                blend=0.25,
            ),
            TerrainRegionCfg(
                generator=FlatTerrainGeneratorCfg(height=1.0, height_scale=0.4),
                center=(0.5, 0.75),
                size=(0.25, 0.25),
            ),
        ),
    )

    heights = composite.generate((16.0, 16.0), (16, 16))

    assert np.all(heights >= 0.0)
    assert np.all(heights <= 1.0)
    # Later regions overwrite earlier ones: the flat patch shows through at full
    # height where it overlaps the stairs region.
    assert heights[8, 11] == pytest.approx(1.0)
    # Inside the stairs region (rows flush, right edge flush; the left edge is an
    # interior seam) the physical step size is conserved after sub-generator rescaling.
    core = np.unique(heights[6:10, 15])
    assert np.diff(core) * 0.4 == pytest.approx(0.05)
    assert core[0] * 0.4 == pytest.approx(0.05)
    # The blend margin mixes the flat base into the stairs across the interior seam
    # (row 3 lies on stairs level 0, i.e. normalized height 0).
    assert heights[3, 8] == pytest.approx(0.25)
    assert heights[3, 9] == pytest.approx(0.5 * 0.25)
    assert heights[3, 8] > heights[3, 9] > heights[3, 12]


def test_composite_terrain_rejects_sub_generator_above_height_scale():
    composite = CompositeTerrainGeneratorCfg(
        base=FlatTerrainGeneratorCfg(),
        height_scale=0.05,
        regions=(
            TerrainRegionCfg(
                generator=NoiseTerrainGeneratorCfg(seed=1, height_scale=0.2),
                center=(0.5, 0.5),
                size=(0.5, 0.5),
            ),
        ),
    )

    with pytest.raises(ValueError, match="exceeds its composite height_scale"):
        composite.generate((16.0, 16.0), (16, 16))


def test_grid_terrain_lays_out_cells():
    composite = grid_terrain(
        [
            [FlatTerrainGeneratorCfg(), StairsTerrainGeneratorCfg(step_count=3, step_height=0.05, height_scale=0.1)],
            [StairsTerrainGeneratorCfg(step_count=3, step_height=0.05, height_scale=0.1), FlatTerrainGeneratorCfg()],
        ],
        blend=0.1,
    )

    assert isinstance(composite, CompositeTerrainGeneratorCfg)
    assert len(composite.regions) == 4
    assert composite.height_scale == pytest.approx(0.1)
    assert composite.regions[0].center == pytest.approx((0.25, 0.25))
    assert composite.regions[3].center == pytest.approx((0.75, 0.75))
    assert all(region.blend == pytest.approx(0.1) for region in composite.regions)

    heights = composite.generate((16.0, 16.0), (16, 16))
    assert np.all(heights >= 0.0)
    assert np.all(heights <= 1.0)


def test_composite_terrain_supports_hydra_overrides():
    cfg = OmegaConf.structured(
        ProceduralHFieldAssetCfg(
            generator=grid_terrain(
                [
                    [
                        FlatTerrainGeneratorCfg(),
                        StairsTerrainGeneratorCfg(step_count=8, step_height=0.01, height_scale=0.15),
                    ],
                    [
                        StairsTerrainGeneratorCfg(step_count=8, step_height=0.01, height_scale=0.15),
                        FlatTerrainGeneratorCfg(),
                    ],
                ]
            ),
            size=(4.0, 3.0),
            shape=(8, 6),
        )
    )

    OmegaConf.update(cfg, "generator.height_scale", 0.2)
    OmegaConf.update(cfg, "generator.regions.1.generator.step_height", 0.02)
    OmegaConf.update(cfg, "generator.regions.1.blend", 0.2)
    asset = OmegaConf.to_object(cfg)

    assert isinstance(asset.generator, CompositeTerrainGeneratorCfg)
    assert asset.generator.height_scale == pytest.approx(0.2)
    assert asset.generator.regions[1].generator.step_height == pytest.approx(0.02)
    assert asset.generator.regions[1].blend == pytest.approx(0.2)

    @configclass
    class TerrainAssetsCfg(SceneAssetsCfg):
        terrain: ProceduralHFieldAssetCfg = asset

    world = build_scene_world(SceneCfg(assets=TerrainAssetsCfg()))
    hfield = world.assets.hfields["terrain"]
    assert hfield.nrow == 8
    assert hfield.ncol == 6
    assert hfield.height_scale == pytest.approx(0.2)
    heights = hfield.source_type.value["hfield"].reshape(8, 6)
    assert np.all(heights >= 0.0)
    assert np.all(heights <= 1.0)


@pytest.mark.parametrize(
    ("cfg", "expected_body", "expected_hfields"),
    [
        (make_g129dof_walk_flat_cfg(), "pelvis", 0),
        (make_g129dof_walk_rough_cfg(), "pelvis", 1),
        (make_dex_evt_walk_flat_cfg(), "pelvis", 0),
        (make_dex_evt_walk_rough_cfg(), "pelvis", 1),
        (make_k1_walk_flat_cfg(), "Trunk", 0),
        (make_k1_walk_rough_cfg(), "Trunk", 1),
    ],
)
def test_humanoid_walk_scene_is_assembled_from_robot_and_floor(cfg, expected_body, expected_hfields):
    cfg = OmegaConf.to_object(OmegaConf.structured(cfg))
    assert cfg.scene.file is None

    model = build_scene_model(cfg.scene, cfg.sim)

    assert expected_body in model.body_names
    assert "floor" in model.geom_names
    assert model.num_hfields == expected_hfields


def test_scene_objs_preserve_inherited_field_order():
    @configclass
    class RobotSceneObjsCfg(StandardSceneObjsCfg):
        floor: FlatTerrainCfg = FlatTerrainCfg(
            material="mat_ground",
            height=0.5,
        )
        sun: LightCfg | None = None
        robot: RobotCfg = RobotCfg(
            model=MjcfFileCfg(file=_CARTPOLE_XML),
            base_link_name="cart",
        )

    objs = RobotSceneObjsCfg()

    assert [cfg_field.name for cfg_field in fields(RobotSceneObjsCfg)] == ["robot", "floor", "sun"]
    assert list(objs) == ["robot", "floor"]
    assert objs["floor"].height == pytest.approx(0.5)


def test_scene_assets_use_field_names_and_support_inheritance():
    @configclass
    class CustomAssetsCfg(StandardSceneAssetsCfg):
        mat_ground: MaterialCfg = MaterialCfg(
            texture="tex_ground",
            roughness=0.8,
        )
        tex_detail: TextureCfg = TextureCfg(file=_GROUND_TEXTURE, color_space="linear")

    assets = CustomAssetsCfg()

    assert list(assets) == ["skybox", "tex_ground", "mat_ground", "tex_detail"]
    assert len(assets) == 4
    assert assets["mat_ground"].roughness == pytest.approx(0.8)
    assert assets["tex_detail"].color_space == "linear"
    assert "name" not in {cfg_field.name for cfg_field in fields(SceneAssetCfg)}
    assert "name" not in {cfg_field.name for cfg_field in fields(TextureCfg)}


def test_standard_scene_supports_hydra_field_overrides():
    @configclass
    class HydraSceneCfg(StandardSceneCfg):
        objs: StandardSceneObjsCfg = StandardSceneObjsCfg(
            robot=RobotCfg(
                model=MjcfFileCfg(file=_CARTPOLE_XML),
                base_link_name="cart",
            )
        )

    cfg = OmegaConf.structured(HydraSceneCfg)

    OmegaConf.update(cfg, "assets.mat_ground.roughness", 0.7)
    OmegaConf.update(cfg, "assets.skybox.color_top", [0.5, 0.5, 0.5])
    OmegaConf.update(cfg, "visual.ambient_light_brightness", 900.0)
    OmegaConf.update(cfg, "visual.tone_mapping", "aces")
    OmegaConf.update(cfg, "system_camera.distance", 4.0)
    OmegaConf.update(cfg, "objs.floor.height", 0.25)
    OmegaConf.update(cfg, "objs.sun.illuminance", 8_000.0)
    OmegaConf.update(cfg, "objs.robot.prefix", "robot0_")
    typed_cfg = OmegaConf.to_object(cfg)

    assert OmegaConf.select(cfg, "assets.mat_ground.texture") == "tex_ground"
    assert OmegaConf.select(cfg, "assets.mat_ground.roughness") == pytest.approx(0.7)
    assert isinstance(typed_cfg, HydraSceneCfg)
    assert isinstance(typed_cfg.assets, StandardSceneAssetsCfg)
    assert typed_cfg.assets.mat_ground.roughness == pytest.approx(0.7)
    assert typed_cfg.assets.skybox.color_top == pytest.approx((0.5, 0.5, 0.5))
    assert typed_cfg.visual.ambient_light_brightness == pytest.approx(900.0)
    assert typed_cfg.visual.tone_mapping == "aces"
    assert typed_cfg.system_camera.distance == pytest.approx(4.0)
    assert isinstance(typed_cfg.objs, StandardSceneObjsCfg)
    assert typed_cfg.objs.floor.height == pytest.approx(0.25)
    assert typed_cfg.objs.sun.illuminance == pytest.approx(8_000.0)
    assert typed_cfg.objs.robot.prefix == "robot0_"


def test_standard_scene_components_can_be_overridden_or_disabled():
    scene = StandardSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                height=0.5,
            ),
            sun=None,
            robot=RobotCfg(
                model=MjcfFileCfg(file=_CARTPOLE_XML),
                base_link_name="cart",
            ),
        )
    )

    assert [name for name, _ in scene.iter_assets()] == ["skybox", "tex_ground", "mat_ground"]
    assert [name for name, _ in scene.iter_objs()] == ["robot", "floor"]


def test_scene_cfg_rejects_invalid_asset_references():
    @configclass
    class MissingTextureAssetsCfg(SceneAssetsCfg):
        ground_material: MaterialCfg = MaterialCfg(texture="missing_texture")

    @configclass
    class MissingMaterialSceneObjsCfg(SceneObjsCfg):
        floor: FlatTerrainCfg = FlatTerrainCfg(material="missing_material")

    @configclass
    class MissingHFieldSceneObjsCfg(SceneObjsCfg):
        floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="missing_hfield")

    with pytest.raises(ValueError, match="must reference a TextureCfg"):
        validate_scene_cfg(SceneCfg(assets=MissingTextureAssetsCfg()))

    with pytest.raises(ValueError, match="must reference a MaterialCfg"):
        validate_scene_cfg(SceneCfg(objs=MissingMaterialSceneObjsCfg()))

    with pytest.raises(ValueError, match="must reference an HFieldAssetCfg"):
        validate_scene_cfg(SceneCfg(objs=MissingHFieldSceneObjsCfg()))


def test_scene_assets_reject_non_asset_fields():
    @configclass
    class InvalidAssetsCfg(SceneAssetsCfg):
        invalid: int = 1

    with pytest.raises(TypeError, match="'invalid' must contain SceneAssetCfg or None"):
        validate_scene_cfg(SceneCfg(assets=InvalidAssetsCfg()))


def test_scene_sensors_reject_invalid_fields_and_contact_options():
    @configclass
    class MissingSensorsCfg(SceneSensorsCfg):
        required: ContactSensorCfg = MISSING

    @configclass
    class InvalidSensorsCfg(SceneSensorsCfg):
        invalid: int = 1

    @configclass
    class InvalidContactSensorsCfg(SceneSensorsCfg):
        invalid_contact: ContactSensorCfg = ContactSensorCfg(
            geom1="floor",
            geom2="foot",
            data=["unknown"],
        )

    with pytest.raises(ValueError, match="SceneSensorsCfg field 'required' is mandatory and must be provided"):
        validate_scene_cfg(SceneCfg(sensors=MissingSensorsCfg()))

    with pytest.raises(TypeError, match="'invalid' must contain SceneSensorCfg or None"):
        validate_scene_cfg(SceneCfg(sensors=InvalidSensorsCfg()))

    with pytest.raises(ValueError, match="unsupported fields"):
        validate_scene_cfg(SceneCfg(sensors=InvalidContactSensorsCfg()))

    assert "name" not in {cfg_field.name for cfg_field in fields(SceneSensorCfg)}
    assert "name" not in {cfg_field.name for cfg_field in fields(ContactSensorCfg)}


def test_scene_cfg_rejects_multiple_skyboxes():
    @configclass
    class MultipleSkyboxesCfg(SceneAssetsCfg):
        skybox: SkyboxCfg = SkyboxCfg()
        alternate_skybox: SkyboxCfg = SkyboxCfg()

    with pytest.raises(ValueError, match="at most one SkyboxCfg"):
        validate_scene_cfg(SceneCfg(assets=MultipleSkyboxesCfg()))


@pytest.mark.parametrize(
    "visual, match",
    [
        (SceneVisualCfg(haze=(0.1, 0.2, 0.3)), "haze must contain 4 values"),
        (SceneVisualCfg(ambient_light_brightness=-1.0), "ambient_light_brightness must be non-negative"),
        (SceneVisualCfg(head_light_luminous_power=-1.0), "head_light_luminous_power must be non-negative"),
        (SceneVisualCfg(tone_mapping="invalid"), "tone_mapping must be None, 'none', or 'aces'"),
    ],
)
def test_scene_visual_cfg_rejects_invalid_values(visual, match):
    with pytest.raises(ValueError, match=match):
        validate_scene_cfg(SceneCfg(visual=visual))


@pytest.mark.parametrize(
    "system_camera, match",
    [
        (SystemCameraCfg(lookat=(0.0, 0.0)), "scene.system_camera.lookat must contain 3 values"),
        (SystemCameraCfg(distance=0.0), "scene.system_camera.distance must be positive"),
    ],
)
def test_system_camera_cfg_rejects_invalid_values(system_camera, match):
    with pytest.raises(ValueError, match=match):
        validate_scene_cfg(SceneCfg(system_camera=system_camera))


def test_scene_objs_reject_non_object_fields():
    @configclass
    class InvalidSceneObjsCfg(SceneObjsCfg):
        invalid: int = 1

    with pytest.raises(TypeError, match="'invalid' must contain SceneObjCfg or None"):
        validate_scene_cfg(SceneCfg(objs=InvalidSceneObjsCfg()))


def test_scene_objects_share_base_contract():
    assert issubclass(FlatTerrainCfg, SceneObjCfg)
    assert issubclass(HFieldTerrainCfg, SceneObjCfg)
    assert LightCfg.__bases__ == (SceneObjCfg,)
    assert issubclass(FlatTerrainCfg, GeomCfg)
    assert issubclass(HFieldTerrainCfg, GeomCfg)
    assert RobotCfg.__bases__ == (BodyCfg,)
    assert BodyCfg.__bases__ == (SceneObjCfg,)
    assert MjcfFileCfg.__bases__ == (ModelFileCfg,)
    assert "size" not in {cfg_field.name for cfg_field in fields(FlatTerrainCfg)}
    assert "texture_file" not in {cfg_field.name for cfg_field in fields(FlatTerrainCfg)}
    assert "material" in {cfg_field.name for cfg_field in fields(FlatTerrainCfg)}
    assert "file" not in {cfg_field.name for cfg_field in fields(HFieldTerrainCfg)}
    assert "hfield" in {cfg_field.name for cfg_field in fields(HFieldTerrainCfg)}
    assert "name" not in {cfg_field.name for cfg_field in fields(SceneObjCfg)}
    assert "model" in {cfg_field.name for cfg_field in fields(RobotCfg)}
    assert "base_link_name" in {cfg_field.name for cfg_field in fields(RobotCfg)}
    assert [cfg_field.name for cfg_field in fields(MjcfFileCfg)] == ["file"]


@dataclass
class SceneBackedCartPoleCfg(DirectEnvCfg):
    scene: SceneCfg | None = field(default_factory=lambda: SceneCfg(file=_CARTPOLE_XML))


class SceneBackedCartPoleEnv(DirectEnv[SceneBackedCartPoleCfg]):
    def __init__(self, cfg: SceneBackedCartPoleCfg):
        super().__init__(cfg)
        self.model = self.sim.compile_model({})

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        return state

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        return state

    def reset(self, env_ids) -> dict:
        return {}


def test_direct_env_loads_model_from_scene_cfg():
    cfg = SceneBackedCartPoleCfg(
        sim=SimCfg(
            dt=0.005,
            solver_iterations=3,
            solver_tolerance=1e-4,
        )
    )
    env = SceneBackedCartPoleEnv(cfg)

    assert tuple(spec.target_name for spec in env.model.actuators) == ("slider",)
    assert env.cfg.scene is not None
    assert env.cfg.scene.file == _CARTPOLE_XML
    # Sim parameters flow through the same public compile API the backend uses.
    model = build_scene_model(cfg.scene, cfg.sim)
    assert model.options.timestep == pytest.approx(0.005)
    assert model.options.max_iterations == 3
    assert model.options.solver_tolerance == pytest.approx(1e-4)


def test_system_camera_follow_must_name_a_body_object():
    @configclass
    class FollowSceneObjsCfg(SceneObjsCfg):
        floor: FlatTerrainCfg = FlatTerrainCfg()
        cartpole: RobotCfg = RobotCfg(
            model=MjcfFileCfg(file=_CARTPOLE_XML),
            base_link_name="cart",
        )

    def scene_with_follow(target: str | None) -> SceneCfg:
        return SceneCfg(
            objs=FollowSceneObjsCfg(),
            system_camera=SystemCameraCfg(follow=target),
        )

    # A declared body object resolves; validation passes.
    validate_scene_cfg(scene_with_follow("cartpole"))

    with pytest.raises(ValueError, match="system_camera.follow must name a body object"):
        validate_scene_cfg(scene_with_follow("missing"))
    with pytest.raises(ValueError, match="system_camera.follow must name a body object"):
        # Declared but not a body: terrain objects cannot be followed.
        validate_scene_cfg(scene_with_follow("floor"))
