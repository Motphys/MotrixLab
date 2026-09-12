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
    ContactReportField,
    ContactSensorCfg,
    ContactSensorReduce,
    FlatTerrainCfg,
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
    SystemCameraCfg,
    TerrainGeneratorCfg,
    TextureCfg,
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
            "height_scale must be finite and non-negative",
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


@pytest.mark.parametrize(
    ("cfg", "expected_body", "expected_hfields", "solver_iterations", "solver_tolerance"),
    [
        (make_g129dof_walk_flat_cfg(), "pelvis", 0, 3, 1e-4),
        (make_g129dof_walk_rough_cfg(), "pelvis", 1, 3, 1e-4),
        (make_dex_evt_walk_flat_cfg(), "pelvis", 0, 6, 1e-4),
        (make_dex_evt_walk_rough_cfg(), "pelvis", 1, 6, 1e-4),
        (make_k1_walk_flat_cfg(), "Trunk", 0, 6, 1e-4),
        (make_k1_walk_rough_cfg(), "Trunk", 1, 6, 1e-4),
    ],
)
def test_humanoid_walk_scene_is_assembled_from_robot_and_floor(
    cfg,
    expected_body,
    expected_hfields,
    solver_iterations,
    solver_tolerance,
):
    cfg = OmegaConf.to_object(OmegaConf.structured(cfg))
    assert cfg.scene.file is None

    model = build_scene_model(cfg.scene, cfg.sim)

    assert expected_body in model.body_names
    assert "floor" in model.geom_names
    assert model.num_hfields == expected_hfields
    assert model.options.timestep == pytest.approx(0.005)
    assert model.options.max_iterations == solver_iterations
    assert model.options.solver_tolerance == pytest.approx(solver_tolerance)


def test_humanoid_walk_terrain_presets_use_procedural_hfield():
    configs = [
        make_g129dof_walk_rough_cfg(),
        make_dex_evt_walk_rough_cfg(),
        make_k1_walk_rough_cfg(),
    ]

    for cfg in configs:
        cfg = OmegaConf.to_object(OmegaConf.structured(cfg))
        assert isinstance(cfg.scene.assets.terrain, ProceduralHFieldAssetCfg)

        hfield = build_scene_world(cfg.scene).assets.hfields["terrain"]
        heights = hfield.source_type.value["hfield"]
        assert hfield.source_type.variant == "buffer"
        assert hfield.height_scale == pytest.approx(0.05)
        assert hfield.nrow == 320
        assert hfield.ncol == 320
        assert np.min(heights) >= 0.0
        assert np.max(heights) <= 1.0

        compiled_hfield = build_scene_model(cfg.scene).get_hfield(0)
        assert compiled_hfield.bound[5] == pytest.approx(0.05)


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
        self.model = self.sim.model_query_compiler.compile({})

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
