# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import motrixsim as mtx
import numpy as np

from motrix_env_core.config.scene._utils import resolve_path
from motrix_env_core.config.scene.asset import (
    HFieldAssetCfg,
    MaterialCfg,
    ProceduralHFieldAssetCfg,
    SkyboxCfg,
    TextureCfg,
)
from motrix_env_core.config.scene.base import (
    BodyCfg,
    ModelFileCfg,
    SceneAssetCfg,
    SceneCfg,
    SceneObjCfg,
    SceneSensorCfg,
    SceneVisualCfg,
)
from motrix_env_core.config.scene.compiler import SceneCompiler
from motrix_env_core.config.scene.geometry import FlatTerrainCfg, GeomCfg, HFieldTerrainCfg
from motrix_env_core.config.scene.light import LightCfg
from motrix_env_core.config.scene.mjcf import MjcfFileCfg
from motrix_env_core.config.scene.sensor import (
    ContactReportField,
    ContactSensorCfg,
    ContactSensorReduce,
    FrameObjectKind,
    FrameRefKind,
    FrameSensorCfg,
    FrameSensorType,
)
from motrix_env_core.config.scene.urdf import UrdfFileCfg
from motrix_env_core.config.scene.validation import validate_scene_cfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_motrixsim.urdf import apply_urdf_cfgs

_CONTACT_REDUCE = {
    ContactSensorReduce.none: mtx.msd.ContactSensorReduce.None_,
    ContactSensorReduce.mindist: mtx.msd.ContactSensorReduce.MinDist,
    ContactSensorReduce.maxforce: mtx.msd.ContactSensorReduce.MaxForce,
    ContactSensorReduce.netforce: mtx.msd.ContactSensorReduce.NetForce,
}

_OBJECT_TYPE_FACTORY = {
    FrameObjectKind.site: mtx.msd.ObjectType.site,
    FrameObjectKind.geom: mtx.msd.ObjectType.geom,
    FrameObjectKind.link: mtx.msd.ObjectType.link,
    FrameObjectKind.link_inertia: mtx.msd.ObjectType.link_inertia,
}

_FRAME_SENSOR_TYPE = {
    FrameSensorType.framepos: mtx.msd.FrameSensorType.FramePos,
    FrameSensorType.framequat: mtx.msd.FrameSensorType.FrameQuat,
    FrameSensorType.framelinvel: mtx.msd.FrameSensorType.FrameLinVel,
    FrameSensorType.frameangvel: mtx.msd.FrameSensorType.FrameAngVel,
    FrameSensorType.framelinacc: mtx.msd.FrameSensorType.FrameLinAcc,
    FrameSensorType.xaxis: mtx.msd.FrameSensorType.XAxis,
    FrameSensorType.yaxis: mtx.msd.FrameSensorType.YAxis,
    FrameSensorType.zaxis: mtx.msd.FrameSensorType.ZAxis,
}


class MotrixSimSceneCompiler(SceneCompiler[mtx.SceneModel]):
    """Compile declarative scene configuration into a MotrixSim model."""

    def build_world(self, scene: SceneCfg) -> mtx.msd.World:
        validate_scene_cfg(scene)
        world = mtx.msd.World() if scene.file is None else mtx.msd.from_file(str(resolve_path(scene.file)))
        for name, asset in scene.iter_assets():
            self._add_asset(world, name, asset)
        self._apply_visual(world, scene.visual)
        for name, obj in scene.iter_objs():
            self._append_obj(world, name, obj)
        for name, sensor in scene.iter_sensors():
            self._append_sensor(world, name, sensor)
        return world

    def configure_world(self, world: mtx.msd.World, sim: SimCfg) -> None:
        world.simulate_option.timestep = sim.dt
        if sim.solver_iterations is not None:
            world.simulate_option.constraint_solver_iterations = sim.solver_iterations
        if sim.solver_tolerance is not None:
            world.simulate_option.constraint_solver_tolerance = sim.solver_tolerance
        if sim.gravity is not None:
            world.simulate_option.gravity = list(sim.gravity)

    def compile(self, scene: SceneCfg, sim: SimCfg) -> mtx.SceneModel:
        sim.validate()
        world = self.build_world(scene)
        self.configure_world(world, sim)
        return world.build()

    def _load_model_world(self, model: ModelFileCfg) -> mtx.msd.World:
        if isinstance(model, MjcfFileCfg):
            return mtx.msd.from_file(str(resolve_path(model.file)))
        if isinstance(model, UrdfFileCfg):
            path = resolve_path(model.file)
            world = mtx.msd.from_str(path.read_text(), format="urdf", file_path=str(path))
            # urdf2msd (since 0.10.1.dev120408) synthesizes a default
            # `<joint>_motor` actuator per joint. The declarative
            # ``UrdfFileCfg.actuators`` are the sole actuation authority, so
            # drop the synthesized defaults before applying the config.
            # Track merging declarative configs onto the synthesized motors
            # in https://gitlab.mp/motphys/morphos-lab/-/issues/226.
            world.actuators.clear()
            apply_urdf_cfgs(world, model)
            return world
        raise TypeError(f"MotrixSim does not support model file config {type(model).__name__}")

    def _add_asset(self, world: mtx.msd.World, name: str, cfg: SceneAssetCfg) -> None:
        if isinstance(cfg, TextureCfg):
            texture = mtx.msd.Texture()
            texture.name = name
            texture.source = mtx.msd.TextureSource.file(str(resolve_path(cfg.file)))
            texture.type_ = mtx.msd.TextureType.D2
            texture.color_space = mtx.msd.ColorSpace.Srgb if cfg.color_space == "srgb" else mtx.msd.ColorSpace.Linear
            texture.gen_mipmaps = cfg.gen_mipmaps
            world.assets.textures[name] = texture
            return
        if isinstance(cfg, SkyboxCfg):
            source = mtx.msd.BuildinAttr()
            source.rgb1 = list(cfg.color_top)
            source.rgb2 = list(cfg.color_bottom)
            source.width = float(cfg.width)
            source.height = float(cfg.height)
            source.markrgb = [1.0, 1.0, 1.0]

            texture = mtx.msd.Texture()
            texture.name = name
            texture.source = mtx.msd.TextureSource.gradient(source)
            texture.type_ = mtx.msd.TextureType.Skybox
            texture.color_space = mtx.msd.ColorSpace.Srgb if cfg.color_space == "srgb" else mtx.msd.ColorSpace.Linear
            texture.gen_mipmaps = cfg.gen_mipmaps
            world.assets.textures[name] = texture
            world.assets.skybox = name
            return
        if isinstance(cfg, MaterialCfg):
            material = mtx.msd.Material()
            material.name = name
            material.color = list(cfg.color)
            material.texture_name = cfg.texture
            material.tex_repeat = list(cfg.texture_repeat)
            material.tex_uniform = cfg.texture_uniform
            material.metallic = cfg.metallic
            material.roughness = cfg.roughness
            world.assets.materials[name] = material
            return
        if isinstance(cfg, HFieldAssetCfg):
            hfield = mtx.msd.HFieldSource()
            hfield.source_type = mtx.msd.HFieldSourceType.path(resolve_path(cfg.file))
            hfield.size = [extent / 2.0 for extent in cfg.size]
            hfield.height_scale = cfg.height_scale
            world.assets.hfields[name] = hfield
            return
        if isinstance(cfg, ProceduralHFieldAssetCfg):
            expected_shape = tuple(cfg.shape)
            heights = np.asarray(cfg.generator.generate(cfg.size, expected_shape), dtype=np.float32)
            if heights.shape != expected_shape:
                raise ValueError(
                    f"Terrain generator returned shape {heights.shape!r}, expected {expected_shape!r} "
                    f"for asset {name!r}"
                )
            if not np.all(np.isfinite(heights)):
                raise ValueError(f"Terrain generator returned non-finite heights for asset {name!r}")
            heights = np.ascontiguousarray(heights)

            hfield = mtx.msd.HFieldSource()
            hfield.source_type = mtx.msd.HFieldSourceType.buffer(heights.reshape(-1), name)
            hfield.nrow, hfield.ncol = heights.shape
            hfield.size = [extent / 2.0 for extent in cfg.size]
            hfield.height_scale = cfg.generator.height_scale
            world.assets.hfields[name] = hfield
            return
        raise TypeError(f"MotrixSim does not support scene asset config {type(cfg).__name__}")

    def _apply_visual(self, world: mtx.msd.World, cfg: SceneVisualCfg) -> None:
        if cfg.ambient_light_color is not None:
            world.visual.ambient_light.color = list(cfg.ambient_light_color)
        if cfg.ambient_light_brightness is not None:
            world.visual.ambient_light.brightness = cfg.ambient_light_brightness
        if cfg.head_light_color is not None:
            world.visual.head_light.color = list(cfg.head_light_color)
        if cfg.head_light_luminous_power is not None:
            world.visual.head_light.luminous_power = cfg.head_light_luminous_power
        if cfg.haze is not None:
            world.visual.haze = list(cfg.haze)
        if cfg.tone_mapping is not None:
            world.visual.tonemapping.method = (
                mtx.msd.ToneMappingMethod.None_ if cfg.tone_mapping == "none" else mtx.msd.ToneMappingMethod.Aces
            )

    def _configure_geom(self, geom: mtx.msd.Geometry, cfg: GeomCfg) -> None:
        geom.priority = cfg.priority
        geom.physics_material.friction = list(cfg.friction)
        geom.physics_material.condim = cfg.condim
        geom.collision_mask.collide_group = cfg.contype
        geom.collision_mask.collide_with = cfg.conaffinity
        if cfg.material is not None:
            geom.visual.material = cfg.material

    def _append_obj(self, world: mtx.msd.World, name: str, cfg: SceneObjCfg) -> None:
        if isinstance(cfg, FlatTerrainCfg):
            geom = mtx.msd.Geometry()
            geom.name = name
            geom.shape = mtx.msd.ShapeType.InfinitePlane
            geom.position = [0.0, 0.0, cfg.height]
            self._configure_geom(geom, cfg)
            world.hierarchy.geoms.append(geom)
            return
        if isinstance(cfg, HFieldTerrainCfg):
            geom = mtx.msd.Geometry()
            geom.name = name
            geom.shape = mtx.msd.ShapeType.HField
            geom.hfield = cfg.hfield
            self._configure_geom(geom, cfg)
            world.hierarchy.geoms.append(geom)
            return
        if isinstance(cfg, LightCfg):
            desc = mtx.msd.DirectionalLightDesc()
            desc.illuminance = cfg.illuminance

            light = mtx.msd.Light()
            light.name = name
            light.type_ = mtx.msd.LightType.directional(desc)
            light.position = list(cfg.position)
            light.direction = list(cfg.direction)
            light.color = list(cfg.color)
            light.cast_shadows = cfg.cast_shadows
            world.hierarchy.lights.append(light)
            return
        if isinstance(cfg, BodyCfg):
            world.attach(
                self._load_model_world(cfg.model),
                other_link_name=cfg.base_link_name,
                other_translation=cfg.translation,
                other_rotation=cfg.rotation,
                other_prefix=cfg.prefix,
                other_suffix=cfg.suffix,
            )
            return
        raise TypeError(f"MotrixSim does not support scene object config {type(cfg).__name__}")

    def _object_type(self, kind: FrameObjectKind, name: str) -> mtx.msd.ObjectType:
        return _OBJECT_TYPE_FACTORY[kind](name)

    def _frame_ref(self, cfg: FrameSensorCfg) -> mtx.msd.FrameSensorRef:
        if cfg.ref_kind is FrameRefKind.local:
            return mtx.msd.FrameSensorRef.local()
        if cfg.ref_kind is FrameRefKind.world:
            return mtx.msd.FrameSensorRef.world()
        assert cfg.ref_object_type is not None and cfg.ref_object_name is not None
        return mtx.msd.FrameSensorRef.object(self._object_type(cfg.ref_object_type, cfg.ref_object_name))

    def _append_sensor(self, world: mtx.msd.World, name: str, cfg: SceneSensorCfg) -> None:
        if isinstance(cfg, ContactSensorCfg):
            sensor = mtx.msd.ContactSensor()
            sensor.name = name
            sensor.match_ = mtx.msd.ContactMatch.geom_pair(cfg.geom1, cfg.geom2)
            sensor.max_num = cfg.num
            sensor.reduce = _CONTACT_REDUCE[cfg.reduce]

            report = mtx.msd.ContactSensorReport()
            for field in ContactReportField:
                setattr(report, field.value, field in cfg.data)
            sensor.report = report
            world.sensors.contact.append(sensor)
            return
        if isinstance(cfg, FrameSensorCfg):
            sensor = mtx.msd.FrameSensor()
            sensor.name = name
            sensor.object_type = self._object_type(cfg.object_type, cfg.object_name)
            sensor.sensor_type = _FRAME_SENSOR_TYPE[cfg.sensor_type]
            sensor.ref_frame = self._frame_ref(cfg)
            world.sensors.frame.append(sensor)
            return
        raise TypeError(f"MotrixSim does not support scene sensor config {type(cfg).__name__}")


def build_scene_world(scene: SceneCfg) -> mtx.msd.World:
    """Build a MotrixSim MSD world without compiling it into a model."""
    return MotrixSimSceneCompiler().build_world(scene)


def build_scene_model(scene: SceneCfg, sim: SimCfg | None = None) -> mtx.SceneModel:
    """Compile a scene with the MotrixSim backend."""
    return MotrixSimSceneCompiler().compile(scene, SimCfg() if sim is None else sim)
