# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

try:
    import mujoco as mj
except ModuleNotFoundError as exc:
    if exc.name == "mujoco":
        raise ModuleNotFoundError(
            "The MuJoCo scene compiler requires the optional dependency; install motrix-env-mujoco"
        ) from exc
    raise
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
from motrix_env_mujoco.urdf import apply_urdf_cfgs

_CONTACT_DATA_BITS = {
    ContactReportField.found: 1,
    ContactReportField.force: 2,
    ContactReportField.torque: 4,
    ContactReportField.dist: 8,
    ContactReportField.pos: 16,
    ContactReportField.normal: 32,
    ContactReportField.tangent: 64,
}

_CONTACT_REDUCE = {
    ContactSensorReduce.none: 0,
    ContactSensorReduce.mindist: 1,
    ContactSensorReduce.maxforce: 2,
    ContactSensorReduce.netforce: 3,
}

_OBJECT_TYPE = {
    FrameObjectKind.site: mj.mjtObj.mjOBJ_SITE,
    FrameObjectKind.geom: mj.mjtObj.mjOBJ_GEOM,
    FrameObjectKind.link: mj.mjtObj.mjOBJ_BODY,
    FrameObjectKind.link_inertia: mj.mjtObj.mjOBJ_XBODY,
}

_FRAME_SENSOR_TYPE = {
    FrameSensorType.framepos: mj.mjtSensor.mjSENS_FRAMEPOS,
    FrameSensorType.framequat: mj.mjtSensor.mjSENS_FRAMEQUAT,
    FrameSensorType.framelinvel: mj.mjtSensor.mjSENS_FRAMELINVEL,
    FrameSensorType.frameangvel: mj.mjtSensor.mjSENS_FRAMEANGVEL,
    FrameSensorType.framelinacc: mj.mjtSensor.mjSENS_FRAMELINACC,
    FrameSensorType.xaxis: mj.mjtSensor.mjSENS_FRAMEXAXIS,
    FrameSensorType.yaxis: mj.mjtSensor.mjSENS_FRAMEYAXIS,
    FrameSensorType.zaxis: mj.mjtSensor.mjSENS_FRAMEZAXIS,
}

_MIN_HFIELD_SIZE = float(np.finfo(np.float32).eps)


def _mujoco_quat(rotation: tuple[float, float, float, float] | None) -> list[float] | None:
    if rotation is None:
        return None
    x, y, z, w = rotation
    return [w, x, y, z]


class MuJoCoSceneCompiler(SceneCompiler[mj.MjModel]):
    """Compile declarative scene configuration into a MuJoCo model."""

    def build_spec(self, scene: SceneCfg) -> mj.MjSpec:
        validate_scene_cfg(scene)
        spec = mj.MjSpec() if scene.file is None else mj.MjSpec.from_file(str(resolve_path(scene.file)))
        for name, asset in scene.iter_assets():
            self._add_asset(spec, name, asset)
        self._apply_visual(spec, scene.visual)
        for name, obj in scene.iter_objs():
            self._append_obj(spec, name, obj)
        for name, sensor in scene.iter_sensors():
            self._append_sensor(spec, name, sensor)
        return spec

    def configure_spec(self, spec: mj.MjSpec, sim: SimCfg) -> None:
        spec.option.timestep = sim.dt
        if sim.solver_iterations is not None:
            spec.option.iterations = sim.solver_iterations
        if sim.solver_tolerance is not None:
            spec.option.tolerance = sim.solver_tolerance
        if sim.gravity is not None:
            spec.option.gravity = sim.gravity

    def create_spec(self, scene: SceneCfg, sim: SimCfg) -> mj.MjSpec:
        """Create a configured MuJoCo spec that callers may modify before compilation."""
        sim.validate()
        spec = self.build_spec(scene)
        self.configure_spec(spec, sim)
        return spec

    def compile(self, scene: SceneCfg, sim: SimCfg) -> mj.MjModel:
        return self.create_spec(scene, sim).compile()

    def _load_model_spec(self, model: ModelFileCfg) -> mj.MjSpec:
        if isinstance(model, MjcfFileCfg):
            return mj.MjSpec.from_file(str(resolve_path(model.file)))
        if isinstance(model, UrdfFileCfg):
            spec = mj.MjSpec.from_file(str(resolve_path(model.file)))
            apply_urdf_cfgs(spec, model)
            return spec
        raise TypeError(f"MuJoCo does not support model file config {type(model).__name__}")

    def _add_asset(self, spec: mj.MjSpec, name: str, cfg: SceneAssetCfg) -> None:
        if isinstance(cfg, TextureCfg):
            spec.add_texture(
                name=name,
                type=mj.mjtTexture.mjTEXTURE_2D,
                colorspace=(
                    mj.mjtColorSpace.mjCOLORSPACE_SRGB
                    if cfg.color_space == "srgb"
                    else mj.mjtColorSpace.mjCOLORSPACE_LINEAR
                ),
                file=str(resolve_path(cfg.file)),
            )
            return
        if isinstance(cfg, SkyboxCfg):
            spec.add_texture(
                name=name,
                type=mj.mjtTexture.mjTEXTURE_SKYBOX,
                colorspace=(
                    mj.mjtColorSpace.mjCOLORSPACE_SRGB
                    if cfg.color_space == "srgb"
                    else mj.mjtColorSpace.mjCOLORSPACE_LINEAR
                ),
                builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
                rgb1=cfg.color_top,
                rgb2=cfg.color_bottom,
                width=cfg.width,
                height=cfg.height,
            )
            return
        if isinstance(cfg, MaterialCfg):
            spec.add_material(
                name=name,
                textures=[cfg.texture] if cfg.texture is not None else None,
                rgba=cfg.color,
                texrepeat=cfg.texture_repeat,
                texuniform=cfg.texture_uniform,
                metallic=cfg.metallic,
                roughness=cfg.roughness,
            )
            return
        if isinstance(cfg, HFieldAssetCfg):
            spec.add_hfield(
                name=name,
                file=str(resolve_path(cfg.file)),
                size=[
                    cfg.size[0] / 2.0,
                    cfg.size[1] / 2.0,
                    max(cfg.height_scale, _MIN_HFIELD_SIZE),
                    _MIN_HFIELD_SIZE,
                ],
            )
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
            spec.add_hfield(
                name=name,
                size=[
                    cfg.size[0] / 2.0,
                    cfg.size[1] / 2.0,
                    max(cfg.generator.height_scale, _MIN_HFIELD_SIZE),
                    _MIN_HFIELD_SIZE,
                ],
                nrow=heights.shape[0],
                ncol=heights.shape[1],
                userdata=np.ascontiguousarray(heights).reshape(-1),
            )
            return
        raise TypeError(f"MuJoCo does not support scene asset config {type(cfg).__name__}")

    def _apply_visual(self, spec: mj.MjSpec, cfg: SceneVisualCfg) -> None:
        # MuJoCo exposes headlight colors but no physical brightness/power or tone-mapping controls.
        if cfg.ambient_light_color is not None:
            spec.visual.headlight.ambient = cfg.ambient_light_color
        if cfg.head_light_color is not None:
            spec.visual.headlight.diffuse = cfg.head_light_color
        if cfg.haze is not None:
            spec.visual.rgba.haze = cfg.haze

    def _geom_kwargs(self, cfg: GeomCfg) -> dict[str, object]:
        return {
            "material": cfg.material,
            "friction": cfg.friction,
            "condim": cfg.condim,
            "contype": cfg.contype,
            "conaffinity": cfg.conaffinity,
            "priority": cfg.priority,
        }

    def _append_obj(self, spec: mj.MjSpec, name: str, cfg: SceneObjCfg) -> None:
        if isinstance(cfg, FlatTerrainCfg):
            spec.worldbody.add_geom(
                name=name,
                type=mj.mjtGeom.mjGEOM_PLANE,
                pos=[0.0, 0.0, cfg.height],
                size=[0.0, 0.0, 0.1],
                **self._geom_kwargs(cfg),
            )
            return
        if isinstance(cfg, HFieldTerrainCfg):
            spec.worldbody.add_geom(
                name=name,
                type=mj.mjtGeom.mjGEOM_HFIELD,
                hfieldname=cfg.hfield,
                **self._geom_kwargs(cfg),
            )
            return
        if isinstance(cfg, LightCfg):
            spec.worldbody.add_light(
                name=name,
                type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
                pos=cfg.position,
                dir=cfg.direction,
                diffuse=cfg.color,
                intensity=cfg.illuminance,
                castshadow=cfg.cast_shadows,
            )
            return
        if isinstance(cfg, BodyCfg):
            robot_spec = self._load_model_spec(cfg.model)
            base_link = robot_spec.body(cfg.base_link_name)
            if base_link is None:
                raise ValueError(f"Attached body base link {cfg.base_link_name!r} does not exist in model")
            if base_link.parent is not robot_spec.worldbody:
                raise ValueError(f"MuJoCo requires BodyCfg.base_link_name {cfg.base_link_name!r} to be a root body")
            first_root = robot_spec.worldbody.first_body()
            if first_root is None or first_root.name != cfg.base_link_name:
                raise ValueError(
                    f"MuJoCo requires BodyCfg.base_link_name {cfg.base_link_name!r} to be the first root body"
                )
            placement = spec.worldbody.add_frame(
                pos=cfg.translation,
                quat=_mujoco_quat(cfg.rotation),
            )
            spec.attach(robot_spec, prefix=cfg.prefix or "", suffix=cfg.suffix or "", frame=placement)
            return
        raise TypeError(f"MuJoCo does not support scene object config {type(cfg).__name__}")

    def _append_sensor(self, spec: mj.MjSpec, name: str, cfg: SceneSensorCfg) -> None:
        if isinstance(cfg, ContactSensorCfg):
            data_mask = sum(_CONTACT_DATA_BITS[field] for field in cfg.data)
            spec.add_sensor(
                name=name,
                type=mj.mjtSensor.mjSENS_CONTACT,
                objtype=mj.mjtObj.mjOBJ_GEOM,
                objname=cfg.geom1,
                reftype=mj.mjtObj.mjOBJ_GEOM,
                refname=cfg.geom2,
                intprm=[data_mask, cfg.num, _CONTACT_REDUCE[cfg.reduce]],
            )
            return
        if isinstance(cfg, FrameSensorCfg):
            if cfg.ref_kind is FrameRefKind.local:
                if cfg.sensor_type is FrameSensorType.framelinvel and cfg.object_type is FrameObjectKind.site:
                    spec.add_sensor(
                        name=name,
                        type=mj.mjtSensor.mjSENS_VELOCIMETER,
                        objtype=mj.mjtObj.mjOBJ_SITE,
                        objname=cfg.object_name,
                    )
                    return
                raise ValueError(
                    f"MuJoCo does not support local-frame semantics for {cfg.sensor_type.value} "
                    f"sensor {name!r} on {cfg.object_type.value} {cfg.object_name!r}"
                )
            ref_type = None
            ref_name = None
            if cfg.ref_kind is FrameRefKind.object:
                assert cfg.ref_object_type is not None and cfg.ref_object_name is not None
                ref_type = _OBJECT_TYPE[cfg.ref_object_type]
                ref_name = cfg.ref_object_name
            spec.add_sensor(
                name=name,
                type=_FRAME_SENSOR_TYPE[cfg.sensor_type],
                objtype=_OBJECT_TYPE[cfg.object_type],
                objname=cfg.object_name,
                reftype=ref_type,
                refname=ref_name,
            )
            return
        raise TypeError(f"MuJoCo does not support scene sensor config {type(cfg).__name__}")


def build_mujoco_spec(scene: SceneCfg) -> mj.MjSpec:
    """Build a MuJoCo specification without compiling it into a model."""
    return MuJoCoSceneCompiler().build_spec(scene)


def build_mujoco_model(scene: SceneCfg, sim: SimCfg | None = None) -> mj.MjModel:
    """Compile a scene with the MuJoCo backend."""
    return MuJoCoSceneCompiler().compile(scene, SimCfg() if sim is None else sim)
