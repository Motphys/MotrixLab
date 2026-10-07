# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, get_type_hints

from motrix_env_core.base import ABEnv, EnvCfg
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.sim.registry import default_sim_backend_name

_BACKEND_BASE_CLASSES = {
    "np": ("ArrayEnv", "DirectEnv"),
    "torch": ("TorchEnv",),
}


def _infer_sim_backend(cls: type[ABEnv]) -> str:
    """Infer the environment data backend from its base class."""
    if not isinstance(cls, type) or not issubclass(cls, ABEnv):
        raise TypeError(f"Environment class must inherit ABEnv, got {cls!r}")
    ancestor_names = {base.__name__ for base in cls.__mro__}
    for backend, base_names in _BACKEND_BASE_CLASSES.items():
        if any(base_name in ancestor_names for base_name in base_names):
            return backend
    raise ValueError(f"Environment class '{cls.__name__}' must inherit DirectEnv or TorchEnv.")


EnvCfgFactory = Callable[[], EnvCfg]
RobotCfgFactory = Callable[[], RobotCfg]


@dataclass
class EnvMeta:
    env_cfg_cls: type[EnvCfg]
    env_cfg_factory: EnvCfgFactory
    env_cls_dict: dict[str, type[ABEnv]] = field(default_factory=dict)
    description: dict[str, str] = field(default_factory=dict)

    def available_sim_backend(self) -> str | None:
        """Return the first available simulation backend."""
        return next(iter(self.env_cls_dict), None)


@dataclass(frozen=True)
class EnvBuildSpec:
    """Resolved, spawn-safe environment construction metadata."""

    env_cls: type[ABEnv]
    env_cfg: EnvCfg
    sim: str | None = None

    def make(self, *, num_envs: int = 1, mode: str = "train", seed: int | None = None) -> ABEnv:
        env_cfg = copy.deepcopy(self.env_cfg)
        if mode == "play":
            env_cfg = env_cfg.for_play()
        env_cfg.validate()
        return _construct_env(self.env_cls, env_cfg, num_envs, self.sim, seed)


def _construct_env(
    env_cls: type[ABEnv], env_cfg: EnvCfg, num_envs: int, sim: str | None, seed: int | None = None
) -> ABEnv:
    """Instantiate ``env_cls``, resolving ``sim`` into a SimBackend for backend-injected envs.

    Direct and manager environments receive a backend: ``sim=None`` falls back
    to the registered default simulator. TorchEnv constructs its own simulator
    runtime and rejects an explicit ``sim``.
    """
    from motrix_env_core.direct.env import DirectEnv
    from motrix_env_core.numba.manager.env import ManagerEnv

    if issubclass(env_cls, ManagerEnv):
        return env_cls(
            env_cfg,
            num_envs=num_envs,
            backend=sim or default_sim_backend_name(),
            seed=seed,
        )
    if issubclass(env_cls, DirectEnv):
        return env_cls(env_cfg, num_envs=num_envs, backend=sim or default_sim_backend_name())
    if sim is not None:
        raise ValueError(
            f"sim={sim!r} is only used by backend-injected environments; "
            f"{env_cls.__name__} constructs its own simulator runtime."
        )
    return env_cls(env_cfg, num_envs=num_envs)


_envs: dict[str, EnvMeta] = {}


@dataclass
class RobotMeta:
    robot_cfg_cls: type[RobotCfg]
    robot_cfg_factory: RobotCfgFactory


ROBOT_ENTRY_POINT_GROUP = "motrix_env_core.robots"

_robots: dict[str, RobotMeta] = {}
_robots_discovered = False
_robots_discovering = False
_robot_entry_points: tuple[EntryPoint, ...] | None = None
_completed_robot_entry_points = 0


def _discover_robots() -> None:
    """Invoke installed robot registration callbacks on the first query.

    Completed callbacks are not repeated if a later entry point fails. Failed
    loads/callbacks propagate and are retried on the next query; any registration
    side effects of a failed callback remain. Reentrant queries see the registry
    as populated so far, without starting another discovery pass.
    """
    global _robots_discovered, _robots_discovering, _robot_entry_points, _completed_robot_entry_points
    if _robots_discovered or _robots_discovering:
        return
    _robots_discovering = True
    try:
        if _robot_entry_points is None:
            _robot_entry_points = tuple(
                sorted(entry_points(group=ROBOT_ENTRY_POINT_GROUP), key=lambda ep: (ep.name, ep.value))
            )
        for entry_point in _robot_entry_points[_completed_robot_entry_points:]:
            register = entry_point.load()
            if not callable(register):
                raise TypeError(f"Robot entry point {entry_point.name!r} must load a callable")
            register()
            _completed_robot_entry_points += 1
        _robots_discovered = True
    finally:
        _robots_discovering = False


def _get_env_meta(name: str) -> EnvMeta:
    try:
        return _envs[name]
    except KeyError:
        raise ValueError(f"Environment '{name}' is not registered.") from None


def _resolve_backend(name: str, meta: EnvMeta) -> tuple[str, type[ABEnv]]:
    """Return the registered environment class for ``name``."""
    resolved_backend = meta.available_sim_backend()
    if resolved_backend is None:
        raise ValueError(f"Environment '{name}' does not support any simulation backend.")
    return resolved_backend, meta.env_cls_dict[resolved_backend]


def contains(name: str) -> bool:
    """Check if an environment configuration is registered."""
    return name in _envs


def _description_from_docstring(env_cfg_provider: type[EnvCfg] | EnvCfgFactory) -> dict[str, str]:
    docstring = inspect.getdoc(env_cfg_provider)
    if not docstring:
        return {}

    lines = [line.strip() for line in docstring.splitlines() if line.strip()]
    descriptions = {"en": lines[0]}
    for line in lines[1:]:
        if line.startswith("zh_CN:"):
            description = line.removeprefix("zh_CN:").strip()
            if description:
                descriptions["zh_CN"] = description
            break
    return descriptions


def register_env_config(
    name: str,
    env_cfg_provider: type[EnvCfg] | EnvCfgFactory,
    *,
    cfg_type: type[EnvCfg] | None = None,
):
    """Register an environment config class or a typed zero-argument factory."""
    if name in _envs.keys():
        raise ValueError(f"Environment '{name}' is already registered.")

    if isinstance(env_cfg_provider, type):
        if not issubclass(env_cfg_provider, EnvCfg):
            raise TypeError(f"Environment config class must inherit EnvCfg, got {env_cfg_provider!r}")
        if cfg_type is not None and cfg_type is not env_cfg_provider:
            raise TypeError("cfg_type must not differ from the registered environment config class")
        cfg_type = env_cfg_provider
    else:
        if cfg_type is None:
            cfg_type = get_type_hints(env_cfg_provider).get("return")
        if cfg_type is None:
            raise TypeError("Environment config factory requires a return type annotation or explicit cfg_type")
        if not isinstance(cfg_type, type) or not issubclass(cfg_type, EnvCfg):
            raise TypeError(f"Environment config factory type must inherit EnvCfg, got {cfg_type!r}")

    _envs[name] = EnvMeta(
        env_cfg_cls=cfg_type,
        env_cfg_factory=env_cfg_provider,
        description=_description_from_docstring(env_cfg_provider),
    )


def envcfg(
    name: str,
    *,
    cfg_type: type[EnvCfg] | None = None,
) -> Callable[[Any], Any]:
    """
    Decorator to register an environment configuration class with a name.

    Usage:
        @envcfg("my-env")
        @configclass
        class MyEnvCfg(EnvCfg):
            ...

        @envcfg("my-preset")
        def make_my_preset() -> MyEnvCfg:
            return MyEnvCfg(...)
    """

    def decorator(provider: Any) -> Any:
        register_env_config(name, provider, cfg_type=cfg_type)
        return provider

    return decorator


def register_robot_config(
    name: str,
    robot_cfg_provider: type[RobotCfg] | RobotCfgFactory,
    *,
    cfg_type: type[RobotCfg] | None = None,
) -> None:
    """Register a robot config class or a typed zero-argument factory."""
    if name in _robots:
        raise ValueError(f"Robot '{name}' is already registered.")

    if isinstance(robot_cfg_provider, type):
        if not issubclass(robot_cfg_provider, RobotCfg):
            raise TypeError(f"Robot config class must inherit RobotCfg, got {robot_cfg_provider!r}")
        if cfg_type is not None and cfg_type is not robot_cfg_provider:
            raise TypeError("cfg_type must not differ from the registered robot config class")
        cfg_type = robot_cfg_provider
    else:
        if cfg_type is None:
            cfg_type = get_type_hints(robot_cfg_provider).get("return")
        if cfg_type is None:
            raise TypeError("Robot config factory requires a return type annotation or explicit cfg_type")
        if not isinstance(cfg_type, type) or not issubclass(cfg_type, RobotCfg):
            raise TypeError(f"Robot config factory type must inherit RobotCfg, got {cfg_type!r}")

    _robots[name] = RobotMeta(robot_cfg_cls=cfg_type, robot_cfg_factory=robot_cfg_provider)


def robotcfg(name: str, *, cfg_type: type[RobotCfg] | None = None) -> Callable[[Any], Any]:
    """Decorate a robot config class or factory for registration under ``name``."""

    def decorator(provider: Any) -> Any:
        register_robot_config(name, provider, cfg_type=cfg_type)
        return provider

    return decorator


def make_robot_config(name: str) -> RobotCfg:
    """Discover installed robot plugins, then create and validate a fresh config."""
    _discover_robots()
    if name not in _robots:
        raise ValueError(f"Robot '{name}' is not registered.")

    meta = _robots[name]
    robot_cfg = meta.robot_cfg_factory()
    if not isinstance(robot_cfg, meta.robot_cfg_cls):
        raise TypeError(
            f"Robot '{name}' config factory must return {meta.robot_cfg_cls.__name__}, got {type(robot_cfg).__name__}"
        )
    robot_cfg.validate("robot")
    return robot_cfg


def list_registered_robots() -> dict[str, dict[str, Any]]:
    """Discover installed robot plugins and list their registered configs."""
    _discover_robots()
    return {
        name: {
            "config_class": meta.robot_cfg_cls.__name__,
        }
        for name, meta in _robots.items()
    }


def make_env_config(
    name: str,
    env_cfg_override: dict[str, Any] | None = None,
    mode: str = "train",
) -> EnvCfg:
    """Create, override, and validate a fresh config for a registered environment."""
    meta = _get_env_meta(name)
    env_cfg = meta.env_cfg_factory()
    if not isinstance(env_cfg, meta.env_cfg_cls):
        raise TypeError(
            f"Environment '{name}' config factory must return {meta.env_cfg_cls.__name__}, got {type(env_cfg).__name__}"
        )

    if mode == "play":
        env_cfg = env_cfg.for_play()
    if env_cfg_override is not None:
        for key, value in env_cfg_override.items():
            if hasattr(env_cfg, key):
                setattr(env_cfg, key, value)
            else:
                raise ValueError(f"Config class '{env_cfg.__class__.__name__}' has no attribute '{key}'")

    env_cfg.validate()
    return env_cfg


def register_env(name: str, env_cls: type[ABEnv]):
    """Register an environment class with a name. Backend is inferred from the class hierarchy."""
    sim_backend = _infer_sim_backend(env_cls)

    if name not in _envs:
        raise ValueError(f"Environment '{name}' is not registered. Please register the config first.")

    if sim_backend in _envs[name].env_cls_dict:
        raise ValueError(f"Environment '{name}' with sim backend '{sim_backend}' is already registered.")

    _envs[name].env_cls_dict[sim_backend] = env_cls


def env(name: str) -> Callable[[type[ABEnv]], type[ABEnv]]:
    """
    Decorator to register an environment class with a name.
    The simulation backend is automatically inferred from the class hierarchy
    (ArrayEnv/DirectEnv -> "np").

    Usage:
        @registry.env("my-env")
        class MyEnv(DirectEnv):
            ...
    """

    def decorator(cls: type[ABEnv]) -> type[ABEnv]:
        register_env(name, cls)
        return cls

    return decorator


def resolve(
    name: str,
    env_cfg: EnvCfg | None = None,
    sim: str | None = None,
) -> EnvBuildSpec:
    """Resolve a registered environment into spawn-safe construction metadata."""

    meta = _get_env_meta(name)
    _, env_cls = _resolve_backend(name, meta)
    if env_cfg is None:
        env_cfg = make_env_config(name)
    else:
        env_cfg = copy.deepcopy(env_cfg)
        env_cfg.validate()

    return EnvBuildSpec(
        env_cls=env_cls,
        env_cfg=env_cfg,
        sim=sim,
    )


def make(
    name: str,
    env_cfg_override: dict[str, Any] | None = None,
    num_envs: int = 1,
    mode: str = "train",
    sim: str | None = None,
    seed: int | None = None,
) -> ABEnv:
    """
    Create an environment instance by name.

    Args:
        name: Environment name
        env_cfg_override: Dictionary of config overrides.
        num_envs: Number of environments to create
        mode: Rollout mode, "train" or "play". "play" selects the config's
            ``for_play()`` variant; the environment itself never sees the mode.
        sim: Simulator name resolved through the SimBackend registry
            (:func:`motrix_env_core.sim.registry.create_sim_backend`) and
            injected into manager-based environments.
        seed: Runtime seed applied to manager-based environment configs.
    Returns:
        Environment instance
    """
    meta = _get_env_meta(name)
    env_cfg = make_env_config(
        name,
        env_cfg_override=env_cfg_override,
        mode=mode,
    )
    _, env_cls = _resolve_backend(name, meta)
    return _construct_env(env_cls, env_cfg, num_envs, sim, seed)


def list_registered_envs() -> dict[str, dict[str, Any]]:
    """List registered environments with backends and localized docstring summaries."""
    result = {}
    for name, meta in _envs.items():
        result[name] = {
            "config_class": meta.env_cfg_cls.__name__,
            "available_backends": list(meta.env_cls_dict.keys()),
            "description": dict(meta.description),
        }
    return result
