# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Name-keyed simulator registry with entry-point discovery.

A simulator registers once via :func:`register_sim_backend` — either
programmatically (tests may stub a backend under a real backend's name) or
through the ``motrix_env.sim_backends`` entry-point group declared by an
installed backend distribution such as ``motrix-env-motrixsim``. Factories
stay lazy so importing this package — and discovery itself — never pulls in a
native simulator.
"""

from collections.abc import Callable
from importlib.metadata import entry_points

import torch

from motrix_env_core.sim.backend import SimBackend

SimBackendFactory = Callable[[torch.device], type[SimBackend]]

SIM_BACKEND_ENTRY_POINT_GROUP = "motrix_env.sim_backends"

_sim_backends: dict[str, SimBackendFactory] = {}
_default_sim_backend: str | None = None
_discovered = False
_discovering = False


def register_sim_backend(
    name: str,
    factory: SimBackendFactory,
    *,
    default: bool = False,
) -> None:
    """Register a :class:`SimBackend` factory under a simulator name.

    Args:
        name: Simulator name, such as ``"motrixsim"``.
        factory: Resolves the concrete backend class for the requested device.
        default: Designates this simulator as the one used when none is
            configured explicitly.
    """
    global _default_sim_backend
    if not name:
        raise ValueError("Sim backend name must not be empty")
    if name in _sim_backends:
        if _discovering:
            # Entry-point discovery never overrides an explicit registration;
            # tests stub real backend names this way.
            return
        raise ValueError(f"Sim backend {name!r} is already registered")
    if default:
        if _default_sim_backend is not None:
            raise ValueError(f"Default sim backend is already set to {_default_sim_backend!r}")
        _default_sim_backend = name
    _sim_backends[name] = factory


def default_sim_backend_name() -> str:
    """Return the simulator used when none is configured explicitly."""
    _discover()
    if _default_sim_backend is None:
        raise ValueError("No default sim backend registered; install a backend package such as motrix-env-motrixsim.")
    return _default_sim_backend


def create_sim_backend(name: str | None = None, device: torch.device | str = "cpu") -> type[SimBackend]:
    """Resolve the backend class before compiling a scene or allocating frontend buffers."""
    name = name if name is not None else default_sim_backend_name()
    factory = _lookup(name)(torch.device(device))
    if not (isinstance(factory, type) and issubclass(factory, SimBackend)):
        raise TypeError(f"Sim backend factory for {name!r} returned {type(factory).__name__}")
    return factory


def list_sim_backends() -> tuple[str, ...]:
    """Return registered simulator names in registration order."""
    _discover()
    return tuple(_sim_backends)


def _lookup(name: str) -> SimBackendFactory:
    _discover()
    try:
        return _sim_backends[name]
    except KeyError:
        _raise_unknown(name)


def _raise_unknown(name: str) -> None:
    available = sorted(_sim_backends)
    raise ValueError(f"Unknown sim backend {name!r}; available backends: {available}")


def _discover() -> None:
    """Load backend registrations advertised by installed distributions.

    Runs at most once per process, and only when a registry lookup needs it.
    """
    global _discovered, _discovering
    if _discovered:
        return
    _discovered = True
    _discovering = True
    try:
        for entry_point in entry_points(group=SIM_BACKEND_ENTRY_POINT_GROUP):
            register = entry_point.load()
            if not callable(register):
                raise TypeError(f"Sim backend entry point {entry_point.name!r} must load a callable")
            register()
    finally:
        _discovering = False


__all__ = [
    "SIM_BACKEND_ENTRY_POINT_GROUP",
    "SimBackendFactory",
    "create_sim_backend",
    "default_sim_backend_name",
    "list_sim_backends",
    "register_sim_backend",
]
