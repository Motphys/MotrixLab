# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point registration for the ``motrix_env.sim_backends`` group.

Importing this module must stay cheap: factories are lazy so registry
discovery never imports the native simulator. This backend provides no live
simulation, so it never claims the default slot.
"""

import torch

from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.registry import register_sim_backend


def _install_hint(exc: ModuleNotFoundError) -> ModuleNotFoundError:
    if exc.name == "mujoco":
        raise ModuleNotFoundError(
            "The MuJoCo sim backend requires the optional dependency; install motrix-env-mujoco"
        ) from exc
    return exc


def _sim_backend(device: torch.device) -> type[SimBackend]:
    if device.type != "cpu":
        raise ValueError(f"MuJoCo does not support simulation on {device}")
    try:
        from motrix_env_mujoco.backend import MuJoCoSimBackend
    except ModuleNotFoundError as exc:
        raise _install_hint(exc) from exc
    return MuJoCoSimBackend


def register() -> None:
    """Register the compile-only MuJoCo backend."""
    register_sim_backend("mujoco", _sim_backend)


__all__ = ["register"]
