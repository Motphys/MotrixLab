# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point registration for the ``motrix_env.sim_backends`` group.

Importing this module must stay cheap: factories are lazy so registry
discovery never imports the native simulator.
"""

from importlib.util import find_spec

import torch

from motrix_env_core.sim.backend import SimBackend
from motrix_env_core.sim.registry import register_sim_backend


def _sim_backend(device: torch.device) -> type[SimBackend]:
    if device.type == "cpu":
        from motrix_env_motrixsim.runtime import MotrixSimBackend

        return MotrixSimBackend
    if device.type != "cuda":
        raise ValueError(f"MotrixSim does not support simulation on {device}")
    if find_spec("motrixsim.gpu") is None:
        raise ImportError("MotrixSim CUDA requires motrix-env-motrixsim[gpu]")
    if not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count()):
        raise ValueError(f"Requested MotrixSim CUDA device {device} is unavailable")
    from motrix_env_motrixsim.gpu_runtime import MotrixCudaBackend

    return MotrixCudaBackend


def register() -> None:
    """Register the MotrixSim backend as the default simulator."""
    register_sim_backend("motrixsim", _sim_backend, default=True)


__all__ = ["register"]
