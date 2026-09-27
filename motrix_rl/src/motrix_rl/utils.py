# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass
class DeviceSupports:
    torch: bool = False
    torch_gpu: bool = False
    jax: bool = False
    jax_gpu: bool = False


def _check_gpu_available_for_torch():
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros((1,)).cuda().numpy(force=True)
        return True
    except Exception:
        return False


def get_device_supports() -> DeviceSupports:
    supports = DeviceSupports()
    torch_prior_device = None
    try:
        import torch  # noqa: F401

        supports.torch = True
        supports.torch_gpu = _check_gpu_available_for_torch()
        if supports.torch_gpu:
            torch_prior_device = torch.cuda.current_device()
    except ImportError:
        pass

    try:
        import jax  # noqa: F401

        supports.jax = True
        from jax.lib import xla_bridge

        platform = xla_bridge.get_backend().platform
        if platform == "gpu":
            supports.jax_gpu = True
    except ImportError:
        pass

    # XLA's GPU probe enumerates every visible device and can leave the CUDA
    # runtime's current device on the LAST probed GPU; torch lazily inherits
    # that value, silently poisoning every later torch.cuda.current_device()
    # consumer in this process. Restore the pre-probe device.
    if torch_prior_device is not None:
        import torch

        torch.cuda.set_device(torch_prior_device)

    return supports


def class_to_dict(obj) -> dict | list | Any:
    """Recursively convert a dataclass to a dictionary.

    Args:
        obj: The object to convert (dataclass, list, dict, or primitive)

    Returns:
        Dictionary representation with nested dataclasses recursively converted
    """
    if dataclasses.is_dataclass(obj):
        return {k: class_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    elif isinstance(obj, list):
        return [class_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: class_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def env_infos(state) -> dict:
    """Compose Gym-style step infos for the RL-library boundary.

    The env state keeps the per-term reward breakdown (``reward_terms``) and
    metrics as first-class fields; RL libraries receive the breakdown under
    ``infos["Reward"]`` and reduced batch-level scalar metrics under
    ``infos["metrics"]``.
    """
    return {"Reward": state.reward_terms, "metrics": state.process_metrics()}
