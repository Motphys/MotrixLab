# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""PyTorch backend for RSLRL integration."""

import torch

from motrix_env_core.renderer import RenderConfig
from motrix_rl_rslrl.torch.wrap import RslrlEnvWrap


def wrap_env(env, device: torch.device, render: RenderConfig | None = None):
    """Wrap the environment for RSLRL."""
    from motrix_env_core.array.env import ArrayEnv

    if isinstance(env, ArrayEnv):
        from motrix_rl_rslrl.torch.wrap import RslrlEnvWrap

        return RslrlEnvWrap(env, device, render=render)
    raise TypeError(f"RSLRL does not support environment type '{type(env).__name__}'.")


__all__ = ["RslrlEnvWrap", "wrap_env"]
