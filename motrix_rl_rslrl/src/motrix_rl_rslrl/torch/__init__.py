# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""PyTorch backend for RSLRL integration."""

import torch

from motrix_env_core.renderer import RenderConfig
from motrix_rl_rslrl.torch.wrap_np import RslrlNpEnvWrap


def wrap_env(env, device: torch.device, render: RenderConfig | None = None):
    """Wrap the environment for RSLRL."""
    from motrix_env_core.array.env import ArrayEnv
    from motrix_env_motrixsim.torch_env import TorchEnv

    if isinstance(env, TorchEnv):
        from motrix_rl_rslrl.torch.wrap_torch import RslrlTorchEnvWrap

        return RslrlTorchEnvWrap(env, device, render=render)
    if isinstance(env, ArrayEnv):
        from motrix_rl_rslrl.torch.wrap_np import RslrlNpEnvWrap

        return RslrlNpEnvWrap(env, device, render=render)
    raise TypeError(f"RSLRL does not support environment type '{type(env).__name__}'.")


__all__ = ["RslrlNpEnvWrap", "wrap_env"]
