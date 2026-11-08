# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.renderer import RenderConfig


def wrap_env(env, render: RenderConfig | None = None):
    """Wrap the environment for SKRL Torch."""
    from motrix_env_core.array.env import ArrayEnv

    if isinstance(env, ArrayEnv):
        from motrix_rl_skrl.torch.wrap import SkrlTorchEnvWrap

        return SkrlTorchEnvWrap(env, render=render)
    raise TypeError(f"SKRL Torch does not support environment type '{type(env).__name__}'.")
