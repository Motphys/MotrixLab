# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.renderer import RenderConfig


def wrap_env(env, render: RenderConfig | None = None):
    """Wrap the environment for SKRL Torch."""
    from motrix_env_core.array.env import ArrayEnv
    from motrix_env_motrixsim.torch_env import TorchEnv

    if isinstance(env, TorchEnv):
        from motrix_rl_skrl.torch.wrap_torch import SkrlTorchWrapper

        return SkrlTorchWrapper(env, render=render)
    if isinstance(env, ArrayEnv):
        from motrix_rl_skrl.torch.wrap_np import SkrlNpWrapper

        return SkrlNpWrapper(env, render=render)
    raise TypeError(f"SKRL Torch does not support environment type '{type(env).__name__}'.")
