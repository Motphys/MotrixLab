# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager metadata binding for the Torch executor."""

from motrix_env_core.manager.env import ManagerEnv
from motrix_env_core.manager.resolver import ManagerProgramResolver, ResolvedManagerProgram


def build_manager_binding(env: ManagerEnv) -> ResolvedManagerProgram:
    """Resolve an already constructed manager environment before ``init_state``.

    The result supplies the read plan, layouts, prepared inputs, and Python
    term entries to the Torch runtime; this helper neither initializes the
    environment nor compiles any numerical kernels.
    """
    return ManagerProgramResolver(env).build()


__all__ = ["build_manager_binding"]
