# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment profile compilers for built-in MotrixLab environments."""

# g1_wbt is imported for registration, not as a second public API facade.
from motrix_deploy.profile import build_deployment_profile, registered_profile_compilers
from motrix_envs.deploy import g1_wbt as _g1_wbt
from motrix_envs.deploy.go2_walk import build_go2_walk_profile

__all__ = ["build_deployment_profile", "build_go2_walk_profile", "registered_profile_compilers"]
