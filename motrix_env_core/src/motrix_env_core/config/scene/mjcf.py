# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import ModelFileCfg
from motrix_env_core.config.scene.site import SiteCfg


@configclass(kw_only=True)
class MjcfFileCfg(ModelFileCfg):
    """An MJCF model loaded from a file, augmented with simulation-only sites."""

    sites: list[SiteCfg] = []

    def validate(self) -> None:
        super().validate()
        for site in self.sites:
            site.validate()
