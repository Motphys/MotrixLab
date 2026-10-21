# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Install the Unitree hardware recipe schema for the deployment application."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from motrix_deploy_unitree.config.schema import install_hardware_runtime_schema


class MotrixDeployUnitreeSchemaPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Hydra imports search-path plugins before composing; register the schema
        # only for the deployment application's configuration root.
        if any(
            entry.provider == "main" and entry.path == "pkg://motrix_deploy.config" for entry in search_path.get_path()
        ):
            install_hardware_runtime_schema()
