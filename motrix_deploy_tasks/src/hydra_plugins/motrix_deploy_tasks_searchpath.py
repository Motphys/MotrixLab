# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Expose installed deployment recipes without importing task implementations."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

from motrix_deploy_tasks.programs.g1_hot_swapping import install_program_schema


class MotrixDeployTasksSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Deployment task options share Hydra's task group name with training;
        # expose them only for the deployment application's configuration root.
        if any(
            entry.provider == "main" and entry.path == "pkg://motrix_deploy.config" for entry in search_path.get_path()
        ):
            install_program_schema()
            search_path.append(provider="motrix-deploy-tasks", path="pkg://motrix_deploy_tasks.config")
