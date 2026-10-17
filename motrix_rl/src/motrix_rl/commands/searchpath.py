# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Give project configuration precedence over the packaged RL defaults."""

import argparse
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class ProjectConfigSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        if not any(
            entry.provider == "main" and entry.path == "pkg://motrix_rl.configs" for entry in search_path.get_path()
        ):
            return
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--config-dir")
        args, _ = parser.parse_known_args()
        if args.config_dir is not None:
            # Hydra validates --config-dir itself after search-path plugins run.
            # Its native entry is appended near the schema source; add an earlier
            # project entry so same-name groups override the packaged defaults.
            search_path.prepend(provider="motrix-project", path=Path(args.config_dir).resolve().as_uri())
