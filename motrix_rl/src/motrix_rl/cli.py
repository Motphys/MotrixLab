# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Helpers for Hydra-based train/play/view command-line entry points."""

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

from motrix_rl.commands.searchpath import ProjectConfigSearchPath

logger = logging.getLogger(__name__)

CfgT = TypeVar("CfgT")


def run_cli(command: Callable[[], None]) -> None:
    """Use the launch directory's configs unless Hydra receives an explicit path.

    Pass the directory through Hydra's native --config-dir handling so package
    configs remain available and Hydra controls composition and diagnostics.
    """
    Plugins.instance().register(ProjectConfigSearchPath)
    args = sys.argv[1:]
    explicit_path = any(
        arg in ("--config-dir", "--config-path", "-cp") or arg.startswith(("--config-dir=", "--config-path=", "-cp="))
        for arg in args
    )
    config_dir = Path.cwd() / "configs"
    if explicit_path or not config_dir.is_dir():
        command()
        return
    original_argv = sys.argv
    sys.argv = [original_argv[0], "--config-dir", str(config_dir), *args]
    try:
        command()
    finally:
        sys.argv = original_argv


def to_typed_config(cfg: DictConfig, expected_type: type[CfgT]) -> CfgT:
    """Convert a Hydra ``DictConfig`` to its registered dataclass schema.

    Hydra always calls ``@hydra.main`` with a ``DictConfig``. This helper keeps
    that dynamic value at the CLI boundary and gives the rest of the application
    a real dataclass with static field types.
    """
    obj = OmegaConf.to_object(cfg)
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(obj).__name__}")
    return obj
