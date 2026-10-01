# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import logging
import os

# FastSAC runs learner and collector processes concurrently. Let idle GNU
# OpenMP workers sleep instead of spinning and stealing CPU from the other
# process. These defaults must be set before importing Torch through motrix_rl;
# explicit user settings still take precedence.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

import motrix_envs  # noqa: E402, F401 registers built-in environments
from motrix_env_core.renderer import RenderConfig  # noqa: E402
from motrix_rl import runner  # noqa: E402
from motrix_rl.cli import to_typed_config  # noqa: E402
from motrix_rl.config import TrainConfig  # noqa: E402
from motrix_rl.plugins import load_plugins  # noqa: E402

# Plugin schemas must be available before Hydra composes the selected task.
load_plugins()

logger = logging.getLogger(__name__)


def run(cfg: TrainConfig) -> None:
    render = RenderConfig() if cfg.render else None

    train_result = runner.train(
        runner.TrainRequest(
            config=cfg,
            render=render,
        )
    )

    if cfg.play:
        try:
            train_result.play(
                render=RenderConfig(),
            )
        except FileNotFoundError as e:
            logger.error(f"Unable to play after training: {e}")
            return


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run(to_typed_config(cfg, TrainConfig))


if __name__ == "__main__":
    main()
