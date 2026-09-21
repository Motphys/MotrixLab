# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import hydra
from omegaconf import DictConfig

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core.renderer import RenderConfig
from motrix_rl import checkpoints, runner, runs
from motrix_rl.cli import to_typed_config
from motrix_rl.config import PlayConfig
from motrix_rl.runs import find_metadata_for_policy, latest_metadata_run

logger = logging.getLogger(__name__)

DEFAULT_ENV_NAME = "cartpole"
PLAY_RECORDING_FPS = 60


@dataclass(frozen=True)
class PlayTarget:
    run: runs.RunContext
    policy_path: Path


def _latest_play_target(env_name: str, rllib: str | None = None) -> PlayTarget:
    metadata_run = latest_metadata_run(env_name, rllib=rllib)
    if metadata_run is not None:
        run_dir, metadata = metadata_run
        logger.info(f"Auto-discovered RL run: {run_dir}")
        return PlayTarget(
            run=runs.open_run_context(run_dir, metadata),
            policy_path=checkpoints.best_policy(metadata, run_dir),
        )

    if rllib is not None:
        raise FileNotFoundError(
            f"No metadata-backed training runs found for framework '{rllib}' and environment '{env_name}'"
        )
    raise FileNotFoundError(f"No metadata-backed training runs found for environment '{env_name}' in runs/{env_name}")


def _policy_play_target(
    policy_path: str | Path,
    requested_env_name: str | None,
    sim: str | None = None,
) -> PlayTarget:
    policy_path = Path(policy_path)
    logger.info(f"Using specified policy: {policy_path}")
    metadata_result = find_metadata_for_policy(policy_path)
    if metadata_result is None:
        raise FileNotFoundError(
            f"No metadata.json found for policy {policy_path}. Pass a policy from a metadata-backed run."
        )

    run_dir, metadata = metadata_result
    if requested_env_name is not None and requested_env_name != metadata.env_name:
        raise ValueError(f"env={requested_env_name!r} does not match policy metadata env {metadata.env_name!r}")
    if sim is not None:
        metadata = replace(metadata, sim=sim)
    logger.info(f"Using policy metadata: {metadata.rllib}.{metadata.algo} ({metadata.train_backend})")
    return PlayTarget(run=runs.open_run_context(run_dir, metadata), policy_path=policy_path)


def _resolve_play_target(cfg: PlayConfig) -> PlayTarget:
    if cfg.policy is not None:
        return _policy_play_target(cfg.policy, cfg.env, sim=cfg.sim)

    rllib = cfg.rllib
    if rllib is not None:
        logger.warning("rllib= is deprecated for play and is only used to filter metadata-backed runs.")
        logger.info(f"Using specified RL framework: {rllib}")

    target = _latest_play_target(cfg.env or DEFAULT_ENV_NAME, rllib=rllib)
    logger.info(f"Auto-discovered best policy: {target.policy_path}")
    if cfg.sim is not None:
        metadata = replace(target.run.metadata, sim=cfg.sim)
        target = replace(target, run=runs.open_run_context(target.run.run_dir, metadata))
    return target


def _play_render_config(target: PlayTarget, cfg: PlayConfig) -> RenderConfig:
    if not cfg.record_video:
        return RenderConfig()
    if cfg.record_seconds <= 0.0:
        raise ValueError(f"record_seconds must be positive, got {cfg.record_seconds}")
    if cfg.record_width <= 0:
        raise ValueError(f"record_width must be positive, got {cfg.record_width}")
    if cfg.record_height <= 0:
        raise ValueError(f"record_height must be positive, got {cfg.record_height}")
    path = target.run.run_dir / "play_video.mp4"
    num_frames = max(1, int(round(cfg.record_seconds * PLAY_RECORDING_FPS)))
    return RenderConfig(
        headless=True,
        path=path,
        fps=PLAY_RECORDING_FPS,
        num_frames=num_frames,
        width=cfg.record_width,
        height=cfg.record_height,
    )


def run(cfg: PlayConfig) -> None:
    try:
        target = _resolve_play_target(cfg)
        render = _play_render_config(target, cfg)
        trainer = runner.create_run_handle(
            target.run,
            cfg_override=cfg.rl,
            play_num_envs=cfg.num_envs,
            seed=cfg.seed,
            randomize_seed=cfg.rand_seed,
            render=render,
        )
        trainer.play(str(target.policy_path))
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
        logger.error("Please train a model first or specify a metadata-backed policy path")


@hydra.main(version_base=None, config_path="../configs", config_name="play")
def main(cfg: DictConfig) -> None:
    run(to_typed_config(cfg, PlayConfig))


if __name__ == "__main__":
    main()
