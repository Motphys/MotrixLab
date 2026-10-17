# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate a documentation video from the latest playable environment policy.

Examples:
    python docs/scripts/generate_video.py cartpole
    python docs/scripts/generate_video.py cartpole --output cartpole_demo.mp4 --force
    python docs/scripts/generate_video.py g1-wbt-dance  # records one complete motion clip
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from motrix_env_core import registry
from motrix_env_core.renderer import RenderConfig
from motrix_envs.locomotion.wbt.cfg import WbtEnvCfg
from motrix_envs.motion.library import expand_motion_paths
from motrix_envs.motion.loader import MotrixMotion
from motrix_rl import checkpoints, runner, runs

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
VIDEO_DIR = REPO_ROOT / "docs" / "source" / "_static" / "videos"
DEFAULT_FPS = 30
DEFAULT_NUM_ENVS = 16
DEFAULT_RECORD_SECONDS = 10.0
DEFAULT_RECORD_WIDTH = 1280
DEFAULT_RECORD_HEIGHT = 720


class PolicyUnavailableError(RuntimeError):
    """Raised when an environment has no metadata-backed playable policy."""


@dataclass(frozen=True)
class PlayTarget:
    run: runs.RunContext
    policy_path: Path


def _target_from_policy(env_name: str, policy: Path, sim: str | None) -> PlayTarget:
    metadata_result = runs.find_metadata_for_policy(policy)
    if metadata_result is None:
        raise PolicyUnavailableError(f"No metadata.json was found for policy {policy}.")

    run_dir, metadata = metadata_result
    if metadata.env_name != env_name:
        raise ValueError(f"Environment {env_name!r} does not match policy metadata {metadata.env_name!r}.")
    if sim is not None:
        metadata = replace(metadata, sim=sim)
    policy_path = checkpoints.resolve_checkpoint_path(policy, metadata=metadata)
    return PlayTarget(runs.open_run_context(run_dir, metadata), policy_path)


def _latest_playable_target(env_name: str, sim: str | None) -> PlayTarget:
    metadata_runs = sorted(
        runs.iter_metadata_runs(env_name, runs_root=RUNS_ROOT),
        key=lambda item: item[0].stat().st_mtime,
        reverse=True,
    )
    for run_dir, metadata in metadata_runs:
        try:
            policy_path = checkpoints.best_policy(metadata, run_dir)
        except FileNotFoundError:
            continue
        if not runs.task_config_path(run_dir).is_file():
            continue
        if sim is not None:
            metadata = replace(metadata, sim=sim)
        return PlayTarget(runs.open_run_context(run_dir, metadata), policy_path)

    raise PolicyUnavailableError(f"No playable policy was found for environment {env_name!r}.")


def _resolve_target(env_name: str, policy: Path | None, sim: str | None) -> PlayTarget:
    if not registry.contains(env_name):
        raise ValueError(f"Environment {env_name!r} is not registered.")
    if policy is not None:
        return _target_from_policy(env_name, policy, sim)
    return _latest_playable_target(env_name, sim)


def _default_train_command(env_name: str) -> str | None:
    config_dir = REPO_ROOT / "configs" / "task" / env_name
    candidates = {path.name: path for path in config_dir.glob("*.yaml")}
    preferred = ("skrl.ppo.yaml", "rslrl.ppo.yaml", "motrix.fastsac.yaml")
    config = next((candidates[name] for name in preferred if name in candidates), None)
    if config is None and candidates:
        config = candidates[sorted(candidates)[0]]
    if config is None:
        return None
    return f"python -m motrix_rl.commands.train task={env_name}/{config.name.removesuffix('.yaml')}"


def _recording_path(env_name: str, output: Path | None) -> Path:
    if output is None:
        return VIDEO_DIR / f"{env_name}.mp4"
    if output.is_absolute():
        return output
    if output.parent == Path("."):
        return VIDEO_DIR / output
    return REPO_ROOT / output


def _record_seconds(env_name: str, requested_seconds: float | None) -> float:
    """Resolve an explicit duration or derive one complete pass of the WBT motion source."""
    if requested_seconds is not None:
        if requested_seconds <= 0:
            raise ValueError(f"--seconds must be positive, got {requested_seconds}.")
        return requested_seconds

    env_cfg = registry.make_env_config(env_name, mode="play")
    if isinstance(env_cfg, WbtEnvCfg):
        motions = [MotrixMotion(path) for path in expand_motion_paths(env_cfg.commands.motion.motion_files)]
        return sum(motion.num_frames / motion.fps for motion in motions)
    return DEFAULT_RECORD_SECONDS


def _render_config(args: argparse.Namespace) -> RenderConfig:
    seconds = _record_seconds(args.env, args.seconds)
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}.")
    if args.width <= 0 or args.width % 2:
        raise ValueError(f"--width must be a positive even integer, got {args.width}.")
    if args.height <= 0 or args.height % 2:
        raise ValueError(f"--height must be a positive even integer, got {args.height}.")

    path = _recording_path(args.env, args.output)
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Video output must use the .mp4 extension: {path}")
    if path.exists() and not args.force:
        raise FileExistsError(f"Video already exists: {path}. Pass --force to overwrite it.")
    return RenderConfig(
        headless=True,
        path=path,
        fps=args.fps,
        num_frames=max(1, round(seconds * args.fps)),
        width=args.width,
        height=args.height,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("env", help="registered environment name")
    parser.add_argument("--policy", type=Path, help="checkpoint file or metadata-backed run directory")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=DEFAULT_NUM_ENVS,
        help="number of environments to play (default: 16, rendered as a 4x4 grid)",
    )
    parser.add_argument("--sim", help="override the simulator name stored in run metadata (manager envs)")
    parser.add_argument("--seed", type=int, help="override the seed stored in run metadata")
    parser.add_argument("--randomize-seed", action="store_true", help="ignore the run seed")
    parser.add_argument(
        "--seconds",
        type=float,
        help="recording duration; defaults to one full motion for WBT tasks and 10 seconds otherwise",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="recording frame rate")
    parser.add_argument("--width", type=int, default=DEFAULT_RECORD_WIDTH, help="recording width")
    parser.add_argument("--height", type=int, default=DEFAULT_RECORD_HEIGHT, help="recording height")
    parser.add_argument(
        "--output",
        type=Path,
        help="video path; a bare filename is placed in docs/source/_static/videos",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing video")
    return parser


def run(args: argparse.Namespace) -> Path:
    if args.num_envs <= 0:
        raise ValueError(f"--num-envs must be positive, got {args.num_envs}.")

    target = _resolve_target(args.env, args.policy, args.sim)
    render = _render_config(args)
    logging.info("Using policy: %s", target.policy_path)
    logging.info("Recording %d frames to %s", render.num_frames, render.path)
    trainer = runner.create_run_handle(
        target.run,
        play_num_envs=args.num_envs,
        seed=args.seed,
        randomize_seed=args.randomize_seed,
        render=render,
    )
    trainer.play(str(target.policy_path))

    if not render.path.is_file():
        raise RuntimeError(f"Playback finished without creating video {render.path}.")
    return render.path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parser().parse_args(argv)
    try:
        output = run(args)
    except PolicyUnavailableError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        train_command = _default_train_command(args.env)
        if train_command is None:
            print(
                "Train this environment and record a best_policy artifact before generating a video.", file=sys.stderr
            )
        else:
            print(f"Train a policy first, for example:\n  {train_command}", file=sys.stderr)
        return 2
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

    print(f"Video saved to {output.relative_to(REPO_ROOT) if output.is_relative_to(REPO_ROOT) else output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
