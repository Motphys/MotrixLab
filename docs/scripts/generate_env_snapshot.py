# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate a headless JPEG snapshot of a registered environment grid."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.sim.backend import RenderConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
POSTER_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "poster"
DEFAULT_NUM_ENVS = 16
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540
DEFAULT_WARMUP_SECONDS = 1.0


def _validate_options(num_envs: int, width: int, height: int, warmup_seconds: float) -> None:
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    if warmup_seconds < 0.0:
        raise ValueError(f"warmup_seconds must be non-negative, got {warmup_seconds}")


def _save_snapshot(
    env: ArrayEnv,
    path: Path,
    *,
    width: int,
    height: int,
    warmup_seconds: float,
) -> Path:
    camera = env.cfg.scene.system_camera
    # Headless RenderConfig is shaped for video recording, so the unused
    # recording fields carry inert one-frame values; snapshots are taken as
    # stills from the renderer's system-camera capture.
    renderer = env.create_renderer(
        RenderConfig(
            headless=True,
            path=path.with_suffix(".mp4"),
            fps=1,
            num_frames=1,
            width=width,
            height=height,
            camera_lookat=camera.lookat,
            camera_distance=camera.distance,
            camera_elevation=camera.elevation,
            camera_azimuth=camera.azimuth,
        )
    )
    try:
        renderer.render()
        if warmup_seconds > 0.0:
            time.sleep(warmup_seconds)
        renderer.capture()  # Discard the first capture while render assets finish loading.

        frame = renderer.capture()
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(str(path), frame)
        return path
    finally:
        renderer.close()


def generate_snapshot(
    env_name: str,
    *,
    force: bool = False,
    num_envs: int = DEFAULT_NUM_ENVS,
    output_dir: Path = POSTER_DIR,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
) -> Path:
    """Generate one overview image using the environment's configured system camera."""
    _validate_options(num_envs, width, height, warmup_seconds)
    output = output_dir / f"{env_name}.jpg"
    if output.is_file() and not force:
        return output

    env = registry.make(env_name, num_envs=num_envs)
    if not isinstance(env, ArrayEnv):
        raise TypeError(f"Snapshot rendering only supports ArrayEnv environments, got {type(env).__name__}")
    env.init_state()

    return _save_snapshot(
        env,
        output,
        width=width,
        height=height,
        warmup_seconds=warmup_seconds,
    )


def generate_missing_snapshots(
    *,
    output_dir: Path = POSTER_DIR,
    num_envs: int = DEFAULT_NUM_ENVS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS,
) -> tuple[list[Path], list[Path]]:
    """Generate snapshots for registered environments that have no output image yet."""
    _validate_options(num_envs, width, height, warmup_seconds)
    generated = []
    skipped = []
    for env_name in sorted(registry.list_registered_envs()):
        path = output_dir / f"{env_name}.jpg"
        if path.is_file():
            skipped.append(path)
            continue
        generated.append(
            generate_snapshot(
                env_name,
                num_envs=num_envs,
                output_dir=output_dir,
                width=width,
                height=height,
                warmup_seconds=warmup_seconds,
            )
        )
    return generated, skipped


def _display_path(path: Path) -> Path:
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", help="generate one registered environment; omit to generate all missing snapshots")
    parser.add_argument("--force", action="store_true", help="overwrite an existing snapshot for the selected --env")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS, help="number of rendered environments")
    parser.add_argument("--output-dir", type=Path, default=POSTER_DIR, help="output directory")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--warmup-seconds", type=float, default=DEFAULT_WARMUP_SECONDS)
    args = parser.parse_args()
    if args.force and args.env is None:
        parser.error("--force requires --env")

    try:
        if args.env is not None:
            existed = (args.output_dir / f"{args.env}.jpg").is_file()
            path = generate_snapshot(
                args.env,
                force=args.force,
                num_envs=args.num_envs,
                output_dir=args.output_dir,
                width=args.width,
                height=args.height,
                warmup_seconds=args.warmup_seconds,
            )
            action = "refreshed" if existed and args.force else "skipped existing" if existed else "saved"
            print(f"{action} environment snapshot: {_display_path(path)}")
        else:
            generated, skipped = generate_missing_snapshots(
                output_dir=args.output_dir,
                num_envs=args.num_envs,
                width=args.width,
                height=args.height,
                warmup_seconds=args.warmup_seconds,
            )
            for path in skipped:
                print(f"skipped existing environment snapshot: {_display_path(path)}")
            for path in generated:
                print(f"saved environment snapshot: {_display_path(path)}")
            print(f"environment snapshots: {len(generated)} generated, {len(skipped)} skipped")
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
