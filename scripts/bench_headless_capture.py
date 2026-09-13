# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Standalone motrixsim headless capture benchmark (issue #37 probe).

Bypasses MotrixLab entirely: loads one MJCF scene, launches a headless
``RenderApp`` with ``batch=N`` instances on a grid, and records a fixed number
of frames through ``system_camera.capture()`` + ``sync(wait=True)``, the same
path as ``MotrixSimRenderer.capture``.

Phases are timed separately so the superlinear term can be attributed to
submission, blocking readback, or host-side encode:

- ``render``:   ``sync(data=...)`` without wait (submission only)
- ``wait``:     ``sync(data=..., wait=True)`` (blocking GPU->CPU readback)
- ``take``:     ``take_image()`` + pixel normalization
- ``encode``:   ``imageio.append_data`` (libx264, same settings as VideoRecorder)

Examples:
    python scripts/bench_headless_capture.py --batches 1 4 16 --frames 90
    python scripts/bench_headless_capture.py --batch 16 --save /tmp/out.mp4
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import motrixsim as mtx
import numpy as np
from motrixsim.render import RenderApp, RenderSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = REPO_ROOT / "motrix_envs/src/motrix_envs/locomotion/ball_balance/assets/basketball.xml"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 90
DEFAULT_FPS = 30
DEFAULT_SPACING = 2.0


def _render_layout(num_envs: int, spacing: float) -> list[list[float]]:
    cols = int(math.ceil(math.sqrt(num_envs)))
    return [[(i % cols) * spacing, (i // cols) * spacing, 0.0] for i in range(num_envs)]


def _frame_pixels(image) -> np.ndarray:
    frame = np.asarray(image.pixels)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return np.ascontiguousarray(frame, dtype=np.uint8)


def bench(
    batch: int,
    num_frames: int,
    width: int,
    height: int,
    fps: int,
    scene_path: Path,
    spacing: float,
    save: Path | None,
    pipelined: bool = False,
    render_fps: int | None = 30,
) -> dict[str, float]:
    model = mtx.load_model(str(scene_path))
    data = mtx.SceneData(model, batch=[batch])
    data.dof_pos[:, 2] += 1.0  # something nontrivial to look at

    model.cameras.set_system_render_target("image", width, height)
    offsets = _render_layout(batch, spacing)
    settings = RenderSettings.performance()
    settings.enable_shadow = True

    t0 = time.perf_counter()
    render = RenderApp(headless=True, fps=render_fps)
    render.launch(model, batch=batch, render_offset=offsets, render_settings=settings)
    render.system_camera.set_view([offsets[-1][0] / 2, offsets[-1][1] / 2, 0.75], 6.0, -10.0, 180.0)
    render.system_camera.active = True
    launch_s = time.perf_counter() - t0

    writer = (
        imageio.get_writer(str(save), fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
        if save
        else None
    )
    totals = {"render": 0.0, "wait": 0.0, "take": 0.0, "encode": 0.0}

    def _grab(task):
        image = task.take_image()
        if image is None:
            raise RuntimeError("capture returned no image")
        return _frame_pixels(image)

    if pipelined:
        # Fix-shaped schedule: submit once per frame, never block in the same
        # iteration that submitted the capture. The blocking sync of frame N
        # drains the capture of frame N-1, whose pixels are encoded while the
        # GPU is already rendering frame N.
        pending = None
        for step in range(num_frames):
            data.dof_pos[:, 2] += 0.01  # pretend physics stepped

            t = time.perf_counter()
            render.sync(data=data)
            task = render.system_camera.capture()
            totals["render"] += time.perf_counter() - t

            if pending is not None:
                t = time.perf_counter()
                render.sync(data=data, wait=True)  # drains frame N-1, not N
                frame = _grab(pending)
                totals["wait"] += time.perf_counter() - t
                t = time.perf_counter()
                totals["take"] += time.perf_counter() - t
                if writer is not None:
                    t = time.perf_counter()
                    writer.append_data(frame)
                    totals["encode"] += time.perf_counter() - t
            pending = task

        if pending is not None:
            render.sync(data=None, wait=True)
            frame = _grab(pending)
            if writer is not None:
                writer.append_data(frame)
    else:
        # Current MotrixSimRenderer.capture() shape: blocking readback in the
        # same iteration as the submission.
        for step in range(num_frames):
            data.dof_pos[:, 2] += 0.01  # pretend physics stepped

            t = time.perf_counter()
            render.sync(data=data)  # submission only, mirrors MotrixSimRenderer.render
            totals["render"] += time.perf_counter() - t

            t = time.perf_counter()
            task = render.system_camera.capture()
            render.sync(data=data, wait=True)  # blocking readback, mirrors capture()
            totals["wait"] += time.perf_counter() - t

            t = time.perf_counter()
            frame = _grab(task)
            totals["take"] += time.perf_counter() - t

            if writer is not None:
                t = time.perf_counter()
                writer.append_data(frame)
                totals["encode"] += time.perf_counter() - t

    if writer is not None:
        writer.close()
    render.__exit__(None, None, None)

    return {"launch": launch_s, **totals}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--batch", type=int, help="run a single batch size only")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    parser.add_argument("--save", type=Path, help="also encode frames to this mp4")
    parser.add_argument("--save-all", action="store_true", help="save one mp4 per batch size")
    parser.add_argument(
        "--pipelined",
        action="store_true",
        help="fix-shaped schedule: readback of frame N-1 drains while frame N renders",
    )
    parser.add_argument(
        "--render-fps",
        type=int,
        default=30,
        help="headless render loop cap; 0 means unlimited (default: 30)",
    )
    args = parser.parse_args(argv)

    batches = [args.batch] if args.batch is not None else args.batches
    print(
        f"scene={args.scene.name} frames={args.frames} {args.width}x{args.height}"
        f" shadow=on encode={'yes' if args.save or args.save_all else 'no'}"
        f" schedule={'pipelined' if args.pipelined else 'serial'}"
    )
    header = (
        f"{'batch':>5} {'launch':>8} {'render':>8} {'wait':>8} {'take':>8} {'encode':>8} {'total':>9} {'s/frame':>8}"
    )
    print(header)
    for batch in batches:
        save = None
        if args.save:
            save = (
                args.save.with_name(f"{args.save.stem}_b{batch}{args.save.suffix}") if len(batches) > 1 else args.save
            )
        elif args.save_all:
            save = Path(f"/tmp/headless_bench_b{batch}.mp4")
        timings = bench(
            batch=batch,
            num_frames=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            scene_path=args.scene,
            spacing=args.spacing,
            save=save,
            pipelined=args.pipelined,
            render_fps=(args.render_fps or None),
        )
        total = sum(timings[k] for k in ("render", "wait", "take", "encode"))
        per_frame = total / args.frames
        if save is not None and save.is_file():
            print(f"  saved {save}")
        print(
            f"{batch:>5} {timings['launch']:>7.2f}s {timings['render']:>7.2f}s"
            f" {timings['wait']:>7.2f}s {timings['take']:>7.2f}s {timings['encode']:>7.2f}s"
            f" {total:>8.2f}s {per_frame:>7.3f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
