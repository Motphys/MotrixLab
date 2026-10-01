# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from motrix_env_core.renderer import RenderConfig
from motrix_envs.locomotion.wbt.g1.common import G1WbtEnvCfg
from motrix_rl import checkpoints, runs

GENERATE_VIDEO_SCRIPT = Path(__file__).resolve().parents[1] / "docs" / "scripts" / "generate_video.py"
GENERATE_VIDEO_MODULE_SPEC = importlib.util.spec_from_file_location(
    "motrix_test_doc_generate_video_script", GENERATE_VIDEO_SCRIPT
)
if GENERATE_VIDEO_MODULE_SPEC is None or GENERATE_VIDEO_MODULE_SPEC.loader is None:
    raise ImportError(f"Unable to load {GENERATE_VIDEO_SCRIPT}")
generate_video = importlib.util.module_from_spec(GENERATE_VIDEO_MODULE_SPEC)
sys.modules[GENERATE_VIDEO_MODULE_SPEC.name] = generate_video
GENERATE_VIDEO_MODULE_SPEC.loader.exec_module(generate_video)


def _metadata_run(root: Path, name: str, *, playable: bool) -> Path:
    run_dir = root / "cartpole" / name
    runs.write_metadata(run_dir, runs.make_metadata("cartpole", "skrl", "torch", "ppo", "np", 1, "pt"))
    runs.task_config_path(run_dir).write_text("task: {}\n", encoding="utf-8")
    if playable:
        policy = run_dir / "checkpoints" / "best.pt"
        policy.parent.mkdir(parents=True, exist_ok=True)
        policy.write_text("checkpoint", encoding="utf-8")
        checkpoints.record_checkpoint_artifact(run_dir, checkpoints.BEST_POLICY, policy, checkpoints.POLICY)
    return run_dir


def test_latest_playable_target_skips_newer_incomplete_run(monkeypatch, tmp_path):
    playable_run = _metadata_run(tmp_path, "playable", playable=True)
    incomplete_run = _metadata_run(tmp_path, "incomplete", playable=False)
    os.utime(playable_run, (1, 1))
    os.utime(incomplete_run, (2, 2))
    monkeypatch.setattr(generate_video, "RUNS_ROOT", tmp_path)

    target = generate_video._latest_playable_target("cartpole", None)

    assert target.run.run_dir == playable_run
    assert target.policy_path.name == "best.pt"


def test_sim_override_replaces_metadata_sim_field(monkeypatch, tmp_path):
    _metadata_run(tmp_path, "playable", playable=True)
    monkeypatch.setattr(generate_video, "RUNS_ROOT", tmp_path)

    target = generate_video._latest_playable_target("cartpole", "motrixsim")

    assert target.run.metadata.sim == "motrixsim"


def test_missing_policy_prints_train_command(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(generate_video, "RUNS_ROOT", tmp_path)

    assert generate_video.main(["cartpole"]) == 2

    error = capsys.readouterr().err
    assert "No playable policy" in error
    assert "uv run scripts/train.py task=cartpole/skrl.ppo" in error


def test_video_config_records_by_default_and_places_bare_output_in_docs_video_dir():
    args = generate_video._parser().parse_args(["cartpole", "--output", "demo.mp4"])

    render = generate_video._render_config(args)

    assert args.num_envs == 16
    assert isinstance(render, RenderConfig)
    assert render.headless
    assert render.path == generate_video.VIDEO_DIR / "demo.mp4"
    assert render.num_frames == generate_video.DEFAULT_FPS * int(generate_video.DEFAULT_RECORD_SECONDS)


def test_record_config_requires_force_to_overwrite(tmp_path):
    output = tmp_path / "demo.mp4"
    output.write_bytes(b"existing")
    args = generate_video._parser().parse_args(["cartpole", "--output", str(output)])

    with pytest.raises(FileExistsError, match="--force"):
        generate_video._render_config(args)


def test_wbt_default_duration_spans_the_whole_motion_source(monkeypatch):
    cfg = G1WbtEnvCfg()
    cfg.commands.motion.motion_files = ("demo_a.npz", "demo_b.npz")

    class FakeMotion:
        def __init__(self, path):
            assert path in (Path("demo_a.npz"), Path("demo_b.npz"))
            self.num_frames = 625 if path.name == "demo_a.npz" else 375
            self.fps = 50

    monkeypatch.setattr(generate_video.registry, "make_env_config", lambda env, mode: cfg)
    monkeypatch.setattr(generate_video, "MotrixMotion", FakeMotion)

    assert generate_video._record_seconds("demo-wbt", None) == pytest.approx(20.0)
    assert generate_video._record_seconds("demo-wbt", 3.0) == 3.0
