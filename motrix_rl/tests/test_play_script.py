# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

import pytest

from motrix_rl import runs
from motrix_rl.config import PlayConfig

PLAY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "play.py"
PLAY_MODULE_SPEC = importlib.util.spec_from_file_location("motrix_test_play_script", PLAY_SCRIPT)
if PLAY_MODULE_SPEC is None or PLAY_MODULE_SPEC.loader is None:
    raise ImportError(f"Unable to load {PLAY_SCRIPT}")
play = importlib.util.module_from_spec(PLAY_MODULE_SPEC)
sys.modules[PLAY_MODULE_SPEC.name] = play
PLAY_MODULE_SPEC.loader.exec_module(play)


def _metadata_run(tmp_path: Path, env_name: str = "go2", sim: str | None = None) -> Path:
    run_dir = tmp_path / "run"
    policy_path = run_dir / "checkpoints" / "best.pt"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("checkpoint", encoding="utf-8")
    runs.write_metadata(
        run_dir,
        runs.make_metadata(env_name, "rslrl", "torch", "ppo", 1, "pt", sim=sim),
    )
    return policy_path


def test_policy_target_uses_metadata_when_optional_overrides_are_unset(tmp_path):
    policy_path = _metadata_run(tmp_path)

    target = play._policy_play_target(policy_path, None)

    assert target.run.metadata.env_name == "go2"
    assert target.run.metadata.sim is None


def test_policy_target_validates_explicit_env_and_overrides_sim(tmp_path):
    policy_path = _metadata_run(tmp_path)

    with pytest.raises(ValueError, match="does not match policy metadata"):
        play._policy_play_target(policy_path, "cartpole")

    target = play._policy_play_target(policy_path, "go2", sim="motrixsim")
    assert target.run.metadata.sim == "motrixsim"


def test_default_play_target_uses_cartpole(monkeypatch, tmp_path):
    policy_path = _metadata_run(tmp_path, env_name="cartpole")
    expected = play._policy_play_target(policy_path, None)
    requested = []

    def latest_target(env_name: str, rllib: str | None = None):
        requested.append((env_name, rllib))
        return expected

    monkeypatch.setattr(play, "_latest_play_target", latest_target)

    assert play._resolve_play_target(PlayConfig()) == expected
    assert requested == [("cartpole", None)]
