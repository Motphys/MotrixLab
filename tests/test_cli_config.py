# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir, initialize_config_module
from hydra.core.global_hydra import GlobalHydra

from motrix_rl import frameworks
from motrix_rl.cli import to_typed_config
from motrix_rl.config import OnnxExportConfig, TrainConfig
from motrix_rl.plugins import load_plugins

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(config_name: str, overrides: list[str]):
    load_plugins()
    initializer = (
        initialize_config_module(version_base=None, config_module="motrix_rl.configs")
        if config_name in ("train", "play", "export_onnx")
        else initialize_config_dir(version_base=None, config_dir=CONFIG_DIR)
    )
    if config_name == "train":
        overrides = [f"hydra.searchpath=[file://{CONFIG_DIR}]", *overrides]
    with initializer:
        return compose(config_name=config_name, overrides=overrides)


def test_selected_train_config_composes_to_selected_provider_type():
    cfg = _compose("train", ["task=cartpole/skrl.ppo"])

    typed_cfg = to_typed_config(cfg, TrainConfig)

    assert isinstance(typed_cfg, TrainConfig)
    assert isinstance(typed_cfg.algo, frameworks.get_config_type(typed_cfg.task.rllib, typed_cfg.task.algo))


def test_train_scalar_overrides():
    cfg = _compose(
        "train",
        [
            "task=dm-walker/rslrl.ppo",
            "num_envs=4096",
            "seed=123",
            "logging.interval=20",
            "checkpoint.interval=50",
        ],
    )
    assert cfg.task.env == "dm-walker"
    assert cfg.task.rllib == "rslrl"
    assert cfg.task.algo == "ppo"
    assert cfg.num_envs == 4096
    assert cfg.seed == 123
    assert cfg.logging.interval == 20
    assert cfg.checkpoint.interval == 50


def test_train_nested_algo_override():
    cfg = _compose("train", ["task=cartpole/skrl.ppo", "algo.agent.learning_rate=1e-3", "seed=7"])
    assert cfg.algo.agent.learning_rate == 1e-3
    assert cfg.seed == 7


def test_play_and_view_compose():
    play = _compose(
        "play",
        ["policy=/tmp/best.pt", "num_envs=16", "record_video=true", "record_seconds=2.5"],
    )
    assert play.policy == "/tmp/best.pt"
    assert play.num_envs == 16
    assert play.record_video is True
    assert play.record_seconds == 2.5

    default_play = _compose("play", [])
    assert default_play.env is None

    view = _compose("view", ["env=cartpole", "num_envs=1"])
    assert view.env == "cartpole"
    assert view.robot is None
    assert view.num_envs == 1

    robot_view = _compose("view", ["robot=go2"])
    assert robot_view.env is None
    assert robot_view.robot == "go2"


def test_export_onnx_compose():
    cfg = _compose(
        "export_onnx",
        [
            "run_dir=/tmp/training-run",
            "output=/tmp/policy.onnx",
            "opset=17",
            "parity.seed=7",
            "parity.samples=64",
            "parity.atol=1e-3",
        ],
    )

    typed_cfg = to_typed_config(cfg, OnnxExportConfig)

    assert typed_cfg.run_dir == "/tmp/training-run"
    assert typed_cfg.output == "/tmp/policy.onnx"
    assert typed_cfg.opset == 17
    assert typed_cfg.parity.seed == 7
    assert typed_cfg.parity.samples == 64
    assert typed_cfg.parity.atol == 1e-3
