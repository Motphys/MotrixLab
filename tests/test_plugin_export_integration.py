# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from motrix_rl import checkpoints, frameworks, runs
from motrix_rl.cli import to_typed_config
from motrix_rl.config import TrainConfig
from motrix_rl.deploy import export_onnx

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _train_config(task: str, overrides: list[str]) -> TrainConfig:
    from motrix_rl.plugins import load_plugins

    load_plugins()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        config = compose(config_name="train", overrides=[f"task={task}", *overrides])
    return to_typed_config(config, TrainConfig)


def _run_with_checkpoint(tmp_path: Path, config: TrainConfig, checkpoint: dict) -> runs.RunContext:
    run = runs.create_run_context(
        env_name=config.task.env,
        rllib=config.task.rllib,
        train_backend="torch",
        algo=config.task.algo,
        seed=config.seed,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    runs.write_task_config(run.run_dir, config)
    checkpoint_path = run.checkpoint_dir / "best.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoints.record_checkpoint_artifact(
        run.run_dir,
        checkpoints.BEST_POLICY,
        checkpoint_path,
        checkpoints.POLICY,
        checkpoint_format="pt",
    )
    return run


def test_framework_export_support_matrix() -> None:
    assert frameworks.get_framework("rslrl").supported_policy_exports() == (("torch", "ppo"),)
    assert frameworks.get_framework("skrl").supported_policy_exports() == (("torch", "ppo"),)
    assert frameworks.get_framework("motrix").supported_policy_exports() == (("torch", "fastsac"),)


def test_export_rejects_non_onnx_output_before_replacing_file(tmp_path: Path) -> None:
    config = _train_config(
        "cartpole/skrl.ppo",
        ["task.train_backend=torch", "algo.models.policy.hiddens=[2]"],
    )
    policy = {
        "net.0.weight": torch.ones(2, 3),
        "net.0.bias": torch.zeros(2),
        "mean_layer.weight": torch.ones(1, 2),
        "mean_layer.bias": torch.zeros(1),
    }
    normalizer = {
        "running_mean": torch.zeros(3, dtype=torch.float64),
        "running_variance": torch.ones(3, dtype=torch.float64),
        "current_count": torch.tensor(1.0, dtype=torch.float64),
    }
    run = _run_with_checkpoint(
        tmp_path,
        config,
        {"policy": policy, "observation_preprocessor": normalizer},
    )

    with pytest.raises(ValueError, match="must end in .onnx"):
        export_onnx(run.run_dir, tmp_path / "policy.bin")
    assert not (tmp_path / "policy.bin").exists()
