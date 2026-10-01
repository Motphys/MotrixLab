# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from motrix_rl import checkpoints, runs
from motrix_rl.cli import to_typed_config
from motrix_rl.config import TrainConfig
from motrix_rl.deploy import OnnxParityConfig, export_onnx
from motrix_rl.deploy.onnx_validation import validate_onnx_policy

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


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


def test_skrl_torch_export_bakes_observation_preprocessor(tmp_path: Path) -> None:
    config = _train_config(
        "cartpole/skrl.ppo",
        ["task.train_backend=torch", "algo.models.policy.hiddens=[5,4]"],
    )
    generator = torch.Generator().manual_seed(11)
    policy = {
        "net.0.weight": torch.randn(5, 3, generator=generator) * 0.1,
        "net.0.bias": torch.randn(5, generator=generator) * 0.1,
        "net.2.weight": torch.randn(4, 5, generator=generator) * 0.1,
        "net.2.bias": torch.randn(4, generator=generator) * 0.1,
        "mean_layer.weight": torch.randn(2, 4, generator=generator) * 0.1,
        "mean_layer.bias": torch.randn(2, generator=generator) * 0.1,
        "log_std_parameter": torch.zeros(2),
    }
    normalizer = {
        "running_mean": torch.tensor([0.5, -0.25, 1.0], dtype=torch.float64),
        "running_variance": torch.tensor([4.0, 0.25, 2.0], dtype=torch.float64),
        "current_count": torch.tensor(128.0, dtype=torch.float64),
    }
    run = _run_with_checkpoint(
        tmp_path,
        config,
        {"policy": policy, "observation_preprocessor": normalizer},
    )

    result = export_onnx(run.run_dir, validation_seed=13, validation_samples=8)

    assert result.path.is_file()
    assert result.report.input_spec.shape == (None, 3)
    assert result.report.output_spec.shape == (None, 2)
    assert result.report.parity.max_abs_error < 1e-5
    model = onnx.load(result.path)
    assert any(node.op_type == "Clip" for node in model.graph.node)
    properties = {item.key: item.value for item in model.metadata_props}
    assert properties["rllib"] == "skrl"
    assert properties["train_backend"] == "torch"

    with pytest.raises(ValueError, match="parity failed"):
        validate_onnx_policy(
            result.path.read_bytes(),
            lambda observations: np.full((observations.shape[0], 2), 100.0, dtype=np.float32),
            observation_size=3,
            action_size=2,
            config=OnnxParityConfig(seed=1, samples=4, atol=0.0, rtol=0.0),
            source="test",
        )


def test_skrl_checkpoint_requires_observation_preprocessor(tmp_path: Path) -> None:
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
    run = _run_with_checkpoint(tmp_path, config, {"policy": policy})

    with pytest.raises(ValueError, match="observation_preprocessor"):
        export_onnx(run.run_dir)
