# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0


from dataclasses import asdict
from pathlib import Path

import onnx
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

from motrix_rl import checkpoints, runs
from motrix_rl.cli import to_typed_config
from motrix_rl.config import TrainConfig
from motrix_rl.deploy import export_onnx

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


def test_rslrl_torch_export_uses_framework_normalizer(tmp_path: Path) -> None:
    from rsl_rl.models import MLPModel

    config = _train_config(
        "cartpole/rslrl.ppo",
        ["algo.actor.hidden_dims=[5,4]", "algo.actor.obs_normalization=true"],
    )
    actor_config = asdict(config.algo.actor)
    actor_config.pop("class_name")
    actor = MLPModel(
        TensorDict({"policy": torch.zeros(1, 3)}, batch_size=[1]),
        config.algo.obs_groups,
        "actor",
        output_dim=2,
        **actor_config,
    )
    with torch.no_grad():
        actor.obs_normalizer._mean.copy_(torch.tensor([[0.5, -0.25, 1.0]]))
        actor.obs_normalizer._var.copy_(torch.tensor([[4.0, 0.25, 2.0]]))
        actor.obs_normalizer._std.copy_(torch.sqrt(actor.obs_normalizer._var))
        for index, parameter in enumerate(actor.parameters()):
            parameter.fill_(0.02 * (index + 1))
    run = _run_with_checkpoint(tmp_path, config, {"actor_state_dict": actor.state_dict()})

    result = export_onnx(run.run_dir, validation_seed=7, validation_samples=8)

    assert result.path.is_file()
    assert result.report.input_spec.name == "obs"
    assert result.report.input_spec.shape == (None, 3)
    assert result.report.output_spec.name == "actions"
    assert result.report.output_spec.shape == (None, 2)
    assert result.report.parity.samples == 8
    assert result.report.parity.max_abs_error < 1e-5
    properties = {item.key: item.value for item in onnx.load(result.path).metadata_props}
    assert properties["rllib"] == "rslrl"
    assert properties["obs_dim"] == "3"
