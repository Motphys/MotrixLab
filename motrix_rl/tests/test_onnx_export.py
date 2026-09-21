# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

from motrix_rl import checkpoints, frameworks, runs
from motrix_rl.cli import to_typed_config
from motrix_rl.config import TrainConfig
from motrix_rl.deploy import OnnxParityConfig, OnnxPolicyExporter, PolicyTensorSpec, export_onnx
from motrix_rl.deploy.onnx_validation import validate_onnx_policy

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _train_config(task: str, overrides: list[str]) -> TrainConfig:
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


def test_fastsac_export_restores_actor_and_normalizer(tmp_path: Path) -> None:
    from motrix_rl.fastsac.buffer import EmpiricalNormalization
    from motrix_rl.fastsac.networks import Actor

    config = _train_config(
        "g1-wbt-dance/motrix.fastsac",
        ["algo.agent.actor_hidden_dim=16", "algo.agent.compile=false", "algo.agent.amp=false"],
    )
    actor_cfg = config.algo.agent
    actor = Actor(
        n_obs=5,
        n_act=3,
        hidden_dim=actor_cfg.actor_hidden_dim,
        log_std_max=actor_cfg.log_std_max,
        log_std_min=actor_cfg.log_std_min,
        use_tanh=actor_cfg.use_tanh,
        use_layer_norm=actor_cfg.use_layer_norm,
        action_scale=torch.tensor([0.5, 1.0, 2.0]),
        action_bias=torch.tensor([-0.25, 0.0, 0.5]),
        device="cpu",
    )
    with torch.no_grad():
        for index, parameter in enumerate(actor.parameters()):
            parameter.fill_(0.01 * (index + 1))
    normalizer = EmpiricalNormalization(shape=5, device="cpu")
    normalizer._mean.copy_(torch.tensor([[0.5, -0.25, 1.0, 0.0, 0.75]]))
    normalizer._var.copy_(torch.tensor([[4.0, 0.25, 2.0, 1.0, 0.5]]))
    normalizer._std.copy_(torch.sqrt(normalizer._var))
    normalizer.count.fill_(128)
    actor_state = actor.state_dict()
    run = _run_with_checkpoint(
        tmp_path,
        config,
        {"actor": actor_state, "obs_normalizer": normalizer.state_dict()},
    )

    result = export_onnx(run.run_dir, validation_seed=17, validation_samples=8)

    assert result.path.is_file()
    assert result.report.input_spec.shape == (None, 5)
    assert result.report.output_spec.shape == (None, 3)
    assert result.report.parity.max_abs_error < 1e-5
    properties = {item.key: item.value for item in onnx.load(result.path).metadata_props}
    assert properties["rllib"] == "motrix"
    assert properties["algo"] == "fastsac"


def test_fastsac_checkpoint_excludes_runtime_wrappers() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _train_config(
        "g1-wbt-dance/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dim=16",
            "algo.agent.critic_hidden_dim=16",
            "algo.agent.num_atoms=5",
            "algo.agent.buffer_size=2",
            "algo.agent.batch_size=1",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
        ],
    )
    agent = FastSacAgent(
        obs_dim=5,
        critic_obs_dim=7,
        act_dim=3,
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )

    class RuntimeWrapper:
        def __init__(self, module) -> None:
            self._orig_mod = module

    agent._actor_runtime = RuntimeWrapper(agent.actor)
    agent._qnet_runtime = RuntimeWrapper(agent.qnet)
    agent._qnet_target_runtime = RuntimeWrapper(agent.qnet_target)

    checkpoint = agent.state_dict()

    for name in ("actor", "qnet", "qnet_target"):
        assert checkpoint[name]
        assert not any(key.startswith("_orig_mod.") for key in checkpoint[name])


def test_framework_export_support_matrix() -> None:
    assert frameworks.get_framework("rslrl").supported_policy_exports() == (("torch", "ppo"),)
    assert frameworks.get_framework("skrl").supported_policy_exports() == (("torch", "ppo"),)
    assert frameworks.get_framework("motrix").supported_policy_exports() == (("torch", "fastsac"),)


def test_onnx_policy_exporter_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        OnnxPolicyExporter()


def test_policy_tensor_spec_rejects_non_tuple_shape() -> None:
    with pytest.raises(TypeError, match="shape must be a tuple, got list"):
        PolicyTensorSpec(name="obs", shape=[None, 3], dtype="float32")  # type: ignore[arg-type]


def test_import_does_not_load_optional_export_dependencies() -> None:
    script = (
        "import sys; import motrix_rl; import motrix_rl.deploy; "
        "assert 'torch' not in sys.modules; assert 'onnx' not in sys.modules; "
        "assert 'onnxruntime' not in sys.modules; assert 'skrl' not in sys.modules; "
        "assert 'rsl_rl' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


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
