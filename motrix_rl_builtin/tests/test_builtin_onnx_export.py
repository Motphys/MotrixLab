# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import onnx
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

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


def test_fastsac_export_restores_actor_and_normalizer(tmp_path: Path) -> None:
    from motrix_rl_builtin.fastsac.buffer import EmpiricalNormalization
    from motrix_rl_builtin.fastsac.networks import Actor

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
    from motrix_rl_builtin.fastsac.agent import FastSacAgent

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
        cfg=config.algo.agent,
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
