# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from motrix_rl.fastsac.factory import resolve_policy_variant
from motrix_rl.fastsac.sonic import SONIC_ENCODER_COUNT

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose(task: str, overrides: list[str]) -> TrainConfig:
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


def test_policy_variant_registry_resolves_names() -> None:
    default = resolve_policy_variant("default")
    sonic = resolve_policy_variant("sonic")

    assert default.name == "default"
    assert sonic.name == "sonic"
    with pytest.raises(ValueError, match=r"unknown FastSAC policy variant 'missing';.*default, sonic"):
        resolve_policy_variant("missing")


def test_sonic_canonical_variant_builds_actor() -> None:
    config = _compose("g1-sonic/motrix.fastsac", [])
    variant = resolve_policy_variant(config.algo.policy_variant)
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    act_dim = model["action_dim"]

    actor = variant.build_actor(
        config.algo,
        (obs_dim, act_dim),
        torch.ones(act_dim),
        torch.zeros(act_dim),
        device="cpu",
    )

    assert actor.n_obs == obs_dim
    assert actor.n_act == act_dim
    assert variant.passthrough_dims(config.algo) == SONIC_ENCODER_COUNT
    assert variant.aux_loss_weights(config.algo) == {
        name: float(weight) for name, weight in config.algo.variant["auxiliary"].items()
    }
    torch.testing.assert_close(actor.policy_head.action_scale, torch.ones(act_dim))
    torch.testing.assert_close(actor.policy_head.action_bias, torch.zeros(act_dim))
    observations = torch.zeros(2, obs_dim)
    observations[:, -2] = 1.0
    actions, log_probs, auxiliary = actor.get_actions_and_log_probs_with_aux(observations)
    assert actions.shape == (2, act_dim)
    assert log_probs.shape == (2,)
    assert set(auxiliary) == set(config.algo.variant["auxiliary"])
    assert all(torch.isfinite(loss) for loss in auxiliary.values())


def _tiny_sonic_config() -> TrainConfig:
    return _compose(
        "g1-sonic/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dim=16",
            "algo.agent.critic_hidden_dim=16",
            "algo.agent.num_q_networks=1",
            "algo.agent.num_atoms=5",
            "algo.agent.v_min=-5.0",
            "algo.agent.v_max=5.0",
            "algo.agent.buffer_size=2",
            "algo.agent.batch_size=2",
            "algo.agent.num_updates=1",
            "algo.agent.policy_frequency=1",
            "algo.agent.obs_normalization=false",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
            "algo.variant.model.num_future_frames=1",
            "algo.variant.model.num_tokens=1",
            "algo.variant.model.token_dim=4",
            "algo.variant.model.fsq_levels=8",
            "algo.variant.model.action_dim=7",
            "+algo.variant.model.g1_encoder_hidden_dims=[32,16]",
            "+algo.variant.model.smpl_encoder_hidden_dims=[32,16]",
            "+algo.variant.model.g1_motion_decoder_hidden_dims=[32,16]",
        ],
    )


def _sonic_obs_dim(config: TrainConfig) -> int:
    model = config.algo.variant["model"]
    action_dim = model["action_dim"]
    proprioceptive = 3 * action_dim + 6
    g1_reference = 2 * action_dim + 6
    smpl_reference = 24 * 3 + 6 + 2 * 3
    return model["num_future_frames"] * (proprioceptive + g1_reference + smpl_reference) + 2


def _packed(obs_dim: int, count: int) -> torch.Tensor:
    observations = torch.randn(count, obs_dim)
    observations[:, -2] = 1.0
    observations[:, -1] = 0.0
    return observations


def test_agent_weights_and_logs_sonic_auxiliary_losses() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _tiny_sonic_config()
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    agent = FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=3,
        act_dim=model["action_dim"],
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    agent.rb.extend(
        _packed(obs_dim, 1),
        torch.zeros(1, 3),
        torch.zeros(1, model["action_dim"]),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
        _packed(obs_dim, 1),
        torch.zeros(1, 3),
    )
    agent.rb.ptr = 1

    metrics = agent.update(1)

    assert set(metrics) >= {"aux_reconstruction", "aux_latent_alignment", "aux_cycle_consistency", "aux_loss"}
    assert torch.isfinite(metrics["aux_reconstruction"])
    assert torch.isfinite(metrics["aux_latent_alignment"])
    assert torch.isfinite(metrics["aux_cycle_consistency"])
    assert torch.isfinite(metrics["aux_loss"])


def test_default_variant_has_no_auxiliary_metrics() -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _compose(
        "g1-wbt-dance/motrix.fastsac",
        [
            "algo.agent.actor_hidden_dim=8",
            "algo.agent.critic_hidden_dim=8",
            "algo.agent.num_q_networks=1",
            "algo.agent.num_atoms=5",
            "algo.agent.v_min=-5.0",
            "algo.agent.v_max=5.0",
            "algo.agent.buffer_size=2",
            "algo.agent.batch_size=2",
            "algo.agent.num_updates=1",
            "algo.agent.policy_frequency=1",
            "algo.agent.obs_normalization=false",
            "algo.agent.compile=false",
            "algo.agent.amp=false",
        ],
    )
    assert config.algo.policy_variant == "default"
    assert config.algo.variant == {}
    agent = FastSacAgent(
        obs_dim=5,
        critic_obs_dim=7,
        act_dim=3,
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    agent.rb.extend(
        torch.randn(1, 5),
        torch.randn(1, 7),
        torch.zeros(1, 3),
        torch.ones(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.randn(1, 5),
        torch.randn(1, 7),
    )
    agent.rb.ptr = 1

    metrics = agent.update(1)

    assert not any(name.startswith("aux_") for name in metrics)


def test_sonic_checkpoint_export_uses_variant_metadata(tmp_path: Path) -> None:
    from motrix_rl.fastsac.agent import FastSacAgent

    config = _tiny_sonic_config()
    model = config.algo.variant["model"]
    obs_dim = _sonic_obs_dim(config)
    agent = FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=3,
        act_dim=model["action_dim"],
        num_envs=1,
        cfg=config.algo,
        device=torch.device("cpu"),
    )
    checkpoint = agent.state_dict()
    assert checkpoint["policy_variant"] == "sonic"
    assert checkpoint["policy_variant_metadata"] == {
        **config.algo.variant,
        "actor_hidden_dim": config.algo.agent.actor_hidden_dim,
        "log_std_min": config.algo.agent.log_std_min,
        "log_std_max": config.algo.agent.log_std_max,
        "use_layer_norm": config.algo.agent.use_layer_norm,
    }

    exported_config = _tiny_sonic_config()
    exported_config.algo.variant["model"]["fsq_levels"] = 16
    exported_config.algo.variant["model"]["action_dim"] = 8
    exported_config.algo.variant["model"]["num_future_frames"] = 2
    exported_config.algo.variant["model"]["g1_encoder_hidden_dims"] = [64, 32]
    run = _run_with_checkpoint(tmp_path, exported_config, checkpoint)

    result = export_onnx(run.run_dir, validation_seed=7, validation_samples=4)

    assert result.report.input_spec.shape == (None, obs_dim)
    assert result.report.output_spec.shape == (None, model["action_dim"])
    assert result.report.parity.max_abs_error < 1e-5
    properties = {item.key: item.value for item in onnx.load(result.path).metadata_props}
    assert properties["obs_dim"] == str(obs_dim)


def test_export_rejects_pre_variant_sonic_checkpoint(tmp_path: Path) -> None:
    config = _tiny_sonic_config()
    run = _run_with_checkpoint(tmp_path, config, {"actor": {}, "obs_normalizer": {}, "sonic": {"enabled": True}})

    with pytest.raises(ValueError, match="pre-PolicyVariant SONIC format"):
        export_onnx(run.run_dir, validation_samples=4)


def test_default_checkpoint_without_variant_metadata_exports(tmp_path: Path) -> None:
    from motrix_rl.fastsac.networks import Actor

    config = _compose(
        "g1-wbt-dance/motrix.fastsac",
        ["algo.agent.actor_hidden_dim=8", "algo.agent.obs_normalization=false"],
    )
    actor = Actor(
        n_obs=5,
        n_act=3,
        hidden_dim=config.algo.agent.actor_hidden_dim,
        log_std_max=config.algo.agent.log_std_max,
        log_std_min=config.algo.agent.log_std_min,
        use_tanh=config.algo.agent.use_tanh,
        use_layer_norm=config.algo.agent.use_layer_norm,
        action_scale=torch.ones(3),
        action_bias=torch.zeros(3),
        device="cpu",
    )
    run = _run_with_checkpoint(
        tmp_path,
        config,
        {"actor": actor.state_dict(), "obs_normalizer": {}},
    )

    result = export_onnx(run.run_dir, validation_seed=9, validation_samples=4)

    assert result.report.input_spec.shape == (None, 5)
    assert result.report.output_spec.shape == (None, 3)
    assert result.report.parity.max_abs_error < 1e-5
