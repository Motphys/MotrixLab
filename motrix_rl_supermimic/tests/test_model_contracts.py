# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Model precision isolation and execution-option boundary contracts."""

import pytest
import torch
from _helpers import make_toy_task
from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.networks import Policy, resolve_amp_dtype, run_mlp
from motrix_rl_supermimic.toy import ToyEnvCfg


def test_amp_modes_resolve_and_unknown_precision_is_rejected() -> None:
    assert resolve_amp_dtype("none") is None
    assert resolve_amp_dtype("bf16") == torch.bfloat16
    with pytest.raises(ValueError, match="Unknown AMP mode"):
        resolve_amp_dtype("fp16")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"amp": "fp16"}, "Unknown AMP mode"),
        ({"overlap_wm": True}, "overlap_wm requires a CUDA device"),
        ({"compile_actor_kernel": True}, "compile_actor_kernel requires cuda_graph_actor_loss"),
        ({"compile_wm_kernel": True}, "compile_wm_kernel requires cuda_graph_wm_loss"),
        ({"cuda_graph_actor_loss": True}, "require a CUDA device"),
        ({"cuda_graph_wm_loss": True}, "require a CUDA device"),
    ],
)
def test_learner_rejects_unavailable_execution_modes(tmp_path, options: dict, message: str) -> None:
    cfg = SuperMimicCfg(agent=AgentCfg(hidden=8), trainer=TrainerCfg(**options))
    with pytest.raises(ValueError, match=message):
        Learner(cfg, make_toy_task(), 2, 0.02, "cpu", tmp_path)


def test_learner_uses_task_goal_window_and_rejects_mismatched_checkpoint(tmp_path) -> None:
    cfg = SuperMimicCfg(agent=AgentCfg(hidden=8))
    task = make_toy_task(cfg=ToyEnvCfg(goal_frames=3))
    learner = Learner(cfg, task, 2, 0.02, "cpu", tmp_path)
    assert learner.mdp is task
    assert learner.goal_dim == task.observation_spec["goal"].space.shape[0]
    assert learner.pi.enc[0].in_features == learner.goal_dim
    learner.save(0.0)
    checkpoint = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    snapshot = torch.load(tmp_path / "policy_0000000.pt", map_location="cpu", weights_only=False)
    assert checkpoint["goal_frames"] == snapshot["goal_frames"] == task.goal_frames
    before = next(learner.pi.parameters()).detach().clone()
    checkpoint["goal_frames"] = 2
    with pytest.raises(ValueError, match="goal_frames"):
        learner.load_checkpoint(checkpoint)
    torch.testing.assert_close(next(learner.pi.parameters()), before)


def test_learner_world_model_uses_task_control_duration(tmp_path) -> None:
    task = make_toy_task(cfg=ToyEnvCfg(ctrl_dt=0.06))
    learner = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8)), task, 1, task.dt, "cpu", tmp_path)
    assert learner.wm.dt == task.dt == 0.06


def test_bf16_policy_on_cpu_preserves_fp32_actions_and_gradients() -> None:
    """CUDA-only AMP must not change CPU inference or its gradient path."""
    plain = Policy(4, 2, 1, hist=2, hidden=(8,))
    mixed = Policy(4, 2, 1, hist=2, hidden=(8,), amp_dtype=torch.bfloat16)
    mixed.load_state_dict(plain.state_dict())
    obs = torch.randn(3, 2, 4)
    goal = torch.randn(3, 2)
    expected = plain(obs, goal)
    actual = mixed(obs, goal)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for actual_parameter, expected_parameter in zip(mixed.parameters(), plain.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.integration
def test_amp_instances_do_not_change_each_others_cuda_precision() -> None:
    """Constructing another model cannot mutate an existing model's precision."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    plain = Policy(4, 2, 1, hist=2, hidden=(8,)).cuda()
    mixed = Policy(4, 2, 1, hist=2, hidden=(8,), amp_dtype=torch.bfloat16).cuda()
    mixed.load_state_dict(plain.state_dict())
    obs = torch.randn(3, 2, 4, device="cuda")
    goal = torch.randn(3, 2, device="cuda")
    expected = plain(obs, goal)
    mixed(obs, goal).sum().backward()
    torch.testing.assert_close(plain(obs, goal), expected, rtol=0, atol=0)
    x = plain.goal_norm(goal)
    torch.testing.assert_close(run_mlp(plain.enc, x), plain.enc(x), rtol=0, atol=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected_mixed = mixed.enc(x).float()
    torch.testing.assert_close(run_mlp(mixed.enc, x, mixed.amp_dtype), expected_mixed, rtol=0, atol=0)
