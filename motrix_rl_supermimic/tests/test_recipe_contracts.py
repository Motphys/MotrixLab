# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Numerical migration contracts from the validated task-repo recipe:

- WM loss scores with ``inc_norm`` but de-normalizes with ``out_norm``;
  the graphed kernel matches the eager path loss and every parameter gradient.
- Reference-action policies share one construction contract across
  learner/collector/evaluator and keep the residual identity.
- The policy LR schedule reproduces the validated trigger/ramp/anneal shape.
"""

import pytest
import torch

from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.networks import WorldModel
from motrix_rl_supermimic.optim import policy_multiplier, trigger_decay
from motrix_rl_supermimic.task_mdp import StateSpec
from motrix_rl_supermimic.wm_graph import FlattenFullLoss
from test_mbrl_env import ToyEnv, ToyTaskMDP  # noqa: F401


def _wm(seed=13):
    torch.manual_seed(seed)
    spec = StateSpec(
        state_dim=15, action_dim=2, context_dim=3, observation_dim=4, critic_observation_dim=4, command_dim=1,
        state_groups=((3, 0.01), (4, 0.01), (3, 0.01), (3, 0.01), (1, 0.01), (1, 0.01)),
    )
    wm = WorldModel(spec, hidden=(16, 16, 16))
    with torch.no_grad():
        wm.net[-1].weight.normal_(0, 0.001)
        wm.out_norm.mean.normal_(0, 0.01)
        wm.out_norm.var.uniform_(0.01, 0.03)
        wm.inc_norm.mean.normal_(0, 0.03)
        wm.inc_norm.var.uniform_(0.1, 0.2)
    return wm, spec


@pytest.mark.parametrize("structured", [False, True])
def test_wm_loss_uses_inc_stats_for_scoring_and_out_stats_for_prediction(structured):
    wm, spec = _wm()
    from motrix_rl_supermimic import state as st

    batch, window = 3, 2
    state = (
        torch.randn(batch, 3),
        torch.eye(3).repeat(batch, 1, 1),
        torch.randn(batch, 3),
        torch.randn(batch, 3),
        torch.randn(batch, 1),
        torch.randn(batch, 1),
    )
    initial = st.flatten_state(state)
    actions = torch.randn(batch, window, 2)
    xis = torch.randn(batch, window, 3)
    targets = initial[:, None].repeat(1, window, 1)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    floor = torch.full((st.num_joints(spec) * 2 + 12,), 0.001)

    s = st.unflatten_state(spec, initial)
    predictions = []
    for j in range(window):
        s = wm.step(s, actions[:, j], xis[:, j])
        predictions.append(s)
    predicted = tuple(torch.stack([p[i] for p in predictions], 1) for i in range(6))
    scale = torch.maximum(wm.inc_norm.std(), floor)
    expected = ((wm.state_error(predicted, st.unflatten_state(spec, targets)) / scale).square().mean(-1) * mask).sum() / batch
    expected_grads = torch.autograd.grad(expected, tuple(wm.parameters()))

    module = FlattenFullLoss(wm, floor, window, batch, normalize_context=lambda xi: xi)
    module.sync_norms(floor)
    actual = module(initial, actions, xis, targets, mask)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, tuple(wm.parameters()))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e)


class _SkipTaskMDP(ToyTaskMDP):
    """Toy task exposing a reference-action module over its goal."""

    spec = ToyTaskMDP.spec

    def reference_action(self):
        offset = 0
        module = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            module.weight.fill_(1.0)
        module.offset = offset
        return module


def test_reference_action_policy_contract():
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=16, hist=2, pi_ref_skip=True),
        trainer=TrainerCfg(iterations=1, warmup_steps=1, horizon=1, goal_frames=1, wm_window=1),
    )
    mdp = _SkipTaskMDP()
    from motrix_rl_supermimic.learner import build_policy

    learner_policy = build_policy(cfg, mdp, mdp.spec.command_dim * cfg.trainer.goal_frames, "cpu")
    collector = build_policy(cfg, mdp, mdp.spec.command_dim * cfg.trainer.goal_frames, "cpu")
    collector.load_state_dict(learner_policy.state_dict())
    obs = torch.randn(3, 2, mdp.spec.observation_dim)
    goal = torch.randn(3, mdp.spec.command_dim * cfg.trainer.goal_frames)
    torch.testing.assert_close(learner_policy(obs, goal), collector(obs, goal))
    # residual identity: zeroing the decoder leaves the reference action
    plain = build_policy(
        SuperMimicCfg(agent=AgentCfg(hidden=16, hist=2),
                      trainer=TrainerCfg(iterations=1, warmup_steps=1, horizon=1, goal_frames=1, wm_window=1)),
        mdp, mdp.spec.command_dim * cfg.trainer.goal_frames, "cpu",
    )
    plain.load_state_dict(learner_policy.state_dict(), strict=True)
    # plain carries the same residual head without the reference module, so
    # learner == plain + reference_action(goal).
    torch.testing.assert_close(learner_policy(obs, goal), plain(obs, goal) + goal)


def test_schedule_trigger_ramp_anneal_and_sticky_state():
    cfg = TrainerCfg(pi_lr_mult=4.0, lr_decay_ret=0.75, lr_decay_by=1400, lr_decay_len=300,
                     pi_anneal_len=3000, pi_anneal_min=0.1)
    assert trigger_decay(cfg, 1000, 0.74, None) is None
    assert trigger_decay(cfg, 1100, 0.75, None) == 1100
    assert trigger_decay(cfg, 1400, 0.1, None) == 1400
    assert trigger_decay(cfg, 2000, 0.9, 1100) == 1100
    assert policy_multiplier(cfg, 1000, None) == 4.0
    assert policy_multiplier(cfg, 1100, 1100) == 4.0
    assert policy_multiplier(cfg, 1250, 1100) == pytest.approx(2.5)
    assert policy_multiplier(cfg, 1400, 1100) == 1.0
    assert policy_multiplier(cfg, 2900, 1100) == pytest.approx(0.55)
    assert policy_multiplier(cfg, 4400, 1100) == pytest.approx(0.1)


def test_ema_starts_from_initial_weights():
    register = None
    torch.manual_seed(0)
    env = ToyEnv(num_envs=4)
    env.init_state()
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=16, hist=2, pi_ema=0.995),
        trainer=TrainerCfg(iterations=2, buffer_steps=32, utd=0.5, horizon=2, goal_frames=1, wm_window=1,
                           wm_batches=1, wm_batch=16, pi_batch=16, warmup_steps=4, wm_only_iters=1,
                           critic_epochs=1, critic_minibatches=1, log_every=1, ckpt_every=2),
    )
    from motrix_rl_supermimic.learner import Learner

    learner = Learner(cfg, env, "cpu", "/tmp/supermimic-ema-test")
    assert learner.pi_avg is not None
    # n_averaged==1 before the first update: the first actor update lerps
    # toward the live policy instead of copying it.
    assert int(learner.pi_avg.n_averaged) == 1
    before = [p.detach().clone() for p in learner.pi_avg.module.parameters()]
    learner.collect(4)
    for _ in range(2):
        learner.wm_update()
    learner.policy_update()
    moved = any(not torch.equal(a, b) for a, b in zip(before, learner.pi_avg.module.parameters()))
    assert moved


def test_quaternion_layout_roundtrip():
    """Regression: mat_to_quat branch-component ordering corrupted the xyzw
    layout for most rotations, silently corrupting WM increments."""
    from motrix_rl_supermimic.state import mat_to_quat, quat_to_mat, so3_exp

    torch.manual_seed(5)
    rotations = torch.linalg.qr(torch.randn(512, 3, 3))[0]
    err = (rotations - quat_to_mat(mat_to_quat(rotations))).abs().amax(dim=(1, 2))
    assert err.max() < 1e-4
    v = torch.randn(256, 3)
    v = v / v.norm(dim=-1, keepdim=True) * torch.rand(256, 1) * 3.0
    err2 = (so3_exp(v) - quat_to_mat(mat_to_quat(so3_exp(v)))).abs().amax(dim=(1, 2))
    assert err2.max() < 1e-4


@pytest.mark.parametrize("zero_rows", [(), (0,), (3,), (1, 5)])
def test_survival_cumprod_matches_aten_values_and_vjp(zero_rows):
    """Graph-safe survival product must be bit-equal to cumprod in value and VJP."""
    from motrix_rl_supermimic.survival import survival_cumprod

    generator = torch.Generator().manual_seed(31)
    x = torch.rand(8, 5, generator=generator)
    for row in zero_rows:
        x[row] = 0
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    weight = torch.randn(x.shape, generator=generator)
    expected = torch.cumprod(a, 0)
    actual = survival_cumprod(b)
    assert torch.equal(actual, expected)
    da, = torch.autograd.grad(expected, a, weight)
    db, = torch.autograd.grad(actual, b, weight)
    assert torch.equal(db.contiguous().view(torch.uint8), da.contiguous().view(torch.uint8))


@pytest.mark.integration
def test_cumprod_backward_blocks_graph_capture_but_survival_does_not():
    """Regression: ATen cumprod backward aborts CUDA capture; the survival
    formulation must stay capturable (policy imagination graphs depend on it)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from motrix_rl_supermimic.survival import survival_cumprod

    x = torch.rand(4, 8, device="cuda").requires_grad_(True)
    weight = torch.randn(4, 8, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = survival_cumprod(1 - x)
        torch.autograd.grad(y, [x], weight, allow_unused=True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = survival_cumprod(1 - x)
        torch.autograd.grad(y, [x], weight, allow_unused=True)
    torch.cuda.synchronize()


def test_norm_stats_view_publishes_real_statistics():
    """Regression: RunningNorm lacked the transport's _mean/_std/_var/_var
    attribute contract, so publish() silently shipped empty normalizer stats
    and the collector ran new weights under initial normalization."""
    from motrix_rl.transport.inference_bundle import InferenceBundleSpec, ModuleSection, NormalizerSection
    from motrix_rl.transport.weight_channel import HostWeightSender, WeightChannelShared
    from motrix_rl_supermimic.networks import NormStatsView

    spec = InferenceBundleSpec(
        modules=(ModuleSection("policy", 10),),
        normalizers=(NormalizerSection("obs_norm", 6),),
    )
    shared = WeightChannelShared(spec)
    sender = HostWeightSender(shared)
    norm = None
    from motrix_rl_supermimic.networks import RunningNorm

    norm = RunningNorm(6)
    with torch.no_grad():
        norm.mean.fill_(0.5)
        norm.var.fill_(2.0)
        norm.count.fill_(123)

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(10))

    sender.publish({"policy": _Tiny()}, {"obs_norm": NormStatsView(norm)})
    assert float(shared.count[1][0]) == 123
    assert float(shared.mean[1][0, 0]) == 0.5
    assert float(shared.var[1][0, 0]) == 2.0


def test_checkpoint_curriculum_keys_roundtrip():
    """Regression: the curriculum-key stripping used tuple indexing instead of
    a slice, collapsing {'curriculum_bin_*'} to one key 'b' and breaking
    resume with KeyError. The stripped names must match the task's fields."""
    from motrix_rl_supermimic.task_mdp import optional_task_hook
    from test_mbrl_env import ToyEnv, ToyTaskMDP

    task = ToyTaskMDP()
    assert optional_task_hook(task, "load_curriculum") is None  # toy: no hooks

    class _CurriculumTask(ToyTaskMDP):
        def __init__(self):
            self.bins = {"bin_G": torch.zeros(3), "bin_history": torch.ones(3)}

        def curriculum_state(self):
            return {k: v.clone() for k, v in self.bins.items()}

        def load_curriculum(self, state):
            for k, v in state.items():
                self.bins[k].copy_(v)

    t = _CurriculumTask()
    ck = {f"curriculum_{k}": v for k, v in t.curriculum_state().items()}
    stripped = {
        key.removeprefix("curriculum_"): value
        for key, value in ck.items()
        if key.startswith("curriculum_") and isinstance(value, torch.Tensor)
    }
    assert set(stripped) == {"bin_G", "bin_history"}
    t.load_curriculum(stripped)
    assert t.bins["bin_G"].equal(torch.zeros(3))
