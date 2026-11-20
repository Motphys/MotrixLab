# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Actor capture contracts: live policy weights, private norms and replay gradients."""

import copy

import pytest
import torch
from _helpers import make_toy_task
from motrix_rl_supermimic.actor_graph import ActorLoss, ActorLossGraph, ActorModels
from motrix_rl_supermimic.config import TrainerCfg
from motrix_rl_supermimic.imagination import ImaginationRuntime
from motrix_rl_supermimic.manager_task import ManagerTaskMDP
from motrix_rl_supermimic.networks import Critic, Policy, WorldModel
from motrix_rl_supermimic.toy import ToyEnvCfg


def actor_inputs(
    device: str,
) -> tuple[Policy, ActorModels, ManagerTaskMDP, ImaginationRuntime, TrainerCfg, tuple[torch.Tensor, ...]]:
    mdp, mdp_binding = make_toy_task(device, cfg=ToyEnvCfg(goal_frames=2))
    cfg = TrainerCfg(horizon=2, pi_batch=3)
    pi = Policy(
        mdp.observation_dim, mdp.goal_dim, mdp.action_dim, hist=mdp.observation_spec.policy.history_length, hidden=(8,)
    ).to(device)
    wm = WorldModel(mdp.world_state, mdp.world_context.size, hidden=(8,), action_dim=mdp.action_dim).to(device)
    critic = Critic(mdp.critic_observation_dim + mdp.goal_dim, hidden=(8,)).to(device)
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    state = torch.zeros(3, mdp.world_state.size, device=device)
    xi = torch.zeros(3, 0, device=device)
    frame = torch.zeros(3, dtype=torch.long, device=device)
    previous = torch.zeros(3, 1, device=device)
    obs = (
        mdp.observation(mdp_binding.bind(frame, state=state, action=previous, previous=previous, context=xi))[:, None]
        .expand(-1, 2, -1)
        .contiguous()
    )
    goal = mdp.goal(mdp_binding.bind_reference(frame))
    return pi, ActorModels(wm, critic), mdp, mdp_binding, cfg, (state, xi, frame, previous, obs, goal)


def test_actor_views_share_weights_but_refresh_replaced_normalizer_buffers():
    pi, models, mdp, mdp_binding, cfg, args = actor_inputs("cpu")
    module = ActorLoss(pi, models, mdp, mdp_binding, cfg)
    assert list(module.parameters()) == list(pi.parameters())
    for actual, borrowed in ((models.wm, module.models.wm), (models.critic, module.models.critic)):
        assert all(a is b for a, b in zip(actual.parameters(), borrowed.parameters()))

    pairs = (
        (pi.obs_norm, module.pi.obs_norm),
        (pi.goal_norm, module.pi.goal_norm),
        (models.wm.in_norm, module.models.wm.in_norm),
        (models.wm.out_norm, module.models.wm.out_norm),
        (models.wm.inc_norm, module.models.wm.inc_norm),
        (models.critic.norm, module.models.critic.norm),
    )
    addresses = [snapshot.var.data_ptr() for _, snapshot in pairs]
    for actual, snapshot in pairs:
        actual.update(torch.randn(8, actual.mean.numel()))
        assert actual.var.data_ptr() != snapshot.var.data_ptr()
    module.sync_norms()
    for (actual, snapshot), address in zip(pairs, addresses):
        torch.testing.assert_close(snapshot.mean, actual.mean)
        torch.testing.assert_close(snapshot.var, actual.var)
        assert snapshot.var.data_ptr() == address

    module(*args)[0].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pi.parameters())
    assert all(p.grad is None for p in models.wm.parameters())
    assert all(p.grad is None for p in models.critic.parameters())


def test_batched_goal_encoding_matches_per_step_loss_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    pi, models, mdp, mdp_binding, cfg, args = actor_inputs("cpu")
    direct = ActorLoss(pi, models, mdp, mdp_binding, cfg)
    encode_goal = direct.pi.encode_goal

    def encode_per_step(goals: torch.Tensor) -> torch.Tensor:
        return torch.stack([encode_goal(step_goals) for step_goals in goals])

    monkeypatch.setattr(direct.pi, "encode_goal", encode_per_step)
    batched_pi = copy.deepcopy(pi)
    batched = ActorLoss(batched_pi, models, mdp, mdp_binding, cfg)
    rng = torch.get_rng_state()
    expected = direct(*args)
    assert len(expected) == 7
    assert expected[1].shape == (cfg.horizon * cfg.pi_batch, mdp.world_state.size)
    torch.set_rng_state(rng)
    actual = batched(*args)
    torch.testing.assert_close(actual[0], expected[0])
    expected[0].backward()
    actual[0].backward()
    for parameter, reference in zip(batched_pi.parameters(), pi.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference.grad)


def test_actor_loss_uses_configured_reward_term_and_preserves_action_gradients() -> None:
    from motrix_env_core.manager.context import ManagerContext
    from motrix_env_core.manager.resolver import ManagerTermResolver
    from motrix_env_core.manager.rewards import RewardTerm

    pi, models, mdp, mdp_binding, cfg, _ = actor_inputs("cpu")

    def reward(ctx: ManagerContext) -> torch.Tensor:
        return ctx.actions["velocity"].current().sum(-1) + 3

    mdp.reward_call = ManagerTermResolver().resolve_reward("configured", 0, RewardTerm(reward))
    module = ActorLoss(pi, models, mdp, mdp_binding, cfg)
    state = torch.zeros(3, mdp.world_state.size)
    frame = torch.zeros(3, dtype=torch.long)
    action = torch.ones(3, mdp.action_dim, requires_grad=True)
    rewards, _ = module.reward_termination(state, frame, action, torch.zeros_like(action), torch.empty(3, 0))
    scale = mdp.reward_weight * mdp.dt
    torch.testing.assert_close(rewards, torch.full_like(rewards, 4 * scale))
    torch.testing.assert_close(torch.autograd.grad(rewards.sum(), action)[0], torch.full_like(action, scale))


@pytest.mark.integration
def test_actor_graph_matches_direct_loss_and_gradients():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(37)
    pi, models, mdp, mdp_binding, cfg, args = actor_inputs("cuda")
    direct_pi = copy.deepcopy(pi)
    direct = ActorLoss(direct_pi, models, mdp, mdp_binding, cfg)
    graph = ActorLossGraph(pi, models, mdp, mdp_binding, cfg)
    # Repeated calls include capture, replay, live-weight and norm updates.
    for _ in range(2):
        observations = torch.randn(8, 4, device="cuda")
        pi.obs_norm.update(observations)
        direct_pi.obs_norm.update(observations)
        direct.sync_norms()
        torch.cuda.manual_seed(51)
        expected = direct(*args)
        expected[0].backward()
        gradients = [p.grad.clone() for p in direct_pi.parameters()]
        direct_pi.zero_grad(set_to_none=True)
        torch.cuda.manual_seed(51)
        actual = graph(*args)
        torch.testing.assert_close(actual.loss, expected[0])
        torch.testing.assert_close(actual.rewards, expected[-3])
        torch.testing.assert_close(actual.terminations, expected[-2])
        actual.loss.backward()
        for parameter, gradient in zip(pi.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient)
        with torch.no_grad():
            for parameter, reference in zip(pi.parameters(), direct_pi.parameters()):
                parameter.add_(parameter.grad, alpha=-0.01)
                reference.copy_(parameter)
        pi.zero_grad(set_to_none=True)

    with pytest.raises(RuntimeError, match="signature changed"):
        graph(*(x[:1] for x in args))
