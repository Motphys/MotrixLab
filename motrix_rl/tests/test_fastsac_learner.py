# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""UTD governance tests for the async learner."""

from types import SimpleNamespace

from motrix_rl.fastsac.async_impl.learner import Learner


def _make_learner(utd_mode: str, num_updates: int) -> Learner:
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(cfg=SimpleNamespace(num_updates=num_updates))
    learner.async_options = SimpleNamespace(utd_mode=utd_mode)
    learner.control = SimpleNamespace(collector_steps=0)
    return learner


def test_strict_scales_num_updates_by_ingested_batches() -> None:
    learner = _make_learner("strict", 4)
    assert learner._num_updates_for(3) == 12
    assert learner._num_updates_for(0) == 0


def test_learner_bound_runs_full_batch() -> None:
    learner = _make_learner("learner_bound", 4)
    assert learner._num_updates_for(0) == 4
    assert learner._num_updates_for(2) == 4


def test_own_copies_compiled_outputs_out_of_the_graph_pool() -> None:
    """Regression: metrics read at log time came from an invalidated CUDA graph.

    `_update_pol` is gated by `policy_frequency`, so its outputs are carried
    across later loop iterations -- each of which calls
    `cudagraph_mark_step_begin()` and invalidates the generation they live in.
    Reading them afterwards raised rather than returning stale numbers:

        RuntimeError: Error: accessing tensor output of CUDAGraphs that has
        been overwritten by a subsequent run

    Needs a GPU: `reduce-overhead` compiles to CUDA graphs and is a no-op
    without one, so there is nothing to invalidate on CPU.
    """
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs require a GPU")

    from motrix_rl.fastsac.agent import _own

    @torch.compile(mode="reduce-overhead")
    def step(x):
        y = x @ x
        return y.sum(), y.mean()

    x = torch.randn(64, 64, device="cuda")

    torch.compiler.cudagraph_mark_step_begin()
    owned = _own(step(x))
    # Two further generations, as the gated loop would produce.
    for _ in range(2):
        torch.compiler.cudagraph_mark_step_begin()
        step(x)

    assert all(torch.isfinite(torch.as_tensor(float(v))) for v in owned)


def test_own_leaves_non_tensors_alone() -> None:
    from motrix_rl.fastsac.agent import _own

    assert _own((1, "a", None)) == (1, "a", None)


def _tiny_agent_cfg(**overrides):
    """Minimal valid FastSacAgentCfg namespace for a real (tiny) CUDA agent."""
    values = dict(
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        num_q_networks=2,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        alpha_learning_rate=1e-3,
        weight_decay=0.0,
        max_grad_norm=0.0,
        use_layer_norm=False,
        use_tanh=True,
        log_std_max=0.0,
        log_std_min=-5.0,
        num_atoms=5,
        v_min=-20.0,
        v_max=20.0,
        gamma=0.97,
        tau=0.125,
        alpha_init=0.001,
        use_autotune=True,
        target_entropy_ratio=0.0,
        buffer_size=64,
        num_steps=1,
        batch_size=8,
        learning_starts=1,
        policy_frequency=4,
        num_updates=4,
        obs_normalization=True,
        compile=True,
        amp=True,
        amp_dtype="bf16",
        device=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_update_metrics_survive_later_graph_generations() -> None:
    """Outputs carried across `cudagraph_mark_step_begin` must stay readable.

    Regression for the `_own` placement in ``FastSacAgent.update``: the actor
    pair is carried across policy-frequency gating (produced at a non-final
    iteration), and the main outputs are owned only at the final iteration.
    Both must remain readable after FURTHER update calls open new CUDA graph
    generations — reading graph-pool memory invalidated by a later replay
    raises instead of returning stale numbers.

    Needs a GPU: reduce-overhead compiles to CUDA graphs and is a no-op
    without one, so there is nothing to invalidate on CPU.
    """
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs require a GPU")

    from motrix_rl.fastsac.agent import FastSacAgent

    n_env, obs, cri, act = 8, 5, 7, 3
    agent = FastSacAgent(
        obs_dim=obs,
        critic_obs_dim=cri,
        act_dim=act,
        num_envs=n_env,
        cfg=_tiny_agent_cfg(),
        device=torch.device("cuda"),
    )
    for _ in range(40):
        agent.rb.extend(
            torch.randn(n_env, obs),
            torch.randn(n_env, cri),
            torch.randn(n_env, act),
            torch.randn(n_env),
            torch.zeros(n_env, dtype=torch.long),
            torch.zeros(n_env, dtype=torch.long),
        )

    # warmup/compile, then exercise both gating positions:
    agent.update(4)  # update_idx 0 -> 4: actor gated at i=0 (non-final, carried)
    first = agent.update(4)  # update_idx 4 -> 8: actor gated at i=0 (non-final)
    agent.update_idx = 5
    second = agent.update(4)  # update_idx 5 -> 9: actor gated at i=3 (final iteration)
    # Further generations have invalidated earlier graph pools; the returned
    # metrics must be owned copies and therefore still readable.
    agent.update(4)
    for metrics in (first, second):
        values = {key: float(value) for key, value in metrics.items()}
        assert all(value == value for value in values.values())  # no NaNs from torn reads


def test_nest_timing_path_merges_scalar_total_with_children_in_any_order() -> None:
    from motrix_rl.fastsac.async_impl.worker import _nest_timing_path

    # the collector emits a stage's scalar before its dotted sub-stages; the
    # rebuild must fold it into a "total" instead of nesting under a float
    tree: dict = {}
    _nest_timing_path(tree, ("physics",), 15.0)
    _nest_timing_path(tree, ("physics", "read"), 12.0)
    assert tree == {"physics": {"total": 15.0, "read": 12.0}}

    # reverse order must converge to the same tree
    tree = {}
    _nest_timing_path(tree, ("physics", "read"), 12.0)
    _nest_timing_path(tree, ("physics",), 15.0)
    assert tree == {"physics": {"total": 15.0, "read": 12.0}}

    # stages without sub-stages stay scalar
    tree = {}
    _nest_timing_path(tree, ("apply_action",), 1.5)
    assert tree == {"apply_action": 1.5}
