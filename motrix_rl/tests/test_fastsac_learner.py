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
