# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_env_core.perf import Perf, active_perf_scope, perf_root, perf_scope


class _Clock:
    def __init__(self, *values: int) -> None:
        self._values = iter(values)
        self.call_count = 0

    def __call__(self) -> int:
        self.call_count += 1
        return next(self._values)


def test_perf_aggregates_repeated_scopes_as_a_tree() -> None:
    clock = _Clock(0, 2, 5, 6, 10, 12)
    perf = Perf(enabled=True, clock=clock)

    with perf.scope("root"):
        with perf.scope("child"):
            pass
        with perf.scope("child"):
            pass

    root = perf.snapshot()[0]
    child = root.child("child")
    assert root.count == 1
    assert root.total_ns == 12
    assert root.self_ns == 5
    assert child.count == 2
    assert child.total_ns == 7
    assert child.self_ns == 7
    assert child.min_ns == 3
    assert child.max_ns == 4
    assert child.mean_ns == 3.5


def test_disabled_perf_does_not_read_clock_or_collect_data() -> None:
    clock = _Clock()
    perf = Perf(clock=clock)

    perf.begin("manual")
    perf.end()
    with perf.scope("context"):
        pass

    assert clock.call_count == 0
    assert perf.snapshot() == ()
    assert perf.scope("context") is perf.scope("another")


def test_scope_is_exception_safe_and_scope_objects_are_reused() -> None:
    clock = _Clock(0, 4, 5, 8)
    perf = Perf(enabled=True, clock=clock)
    scope = perf.scope("work")
    assert scope is perf.scope("work")

    with pytest.raises(ValueError, match="failed"):
        with scope:
            raise ValueError("failed")
    with scope:
        pass

    node = perf.snapshot()[0]
    assert node.count == 2
    assert node.total_ns == 7


def test_end_validates_scope_without_corrupting_stack() -> None:
    clock = _Clock(0, 3)
    perf = Perf(enabled=True, clock=clock)
    perf.begin("root")

    with pytest.raises(RuntimeError, match="scope mismatch"):
        perf.end("other")
    assert perf.end("root") == 3
    with pytest.raises(RuntimeError, match="without an active scope"):
        perf.end()


def test_active_scope_rejects_state_changes_and_snapshot() -> None:
    perf = Perf(enabled=True, clock=_Clock(0, 1))
    perf.begin("root")

    with pytest.raises(RuntimeError, match="change Perf.enabled"):
        perf.disable()
    with pytest.raises(RuntimeError, match="reset Perf"):
        perf.reset()
    with pytest.raises(RuntimeError, match="snapshot Perf"):
        perf.snapshot()

    perf.end()
    perf.reset()
    assert perf.snapshot() == ()


def test_decorators_route_nested_scopes_to_the_owning_perf() -> None:
    class _Owner:
        def __init__(self, clock: _Clock) -> None:
            self.perf = Perf(enabled=True, clock=clock)

        @perf_scope("inner")
        def inner(self) -> None:
            pass

        @perf_root("outer")
        def outer(self) -> None:
            self.inner()

    first = _Owner(_Clock(0, 1, 3, 5))
    second = _Owner(_Clock(10, 11, 14, 16))
    first.outer()
    second.outer()

    first_outer = first.perf.snapshot()[0]
    second_outer = second.perf.snapshot()[0]
    assert first_outer.total_ns == 5
    assert first_outer.child("inner").total_ns == 2
    assert second_outer.total_ns == 6
    assert second_outer.child("inner").total_ns == 3


def test_active_perf_scope_contributes_to_activated_profiler() -> None:
    perf = Perf(enabled=True, clock=_Clock(0, 2, 5, 8))

    with perf.activate(), perf.scope("root"):
        with active_perf_scope("backend"):
            pass

    root = perf.snapshot()[0]
    assert root.total_ns == 8
    assert root.child("backend").total_ns == 3


def test_active_perf_scope_is_disabled_without_an_activated_profiler() -> None:
    with active_perf_scope("backend"):
        pass


def test_stage_mean_ms_flattens_nested_scopes_to_dotted_paths() -> None:
    # two step calls; each scope entry/exit consumes one clock tick (8 per
    # iteration). physics runs 3ms per call, transition 9ms with a 2ms read.
    base = (0, 2_000_000, 5_000_000, 6_000_000, 10_000_000, 12_000_000, 15_000_000, 26_000_000)
    ticks = base + tuple(t + 26_000_000 for t in base)
    perf = Perf(enabled=True, clock=_Clock(*ticks))
    for _ in range(2):
        with perf.scope("step"):
            with perf.scope("physics"):
                pass
            with perf.scope("transition"):
                with perf.scope("read"):
                    pass

    means = perf.stage_mean_ms("step")

    assert means["physics"] == pytest.approx(3.0)
    assert means["transition"] == pytest.approx(9.0)
    assert means["transition.read"] == pytest.approx(2.0)
    assert "step" not in means  # the root itself is not part of the paths


def test_stage_mean_ms_returns_empty_for_unrun_root() -> None:
    assert Perf(enabled=True).stage_mean_ms("step") == {}
