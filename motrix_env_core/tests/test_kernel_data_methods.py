# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from pathlib import Path

import numba
import numpy as np
import pytest

from motrix_env_core.numba.kernel_data import (
    KernelDataLowering,
    KernelDataScope,
    SharedArray,
    flatten_kernel_data,
    iter_layout_leaves,
    kernel_data,
    rebuild_lowered,
)
from motrix_env_core.numba.manager.dispatch import dispatch


@kernel_data
class _MethodData:
    values: np.ndarray
    bias: SharedArray
    scale: np.float32

    @dispatch
    def total(self):
        return self.values.sum() * self.scale + self.bias[0]

    @dispatch
    def scaled(self):
        return self.values * self.scale

    @dispatch
    def combined(self, factor):
        return self.total() * factor

    def host_only(self):
        return self.values.sum()


@kernel_data
class _InheritedMethodData(_MethodData):
    extra: np.float32

    @dispatch
    def with_extra(self):
        return self.total() + self.extra


@kernel_data
class _OverriddenMethodData(_InheritedMethodData):
    @dispatch
    def total(self):
        return self.values.sum() * self.scale - self.bias[0]


def _lower_lane(value, env_id):
    """Use the same leaf scopes as manager lowering, retaining lane views."""
    leaves, tree_def = flatten_kernel_data(value)
    layout = KernelDataLowering().lower(tree_def, context="kernel data methods test")
    lane_leaves = tuple(
        leaves[leaf.slot_index][env_id] if leaf.scope is KernelDataScope.PER_ENV else leaves[leaf.slot_index]
        for leaf in iter_layout_leaves(layout)
    )
    return layout, rebuild_lowered(layout, lane_leaves)


def _read_total(value):
    # Kept at module scope so the subprocess test has a stable disk-cache locator.
    return value.total()


def _method_value(data_type=_MethodData):
    args = (
        np.asarray([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
        np.float32(2.0),
    )
    return data_type(*args) if data_type is _MethodData else data_type(*args, np.float32(3.0))


def test_marked_methods_compile_on_lowered_lanes_and_preserve_host_methods() -> None:
    value = _method_value()
    original_method = _MethodData.scaled
    layout, lane = _lower_lane(value, 1)
    assert layout.logical_type is _MethodData
    assert np.shares_memory(lane.values, value.values)
    assert lane.bias is value.bias

    read_total = numba.njit(_read_total)

    @numba.njit
    def scaled(kernel_value):
        return kernel_value.scaled()

    assert read_total(lane) == value.values[1].sum() * value.scale + value.bias[0]
    np.testing.assert_array_equal(scaled(lane), value.values[1] * value.scale)
    assert read_total.nopython_signatures
    assert scaled.nopython_signatures

    # Lowering must not replace host Python functions or their batched behavior.
    assert _MethodData.scaled is original_method
    assert not isinstance(_MethodData.scaled, numba.core.registry.CPUDispatcher)
    np.testing.assert_array_equal(value.scaled(), value.values * value.scale)
    assert value.total() == value.values.sum() * value.scale + value.bias[0]
    assert value.host_only() == value.values.sum()

    value.values[1, 0] = 7.0
    assert read_total(lane) == value.values[1].sum() * value.scale + value.bias[0]


@pytest.mark.parametrize("data_type", [_InheritedMethodData, _OverriddenMethodData])
def test_marked_methods_follow_inheritance_and_overrides(data_type) -> None:
    value = _method_value(data_type)
    _, lane = _lower_lane(value, 0)

    @numba.njit
    def read_methods(kernel_value):
        return kernel_value.total(), kernel_value.with_extra(), kernel_value.scaled()

    total, with_extra, scaled = read_methods(lane)
    expected = value.values[0].sum() * value.scale
    expected += -value.bias[0] if data_type is _OverriddenMethodData else value.bias[0]
    assert total == expected
    assert with_extra == expected + value.extra
    np.testing.assert_array_equal(scaled, value.values[0] * value.scale)
    assert read_methods.nopython_signatures


@pytest.mark.parametrize("data_type", [_MethodData, _OverriddenMethodData])
def test_marked_method_can_call_another_marked_method(data_type) -> None:
    value = _method_value(data_type)
    _, lane = _lower_lane(value, 1)

    @numba.njit
    def combined(kernel_value, factor):
        return kernel_value.combined(factor)

    expected = value.values[1].sum() * value.scale
    expected += -value.bias[0] if data_type is _OverriddenMethodData else value.bias[0]
    assert combined(lane, np.float32(4.0)) == expected * np.float32(4.0)
    assert combined.nopython_signatures


def test_unmarked_methods_remain_host_only() -> None:
    value = _method_value()
    _, lane = _lower_lane(value, 0)

    @numba.njit
    def host_only(kernel_value):
        return kernel_value.host_only()

    with pytest.raises(numba.TypingError, match="host_only"):
        host_only(lane)
    assert value.host_only() == value.values.sum()


def test_marked_method_cannot_call_an_unmarked_method() -> None:
    @kernel_data
    class _CallsHostMethod(_MethodData):
        @dispatch
        def call_host(self):
            return self.host_only()

    base = _method_value()
    value = _CallsHostMethod(base.values, base.bias, base.scale)
    _, lane = _lower_lane(value, 0)

    @numba.njit
    def call_host(kernel_value):
        return kernel_value.call_host()

    with pytest.raises(numba.TypingError, match="host_only"):
        call_host(lane)
    assert value.call_host() == value.values.sum()


def test_unmarked_override_does_not_inherit_a_marked_base_method() -> None:
    @kernel_data
    class _HostOverride(_MethodData):
        def total(self):
            return self.values.sum() - self.bias[0]

    base = _method_value()
    value = _HostOverride(base.values, base.bias, base.scale)
    _, lane = _lower_lane(value, 0)
    read_total = numba.njit(_read_total)
    with pytest.raises(numba.TypingError, match="total"):
        read_total(lane)
    assert value.total() == value.values.sum() - value.bias[0]


def test_marked_method_implementation_changes_layout_fingerprint(monkeypatch) -> None:
    @kernel_data
    class _MutableMethod:
        values: np.ndarray

        @dispatch
        def total(self):
            return self.values.sum() + 1.0

    value = _MutableMethod(np.asarray([[2.0, 3.0]], dtype=np.float32))
    leaves, tree_def = flatten_kernel_data(value)
    lowering = KernelDataLowering()
    first = lowering.lower(tree_def, context="before method replacement")
    first_lane = rebuild_lowered(first, (leaves[0][0],))

    @dispatch
    def replacement(self):
        return self.values.sum() + 2.0

    monkeypatch.setattr(_MutableMethod, "total", replacement)
    _, changed_tree = flatten_kernel_data(value)
    second = lowering.lower(changed_tree, context="after method replacement")
    second_lane = rebuild_lowered(second, (leaves[0][0],))
    assert first.fingerprint != second.fingerprint
    assert first.lowered_type is not second.lowered_type
    read_total = numba.njit(_read_total)
    assert read_total(first_lane) == 6.0
    assert read_total(second_lane) == 7.0
    assert value.total() == 7.0


def test_marked_methods_load_numba_disk_cache_in_a_second_process(tmp_path) -> None:
    # Import the real test module rather than duplicating its classes in a script:
    # dynamically generated proxies need a stable importable module for unpickling.
    script = """
import importlib
import json
import numba

module = importlib.import_module('test_kernel_data_methods')
value = module._method_value()
layout, lane = module._lower_lane(value, 1)
read_total = numba.njit(cache=True)(module._read_total)
result = read_total(lane)
print(json.dumps({
    'result': result,
    'fingerprint': layout.fingerprint,
    'hits': sum(read_total.stats.cache_hits.values()),
    'misses': sum(read_total.stats.cache_misses.values()),
}))
"""
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path(__file__).parent), env.get("PYTHONPATH"))))
    env["NUMBA_NUM_THREADS"] = "1"

    def run_process():
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return json.loads(completed.stdout.splitlines()[-1])

    cold = run_process()
    warm = run_process()
    assert cold["result"] == warm["result"] == 24.5
    assert cold["fingerprint"] == warm["fingerprint"]
    assert cold["misses"] == 1
    assert cold["hits"] == 0
    assert warm["misses"] == 0
    assert warm["hits"] == 1
