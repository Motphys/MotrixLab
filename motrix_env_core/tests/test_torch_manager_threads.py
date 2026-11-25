# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager thread caps work with SMT defaults and explicit caller settings."""

import pytest

import motrix_env_core.torch_manager.runtime as runtime


@pytest.mark.parametrize(
    "num_envs,current,expected",
    [(2048, 96, 8), (4096, 96, 16), (8192, 96, 16), (2048, 4, 4), (2048, 8, 8)],
)
def test_default_cap_does_not_mistake_physical_cores_for_a_caller_override(
    monkeypatch: pytest.MonkeyPatch, num_envs: int, current: int, expected: int
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 192)
    monkeypatch.setattr(runtime.torch, "get_num_threads", lambda: current)
    updates = []
    monkeypatch.setattr(runtime.torch, "set_num_threads", updates.append)

    runtime._apply_default_thread_cap(num_envs)

    assert updates == ([expected] if expected < current else [])
    monkeypatch.setattr(runtime.torch, "get_num_threads", lambda: updates[-1] if updates else current)
    monkeypatch.delenv("MOTRIX_TORCH_COMPILE_THREADS", raising=False)
    assert runtime._compile_options()["cpp.threads"] == expected


def test_explicit_omp_threads_bypasses_the_default_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    monkeypatch.setattr(runtime.torch, "get_num_threads", lambda: 32)
    updates = []
    monkeypatch.setattr(runtime.torch, "set_num_threads", updates.append)

    runtime._apply_default_thread_cap(2048)

    assert updates == []
