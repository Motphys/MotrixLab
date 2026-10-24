# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared Torch worker settings preserve explicit caller-owned policies."""

import random
import signal
from threading import Thread

import numpy as np
import pytest
import torch

from motrix_rl import numa
from motrix_rl.execution import worker_runtime


def test_seed_none_is_noop_and_explicit_seed_reproduces_all_rngs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        worker_runtime.set_seed(None)
        assert random.getstate() == python_state
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert torch.equal(torch.get_rng_state(), torch_state)
        worker_runtime.set_seed(19)
        values = random.random(), np.random.rand(), torch.rand(1)
        worker_runtime.set_seed(19)
        assert random.random() == values[0]
        assert np.random.rand() == values[1]
        assert torch.equal(torch.rand(1), values[2])
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def test_cpu_spec_clamps_huge_ranges_to_sparse_inherited_affinity(monkeypatch):
    monkeypatch.setattr(numa, "available_cpu_ids", lambda: {2, 5, 8})
    assert numa.resolve_cpu_set("0:1000000000", "worker_cpus") == {2, 5, 8}
    assert numa.resolve_cpu_set("3:7", "worker_cpus") == {5}
    assert numa.resolve_cpu_set("0,1", "worker_cpus") == set()
    assert numa.resolve_cpu_set(None, "worker_cpus") == set()
    with pytest.raises(ValueError, match="worker_cpus"):
        numa.resolve_cpu_set("8:2", "worker_cpus")


def test_thread_budget_still_applies_if_os_affinity_fails(monkeypatch):
    threads = []

    def fail(_cpus):
        raise OSError("restricted")

    monkeypatch.setattr(worker_runtime, "set_cpu_affinity", fail)
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    worker_runtime.pin_worker_cpus({2, 5})
    worker_runtime.pin_worker_cpus(set())
    assert threads == [2]


def test_sigint_handler_only_requests_stop_and_requires_main_thread(monkeypatch):
    installed, stopped, errors = [], [], []
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.append((signum, handler)))
    worker_runtime.install_graceful_sigint(lambda: stopped.append(True))
    assert installed[0][0] == signal.SIGINT
    installed[0][1](signal.SIGINT, None)
    assert stopped == [True]

    def install_in_thread():
        try:
            worker_runtime.install_graceful_sigint(lambda: None)
        except RuntimeError as error:
            errors.append(error)

    thread = Thread(target=install_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive() and len(errors) == 1
    assert len(installed) == 1
