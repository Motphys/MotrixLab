# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared white-box mocks for the async FastSAC trainer's process-free tests.

``make_mock_learner`` builds a ``Learner`` without running ``__init__`` (no
agent build, no CUDA): the drain path only needs the rings, a capturing
replay-buffer stub and the generation assembler. Used by
test_fastsac_async_multi.py and test_fastsac_pipeline_equivalence.py.

``ThreadSpawnContext`` stands in for the ``spawn`` multiprocessing context of
the parent trainer: workers become threads running a test-provided script,
so the parent's boot barrier, monitor loop and panel run unchanged. Used by
test_fastsac_live_metrics.py.
"""

from __future__ import annotations

import inspect
import queue
import threading
from collections.abc import Callable
from types import SimpleNamespace

import torch
from motrix_rl_builtin.fastsac.async_impl.learner import GenerationAssembler, Learner
from motrix_rl_builtin.fastsac.async_impl.transport import Control


def make_mock_learner(rings: list, extends: list | None) -> Learner:
    """A drain-capable Learner over ``rings``; ``extends`` captures extend_batch columns."""
    learner = Learner.__new__(Learner)
    learner.agent = SimpleNamespace(
        device=torch.device("cpu"),
        rb=SimpleNamespace(ptr=0, extend_batch=lambda *c: extends.append([x.clone() for x in c])),
        cfg=SimpleNamespace(learning_starts=0),
    )
    learner.async_options = SimpleNamespace(max_ingest_per_iter=8)
    learner.control = Control(num_collectors=len(rings))
    learner.rings = rings
    learner.weights = []
    learner._pending = GenerationAssembler(num_rings=len(rings))
    learner._staging = None
    learner._pending_copy = False
    return learner


class ThreadQueue(queue.Queue):
    """``queue.Queue`` with the ``close()`` of a multiprocessing queue."""

    def close(self) -> None:
        pass


class ThreadProcess:
    """A worker "process" that runs ``script(self)`` on a thread; its return value is the exit code."""

    def __init__(self, script: Callable[[ThreadProcess], int], target, args=(), kwargs=None, name=None) -> None:
        self.target = target
        self.args = tuple(args)
        self.kwargs = dict(kwargs or {})
        self.name = name
        self.exitcode: int | None = None
        self._script = script
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    @property
    def arguments(self) -> dict:
        """The worker entry point's arguments by parameter name."""
        return inspect.signature(self.target).bind(*self.args, **self.kwargs).arguments

    def _run(self) -> None:
        self.exitcode = self._script(self)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class ThreadSpawnContext:
    """Thread-backed stand-in for ``mp.get_context("spawn")``.

    ``scripts`` maps the worker entry point (``run_learner_process`` /
    ``run_collector_process``) to the script its threads run.
    """

    def __init__(self, scripts: dict[Callable, Callable[[ThreadProcess], int]]) -> None:
        self.scripts = scripts
        self.processes: list[ThreadProcess] = []

    def Queue(self, maxsize: int = 0) -> ThreadQueue:  # noqa: N802 - mirrors the multiprocessing API
        return ThreadQueue(maxsize)

    def Event(self) -> threading.Event:  # noqa: N802 - mirrors the multiprocessing API
        return threading.Event()

    def Process(self, target, args=(), kwargs=None, name=None) -> ThreadProcess:  # noqa: N802
        process = ThreadProcess(self.scripts[target], target, args, kwargs, name)
        self.processes.append(process)
        return process
