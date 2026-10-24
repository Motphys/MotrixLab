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

    def cancel_join_thread(self) -> None:
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
        self._algorithm_target, self._runtime, self._algorithm_args, self._algorithm_kwargs = self._unwrap()

    def _unwrap(self):
        """Expose the algorithm entry shape while retaining raw worker_entry args."""
        from motrix_rl.execution.workers import worker_entry

        if self.target is not worker_entry:
            return self.target, None, self.args, self.kwargs
        runtime, algorithm_target, algorithm_args, algorithm_kwargs, _settings = self.args
        return algorithm_target, runtime, tuple(algorithm_args), dict(algorithm_kwargs)

    @property
    def arguments(self) -> dict:
        """The algorithm entry point's bound arguments, including ``runtime``."""
        return dict(
            inspect.signature(self._algorithm_target)
            .bind(self._runtime, *self._algorithm_args, **self._algorithm_kwargs)
            .arguments
        )

    @property
    def runtime(self):
        return self._runtime

    def _run(self) -> None:
        from motrix_rl.execution.contracts import ExecutionCancelled

        try:
            self.exitcode = self._script(self)
        except ExecutionCancelled:
            if self.runtime is None or not self.runtime.stop_event.is_set():
                self.exitcode = 1
                raise
            self.exitcode = 0
        except BaseException:
            self.exitcode = 1
            raise

    @property
    def pid(self) -> int | None:
        return self._thread.ident

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
        from motrix_rl.execution.workers import worker_entry

        script = self.scripts.get(target)
        if script is None and target is worker_entry:
            script = self.scripts[args[1]]
        process = ThreadProcess(script, target, args, kwargs, name)
        self.processes.append(process)
        return process
