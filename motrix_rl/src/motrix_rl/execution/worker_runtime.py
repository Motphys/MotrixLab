# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small process-worker runtime helpers shared by trainers."""

from __future__ import annotations

import logging
import os
import random
import signal
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from motrix_rl.numa import set_cpu_affinity

logger = logging.getLogger(__name__)


def set_seed(seed: int | None) -> None:
    """Seed Python, NumPy and Torch RNGs without forcing determinism."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pin_worker_cpus(cpus: set[int]) -> None:
    """Best-effort pinning and Torch thread cap for a worker."""
    if not cpus:
        return
    try:
        set_cpu_affinity(cpus)
    except OSError as exc:
        logger.warning("CPU pinning to %s failed (%s); continuing unpinned", sorted(cpus), exc)
    torch.set_num_threads(max(len(cpus), 1))


def install_graceful_sigint(request_stop: Callable[[], object]) -> None:
    """Convert SIGINT into a cooperative stop request from the main thread."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("install_graceful_sigint must run in the main thread")

    def _handler(signum, frame):  # noqa: ARG001
        request_stop()

    signal.signal(signal.SIGINT, _handler)


@contextmanager
def inherit_log_stdio(log_file: Path):
    """Make spawned child fd 1/2 inherit its log file during process start.

    The parent must serialize this guard around ``Process.start()`` calls;
    redirecting process-global descriptors concurrently would mix worker logs.
    """
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(log_fd)


def configure_process_logging(log_file: Path) -> None:
    """Route Python and native worker output to a log file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=str(log_file),
        filemode="a",
        force=True,
    )
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
