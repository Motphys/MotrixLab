# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Live training metrics in the ``motrix.metrics/v1`` protocol.

Setting ``MOTRIX_METRICS_URL`` makes a training run push its records to an
ingest service compatible with ``motrix.metrics/v1``; setting
``MOTRIX_METRICS_JSONL=1`` also (or instead) appends them to
``run_dir/metrics.jsonl``. With neither the logger is disabled: no file, no
thread. Both sinks write from daemon threads, so neither a slow service nor a
slow disk blocks or fails training. See ``wiki/design/live-metrics.md``.
"""

from __future__ import annotations

import atexit
import collections
import gzip
import json
import logging
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections.abc import Callable, Mapping
from http.client import HTTPException
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA = "motrix.metrics/v1"
JSONL_FILENAME = "metrics.jsonl"

_TRUTHY = {"1", "true", "yes", "on"}

_RUN_ID_MAX = 200
_RUN_ID_INVALID = re.compile(r"[^A-Za-z0-9._-]")

# How long the interpreter waits at exit for sinks still writing or sending (see _join_at_exit).
EXIT_TIMEOUT = 5.0
_OPEN: weakref.WeakSet[MetricsLogger] = weakref.WeakSet()


class MetricsLogger:
    """Emits ``motrix.metrics/v1`` records to a jsonl file and an ingest service.

    Every record gets one ``seq`` and goes to every sink, so a jsonl written
    next to the service can later backfill it record for record. Holds a thread and locks: it cannot
    be pickled into worker processes.
    """

    def __init__(
        self,
        run_id: str,
        *,
        role: str = "trainer",
        jsonl: JsonlSink | None = None,
        http: HttpSink | None = None,
        project: str | None = None,
        group: str | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.writer = f"{socket.gethostname()}:{os.getpid()}:{role}"
        self._jsonl = jsonl
        self._http = http
        self._identity = {"project": project, "group": group, "meta": dict(meta or {})}
        # Starts at the wall clock rather than 0: a restarted container can reuse
        # hostname and pid, and the service drops records by (writer, seq).
        self._seq = time.time_ns() // 1000
        self._seq_lock = threading.Lock()
        self._emit_lock = threading.Lock()
        self._nonfinite_keys: set[str] = set()
        self._invalid_keys: set[str] = set()
        self._closed = False
        if http is not None:
            http.start(lambda: self._record("heartbeat", {}))
        if self.enabled:
            _OPEN.add(self)

    @classmethod
    def disabled(cls) -> MetricsLogger:
        """A logger whose methods do nothing."""
        return cls("disabled")

    @classmethod
    def from_env(cls, run_dir: str | Path, *, role: str = "trainer") -> MetricsLogger:
        """Build the logger the ``MOTRIX_METRICS_*`` environment asks for."""
        url = os.environ.get("MOTRIX_METRICS_URL", "")
        jsonl = os.environ.get("MOTRIX_METRICS_JSONL", "").strip().lower() in _TRUTHY
        if not url and not jsonl:
            return cls.disabled()
        run_dir = Path(run_dir)
        meta: dict[str, Any] = {}
        raw_meta = os.environ.get("MOTRIX_METRICS_META", "")
        if raw_meta:
            try:
                meta = dict(json.loads(raw_meta))
            except (ValueError, TypeError):
                logger.warning("metrics: ignoring MOTRIX_METRICS_META, not a JSON object")
        return cls(
            sanitize_run_id(os.environ.get("MOTRIX_METRICS_RUN_ID") or "__".join(run_dir.parts[-5:])),
            role=role,
            jsonl=JsonlSink(run_dir / JSONL_FILENAME) if jsonl else None,
            http=HttpSink(url, os.environ.get("MOTRIX_METRICS_TOKEN", "")) if url else None,
            project=os.environ.get("MOTRIX_METRICS_PROJECT") or None,
            group=os.environ.get("MOTRIX_METRICS_GROUP") or None,
            meta=meta,
        )

    @property
    def enabled(self) -> bool:
        return self._jsonl is not None or self._http is not None

    def run(self, state: str, **data: Any) -> None:
        """Report the run's state; ``running`` also carries project, group and meta."""
        if not self.enabled:
            return
        if state == "running":
            identity = {key: value for key, value in self._identity.items() if value is not None}
            data = {**identity, **data}
        self._emit("run", {"state": state, **data})

    def scalars(self, values: Mapping[str, Any], step: int, *, env_steps: int | None = None) -> None:
        """Report scalar values at *step*; NaN and Inf become ``null``.

        A value that is not a single number (a multi-element tensor, a string)
        is left out with a one-time warning per key.
        """
        if not self.enabled:
            return
        data: dict[str, float | None] = {}
        new_nonfinite = []
        for key, value in values.items():
            try:
                number = None if value is None else float(value)
            except Exception:  # noqa: BLE001 - a bad value must not fail training
                if key not in self._invalid_keys:
                    self._invalid_keys.add(key)
                    logger.warning("metrics: dropping %r, %s is not a single number", key, type(value).__name__)
                continue
            if number is not None and not math.isfinite(number):
                number = None
                if key not in self._nonfinite_keys:
                    self._nonfinite_keys.add(key)
                    new_nonfinite.append(key)
            data[str(key)] = number
        if not data:
            return
        axes = None if env_steps is None else {"env_steps": int(env_steps)}
        self._emit("scalars", data, step=step, axes=axes, clean=False)  # already floats or None
        for key in new_nonfinite:
            self.event("nonfinite", step=step, key=key)

    def event(self, name: str, step: int | None = None, **data: Any) -> None:
        """Report a named event such as ``checkpoint_saved``."""
        if not self.enabled:
            return
        self._emit("event", {"name": name, **data}, step=step)

    def close(self) -> None:
        """No more records: each sink sends or writes what it holds at its own pace, then exits.

        Returns at once; nothing waits for the sinks unless :meth:`join` is
        called. Idempotent, never raises.
        """
        with self._emit_lock:
            if self._closed:
                return
            self._closed = True
        for sink in self._sinks():
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - shutting down must not mask the run's outcome
                logger.warning("metrics: closing %s failed: %s", type(sink).__name__, exc)

    def join(self, timeout: float) -> None:
        """Wait up to *timeout* seconds in total for the sinks to finish after :meth:`close`."""
        deadline = time.monotonic() + timeout
        for sink in self._sinks():
            sink.join(max(0.0, deadline - time.monotonic()))

    def _sinks(self) -> list[JsonlSink | HttpSink]:
        return [sink for sink in (self._jsonl, self._http) if sink is not None]

    def _record(self, kind: str, data: dict[str, Any], step: int | None = None, axes: dict | None = None) -> dict:
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        record: dict[str, Any] = {
            "v": 1,
            "run_id": self.run_id,
            "writer": self.writer,
            "seq": seq,
            "t": time.time(),
            "kind": kind,
        }
        if step is not None:
            record["step"] = int(step)
        if axes is not None:
            record["axes"] = axes
        record["data"] = data
        return record

    def _emit(
        self, kind: str, data: dict[str, Any], step: int | None = None, axes: dict | None = None, *, clean: bool = True
    ) -> None:
        if self._closed:
            return
        if clean:
            data = _clean(data)
        # Both sinks under one lock: the jsonl stays in seq order and holds
        # exactly the records the service was sent.
        with self._emit_lock:
            if self._closed:  # close() ran while *data* was being cleaned
                return
            record = self._record(kind, data, step, axes)
            if self._jsonl is not None:
                self._jsonl.write(record)
            if self._http is not None:
                self._http.put(record)


def sanitize_run_id(raw: str) -> str:
    """Fit *raw* to ``^[A-Za-z0-9._-]{1,200}$``, keeping its distinctive tail."""
    cleaned = _RUN_ID_INVALID.sub("_", raw)[-_RUN_ID_MAX:]
    return cleaned or "run"


def _clean(value: Any) -> Any:
    """Make *value* JSON-safe: numpy/torch to Python, NaN/Inf to None, recursively."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(item) for item in value]
    if hasattr(value, "tolist"):  # numpy scalars/arrays and torch tensors
        try:
            return _clean(value.tolist())
        except Exception:  # noqa: BLE001 - fall back to the text form
            pass
    return str(value)


class JsonlSink:
    """Appends records to a local file, one line each, from a daemon thread.

    Like TensorBoard's event writer, ``write`` only queues: the thread
    serializes, writes and flushes whole batches, so a slow or networked disk
    never stalls the training thread. Unlike it, a full queue drops the oldest
    scalars instead of blocking. After :meth:`close` the writer writes what is
    queued and exits. Errors are logged once, never raised.
    """

    def __init__(self, path: Path, *, flush_interval: float = 2.0, max_queue: int = 100_000) -> None:
        self.path = path
        self._flush_interval = flush_interval
        self._max_queue = max_queue
        self._queue: collections.deque[dict[str, Any]] = collections.deque()
        self._cond = threading.Condition()
        self._closing = False
        self._file = None
        self._warned: set[str] = set()
        self.dropped = 0
        self._thread = threading.Thread(target=self._loop, name="motrix-metrics-jsonl", daemon=True)
        self._thread.start()

    def write(self, record: dict[str, Any]) -> None:
        """Queue *record* without blocking."""
        with self._cond:
            if self._closing:
                return
            if not _bounded_append(self._queue, record, self._max_queue):
                self.dropped += 1
                self._warn_once("full", f"metrics: {self.path} is not keeping up, dropping oldest scalars")
            if len(self._queue) >= self._max_queue // 2:
                self._cond.notify()

    def close(self) -> None:
        """No more records: write what is queued, then exit; does not wait."""
        with self._cond:
            self._closing = True
            self._cond.notify()

    def join(self, timeout: float) -> None:
        """Wait up to *timeout* seconds for the writer to finish after :meth:`close`."""
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _loop(self) -> None:
        while True:
            with self._cond:
                # One write and flush per interval, not per record; close() or a
                # half-full queue cut the wait short.
                deadline = time.monotonic() + self._flush_interval
                while not (self._closing or len(self._queue) >= self._max_queue // 2):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)
                batch = list(self._queue)
                self._queue.clear()
                closing = self._closing
            if batch:
                self._write_batch(batch)
            with self._cond:
                if closing and not self._queue:
                    break
        if self._file is not None:
            try:
                self._file.close()
            except Exception as exc:  # noqa: BLE001 - a local write must never stop training
                self._warn_once("io", f"metrics: cannot write {self.path}: {exc}")

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        lines = []
        for record in batch:
            try:
                lines.append(json.dumps(record, allow_nan=False, separators=(",", ":")))
            except Exception as exc:  # noqa: BLE001 - one bad record must not lose the batch
                self._warn_once("encode", f"metrics: cannot encode a record for {self.path}: {exc}")
        if not lines:
            return
        try:
            if self._file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self.path.open("a", encoding="utf-8")
            self._file.write("\n".join(lines) + "\n")
            self._file.flush()
        except Exception as exc:  # noqa: BLE001 - a local write must never stop training
            self._warn_once("io", f"metrics: cannot write {self.path}: {exc}")

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)


class HttpSink:
    """Pushes records to ``{url}/api/v1/ingest`` in gzip batches from a daemon thread.

    The queue is bounded; when full the oldest scalars are dropped, never run
    or event records. Network errors, 408, 429 and 5xx are retried with
    backoff (503 honours ``Retry-After``); other 4xx drop the batch. After
    :meth:`close` the sender sends what is queued and exits, retrying for at
    most ``flush_budget`` seconds before it drops the rest.
    """

    def __init__(
        self,
        url: str,
        token: str = "",
        *,
        batch_size: int = 500,
        flush_interval: float = 2.0,
        request_timeout: float = 5.0,
        heartbeat_interval: float = 30.0,
        max_queue: int = 10_000,
        backoff: float = 1.0,
        max_backoff: float = 30.0,
        flush_budget: float = 60.0,
    ) -> None:
        self.endpoint = url.rstrip("/") + "/api/v1/ingest"
        self._headers = {"Content-Type": "application/json", "Content-Encoding": "gzip"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._request_timeout = request_timeout
        self._heartbeat_interval = heartbeat_interval
        self._max_queue = max_queue
        self._backoff = backoff
        self._max_backoff = max_backoff
        self._flush_budget = flush_budget
        self._queue: collections.deque[dict[str, Any]] = collections.deque()
        self._cond = threading.Condition()
        self._deadline: float | None = None  # set by close()
        self._heartbeat: Callable[[], dict[str, Any]] | None = None
        self._thread = threading.Thread(target=self._loop, name="motrix-metrics-http", daemon=True)
        self._warned: set[str] = set()
        self.dropped = 0

    def start(self, heartbeat: Callable[[], dict[str, Any]]) -> None:
        self._heartbeat = heartbeat
        self._thread.start()

    def put(self, record: dict[str, Any]) -> None:
        """Queue *record* without blocking."""
        with self._cond:
            if not _bounded_append(self._queue, record, self._max_queue):
                self.dropped += 1
                self._warn_once("full", "metrics: send queue full, dropping oldest scalars")
            if len(self._queue) >= self._batch_size:
                self._cond.notify()

    def close(self) -> None:
        """No more records: send what is queued within ``flush_budget``, then exit; does not wait."""
        with self._cond:
            self._deadline = time.monotonic() + self._flush_budget
            self._cond.notify()

    def join(self, timeout: float) -> None:
        """Wait up to *timeout* seconds for the sender to finish after :meth:`close`."""
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _loop(self) -> None:
        last_sent = time.monotonic()
        while True:
            with self._cond:
                if self._deadline is None and len(self._queue) < self._batch_size:
                    self._cond.wait(self._flush_interval)
                if self._deadline is not None and not self._queue:
                    return
                if not self._queue and time.monotonic() - last_sent < self._heartbeat_interval:
                    continue
                batch = [self._queue.popleft() for _ in range(min(self._batch_size, len(self._queue)))]
            try:
                if not batch:
                    batch = [self._heartbeat()]  # idle: tell the service the run is alive
                if not self._send_with_retry(batch):
                    return
            except Exception as exc:  # noqa: BLE001 - the thread must outlive any one batch
                self._warn_once("crash", f"metrics: dropping a batch after an unexpected error: {exc!r}")
            last_sent = time.monotonic()

    def _send_with_retry(self, batch: list[dict[str, Any]]) -> bool:
        """Deliver or drop *batch*; False when the flush budget ran out first."""
        body = gzip.compress(json.dumps({"schema": SCHEMA, "records": batch}, allow_nan=False).encode("utf-8"))
        backoff = self._backoff
        while True:
            done, retry_after = self._post(body)
            if done:
                return True
            delay = backoff if retry_after is None else retry_after
            backoff = min(backoff * 2, self._max_backoff)
            if not self._sleep(delay):
                return False

    def _sleep(self, delay: float) -> bool:
        """Wait *delay* seconds, cut short only by the flush budget; False once it has passed."""
        until = time.monotonic() + delay
        with self._cond:
            while True:
                # put() and close() notify too; only the flush budget may shorten the wait.
                end = until if self._deadline is None else min(until, self._deadline)
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return self._deadline is None or time.monotonic() < self._deadline
                self._cond.wait(remaining)

    def _post(self, body: bytes) -> tuple[bool, float | None]:
        """Send once. Returns (done, Retry-After); done covers delivered and dropped batches."""
        request = urllib.request.Request(self.endpoint, data=body, headers=self._headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self._request_timeout):
                return True, None
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429) or exc.code >= 500:
                self._warn_once("retry", f"metrics: ingest answered {exc.code}, retrying")
                return False, _seconds(exc.headers.get("Retry-After")) if exc.code == 503 else None
            self._warn_once(f"drop{exc.code}", f"metrics: ingest rejected a batch with {exc.code}; dropping it")
            return True, None
        except (OSError, ValueError, HTTPException) as exc:  # URLError, timeouts, resets, bad replies
            self._warn_once("network", f"metrics: cannot reach {self.endpoint} ({exc}), retrying")
            return False, None

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)


def _bounded_append(queue: collections.deque[dict[str, Any]], record: dict[str, Any], limit: int) -> bool:
    """Append *record*; when *queue* is full, drop its oldest scalars (or *record*) first. False if one was dropped."""
    if len(queue) < limit:
        queue.append(record)
        return True
    victim = next((item for item in queue if item["kind"] == "scalars"), None)
    if victim is not None:
        queue.remove(victim)
    elif record["kind"] == "scalars":
        return False
    queue.append(record)
    return victim is None


@atexit.register
def _join_at_exit() -> None:
    """Give sinks still writing or sending up to :data:`EXIT_TIMEOUT` seconds in total.

    Sinks run in daemon threads, which the interpreter kills at exit; without
    this a script that exits right after training would drop its last records,
    the terminal run state among them. Healthy sinks finish in milliseconds.
    """
    deadline = time.monotonic() + EXIT_TIMEOUT
    for metrics in list(_OPEN):
        metrics.close()
        metrics.join(max(0.0, deadline - time.monotonic()))


def _seconds(value: str | None) -> float | None:
    try:
        return max(float(value), 0.0) if value is not None else None
    except ValueError:
        return None
