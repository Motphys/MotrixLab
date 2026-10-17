# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import queue
import re
import socket
import threading
import time
import weakref
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motrix_rl import metrics as metrics_module
from motrix_rl.metrics import JSONL_FILENAME, SCHEMA, HttpSink, JsonlSink, MetricsLogger

FIXTURES = Path(__file__).parent / "fixtures" / "metrics_v1"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
WRITER_PATTERN = re.compile(r"^[^:]+:\d+:[a-z]+$")
KINDS = {"run", "scalars", "event", "eval", "media", "heartbeat"}
ENV_VARS = (
    "MOTRIX_METRICS_URL",
    "MOTRIX_METRICS_JSONL",
    "MOTRIX_METRICS_TOKEN",
    "MOTRIX_METRICS_RUN_ID",
    "MOTRIX_METRICS_PROJECT",
    "MOTRIX_METRICS_GROUP",
    "MOTRIX_METRICS_META",
)


class _Ingest:
    """A local ingest service that records every request and answers from a script."""

    def __init__(self, statuses=(), headers=None):
        self.requests = []
        self._statuses = list(statuses)
        self._headers = headers or {}
        ingest = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                ingest.requests.append(
                    {"path": self.path, "headers": dict(self.headers), "body": json.loads(gzip.decompress(raw))}
                )
                status = ingest._statuses.pop(0) if ingest._statuses else 200
                self.send_response(status)
                for key, value in ingest._headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, args=(0.05,), daemon=True).start()

    def records(self):
        return [record for request in self.requests for record in request["body"]["records"]]

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def ingest():
    server = _Ingest()
    yield server
    server.stop()


@pytest.fixture
def captured_http(monkeypatch):
    """Capture serialized protocol requests without opening a socket."""
    requests = []

    def urlopen(request, timeout):
        requests.append(
            {
                "headers": {key.lower(): value for key, value in request.header_items()},
                "body": json.loads(gzip.decompress(request.data)),
            }
        )
        return nullcontext()

    monkeypatch.setattr(metrics_module.urllib.request, "urlopen", urlopen)
    return requests


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _logger(tmp_path, url=None, token="", **http_kwargs):
    http = HttpSink(url, token, **http_kwargs) if url else None
    return MetricsLogger("run-1", jsonl=JsonlSink(tmp_path / JSONL_FILENAME), http=http)


def _wait_until(predicate, timeout=5.0):
    """Return as soon as *predicate* holds; the timeout is only reached when the test fails."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def _jsonl(tmp_path):
    return [json.loads(line) for line in (tmp_path / JSONL_FILENAME).read_text().splitlines()]


def _assert_v1_record(record):
    assert record["v"] == 1
    assert RUN_ID_PATTERN.match(record["run_id"])
    assert WRITER_PATTERN.match(record["writer"])
    assert isinstance(record["seq"], int)
    assert isinstance(record["t"], float)
    assert record["kind"] in KINDS
    assert isinstance(record["data"], dict)
    assert set(record) <= {"v", "run_id", "writer", "seq", "t", "kind", "step", "axes", "data"}
    if record["kind"] == "run":
        assert record["data"]["state"] in {"pending", "running", "finished", "failed", "killed"}


def _metrics_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith("motrix-metrics-")]


def test_without_url_or_jsonl_logger_is_disabled_and_inert(tmp_path):
    threads_before = _metrics_threads()
    logger = MetricsLogger.from_env(tmp_path)

    logger.run("running", total_steps=1)
    logger.scalars({"a": 1.0}, step=0)
    logger.event("run_started")
    logger.close()
    logger.join(5)

    assert not logger.enabled
    assert list(tmp_path.iterdir()) == []
    assert set(_metrics_threads()) <= set(threads_before)  # started none; others' may finish meanwhile


def test_records_carry_protocol_fields_and_monotonic_seq(tmp_path):
    before = time.time_ns() // 1000
    logger = _logger(tmp_path)

    def emit(worker):
        for step in range(50):
            logger.scalars({f"w{worker}": step}, step=step, env_steps=step * 4)

    threads = [threading.Thread(target=emit, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logger.event("checkpoint_saved", step=49, path="ckpt.pt")
    logger.close()
    logger.join(5)

    records = _jsonl(tmp_path)
    for record in records:
        _assert_v1_record(record)
    seqs = [record["seq"] for record in records]
    assert len(records) == 201
    assert seqs == sorted(set(seqs))
    assert seqs[0] > before
    assert records[0]["writer"].endswith(":trainer")
    assert records[0]["axes"] == {"env_steps": records[0]["step"] * 4}
    assert records[-1]["data"] == {"name": "checkpoint_saved", "path": "ckpt.pt"}
    assert "axes" not in records[-1]


def test_values_become_floats_or_null_and_nonfinite_is_reported_once(tmp_path):
    logger = _logger(tmp_path)

    logger.scalars(
        {"nan": float("nan"), "inf": np.float32("inf"), "np": np.float64(2.5), "torch": torch.tensor(1.5), "int": 3},
        step=0,
    )
    logger.scalars({"nan": float("nan"), "inf": 1.0}, step=1)
    logger.event("resumed", step=1, stats={"x": np.array([1.0, float("nan")]), "y": torch.tensor(2)})
    logger.close()
    logger.join(5)

    records = _jsonl(tmp_path)
    assert records[0]["data"] == {"nan": None, "inf": None, "np": 2.5, "torch": 1.5, "int": 3.0}
    nonfinite = [record for record in records if record["kind"] == "event" and record["data"]["name"] == "nonfinite"]
    assert sorted(record["data"]["key"] for record in nonfinite) == ["inf", "nan"]
    assert all(record["step"] == 0 for record in nonfinite)
    assert records[-2]["data"] == {"nan": None, "inf": 1.0}
    assert records[-1]["data"] == {"name": "resumed", "stats": {"x": [1.0, None], "y": 2}}


def test_values_that_are_not_one_number_are_dropped_without_raising(tmp_path):
    logger = _logger(tmp_path)

    for step in range(2):
        logger.scalars({"vec": torch.ones(4), "arr": np.zeros(3), "text": "fast", "ok": torch.tensor([2.0])}, step=step)
    logger.scalars({"vec": torch.ones(4)}, step=2)
    logger.close()
    logger.join(5)

    assert [record["data"] for record in _jsonl(tmp_path)] == [{"ok": 2.0}, {"ok": 2.0}]


def test_records_emitted_while_closing_are_discarded(tmp_path):
    logger = _logger(tmp_path)

    cleaning, release = threading.Event(), threading.Event()

    class SlowValue:
        def tolist(self):
            cleaning.set()
            release.wait(5)
            return 1.0

    logger.run("finished")
    late = threading.Thread(target=logger.event, args=("late",), kwargs={"value": SlowValue()})
    late.start()
    cleaning.wait(5)
    logger.close()
    logger.join(5)
    release.set()
    late.join()

    assert [record["data"]["state"] for record in _jsonl(tmp_path)] == ["finished"]


def test_close_returns_at_once_and_join_waits_for_both_sinks_against_one_deadline():
    calls = []

    class Stuck:
        """A sink that never finishes: join() uses up all the time it is given."""

        def __init__(self, name):
            self.name = name

        def start(self, heartbeat):
            pass

        def close(self):
            calls.append(("close", self.name))

        def join(self, timeout):
            calls.append(("join", self.name, timeout))
            time.sleep(timeout)

    logger = MetricsLogger("run-1", jsonl=Stuck("jsonl"), http=Stuck("http"))
    logger.close()
    assert calls == [("close", "jsonl"), ("close", "http")]  # told, not waited for

    logger.join(0.1)

    assert [call[:2] for call in calls[2:]] == [("join", "jsonl"), ("join", "http")]
    assert calls[3][2] < 0.05  # the second sink gets what is left, not another full timeout


def test_exit_writes_what_a_logger_still_holds(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics_module, "_OPEN", weakref.WeakSet())
    logger = MetricsLogger("run-1", jsonl=JsonlSink(tmp_path / JSONL_FILENAME, flush_interval=60))
    logger.run("finished")

    metrics_module._join_at_exit()  # never closed: the exit hook closes it and waits

    assert [record["data"]["state"] for record in _jsonl(tmp_path)] == ["finished"]


def test_run_id_prefers_environment_and_is_sanitized(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTRIX_METRICS_JSONL", "1")
    run_dir = tmp_path / "runs" / "cartpole" / "skrl" / "torch" / "ppo" / "26-10-12_10-00-00-000000"

    derived = MetricsLogger.from_env(run_dir)
    derived.close()
    derived.join(5)
    assert derived.run_id == "cartpole__skrl__torch__ppo__26-10-12_10-00-00-000000"

    monkeypatch.setenv("MOTRIX_METRICS_RUN_ID", "a b/c:" + "x" * 300 + "-tail")
    explicit = MetricsLogger.from_env(run_dir)
    explicit.close()
    explicit.join(5)
    assert RUN_ID_PATTERN.match(explicit.run_id)
    assert len(explicit.run_id) == 200
    assert explicit.run_id.endswith("x-tail")

    monkeypatch.setenv("MOTRIX_METRICS_RUN_ID", "ok.run_1-2")
    kept = MetricsLogger.from_env(run_dir)
    kept.close()
    kept.join(5)
    assert kept.run_id == "ok.run_1-2"


@pytest.mark.integration
def test_url_alone_pushes_without_writing_a_local_file(monkeypatch, tmp_path, ingest):
    monkeypatch.setenv("MOTRIX_METRICS_URL", ingest.url)
    logger = MetricsLogger.from_env(tmp_path)

    logger.run("running")
    logger.close()
    logger.join(5)

    assert [record["data"]["state"] for record in ingest.records()] == ["running"]
    assert not (tmp_path / JSONL_FILENAME).exists()


@pytest.mark.parametrize("flag", ["1", "true", "YES", "on"])
def test_jsonl_flag_alone_writes_locally_without_http(monkeypatch, tmp_path, flag):
    monkeypatch.setenv("MOTRIX_METRICS_JSONL", flag)
    before = set(threading.enumerate())  # other tests' senders may still be finishing in the background
    logger = MetricsLogger.from_env(tmp_path)

    assert logger.enabled
    assert [thread.name for thread in set(threading.enumerate()) - before] == ["motrix-metrics-jsonl"]
    logger.run("running")
    logger.close()
    logger.join(5)

    assert [record["data"]["state"] for record in _jsonl(tmp_path)] == ["running"]


def test_jsonl_flag_off_values_keep_the_logger_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("MOTRIX_METRICS_JSONL", "0")

    assert not MetricsLogger.from_env(tmp_path).enabled


def test_stuck_disk_does_not_block_emit_and_loses_nothing(tmp_path):
    writing, unstick = threading.Event(), threading.Event()

    class StuckSink(JsonlSink):
        def __init__(self, path):
            self.batches = []
            super().__init__(path, flush_interval=0.01)

        def _write_batch(self, batch):
            writing.set()
            unstick.wait(5)  # a hung or networked run_dir
            self.batches.append(len(batch))
            super()._write_batch(batch)

    sink = StuckSink(tmp_path / JSONL_FILENAME)
    logger = MetricsLogger("run-1", jsonl=sink)
    logger.scalars({"a": 0.0}, step=0)
    writing.wait(5)  # the writer took the first record and is stuck on it

    for step in range(1, 500):
        logger.scalars({"a": float(step)}, step=step)
    finished_while_stuck = sink.batches == []
    unstick.set()
    logger.close()
    logger.join(5)

    assert finished_while_stuck  # every emit returned while the disk was still stuck
    assert [record["step"] for record in _jsonl(tmp_path)] == list(range(500))
    assert sink.batches == [1, 499]  # the backlog goes out as one write and flush, not one per record


@pytest.mark.integration
def test_from_env_pushes_gzip_batches_with_bearer_token(monkeypatch, tmp_path, ingest):
    monkeypatch.setenv("MOTRIX_METRICS_URL", ingest.url + "/")
    monkeypatch.setenv("MOTRIX_METRICS_JSONL", "1")
    monkeypatch.setenv("MOTRIX_METRICS_TOKEN", "secret-token")
    monkeypatch.setenv("MOTRIX_METRICS_PROJECT", "proj")
    monkeypatch.setenv("MOTRIX_METRICS_GROUP", "grp")
    monkeypatch.setenv("MOTRIX_METRICS_META", '{"flow_run_id": "7"}')
    logger = MetricsLogger.from_env(tmp_path)

    logger.run("running", total_steps=10)
    logger.scalars({"rollout/mean_return": 1.0}, step=1, env_steps=8)
    logger.run("finished")
    logger.close()
    logger.join(5)

    assert len(ingest.requests) == 1
    request = ingest.requests[0]
    assert request["path"] == "/api/v1/ingest"
    assert request["headers"]["Authorization"] == "Bearer secret-token"
    assert request["headers"]["Content-Encoding"] == "gzip"
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["body"]["schema"] == SCHEMA
    assert request["body"]["records"] == _jsonl(tmp_path)
    running = request["body"]["records"][0]["data"]
    assert running == {
        "state": "running",
        "project": "proj",
        "group": "grp",
        "meta": {"flow_run_id": "7"},
        "total_steps": 10,
    }
    assert "secret-token" not in (tmp_path / JSONL_FILENAME).read_text()


@pytest.mark.integration
def test_server_errors_are_retried_without_duplicating_records(tmp_path):
    ingest = _Ingest(statuses=[500])
    try:
        # Nothing is sent before close(), so both records share the first batch.
        logger = _logger(tmp_path, ingest.url, flush_interval=60, backoff=0.01)
        logger.scalars({"a": 1.0}, step=0)
        logger.event("warmup_done", step=0)
        logger.close()
        logger.join(10)
    finally:
        ingest.stop()

    assert len(ingest.requests) == 2
    assert ingest.requests[0]["body"] == ingest.requests[1]["body"]
    assert [record["seq"] for record in ingest.records()[2:]] == [record["seq"] for record in _jsonl(tmp_path)]


@pytest.mark.integration
def test_service_unavailable_honours_retry_after(tmp_path):
    ingest = _Ingest(statuses=[503], headers={"Retry-After": "0"})
    try:
        # Only Retry-After: 0 can explain a prompt retry under a 60 s backoff.
        logger = _logger(tmp_path, ingest.url, flush_interval=0.01, backoff=60)
        logger.event("run_started", step=0)
        _wait_until(lambda: len(ingest.requests) == 2)
        logger.close()
        logger.join(5)
    finally:
        ingest.stop()


def test_records_queued_during_backoff_do_not_trigger_early_retries(monkeypatch):
    now = [0.0]
    waits = queue.Queue()
    posts = []

    class ControlledCondition(threading.Condition):
        def wait(self, timeout=None):
            # Report every worker wait; notifications, not elapsed real time, drive it.
            waits.put(timeout)
            return super().wait()

    monkeypatch.setattr(
        metrics_module, "time", SimpleNamespace(monotonic=lambda: now[0], time=time.time, time_ns=time.time_ns)
    )
    http = HttpSink("http://metrics.invalid", batch_size=2, flush_budget=1)
    http._cond = ControlledCondition()

    def unavailable(body):
        posts.append(json.loads(gzip.decompress(body)))
        return False, 30.0  # the transport's parsed Retry-After

    monkeypatch.setattr(http, "_post", unavailable)
    # Start with a full batch so the first observed wait is the retry backoff.
    for step in range(2):
        http.put({"kind": "scalars", "step": step, "data": {"a": float(step)}})
    http.start(lambda: {})
    try:
        assert waits.get(timeout=5) == 30
        for step in range(2, 50, 2):
            with http._cond:
                http.put({"kind": "scalars", "step": step, "data": {}})
                http.put({"kind": "scalars", "step": step + 1, "data": {}})
            # Acquiring the condition ensures the worker has entered wait(). Each
            # notification must be processed and re-enter backoff, not retry HTTP.
            assert waits.get(timeout=5) == 30
            assert len(posts) == 1
        with http._cond:
            now[0] = 29
            http._cond.notify()
        assert waits.get(timeout=5) == 1
        http.close()  # flush deadline is now 30, equal to the retry deadline
        assert waits.get(timeout=5) == 1
        with http._cond:
            now[0] = 30
            http._cond.notify()
        http.join(5)
        assert not http._thread.is_alive()
        assert len(posts) == 1
        assert [record["step"] for record in posts[0]["records"]] == [0, 1]
    finally:
        # Also stop the worker when an assertion fails: no sender leaks into
        # subsequent tests or retries after monkeypatch restores the real clock.
        http._flush_budget = 0
        http.close()
        http.join(5)
        assert not http._thread.is_alive()


class _GarbageThenOk:
    """A TCP server whose first reply is not HTTP; later requests get 200."""

    def __init__(self):
        self.bodies = []
        self._sock = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        replies = 0
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                head, _, body = conn.recv(65536).partition(b"\r\n\r\n")
                length = int(re.search(rb"Content-Length: (\d+)", head).group(1))
                while len(body) < length:
                    body += conn.recv(65536)
                replies += 1
                if replies == 1:
                    conn.sendall(b"garbage\r\n\r\n")
                else:
                    self.bodies.append(json.loads(gzip.decompress(body)))
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")

    def stop(self):
        self._sock.close()


@pytest.mark.integration
def test_malformed_replies_are_retried_and_the_sink_keeps_running(tmp_path):
    server = _GarbageThenOk()
    try:
        logger = _logger(tmp_path, server.url, flush_interval=0.01, backoff=0.01)
        logger.event("run_started", step=0)
        _wait_until(lambda: server.bodies)
        logger.event("warmup_done", step=1)
        logger.close()
        logger.join(5)
    finally:
        server.stop()

    sent = [record["data"]["name"] for body in server.bodies for record in body["records"]]
    assert sent == ["run_started", "warmup_done"]


@pytest.mark.integration
def test_unexpected_sender_error_does_not_stop_delivery(tmp_path, ingest):
    calls = []

    def heartbeat():
        calls.append(None)
        raise RuntimeError("boom")

    http = HttpSink(ingest.url, flush_interval=0.02, heartbeat_interval=0.0)
    http.start(heartbeat)
    _wait_until(lambda: calls)
    http.put({"kind": "event", "data": {"name": "warmup_done"}})
    http.close()
    http.join(5)

    assert calls
    assert [record["data"]["name"] for record in ingest.records()] == ["warmup_done"]


@pytest.mark.integration
def test_client_errors_drop_the_batch(tmp_path):
    ingest = _Ingest(statuses=[401, 401])
    try:
        logger = _logger(tmp_path, ingest.url, flush_interval=0.01)
        logger.event("run_started", step=0)
        _wait_until(lambda: len(ingest.requests) == 1)
        logger.event("warmup_done", step=1)
        logger.close()
        logger.join(5)
    finally:
        ingest.stop()

    assert [[r["data"]["name"] for r in req["body"]["records"]] for req in ingest.requests] == [
        ["run_started"],
        ["warmup_done"],
    ]
    assert len(_jsonl(tmp_path)) == 2


@pytest.mark.integration
def test_unreachable_service_keeps_the_jsonl_and_gives_up_within_its_flush_budget(tmp_path):
    # A bound, non-listening socket reserves an endpoint that cannot accept HTTP;
    # unlike a hard-coded port it cannot accidentally address another service.
    with socket.socket() as endpoint:
        endpoint.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{endpoint.getsockname()[1]}"
        http = HttpSink(url, flush_interval=0.01, backoff=0.01, flush_budget=0.2)
        logger = MetricsLogger("run-1", jsonl=JsonlSink(tmp_path / JSONL_FILENAME), http=http)

        for step in range(1000):
            logger.scalars({"a": float(step)}, step=step)
        logger.close()
        logger.join(5)

    assert not http._thread.is_alive()  # gave up after its budget, well before the join timeout
    assert len(_jsonl(tmp_path)) == 1000


@pytest.mark.integration
def test_full_queue_drops_oldest_scalars_and_keeps_events(tmp_path, ingest):
    logger = _logger(tmp_path, ingest.url, flush_interval=60, max_queue=3)

    logger.event("run_started", step=0)
    for step in range(5):
        logger.scalars({"a": float(step)}, step=step)
    logger.close()
    logger.join(5)

    sent = ingest.records()
    assert [record["kind"] for record in sent] == ["event", "scalars", "scalars"]
    assert [record["step"] for record in sent[1:]] == [3, 4]
    assert len(_jsonl(tmp_path)) == 6


@pytest.mark.integration
def test_idle_sink_sends_heartbeats_only_over_http(tmp_path, ingest):
    logger = _logger(tmp_path, ingest.url, flush_interval=0.01, heartbeat_interval=0.01)

    logger.run("running")
    _wait_until(lambda: any(record["kind"] == "heartbeat" for record in ingest.records()))
    logger.close()
    logger.join(5)

    heartbeats = [record for record in ingest.records() if record["kind"] == "heartbeat"]
    assert heartbeats
    for record in heartbeats:
        _assert_v1_record(record)
        assert record["data"] == {}
    assert [record["kind"] for record in _jsonl(tmp_path)] == ["run"]


@pytest.mark.parametrize("name", ["pipeline_pending.json", "pipeline_failed.json"])
def test_orchestrator_fixtures_share_the_record_format(name, captured_http):
    fixture = json.loads((FIXTURES / name).read_text())
    logger = MetricsLogger("run-1", http=HttpSink("http://metrics.invalid", "fixture-token"))
    logger.run("running")
    logger.close()
    logger.join(5)

    assert fixture["body"]["schema"] == SCHEMA
    assert fixture["headers"] == {key: captured_http[0]["headers"][key.lower()] for key in fixture["headers"]}
    for record in fixture["body"]["records"]:
        _assert_v1_record(record)
        assert record["writer"].endswith(":pipeline")


@pytest.mark.parametrize("name", ["motrixlab_running.json", "motrixlab_scalars.json", "motrixlab_event.json"])
def test_logger_reproduces_motrixlab_fixtures(name, captured_http):
    fixture = json.loads((FIXTURES / name).read_text())
    expected = fixture["body"]["records"]
    logger = MetricsLogger(
        expected[0]["run_id"], http=HttpSink("http://metrics.invalid", "fixture-token", flush_interval=60)
    )

    for record in expected:
        data = dict(record["data"])
        if record["kind"] == "run":
            logger.run(data.pop("state"), **data)
        elif record["kind"] == "scalars":
            values = {key: float("nan") if value is None else value for key, value in data.items()}
            logger.scalars(values, record["step"], env_steps=record["axes"]["env_steps"])
        elif data["name"] != "nonfinite":  # emitted by the logger itself
            logger.event(data.pop("name"), step=record.get("step"), **data)
    logger.close()
    logger.join(5)

    assert len(captured_http) == 1
    request = captured_http[0]
    assert fixture["headers"] == {key: request["headers"][key.lower()] for key in fixture["headers"]}
    sent = request["body"]["records"]
    for record in sent:
        _assert_v1_record(record)
    assert [record["seq"] for record in sent] == sorted(record["seq"] for record in sent)
    normalized = [
        {**record, "writer": want["writer"], "seq": want["seq"], "t": want["t"]}
        for record, want in zip(sent, expected, strict=True)
    ]
    assert {**request["body"], "records": normalized} == fixture["body"]
