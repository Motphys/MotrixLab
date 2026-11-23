# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
from motrix_bench.execution import WorkerExecution, worker_execution


class Resource:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_worker_execution_closes_on_success_and_is_idempotent():
    resource = Resource()
    execution = WorkerExecution(resource, lambda value: value.close())
    with execution as acquired:
        assert acquired is resource
    execution.close()
    assert resource.closed == 1


def test_worker_execution_closes_when_body_fails():
    resource = Resource()
    with pytest.raises(RuntimeError, match="boom"):
        with worker_execution(lambda: resource):
            raise RuntimeError("boom")
    assert resource.closed == 1


def test_cleanup_error_is_attached_to_primary_error():
    resource = object()

    def cleanup(_):
        raise RuntimeError("cleanup")

    with pytest.raises(ValueError, match="primary") as error:
        with worker_execution(lambda: resource, cleanup):
            raise ValueError("primary")
    assert "worker cleanup failed" in " ".join(error.value.__notes__)
