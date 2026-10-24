# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Build resources unwind even when an algorithm cannot return a plan."""

from contextlib import ExitStack
from unittest.mock import Mock

import pytest

from motrix_rl.execution.building import BuildRuntime


def test_algorithm_resource_ownership_unwinds_on_build_failure():
    resource = Mock()
    error = RuntimeError("invalid algorithm layout")
    with pytest.raises(RuntimeError) as caught:
        with ExitStack() as scope:
            runtime = BuildRuntime(scope)
            assert runtime.own(resource) is resource
            raise error
    assert caught.value is error
    resource.close.assert_called_once_with()


def test_algorithm_explicit_cleanup_and_shared_events():
    released = []
    with ExitStack() as scope:
        runtime = BuildRuntime(scope)
        token = object()
        assert runtime.own(token, lambda: released.append(token)) is token
        event = runtime.event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
    assert released == [token]
