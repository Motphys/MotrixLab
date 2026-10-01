# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numba

from motrix_env_core.numba.fingerprint import function_fingerprint


def test_source_unavailable_constants_change_fingerprint():
    namespace = {"__name__": "fingerprint_test"}
    exec("def entry(): return 1", namespace)
    first = function_fingerprint(namespace["entry"])
    exec("def entry(): return 2", namespace)
    assert function_fingerprint(namespace["entry"]) != first


def test_helper_changes_and_cycles():
    namespace = {"__name__": "fingerprint_test"}
    exec("def helper(): return entry() + 1\ndef entry(): return helper()", namespace)
    entry = namespace["entry"]
    first = function_fingerprint(entry)
    assert function_fingerprint(entry) == first
    exec("def helper(): return entry() + 2", namespace)
    assert function_fingerprint(entry) != first


def test_closure_constants_and_helpers():
    def make(value):
        def entry():
            return value

        return entry

    assert function_fingerprint(make(1)) != function_fingerprint(make(2))

    def helper():
        return 1

    entry = make(helper)
    first = function_fingerprint(entry)
    helper.__code__ = (lambda: 2).__code__
    assert function_fingerprint(entry) != first


def test_defaults_keyword_defaults_and_helper_dependencies():
    def entry(value=1, *, scale=2):
        return value * scale

    first = function_fingerprint(entry)
    entry.__defaults__ = (3,)
    second = function_fingerprint(entry)
    assert second != first
    entry.__kwdefaults__ = {"scale": 4}
    assert function_fingerprint(entry) != second

    def helper():
        return 1

    entry.__defaults__ = (helper,)
    first = function_fingerprint(entry)
    helper.__code__ = (lambda: 2).__code__
    assert function_fingerprint(entry) != first


def test_numba_dispatcher_matches_python_function():
    def helper(value):
        return value + 1

    assert function_fingerprint(numba.njit(helper)) == function_fingerprint(helper)
