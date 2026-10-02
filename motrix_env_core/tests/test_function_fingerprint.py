# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

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


def test_fingerprint_is_stable_across_hash_seeds(tmp_path):
    script = tmp_path / "fingerprint.py"
    script.write_text(
        "from motrix_env_core.numba.fingerprint import function_fingerprint\n"
        "def entry(value):\n"
        "    return value in {'alpha', 'beta', 'gamma'}\n"
        "print(function_fingerprint(entry))\n",
        encoding="utf-8",
    )
    fingerprints = []
    for seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, str(script)],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": "motrix_env_core/src"},
        )
        fingerprints.append(result.stdout.strip())
    assert fingerprints[0] == fingerprints[1]


def test_fingerprint_is_stable_after_inline_helper_compilation():
    namespace = {"__name__": "fingerprint_test", "numba": numba}
    exec(
        "@numba.njit(inline='always')\n"
        "def helper(value):\n"
        "    if value > 0:\n"
        "        return min(value + 1, 10)\n"
        "    return max(value - 1, -10)\n"
        "def entry(value):\n"
        "    return helper(value)\n",
        namespace,
    )
    entry = namespace["entry"]
    first = function_fingerprint(entry)
    assert numba.njit(entry)(2) == 3
    assert function_fingerprint(entry) == first
