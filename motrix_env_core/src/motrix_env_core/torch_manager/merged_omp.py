# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Experiment: drop needless OpenMP barriers in Inductor-generated CPU kernels.

Inductor (torch 2.7) already emits ONE ``#pragma omp parallel`` region per
compiled graph containing a sequence of ``#pragma omp for`` loops. Every one
of those loops carries an implicit barrier, so a graph with 30 loops pays 30
full-team barriers per invocation — the fork/join cost is shared, but the
barrier cost is not, which is the dominant overhead when thread counts grow.

This module post-processes the generated C++ source before Inductor compiles
it: it converts an ``#pragma omp for`` to ``#pragma omp for nowait`` when no
later loop in the same parallel region reads (or rewrites) a buffer that an
earlier loop wrote since the previous surviving barrier. Barriers are kept
conservatively whenever a write/read or write/write pair touches the same
buffer across a loop boundary.

Enabled only through ``MOTRIX_TORCH_MERGED_OMP=1`` (set ``2`` for a per-graph
report on stderr). Integration is a monkeypatch of
``torch._inductor.codecache.CppCodeCache.load_async`` — the single choke point
every CPU kernel source passes through — applied before the first
``torch.compile`` call (async-compile workers fork afterwards and inherit it),
and it is idempotent.
"""

from __future__ import annotations

import os
import re
import sys

_PTR = r"(?:in|out|in_out)_ptr\d+"
_ALIAS = re.compile(rf"^\s*auto\s+({_PTR})\s*=\s*({_PTR})\s*;")
_WRITE = re.compile(rf"^\s*({_PTR})\s*\[[^]]*\]\s*=(?!=)", re.M)
_READ = re.compile(rf"\b({_PTR})\b")
_LOOP_PRAGMA = re.compile(r"^\s*#pragma omp for\b(?P<rest>.*)$")
_PARALLEL = re.compile(r"^\s*#pragma omp parallel\b")


def _collect_aliases(lines: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for line in lines:
        m = _ALIAS.match(line)
        if m:
            aliases[m.group(1)] = aliases.get(m.group(2), m.group(2))
    return aliases


def _region_loops(lines: list[str], start: int) -> list[dict]:
    """Describe the ``#pragma omp for`` loop blocks of one parallel region.

    ``start`` indexes the line holding the region's opening brace. Returns one
    record per loop: its ``#pragma omp for`` line index, end line, and the
    canonical read/write pointer sets. The final record carries the region's
    exclusive end line.
    """
    aliases = _collect_aliases(lines)
    loops: list[dict] = []
    depth = 0
    i = start
    n = len(lines)
    while i < n:
        line = lines[i]
        pragma = _LOOP_PRAGMA.match(line)
        if pragma and depth >= 1:
            # Loop body extends from the pragma to the closing brace that
            # returns to the depth held just before the enclosing block.
            loop_depth = depth
            j = i + 1
            while j < n:
                for ch in lines[j]:
                    if ch == "{":
                        loop_depth += 1
                    elif ch == "}":
                        loop_depth -= 1
                if loop_depth < depth:
                    break
                j += 1
            body = "\n".join(lines[i : j + 1])
            writes = {_ALIAS_TO(aliases, w) for w in _WRITE.findall(body)}
            reads = {_ALIAS_TO(aliases, r) for r in _READ.findall(body)}
            loops.append({"pragma": i, "end": j, "reads": reads, "writes": writes})
            i = j
            continue
        depth += line.count("{") - line.count("}")
        if depth == 0:
            loops.append({"region_end": i})
            return loops
        i += 1
    return loops


def _ALIAS_TO(aliases: dict[str, str], name: str) -> str:
    return aliases.get(name, name)


def transform_source(source: str) -> tuple[str, dict[str, int]]:
    """Rewrite ``#pragma omp for`` loops to ``nowait`` where barrier-free.

    Returns the (possibly unchanged) source and a stats dict. Any parse doubt
    leaves the source untouched.
    """
    if "#pragma omp parallel" not in source:
        return source, {"regions": 0, "loops": 0, "nowait": 0, "barriers_kept": 0}
    lines = source.splitlines(keepends=True)
    out = list(lines)
    stats = {"regions": 0, "loops": 0, "nowait": 0, "barriers_kept": 0}
    i = 0
    n = len(lines)
    while i < n:
        if not _PARALLEL.match(lines[i]):
            i += 1
            continue
        # Find the region's opening brace.
        j = i + 1
        while j < n and "{" not in lines[j]:
            if ";" in lines[j]:  # pragma with clause then body on next lines
                break
            j += 1
        if j >= n or "{" not in lines[j]:
            i += 1
            continue
        loops = _region_loops(lines, j)
        region_end = loops.pop()["region_end"] if loops and "region_end" in loops[-1] else None
        if region_end is None or not loops:
            i = region_end + 1 if region_end is not None else j + 1
            continue
        stats["regions"] += 1
        stats["loops"] += len(loops)
        # Forward scan: decide, for each loop, whether the barrier AFTER it
        # can be dropped. A barrier must stay when any later loop before the
        # next surviving barrier reads or writes a buffer written since the
        # last surviving barrier.
        pending_writes: set[str] = set()
        keep_barrier = [False] * len(loops)
        for idx, loop in enumerate(loops):
            if (loop["reads"] | loop["writes"]) & pending_writes:
                # This loop consumes buffers written since the last surviving
                # barrier (idx is > 0 here because pending starts empty), so
                # the barrier after loop idx-1 must stay.
                keep_barrier[idx - 1] = True
                pending_writes = set()
            pending_writes |= loop["writes"]
        for idx, loop in enumerate(loops):
            if idx == len(loops) - 1 or keep_barrier[idx]:
                stats["barriers_kept"] += 1
                continue
            m = _LOOP_PRAGMA.match(out[loop["pragma"]])
            out[loop["pragma"]] = f"#pragma omp for nowait{m.group('rest')}\n"
            stats["nowait"] += 1
        i = region_end + 1
    return "".join(out), stats


_installed = False


def install() -> None:
    """Patch ``CppCodeCache.load_async`` once; no-op when already installed."""
    global _installed
    if _installed:
        return
    from torch._inductor.codecache import CppCodeCache

    verbose = os.environ.get("MOTRIX_TORCH_MERGED_OMP", "0") == "2"
    original = CppCodeCache.load_async.__func__

    def load_async(cls, source_code, *args, **kwargs):
        source_code, stats = transform_source(source_code)
        if verbose and stats.get("loops"):
            print(
                f"[merged-omp] regions={stats['regions']} loops={stats['loops']} "
                f"nowait={stats['nowait']} barriers_kept={stats['barriers_kept']}",
                file=sys.stderr,
            )
        return original(cls, source_code, *args, **kwargs)

    CppCodeCache.load_async = classmethod(load_async)
    CppCodeCache.load = classmethod(
        lambda cls, source_code, *args, **kwargs: cls.load_async(source_code, *args, **kwargs)()
    )
    _installed = True


def maybe_install() -> None:
    if os.environ.get("MOTRIX_TORCH_MERGED_OMP", "0") not in ("0", ""):
        install()
