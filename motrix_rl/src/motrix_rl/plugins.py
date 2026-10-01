# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Discover installed RL integrations without importing them from the core initializer."""

from importlib.metadata import entry_points

_loaded: set[str] = set()


def load_plugins() -> None:
    """Register installed plugins once, propagating plugin failures to the caller.

    Called before Hydra task composition and by registry queries. A failed load
    is not marked successful, so fixing an installation allows a subsequent retry.
    """
    for entry in sorted(entry_points(group="motrix_rl.frameworks"), key=lambda item: item.name):
        if entry.name in _loaded:
            continue
        entry.load()()
        _loaded.add(entry.name)
