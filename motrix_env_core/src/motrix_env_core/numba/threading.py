# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime threading-layer helpers for numba parallel kernels.

On many-core machines the default OpenMP layer pays a per-region thread-wakeup
storm (``OMP_WAIT_POLICY=PASSIVE`` parks every worker between calls, and each
``prange`` region must wake them all before any work starts). For short kernels
called every environment step this overhead dominates the kernel itself. The
TBB layer keeps a resident work-stealing pool and does not have this problem.

The PyPI ``tbb`` wheel installs ``libtbb.so.12`` into ``<venv>/lib`` instead of
``site-packages``, so numba's plain ``CDLL("libtbb.so.12")`` lookup cannot find
it without ``LD_LIBRARY_PATH``. Preloading the library by full path with
``RTLD_GLOBAL`` registers its SONAME in the process, and numba's later lookup by
the same SONAME reuses the loaded handle.
"""

from __future__ import annotations

import ctypes
import glob
import sys
from pathlib import Path

_TBB_SONAME = "libtbb.so.12"
_tbb_preload_result: bool | None = None


def preload_tbb() -> bool:
    """Preload ``libtbb.so.12`` from the active environment, if present.

    Returns ``True`` when the library was loaded (or was already loaded), which
    lets numba select the TBB threading layer for parallel kernels. Returns
    ``False`` when the ``tbb`` extra is not installed — callers keep the numba
    default layer in that case. The probe runs once and is cached: every
    ``ManagerEnv`` construction calls this, and the glob + CDLL work should
    not repeat per instance.
    """
    global _tbb_preload_result
    if _tbb_preload_result is None:
        _tbb_preload_result = _probe_tbb()
    return _tbb_preload_result


def _probe_tbb() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    candidates = sorted(glob.glob(str(Path(sys.prefix) / "lib" / f"{_TBB_SONAME}*")), reverse=True)
    for path in candidates:
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        return True
    return False


__all__ = ["preload_tbb"]
