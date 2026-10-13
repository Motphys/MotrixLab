# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The one interface a trainer logs its training metrics through.

``TrainingLogger`` puts TensorBoard and live metrics (``motrix.metrics/v1``,
see :mod:`motrix_rl.metrics`) behind one call per logging window, so a trainer
builds its scalars once and never talks to either output directly. See
``wiki/design/live-metrics.md``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

from motrix_rl.metrics import MetricsLogger

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Writes a trainer's scalars to TensorBoard and sends them, with its events, to live metrics.

    The TensorBoard writer under ``log_dir`` is opened on the first
    :meth:`scalars` call, so a run that fails before logging anything leaves no
    event file. TensorBoard is optional: when it cannot be opened, scalars still
    go live. The live :class:`MetricsLogger` belongs to the control plane, which
    reports the run state and closes it; :meth:`close` closes only TensorBoard.
    """

    def __init__(self, log_dir: str | Path, live: MetricsLogger) -> None:
        self._log_dir = Path(log_dir)
        self.live = live
        self._tensorboard: Any = None
        self._tensorboard_opened = False

    def scalars(
        self,
        values: Mapping[str, float],
        step: int,
        *,
        env_steps: int,
        live_extra: Callable[[], Mapping[str, float]] | None = None,
        placeholders: Collection[str] = (),
    ) -> None:
        """Log one window: *values* to TensorBoard under their keys as tags, and live.

        *live_extra* returns scalars sent live only (host samples TensorBoard
        never had); it is called only when live metrics are on.
        *placeholders* names keys whose value is not defined yet, such as a NaN
        mean before the first episode ends: TensorBoard still gets them, live
        leaves them out so its once-per-key ``nonfinite`` event stays free for a
        real divergence.
        """
        tensorboard = self._open_tensorboard()
        if tensorboard is not None:
            for tag, value in values.items():
                tensorboard.add_scalar(tag, value, step)
        if self.live.enabled:
            live = {key: value for key, value in values.items() if key not in placeholders}
            if live_extra is not None:
                live.update(live_extra())
            self.live.scalars(live, step, env_steps=env_steps)

    def event(self, name: str, step: int | None = None, **data: Any) -> None:
        """Report a named event such as ``checkpoint_saved``; events go live only."""
        self.live.event(name, step, **data)

    def close(self) -> None:
        """Flush and close TensorBoard; never raises, so it is safe in a ``finally``."""
        if self._tensorboard is None:
            return
        try:
            self._tensorboard.close()
        except Exception as exc:  # noqa: BLE001 - closing must not mask the run's outcome
            logger.warning("closing TensorBoard failed: %s", exc)
        self._tensorboard = None

    def _open_tensorboard(self) -> Any:
        if not self._tensorboard_opened:
            self._tensorboard_opened = True
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tensorboard = SummaryWriter(log_dir=str(self._log_dir))
            except Exception as exc:  # noqa: BLE001 - TensorBoard is optional
                logger.warning("TensorBoard is unavailable, logging live metrics only: %s", exc)
        return self._tensorboard
