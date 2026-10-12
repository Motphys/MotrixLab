# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Standalone hardware runtime driver."""

from motrix_deploy.errors import ValidationError
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy.runtime.base import DeploymentRuntime
from motrix_deploy.runtime.result import RolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler, LoopScheduler, RealtimeScheduler
from motrix_env_core.input import GamePadDevice


class HardwareRuntime(DeploymentRuntime):
    """Own the hardware host and pace bound control session without advancing physics."""

    def __init__(
        self,
        robot: RobotInterface,
        *,
        scheduler: LoopScheduler | None = None,
        realtime: bool = True,
        gamepad: GamePadDevice | None = None,
    ) -> None:
        super().__init__()
        self._robot = robot
        self.scheduler = scheduler
        self.realtime = realtime
        self._gamepad = gamepad if gamepad is not None else robot.get_gamepad_device()

    @property
    def robot(self) -> RobotInterface:
        return self._robot

    @property
    def gamepad(self) -> GamePadDevice | None:
        """Operator gamepad supplied by the robot's remote channel, if any."""
        return self._gamepad

    def run(self) -> RolloutResult:
        if self.session is None:
            raise ValidationError("runtime.session", "a bound control session", None)
        scheduler = self.scheduler
        if scheduler is None:
            scheduler = (
                RealtimeScheduler(self.session.period_s) if self.realtime else FixedStepScheduler(self.session.period_s)
            )
        elif scheduler.period_s != self.session.period_s:
            raise ValidationError("runtime.scheduler.period_s", self.session.period_s, scheduler.period_s)
        self.open()
        try:
            return self._run_session(scheduler)
        finally:
            self.close()


__all__ = ["HardwareRuntime"]
