# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Standalone hardware runtime driver."""

from motrix_deploy.errors import ValidationError
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy.runtime.base import DeploymentRuntime
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.result import RolloutResult
from motrix_deploy.runtime.scheduler import FixedStepScheduler, LoopScheduler, RealtimeScheduler


class HardwareRuntime(DeploymentRuntime):
    """Drive a control session without owning or advancing robot physics.

    The control session owns robot open/stop/close. This runtime only owns the
    scheduler and ensures an active session is stopped when its lifecycle closes.
    """

    def __init__(
        self,
        robot: RobotInterface,
        *,
        scheduler: LoopScheduler | None = None,
        realtime: bool = True,
    ) -> None:
        super().__init__()
        self._robot = robot
        self.scheduler = scheduler
        self.realtime = realtime

    @property
    def robot(self) -> RobotInterface:
        return self._robot

    def close(self) -> None:
        control = self.control
        if control is not None and control.active:
            control.stop()
        super().close()

    def run(self, control: ControlSession | None = None, *, steps: int | None = None) -> RolloutResult:
        control = self._resolve_control(control)
        if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0):
            raise ValidationError("steps", "a positive integer", steps)
        scheduler = self.scheduler
        if scheduler is None:
            scheduler = RealtimeScheduler(control.period_s) if self.realtime else FixedStepScheduler(control.period_s)
        elif scheduler.period_s != control.period_s:
            raise ValidationError("runtime.scheduler.period_s", control.period_s, scheduler.period_s)
        self.open()
        try:
            if control.start():
                scheduler.reset()
                while control.active and (steps is None or control.completed_steps < steps):
                    step = control.completed_steps
                    scheduler.wait(step)
                    control.tick(elapsed_time_s=scheduler.elapsed_time_s(step))
        except (Exception, KeyboardInterrupt) as error:
            control.fail(error)
        finally:
            control.stop()
            self.close()
        return control.result(steps=steps, overrun_count=scheduler.overrun_count)


__all__ = ["HardwareRuntime"]
