# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime resources, scheduling and physics advancement of a bound control session."""

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING

from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.result import RolloutResult, SimulationRolloutResult
from motrix_deploy.runtime.scheduler import LoopScheduler

if TYPE_CHECKING:
    from motrix_deploy.robot.interface import RobotInterface


class DeploymentRuntime(ABC):
    """Own world resources and scheduling, independent of controller semantics."""

    def __init__(self) -> None:
        self._opened = False
        self.session: ControlSession | None = None
        self._used = False

    def bind_session(self, session: ControlSession) -> None:
        if self.session is not None and self.session is not session:
            raise ValidationError("runtime.session", "the already bound control session", session)
        if session.robot is not self.robot:
            raise ValidationError("runtime.session.robot", self.robot, session.robot)
        self.session = session

    @property
    @abstractmethod
    def robot(self) -> "RobotInterface": ...

    @property
    def opened(self) -> bool:
        return self._opened

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def __enter__(self) -> "DeploymentRuntime":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _advance_interval(self) -> None:
        """Apply host-specific execution after an accepted command."""

    def _check_running(self) -> None:
        """Check whether the host permits another control interval."""

    def _run_session(self, scheduler: LoopScheduler) -> RolloutResult:
        if self.session is None:
            raise ValidationError("runtime.session", "a bound control session", None)
        if self._used:
            raise ValidationError("runtime.session", "a single-use execution", "already executed")
        self._used = True
        session = self.session
        try:
            if session.start():
                scheduler.reset()
                interval_step = 0
                while session.active:
                    self._check_running()
                    if not session.tick(elapsed_time_s=scheduler.elapsed_time_s(interval_step)):
                        break
                    self._advance_interval()
                    interval_step += 1
                    scheduler.wait(interval_step)
        except (Exception, KeyboardInterrupt) as error:
            session.fail(error)
        finally:
            session.stop()
        return session.result(overrun_count=scheduler.overrun_count)

    @abstractmethod
    def run(self) -> RolloutResult: ...


class SimulationRuntime(DeploymentRuntime, ABC):
    """Runtime that additionally owns simulation advancement and viewer resources."""

    def __init__(self) -> None:
        super().__init__()
        self._policy_simulation_time_s = 0.0

    def bind_session(self, session: ControlSession) -> None:
        if self.session is not None and self.session is not session:
            raise ValidationError("runtime.session", "the already bound control session", session)
        if session.robot is not self.robot:
            raise ValidationError("runtime.session.robot", self.robot, session.robot)
        self._configure_session_period(session.period_s)
        super().bind_session(session)

    def _configure_session_period(self, period_s: float) -> None:
        """Configure simulation substeps for the control interval."""

    @abstractmethod
    def advance_control_period(self) -> None:
        """Advance physics through one control interval."""

    @abstractmethod
    def _physics_time_s(self) -> float:
        """Read the physical simulation clock."""

    def _advance_interval(self) -> None:
        assert self.session is not None
        before = self._physics_time_s()
        try:
            self.advance_control_period()
        finally:
            if self.session.policy_tick:
                self._policy_simulation_time_s += self._physics_time_s() - before

    def _run_session(self, scheduler: LoopScheduler) -> SimulationRolloutResult:
        self._policy_simulation_time_s = 0.0
        result = super()._run_session(scheduler)
        return SimulationRolloutResult(
            **{item.name: getattr(result, item.name) for item in fields(RolloutResult)},
            policy_simulation_time_s=self._policy_simulation_time_s,
            real_time_factor=self._policy_simulation_time_s / result.wall_time_s if result.wall_time_s > 0 else 0.0,
        )

    def _check_running(self) -> None:
        self.check_viewer_running()

    def check_viewer_running(self) -> None:
        """Reject viewer shutdown; headless simulation hosts need no check."""


__all__ = ["DeploymentRuntime", "SimulationRuntime"]
