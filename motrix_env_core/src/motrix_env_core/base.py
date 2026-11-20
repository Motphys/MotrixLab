# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc

import gymnasium as gym

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, validate_scene_cfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.observations import ObservationSpec


@configclass
class EnvCfg:
    """
    Config for the environment

    """

    scene: SceneCfg | None = None
    sim: SimCfg = SimCfg()
    max_episode_seconds: float = None
    ctrl_dt: float = 0.01
    render_spacing: float = 1.0

    @property
    def max_episode_steps(self) -> int | None:
        """
        return the max episode steps
        """
        if self.max_episode_seconds is None:
            return None
        return int(self.max_episode_seconds / self.ctrl_dt)

    @property
    def sim_substeps(self) -> int:
        """
        return the number of simulation steps per control step
        """
        return int(round(self.ctrl_dt / self.sim.dt))

    def validate(self):
        """
        validate the config
        """
        self.sim.validate()
        if self.sim.dt > self.ctrl_dt:
            raise ValueError("sim.dt must be less than or equal to ctrl_dt")
        if self.scene is None:
            raise ValueError("EnvCfg.scene must be configured")
        validate_scene_cfg(self.scene)

    def for_play(self) -> "EnvCfg":
        """Return the config variant used for play/eval rollouts.

        The default is mode-agnostic (play behaves exactly like train). Envs
        whose play behavior differs override this to return a modified copy
        (e.g. no episode cap, deterministic start, disabled curriculum/noise).
        Keeping the difference on the config makes it the single source of
        truth and keeps the environment itself unaware of any runtime "mode".
        """
        return self


class ABEnv(abc.ABC):
    @property
    @abc.abstractmethod
    def num_envs(self) -> int:
        """
        return the size of the env if it is vectorized
        """

    @property
    @abc.abstractmethod
    def cfg(self) -> EnvCfg:
        """
        The configuration of the environment
        """

    @property
    @abc.abstractmethod
    def observation_spec(self) -> ObservationSpec:
        """Policy/value frame spaces and their history definitions."""

    @property
    @abc.abstractmethod
    def action_space(self) -> gym.Space:
        """Action space"""
