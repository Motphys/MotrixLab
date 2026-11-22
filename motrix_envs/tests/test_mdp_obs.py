# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.config.scene import KeyPoseCfg, ModelFileCfg, RobotCfg
from motrix_env_core.manager.actions import ActionState, ActionTerm
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.random import RandomSource
from motrix_env_core.mdp.observations import (
    ActionsObsCfg,
    BodyAngularVelocityObsCfg,
    BodyJointVelObsCfg,
    BodyLinearVelocityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.sim.model import BodyModel, SimModel
from motrix_envs.locomotion.wbt.mdp.observations import (
    DofPosRelObsCfg,
)


class _Action(ActionTerm):
    def _process(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def _reset(self, ids: torch.Tensor) -> None:
        pass


def _action() -> ActionTerm:
    return _Action(gym.spaces.Box(-1.0, 1.0, (3,)), ActionState(torch.zeros(1, 2, 3), torch.zeros(1, dtype=torch.long)))


class _FakeObjs:
    """Attribute- and item-accessible objs stand-in (like SceneObjsCfg)."""

    def __init__(self, robot):
        self.robot = robot

    def __getitem__(self, name):
        return getattr(self, name)


class _FakeSimData:
    """Minimal read-program stand-in: query/view plus key-based reads."""

    def __init__(self, queries: dict, views: dict) -> None:
        self._queries = queries
        self._views = views

    def query(self, key: str):
        return self._queries[key]

    def view(self, key: str):
        return self._views[key]

    def __getitem__(self, key: str):
        return self.view(key)


def _env() -> SimpleNamespace:
    robot = RobotCfg(
        model=ModelFileCfg(file="unused.xml"),
        base_link_name="resolved_robot",
        key_pose=KeyPoseCfg(joint_names=["a", "b", "c"], poses={"default": [0.5, 1.0, 1.5]}),
    )
    joint_names = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
    body_model = BodyModel(
        name="robot",
        base_link_name="resolved_robot",
        link_names=("resolved_robot",),
        joint_names=joint_names,
        joint_pos_limits=None,
        actuators=(),
        init_base_position=torch.zeros(3),
        init_base_quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        init_joint_pos=torch.zeros(len(joint_names)),
        init_link_positions=torch.zeros(1, 3),
        init_link_quats=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    )
    return BuildContext(
        cfg=SimpleNamespace(scene=SimpleNamespace(objs=_FakeObjs(robot))),
        num_envs=1,
        seed=17,
        actions={"joint_position": _action()},
        commands={},
        model=SimModel(actuators=(), init_dof_pos=torch.zeros(3), bodies={"robot": body_model}),
    )


def test_actions_observation_reads_current_actions() -> None:
    env = BuildContext(
        cfg=SimpleNamespace(),
        num_envs=1,
        seed=17,
        actions={"joint_position": _action()},
        commands={},
    )
    cfg = ActionsObsCfg()
    term = cfg.__call__(env)
    current = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = torch.empty((1, term.size), dtype=torch.float32)

    assert term.size == 3
    # The batched context exposes the whole-batch queue; current() is the
    # queue row the action pointer selects.
    queue = torch.zeros((1, 2, 3), dtype=torch.float32)
    queue[:, 1] = torch.from_numpy(current)
    batch_state = SimpleNamespace(
        current=lambda: queue[:, 1],
        action_queue=queue,
        action_ptr=torch.asarray([1], dtype=torch.int64),
    )
    ctx = SimpleNamespace(actions={"joint_position": batch_state})
    term.dispatch(ctx, out, *term.args)
    np.testing.assert_array_equal(out.numpy(), current)


def test_robot_observation_cfg_uses_sim_query_layout_and_robot_key_pose() -> None:
    env = _env()
    relative_position_cfg = DofPosRelObsCfg()
    velocity_cfg = BodyJointVelObsCfg()
    linear_cfg = BodyLinearVelocityObsCfg()
    angular_cfg = BodyAngularVelocityObsCfg()

    relative_position = relative_position_cfg.__call__(env)
    velocity = velocity_cfg.__call__(env)
    linear = linear_cfg.__call__(env)
    angular = angular_cfg.__call__(env)

    assert relative_position.size == 3
    assert velocity.size == 3  # size = len(bodies["robot"].joint_names)
    assert linear.size == angular.size == 3
    np.testing.assert_array_equal(relative_position.args[0].reference, np.asarray([0.5, 1.0, 1.5], dtype=np.float32))


def test_robot_dof_pos_rel_obs_cfg_requires_key_pose() -> None:
    with pytest.raises(ValueError, match="key pose 'unknown_pose'"):
        DofPosRelObsCfg(reference_key_pose="unknown_pose").__call__(_env())


def test_robot_dof_pos_rel_obs_cfg_requires_scene_robot() -> None:
    ctx = _env()
    ctx = replace(ctx, cfg=SimpleNamespace(scene=SimpleNamespace(objs=_FakeObjs(robot=None))))

    with pytest.raises(TypeError, match="RobotCfg"):
        DofPosRelObsCfg().__call__(ctx)


def test_robot_base_velocities_are_expressed_in_the_base_local_frame() -> None:
    half_sqrt_two = np.float32(np.sqrt(0.5))
    base_quat = torch.from_numpy(np.asarray([[0.0, 0.0, half_sqrt_two, half_sqrt_two]], dtype=np.float32))
    linear_velocity = torch.from_numpy(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32))
    angular_velocity = torch.from_numpy(np.asarray([[0.0, 2.0, 0.0]], dtype=np.float32))
    linear = BodyLinearVelocityObsCfg().__call__(_env())
    angular = BodyAngularVelocityObsCfg().__call__(_env())
    linear_out = torch.empty((1, 3), dtype=torch.float32)
    angular_out = torch.empty((1, 3), dtype=torch.float32)

    # Str args name sim keys; the resolver binds them to whole-batch arrays,
    # so a direct dispatch call substitutes them explicitly.
    linear.dispatch(None, linear_out, base_quat, linear_velocity, *linear.args[2:])
    angular.dispatch(None, angular_out, base_quat, angular_velocity, *angular.args[2:])

    np.testing.assert_allclose(linear_out.numpy()[0], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(angular_out.numpy()[0], [2.0, 0.0, 0.0], atol=1e-6)


def test_robot_observation_noise_follows_training_seed_and_default_rng() -> None:
    dof_vel = torch.from_numpy(np.tile(np.asarray([1.0, 2.0], dtype=np.float32), (2, 1)))
    term = BodyJointVelObsCfg(noise=UniformNoiseCfg(amplitude=0.25)).__call__(_env())

    def evaluate(ctx: SimpleNamespace) -> np.ndarray:
        out = torch.empty((2, dof_vel.shape[1]), dtype=torch.float32)
        term.dispatch(ctx, out, dof_vel, *term.args[1:])
        return out.numpy()

    torch.manual_seed(7)
    default_state = torch.get_rng_state().clone()
    ctx = SimpleNamespace(rand=RandomSource())
    first = evaluate(ctx)
    continued = evaluate(ctx)
    assert not torch.equal(default_state, torch.get_rng_state())
    torch.manual_seed(7)
    reproduced = evaluate(SimpleNamespace(rand=RandomSource()))
    torch.manual_seed(8)
    different_seed = evaluate(SimpleNamespace(rand=RandomSource()))

    np.testing.assert_array_equal(reproduced, first)
    assert np.any(continued != first)
    assert np.any(different_seed != first)
    assert np.all(first >= dof_vel.numpy() - 0.25)
    assert np.all(first < dof_vel.numpy() + 0.25)
    assert not np.array_equal(first[0], first[1])
