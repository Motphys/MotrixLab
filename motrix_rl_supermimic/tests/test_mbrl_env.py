# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MBRL replay contracts over real Manager schedules and CPU simulator I/O."""

import os
from dataclasses import dataclass, replace

import pytest
import torch
from _helpers import ReferenceContextCfg, make_toy, make_toy_task
from gymnasium.spaces import Box
from motrix_rl_supermimic.mbrl_env import MbrlEnv, assemble_task
from motrix_rl_supermimic.task_mdp import compile_task_heads
from motrix_rl_supermimic.toy import (
    TARGETS,
    ToyEnvCfg,
    _flat_state,
    _ToyCommand,
    _ToyCommandCfg,
    _VelocityActionCfg,
)
from motrix_rl_supermimic.world_context import WorldContextTerm, WorldContextTermCfg

from motrix_env_core.config import configclass
from motrix_env_core.manager.actions import ActionCfg, ActionState, ActionTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.env import ManagerEnv
from motrix_env_core.sim.model import ActuatorSpec


@dataclass(frozen=True, slots=True)
class _MutatingCommand(_ToyCommand):
    def update(self, ctx: ManagerContext) -> None:
        super(_MutatingCommand, self).update(ctx)
        self.context.add_(3)
        ctx.sim["position"].add_(0.5)


@configclass
class _MutatingCommandCfg(_ToyCommandCfg):
    def __call__(self, ctx: BuildContext) -> _MutatingCommand:
        command = super().__call__(ctx)
        return _MutatingCommand(
            command=command.command,
            frame=command.frame,
            context=torch.full((ctx.num_envs, 1), 2.0),
            targets=command.targets,
        )


def _capture_position_and_previous_action(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(torch.cat((ctx.sim["position"], ctx.actions["velocity"].current()), -1))


@configclass
class _PhysicalContextCfg(WorldContextTermCfg):
    def __call__(self, ctx: BuildContext) -> WorldContextTerm:
        return WorldContextTerm(2, _capture_position_and_previous_action, (1.0, -2.0), (2.0, 4.0))


def test_named_context_terms_derive_width_and_capture_live_partial_reset_data(dynamo_eager: None) -> None:
    cfg = ToyEnvCfg(world_model_context={"reference": ReferenceContextCfg(), "physical": _PhysicalContextCfg()})
    cfg.commands["reference"] = _MutatingCommandCfg()
    env = make_toy(cfg=cfg, num_envs=2)
    env.init_state()
    task = assemble_task(cfg.for_imagined(), "cpu")
    assert env.state_spec.context_dim == task.spec.context_dim == env.world_context.size == 3
    assert env.world_context.slices == task.world_context.slices == {"reference": slice(0, 1), "physical": slice(1, 3)}
    env.command_terms["reference"].context.copy_(torch.tensor([[2.0], [4.0]]))
    env.sim.position.copy_(torch.tensor([[0.2], [-0.3]]))
    env.sim_data.execute()
    first = env.snapshot_transition(env.step(torch.tensor([[0.1], [-0.2]])))
    torch.testing.assert_close(first["xi"], torch.tensor([[2.0, 0.2, 0.0], [4.0, -0.3, 0.0]]))
    expected_normalized = (first["xi"] - torch.tensor([0.0, 1.0, -2.0])) / torch.tensor([1.0, 2.0, 4.0])
    torch.testing.assert_close(task.normalize_context(first["xi"]), expected_normalized)
    env.reset(torch.tensor([1]))
    reference_before = env.command_terms["reference"].context.clone()
    position_before = env.sim_data.view("position").clone()
    previous_before = env.action_terms["velocity"].state.current().clone()
    expected = torch.cat((reference_before, position_before, previous_before), -1)
    second = env.snapshot_transition(env.step(torch.zeros(2, 1)))
    torch.testing.assert_close(second["xi"], expected)
    assert second["xi"][1, 0].item() == second["xi"][1, 2].item() == 0.0
    # The earlier replay row owns its inputs even as reset and command updates mutate live data.
    torch.testing.assert_close(first["xi"], torch.tensor([[2.0, 0.2, 0.0], [4.0, -0.3, 0.0]]))


def test_term_factories_follow_default_device() -> None:
    cfg = ToyEnvCfg()
    ctx = BuildContext(cfg=cfg, actions={}, commands={}, num_envs=2, seed=0)
    with torch.device("meta"):
        command = _ToyCommandCfg()(ctx)
        action = _VelocityActionCfg()(ctx, ())
    for tensor in (command.command, command.frame, command.context, command.targets, action.state.action_queue):
        assert tensor.device.type == "meta"


def test_task_assembly_scopes_device_without_changing_callers_default() -> None:
    cfg = ToyEnvCfg()
    with torch.device("meta"):
        task = assemble_task(cfg.for_imagined(), "cpu")
        assert torch.empty(0).device.type == "meta"
    assert task.world_context.center.device.type == "cpu"
    assert task.goal(task.imagination.bind_context(torch.tensor([0]))).device.type == "cpu"


@pytest.fixture()
def dynamo_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    compile = torch.compile

    def eager(function, **kwargs):
        kwargs.pop("options", None)
        return compile(function, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", eager)


@pytest.fixture()
def env(dynamo_eager: None) -> ManagerEnv:
    manager = make_toy(num_envs=3, cfg=ToyEnvCfg(goal_frames=3))
    manager.init_state()
    return manager


def test_registered_toy_uses_configured_backend_and_records_transition(dynamo_eager: None) -> None:
    from motrix_rl_supermimic.toy import register

    from motrix_env_core import registry

    register()
    manager = registry.make("supermimic-toy", num_envs=2)
    manager.init_state()
    result = manager.step(torch.full((2, 1), 0.1))
    transition = manager.snapshot_transition(result)
    assert transition["ep_step"].tolist() == [0, 0]
    assert transition["f"].tolist() == [0, 0]
    torch.testing.assert_close(transition["s1"][:, 0], torch.full((2,), 0.1 * manager.cfg.sim_substeps))
    imagined = assemble_task(manager.cfg.for_imagined(), "cpu")
    torch.testing.assert_close(
        transition["r"],
        imagined.reward(
            imagined.imagination.bind_context(
                manager.command_terms["reference"].frame,
                state=transition["s1"],
                action=transition["a"],
                previous=transition["a_prev"],
                context=transition["s1"].new_zeros(
                    (*(manager.command_terms["reference"].frame).shape, imagined.spec.context_dim)
                ),
            )
        ),
    )


def test_transition_dispatch_captures_before_command_mutation(dynamo_eager: None) -> None:
    cfg = ToyEnvCfg()
    cfg.world_model_context = {"reference": ReferenceContextCfg()}
    cfg.commands["reference"] = _MutatingCommandCfg()
    env = make_toy(cfg=cfg, num_envs=2)
    env.init_state()
    env.command_terms["reference"].context.fill_(2.0)
    action = torch.full((2, 1), 0.1)
    transition = env.snapshot_transition(env.step(action))
    torch.testing.assert_close(transition["s"][:, :1], torch.zeros_like(action))
    torch.testing.assert_close(transition["s"][:, :], torch.zeros(2, 2))
    torch.testing.assert_close(transition["s1"][:, 0], action[:, 0] * cfg.sim_substeps)
    torch.testing.assert_close(transition["s1"][:, 1], action[:, 0])
    torch.testing.assert_close(transition["xi"], torch.full((2, 1), 2.0))
    torch.testing.assert_close(env.command_terms["reference"].context, torch.full((2, 1), 5.0))
    assert transition["f"].tolist() == [0, 0]
    assert env.command_terms["reference"].frame.tolist() == [1, 1]
    torch.testing.assert_close(transition["a"], action)
    torch.testing.assert_close(transition["a_prev"], torch.zeros_like(action))
    # The observation sees the command-mutated views; replay keeps the physical endpoint.
    torch.testing.assert_close(env.sim_data.view("position")[:, 0], transition["s1"][:, 0] + 0.5)
    torch.testing.assert_close(env.state.obs["goal"], torch.full((2, cfg.goal_frames), TARGETS[1]))


class _AuxiliaryAction(ActionTerm):
    def _process(self, actions: torch.Tensor) -> None:
        return None

    def _reset(self, env_ids: torch.Tensor) -> None:
        pass


@configclass
class _AuxiliaryActionCfg(ActionCfg):
    actuator_names: tuple[str, ...] | None = ()

    def __call__(self, ctx: BuildContext, actuators: tuple[ActuatorSpec, ...] | None) -> ActionTerm:
        return _AuxiliaryAction(
            Box(-1.0, 1.0, (2,)),
            ActionState(torch.zeros(ctx.num_envs, 2, 2), torch.zeros(1, dtype=torch.long)),
        )


def test_term_widths_drive_real_and_learner_assembly(dynamo_eager: None, tmp_path) -> None:
    from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg
    from motrix_rl_supermimic.learner import Learner

    from motrix_env_core.manager.observations import ObservationTermCfg, ObsTerm

    def observe(ctx: ManagerContext, out: torch.Tensor) -> None:
        out.fill_(2.0)

    @configclass(kw_only=True)
    class Observation(ObservationTermCfg):
        width: int

        def __call__(self, ctx: BuildContext) -> ObsTerm:
            return ObsTerm(self.width, observe)

    cfg = ToyEnvCfg()
    original = assemble_task(cfg.for_imagined(), "cpu")
    cfg.goal_frames = 1
    cfg.actions = {"auxiliary": _AuxiliaryActionCfg(), **cfg.actions}
    cfg.observations["policy"] = {"features": Observation(width=6)}
    cfg.observations["value"] = {"features": Observation(width=5)}
    cfg.observations["goal"] = {"features": Observation(width=7)}
    env = make_toy(cfg=cfg, num_envs=2)
    result = env.init_state()
    task = assemble_task(cfg.for_imagined(), "cpu")
    assert task.spec == original.spec
    assert task.action_dim == original.action_dim + 2 == env.action_space.shape[0]
    assert task.observation_dim == result.obs.policy.shape[-1] == 6
    assert task.critic_observation_dim == result.obs.value.shape[-1] == 5
    assert task.goal_dim == result.obs["goal"].shape[-1] == 7
    learner = Learner(SuperMimicCfg(agent=AgentCfg(hidden=8)), task, 2, cfg.ctrl_dt, "cpu", tmp_path)
    action = learner.pi(result.obs.policy[:, None], result.obs["goal"])
    assert action.shape == (env.num_envs, task.action_dim)
    assert learner.pi.obs_norm.mean.shape[-1] == task.observation_dim
    assert learner.pi.goal_norm.mean.shape[-1] == task.goal_dim
    assert learner.critic.norm.mean.shape[-1] == task.critic_observation_dim + task.goal_dim
    transition = env.snapshot_transition(env.step(action.detach()))
    learner.buffer.add(transition)
    torch.testing.assert_close(learner.buffer.data["a"][0], action.detach())
    state = torch.zeros(env.num_envs, task.spec.state_dim)
    predicted = learner.wm(state, action, torch.empty(env.num_envs, task.spec.context_dim))
    assert predicted.shape == state.shape


def test_transition_action_history_follows_manager_slices_and_partial_reset(dynamo_eager: None) -> None:
    cfg = ToyEnvCfg()
    cfg.actions = {"auxiliary": _AuxiliaryActionCfg(), **cfg.actions}
    env = make_toy(cfg=cfg, num_envs=2)
    env.init_state()
    first = torch.tensor([[0.2, -0.3, 0.1], [-0.4, 0.5, -0.1]])
    snapshot = env.snapshot_transition(env.step(first))
    torch.testing.assert_close(snapshot["a"], first)
    torch.testing.assert_close(snapshot["a_prev"], torch.zeros_like(first))
    second = -first
    snapshot = env.snapshot_transition(env.step(second))
    torch.testing.assert_close(snapshot["a"], second)
    torch.testing.assert_close(snapshot["a_prev"], first)
    env.reset(torch.tensor([1]))
    snapshot = env.snapshot_transition(env.step(first))
    expected_previous = second.clone()
    expected_previous[1] = 0
    torch.testing.assert_close(snapshot["a"], first)
    torch.testing.assert_close(snapshot["a_prev"], expected_previous)


def test_transition_frame_uses_selected_reference_command_before_update(dynamo_eager: None) -> None:
    cfg = ToyEnvCfg(reference_command="timeline")
    cfg.commands["timeline"] = _ToyCommandCfg()
    env = make_toy(cfg=cfg, num_envs=2)
    env.init_state()
    reference = env.command_terms["timeline"]
    reference.frame.copy_(torch.tensor([1, 2]))
    snapshot = env.snapshot_transition(env.step(torch.zeros(2, 1)))
    assert snapshot["f"].tolist() == [1, 2]
    assert reference.frame.tolist() == [2, 3]
    assert env.command_terms["reference"].frame.tolist() == [1, 1]


@configclass
class _InvalidFrameCommandCfg(_ToyCommandCfg):
    def __call__(self, ctx: BuildContext) -> _ToyCommand:
        return replace(super().__call__(ctx), frame=torch.zeros(ctx.num_envs))


def test_reference_command_rejects_non_integer_frame() -> None:
    cfg = ToyEnvCfg()
    cfg.commands["reference"] = _InvalidFrameCommandCfg()
    with pytest.raises(ValueError, match="ReferenceCommand.frame"):
        make_toy(cfg=cfg, num_envs=2)


def test_initial_observation_and_learner_task(env: MbrlEnv) -> None:
    imagined = assemble_task(env.cfg.for_imagined(), "cpu")
    assert imagined.spec == env.state_spec
    assert env.command_terms["reference"].frame.tolist() == [0, 0, 0]
    assert env.observation_spec.policy.space.shape == (
        imagined.observation_dim * imagined.observation_spec.policy.history_length,
    )
    assert env.observation_spec.value_or_policy.space.shape == (imagined.critic_observation_dim,)
    assert env.action_space.shape == (imagined.action_dim,)
    assert env.observation_spec["goal"].space.shape == (imagined.goal_dim,)
    assert imagined.observation_spec["goal"].space.shape == env.observation_spec["goal"].space.shape
    torch.testing.assert_close(
        env.state.obs["goal"], imagined.goal(imagined.imagination.bind_context(env.command_terms["reference"].frame))
    )
    torch.testing.assert_close(env.command_terms["reference"].context, torch.zeros(3, 0))
    torch.testing.assert_close(env.action_terms["velocity"].state.current(), torch.zeros(3, 1))
    torch.testing.assert_close(
        env.state.obs.value,
        imagined.critic_observation(
            imagined.imagination.bind_context(
                env.command_terms["reference"].frame,
                state=_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity")),
                action=(_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))).new_zeros(
                    (*env.command_terms["reference"].frame.shape, imagined.action_dim)
                ),
                previous=(_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))).new_zeros(
                    (*env.command_terms["reference"].frame.shape, imagined.action_dim)
                ),
                context=env.command_terms["reference"].context,
            )
        ),
    )
    torch.testing.assert_close(
        env.state.obs.policy[:, -imagined.observation_dim :],
        imagined.observation(
            imagined.imagination.bind_context(
                env.command_terms["reference"].frame,
                state=_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity")),
                action=env.action_terms["velocity"].state.current(),
                previous=env.action_terms["velocity"].state.current(),
                context=env.command_terms["reference"].context,
            )
        ),
        atol=1e-3,
        rtol=0,
    )


@pytest.mark.parametrize("substeps", [1, 2])
@pytest.mark.parametrize("mutate_cfg", [False, True])
def test_shared_heads_frame_reward_units_and_action_history(
    dynamo_eager: None, substeps: int, mutate_cfg: bool
) -> None:
    dt = ToyEnvCfg().sim.dt * substeps
    cfg = ToyEnvCfg(ctrl_dt=dt * 2 if mutate_cfg else dt)
    cfg.ctrl_dt = dt
    env = make_toy(cfg=cfg, num_envs=3)
    env.init_state()
    expected_position = torch.zeros(3)
    previous = torch.zeros(3, 1)
    for step, value in enumerate((0.1, -0.2, 0.05, 0.0, 0.1)):
        action = torch.full((3, 1), value)
        env.step(action)
        transition = env.snapshot_transition(env.state)
        expected_position += value * substeps
        torch.testing.assert_close(
            _flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))[:, 0], expected_position
        )
        torch.testing.assert_close(
            _flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))[:, 1], action[:, 0]
        )
        torch.testing.assert_close(env.action_terms["velocity"].state.current(), action)
        assert env.command_terms["reference"].frame.tolist() == [(step + 1) % len(TARGETS)] * 3
        assert transition["f"].tolist() == [step % len(TARGETS)] * 3
        assert transition["ep_step"].tolist() == [step] * 3
        torch.testing.assert_close(transition["a_prev"], previous)
        imagined = assemble_task(env.cfg.for_imagined(), "cpu")
        torch.testing.assert_close(
            transition["r"],
            imagined.reward(
                imagined.imagination.bind_context(
                    env.command_terms["reference"].frame,
                    state=transition["s1"],
                    action=action,
                    previous=previous,
                    context=transition["s1"].new_zeros(
                        (*env.command_terms["reference"].frame.shape, imagined.spec.context_dim)
                    ),
                )
            ),
        )
        imagined = assemble_task(env.cfg.for_imagined(), "cpu")
        torch.testing.assert_close(
            env.state.obs.policy[:, -imagined.observation_dim :],
            imagined.observation(
                imagined.imagination.bind_context(
                    env.command_terms["reference"].frame,
                    state=_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity")),
                    action=action,
                    previous=action,
                    context=env.command_terms["reference"].context,
                )
            ),
            atol=1e-3,
            rtol=0,
        )
        previous = action.clone()


def test_partial_autoreset_preserves_terminal_snapshots_and_live_rows(env: MbrlEnv) -> None:
    env.step(torch.tensor([[0.1], [0.2], [0.3]]))
    env.sim.position[1] = 2.5
    env.sim_data.execute()
    action = torch.tensor([[0.05], [0.1], [-0.05]])
    result = env.step(action)
    transition = env.snapshot_transition(env.state)
    assert result.terminated.tolist() == [False, True, False]
    assert result.truncated.tolist() == [False, False, False]
    assert transition["done"].tolist() == [False, True, False]
    assert transition["ep_step"].tolist() == [1, 1, 1]
    assert transition["s1"][1, 0] > 2.5
    assert _flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))[1, :].tolist() == [0.0, 0.0]
    assert env.command_terms["reference"].frame.tolist() == [2, 0, 2]
    torch.testing.assert_close(
        result.obs["goal"], torch.tensor([TARGETS[2], TARGETS[0], TARGETS[2]])[:, None].expand(-1, env.cfg.goal_frames)
    )
    torch.testing.assert_close(env.action_terms["velocity"].state.current(), torch.tensor([[0.05], [0.0], [-0.05]]))
    assert env.sim.reset_rows[-1].tolist() == [1]
    torch.testing.assert_close(
        result.obs.policy[1].reshape(-1, 4), torch.tensor([[0.0, 1.0, 0.0, 0.0]]).expand(2, -1), atol=1e-3, rtol=0
    )
    frozen = {key: value.clone() for key, value in transition.items()}
    action.fill_(0.7)
    env.step(torch.zeros(3, 1))
    for key, expected in frozen.items():
        torch.testing.assert_close(transition[key], expected)
    assert env.snapshot_transition(env.state)["ep_step"].tolist() == [2, 0, 2]
    assert env.snapshot_transition(env.state)["a_prev"][1].item() == 0.0


def test_public_partial_reset_uses_manager_actions_commands_and_read_program(env: MbrlEnv) -> None:
    env.step(torch.full((3, 1), 0.2))
    before = _flat_state(env.sim_data.view("position"), env.sim_data.view("velocity")).clone()
    env.reset(torch.tensor([0, 2]))
    assert env.state.episode_steps.tolist() == [0, 1, 0]
    torch.testing.assert_close(
        _flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))[[0, 2]],
        torch.zeros(2, env.state_spec.state_dim),
    )
    torch.testing.assert_close(_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))[1], before[1])
    assert env.command_terms["reference"].frame.tolist() == [0, 1, 0]
    assert env.action_terms["velocity"].state.current()[:, 0].tolist() == pytest.approx([0.0, 0.2, 0.0])
    env.step(torch.full((3, 1), 0.1))
    assert env.command_terms["reference"].frame.tolist() == [1, 2, 1]
    assert env.snapshot_transition(env.state)["a_prev"][:, 0].tolist() == pytest.approx([0.0, 0.2, 0.0])


def test_simulator_only_reset_preserves_counter_and_latest_snapshot(env: MbrlEnv) -> None:
    first = env.step(torch.full((3, 1), 0.1))
    retained = env.snapshot_transition(first)
    steps = env.state.episode_steps.clone()
    frame = env.command_terms["reference"].frame.clone()
    env.reset(torch.empty(0, dtype=torch.long), sim_reset_ids=torch.tensor([1]))
    torch.testing.assert_close(env.state.episode_steps, steps)
    torch.testing.assert_close(env.command_terms["reference"].frame, frame)
    torch.testing.assert_close(env.command_terms["reference"].command, torch.tensor(TARGETS)[frame].unsqueeze(-1))
    assert _flat_state(env.sim_data.view("position"), env.sim_data.view("velocity"))[1, :].tolist() == [0.0, 0.0]
    for name, value in retained.items():
        torch.testing.assert_close(env.snapshot_transition(first)[name], value)


def test_truncation_keeps_endpoint_and_resets_episode(dynamo_eager: None) -> None:
    cfg = ToyEnvCfg(goal_frames=3)
    cfg.max_episode_seconds = cfg.ctrl_dt
    env = make_toy(cfg=cfg, num_envs=2)
    env.init_state()
    result = env.step(torch.full((2, 1), 0.1))
    assert result.truncated.tolist() == [True, True]
    assert result.terminated.tolist() == [False, False]
    assert env.snapshot_transition(env.state)["done"].tolist() == [True, True]
    assert bool((env.snapshot_transition(env.state)["s1"][:, 0] > 0).all())
    assert env.state.episode_steps.tolist() == [0, 0]
    assert env.command_terms["reference"].frame.tolist() == [0, 0]
    assert result.final_obs is not None
    torch.testing.assert_close(result.obs["goal"], torch.full((2, cfg.goal_frames), TARGETS[0]))
    torch.testing.assert_close(result.final_obs.obs["goal"], torch.full((2, cfg.goal_frames), TARGETS[1]))
    imagined = assemble_task(env.cfg.for_imagined(), "cpu")
    torch.testing.assert_close(
        result.final_obs.obs.value,
        imagined.critic_observation(
            imagined.imagination.bind_context(
                torch.ones(2, dtype=torch.long),
                state=env.snapshot_transition(env.state)["s1"],
                action=(env.snapshot_transition(env.state)["s1"]).new_zeros(
                    (*(torch.ones(2, dtype=torch.long)).shape, imagined.action_dim)
                ),
                previous=(env.snapshot_transition(env.state)["s1"]).new_zeros(
                    (*(torch.ones(2, dtype=torch.long)).shape, imagined.action_dim)
                ),
                context=env.command_terms["reference"].context,
            )
        ),
    )


def test_soft_termination_is_not_the_hard_environment_boundary(env: MbrlEnv) -> None:
    env.sim.position.fill_(1.6)
    env.sim_data.execute()
    task = assemble_task(env.cfg.for_imagined(), "cpu")
    action = torch.zeros(3, task.action_dim)
    ctx = task.imagination.bind_context(
        env.command_terms["reference"].frame,
        state=_flat_state(env.sim_data.view("position"), env.sim_data.view("velocity")),
        action=action,
        previous=action,
        context=env.command_terms["reference"].context,
    )
    assert bool((task.soft_termination(ctx) > 0.5).all())
    result = env.step(torch.zeros(3, 1))
    assert not bool(result.terminated.any())


def test_status_buffer_and_control_duration(env: MbrlEnv) -> None:
    status = env.engine_error_codes
    status.fill_(7)
    env.step(torch.zeros(3, 1))
    assert env.engine_error_codes is status
    torch.testing.assert_close(status, torch.zeros(3, dtype=torch.int32))
    assert env.cfg.ctrl_dt > 0


def test_compiled_task_heads_preserve_values_and_gradients(dynamo_eager: None) -> None:
    eager, compiled = make_toy_task(), make_toy_task()
    compile_task_heads(compiled)
    state = torch.randn(3, eager.spec.state_dim, requires_grad=True)
    action = torch.randn(3, 1, requires_grad=True)
    previous, frame, context = torch.zeros_like(action), torch.arange(3), torch.zeros(3, 0)
    outputs = []
    for task in (eager, compiled):
        torch.manual_seed(0)
        observed = task.imagination.bind_context(
            frame, state=state, action=previous, previous=previous, context=context
        )
        physical = task.imagination.bind_context(frame, state=state, action=action, previous=previous, context=context)
        critic_ctx = task.imagination.bind_context(
            frame, state=state, action=previous, previous=previous, context=context
        )
        outputs.append(
            (
                task.observation(observed),
                task.reward(physical),
                task.soft_termination(physical),
                task.critic_observation(critic_ctx),
                task.goal(task.imagination.bind_context(frame)),
                task.normalize_context(context),
            )
        )
    for result, reference in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(result, reference)
    gradients = [
        torch.autograd.grad(
            task.reward(
                task.imagination.bind_context(
                    frame,
                    state=state,
                    action=action,
                    previous=previous,
                    context=state.new_zeros((*frame.shape, task.spec.context_dim)),
                )
            ).sum(),
            (state, action),
        )
        for task in (eager, compiled)
    ]
    for result, reference in zip(*gradients, strict=True):
        torch.testing.assert_close(result, reference)


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("MOTRIX_TEST_MANAGER_COMPILE") != "1", reason="opt-in real Inductor runtime")
def test_inductor_manager_schedule_matches_shared_heads() -> None:
    env = make_toy(num_envs=2)
    env.init_state()
    for action in (torch.full((2, 1), 0.1), torch.full((2, 1), -0.2)):
        result = env.step(action)
        transition = env.snapshot_transition(env.state)
        imagined = assemble_task(env.cfg.for_imagined(), "cpu")
        torch.testing.assert_close(
            result.reward,
            imagined.reward(
                imagined.imagination.bind_context(
                    (transition["f"] + 1) % len(TARGETS),
                    state=transition["s1"],
                    action=transition["a"],
                    previous=transition["a_prev"],
                    context=transition["s1"].new_zeros(
                        (*((transition["f"] + 1) % len(TARGETS)).shape, imagined.spec.context_dim)
                    ),
                )
            ),
        )
    env.reset(torch.tensor([1]))
    env.step(torch.zeros(2, 1))
    assert env.command_terms["reference"].frame.tolist() == [3, 1]
