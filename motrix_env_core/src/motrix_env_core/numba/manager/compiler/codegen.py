# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from motrix_env_core.numba.manager.compiler.plan import ObservationGroupLayout
from motrix_env_core.numba.manager.compiler.program import PreparedInvocation, ResolvedManagerContext, ResolvedSimReset


@dataclass(frozen=True)
class KernelSourceGenerator:
    """Generate schema-specialized manager kernels for one compilation plan."""

    flat_input_count: int

    def generate(
        self,
        observations: dict[str, tuple[PreparedInvocation, ...]],
        observation_layout: dict[str, ObservationGroupLayout],
        rewards: tuple[PreparedInvocation, ...],
        terminations: tuple[PreparedInvocation, ...],
        context: ResolvedManagerContext,
        command_updates: tuple[PreparedInvocation, ...],
        command_advances: tuple[PreparedInvocation, ...],
        command_resets: tuple[PreparedInvocation, ...],
        reset: tuple[ResolvedSimReset, ...],
    ) -> str:
        evaluate_lane_body, observe_lane_body = self._generate_lane_bodies(
            observations,
            observation_layout,
            rewards,
            terminations,
            command_updates,
            command_advances,
        )
        lines = [
            "def generated_evaluate_kernel(inputs, reward_weights, buffers, outputs):",
        ]
        lines.extend(f"    input_{index} = inputs[{index}]" for index in range(self.flat_input_count))
        lines.extend(
            [
                "    def _evaluate_lane(",
                "        ctx, reward_weights, reward_terms, weighted_reward_terms, termination_masks,",
                "        reward, terminated, env_id,",
                "    ):",
            ]
        )
        lines.extend(f"        {line}" for line in evaluate_lane_body)
        lines.extend(
            [
                "    reward_terms = buffers[0]",
                "    weighted_reward_terms = buffers[1]",
                "    termination_masks = buffers[2]",
                "    reward = outputs.reward",
                "    terminated = outputs.terminated",
                "    for env_id in numba.prange(reward.shape[0]):",
            ]
        )
        lines.append(f"        ctx = {context.expression}")
        lines.extend(
            [
                "        _evaluate_lane(",
                "            ctx,",
                "            reward_weights,",
                "            reward_terms,",
                "            weighted_reward_terms,",
                "            termination_masks,",
                "            reward,",
                "            terminated,",
                "            env_id,",
                "        )",
            ]
        )

        lines.extend(
            [
                "",
                "def generated_observe_kernel(inputs, outputs):",
            ]
        )
        lines.extend(f"    input_{index} = inputs[{index}]" for index in range(self.flat_input_count))
        lines.extend(
            [
                "    policy_obs = outputs.policy_obs",
                "    value_obs = outputs.value_obs",
                "    for env_id in numba.prange(policy_obs.shape[0]):",
                f"        ctx = {context.expression}",
            ]
        )
        lines.extend(f"        {line}" for line in observe_lane_body)

        lines.extend(
            [
                "",
                "def generated_reset_kernel(inputs, env_ids, reset_buffers):",
            ]
        )
        lines.extend(f"    input_{index} = inputs[{index}]" for index in range(self.flat_input_count))
        lines.extend(["    for row in numba.prange(env_ids.shape[0]):", "        env_id = env_ids[row]"])
        lines.append(f"        ctx = {context.expression}")
        for index, command_reset in enumerate(command_resets):
            lines.append(f"        {self._term_call(command_reset, index, 'ctx')}")
        for index, reset_term in enumerate(reset):
            output_args = tuple(
                f"reset_buffers[{reset_term.output_offset + field_index}][row]"
                for field_index in range(reset_term.output_count)
            )
            lines.append(f"        sim_writes_{index} = {reset_term.sim_writes_type}({', '.join(output_args)})")
            call = self._term_call(
                reset_term.invocation,
                index,
                "ctx",
                f"sim_writes_{index}",
            )
            lines.append(f"        {call}")

        return "\n".join(lines) + "\n"

    @classmethod
    def _generate_lane_bodies(
        cls,
        observations: dict[str, tuple[PreparedInvocation, ...]],
        observation_layout: dict[str, ObservationGroupLayout],
        rewards: tuple[PreparedInvocation, ...],
        terminations: tuple[PreparedInvocation, ...],
        command_updates: tuple[PreparedInvocation, ...],
        command_advances: tuple[PreparedInvocation, ...],
    ) -> tuple[list[str], list[str]]:
        """Emit straight-line local helpers that Numba inlines during the kernel frontend pass.

        Keeping these helpers local avoids an extra dispatcher compilation. Their straight-line control flow also lets
        the enclosing ``prange`` remain parallel, unlike heterogeneous ``literal_unroll`` loops.
        """
        evaluate_lines = []
        for index, update in enumerate(command_updates):
            evaluate_lines.append(cls._term_call(update, index, "ctx"))
        evaluate_lines.append("total_reward = 0.0")
        for index, term in enumerate(rewards):
            evaluate_lines.extend(
                [
                    f"reward_value_{index} = {cls._term_call(term, index, 'ctx')}",
                    f"reward_terms[env_id, {index}] = reward_value_{index}",
                    f"weighted_reward_{index} = reward_value_{index} * reward_weights[{index}] * ctx.dt",
                    f"weighted_reward_terms[env_id, {index}] = weighted_reward_{index}",
                    f"total_reward += weighted_reward_{index}",
                ]
            )
        evaluate_lines.append("reward[env_id] = total_reward")
        evaluate_lines.append("is_terminated = False")
        for index, term in enumerate(terminations):
            evaluate_lines.extend(
                [
                    f"termination_value_{index} = {cls._term_call(term, index, 'ctx')}",
                    f"termination_masks[env_id, {index}] = termination_value_{index}",
                    f"is_terminated = is_terminated or termination_value_{index}",
                ]
            )
        evaluate_lines.append("terminated[env_id] = is_terminated")
        for index, advance in enumerate(command_advances):
            evaluate_lines.append(cls._term_call(advance, index, "ctx"))

        observe_lines = []
        observation_index = 0
        for group_name, group_terms in observations.items():
            if group_name not in {"policy", "value"}:
                raise ValueError(f"Unsupported observation group {group_name!r}.")
            output_name = "policy_obs" if group_name == "policy" else "value_obs"
            for term, term_layout in zip(group_terms, observation_layout[group_name].terms, strict=True):
                output_slice = (
                    f"{output_name}[env_id, {term_layout.output_slice.start}:{term_layout.output_slice.stop}]"
                )
                observe_lines.append(cls._term_call(term, observation_index, "ctx", output_slice))
                observation_index += 1
        if not observe_lines:
            observe_lines.append("pass")
        return evaluate_lines, observe_lines

    @staticmethod
    def _term_call(invocation: PreparedInvocation, term_index: int, *arguments: str) -> str:
        """Emit one positional call for a dispatch term or legacy receiver term."""
        symbol = f"term_{invocation.kind}_{term_index}"
        if invocation.receiver_key is not None:
            store_name = "commands"
            arguments = (f"ctx.{store_name}[{invocation.receiver_key!r}]", *arguments)
        arguments = (*arguments, *invocation.args_expressions)
        return f"{symbol}({', '.join(arguments)})"


__all__ = ["KernelSourceGenerator"]
