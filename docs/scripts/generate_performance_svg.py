# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate static TensorBoard convergence SVGs and their JSON snapshots."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "docs" / "performance.yaml"
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs"
STATIC_DIR = REPO_ROOT / "docs" / "source" / "_static"
METRIC_LABEL_LANGUAGES = ("en", "zh_CN")
TEMPLATES = {"default", "curriculum", "wbt"}
PREFERRED_METRICS = (
    "rollout/mean_return",
    "Reward / Total reward (mean)",
    "episode_return",
    "Train/mean_reward",
    "train/mean_reward",
)
METRIC_LABEL_ALIASES = {
    "rollout/mean_return": "mean_episode_return",
    "reward / total reward (mean)": "mean_episode_return",
    "episode_return": "mean_episode_return",
    "train/mean_reward": "mean_episode_return",
}
METRIC_LABELS = {
    "mean_episode_return": {"en": "Mean episode return", "zh_CN": "平均 Episode 回报"},
}


@dataclass(frozen=True)
class MetricConfig:
    """Optional metric presentation overrides for one environment."""

    metric: str
    metric_label: dict[str, str] | None = None
    template: str = "default"
    curriculum_metric: str | None = None
    curriculum_metric_label: dict[str, str] | None = None
    survival_metric: str | None = None
    survival_metric_label: dict[str, str] | None = None
    episode_length_max: float | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    """Resolved performance-series metadata for one environment run."""

    benchmark_id: str
    task: str
    metric: str
    metric_label: dict[str, str]
    template: str = "default"
    curriculum_metric: str | None = None
    curriculum_metric_label: dict[str, str] | None = None
    survival_metric: str | None = None
    survival_metric_label: dict[str, str] | None = None
    episode_length_max: float | None = None

    @property
    def env_id(self) -> str:
        return self.task.split("/", maxsplit=1)[0]

    @property
    def recipe(self) -> str:
        return self.task.split("/", maxsplit=1)[1]


@dataclass(frozen=True)
class TensorBoardSeries:
    """One scalar series in the coordinate system recorded by TensorBoard."""

    steps: np.ndarray
    elapsed_seconds: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class RunSeries:
    """One scalar series normalized to total environment transitions."""

    run_dir: Path
    environment_steps: np.ndarray
    elapsed_seconds: np.ndarray
    values: np.ndarray
    curriculum_values: np.ndarray | None = None
    survival_values: np.ndarray | None = None
    provenance: dict | None = None


@dataclass(frozen=True)
class Aggregate:
    """Aligned scalar statistics ready for chart rendering."""

    environment_steps: np.ndarray
    elapsed_seconds: np.ndarray | None
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    curriculum_values: np.ndarray | None = None
    survival_values: np.ndarray | None = None


def load_metric_configs(path: Path = DEFAULT_CONFIG) -> dict[str, MetricConfig]:
    """Load optional metric overrides keyed by environment ID."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise RuntimeError(f"{path} must contain performance schema version 1")
    benchmarks = raw.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise RuntimeError(f"{path} must contain a benchmarks mapping")

    result: dict[str, MetricConfig] = {}
    for env_id, value in benchmarks.items():
        if not isinstance(env_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", env_id):
            raise RuntimeError(f"Invalid environment ID: {env_id!r}")
        if not isinstance(value, dict):
            raise RuntimeError(f"Environment {env_id!r} must be a mapping")
        missing = {"metric"} - value.keys()
        if missing:
            raise RuntimeError(f"Environment {env_id!r} is missing fields: {sorted(missing)}")
        labels = value.get("metric_label")
        if labels is not None and (
            not isinstance(labels, dict)
            or any(not isinstance(labels.get(language), str) for language in METRIC_LABEL_LANGUAGES)
        ):
            raise RuntimeError(f"Environment {env_id!r} metric_label must provide en and zh_CN labels")
        template = str(value.get("template", "default"))
        if template not in TEMPLATES:
            raise RuntimeError(f"Environment {env_id!r} has unsupported template {template!r}")
        curriculum_metric = value.get("curriculum_metric")
        curriculum_labels = value.get("curriculum_metric_label")
        survival_metric = value.get("survival_metric")
        survival_labels = value.get("survival_metric_label")
        episode_length_max = value.get("episode_length_max")
        if template == "curriculum":
            if not isinstance(curriculum_metric, str) or not curriculum_metric:
                raise RuntimeError(f"Environment {env_id!r} curriculum template requires curriculum_metric")
            if not isinstance(curriculum_labels, dict) or any(
                not isinstance(curriculum_labels.get(language), str) for language in METRIC_LABEL_LANGUAGES
            ):
                raise RuntimeError(
                    f"Environment {env_id!r} curriculum template requires en and zh_CN curriculum_metric_label"
                )
        if template == "wbt":
            if not isinstance(survival_metric, str) or not survival_metric:
                raise RuntimeError(f"Environment {env_id!r} WBT template requires survival_metric")
            if not isinstance(survival_labels, dict) or any(
                not isinstance(survival_labels.get(language), str) for language in METRIC_LABEL_LANGUAGES
            ):
                raise RuntimeError(f"Environment {env_id!r} WBT template requires en and zh_CN survival_metric_label")
            if not isinstance(episode_length_max, (int, float)) or episode_length_max <= 0:
                raise RuntimeError(f"Environment {env_id!r} WBT template requires a positive episode_length_max")
        result[env_id] = MetricConfig(
            metric=str(value["metric"]),
            metric_label=(
                {language: labels[language] for language in METRIC_LABEL_LANGUAGES}
                if isinstance(labels, dict)
                else None
            ),
            template=template,
            curriculum_metric=str(curriculum_metric) if curriculum_metric is not None else None,
            curriculum_metric_label=(
                {language: curriculum_labels[language] for language in METRIC_LABEL_LANGUAGES}
                if isinstance(curriculum_labels, dict)
                else None
            ),
            survival_metric=str(survival_metric) if survival_metric is not None else None,
            survival_metric_label=(
                {language: survival_labels[language] for language in METRIC_LABEL_LANGUAGES}
                if isinstance(survival_labels, dict)
                else None
            ),
            episode_length_max=float(episode_length_max) if episode_length_max is not None else None,
        )
    return result


def _has_checkpoint(run_dir: Path) -> bool:
    """Return whether a run has a manifest-backed checkpoint artifact."""
    manifest_path = run_dir / "checkpoints" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return False
    return any(
        isinstance(artifact, dict)
        and isinstance(artifact.get("path"), str)
        and (manifest_path.parent / artifact["path"]).is_file()
        for artifact in artifacts.values()
    )


def discover_run_candidates(env_id: str, runs_root: Path) -> list[Path]:
    """Discover checkpoint-backed runs for an environment, newest first."""
    candidates: list[tuple[str, Path]] = []
    env_root = runs_root / env_id
    if not env_root.is_dir():
        return []

    for metadata_path in env_root.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_dir = metadata_path.parent
        if metadata.get("env_name") != env_id or not _has_checkpoint(run_dir):
            continue
        sort_key = str(metadata.get("created_at") or metadata_path.stat().st_mtime_ns)
        candidates.append((sort_key, run_dir))

    return [run_dir for _, run_dir in sorted(candidates, key=lambda item: item[0], reverse=True)]


def list_tensorboard_scalars(run_dir: Path) -> tuple[str, ...]:
    """List scalar tags recorded by one TensorBoard run."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install a training extra such as --extra skrl-torch") from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return tuple(accumulator.Tags().get("scalars", []))


def select_metric(tags: tuple[str, ...], requested_metric: str | None = None) -> str | None:
    """Select an explicit or conventional return/reward scalar tag."""
    if requested_metric is not None:
        return requested_metric if requested_metric in tags else None
    for metric in PREFERRED_METRICS:
        if metric in tags:
            return metric
    lowered = {tag: tag.lower() for tag in tags}
    for token in ("return", "reward"):
        matches = sorted(tag for tag, value in lowered.items() if token in value and "mean" in value)
        if matches:
            return matches[0]
    return None


def metric_labels(metric: str, config: MetricConfig | None) -> dict[str, str]:
    """Resolve localized labels for an automatically or explicitly selected metric."""
    if config is not None and config.metric == metric and config.metric_label is not None:
        return config.metric_label
    label_key = METRIC_LABEL_ALIASES.get(metric.casefold())
    return METRIC_LABELS.get(label_key, {language: metric for language in METRIC_LABEL_LANGUAGES})


def read_tensorboard_scalar(run_dir: Path, metric: str) -> TensorBoardSeries | None:
    """Read all finite values for one scalar tag from a TensorBoard run."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required; install a training extra such as --extra skrl-torch") from exc

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if metric not in accumulator.Tags().get("scalars", []):
        return None
    events = accumulator.Scalars(metric)
    points = [
        (float(event.step), float(event.wall_time), float(event.value))
        for event in events
        if math.isfinite(event.wall_time) and math.isfinite(event.value)
    ]
    if not points:
        return None

    # Keep the last value when event files contain duplicate steps after a resumed run.
    unique = {step: (wall_time, value) for step, wall_time, value in points}
    ordered = sorted(unique.items())
    first_event_time = float(accumulator.FirstEventTimestamp())
    elapsed_seconds = np.asarray(
        [max(0.0, wall_time - first_event_time) for _, (wall_time, _) in ordered],
        dtype=np.float64,
    )
    # A secondary time axis must be monotonic even if a resumed event stream
    # contains overlapping timestamps or the system clock moves backwards.
    elapsed_seconds = np.maximum.accumulate(elapsed_seconds)
    return TensorBoardSeries(
        steps=np.asarray([step for step, _ in ordered], dtype=np.float64),
        elapsed_seconds=elapsed_seconds,
        values=np.asarray([value for _, (_, value) in ordered], dtype=np.float64),
    )


def normalize_environment_steps(spec: BenchmarkSpec, run_dir: Path, tensorboard_steps: np.ndarray) -> np.ndarray:
    """Convert a framework-specific TensorBoard counter to total environment transitions."""
    task_config_path = run_dir / "task_config.yaml"
    if not task_config_path.is_file():
        raise FileNotFoundError(f"Run {run_dir} has no task_config.yaml; environment-step normalization is unavailable")
    raw = OmegaConf.to_container(OmegaConf.load(task_config_path), resolve=True)
    if not isinstance(raw, dict) or not isinstance(raw.get("task"), dict) or not isinstance(raw.get("algo"), dict):
        raise RuntimeError(f"Invalid task config snapshot: {task_config_path}")

    task = raw["task"]
    algo = raw["algo"]
    rllib, algorithm = spec.recipe.split(".")[:2]
    if task.get("env") != spec.env_id or task.get("rllib") != rllib or task.get("algo") != algorithm:
        raise RuntimeError(f"Task config snapshot does not match benchmark {spec.benchmark_id!r}: {task_config_path}")
    num_envs = raw.get("num_envs")
    if not isinstance(num_envs, int) or num_envs <= 0:
        raise RuntimeError(f"Run {run_dir} must have a positive integer num_envs")

    if rllib == "skrl" and algorithm == "ppo":
        completed_vector_steps = tensorboard_steps
    elif rllib == "rslrl" and algorithm == "ppo":
        num_steps_per_env = algo.get("num_steps_per_env")
        if not isinstance(num_steps_per_env, int) or num_steps_per_env <= 0:
            raise RuntimeError(f"RSL-RL run {run_dir} must have a positive integer algo.num_steps_per_env")
        # RSL-RL records learning iteration 0 after completing its first rollout.
        completed_vector_steps = (tensorboard_steps + 1.0) * num_steps_per_env
    elif rllib == "motrix" and algorithm == "fastsac":
        asynchronous = algo.get("asynchronous")
        if not isinstance(asynchronous, bool):
            raise RuntimeError(f"FastSAC run {run_dir} must define boolean algo.asynchronous")
        # TODO(#191): unify sync/async TensorBoard step semantics in the trainers
        # so documentation does not need a topology-specific offset here.
        # The synchronous collector logs step 0 after its first env interaction;
        # the asynchronous collector records completed environment steps directly.
        completed_vector_steps = tensorboard_steps if asynchronous else tensorboard_steps + 1.0
    else:
        raise RuntimeError(f"Unsupported TensorBoard step semantics for benchmark task {spec.task!r}")
    return np.asarray(completed_vector_steps * num_envs, dtype=np.float64)


def select_run_series(
    env_id: str,
    runs_root: Path,
    *,
    requested_metric: str | None,
    metric_config: MetricConfig | None,
    strict: bool,
) -> tuple[BenchmarkSpec | None, list[RunSeries]]:
    """Select the newest usable checkpoint-backed run for an environment."""
    candidates = discover_run_candidates(env_id, runs_root)
    if not candidates:
        if strict:
            raise RuntimeError(f"Environment {env_id!r} has no checkpoint-backed runs")
        return None, []

    run_dir = candidates[0]
    try:
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid run metadata: {run_dir / 'metadata.json'}") from exc
    rllib = metadata.get("rllib")
    algo = metadata.get("algo")
    if not isinstance(rllib, str) or not isinstance(algo, str):
        raise RuntimeError(f"Run metadata does not identify rllib and algo: {run_dir / 'metadata.json'}")
    metric = select_metric(list_tensorboard_scalars(run_dir), requested_metric)
    if metric is None:
        metric_description = repr(requested_metric) if requested_metric is not None else "a return/reward scalar"
        if strict:
            raise RuntimeError(f"Latest run {run_dir} has no {metric_description}")
        return None, []
    spec = BenchmarkSpec(
        benchmark_id=env_id,
        task=f"{env_id}/{rllib}.{algo}",
        metric=metric,
        metric_label=metric_labels(metric, metric_config),
        template=metric_config.template if metric_config is not None else "default",
        curriculum_metric=metric_config.curriculum_metric if metric_config is not None else None,
        curriculum_metric_label=metric_config.curriculum_metric_label if metric_config is not None else None,
        survival_metric=metric_config.survival_metric if metric_config is not None else None,
        survival_metric_label=metric_config.survival_metric_label if metric_config is not None else None,
        episode_length_max=metric_config.episode_length_max if metric_config is not None else None,
    )
    scalar = read_tensorboard_scalar(run_dir, metric)
    if scalar is None:
        if strict:
            raise RuntimeError(f"Latest run {run_dir} has no finite values for {metric!r}")
        return None, []
    try:
        environment_steps = normalize_environment_steps(spec, run_dir, scalar.steps)
    except FileNotFoundError:
        if strict:
            raise
        return None, []
    curriculum_values = None
    if spec.template == "curriculum":
        assert spec.curriculum_metric is not None
        curriculum_scalar = read_tensorboard_scalar(run_dir, spec.curriculum_metric)
        if curriculum_scalar is None:
            if strict:
                raise RuntimeError(
                    f"Latest run {run_dir} has no finite values for curriculum metric {spec.curriculum_metric!r}"
                )
            return None, []
        curriculum_steps = normalize_environment_steps(spec, run_dir, curriculum_scalar.steps)
        curriculum_values = np.interp(environment_steps, curriculum_steps, curriculum_scalar.values)
    survival_values = None
    if spec.template == "wbt":
        assert spec.survival_metric is not None
        assert spec.episode_length_max is not None
        survival_scalar = read_tensorboard_scalar(run_dir, spec.survival_metric)
        if survival_scalar is None:
            if strict:
                raise RuntimeError(
                    f"Latest run {run_dir} has no finite values for survival metric {spec.survival_metric!r}"
                )
            return None, []
        survival_steps = normalize_environment_steps(spec, run_dir, survival_scalar.steps)
        episode_lengths = np.interp(environment_steps, survival_steps, survival_scalar.values)
        survival_values = np.clip(episode_lengths / spec.episode_length_max * 100.0, 0.0, 100.0)
    return (
        spec,
        [
            RunSeries(
                run_dir=run_dir,
                environment_steps=environment_steps,
                elapsed_seconds=scalar.elapsed_seconds,
                values=scalar.values,
                curriculum_values=curriculum_values,
                survival_values=survival_values,
                provenance=metadata,
            )
        ],
    )


def aggregate_series(series: list[RunSeries], max_points: int = 300) -> Aggregate:
    """Align runs on total environment steps and compute curve and elapsed-time aggregates."""
    if not series:
        raise ValueError("At least one run series is required")
    if len(series) == 1:
        item = series[0]
        if len(item.environment_steps) <= max_points:
            environment_steps, elapsed_seconds, mean = item.environment_steps, item.elapsed_seconds, item.values
        else:
            indices = np.linspace(0, len(item.environment_steps) - 1, max_points).astype(int)
            environment_steps = item.environment_steps[indices]
            elapsed_seconds = item.elapsed_seconds[indices]
            mean = item.values[indices]
        return Aggregate(
            environment_steps=environment_steps,
            elapsed_seconds=elapsed_seconds,
            mean=mean,
            lower=mean.copy(),
            upper=mean.copy(),
            curriculum_values=(
                item.curriculum_values if len(item.environment_steps) <= max_points else item.curriculum_values[indices]
            )
            if item.curriculum_values is not None
            else None,
            survival_values=(
                item.survival_values if len(item.environment_steps) <= max_points else item.survival_values[indices]
            )
            if item.survival_values is not None
            else None,
        )

    start = max(item.environment_steps[0] for item in series)
    end = min(item.environment_steps[-1] for item in series)
    if end <= start:
        raise RuntimeError("Selected TensorBoard runs do not have an overlapping environment-step range")
    point_count = min(max_points, max(2, min(len(item.environment_steps) for item in series)))
    environment_steps = np.linspace(start, end, point_count)
    values = np.vstack([np.interp(environment_steps, item.environment_steps, item.values) for item in series])
    elapsed = np.vstack([np.interp(environment_steps, item.environment_steps, item.elapsed_seconds) for item in series])
    curriculum_values = None
    if all(item.curriculum_values is not None for item in series):
        curriculum_values = np.vstack(
            [np.interp(environment_steps, item.environment_steps, item.curriculum_values) for item in series]
        ).mean(axis=0)
    survival_values = None
    if all(item.survival_values is not None for item in series):
        survival_values = np.vstack(
            [np.interp(environment_steps, item.environment_steps, item.survival_values) for item in series]
        ).mean(axis=0)
    mean = values.mean(axis=0)
    confidence = 1.96 * values.std(axis=0, ddof=1) / math.sqrt(len(series))
    return Aggregate(
        environment_steps=environment_steps,
        elapsed_seconds=np.median(elapsed, axis=0),
        mean=mean,
        lower=mean - confidence,
        upper=mean + confidence,
        curriculum_values=curriculum_values,
        survival_values=survival_values,
    )


def _axis_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.3g}"


def _duration_label(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds >= 3600:
        hours, remainder = divmod(int(round(seconds)), 3600)
        minutes = remainder // 60
        return f"{hours}h {minutes:02d}m"
    if seconds >= 60:
        minutes, remainder = divmod(int(round(seconds)), 60)
        return f"{minutes}m {remainder:02d}s"
    if seconds >= 10:
        return f"{seconds:.0f}s"
    return f"{seconds:.1f}s"


def _nice_step(span: float, target_ticks: int) -> float:
    """Choose a conventional 1/2/2.5/5 × 10ⁿ tick interval."""
    if span <= 0:
        return 1.0
    raw_step = span / max(target_ticks - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    multiplier = next(value for value in (1.0, 2.0, 2.5, 5.0, 10.0) if residual <= value)
    return multiplier * magnitude


def _nice_ticks(start: float, end: float, target_ticks: int, *, expand: bool) -> list[float]:
    """Return readable ticks, optionally expanding both axis bounds."""
    if end <= start:
        end = start + 1.0
    step = _nice_step(end - start, target_ticks)
    first = math.floor(start / step) * step if expand else math.ceil(start / step) * step
    last = math.ceil(end / step) * step if expand else math.floor(end / step) * step
    count = max(1, int(round((last - first) / step)) + 1)
    return [first + index * step for index in range(count)]


def _padded_tick_bounds(ticks: list[float], padding_ratio: float = 0.05) -> tuple[float, float]:
    """Add proportional visual headroom without changing labeled ticks."""
    tick_span = ticks[-1] - ticks[0]
    padding = tick_span * padding_ratio
    return ticks[0] - padding, ticks[-1] + padding


def _render_svg(
    spec: BenchmarkSpec,
    aggregate: Aggregate,
    run_count: int,
    *,
    has_curriculum: bool,
    has_survival: bool = False,
    annotation: str | None = None,
) -> str:
    """Render the shared SVG layout for the configured template."""
    if has_curriculum and has_survival:
        raise ValueError("A performance chart can have only one secondary-axis series")
    has_secondary = has_curriculum or has_survival
    has_wall_time = aggregate.elapsed_seconds is not None
    width, height = 960, 570 if has_wall_time else 540
    left, right, top, bottom = 92, 100 if has_secondary else 30, 90 if has_wall_time else 58, 76
    plot_width, plot_height = width - left - right, height - top - bottom
    x_min, x_max = 0.0, float(aggregate.environment_steps[-1])
    raw_y_min = float(min(aggregate.lower.min(), aggregate.mean.min()))
    if x_max <= x_min:
        x_max = x_min + 1.0
    raw_y_max = float(max(aggregate.upper.max(), aggregate.mean.max()))
    if raw_y_min >= 0:
        raw_y_min = 0.0
    y_ticks = _nice_ticks(raw_y_min, raw_y_max, 8, expand=True)
    y_min, y_max = y_ticks[0], y_ticks[-1]
    x_ticks = _nice_ticks(x_min, x_max, 6, expand=False)

    secondary_values: np.ndarray | None = None
    secondary_label: dict[str, str] | None = None
    secondary_css_class = ""
    if has_curriculum:
        secondary_values = aggregate.curriculum_values
        secondary_label = spec.curriculum_metric_label
        secondary_css_class = "curriculum"
    elif has_survival:
        secondary_values = aggregate.survival_values
        secondary_label = spec.survival_metric_label
        secondary_css_class = "survival"

    secondary_min = secondary_max = 0.0
    secondary_ticks: list[float] = []
    if has_secondary:
        assert secondary_values is not None
        if has_survival:
            secondary_ticks = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        else:
            secondary_ticks = _nice_ticks(
                float(secondary_values.min()),
                float(secondary_values.max()),
                6,
                expand=True,
            )
        secondary_min, secondary_max = _padded_tick_bounds(secondary_ticks)

    def x_coord(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_coord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    def secondary_y_coord(value: float) -> float:
        return top + (secondary_max - value) / (secondary_max - secondary_min) * plot_height

    mean_points = " ".join(
        f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}" for x, y in zip(aggregate.environment_steps, aggregate.mean)
    )
    upper = [
        f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}" for x, y in zip(aggregate.environment_steps, aggregate.upper)
    ]
    lower = [
        f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}"
        for x, y in zip(aggregate.environment_steps[::-1], aggregate.lower[::-1])
    ]
    band = " ".join([*upper, *lower])
    secondary_points = ""
    if has_curriculum and secondary_values is not None:
        step_points = [(aggregate.environment_steps[0], secondary_values[0])]
        for index in range(1, len(aggregate.environment_steps)):
            step_points.extend(
                [
                    (aggregate.environment_steps[index], secondary_values[index - 1]),
                    (aggregate.environment_steps[index], secondary_values[index]),
                ]
            )
        secondary_points = " ".join(
            f"{x_coord(float(x)):.2f},{secondary_y_coord(float(y)):.2f}" for x, y in step_points
        )
    elif has_survival and secondary_values is not None:
        secondary_points = " ".join(
            f"{x_coord(float(x)):.2f},{secondary_y_coord(float(y)):.2f}"
            for x, y in zip(aggregate.environment_steps, secondary_values)
        )

    style = (
        "text{font-family:Inter,Arial,sans-serif;fill:#29313d}.grid{stroke:#dfe3e8;stroke-width:1}"
        ".axis{stroke:#67717e;stroke-width:1.2}.curve{fill:none;stroke:#3b82f6;stroke-width:3;stroke-linejoin:round}"
        ".band{fill:#3b82f6;fill-opacity:.18;stroke:none}"
        ".curriculum{fill:none;stroke:#d97706;stroke-width:2.5;stroke-linejoin:round}"
        ".curriculum-axis{stroke:#d97706;stroke-width:1.2}.curriculum-label{fill:#b45309}"
    )
    if has_survival:
        style += (
            ".survival{fill:none;stroke:#059669;stroke-width:2.5;stroke-linejoin:round}"
            ".survival-axis{stroke:#059669;stroke-width:1.2}.survival-label{fill:#047857}"
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f"<style>{style}</style>",
    ]
    if annotation:
        # Header metadata line centered above the legend: the bottom rows are
        # reserved for the axis titles, and the earlier bottom-right footer
        # placement sat too close to the canvas edge (a top-right placement
        # collided with the centered legend at this font size).
        lines.append(
            f'<text x="{width / 2:.2f}" y="18" text-anchor="middle" font-size="11" '
            f'fill="#6b7280">{escape(annotation)}</text>'
        )
    legend_entries = [("curve", spec.metric_label["en"])]
    if has_secondary:
        assert secondary_label is not None
        legend_entries.append((secondary_css_class, secondary_label["en"]))
    legend_widths = [38 + len(label) * 7.2 for _, label in legend_entries]
    legend_total_width = sum(legend_widths) + 28 * (len(legend_entries) - 1)
    legend_x = (width - legend_total_width) / 2
    for (css_class, label), entry_width in zip(legend_entries, legend_widths):
        lines.extend(
            [
                f'<line class="{css_class}" x1="{legend_x:.2f}" y1="34" x2="{legend_x + 28:.2f}" y2="34"/>',
                f'<text x="{legend_x + 36:.2f}" y="38" font-size="13">{escape(label)}</text>',
            ]
        )
        legend_x += entry_width + 28

    elapsed_steps = aggregate.environment_steps
    elapsed_values = aggregate.elapsed_seconds
    if elapsed_values is not None and aggregate.environment_steps[0] > 0:
        elapsed_steps = np.insert(elapsed_steps, 0, 0.0)
        elapsed_values = np.insert(elapsed_values, 0, 0.0)
    for value in x_ticks:
        x = x_coord(value)
        lines.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-size="13">'
                f"{escape(_axis_label(value))}</text>",
            ]
        )
        if elapsed_values is not None:
            elapsed = float(np.interp(value, elapsed_steps, elapsed_values))
            lines.append(
                f'<text x="{x:.2f}" y="{top - 10}" text-anchor="middle" font-size="13">'
                f"{escape(_duration_label(elapsed))}</text>"
            )
    for value in y_ticks:
        y = y_coord(value)
        lines.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="13">'
                f"{escape(_axis_label(value))}</text>",
            ]
        )
    for value in secondary_ticks:
        y = secondary_y_coord(value)
        lines.append(
            f'<text class="{secondary_css_class}-label" x="{left + plot_width + 12:.2f}" y="{y + 4:.2f}" '
            f'font-size="13">{escape(_axis_label(value))}</text>'
        )
    lines.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
            f'y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
        ]
    )
    if has_wall_time:
        lines.extend(
            [
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left + plot_width}" y2="{top}"/>',
                f'<text x="{left + plot_width / 2:.2f}" y="57" text-anchor="middle" font-size="15">'
                "Elapsed wall time</text>",
            ]
        )
    if has_secondary:
        assert secondary_label is not None
        lines.extend(
            [
                f'<line class="{secondary_css_class}-axis" x1="{left + plot_width}" y1="{top}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
                f'<text class="{secondary_css_class}-label" x="{width - 18}" y="{top + plot_height / 2:.2f}" '
                f'text-anchor="middle" font-size="15" transform="rotate(90 {width - 18} '
                f'{top + plot_height / 2:.2f})">{escape(secondary_label["en"])}</text>',
                f'<polyline class="{secondary_css_class}" points="{secondary_points}"/>',
            ]
        )
    if run_count > 1:
        lines.append(f'<polygon class="band" points="{band}"/>')
    lines.extend(
        [
            f'<polyline class="curve" points="{mean_points}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" '
            'font-size="15">Environment steps (total)</text>',
            f'<text x="23" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="15" '
            f'transform="rotate(-90 23 {top + plot_height / 2:.2f})">'
            f"{escape(spec.metric_label['en'])}</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def render_default_svg(spec: BenchmarkSpec, aggregate: Aggregate, run_count: int, annotation: str | None = None) -> str:
    """Render the standard single-axis performance template."""
    return _render_svg(spec, aggregate, run_count, has_curriculum=False, annotation=annotation)


def render_curriculum_svg(
    spec: BenchmarkSpec, aggregate: Aggregate, run_count: int, annotation: str | None = None
) -> str:
    """Render performance and curriculum progress on independent y axes."""
    if aggregate.curriculum_values is None or spec.curriculum_metric_label is None:
        raise RuntimeError(f"Curriculum template for {spec.benchmark_id!r} has no curriculum series")
    return _render_svg(spec, aggregate, run_count, has_curriculum=True, annotation=annotation)


def render_wbt_svg(spec: BenchmarkSpec, aggregate: Aggregate, run_count: int, annotation: str | None = None) -> str:
    """Render WBT return and normalized episode survival on independent y axes."""
    if aggregate.survival_values is None or spec.survival_metric_label is None:
        raise RuntimeError(f"WBT template for {spec.benchmark_id!r} has no survival series")
    return _render_svg(spec, aggregate, run_count, has_curriculum=False, has_survival=True, annotation=annotation)


SVG_RENDERERS = {
    "default": render_default_svg,
    "curriculum": render_curriculum_svg,
    "wbt": render_wbt_svg,
}


def render_svg(spec: BenchmarkSpec, aggregate: Aggregate, run_count: int, annotation: str | None = None) -> str:
    """Dispatch SVG rendering through the selected template."""
    try:
        renderer = SVG_RENDERERS[spec.template]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported performance template: {spec.template!r}") from exc
    return renderer(spec, aggregate, run_count, annotation=annotation)


def _hardware_annotation(series: list[RunSeries]) -> str | None:
    """Concise used-hardware line for the chart, from the runs' provenance."""

    for item in series:
        provenance = item.provenance
        if not (isinstance(provenance, dict) and isinstance(provenance.get("system"), dict)):
            continue
        system = provenance["system"]
        parts = []
        gpus = system.get("gpus_used") or []
        if gpus:
            model = str(gpus[0].get("model", "GPU")).replace("NVIDIA GeForce ", "")
            parts.append(f"{len(gpus)}× {model}")
        cpu_model = str(system.get("cpu", {}).get("model") or "")
        if cpu_model:
            parts.append(cpu_model)
        performance = provenance.get("performance")
        if isinstance(performance, dict):
            wall_minutes = performance.get("wall_time_s")
            if isinstance(wall_minutes, (int, float)) and wall_minutes > 0:
                parts.append(f"{wall_minutes / 60.0:.0f} min")
            rate = performance.get("mean_env_steps_per_s")
            if isinstance(rate, (int, float)) and rate > 0:
                parts.append(f"{rate / 1000.0:.0f}k env-steps/s")
        return " · ".join(parts) if parts else None
    return None


def _profile_slug(series: list[RunSeries]) -> str | None:
    """Hardware slug of a series' provenance, or None when unrecorded."""

    for item in series:
        provenance = item.provenance
        if isinstance(provenance, dict) and isinstance(provenance.get("system"), dict):
            from motrix_rl.system_metrics import hardware_profile_slug

            return hardware_profile_slug(provenance["system"])
    return None


def render_data(
    spec: BenchmarkSpec, aggregate: Aggregate, series: list[RunSeries], data_path: Path | None = None
) -> str:
    """Render a stable JSON sidecar containing chart provenance and aggregate values.

    Schema v2 keeps the default profile inline at the top level (the shape
    readers consume) and adds hardware provenance: ``default_profile`` names
    the used-hardware slug of the inline points, ``provenance`` carries the
    runs' system/performance records, and ``other_profiles`` preserves full
    payloads generated on other machines so regenerating on one host never
    drops another host's data.
    """
    slug = _profile_slug(series)
    other_profiles: dict[str, dict] = {}
    if data_path is not None and data_path.is_file():
        try:
            previous = json.loads(data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if isinstance(previous, dict):
            other_profiles.update(previous.get("other_profiles") or {})
            previous_slug = previous.get("default_profile")
            if isinstance(previous_slug, str) and previous_slug != slug:
                other_profiles[previous_slug] = {
                    key: previous[key]
                    for key in ("benchmark", "task", "template", "metric", "metric_label", "points")
                    if key in previous
                }
                if "provenance" in previous:
                    other_profiles[previous_slug]["provenance"] = previous["provenance"]
    provenance_runs = []
    for item in series:
        record = {}
        if isinstance(item.provenance, dict):
            for key in ("system", "performance"):
                if item.provenance.get(key) is not None:
                    record[key] = item.provenance[key]
        provenance_runs.append(record)
    payload = {
        "schema_version": 2,
        **({"default_profile": slug} if slug is not None else {}),
        **({"other_profiles": other_profiles} if other_profiles else {}),
        "benchmark": spec.benchmark_id,
        "task": spec.task,
        "template": spec.template,
        "metric": spec.metric,
        "metric_label": spec.metric_label,
        **({"curriculum_metric": spec.curriculum_metric} if spec.curriculum_metric is not None else {}),
        **(
            {"curriculum_metric_label": spec.curriculum_metric_label}
            if spec.curriculum_metric_label is not None
            else {}
        ),
        **({"survival_metric": spec.survival_metric} if spec.survival_metric is not None else {}),
        **({"survival_metric_label": spec.survival_metric_label} if spec.survival_metric_label is not None else {}),
        **({"episode_length_max": spec.episode_length_max} if spec.episode_length_max is not None else {}),
        "runs": [item.run_dir.name for item in series],
        **({"provenance": provenance_runs} if provenance_runs else {}),
        "points": [
            {
                "environment_steps": float(environment_steps),
                **({"elapsed_seconds": float(elapsed_seconds)} if elapsed_seconds is not None else {}),
                **({"curriculum_value": float(curriculum_value)} if curriculum_value is not None else {}),
                **({"survival_value": float(survival_value)} if survival_value is not None else {}),
                "mean": float(mean),
                "lower_95": float(lower),
                "upper_95": float(upper),
            }
            for environment_steps, elapsed_seconds, curriculum_value, survival_value, mean, lower, upper in zip(
                aggregate.environment_steps,
                aggregate.elapsed_seconds
                if aggregate.elapsed_seconds is not None
                else [None] * len(aggregate.environment_steps),
                aggregate.curriculum_values
                if aggregate.curriculum_values is not None
                else [None] * len(aggregate.environment_steps),
                aggregate.survival_values
                if aggregate.survival_values is not None
                else [None] * len(aggregate.environment_steps),
                aggregate.mean,
                aggregate.lower,
                aggregate.upper,
            )
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def read_snapshot(
    env_id: str, metric_config: MetricConfig | None
) -> tuple[BenchmarkSpec, Aggregate, list[RunSeries]] | None:
    """Read a committed aggregate-data snapshot when raw training runs are unavailable."""
    path = STATIC_DIR / "data" / "performance" / f"{env_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["benchmark"] != env_id or not str(payload["task"]).startswith(f"{env_id}/"):
            raise RuntimeError(f"Performance snapshot metadata does not match {env_id!r}")
        metric = str(payload["metric"])
        raw_labels = payload.get("metric_label")
        has_stored_labels = isinstance(raw_labels, dict) and all(
            language in raw_labels for language in METRIC_LABEL_LANGUAGES
        )
        has_config_override = (
            metric_config is not None and metric_config.metric == metric and metric_config.metric_label is not None
        )
        if metric.casefold() in METRIC_LABEL_ALIASES or has_config_override or not has_stored_labels:
            labels = metric_labels(metric, metric_config)
        else:
            labels = {language: str(raw_labels[language]) for language in METRIC_LABEL_LANGUAGES}
        template = str(payload.get("template", "default"))
        if template not in TEMPLATES:
            raise RuntimeError(f"Performance snapshot has unsupported template {template!r}: {path}")
        curriculum_metric = payload.get("curriculum_metric")
        raw_curriculum_labels = payload.get("curriculum_metric_label")
        curriculum_labels = (
            {language: str(raw_curriculum_labels[language]) for language in METRIC_LABEL_LANGUAGES}
            if isinstance(raw_curriculum_labels, dict)
            and all(language in raw_curriculum_labels for language in METRIC_LABEL_LANGUAGES)
            else None
        )
        if template == "curriculum" and (not isinstance(curriculum_metric, str) or curriculum_labels is None):
            raise RuntimeError(f"Curriculum performance snapshot is incomplete: {path}")
        survival_metric = payload.get("survival_metric")
        raw_survival_labels = payload.get("survival_metric_label")
        survival_labels = (
            {language: str(raw_survival_labels[language]) for language in METRIC_LABEL_LANGUAGES}
            if isinstance(raw_survival_labels, dict)
            and all(language in raw_survival_labels for language in METRIC_LABEL_LANGUAGES)
            else None
        )
        episode_length_max = payload.get("episode_length_max")
        if template == "wbt" and (
            not isinstance(survival_metric, str)
            or survival_labels is None
            or not isinstance(episode_length_max, (int, float))
            or episode_length_max <= 0
        ):
            raise RuntimeError(f"WBT performance snapshot is incomplete: {path}")
        spec = BenchmarkSpec(
            benchmark_id=env_id,
            task=str(payload["task"]),
            metric=metric,
            metric_label=labels,
            template=template,
            curriculum_metric=str(curriculum_metric) if curriculum_metric is not None else None,
            curriculum_metric_label=curriculum_labels,
            survival_metric=str(survival_metric) if survival_metric is not None else None,
            survival_metric_label=survival_labels,
            episode_length_max=float(episode_length_max) if episode_length_max is not None else None,
        )
        points = payload["points"]
        runs = payload["runs"]
        if not points or not runs:
            raise RuntimeError(f"Performance snapshot is incomplete: {path}")
        provenance_runs = payload.get("provenance")
        if provenance_runs is not None and (not isinstance(provenance_runs, list) or len(provenance_runs) != len(runs)):
            raise RuntimeError(f"Performance snapshot provenance does not match its runs: {path}")
        if all("environment_steps" in point for point in points):
            environment_steps = [point["environment_steps"] for point in points]
        elif all("step" in point for point in points):
            legacy_scale = payload.get("environment_steps_per_tensorboard_step")
            legacy_offset = payload.get("tensorboard_step_offset", 0)
            if not isinstance(legacy_scale, (int, float)) or legacy_scale <= 0:
                raise RuntimeError(
                    f"Legacy performance snapshot requires environment_steps_per_tensorboard_step: {path}"
                )
            environment_steps = [(point["step"] + legacy_offset) * legacy_scale for point in points]
        else:
            raise RuntimeError(f"Performance snapshot has inconsistent environment-step points: {path}")
        has_elapsed = ["elapsed_seconds" in point for point in points]
        if any(has_elapsed) and not all(has_elapsed):
            raise RuntimeError(f"Performance snapshot has incomplete elapsed-time points: {path}")
        has_curriculum = ["curriculum_value" in point for point in points]
        if any(has_curriculum) and not all(has_curriculum):
            raise RuntimeError(f"Performance snapshot has incomplete curriculum points: {path}")
        if template == "curriculum" and not all(has_curriculum):
            raise RuntimeError(f"Curriculum performance snapshot has no curriculum points: {path}")
        has_survival = ["survival_value" in point for point in points]
        if any(has_survival) and not all(has_survival):
            raise RuntimeError(f"Performance snapshot has incomplete survival points: {path}")
        if template == "wbt" and not all(has_survival):
            raise RuntimeError(f"WBT performance snapshot has no survival points: {path}")
        aggregate = Aggregate(
            environment_steps=np.asarray(environment_steps, dtype=np.float64),
            elapsed_seconds=(
                np.asarray([point["elapsed_seconds"] for point in points], dtype=np.float64)
                if all(has_elapsed)
                else None
            ),
            mean=np.asarray([point["mean"] for point in points], dtype=np.float64),
            lower=np.asarray([point["lower_95"] for point in points], dtype=np.float64),
            upper=np.asarray([point["upper_95"] for point in points], dtype=np.float64),
            curriculum_values=(
                np.asarray([point["curriculum_value"] for point in points], dtype=np.float64)
                if all(has_curriculum)
                else None
            ),
            survival_values=(
                np.asarray([point["survival_value"] for point in points], dtype=np.float64)
                if all(has_survival)
                else None
            ),
        )
        series = [
            RunSeries(
                run_dir=Path(str(run)),
                environment_steps=np.asarray([], dtype=np.float64),
                elapsed_seconds=np.asarray([], dtype=np.float64),
                values=np.asarray([], dtype=np.float64),
                provenance=(
                    {key: record[key] for key in ("system", "performance") if key in record}
                    if isinstance((record := provenance_runs[index]), dict)
                    else None
                ),
            )
            for index, run in enumerate(runs)
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid performance snapshot: {path}") from exc
    return spec, aggregate, series


def _update_text(path: Path, generated: str, *, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    if current == generated:
        return False
    if not check:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated, encoding="utf-8")
    return True


def generate(
    env_ids: list[str],
    *,
    metric_configs: dict[str, MetricConfig],
    runs_root: Path,
    requested_metric: str | None,
    strict: bool,
    refresh: bool,
    check: bool,
) -> list[Path]:
    """Generate selected environment SVGs and snapshots, returning stale paths."""
    stale = []
    for env_id in env_ids:
        metric_config = metric_configs.get(env_id)
        snapshot = read_snapshot(env_id, metric_config) if not refresh else None
        if snapshot is not None and not refresh:
            spec, aggregate, series = snapshot
        else:
            spec, series = select_run_series(
                env_id,
                runs_root,
                requested_metric=requested_metric,
                metric_config=metric_config,
                strict=strict,
            )
            aggregate = aggregate_series(series) if series else None
            if aggregate is None and snapshot is not None:
                spec, aggregate, series = snapshot
        if spec is not None and aggregate is not None:
            svg_path = STATIC_DIR / "images" / "performance" / f"{env_id}.svg"
            data_path = STATIC_DIR / "data" / "performance" / f"{env_id}.json"
            if _update_text(
                svg_path, render_svg(spec, aggregate, len(series), annotation=_hardware_annotation(series)), check=check
            ):
                stale.append(svg_path)
            data_stale = _update_text(data_path, render_data(spec, aggregate, series, data_path=data_path), check=check)
            if data_stale:
                stale.append(data_path)
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("envs", nargs="+", help="environment IDs under the runs directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="optional metric-label config path")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="training runs directory")
    parser.add_argument("--metric", help="TensorBoard scalar tag; defaults to an automatically detected return metric")
    parser.add_argument("--check", action="store_true", help="report stale outputs without writing files")
    args = parser.parse_args()

    invalid = sorted(env_id for env_id in args.envs if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", env_id))
    if invalid:
        parser.error(f"invalid environment IDs: {', '.join(invalid)}")
    metric_configs = load_metric_configs(args.config)
    stale = generate(
        args.envs,
        metric_configs=metric_configs,
        runs_root=args.runs_root,
        requested_metric=args.metric,
        strict=True,
        refresh=True,
        check=args.check,
    )
    if args.check and stale:
        for path in stale:
            print(f"stale performance output: {path.relative_to(REPO_ROOT)}")
        return 1
    if not args.check:
        for path in stale:
            print(f"updated performance output: {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
