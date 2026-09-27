# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from motrix_rl.config import CheckpointConfig, LoggingConfig, TaskConfig, TaskMeta

LOG_DIR_PREFIX = "runs"
METADATA_FILENAME = "metadata.json"
TASK_CONFIG_FILENAME = "task_config.yaml"
CHECKPOINTS_DIRNAME = "checkpoints"


@dataclass(frozen=True)
class RunMetadata:
    env_name: str
    rllib: str
    train_backend: str
    algo: str
    seed: int | None
    created_at: str
    checkpoint_format: str
    sim: str | None = None
    motrixlab_version: str | None = None
    system: dict | None = None
    performance: dict | None = None


@dataclass(frozen=True)
class RunContext:
    """Framework-owned context for a training or play run."""

    run_dir: Path
    metadata: RunMetadata

    @property
    def sim(self) -> str | None:
        """Return the registered simulator name backing manager environments."""
        return self.metadata.sim

    @property
    def checkpoint_dir(self) -> Path:
        """Return the standard checkpoint artifact directory for this run."""
        return self.run_dir / CHECKPOINTS_DIRNAME


def make_metadata(
    env_name: str,
    rllib: str,
    train_backend: str,
    algo: str,
    seed: int | None,
    checkpoint_format: str,
    sim: str | None = None,
) -> RunMetadata:
    return RunMetadata(
        env_name=env_name,
        rllib=rllib,
        train_backend=train_backend,
        algo=algo,
        seed=seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_format=checkpoint_format,
        sim=sim,
    )


def create_run_context(
    env_name: str,
    rllib: str,
    train_backend: str,
    algo: str,
    seed: int | None,
    checkpoint_format: str | None,
    runs_root: Path | str = LOG_DIR_PREFIX,
    sim: str | None = None,
) -> RunContext:
    """Create a framework-owned run directory and write its metadata.

    Args:
        env_name: Registered environment name.
        rllib: RL framework name.
        train_backend: Training backend name.
        algo: Algorithm name.
        seed: Run seed after applying CLI/config overrides.
        checkpoint_format: Preferred checkpoint storage format for this trainer.
        runs_root: Root directory under which run directories are created.
        sim: Registered simulator name backing manager environments, if specified.
    """
    run_dir = _new_run_dir(runs_root, env_name, rllib, train_backend, algo)
    metadata = make_metadata(
        env_name=env_name,
        rllib=rllib,
        train_backend=train_backend,
        algo=algo,
        seed=seed,
        checkpoint_format=_normalize_checkpoint_format(checkpoint_format) or "",
        sim=sim,
    )
    write_metadata(run_dir, metadata)
    context = RunContext(run_dir=run_dir, metadata=metadata)
    context.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return context


def open_run_context(run_dir: Path | str, metadata: RunMetadata | None = None) -> RunContext:
    """Open an existing run directory as a run context.

    Args:
        run_dir: Existing root directory of a training run.
        metadata: Optional run metadata. If omitted, metadata.json is read from ``run_dir``.
    """
    run_path = Path(run_dir)
    return RunContext(run_dir=run_path, metadata=metadata or read_metadata(run_path))


def write_metadata(run_dir: str | Path, metadata: RunMetadata) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / METADATA_FILENAME
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def annotate_run(run_dir: str | Path, **fields: Any) -> Path:
    """Merge annotation fields into an existing run's metadata.json.

    Used by trainers to attach facts that only become known after run
    creation (resolved device layout, end-of-training performance). The
    merge is a raw top-level dict update so future annotation keys do not
    require a RunMetadata field; ``read_metadata`` ignores unknown keys.
    """
    run_dir = Path(run_dir)
    metadata_path = run_dir / METADATA_FILENAME
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    data.update(fields)
    metadata_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata_path


def read_metadata(run_dir: str | Path) -> RunMetadata:
    data = json.loads((Path(run_dir) / METADATA_FILENAME).read_text(encoding="utf-8"))
    # Metadata written by older versions may carry keys the current schema no
    # longer has (e.g. the removed sim_backend axis); ignore unknown keys.
    known = {field.name for field in fields(RunMetadata)}
    return RunMetadata(**{key: value for key, value in data.items() if key in known})


def task_config_path(run_dir: str | Path) -> Path:
    """Return the resolved task-config snapshot path for a run."""
    return Path(run_dir) / TASK_CONFIG_FILENAME


def write_task_config(run_dir: str | Path, config: TaskConfig) -> Path:
    """Persist the resolved task recipe used to create a training run."""
    if not isinstance(config, TaskConfig):
        raise TypeError(f"Expected TaskConfig, got {type(config).__name__}")
    structured = OmegaConf.structured(config)
    resolved = _resolved_mapping(structured)
    snapshot = {
        key: resolved[key] for key in ("task", "num_envs", "play_num_envs", "seed", "logging", "checkpoint", "algo")
    }
    path = task_config_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(snapshot), path)
    return path


def read_task_config(
    run_dir: str | Path,
    algo_config_type: type[Any],
    cfg_override: dict[str, Any] | None = None,
) -> TaskConfig:
    """Load a run's resolved task recipe and restore its typed algorithm config."""
    path = task_config_path(run_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Run {Path(run_dir)} has no {TASK_CONFIG_FILENAME}; "
            "playback requires a run created with a resolved task-config snapshot."
        )
    snapshot = OmegaConf.load(path)
    algo_cfg = OmegaConf.merge(OmegaConf.structured(algo_config_type), snapshot.algo, cfg_override or {})
    return TaskConfig(
        task=TaskMeta(**_resolved_mapping(snapshot.task)),
        num_envs=int(snapshot.num_envs),
        play_num_envs=int(snapshot.play_num_envs),
        seed=snapshot.seed,
        logging=LoggingConfig(**_resolved_mapping(snapshot.logging)),
        checkpoint=CheckpointConfig(**_resolved_mapping(snapshot.checkpoint)),
        algo=OmegaConf.to_object(algo_cfg),
    )


def find_metadata_for_policy(policy_path: str | Path) -> tuple[Path, RunMetadata] | None:
    path = Path(policy_path).resolve()
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        metadata_path = candidate / METADATA_FILENAME
        if metadata_path.exists():
            return candidate, read_metadata(candidate)
    return None


def iter_metadata_runs(env_name: str, runs_root: str | Path = LOG_DIR_PREFIX) -> list[tuple[Path, RunMetadata]]:
    base_dir = Path(runs_root) / env_name
    if not base_dir.exists():
        return []

    runs = []
    for metadata_path in base_dir.rglob(METADATA_FILENAME):
        run_dir = metadata_path.parent
        try:
            runs.append((run_dir, read_metadata(run_dir)))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    return runs


def latest_metadata_run(
    env_name: str,
    rllib: str | None = None,
    train_backend: str | None = None,
    algo: str | None = None,
    runs_root: str | Path = LOG_DIR_PREFIX,
) -> tuple[Path, RunMetadata] | None:
    runs = [
        (run_dir, metadata)
        for run_dir, metadata in iter_metadata_runs(env_name, runs_root=runs_root)
        if (rllib is None or metadata.rllib == rllib)
        and (train_backend is None or metadata.train_backend == train_backend)
        and (algo is None or metadata.algo == algo)
    ]
    if not runs:
        return None
    return max(runs, key=lambda item: item[0].stat().st_mtime)


def _new_run_dir(runs_root: Path | str, env_name: str, rllib: str, train_backend: str, algo: str) -> Path:
    base_dir = Path(runs_root) / env_name / rllib / train_backend / algo
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")
    run_dir = base_dir / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"{timestamp}_{suffix}"
        suffix += 1
    return run_dir


def _normalize_checkpoint_format(checkpoint_format: str | None) -> str | None:
    if checkpoint_format is None:
        return None
    normalized = checkpoint_format.lstrip(".").lower()
    return normalized or None


def _resolved_mapping(config) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in task-config snapshot, got {type(value).__name__}")
    return value
