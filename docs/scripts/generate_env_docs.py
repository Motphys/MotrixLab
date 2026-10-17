# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generate localized environment overview tables for the user guide."""

from __future__ import annotations

import argparse
from pathlib import Path

from motrix_env_core import registry

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "configs" / "task"
POSTER_DIR = REPO_ROOT / "docs" / "source" / "_static" / "images" / "poster"
POSTER_SRC_PREFIX = "../../_static/images/poster"
START_MARKER = "<!-- ENV_OVERVIEW_TABLE_START -->"
END_MARKER = "<!-- ENV_OVERVIEW_TABLE_END -->"

_DOC_CONFIGS = {
    "en": {
        "path": REPO_ROOT / "docs" / "source" / "en" / "user_guide" / "envs" / "index.md",
        "headers": ("Poster", "Env ID", "Description", "Training Algorithms"),
    },
    "zh_CN": {
        "path": REPO_ROOT / "docs" / "source" / "zh_CN" / "user_guide" / "envs" / "index.md",
        "headers": ("预览图", "Env ID", "描述", "训练算法"),
    },
}


def discover_training_algorithms(task_dir: Path = TASK_DIR) -> dict[str, tuple[str, ...]]:
    """Discover each environment's unique ``rllib.algo`` Task options."""
    if not task_dir.is_dir():
        raise RuntimeError(f"Task config directory does not exist: {task_dir}")

    result = {}
    for env_dir in sorted(path for path in task_dir.iterdir() if path.is_dir()):
        algorithms = set()
        for task_path in sorted(env_dir.glob("*.yaml")):
            parts = task_path.stem.split(".")
            if len(parts) not in (2, 3) or any(not part for part in parts):
                raise RuntimeError(f"Invalid Task option {task_path!s}; expected <rllib>.<algo>[.<backend>].yaml")
            algorithms.add(".".join(parts[:2]))
        result[env_dir.name] = tuple(sorted(algorithms))
    return result


def _validate_inputs(
    registered_envs: dict[str, dict[str, object]],
    training_algorithms: dict[str, tuple[str, ...]],
    poster_dir: Path,
) -> None:
    missing_descriptions = []
    for env_id, item in registered_envs.items():
        descriptions = item.get("description")
        for language in _DOC_CONFIGS:
            description = descriptions.get(language) if isinstance(descriptions, dict) else None
            if not isinstance(description, str) or not description.strip():
                missing_descriptions.append(f"{env_id}:{language}")
    if missing_descriptions:
        raise RuntimeError(f"Registered environments have missing docstring descriptions: {missing_descriptions}")

    missing_posters = sorted(env_id for env_id in registered_envs if not (poster_dir / f"{env_id}.jpg").is_file())
    if missing_posters:
        raise RuntimeError(f"Registered environments have missing poster images: {missing_posters}")

    unknown_task_envs = sorted(set(training_algorithms) - set(registered_envs))
    if unknown_task_envs:
        raise RuntimeError(f"Task configs reference unregistered environments: {unknown_task_envs}")


def _markdown_cell(value: str) -> str:
    return " ".join(value.splitlines()).replace("|", "\\|")


def _render_env_table(
    language: str,
    registered_envs: dict[str, dict[str, object]],
    training_algorithms: dict[str, tuple[str, ...]],
) -> str:
    headers = _DOC_CONFIGS[language]["headers"]
    lines = [
        "<!-- This table is generated; do not edit this block manually. -->",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
    ]
    for env_id in sorted(registered_envs):
        poster = f'<img src="{POSTER_SRC_PREFIX}/{env_id}.jpg" alt="{env_id}" width="240">'
        description = registered_envs[env_id]["description"][language]
        algorithms = training_algorithms.get(env_id, ())
        algorithm_cell = ", ".join(f"`{algorithm}`" for algorithm in algorithms) or "—"
        lines.append(f"| {poster} | `{env_id}` | {_markdown_cell(description)} | {algorithm_cell} |")
    return "\n".join(lines)


def render_env_table(
    language: str,
    *,
    registered_envs: dict[str, dict[str, object]] | None = None,
    task_dir: Path = TASK_DIR,
    poster_dir: Path = POSTER_DIR,
) -> str:
    """Render a localized Markdown table from current registry and Task data."""
    if language not in _DOC_CONFIGS:
        raise ValueError(f"Unsupported documentation language: {language}")
    if registered_envs is None:
        registered_envs = registry.list_registered_envs()
    training_algorithms = discover_training_algorithms(task_dir)
    _validate_inputs(registered_envs, training_algorithms, poster_dir)
    return _render_env_table(language, registered_envs, training_algorithms)


def _replace_generated_table(content: str, table: str, path: Path) -> str:
    if content.count(START_MARKER) != 1 or content.count(END_MARKER) != 1:
        raise RuntimeError(f"{path} must contain exactly one generated environment table marker pair")
    if content.index(START_MARKER) > content.index(END_MARKER):
        raise RuntimeError(f"{path} has environment table markers in the wrong order")
    prefix, remainder = content.split(START_MARKER, maxsplit=1)
    _, suffix = remainder.split(END_MARKER, maxsplit=1)
    return f"{prefix}{START_MARKER}\n\n{table}\n\n{END_MARKER}{suffix}"


def generate_tables(*, check: bool) -> list[Path]:
    """Update environment tables, or return stale documents without changing them."""
    registered_envs = registry.list_registered_envs()
    training_algorithms = discover_training_algorithms()
    _validate_inputs(registered_envs, training_algorithms, POSTER_DIR)

    stale = []
    for language, doc_config in _DOC_CONFIGS.items():
        path = doc_config["path"]
        content = path.read_text(encoding="utf-8")
        table = _render_env_table(language, registered_envs, training_algorithms)
        generated = _replace_generated_table(content, table, path)
        if generated == content:
            continue
        stale.append(path)
        if not check:
            path.write_text(generated, encoding="utf-8")
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated environment documentation is stale")
    args = parser.parse_args()

    stale_docs = generate_tables(check=args.check)
    if args.check and stale_docs:
        for path in stale_docs:
            print(f"stale generated environment table: {path.relative_to(REPO_ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
