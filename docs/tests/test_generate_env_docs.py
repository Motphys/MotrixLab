# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_env_docs.py"
MODULE_SPEC = importlib.util.spec_from_file_location("generate_env_docs", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
generate_env_docs = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(generate_env_docs)


def _registered_env(en: str = "English description.", zh_cn: str = "中文描述。") -> dict[str, object]:
    return {
        "config_class": "DemoEnvCfg",
        "description": {"en": en, "zh_CN": zh_cn},
    }


def test_discover_training_algorithms_merges_backend_variants(tmp_path):
    env_dir = tmp_path / "demo-env"
    env_dir.mkdir()
    for filename in ("skrl.ppo.yaml", "skrl.ppo.jax.yaml", "skrl.ppo.torch.yaml", "motrix.fastsac.yaml"):
        (env_dir / filename).write_text("", encoding="utf-8")

    assert generate_env_docs.discover_training_algorithms(tmp_path) == {"demo-env": ("motrix.fastsac", "skrl.ppo")}


def test_discover_training_algorithms_rejects_invalid_option_name(tmp_path):
    env_dir = tmp_path / "demo-env"
    env_dir.mkdir()
    (env_dir / "skrl.yaml").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"expected <rllib>\.<algo>"):
        generate_env_docs.discover_training_algorithms(tmp_path)


def test_render_env_table_localizes_and_marks_env_without_tasks(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    env_dir = task_dir / "with-task"
    env_dir.mkdir()
    (env_dir / "skrl.ppo.jax.yaml").write_text("", encoding="utf-8")
    registered_envs = {
        "with-task": _registered_env(),
        "without-task": _registered_env(),
    }
    poster_dir = tmp_path / "posters"
    poster_dir.mkdir()
    for env_id in registered_envs:
        (poster_dir / f"{env_id}.jpg").write_bytes(b"poster")

    table = generate_env_docs.render_env_table(
        "zh_CN",
        registered_envs=registered_envs,
        task_dir=task_dir,
        poster_dir=poster_dir,
    )

    assert "| 预览图 | Env ID | 描述 | 训练算法 |" in table
    assert (
        '| <img src="../../_static/images/poster/with-task.jpg" alt="with-task" width="240"> '
        "| `with-task` | 中文描述。 | `skrl.ppo` |"
    ) in table
    assert (
        '| <img src="../../_static/images/poster/without-task.jpg" alt="without-task" width="240"> '
        "| `without-task` | 中文描述。 | — |"
    ) in table


def test_render_env_table_validates_docstrings_and_task_envs(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    unknown_task = task_dir / "unknown-env"
    unknown_task.mkdir()
    (unknown_task / "skrl.ppo.yaml").write_text("", encoding="utf-8")
    poster_dir = tmp_path / "posters"
    poster_dir.mkdir()
    (poster_dir / "documented-env.jpg").write_bytes(b"poster")
    (poster_dir / "unknown-env.jpg").write_bytes(b"poster")

    with pytest.raises(RuntimeError, match="unregistered environments"):
        generate_env_docs.render_env_table(
            "en",
            registered_envs={"documented-env": _registered_env()},
            task_dir=task_dir,
            poster_dir=poster_dir,
        )

    with pytest.raises(RuntimeError, match="missing docstring descriptions"):
        generate_env_docs.render_env_table(
            "en",
            registered_envs={"unknown-env": _registered_env(zh_cn="")},
            task_dir=task_dir,
            poster_dir=poster_dir,
        )


def test_render_env_table_rejects_missing_poster(tmp_path):
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    env_dir = task_dir / "demo-env"
    env_dir.mkdir()
    (env_dir / "skrl.ppo.yaml").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing poster images.*demo-env"):
        generate_env_docs.render_env_table(
            "en",
            registered_envs={"demo-env": _registered_env()},
            task_dir=task_dir,
            poster_dir=tmp_path / "posters",
        )


def test_replace_generated_table_preserves_manual_content_and_requires_one_marker_pair():
    path = Path("index.md")
    content = f"before\n{generate_env_docs.START_MARKER}\nold\n{generate_env_docs.END_MARKER}\nafter\n"

    replaced = generate_env_docs._replace_generated_table(content, "new table", path)

    assert replaced == (
        f"before\n{generate_env_docs.START_MARKER}\n\nnew table\n\n{generate_env_docs.END_MARKER}\nafter\n"
    )
    with pytest.raises(RuntimeError, match="exactly one"):
        generate_env_docs._replace_generated_table("no markers", "new table", path)
