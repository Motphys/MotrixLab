---
name: write-motrixlab-task-docs
description: Write, restructure, or review bilingual Sphinx/MyST user documentation for MotrixLab task environments, from simple single-page tasks to reusable task families. Use when documenting a registered environment, its runtime contract, configuration, presets, training evidence, or an implemented extension workflow under docs/source/{zh_CN,en}/user_guide/envs/.
---

# Write MotrixLab Task Docs

Create user-facing task-environment documentation from current repository evidence. Keep Chinese and English pages aligned,
state exact runtime semantics, and validate the rendered Sphinx output.

## Read the standard

Read [references/writing-standard.md](references/writing-standard.md) completely before drafting or restructuring a task
environment page. Apply only sections supported by the target environment; do not add empty boilerplate.

## Establish evidence

1. Read repository instructions, the current documentation page, and the closest maintained task page.
2. Locate the environment registration, config factory, environment implementation, relevant model or scene config, training
   config, tests, and referenced media. Skip surfaces that do not exist for the task.
3. Treat current code and declarative config as authoritative. Use tests as supporting evidence, not as a substitute for the
   implementation.
4. Separate shared environment behavior from model-, preset-, algorithm-, and terrain-specific values when those variants
   exist.
5. Verify every dimension, formula, reward term, reset randomization, termination condition, command, path, and performance
   claim before writing it.

Prefer `rg` and concrete source paths. Useful starting searches include:

```bash
rg -n "registry\.env|registry\.envcfg" motrix_envs/src configs/task
rg -n "action_space|observation_space|reward|terminated|truncated|reset|random" motrix_envs/src
rg -n "<env-id>|<environment-class>" docs motrix_envs/tests
```

## Classify the task first

Choose documentation depth from the implemented reuse boundary and user needs, not from a fixed template:

- **Simple task:** one main implementation or preset, few meaningful knobs, and no supported extension contract. Use one page.
  Keep action, observation, reward, termination, and reset explanations concise on that page.
- **Configurable task:** users need substantial runtime-contract or tuning guidance. Keep an overview and add only the design or
  tuning page that answers a distinct user question.
- **Reusable task family:** one stable environment contract intentionally supports multiple models, assets, terrains, or task
  variants. A multi-page structure may include overview, design, tuning, and extension guidance as justified.

Extend the current navigation rather than inventing a second hierarchy. Do not split a short section into its own page. Do not
create an adding-a-robot, adding-a-model, or adding-a-task page unless the repository implements a stable extension workflow
and users are expected to use it.

Keep model/config responsibility boundaries in a short opening explanation when needed. Do not create a standalone boundary,
lifecycle, or configuration-schema chapter by default.

## Write for users

- Lead with what the task does and what the user can configure or run.
- Use exact public names, Env IDs, config fields, units, shapes, and source-derived formulas.
- Use tables for repeated mappings and comparisons. Prefer concise prose or a list when a simple task has only one or two
  obvious items. Follow the table shapes in the writing standard when a table is warranted.
- Explain why each reward or randomization exists, not only how it is computed.
- Distinguish `terminated` failures from `truncated` time limits.
- Distinguish task, initial-state, and observation randomization from physical domain randomization. State explicitly when
  physical parameters are not randomized.
- Describe only implemented capability. Do not turn a plausible design, unused config, or test fixture into a current claim.
- Avoid internal lifecycle narration, implementation history, compatibility notes, and `dataclasses.replace` examples unless
  the user explicitly asks for them.

## Keep bilingual pages aligned

Edit `docs/source/zh_CN/` and `docs/source/en/` together unless the user explicitly scopes the work to one language. Preserve
technical identifiers across languages and translate meaning rather than sentence structure. Use established terminology from
the neighboring pages.

## Handle media and performance evidence

- Reuse repository assets and Sphinx-relative paths.
- Generate videos and posters with the dedicated scripts under `docs/scripts/`; do not hand-assemble placeholder images or
  record through ad-hoc `play.py` invocations:
  - `python docs/scripts/generate_video.py <env-id> --force` records the doc video to
    `docs/source/_static/videos/<env-id>.mp4` (16 envs rendered as a 4x4 grid, 30 fps, 10 s by default).
  - `python docs/scripts/generate_env_snapshot.py --env <env-id> --force` renders the poster to
    `docs/source/_static/images/poster/<env-id>.jpg`. It supports both DirectEnv and ManagerEnv environments.
- Both scripts render headless with the environment's configured system camera. Verify the camera framing before embedding:
  the poster and video must show the full scene clearly — for the default multi-env grid, the whole grid must be inside the
  frame with every environment identifiable. If the view crops the grid or shows only one environment, fix the
  `SystemCameraCfg` (usually `lookat` at the grid center and a `distance` around 4.5–6.0 for a 4x4 grid) in the environment
  config, regenerate, and inspect the rendered image again before continuing. Re-record both media whenever the camera
  changes.
- Extract and inspect a frame from the recorded video (for example with `imageio`) to confirm framing; do not embed media
  you have not looked at.
- Use the sphinxcontrib-video `{video}` directive with a `:poster:` for preview videos, matching nearby environment pages.
- Use the MyST `figure` directive; place its caption in the directive body, not in a `:caption:` option.
- Add `:alt:`, width, and alignment consistent with nearby pages.
- Describe convergence time, return changes, or curriculum effects only when plots or logs support the claim.
- When a penalty curriculum changes return scale, explain that increasing penalty weight can lower return without implying
  policy degradation.

## Validate

Run checks proportional to the change, at minimum:

```bash
git diff --check -- <changed-docs>
.venv/bin/sphinx-build -W --keep-going -b html docs/source <zh-output-dir>
.venv/bin/sphinx-build -W --keep-going -D language=en -b html docs/source <en-output-dir>
```

Use `uv run --extra docs sphinx-build` if the repository virtual environment is unavailable. If registration or generated
environment overview content changed, also run:

```bash
.venv/bin/python docs/scripts/generate_env_docs.py --check
```

## Proofread the finished page against the code

After drafting and before declaring the work done, re-read the finished page and re-derive every quantitative claim from
the current implementation — do not trust what you wrote from memory during drafting:

- Recompute observation dimensions by summing the configured observation terms per group (actor and critic separately);
  compare against the stated totals.
- Check the interval semantics of every noise and randomization claim by reading the implementing kernel/config, not by
  assuming a convention: uniform samples and observation noise are one-sided `[0, scale)` in this codebase — never write
  `±` unless the code actually generates a centered, two-sided sample.
- Re-check every weight, threshold, scale, sigma, target, and duration against the config factory values.
- Confirm the Chinese and English pages state identical numbers and structure; fix any drift on both sides in one commit.

Report any mismatch you find as a fixed defect; a proofread that changes nothing must still have actually re-derived the
numbers.

Inspect the final HTML for the changed page, especially wide tables, formulas, captions, navigation titles, and internal
links. Report warnings or unrelated failures precisely; do not claim an unobserved build succeeded.
