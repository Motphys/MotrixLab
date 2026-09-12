# MotrixLab Task Environment Documentation Standard

## Contents

1. Evidence and scope
2. Classify the task and choose the architecture
3. Single-page task documentation
4. Overview content
5. Task environment design
6. Configuration tuning
7. Extension guide for reusable task families
8. Media and training evidence
9. Language and formatting
10. Validation and acceptance

## 1. Evidence and scope

Write from the current checkout. Before drafting, identify:

- registered Env IDs and their environment/config providers;
- shared environment class versus model-, terrain-, and preset-specific factories, when these variants exist;
- action construction and actuator semantics;
- actor and critic observation construction;
- every raw reward term, configured scale, timestep factor, and curriculum multiplier;
- failure termination versus time-limit truncation;
- deterministic reset operations and every sampled quantity;
- algorithm task configs, user commands, media, plots, and tests.

State the semantic owner of a value when confusion is likely. For example, a model config may own physical assets and limits,
while a task config owns commands, observations, rewards, and episode behavior. Integrate this boundary into the page
introduction instead of repeating it as a standalone chapter.

Do not infer runtime switching or a public extension workflow merely because several Env IDs use one shared class. Each Env ID
selects a registered full config unless the runtime explicitly implements otherwise.

## 2. Classify the task and choose the architecture

Use the existing `docs/source/{zh_CN,en}/user_guide/envs/` hierarchy. Classify the task before creating pages:

| Task class           | Indicators                                                                                   | Default structure                                           |
| -------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Simple task          | One implementation or preset, few meaningful knobs, no supported extension contract          | One page                                                    |
| Configurable task    | Runtime semantics or stable configuration fields need substantial explanation                | Overview plus only the necessary design or tuning page      |
| Reusable task family | A shared contract intentionally supports multiple models, assets, terrains, or task variants | Overview plus justified design, tuning, and extension pages |

Classification is about public user workflow, not code size. Several internal classes do not make a task family, and several
registered Env IDs do not prove that users can extend it. Prefer the smallest structure that answers real user questions.

Before adding a page, state the question it answers. If that question can be answered clearly in a short section on the main
page, do not create the page. Never add empty placeholder sections to satisfy this standard.

## 3. Single-page task documentation

Use a single page by default for a simple task. A practical order is:

1. task purpose and physical objective;
2. available Env ID and copyable view/train/play commands;
3. concise action and observation contract;
4. reward, termination, and reset behavior;
5. meaningful configuration knobs, media, or training evidence, only when present.

The runtime contract may use short subsections, a compact table, or prose. Do not create separate `env_design`,
`config_tuning`, or extension pages merely because those concepts exist in the implementation. A one-term reward or one
termination condition usually needs a sentence or list item rather than a table.

## 4. Overview content

Recommended order:

1. concise purpose and reuse boundary;
2. representative video, poster, or screenshot when available;
3. training curves only when backed by current artifacts;
4. hidden toctree only when supporting pages exist;
5. built-in model/preset/Env ID table only when there are useful variants to compare;
6. copyable view, train, and play commands.

The opening should define the task in physical terms. Name command frames and components, such as body-frame
`[vx, vy, yaw_rate]`, rather than saying only "velocity tracking."

For built-in variants, distinguish shared task logic from differences such as model, terrain, spawn range, and training
backend. Do not copy the complete reward or config reference into the overview.

### Training curves

Title the section directly, for example `训练曲线` / `Training curves`. State:

- training algorithm and synchronous/asynchronous mode;
- axes and metrics;
- evidence-supported convergence range;
- curriculum or scaling behavior needed to interpret the curve.

Use standard MyST captions:

````markdown
```{figure} /_static/images/performance/<env-id>.svg
:alt: <meaningful alt text>
:width: 100%
:align: center

<Robot or Env ID> 训练曲线
```
````

The directive body is the caption. `:caption:` is not the figure-caption syntax.

## 5. Task environment design

This may be a section on a simple task page or a separate page when the contract is substantial. Use a short introduction
followed by only relevant runtime contracts. The usual order is:

1. action space;
2. observation space;
3. reward design;
4. termination conditions;
5. reset logic.

Do not add lifecycle, configuration structure, or implementation-file inventories by default.

### Action space

Document:

- space type and shape;
- one dimension's semantic target;
- conversion from policy action to physical control;
- scale, offset, clipping, and bounds;
- where physical limits originate.

If bounds are computed per actuator, show the actual formula. Do not simplify a per-joint `Box` to `[-1, 1]` unless that is
the runtime contract. State whether bounds are symmetric and whether the default pose must be centered.

### Observation space

Use a table when there are repeated fields worth comparing:

| Observation | Actor | Critic | Meaning |
| ----------- | ----: | -----: | ------- |

Include frame, units or scale, dimensions, privileged inputs, noise, stacking/history, and a dimension formula when useful.
Do not imply actor access to critic-only state.

### Reward design

Introduce shared reference or task behavior before the table when it changes reward interpretation. When multiple reward
terms warrant a table, use exactly these columns:

| Reward term | Computation | Design purpose |
| ----------- | ----------- | -------------- |

Give one row per implemented raw reward term. For a single obvious reward term, concise prose is sufficient. Describe the real
measure or kernel and why the behavior is useful. After the table, explain the complete weighting order, including configured
scale, control timestep, curriculum multiplier, and logging keys. Identify constraint terms whose non-negative raw
measurements become penalties through negative weights.

Avoid grouping several terms into one unexplained category. Avoid calling return decline a regression when a rising penalty
scale changes the reward definition.

### Termination conditions

When multiple conditions warrant a table, use exactly these columns:

| Type | Status | Condition | Meaning |
| ---- | ------ | --------- | ------- |

Separate failure termination (`terminated`) from time-limit truncation (`truncated`). State the configured selection rule for
contacts or thresholds. Do not describe a time limit as a fall.

### Reset logic and randomization

First describe deterministic reset state: pose, velocity, action/history buffers, commands, solver/model state, and kinematic
refresh as applicable.

When several stochastic behaviors warrant a table, use exactly these columns:

| Randomized quantity | Sampling | Timing | Purpose |
| ------------------- | -------- | ------ | ------- |

Include randomization that occurs after reset too, such as periodic command resampling or per-observation noise, but make its
timing explicit. Distinguish:

- task randomization: commands, goals, motions;
- initial-state randomization: pose, velocity, phase, spawn position;
- observation/action randomization: noise, latency, disturbances;
- physical domain randomization: mass, inertia, friction, actuator gains, delays, sensor parameters.

If no physical parameters are randomized, say so directly. Do not label command sampling or observation noise as physical
domain randomization.

## 6. Configuration tuning

Create a separate tuning page only when the task exposes enough stable, meaningful configuration to justify one. Otherwise,
explain the few useful fields on the main page. Start from the actual public top-level config entrypoint and show direct
construction with only meaningful subconfigs. Organize the rest by user-facing effects, such as control, commands,
normalization, references, rewards, curriculum, terrain, and spawn.

Do not teach `dataclasses.replace` by default. It is a Python copying mechanism, not the task contract.

For normalization, explain:

$$
x_{obs}=scale\cdot x_{raw}
$$

State that scaling changes network input magnitude, not simulation state, commands, or reward. Explain that larger scales
amplify a component and smaller scales attenuate it; typical observation components should remain in comparable numerical
ranges. State whether noise is applied before or after scaling and whether critic inputs remain noise-free.

For every config field, explain effect, unit/shape, safe tuning direction, and coupled fields. Keep actuator effort and position
limits with the model configuration when it is the physical source of truth.

## 7. Extension guide for reusable task families

Do not create an extension guide for a simple or merely configurable task. Add one only when all of these are true:

- the environment exposes a stable public extension contract;
- the shared implementation is intentionally reusable across multiple models, assets, terrains, or task variants;
- users are expected to add such variants without rewriting the environment.

Name the guide after the real extension unit, such as `Adding a robot`, `Adding an object preset`, or `Adding a task variant`.
Do not assume every task extends through a robot.

Begin with the top-level environment config factory, not asset preparation in isolation. A typical sequence, when applicable,
is:

1. define the complete environment config;
2. place the model or task asset in the scene and provide its initial state;
3. map task-required model elements;
4. register matching environment config and implementation IDs;
5. add algorithm task configs;
6. verify view, train, and play flows.

Use current class and field names. Do not preserve obsolete aliases in examples. Explain only extension contracts that users
must satisfy and that are not already documented by the underlying model or asset.

## 8. Media and training evidence

Reuse existing assets and scripts. Keep source-versus-generated ownership explicit:

- posters: `docs/source/_static/images/poster/`;
- performance figures: `docs/source/_static/images/performance/`;
- videos: `docs/source/_static/videos/`.

Generate videos and posters with the dedicated `docs/scripts/` tools instead of ad-hoc recording:

```bash
# Doc video -> docs/source/_static/videos/<env-id>.mp4 (4x4 grid of 16 envs, 30 fps, 10 s)
python docs/scripts/generate_video.py <env-id> --force

# Poster -> docs/source/_static/images/poster/<env-id>.jpg (headless snapshot, DirectEnv and ManagerEnv)
python docs/scripts/generate_env_snapshot.py --env <env-id> --force
```

Both tools render headless with the environment's configured system camera. Before embedding, verify the camera framing by
inspecting the generated image or an extracted video frame: the full scene must be clearly visible, and for the multi-env
grid every environment must fit inside the frame. A close-up `SystemCameraCfg` that shows a single environment is wrong for
grid media — set `lookat` to the grid center and pick a `distance` of roughly 4.5–6.0 for a 4x4 grid in the environment
config, then regenerate the poster and the video together. Never embed media you have not visually inspected, and never
link a nonexistent placeholder asset; use a MyST note/admonition when media is planned but absent. Verify final
Sphinx-relative paths after language files are copied into the build source.

Preview videos use the sphinxcontrib-video `{video}` directive with a `:poster:`, matching nearby environment pages;
static images use the `figure` directive with the caption in the directive body.

Performance prose must match plot/log evidence. Record elapsed wall time only if the artifact encodes it. When a curriculum
raises penalty weight, explain why episode return can decrease after behavioral convergence.

## 9. Language and formatting

- Use direct positive technical wording.
- Prefer `动作空间`, `观察空间`, `奖励设计`, `终止条件`, and `重置逻辑` in Chinese task-design pages.
- Keep code identifiers, units, Env IDs, and info keys exact.
- Use one concept per table column. Split `条件` from `含义`, and `生效时机` from `作用`.
- Use tables for exact repeated mappings; use prose for boundaries, caveats, and interpretation.
- Do not create a table when a sentence or short list is clearer.
- Keep paragraphs short around wide tables.
- Avoid duplicating the same contract across overview, design, tuning, and extension pages.

## 10. Validation and acceptance

Run:

```bash
git diff --check -- <changed-docs>
.venv/bin/sphinx-build -W --keep-going -b html docs/source <zh-output-dir>
.venv/bin/sphinx-build -W --keep-going -D language=en -b html docs/source <en-output-dir>
```

When environment registration or generated overview content changes, run:

```bash
.venv/bin/python docs/scripts/generate_env_docs.py --check
```

Inspect the rendered target page. Acceptance requires:

- Chinese and English structures match;
- navigation title and heading hierarchy are correct;
- tables render with the intended number of columns;
- equations and figure captions render correctly;
- internal links and static assets resolve;
- all behavioral claims are source-backed;
- strict builds finish successfully, or unrelated failures are reported without being attributed to the doc change.
