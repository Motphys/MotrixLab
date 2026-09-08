# Contributing to MotrixLab

Thank you for contributing. Please use GitHub Issues for reproducible bugs and
feature proposals, and Discussions for usage questions and design discussion.

## Development environment

MotrixLab currently targets:

- Python **3.10.x** (the workspace declares `==3.10.*`).
- Linux x86_64 and Windows x86_64 for the documented CPU simulation and viewer
  workflows. The JAX training extra is Linux-only.
- [uv](https://docs.astral.sh/uv/) for environments, dependency resolution,
  and workspace commands.
- Git LFS for meshes, motion data, images, and videos.

Install Git LFS before cloning and fetch the assets needed by the tests:

```bash
git lfs install
git clone https://github.com/Motphys/MotrixLab.git
cd MotrixLab
git lfs pull
```

Create the complete development environment from the repository root:

```bash
uv sync --all-packages --all-groups --all-extras
```

Optional training backends can be installed separately when a full environment
is unnecessary:

```bash
uv sync --all-packages --extra skrl-torch
uv sync --all-packages --extra skrl-jax   # Linux only
uv sync --all-packages --extra rslrl
```

The workspace contains nine packages. Package-local changes should use the
smallest required extra; changes involving the simulator, built-in assets, or
training integrations should be tested with the corresponding package and
extra enabled.

## Branch strategy

MotrixLab uses two long-lived branches:

- `main` is the development and integration branch. All normal feature,
  bug-fix, refactor, documentation, test, and CI pull requests target this
  branch.
- `stable` is the stable release branch. Only reviewed changes merged from
  `main`, or an emergency hotfix, may enter it. Release tags and GitHub
  Releases are created from `stable` only.

Do not push directly to either long-lived branch. External contributors must
use a fork. Maintainers may use the upstream repository for administrative
changes, but the release PR itself must originate from upstream `main` and
target `stable`.

### Normal contribution (Fork → `main`)

Create a topic branch from the latest upstream `main`:

```bash
git clone https://github.com/<your-user>/MotrixLab.git
cd MotrixLab
git remote add upstream https://github.com/Motphys/MotrixLab.git
git fetch upstream
git switch main
git pull --ff-only upstream main
git switch -c fix/123-reset-seed
```

Run the sync commands with a clean working tree. If you have local changes,
commit them on a topic branch or stash them before updating from upstream.

Push the branch to your fork and open the PR in the GitHub web UI:

```bash
git push -u origin fix/123-reset-seed
```

In the compare-across-forks view, select **Motphys/MotrixLab** as the base
repository, `main` as the base branch, your fork as the head repository, and
your topic branch as the compare branch. This is the required target for every
normal feature, bug-fix, documentation, test, and CI PR.

### Hotfix (Fork → `stable`)

Use this path only for an urgent fix that must ship before the next normal
release. Start from the latest upstream `stable`:

```bash
git fetch upstream
git switch -c stable --track upstream/stable
git switch -c hotfix/0.3.1-critical-fix
```

If your fork already has a local `stable` branch, update it instead:

```bash
git switch stable
git pull --ff-only upstream stable
git switch -c hotfix/0.3.1-critical-fix
```

Open the PR with **Motphys/MotrixLab** as the base repository, `stable` as the
base branch, and the hotfix branch in your fork as the compare branch. After a
hotfix is merged and released, a maintainer must open a back-merge PR from
upstream `stable` to `main` and merge it before normal development continues.
This is the only allowed `stable → main` PR.

Before starting new work, sync your fork with upstream; do not build a new PR
on top of stale `main` or `stable`. Delete the fork branch after merge. Keep
each PR focused on one logical change and update it when the target branch
moves.

You may enable **Allow edits from maintainers** for ordinary source-only
changes so maintainers can fix small issues on the PR branch. Do not enable it
when the fork branch contains workflow changes or sensitive material. Review
GitHub's warning carefully because this grants maintainers write access to the
fork branch.

Use one of these prefixes, followed by a short kebab-case description:

```text
feat/<description>       # new user-visible capability
fix/<description>        # bug fix
refactor/<description>   # behavior-preserving restructuring
perf/<description>       # performance change
docs/<description>       # documentation only
test/<description>       # test-only change
chore/<description>      # tooling, dependencies, or maintenance
ci/<description>         # automation change
hotfix/<description>     # urgent patch branched from stable
```

Include an issue number when one exists. Avoid branch names that contain private
project names, credentials, or experimental data identifiers.

## Commit convention

Use a short imperative subject in the Conventional Commits style:

```text
<type>(optional-scope): imperative summary
```

Allowed types are `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `chore`,
`build`, `ci`, and `revert`. Keep the subject concise, start it with a
lowercase type, and add an issue reference at the end when applicable:

```text
fix(deploy): reject an expired artifact (#123)
```

Each commit should be reviewable and buildable where practical. Maintainers
normally use **Squash and merge** for contributor PRs so the target branch has
a concise history; do not rely on a local merge commit to explain the change.
Version bumps, release notes, and tags belong in the
release pull request from `main` to `stable`, not in an individual feature
branch unless the change specifically requires them.

## Tests and checks

Run the full test suite before opening a pull request:

```bash
uv run pytest -q
```

For an iteration on one package, run its tests directly, then run the full
suite before requesting review:

```bash
uv run pytest motrix_env_core/tests -q
uv run pytest motrix_deploy/tests motrix_deploy_mujoco/tests -q
```

Run the repository's formatting, license-header, and lint hooks:

```bash
uv tool install prek==0.5.2
prek install
prek run --all-files
```

The hooks run Copywrite, Ruff, and dprint. For individual checks, use:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

When changing documentation, install the `docs` extra and build with warnings
treated as errors:

```bash
uv sync --all-packages --extra docs
uv run sphinx-build -W -b html docs/source docs/build/html
```

The public CI runs the complete workspace test command on Linux for both
`main` and `stable`. A local test run that cannot install a platform-specific
simulator wheel should be reported in the pull request rather than silently
omitted.

## Pull requests and review

1. For contributor PRs, target `main` for normal work, or target `stable` only
   for a documented `hotfix/*` branch. The maintainer-only `stable → main`
   back-merge is described above.
2. Add or update tests and documentation for behavior changes.
3. Describe compatibility, performance, asset/license, and hardware-safety
   implications in the pull request description when relevant.
4. Include the exact commands used for validation and explain any skipped
   checks.
5. Do not commit credentials, generated training runs, deployment artifacts, or
   unlicensed third-party assets.

Maintainers review code, tests, documentation, dependency changes, and license
implications. A PR should have all required CI checks passing, no unresolved
review conversations, and an approving maintainer review before merge. If a
change is not ready for review, open it as a draft PR. Do not use
`pull_request_target` or execute code from an untrusted fork with write tokens.

## Release flow

1. Merge completed work into `main` through normal pull requests.
2. Open a release pull request from `main` to `stable`, including the
   changelog, version synchronization, dependency lock, and release notes.
3. After it is merged, create an annotated `vMAJOR.MINOR.PATCH` tag on the
   `stable` commit and publish the GitHub Release.
4. If a `hotfix/*` branch is merged directly to `stable`, tag and release it
   from `stable`, then open the maintainer back-merge PR `stable → main`
   immediately.

Never create a release from a contributor fork. Release tags must be created in
the upstream repository by an authorized maintainer after the stable-branch
merge.

For example, an authorized maintainer can publish a release after checking out
the upstream `stable` branch. The commands below assume the upstream repository
is configured as the `upstream` remote; in a direct clone of
`Motphys/MotrixLab`, replace `upstream` with `origin`.

```bash
git fetch upstream
git switch stable
git pull --ff-only upstream stable
git tag -a v0.3.1 -m "Release v0.3.1"
git push upstream v0.3.1
```

Create the GitHub Release from that tag, paste the user-visible entries from
`CHANGELOG.md`, and mark pre-releases explicitly. Do not move or delete a
published version tag.

## GitHub references

The workflow follows GitHub's guidance for [creating a pull request from a
fork](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork),
[syncing a fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/syncing-a-fork),
and [secure use of GitHub Actions](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions).

All source files should retain the Apache-2.0 SPDX header used by this
repository. Keep the nine workspace package versions synchronized when making a
release, and update `THIRD_PARTY_NOTICES.md` whenever a dependency or bundled
asset changes.
