# GitHub Actions Workflows

This directory contains GitHub Actions workflows for automating CI/CD processes.

## Pull Request Checks

### File: `pr-checks.yml`

Runs the repository's configured checks when a pull request is opened, updated, reopened, or marked ready for review.
When new commits are pushed to the same pull request, GitHub cancels the older in-progress run.

The workflow checks out the full Git history, sets up uv, and runs prek 0.5.2. The hooks configured in
`.pre-commit-config.yaml` maintain Apache 2.0 license headers with Copywrite, lint and format Python with Ruff, and format
supported documentation and configuration files with the `dprint-py` development dependency through uv. It needs only read
access to repository contents and uses no repository secrets.

### File: `tests.yml`

Runs on pull requests, pushes to `main` or `stable`, manual dispatch, and daily at 02:00 UTC
(10:00 China Standard Time). Scheduled runs use the default branch. Each job installs the uv
workspace with Git LFS assets on Linux, the supported public CI environment.

PR and push runs have two independent, parallel jobs:

- **fast**: `pytest -m "not slow and not integration and not numba"`.
- **numba**: `pytest -m "not integration and (slow or numba)"`. This includes the two expensive
  WBT kernel regressions on every PR, so failures are not deferred until the next day.
  The entire `test_wbt_numba.py` module stays together: excluding only the first two expensive
  tests shifts first-use compilation into later tests rather than making the fast suite cheap.

These selections partition all non-integration tests, including future `slow` tests. They run on
all PRs rather than relying on path filters that might miss a transitive dependency change.
A failure in either job does not cancel the other job. Both should be required checks for merging;
replace any branch-protection requirement for the old `pytest` job with these two checks.

Nightly and manual dispatch run the **full** suite, including
`test/test_all_envs.py::test_all_demos`. The full-environment subprocess smoke test is excluded
from PR/push jobs. Each job reports the 20 slowest phases and uploads JUnit results, including on
failure. Numba caches have per-suite write keys to avoid parallel jobs competing for one cache.

Local commands (activate `.venv` first):

```bash
# Quick feedback; excludes expensive kernel regressions and full-environment smoke.
python -m pytest -m "not slow and not integration and not numba"
# Dedicated compilation/runtime regressions.
python -m pytest -m "not integration and (slow or numba)"
# Only the full-environment integration tests.
python -m pytest -m integration
# Full coverage remains the default; no tests are silently excluded by addopts.
python -m pytest
```

Markers are registered in the root `pyproject.toml`: `slow` describes cost, `numba` identifies
specialized kernel regressions, and `integration` identifies tests reserved for full runs.
Not every test importing Numba needs the `numba` marker: inexpensive core contract tests remain
in the fast suite.

For new Numba environments, reuse small synthetic-manager tests for shared runtime contracts,
keep task-specific smoke tests minimal, and retain a small set of representative real-environment
regressions. The existing nightly smoke test automatically covers all registered environments.
Do not duplicate a full expensive rollout for every preset. If the dedicated job grows too large,
measure `--durations` and shard by environment family; do not silently drop kernel coverage from PRs.

#### Diagnosing Numba startup costs

Compare a fresh `NUMBA_CACHE_DIR` with a second process using the same directory, then compare
another environment in the same process. `NUMBA_DEBUG_CACHE=1` reports actual specialization
cache loads/saves. Manager INFO logs time each evaluate/observe/reset specialization; DEBUG
logs include signatures. Dispatcher preparation is not the actual compilation phase.

Precompilation uses the real input types, and warmup copies preserve strides and readonly flags,
so warmup and reset/step share signatures. Function fingerprints use explicit code fields rather
than marshal serialization, which can change after Numba inspects an unchanged code object.
Keep these invariants intact when extending the compiler: a warmup must not introduce new
signatures, and constructing the same configuration after execution must reuse the same plan.

A local G1 WBT play-mode benchmark (`num_envs=2`, init and one step; interpreter imports excluded)
measured about 29 seconds with an empty cache, 0.27 seconds in a second process with disk cache,
and 0.07 seconds for another environment using the in-process cache after these fixes.
The original cold benchmark took about 57 seconds and compiled each kernel twice.
These are diagnostic measurements, not CI timing thresholds; hardware and cache state vary.

### File: `codeql.yml`

Runs GitHub CodeQL analysis for Python on pushes, pull requests, and a weekly
scheduled scan. Review results under the repository's **Security** tab.

### File: `branch-policy.yml`

Checks pull request metadata without checking out contributor code. Normal PRs
must target `main`; a post-hotfix back-merge may use upstream `stable → main`.
Only a release PR from the upstream `main` branch or a `hotfix/*` PR may target
`stable`. Add `Branch policy / validate` as a required status check on both
protected branches.

## Docker Image Build Workflow

### File: `docker-build.yml`

Automatically builds and pushes Docker images to Docker Hub when a new version
tag pointing to the `stable` release branch is pushed.

#### Trigger Conditions

The workflow is triggered when you push a git tag that matches the pattern `v*`.
The job then verifies that the tagged commit is reachable from `stable`:

```bash
git switch stable
git pull --ff-only upstream stable
git tag -a v0.3.0 -m "Release v0.3.0"
git push upstream v0.3.0
```

Tags whose commit is not reachable from `stable` are rejected by the
verification step and are not published. Git tags do not retain the branch on
which they were created, so the tag ruleset and maintainer-only permissions
remain the authoritative control for requiring that tags are created from
`stable`.

#### What It Does

1. **Extracts Version**: Reads the version from `pyproject.toml` (currently `0.3.0`)
2. **Builds Docker Image**: Uses the Dockerfile in `docker/Dockerfile`
3. **Pushes Multiple Tags**:
   - `motphys/motrixlab:v0.3.0` (version tag)
   - `motphys/motrixlab:v0.3` (major.minor version tag)
   - `motphys/motrixlab:0.3.0` (version from `pyproject.toml`)
   - `motphys/motrixlab:latest` (always points to the latest version)

#### Required Secrets

You need to configure the following secrets in your GitHub repository settings:

1. **`DOCKER_USERNAME`**: Your Docker Hub username
2. **`DOCKER_PASSWORD`**: Your Docker Hub password or access token

To add secrets:

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the secrets listed above

#### Usage Example

```bash
# 1. Update version in pyproject.toml if needed
# 2. Commit your changes
git add .
git commit -m "Release v0.3.0"

# 3. From the upstream stable commit, create and push an annotated version tag
git switch stable
git pull --ff-only upstream stable
git tag -a v0.3.0 -m "Release v0.3.0"
git push upstream v0.3.0

# 4. The workflow will automatically build and push the Docker image
# 5. Monitor the build at: https://github.com/Motphys/MotrixLab/actions
```

#### Built Image Tags

After the workflow completes, the following Docker images will be available:

```bash
# Pull the latest version
docker pull motphys/motrixlab:latest

# Pull a specific version
docker pull motphys/motrixlab:0.3.0

# Pull the major.minor version tag
docker pull motphys/motrixlab:v0.3
```

#### Workflow Features

- ✅ **Optimized Caching**: Uses GitHub Actions cache to speed up builds
- ✅ **Multi-tag Support**: Automatically tags with version, major.minor, and latest
- ✅ **Version Extraction**: Automatically reads version from pyproject.toml
- ✅ **Docker Layer Caching**: Uses UV cache mounts for faster dependency installation
- ✅ **Tag-based Trigger**: Only builds on version tags, not on every commit

#### Docker Image Contents

The resulting Docker image includes:

- Base: NVIDIA CUDA 12.8.1 Runtime + Ubuntu 24.04
- UV package manager
- MotrixLab workspace packages
- SKRL dependencies for both JAX and PyTorch backends
- TensorBoard
- MotrixSim physics engine

#### Testing the Docker Image Locally

Before tagging a release, you can test the Docker build locally:

```bash
cd docker
docker build -t motphys/motrixlab:test .
docker run --gpus all motphys/motrixlab:test scripts/view.py env=cartpole
```

#### Troubleshooting

**Build fails with authentication error:**

- Verify Docker Hub credentials are correctly set in GitHub secrets
- Ensure your Docker Hub account has permission to push to the `motphys/motrixlab` repository

**Version extraction fails:**

- Ensure `pyproject.toml` has a valid `version = "x.y.z"` line
- Check the workflow logs for the exact extraction command output

**Tag not triggering the workflow:**

- Ensure the tag starts with `v` (e.g., `v0.3.0`, not `0.3.0`)
- Confirm that the tag points to a commit on `stable`, then push it from the
  upstream repository (for example, `git push upstream v0.3.0`)

#### See Also

- [Docker Hub Repository](https://hub.docker.com/r/motphys/motrixlab)
- [Container Deployment Documentation](../../docs/source/zh_CN/user_guide/getting_started/container_deployment.md)
- [Dockerfile](../../docker/Dockerfile)
