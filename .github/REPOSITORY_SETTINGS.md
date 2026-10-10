# GitHub repository settings checklist

These settings cannot be committed as repository files. Apply them in the
GitHub repository after the public repository is created.

## Observed GitHub state

As of 2026-09-07, the public `Motphys/MotrixLab` repository has `main` as its default
branch and allows forks. At the time of this review, the public API showed no
active repository rulesets, no protected branches, no community health files,
and only the Docker publishing workflow. The existing `v0.3.0` release points
at `main`; treat it as historical and create future releases from `stable`.

The new community files and workflows in this checkout take effect only after
they are merged into the public `main` branch.

The repository metadata also has no homepage or topics. After the public
cleanup, set the documentation homepage and add discoverability topics such as
`robotics`, `reinforcement-learning`, `robot-simulation`, `mujoco`, `python`,
and `robot-learning`. Keep the GitHub wiki enabled only if it will be actively
maintained; otherwise make the repository's versioned `wiki/` documentation the
single source of truth.

## Rollout order

1. Publish the cleaned history and these community/CI files to `main`.
2. Create `stable` at the reviewed `main` commit, for example:

   ```bash
   git fetch origin main
   git switch -c stable origin/main
   git push origin stable
   ```

3. Run the new workflows once and confirm their exact check names.
4. Configure the rulesets below, selecting those exact check names as required
   status checks. Configure protections only after the checks exist; otherwise
   GitHub can block every pull request while waiting for a check that has never
   run.
5. Confirm `main` is the default branch, enable Discussions and private
   vulnerability reporting, and review the first external Fork PR end to end.

## Default branch

- Set `main` as the default branch so new pull requests target development
  by default. Keep `stable` as the stable release branch.

## Rulesets and branch protection (`main` and `stable`)

- Prefer repository **Rulesets** so the branch and tag policies are visible and
  can be layered. If Rulesets are unavailable on the plan, use equivalent
  branch protection rules. Protect both long-lived branches.
- Require pull requests, at least one maintainer review, and dismissal of stale
  approvals.
- Require the `Branch policy / validate`, `Tests / pytest (fast)`,
  `Tests / pytest (numba)`, `PR checks / prek`, and `CodeQL / analyze` status
  checks.
- Require approval of all review conversations before merge.
- Require linear history or squash merging; disable merge commits for ordinary
  contributor PRs.
- Require branches to be up to date before merge.
- Block force-pushes and branch deletion.
- Do not allow bypass for normal changes. Handle emergencies through a reviewed
  `hotfix/*` PR; if a ruleset bypass is unavoidable, record the incident and
  immediately follow up with the required hotfix PR.

## Temporary release branch rulesets

- Add branch rulesets targeting `release/**` and `hotfix/*`.
- Block force-pushes and deletion for both patterns.
- Do not require pull requests on the temporary branches themselves. The release
  App must push the validated patch bump to `hotfix/*`, and governance remains
  at the `hotfix/* → stable` review; `release/**` is governed by its PR into
  `main`.
- Restrict direct writes where the ruleset supports actor conditions, but keep
  the release App installation allowed for the automated patch bump.
- Require generated `release/v* → main` version PRs to use squash or rebase so
  the resulting `main` head remains the detectable `chore(release):` commit. If
  that trigger is missed, maintainers can use the documented
  `open_release_pr=true` manual fallback.

## Stable branch and releases

- Configure normal pull requests from forks to target `main`.
- Require a release pull request from `main` to `stable`.
- Allow `hotfix/*` stable patch pull requests to target `stable` directly.
- Allow only the upstream maintainer back-merge `stable` → `main` after a
  stable patch release; reject `stable` → `main` PRs from forks.
- Require every stable patch release to be merged back into `main` after
  release.
- Restrict tag creation for `v*` to maintainers and create tags from `stable`.
- Keep PyPI publishing automated from reviewed `stable` release tags, but keep
  Docker publishing as a manual dispatch from `stable`.
- Restrict who can create, update, or delete `v*` tags.
- Add a tag ruleset for `v*` that allows creation only by maintainers and the
  release App, and prevents deletion or updates.
- Remember that Git tags do not record their source branch; the PyPI and Docker
  workflow ancestry checks are second guards, while the tag ruleset and
  maintainer permissions enforce the `stable`-only release process.

## Security and automation

- Enable Dependabot alerts and security updates.
- Enable secret scanning and push protection where available.
- Enable private vulnerability reporting.
- Require approval before running workflows from first-time or untrusted fork
  contributors, and review workflow changes carefully.
- Review the permissions of every GitHub Actions workflow.
- Pin third-party Actions to full commit SHAs and let Dependabot update those
  references. The workflows in this repository should retain a version comment
  beside each SHA.
- Configure a real maintainer/team in `.github/CODEOWNERS` before requiring
  code-owner approval.

## Releases and collaboration

- Enable Issues and Discussions; pin the contribution and support documents.
- Keep Issues for actionable bugs/features and Discussions for questions,
  support, and open-ended design discussion.
- Create a release from a reviewed tag only after the public dependency and
  third-party asset audit is complete.
- Configure the package/container publishing secrets only in the main
  repository, never in forks.
- Create a release GitHub App, install it only on this repository with Contents
  read/write and Pull requests read/write, and store its client ID/private key as
  `RELEASE_APP_CLIENT_ID` / `RELEASE_APP_PRIVATE_KEY`. The workflow exchanges
  those credentials for short-lived installation tokens; generated PRs, tags,
  and GitHub Releases therefore trigger the downstream workflows and required
  checks normally.
- Check Git LFS bandwidth/storage usage before publishing large releases.

GitHub references: [about rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets),
[protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches),
and [creating a pull request from a fork](https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request-from-a-fork).
