## Target branch

- [ ] This is a normal PR from a fork targeting upstream `main`.
- [ ] This is a `hotfix/*` PR from a fork targeting upstream `stable`.
- [ ] This is a maintainer release PR from `main` to `stable`.
- [ ] This is a maintainer post-hotfix back-merge from upstream `stable` to
      `main`.
- [ ] I selected `Motphys/MotrixLab` as the base repository and the correct base
      branch in GitHub's compare-across-forks UI.
- [ ] For a fork PR, I synchronized my topic branch with the latest upstream
      target branch before opening or updating this PR.

## Summary

<!-- What changed and why? -->

## Issue

<!-- Link the issue, discussion, or design document. Use `Fixes #123` when the
     PR should close an issue automatically. -->

## Validation

- [ ] `python -m pytest -q`
- [ ] `prek run --all-files`
- [ ] CI checks pass on the target branch
- [ ] Documentation updated (if applicable)
- [ ] Third-party attribution updated (if applicable)
- [ ] No credentials, secrets, private URLs, or sensitive robot data are
      included in the commits, logs, or screenshots.

## Compatibility and migration

<!-- Describe breaking changes, configuration changes, supported platforms, or
     migration steps. Write "None" when not applicable. -->

## Safety and release impact

<!-- Mention platform/backend changes, performance impact, migration steps,
     asset/license changes, hardware testing, and safety limitations. -->
