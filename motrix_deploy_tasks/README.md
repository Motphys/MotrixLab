# Motrix Deploy Tasks

`motrix_deploy_tasks` contains concrete, versioned task implementations. It publishes the `motrix-deploy` executable as a
thin bootstrap that imports these tasks before delegating to `motrix_deploy.cli`.
In the MotrixLab workspace, the bootstrap selects `configs/deploy/sim2sim/go2_walk_sim2sim.yaml` for `sim2sim` and
`configs/deploy/sim2real/go2_walk_flat_sim2real.yaml` for `sim2real`. Outside the workspace,
pass an explicit Hydra `--config-path` and `--config-name`.

The default sim2sim recipe targets `go2-walk-rough`; the default sim2real recipe targets `go2-walk-flat`. From the workspace root, select the flat-terrain recipe while retaining the
workspace config path with:

```bash
motrix-deploy sim2sim \
  --config-name go2_walk_flat_sim2sim \
  artifact=artifacts/go2-walk-flat.deploy
```

The package currently provides the `go2_walk/v1` task used by the Go2 flat and rough walking environments.
Importing `motrix_deploy_tasks` registers its concrete implementations with the registry in `motrix_deploy.task`. The
artifact stores this versioned task name; the runtime resolves it to
`motrix_deploy_tasks.go2_walk.Go2WalkDeployTaskV1` for both sim2sim and sim2real. Deployment recipes select the backend and
input binding, while the artifact remains the authority for task selection.
The environment-to-artifact profile compiler is owned by `motrix_envs.deploy`, alongside the source environment config.
