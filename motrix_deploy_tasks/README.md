# Motrix Deploy Tasks

`motrix-deploy-tasks` supplies the versioned Go2 walking task, flat/rough deployment worlds,
and installed Hydra recipes for `motrix-deploy`. It does not provide a simulation or hardware backend.

Install a backend plugin and use an artifact exported for the matching task:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

Use `go2-walk-rough/sim` for the rough scene, `runtime.backend=motrixsim` for native MotrixSim,
or `go2-walk-flat/hardware` for Unitree hardware. Recipes are packaged with the plugin;
no workspace config path is required.

See the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for export, configuration, and hardware safety procedures.
