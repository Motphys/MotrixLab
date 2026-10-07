# Motrix Deploy MuJoCo

`motrix-deploy-mujoco` provides the MuJoCo simulation backend for `motrix-deploy`.
It compiles a complete `SceneCfg`, derives the robot contract, and owns physics stepping,
joint control, and an optional interactive viewer.

With `motrix-deploy-tasks` and a matching exported artifact:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

For programmatic use, see [deploy_to_sim.py](../examples/deploy_to_sim.py).
Configuration and deployment workflows are covered in the
[deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md).
