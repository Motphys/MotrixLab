# Motrix Deploy

`motrix-deploy` runs exported policies without a training framework. It provides deployment artifacts,
robot/task contracts, shared control execution, runtime plugin discovery, and the `motrix-deploy` CLI.
Install the task and runtime plugins needed for your target.

With the Go2 task and a simulation backend installed:

```bash
motrix-deploy task=go2-walk-flat/sim artifact=artifacts/go2-walk-flat.deploy
```

See the [deployment tutorial](../docs/source/en/user_guide/tutorial/advanced/motrix_deploy.md)
for export, simulation, and hardware workflows, or the [Python example](../examples/deploy_to_sim.py)
for programmatic control.
