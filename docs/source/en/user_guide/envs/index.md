# Environments

A MotrixLab Environment is a simulation task created through the Env Registry. It owns the scene, observations, actions, rewards, termination, and reset behavior. Environments are separate from reusable Robot models and from Training Tasks that select training algorithms and parameters.

Preview an environment without starting training:

```bash
uv run scripts/view.py env=<env-id>
```

## Environment topics

-   [Basic Environments](basic/index.md)
-   [DM Control Environments](dm_control/index.md)
-   [Whole-Body Tracking](whole_body_tracking/index.md)
-   [Generic Humanoid Velocity-Tracking Environment](humanoid_velocity_tracking.md)
-   [Generic Quadruped Velocity-Tracking Environment](quadruped_velocity_tracking.md)
-   [Other Quadruped Locomotion Environments](quadruped_locomotion/index.md)
-   [Manipulation Environments](manipulation/index.md)

<!-- ENV_OVERVIEW_TABLE_START -->

<!-- This table is generated; do not edit this block manually. -->
| Poster | Env ID | Description | Training Algorithms |
| --- | --- | --- | --- |
| <img src="../../_static/images/poster/acrobot.jpg" alt="acrobot" width="240"> | `acrobot` | Swing up and balance an underactuated two-link robot. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/anymal_c_navigation_flat.jpg" alt="anymal_c_navigation_flat" width="240"> | `anymal_c_navigation_flat` | Navigate ANYmal-C toward a target on flat ground. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/anymalc-walk-flat.jpg" alt="anymalc-walk-flat" width="240"> | `anymalc-walk-flat` | Track locomotion commands with ANYmal-C on flat ground. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/anymalc-walk-rough.jpg" alt="anymalc-walk-rough" width="240"> | `anymalc-walk-rough` | Track walking commands with ANYmal-C on a procedural rough height field. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/bounce_ball.jpg" alt="bounce_ball" width="240"> | `bounce_ball` | Control a paddle to keep bouncing a table-tennis ball. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/cartpole.jpg" alt="cartpole" width="240"> | `cartpole` | Move a cart to keep an inverted pendulum upright. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dex-evt-walk-flat.jpg" alt="dex-evt-walk-flat" width="240"> | `dex-evt-walk-flat` | Control the Dex-EVT humanoid to walk on flat ground. | `motrix.fastsac` |
| <img src="../../_static/images/poster/dex-evt-walk-rough.jpg" alt="dex-evt-walk-rough" width="240"> | `dex-evt-walk-rough` | Control the Dex-EVT humanoid to walk over uneven terrain. | `motrix.fastsac` |
| <img src="../../_static/images/poster/dex-evt-wbt-dance.jpg" alt="dex-evt-wbt-dance" width="240"> | `dex-evt-wbt-dance` | Track a bundled dance reference motion with Dex-EVT. | `motrix.fastsac` |
| <img src="../../_static/images/poster/dm-cheetah.jpg" alt="dm-cheetah" width="240"> | `dm-cheetah` | Drive the planar Cheetah forward as fast as possible. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-finger-spin.jpg" alt="dm-finger-spin" width="240"> | `dm-finger-spin` | Use the robotic finger to continuously spin a free object. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-finger-turn-easy.jpg" alt="dm-finger-turn-easy" width="240"> | `dm-finger-turn-easy` | Turn the finger object to a target angle with an easy tolerance. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-finger-turn-hard.jpg" alt="dm-finger-turn-hard" width="240"> | `dm-finger-turn-hard` | Turn the finger object precisely to a target angle. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-hopper-hop.jpg" alt="dm-hopper-hop" width="240"> | `dm-hopper-hop` | Keep the one-legged Hopper upright while hopping forward. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-hopper-stand.jpg" alt="dm-hopper-stand" width="240"> | `dm-hopper-stand` | Keep the one-legged Hopper standing upright. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-humanoid-run.jpg" alt="dm-humanoid-run" width="240"> | `dm-humanoid-run` | Control the DM Control humanoid to run forward. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-humanoid-stand.jpg" alt="dm-humanoid-stand" width="240"> | `dm-humanoid-stand` | Control the DM Control humanoid to remain standing. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-humanoid-walk.jpg" alt="dm-humanoid-walk" width="240"> | `dm-humanoid-walk` | Control the DM Control humanoid to walk forward. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-lqr-2-1.jpg" alt="dm-lqr-2-1" width="240"> | `dm-lqr-2-1` | Stabilize a linear system with two state and one control dimensions. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-lqr-6-2.jpg" alt="dm-lqr-6-2" width="240"> | `dm-lqr-6-2` | Stabilize a linear system with six state and two control dimensions. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-manipulator-bring-ball.jpg" alt="dm-manipulator-bring-ball" width="240"> | `dm-manipulator-bring-ball` | Control a manipulator to bring a ball to a target location. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-escape.jpg" alt="dm-quadruped-escape" width="240"> | `dm-quadruped-escape` | Control the DM Control quadruped to escape from a bowl-shaped area. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-fetch.jpg" alt="dm-quadruped-fetch" width="240"> | `dm-quadruped-fetch` | Control the DM Control quadruped to find and fetch a ball. | `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-run.jpg" alt="dm-quadruped-run" width="240"> | `dm-quadruped-run` | Control the DM Control quadruped to run forward. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-quadruped-walk.jpg" alt="dm-quadruped-walk" width="240"> | `dm-quadruped-walk` | Control the DM Control quadruped to walk forward. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-reacher.jpg" alt="dm-reacher" width="240"> | `dm-reacher` | Move a two-link arm endpoint to a random target. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-runner.jpg" alt="dm-runner" width="240"> | `dm-runner` | Control the planar Walker to run at a high target speed. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-stander.jpg" alt="dm-stander" width="240"> | `dm-stander` | Control the planar Walker to remain standing upright. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/dm-walker.jpg" alt="dm-walker" width="240"> | `dm-walker` | Control the planar Walker to walk forward at a target speed. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/franka-lift-cube.jpg" alt="franka-lift-cube" width="240"> | `franka-lift-cube` | Control a Franka arm to grasp and lift a cube. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/franka-open-cabinet.jpg" alt="franka-open-cabinet" width="240"> | `franka-open-cabinet` | Control a Franka arm to grasp a handle and open a drawer. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/g1-29dof-wbt-largebox.jpg" alt="g1-29dof-wbt-largebox" width="240"> | `g1-29dof-wbt-largebox` | Track a large-box carrying reference motion with Unitree G1. | `motrix.fastsac` |
| <img src="../../_static/images/poster/g1-walk-flat.jpg" alt="g1-walk-flat" width="240"> | `g1-walk-flat` | Track walking commands with Unitree G1 on flat ground. | `motrix.fastsac` |
| <img src="../../_static/images/poster/g1-walk-rough.jpg" alt="g1-walk-rough" width="240"> | `g1-walk-rough` | Track walking commands with Unitree G1 over uneven terrain. | `motrix.fastsac` |
| <img src="../../_static/images/poster/g1-wbt-dance.jpg" alt="g1-wbt-dance" width="240"> | `g1-wbt-dance` | Track the bundled G1 dance motion with the manager-based environment. | `motrix.fastsac` |
| <img src="../../_static/images/poster/go1-stairs-terrain-walk.jpg" alt="go1-stairs-terrain-walk" width="240"> | `go1-stairs-terrain-walk` | Control Unitree Go1 to walk over stair terrain. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go1-walk-flat.jpg" alt="go1-walk-flat" width="240"> | `go1-walk-flat` | Track walking commands with Unitree Go1 on flat ground. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go1-walk-rough.jpg" alt="go1-walk-rough" width="240"> | `go1-walk-rough` | Track walking commands with Unitree Go1 on a procedural rough height field. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go2-walk-flat.jpg" alt="go2-walk-flat" width="240"> | `go2-walk-flat` | Track walking commands with Unitree Go2 on flat ground. | `motrix.fastsac`, `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/go2-walk-rough.jpg" alt="go2-walk-rough" width="240"> | `go2-walk-rough` | Track walking commands with Unitree Go2 on a procedural rough height field. | `motrix.fastsac`, `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/k1-walk-flat.jpg" alt="k1-walk-flat" width="240"> | `k1-walk-flat` | Track walking commands with Booster K1 on flat ground. | `motrix.fastsac` |
| <img src="../../_static/images/poster/k1-walk-rough.jpg" alt="k1-walk-rough" width="240"> | `k1-walk-rough` | Track walking commands with Booster K1 over uneven terrain. | `motrix.fastsac` |
| <img src="../../_static/images/poster/k1-wbt-freekick.jpg" alt="k1-wbt-freekick" width="240"> | `k1-wbt-freekick` | Track a free-kick reference motion with Booster K1. | `motrix.fastsac` |
| <img src="../../_static/images/poster/microduck-walk-flat.jpg" alt="microduck-walk-flat" width="240"> | `microduck-walk-flat` | Track walking commands with Microduck on flat ground. | `motrix.fastsac` |
| <img src="../../_static/images/poster/microduck-walk-rough.jpg" alt="microduck-walk-rough" width="240"> | `microduck-walk-rough` | Track walking commands with Microduck over uneven terrain. | `motrix.fastsac` |
| <img src="../../_static/images/poster/peg-insert.jpg" alt="peg-insert" width="240"> | `peg-insert` | Control RM65 to grasp, align, and insert a peg into a socket. | `motrix.fastsac`, `skrl.ppo` |
| <img src="../../_static/images/poster/pendulum.jpg" alt="pendulum" width="240"> | `pendulum` | Apply joint torque to swing up and balance a pendulum. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/point_mass.jpg" alt="point_mass" width="240"> | `point_mass` | Move a two-dimensional point mass to a random target. | `skrl.ppo` |
| <img src="../../_static/images/poster/rm65-open-cabinet.jpg" alt="rm65-open-cabinet" width="240"> | `rm65-open-cabinet` | Control RM65 to grasp a handle and open the cabinet's bottom drawer. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/rm65_insert_peg.jpg" alt="rm65_insert_peg" width="240"> | `rm65_insert_peg` | Control RM65 to grasp, align, and insert a peg into a socket. | `motrix.fastsac`, `skrl.ppo` |
| <img src="../../_static/images/poster/shadow-hand-repose.jpg" alt="shadow-hand-repose" width="240"> | `shadow-hand-repose` | Control a Shadow Hand to reorient a cube in-hand. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/stewart.jpg" alt="stewart" width="240"> | `stewart` | Keep a Stewart platform level without external disturbances. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/stewart-disturb-xy.jpg" alt="stewart-disturb-xy" width="240"> | `stewart-disturb-xy` | Recover and level a Stewart platform under XY disturbances. | `rslrl.ppo`, `skrl.ppo` |
| <img src="../../_static/images/poster/stewart-static.jpg" alt="stewart-static" width="240"> | `stewart-static` | Keep a Stewart platform level without external disturbances. | `rslrl.ppo`, `skrl.ppo` |

<!-- ENV_OVERVIEW_TABLE_END -->
