# Unitree GO1 Fixed-Stairs Walk

`go1-stairs-terrain-walk` uses the independent `Go1WalkStairsTask` and fixed `scene_stairs_terrain.xml` stairs scene.
It does not reuse `QuadrupedWalkTask`; see the
[Generic Quadruped Velocity-Tracking Environment](../quadruped_velocity_tracking.md) for the shared GO1 flat-ground and
procedural rough-terrain tasks.

```{video} /_static/videos/go1_stairs_terrain_walk.mp4
:poster: _static/images/poster/go1_stairs_terrain_walk.jpg
:nocontrols:
:autoplay:
:playsinline:
:muted:
:loop:
:width: 100%
```

The 12D `go1-stairs-terrain-walk` action uses the actuator control bounds directly and is scaled by `action_scale=0.05` into
a PD target around the default pose. Its 60D observation concatenates body linear velocity, gyroscope, local gravity, 12D
joint-position residual, 12D joint velocity, 12D previous action, the 3D velocity command, and 3D contact force for each foot.
Only forward speed is sampled, with $v_x\in[0.5,1.0]$ m/s; $v_y$ and `yaw_rate` remain zero.

Reset positions cycle through 25 fixed locations in the scene. Joints return to the model default pose, and velocities and
history buffers are cleared. A failure termination occurs when the trunk contacts any ground geom or when the squared
horizontal linear-speed sum exceeds $10^8$. The default `20 s` limit produces truncation. This task does not randomize
physical parameters.

```bash
python scripts/view.py env=go1-stairs-terrain-walk
python scripts/train.py task=go1-stairs-terrain-walk/skrl.ppo
python scripts/play.py env=go1-stairs-terrain-walk
```
