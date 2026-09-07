# Microduck robot model

Ported from https://github.com/pollen-robotics/microduck_rl
(`src/mjlab_microduck/robot/microduck/robot_walk.xml`, renamed to
`microduck.xml`). Relative to upstream, a named termination capsule
`trunk_collision` was added on `trunk_base` for the humanoid walk task's
fall-termination contact query; everything else is unmodified.

- Upstream code and MJCF: Apache License 2.0
- 3D mesh files (`assets/*.stl`): Creative Commons BY-SA-NC
  (non-commercial), per the upstream README

The Microduck is a ~800 g, ~25 cm bipedal robot by Pollen Robotics with 14
actuated joints (5 per leg, 4 neck/head).
