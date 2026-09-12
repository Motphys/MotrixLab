# Microduck robot model

Ported from https://github.com/pollen-robotics/microduck_rl
(`src/mjlab_microduck/robot/microduck/robot_walk.xml`, renamed to
`microduck.xml`). Relative to upstream, a named termination capsule
`trunk_collision` was added on `trunk_base` for the humanoid walk task's
fall-termination contact query, and the foot collision geoms
(`left_foot_collision` / `right_foot_collision`) reference decimated convex
proxies (`assets/sole_{left,right}_collision.stl`, ~300 faces, generated from
the convex hulls of the high-poly visual soles) — the original sole meshes
exceeded the single-convex cooking face limit and dominated narrowphase
collision cost. Visual geoms are unmodified.

- Upstream code and MJCF: Apache License 2.0
- 3D mesh files (`assets/*.stl`): Creative Commons BY-SA-NC
  (non-commercial), per the upstream README

The Microduck is a ~800 g, ~25 cm bipedal robot by Pollen Robotics with 14
actuated joints (5 per leg, 4 neck/head).
