# Third-party notices

MotrixLab source code is released under the Apache License, Version 2.0. The
license for the source code does not automatically apply to third-party
dependencies, robot descriptions, meshes, motion data, images, or videos.

This file is the release inventory for those materials. Before publishing a
release, every row must have a source URL, copyright holder, license, and a
clear statement that redistribution in this repository is permitted. If a
material cannot be cleared, remove it from the public tree or make it a
separately downloaded optional asset.

## Bundled assets and data

| Material | Repository path | Upstream/source | License or status |
| --- | --- | --- | --- |
| Franka Emika Panda MJCF and meshes | `motrix_envs/src/motrix_envs/manipulation/franka_lift_cube/xmls/`; `motrix_envs/src/motrix_envs/manipulation/franka_open_cabinet/xmls/` | [frankarobotics/franka_ros](https://github.com/frankarobotics/franka_ros/tree/develop/franka_description), see each directory README | The public source repository declares Apache-2.0; local model licenses are present. Record the exact imported revision and preserve attribution for all derived meshes |
| Shadow Hand E3M5 description | `motrix_envs/src/motrix_envs/manipulation/shadow_hand/xmls/` | Record the exact upstream source and revision before release | A local Apache-2.0 file is present; upstream provenance still needs to be recorded |
| RM65 descriptions and meshes | `motrix_envs/src/motrix_envs/manipulation/rm65_open_cabinet/xmls/`; `motrix_envs/src/motrix_envs/common/rm65/` | Record the exact upstream source and revision before release | A local Apache-2.0 file is present for the cabinet scene; verify every shared mesh |
| Unitree G1, Go1 and Go2 assets | `motrix_envs/src/motrix_envs/robot/assets/`; `motrix_envs/src/motrix_envs/locomotion/go1/` | Record the exact Unitree source and model terms | Release approval required from the rights holder; trademarks and model terms are separate from the code license |
| ANYmal-C assets | `motrix_envs/src/motrix_envs/robot/assets/anymal_c/`; `motrix_envs/src/motrix_envs/locomotion/anymal_c/` | Record the exact upstream source and model terms | Release approval and attribution required |
| Booster K1 assets | `motrix_envs/src/motrix_envs/robot/assets/k1/` | [BoosterRobotics/booster_assets K1 model](https://github.com/BoosterRobotics/booster_assets/blob/main/robots/K1/K1_22dof.xml) | The public repository declares BSD-3-Clause and the local XML identifies this source. Record the exact imported revision and confirm that the derived XML and meshes may be redistributed |
| Dex-EVT assets | `motrix_envs/src/motrix_envs/robot/assets/dex_evt/` | Record the exact upstream source and model terms | No public provenance or redistribution permission is documented in this checkout; release approval and attribution are required |
| LAFAN1 input data and converter | `motrix_envs/src/motrix_envs/motion/converters/`; downloaded inputs are not part of the repository | [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | CC BY-NC-ND 4.0 is documented by the user guide; retain attribution and state the non-commercial/no-derivatives limits before redistributing any converted output |
| Bundled whole-body-tracking motion clips | `motrix_envs/src/motrix_envs/locomotion/wbt/assets/motion/` | Record the source, revision, author, and terms for each `*.npz` clip | Provenance and redistribution permission are not yet documented for every clip; do not publish until cleared |
| Documentation videos, posters and benchmark figures | `docs/source/_static/` | Record whether each item is original, generated, or derived from a third-party asset | Verify the rights for the source material and the resulting media independently; retain attribution where required |

## Runtime dependencies

The following are installed from external distributions and remain under their
own licenses: MotrixSim, MuJoCo, Gymnasium, NumPy, Numba, nvidia-ml-py (NVML
bindings used for training-panel GPU metrics), Hydra/OmegaConf,
SKRL, RSL-RL, PyTorch/JAX, ONNX Runtime, TensorBoard, and the Unitree SDK2
Python package. Their licenses are not replaced by the MotrixLab license.
The release process should generate a dependency license report from the final
lock file (for example with `pip-licenses`) and attach it to each release.

## Release checklist

The status below records what can be verified from this checkout. A checked item
does not replace legal approval for an asset or dependency.

### Repository checks completed (2026-09-07)

- [x] The inventory covers the bundled robot descriptions, meshes, motion data,
      documentation media, and runtime dependency categories currently found in
      the tree.
- [x] Local Apache-2.0 license files are present beside the Franka lift-cube,
      Franka open-cabinet, RM65 open-cabinet, and Shadow Hand model directories.
- [x] Git LFS inventory was enumerated with `git lfs ls-files` (667 tracked
      objects at review time).
- [x] The dependency lock file `uv.lock` is present; the Unitree SDK source is
      pinned to a Git revision in `motrix_deploy_unitree/pyproject.toml`.
- [x] Public upstream metadata was checked for the documented Franka source
      (`frankarobotics/franka_ros`, Apache-2.0) and Booster K1 source
      (`BoosterRobotics/booster_assets`, BSD-3-Clause). These references do not
      by themselves prove that this checkout contains the same revision.

### Required before the first public release

- [ ] Fill in an upstream URL, exact revision, copyright holder, license, and
      redistribution permission for every asset row, including each bundled
      whole-body-tracking motion clip.
- [ ] Obtain written clearance for Unitree, ANYmal-C, Booster K1, Dex-EVT, RM65,
      and other branded models where the upstream terms do not clearly permit
      redistribution. Separate trademark attribution from copyright licensing.
- [ ] Confirm that every Git LFS object has the same redistribution permission
      as its pointer file; remove or externalize any uncleared large object.
- [ ] Generate and attach a dependency license report from the final lock file
      (for example with `pip-licenses`), including transitive dependencies.
- [ ] Run a dependency scan and a secret scan over the complete public Git
      history, not only the current checkout, and resolve every finding.
- [ ] Remove or replace any asset, media file, dependency, or historical object
      that cannot be legally redistributed before creating the first release tag.
