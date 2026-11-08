# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim backend for :mod:`motrix_env_core`.

This package binds the backend-neutral sim boundary (``motrix_env_core.sim``)
to the MotrixSim simulator: scene compilation, the live ``SimBackend``, and the
torch-tensor frontend. Importing the package is cheap on purpose — the native
``motrixsim`` module loads only when a deep module (``compiler``, ``runtime``,
...) is imported, so registry discovery never pulls the
simulator. Registration happens through the ``motrix_env.sim_backends`` entry
point (see :mod:`motrix_env_motrixsim.register`).
"""
