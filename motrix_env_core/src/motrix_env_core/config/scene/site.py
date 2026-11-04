# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec3, Vec4, optional_vec
from motrix_env_core.config.scene.base import SceneObjCfg


@configclass(kw_only=True)
class SiteCfg(SceneObjCfg):
    """Create a site on a named logical link.

    ``parent_link_name`` uses the imported link name before a containing
    ``BodyCfg`` applies its prefix or suffix. When the parent link belongs to a
    ``BodyCfg`` model, the site mounts before attach, so both its name and its
    parent follow that body's decoration. Otherwise the site mounts after all
    bodies are imported with its plain name on the literal parent link. When
    used as a scene object the field name is the site name; in a model file
    config's ``sites`` (``UrdfFileCfg``/``MjcfFileCfg``) ``name`` must be
    provided explicitly.
    """

    parent_link_name: str
    name: str | None = None
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Vec4 = (0.0, 0.0, 0.0, 1.0)

    def validate(self, object_name: str | None = None) -> None:
        super().validate(object_name or self.name or "site")
        if self.name is not None and not self.name:
            raise ValueError("SiteCfg.name must not be empty when provided")
        if not self.parent_link_name:
            raise ValueError("SiteCfg.parent_link_name must not be empty")
        optional_vec("SiteCfg.position", self.position, 3)
        optional_vec("SiteCfg.rotation", self.rotation, 4)
