"""Organ asset helpers (visual + physics proxy)."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.utils import prims as prim_utils
from isaaclab.utils import configclass


@configclass
class OrganAssetCfg:
    """Configuration for a visual organ asset with a rigid physics proxy."""

    prim_path: str = "/World/Organs/Gallbladder"
    visual_prim_name: str = "GaussianSplat"
    physics_prim_name: str = "ProxyMesh"

    visual_size: tuple[float, float, float] = (0.06, 0.04, 0.02)
    proxy_size: tuple[float, float, float] = (0.06, 0.04, 0.02)

    visual_color: tuple[float, float, float] = (0.85, 0.3, 0.3)
    proxy_color: tuple[float, float, float] = (0.6, 0.15, 0.15)

    proxy_density: float = 800.0
    proxy_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )

    proxy_init_pos: tuple[float, float, float] = (0.6, 0.0, 0.02)
    proxy_init_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    activate_contact_sensors: bool = False


@dataclass
class OrganAssetInfo:
    """Returned info from :func:`spawn_organ_asset`."""

    parent_prim_path: str
    visual_prim_path: str
    physics_prim_path: str
    rigid_cfg: RigidObjectCfg


def _ensure_xform(path: str) -> None:
    if not prim_utils.is_prim_path_valid(path):
        prim_utils.create_prim(path, prim_type="Xform")


def spawn_organ_asset(cfg: OrganAssetCfg) -> OrganAssetInfo:
    """Spawn the visual organ prim and prepare a rigid proxy config.

    The physics proxy is returned as a :class:`RigidObjectCfg` and should be instantiated
    by the caller (e.g. inside an environment's scene setup).
    """
    parent_path = cfg.prim_path
    visual_root = f"{parent_path}/Visual"
    physics_root = f"{parent_path}/Physics"
    visual_prim_path = f"{visual_root}/{cfg.visual_prim_name}"
    physics_prim_path = f"{physics_root}/{cfg.physics_prim_name}"

    _ensure_xform(parent_path)
    _ensure_xform(visual_root)
    _ensure_xform(physics_root)

    # spawn visual-only child
    if not prim_utils.is_prim_path_valid(visual_prim_path):
        visual_cfg = sim_utils.CuboidCfg(
            size=cfg.visual_size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.visual_color),
        )
        visual_cfg.func(visual_prim_path, visual_cfg)

    # build physics proxy config (spawned later by RigidObject)
    proxy_spawn_cfg = sim_utils.CuboidCfg(
        size=cfg.proxy_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(density=cfg.proxy_density),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=cfg.proxy_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.proxy_color),
        activate_contact_sensors=cfg.activate_contact_sensors,
    )

    rigid_cfg = RigidObjectCfg(
        prim_path=physics_prim_path,
        spawn=proxy_spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=cfg.proxy_init_pos,
            rot=cfg.proxy_init_rot,
        ),
    )

    return OrganAssetInfo(
        parent_prim_path=parent_path,
        visual_prim_path=visual_prim_path,
        physics_prim_path=physics_prim_path,
        rigid_cfg=rigid_cfg,
    )

