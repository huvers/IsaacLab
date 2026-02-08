"""Tissue interaction interface and rigid proxy implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from isaaclab.assets import RigidObject
from isaaclab.sim.prims import XFormPrim

from huvers_surg.controllers.grasp_anchor import GraspAnchor, GripperState, compute_jaw_midpoint_anchor
from huvers_surg.controllers.grasp_constraint import GraspConstraint
from huvers_surg.utils import wp_to_torch


@dataclass
class CutResult:
    success: bool
    message: str = ""


class TissueInteractable(ABC):
    """Interface for tissue-like interactable objects."""

    @abstractmethod
    def get_physics_proxy_prim(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_visual_prim(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute_grasp_anchor(self, gripper_state: GripperState) -> GraspAnchor:
        raise NotImplementedError

    @abstractmethod
    def apply_grasp_constraint(self, anchor: GraspAnchor, gripper_state: GripperState) -> None:
        raise NotImplementedError

    @abstractmethod
    def release_grasp(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def step_visual_update(self) -> None:
        raise NotImplementedError

    # Roadmap hooks for cutting/tearing
    def request_cut(self, path: torch.Tensor, tool_pose: torch.Tensor) -> CutResult:
        return CutResult(False, "Cutting not implemented for this proxy.")

    def update_collision_proxy(self) -> None:
        return None

    def update_visual_gs(self) -> None:
        return None

    def update_visual_mapping(self) -> None:
        return None


@dataclass
class RigidTissueProxy(TissueInteractable):
    """Rigid tissue proxy backed by a RigidObject and a visual prim."""

    rigid_object: RigidObject
    visual_prim_path: str
    physics_prim_path: str
    grasp_constraint: GraspConstraint

    def __post_init__(self):
        self._visual_xform = XFormPrim(self.visual_prim_path, name="tissue_visual", device=self.rigid_object.device)

    def get_physics_proxy_prim(self) -> str:
        return self.physics_prim_path

    def get_visual_prim(self) -> str:
        return self.visual_prim_path

    def compute_grasp_anchor(self, gripper_state: GripperState) -> GraspAnchor:
        obj_pos = wp_to_torch(self.rigid_object.data.root_pos_w)
        obj_quat = wp_to_torch(self.rigid_object.data.root_quat_w)
        return compute_jaw_midpoint_anchor(gripper_state, obj_pos, obj_quat)

    def apply_grasp_constraint(self, anchor: GraspAnchor, gripper_state: GripperState) -> None:
        if not self.grasp_constraint.engaged:
            self.grasp_constraint.engage(anchor)
        self.grasp_constraint.step(gripper_state, self.rigid_object)

    def release_grasp(self) -> None:
        self.grasp_constraint.release(self.rigid_object)

    def step_visual_update(self) -> None:
        # Mirror physics proxy pose to the visual prim
        pos = wp_to_torch(self.rigid_object.data.root_pos_w)
        quat = wp_to_torch(self.rigid_object.data.root_quat_w)
        self._visual_xform.set_world_poses(pos, quat)


class DeformableTissueProxy(TissueInteractable):
    """Stub for future deformable tissue implementation."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Deformable tissue proxy is not implemented yet.")

    def get_physics_proxy_prim(self) -> str:  # pragma: no cover - stub
        raise NotImplementedError

    def get_visual_prim(self) -> str:  # pragma: no cover - stub
        raise NotImplementedError

    def compute_grasp_anchor(self, gripper_state: GripperState) -> GraspAnchor:  # pragma: no cover - stub
        raise NotImplementedError

    def apply_grasp_constraint(self, anchor: GraspAnchor, gripper_state: GripperState) -> None:  # pragma: no cover
        raise NotImplementedError

    def release_grasp(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def step_visual_update(self) -> None:  # pragma: no cover - stub
        raise NotImplementedError
