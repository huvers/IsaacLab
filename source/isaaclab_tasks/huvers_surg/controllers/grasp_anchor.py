"""Grasp anchor helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from huvers_surg.utils import compose_pose, relative_pose


@dataclass
class GripperState:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    jaw_pos_w: torch.Tensor | None = None  # (N, 2, 3) or (2, 3)
    jaw_quat_w: torch.Tensor | None = None
    jaw_gap: torch.Tensor | None = None


@dataclass
class GraspAnchor:
    """Relative transform between gripper frame and object at grasp time."""

    rel_pos: torch.Tensor
    rel_quat: torch.Tensor

    def target_pose(self, gripper_state: GripperState) -> tuple[torch.Tensor, torch.Tensor]:
        return compose_pose(gripper_state.pos_w, gripper_state.quat_w, self.rel_pos, self.rel_quat)


def compute_jaw_midpoint_anchor(
    gripper_state: GripperState,
    obj_pos_w: torch.Tensor,
    obj_quat_w: torch.Tensor,
) -> GraspAnchor:
    """Compute an anchor using the midpoint between jaw tips (or gripper frame if unavailable)."""
    if gripper_state.jaw_pos_w is not None:
        jaw_pos = gripper_state.jaw_pos_w
        if jaw_pos.ndim == 2:
            jaw_pos = jaw_pos.unsqueeze(0)
        grip_pos = jaw_pos.mean(dim=-2)
    else:
        grip_pos = gripper_state.pos_w

    grip_quat = gripper_state.quat_w

    rel_pos, rel_quat = relative_pose(grip_pos, grip_quat, obj_pos_w, obj_quat_w)
    return GraspAnchor(rel_pos=rel_pos, rel_quat=rel_quat)

