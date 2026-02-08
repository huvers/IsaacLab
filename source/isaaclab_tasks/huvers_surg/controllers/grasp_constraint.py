"""Grasp constraint helpers (runtime fixed or servo attach)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab.assets import RigidObject
from isaaclab.utils import math as math_utils

from huvers_surg.controllers.grasp_anchor import GraspAnchor, GripperState
from huvers_surg.utils import pose_error, wp_to_torch


@dataclass
class GraspConstraintCfg:
    mode: str = "kinematic"  # "kinematic" or "impedance"
    smoothing: float = 0.2
    k_pos: float = 200.0
    d_pos: float = 5.0
    k_rot: float = 50.0
    d_rot: float = 2.0
    max_force: float = 200.0
    max_torque: float = 50.0


class GraspConstraint:
    """Maintain a grasp by attaching the object to a gripper frame."""

    def __init__(self, cfg: GraspConstraintCfg) -> None:
        self.cfg = cfg
        self.anchor: GraspAnchor | None = None
        self.engaged: bool = False

    def engage(self, anchor: GraspAnchor) -> None:
        self.anchor = anchor
        self.engaged = True

    def release(self, rigid_object: RigidObject | None = None) -> None:
        if rigid_object is not None and self.cfg.mode == "impedance":
            # disable external wrench
            rigid_object.set_external_force_and_torque(torch.zeros(0, 3), torch.zeros(0, 3))
        self.anchor = None
        self.engaged = False

    def step(self, gripper_state: GripperState, rigid_object: RigidObject) -> None:
        if not self.engaged or self.anchor is None:
            return

        target_pos, target_quat = self.anchor.target_pose(gripper_state)

        if self.cfg.mode == "kinematic":
            # smooth kinematic attach
            obj_pos = wp_to_torch(rigid_object.data.root_pos_w)
            obj_quat = wp_to_torch(rigid_object.data.root_quat_w)
            alpha = float(self.cfg.smoothing)
            new_pos = obj_pos + alpha * (target_pos - obj_pos)
            new_quat = math_utils.quat_slerp(obj_quat, target_quat, tau=alpha)
            new_pose = torch.cat([new_pos, new_quat], dim=-1)
            rigid_object.write_root_pose_to_sim(new_pose)
            return

        # impedance/servo attach
        obj_pos = wp_to_torch(rigid_object.data.root_pos_w)
        obj_quat = wp_to_torch(rigid_object.data.root_quat_w)
        obj_lin_vel = wp_to_torch(rigid_object.data.root_lin_vel_w)
        obj_ang_vel = wp_to_torch(rigid_object.data.root_ang_vel_w)

        pos_err, rot_err = pose_error(target_pos, target_quat, obj_pos, obj_quat)

        force = self.cfg.k_pos * pos_err - self.cfg.d_pos * obj_lin_vel
        torque = self.cfg.k_rot * rot_err - self.cfg.d_rot * obj_ang_vel

        force = torch.clamp(force, min=-self.cfg.max_force, max=self.cfg.max_force)
        torque = torch.clamp(torque, min=-self.cfg.max_torque, max=self.cfg.max_torque)

        # apply to root body in world frame
        force = force.unsqueeze(1)
        torque = torque.unsqueeze(1)
        rigid_object.set_external_force_and_torque(force, torque, is_global=True)
