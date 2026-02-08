"""Pose and quaternion utilities (xyzw convention)."""

from __future__ import annotations

from typing import Tuple

import torch

from isaaclab.utils import math as math_utils


def quat_xyzw_from_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from (w, x, y, z) to (x, y, z, w)."""
    return math_utils.convert_quat(quat_wxyz, to="xyzw")


def quat_wxyz_from_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion from (x, y, z, w) to (w, x, y, z)."""
    return math_utils.convert_quat(quat_xyzw, to="wxyz")


def compose_pose(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compose two poses (A ∘ B) with quaternions in (x, y, z, w).

    Args:
        pos_a: Position of pose A. Shape (..., 3).
        quat_a: Quaternion of pose A. Shape (..., 4).
        pos_b: Position of pose B. Shape (..., 3).
        quat_b: Quaternion of pose B. Shape (..., 4).

    Returns:
        Composed position and quaternion (xyzw).
    """
    pos = pos_a + math_utils.quat_apply(quat_a, pos_b)
    quat = math_utils.quat_mul(quat_a, quat_b)
    return pos, quat


def invert_pose(pos: torch.Tensor, quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Invert a pose (position + quaternion) in (x, y, z, w).

    Args:
        pos: Position. Shape (..., 3).
        quat: Quaternion in (x, y, z, w). Shape (..., 4).

    Returns:
        Inverted pose (pos, quat).
    """
    quat_inv = math_utils.quat_inv(quat)
    pos_inv = -math_utils.quat_apply(quat_inv, pos)
    return pos_inv, quat_inv


def relative_pose(
    pos_a: torch.Tensor,
    quat_a: torch.Tensor,
    pos_b: torch.Tensor,
    quat_b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute pose of B in frame A (T_a^-1 * T_b).

    Args:
        pos_a: Position of frame A in world.
        quat_a: Quaternion of frame A in world (xyzw).
        pos_b: Position of frame B in world.
        quat_b: Quaternion of frame B in world (xyzw).

    Returns:
        Relative position and quaternion (xyzw).
    """
    inv_pos, inv_quat = invert_pose(pos_a, quat_a)
    return compose_pose(inv_pos, inv_quat, pos_b, quat_b)


def pose_error(
    pos_target: torch.Tensor,
    quat_target: torch.Tensor,
    pos_current: torch.Tensor,
    quat_current: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute pose error between target and current in world frame.

    Returns position error and rotation error (axis-angle).
    """
    pos_err = pos_target - pos_current
    rot_err = math_utils.quat_box_minus(quat_target, quat_current)
    return pos_err, rot_err

