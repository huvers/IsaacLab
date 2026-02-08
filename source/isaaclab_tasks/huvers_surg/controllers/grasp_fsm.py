"""Grasp state machine for behavior B."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class GraspState(str, Enum):
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CANDIDATE_GRASP = "CANDIDATE_GRASP"
    HELD = "HELD"
    OPENING = "OPENING"
    FAILED = "FAILED"


@dataclass
class GraspFsmCfg:
    close_gap_thresh: float = 0.01
    open_gap_thresh: float = 0.03
    drop_drift_thresh: float = 0.02
    drop_force_thresh: float = 0.5
    drop_steps: int = 5


class GraspFsm:
    """Deterministic grasp FSM for scripted demos."""

    def __init__(self, cfg: GraspFsmCfg) -> None:
        self.cfg = cfg
        self.state = GraspState.OPEN
        self._drop_counter = 0

    def reset(self) -> None:
        self.state = GraspState.OPEN
        self._drop_counter = 0

    def _is_drop(self, object_drift: float | torch.Tensor | None, force_left: float | torch.Tensor | None, force_right: float | torch.Tensor | None, jaw_gap: float | torch.Tensor) -> bool:
        drift = float(object_drift) if object_drift is not None else 0.0
        f_l = float(force_left) if force_left is not None else 0.0
        f_r = float(force_right) if force_right is not None else 0.0
        gap = float(jaw_gap)

        drift_bad = drift > self.cfg.drop_drift_thresh
        force_bad = (gap < self.cfg.close_gap_thresh) and (min(f_l, f_r) < self.cfg.drop_force_thresh)
        return drift_bad or force_bad

    def update(
        self,
        *,
        close_cmd: bool,
        open_cmd: bool,
        left_in_contact: bool,
        right_in_contact: bool,
        jaw_gap: float | torch.Tensor,
        constraint_engaged: bool,
        object_drift: float | torch.Tensor | None = None,
        force_left: float | torch.Tensor | None = None,
        force_right: float | torch.Tensor | None = None,
    ) -> GraspState:
        # transition logic
        if self.state == GraspState.OPEN:
            if close_cmd:
                self.state = GraspState.CLOSING

        elif self.state == GraspState.CLOSING:
            if open_cmd:
                self.state = GraspState.OPENING
            elif left_in_contact and right_in_contact and float(jaw_gap) < self.cfg.close_gap_thresh:
                self.state = GraspState.CANDIDATE_GRASP

        elif self.state == GraspState.CANDIDATE_GRASP:
            if constraint_engaged:
                self.state = GraspState.HELD
            elif open_cmd:
                self.state = GraspState.OPENING

        elif self.state == GraspState.HELD:
            if open_cmd:
                self.state = GraspState.OPENING
            else:
                if self._is_drop(object_drift, force_left, force_right, jaw_gap):
                    self._drop_counter += 1
                else:
                    self._drop_counter = 0
                if self._drop_counter >= self.cfg.drop_steps:
                    self.state = GraspState.OPENING

        elif self.state == GraspState.OPENING:
            if float(jaw_gap) > self.cfg.open_gap_thresh:
                self.state = GraspState.OPEN

        return self.state

