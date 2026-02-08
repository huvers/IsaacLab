"""Contact feature extraction for grasping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from isaaclab.sensors import ContactSensor

from huvers_surg.utils import wp_to_torch


@dataclass
class ContactFeatureState:
    force_mag: torch.Tensor
    in_contact: torch.Tensor
    contact_persistence: torch.Tensor


class ContactFeatures:
    """Helper to extract contact forces and contact state with hysteresis."""

    def __init__(
        self,
        sensor: ContactSensor,
        force_threshold: float = 0.2,
        hysteresis: float = 0.05,
        persistence_steps: int = 3,
        reduce: Literal["max", "sum"] = "max",
    ) -> None:
        self.sensor = sensor
        self.force_threshold = force_threshold
        self.hysteresis = hysteresis
        self.persistence_steps = max(int(persistence_steps), 1)
        self.reduce = reduce
        # lazy init state once sensor buffers exist
        self._in_contact: torch.Tensor | None = None
        self._contact_steps: torch.Tensor | None = None

    def _reduce_force(self, force_vec: torch.Tensor) -> torch.Tensor:
        # force_vec: (N, B, 3) or (N, B, M, 3)
        mag = torch.linalg.norm(force_vec, dim=-1)
        if mag.ndim == 3:
            if self.reduce == "sum":
                return mag.sum(dim=(1, 2))
            return mag.amax(dim=(1, 2))
        if mag.ndim == 2:
            if self.reduce == "sum":
                return mag.sum(dim=1)
            return mag.amax(dim=1)
        # fallback
        return mag

    def update(self) -> ContactFeatureState:
        data = self.sensor.data
        force_tensor = None
        if data.force_matrix_w is not None:
            force_tensor = wp_to_torch(data.force_matrix_w)
        elif data.net_forces_w is not None:
            force_tensor = wp_to_torch(data.net_forces_w)

        if force_tensor is None:
            # No data available: return zeros.
            num_envs = self.sensor.num_instances or 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            force_mag = torch.zeros(num_envs, device=device)
        else:
            force_mag = self._reduce_force(force_tensor)

        if self._in_contact is None or self._contact_steps is None:
            self._in_contact = torch.zeros_like(force_mag, dtype=torch.bool)
            self._contact_steps = torch.zeros_like(force_mag, dtype=torch.long)

        on_thresh = self.force_threshold
        off_thresh = max(0.0, self.force_threshold - self.hysteresis)

        new_in_contact = torch.where(self._in_contact, force_mag > off_thresh, force_mag > on_thresh)
        self._contact_steps = torch.where(new_in_contact, self._contact_steps + 1, torch.zeros_like(self._contact_steps))
        contact_persistence = self._contact_steps >= self.persistence_steps

        self._in_contact = new_in_contact

        return ContactFeatureState(
            force_mag=force_mag,
            in_contact=new_in_contact,
            contact_persistence=contact_persistence,
        )

