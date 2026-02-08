"""Warp/Torch interop helpers."""

from __future__ import annotations

from typing import Optional

import torch
import warp as wp


def wp_to_torch(array: wp.array | None, *, dtype: Optional[torch.dtype] = None, device: Optional[str] = None) -> torch.Tensor | None:
    """Convert a Warp array to a torch.Tensor with optional dtype/device overrides."""
    if array is None:
        return None
    tensor = wp.to_torch(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor

