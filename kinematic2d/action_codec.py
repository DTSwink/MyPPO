"""Decode raw MLP outputs into actions (no position/height clamping)."""

from __future__ import annotations

import numpy as np
import torch


def _wrap_angle_np(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def sanitize_action_numpy(raw: np.ndarray) -> np.ndarray:
    out = raw.astype(np.float32, copy=True)
    limbs = out[:9].reshape(3, 3)
    limbs[:, 2] = _wrap_angle_np(limbs[:, 2])
    out[:9] = limbs.reshape(-1)
    return out


def sanitize_action_torch(raw: torch.Tensor) -> torch.Tensor:
    """Out-of-place angle wrap so autograd stays valid."""
    limbs = raw[:, :9].reshape(-1, 3, 3)
    angle = torch.atan2(torch.sin(limbs[..., 2]), torch.cos(limbs[..., 2]))
    limbs_out = torch.stack([limbs[..., 0], limbs[..., 1], angle], dim=-1).reshape(raw.shape[0], 9)
    return torch.cat([limbs_out, raw[:, 9:13]], dim=1)
