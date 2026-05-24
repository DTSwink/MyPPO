"""Vectorized batched environment for fast training."""

from __future__ import annotations

import math

import numpy as np
import torch

from kinematic2d.trajectory import TRAJECTORY_SPEED, forward_constant_trajectory


def _precompute_future_roots(n_roots: int, speed: float, horizon: int) -> np.ndarray:
    """O(n) table for the straight +Y root path (all root frames share +X forward)."""
    frames = np.arange(n_roots, dtype=np.int32)[:, None]
    steps = np.arange(1, horizon + 1, dtype=np.int32)[None, :]
    target = np.minimum(frames + steps, n_roots - 1)
    forward = (target - frames).astype(np.float32) * speed
    table = np.zeros((n_roots, horizon, 3), dtype=np.float32)
    table[:, :, 0] = forward
    return table


def batch_local_xy_to_global(root: torch.Tensor, local_xy: torch.Tensor) -> torch.Tensor:
    """Map 2D points from root-local to global. root: (B, 3), local_xy: (B, 2)."""
    c = torch.cos(root[:, 2])
    s = torch.sin(root[:, 2])
    lx, ly = local_xy[:, 0], local_xy[:, 1]
    gx = c * lx - s * ly + root[:, 0]
    gy = s * lx + c * ly + root[:, 1]
    return torch.stack([gx, gy], dim=-1)


class BatchEnv:
    def __init__(
        self,
        batch_size: int,
        device: torch.device,
        horizon: int = 8,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.horizon = horizon

        roots_list = forward_constant_trajectory()
        self.n_roots = len(roots_list)
        roots_np = np.stack([r.to_array() for r in roots_list], axis=0)
        self.roots = torch.from_numpy(roots_np).to(device)
        future_np = _precompute_future_roots(self.n_roots, TRAJECTORY_SPEED, horizon)
        self.future_table = torch.from_numpy(future_np).to(device)

        self.frame_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.current_limbs = torch.empty(batch_size, 3, 3, device=device)
        self.previous_limbs = torch.empty(batch_size, 3, 3, device=device)
        self.reset_all()

    def reset_all(self) -> None:
        self.frame_idx.zero_()
        self._randomize_limbs(torch.arange(self.batch_size, device=self.device))

    def reset_poses(self, env_ids: torch.Tensor | None = None) -> None:
        """Resample limb poses without rewinding the root trajectory."""
        if env_ids is None:
            env_ids = torch.arange(self.batch_size, device=self.device)
        self._randomize_limbs(env_ids)

    def poses_finite(self) -> bool:
        return bool(torch.isfinite(self.current_limbs).all().item())

    def _randomize_limbs(self, env_ids: torch.Tensor) -> None:
        n = env_ids.shape[0]
        if n == 0:
            return
        self.current_limbs[env_ids] = torch.empty(n, 3, 3, device=self.device).uniform_(-0.15, 0.15)
        self.current_limbs[env_ids, :, 2] = torch.empty(n, 3, device=self.device).uniform_(-math.pi, math.pi)
        self.previous_limbs[env_ids] = self.current_limbs[env_ids].clone()

    def build_obs_from(
        self,
        frame_idx: torch.Tensor,
        current_limbs: torch.Tensor,
        previous_limbs: torch.Tensor,
    ) -> torch.Tensor:
        future = self.future_table[frame_idx]
        return torch.cat(
            [
                future.reshape(self.batch_size, -1),
                current_limbs.reshape(self.batch_size, -1),
                previous_limbs.reshape(self.batch_size, -1),
            ],
            dim=-1,
        )

    def build_obs(self) -> torch.Tensor:
        return self.build_obs_from(self.frame_idx, self.current_limbs, self.previous_limbs)

    def pinned_foot_velocity_loss_at(
        self,
        frame_idx: torch.Tensor,
        current_limbs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize global velocity of the pinned (lower height) foot."""
        root_curr = self.roots[frame_idx]
        next_idx = torch.clamp(frame_idx + 1, max=self.n_roots - 1)
        root_next = self.roots[next_idx]

        curr_left = current_limbs[:, 1, :2]
        curr_right = current_limbs[:, 2, :2]
        next_limbs = action[:, :9].reshape(self.batch_size, 3, 3)
        next_left = next_limbs[:, 1, :2]
        next_right = next_limbs[:, 2, :2]

        g_curr_left = batch_local_xy_to_global(root_curr, curr_left)
        g_curr_right = batch_local_xy_to_global(root_curr, curr_right)
        g_next_left = batch_local_xy_to_global(root_next, next_left)
        g_next_right = batch_local_xy_to_global(root_next, next_right)

        vel_left_sq = ((g_next_left - g_curr_left) ** 2).sum(dim=-1)
        vel_right_sq = ((g_next_right - g_curr_right) ** 2).sum(dim=-1)

        left_h = action[:, 9]
        right_h = action[:, 10]
        left_pinned = left_h <= right_h
        pinned_vel_sq = torch.where(left_pinned, vel_left_sq, vel_right_sq)
        return pinned_vel_sq.mean()

    def pinned_foot_velocity_loss(self, action: torch.Tensor) -> torch.Tensor:
        return self.pinned_foot_velocity_loss_at(self.frame_idx, self.current_limbs, action)

    def unpinned_foot_stride_loss_at(
        self,
        frame_idx: torch.Tensor,
        current_limbs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize unpinned foot global step deviating from 2x root motion."""
        root_curr = self.roots[frame_idx]
        next_idx = torch.clamp(frame_idx + 1, max=self.n_roots - 1)
        root_next = self.roots[next_idx]
        target_disp = 2.0 * (root_next[:, :2] - root_curr[:, :2])

        curr_left = current_limbs[:, 1, :2]
        curr_right = current_limbs[:, 2, :2]
        next_limbs = action[:, :9].reshape(self.batch_size, 3, 3)
        next_left = next_limbs[:, 1, :2]
        next_right = next_limbs[:, 2, :2]

        g_curr_left = batch_local_xy_to_global(root_curr, curr_left)
        g_curr_right = batch_local_xy_to_global(root_curr, curr_right)
        g_next_left = batch_local_xy_to_global(root_next, next_left)
        g_next_right = batch_local_xy_to_global(root_next, next_right)

        err_left = g_next_left - g_curr_left - target_disp
        err_right = g_next_right - g_curr_right - target_disp
        err_left_sq = (err_left * err_left).sum(dim=-1)
        err_right_sq = (err_right * err_right).sum(dim=-1)

        left_h = action[:, 9]
        right_h = action[:, 10]
        left_pinned = left_h <= right_h
        return torch.where(left_pinned, err_right_sq, err_left_sq).mean()

    def unpinned_foot_stride_loss(self, action: torch.Tensor) -> torch.Tensor:
        return self.unpinned_foot_stride_loss_at(self.frame_idx, self.current_limbs, action)

    def step(self, action: torch.Tensor) -> None:
        action = action.detach()
        self.previous_limbs = self.current_limbs.clone()
        self.current_limbs = action[:, :9].reshape(self.batch_size, 3, 3).clone()

        self.frame_idx += 1
        done = self.frame_idx >= self.n_roots - 1
        if done.any():
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(1)
            self.frame_idx[done_ids] = 0
            self._randomize_limbs(done_ids)
