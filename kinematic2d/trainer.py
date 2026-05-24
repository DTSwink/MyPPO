"""Fast batched supervised training loop."""

from __future__ import annotations

import math
import time
import traceback
from pathlib import Path
from threading import Event

import numpy as np
import torch

from kinematic2d.batch_env import BatchEnv
from kinematic2d.checkpoint import append_loss_point, save_checkpoint
from kinematic2d.loss_terms import (
    loss_term_keys,
    normalize_loss_coeffs,
    normalize_loss_terms_enabled,
)
from kinematic2d.settings import SettingsStore
from kinematic2d.torch_policy import TorchMLP

PELVIS_MAX_DEVIATION_RAD = math.radians(25.0)
MAX_LIMB_RADIUS_M = 0.6
SETTINGS_REFRESH_STEPS = 30


def pelvis_tilt_loss(pelvis_angles: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(torch.sin(pelvis_angles), torch.cos(pelvis_angles))
    excess = torch.relu(torch.abs(angles) - PELVIS_MAX_DEVIATION_RAD)
    return (excess * excess).mean()


def limb_radius_loss(limbs: torch.Tensor, max_radius: float = MAX_LIMB_RADIUS_M) -> torch.Tensor:
    """Penalize pelvis/feet xy distance from root origin beyond max_radius (root-local)."""
    dist = torch.linalg.norm(limbs[..., :2], dim=-1)
    excess = torch.relu(dist - max_radius)
    return (excess * excess).mean()


def compute_loss_terms(
    env: BatchEnv,
    frame_idx: torch.Tensor,
    current_limbs: torch.Tensor,
    action: torch.Tensor,
    limbs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "pelvis_tilt": pelvis_tilt_loss(limbs[:, 0, 2]),
        "foot_pin": env.pinned_foot_velocity_loss_at(frame_idx, current_limbs, action),
        "limb_radius": limb_radius_loss(limbs),
        "foot_stride": env.unpinned_foot_stride_loss_at(frame_idx, current_limbs, action),
    }


def _unroll_steps(env: BatchEnv, window: int) -> int:
    if window <= 0:
        return 0
    remaining = (env.n_roots - 1) - int(env.frame_idx[0].item())
    return min(window, max(0, remaining))


def train_loop(
    checkpoint_dir: Path,
    stop_event: Event,
    batch_size: int = 2048,
    lr: float = 3e-3,
    seed: int | None = 0,
    loss_terms_enabled: dict[str, bool] | None = None,
) -> None:
    try:
        _train_loop_impl(checkpoint_dir, stop_event, batch_size, lr, seed, loss_terms_enabled)
    except Exception:
        traceback.print_exc()
        raise


def _train_loop_impl(
    checkpoint_dir: Path,
    stop_event: Event,
    batch_size: int,
    lr: float,
    seed: int | None,
    loss_terms_enabled: dict[str, bool] | None,
) -> None:
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    env = BatchEnv(batch_size=batch_size, device=device)
    policy = TorchMLP().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    settings_store = SettingsStore()

    enabled = normalize_loss_terms_enabled(loss_terms_enabled)
    coeffs = normalize_loss_coeffs(settings_store.loss_coeffs())
    ar_window = settings_store.autoregressive_window()
    keys = loss_term_keys()

    step = 0
    interval_loss = 0.0
    interval_terms = {key: 0.0 for key in keys}
    interval_steps = 0
    last_checkpoint = time.perf_counter()
    checkpoint_interval = settings_store.checkpoint_refresh_sec()

    save_checkpoint(checkpoint_dir, **policy.numpy_weights(), step=0, loss=float("nan"))
    append_loss_point(
        checkpoint_dir,
        step=0,
        loss=1.0,
        terms={key: 1.0 for key in keys},
    )

    while not stop_event.is_set():
        if step % SETTINGS_REFRESH_STEPS == 0:
            checkpoint_interval = settings_store.checkpoint_refresh_sec()
            enabled = normalize_loss_terms_enabled(settings_store.loss_terms_enabled())
            coeffs = normalize_loss_coeffs(settings_store.loss_coeffs())
            ar_window = settings_store.autoregressive_window()

        if not env.poses_finite():
            env.reset_poses()

        steps = _unroll_steps(env, ar_window)
        if steps < 1:
            continue

        frame_idx = env.frame_idx.clone()
        current_limbs = env.current_limbs.clone()
        previous_limbs = env.previous_limbs.clone()

        total_loss = torch.zeros((), device=device)
        mean_terms = {key: torch.zeros((), device=device) for key in keys}
        actions_for_env: list[torch.Tensor] = []

        for _ in range(steps):
            obs = env.build_obs_from(frame_idx, current_limbs, previous_limbs)
            action = policy(obs)
            limbs = action[:, :9].reshape(-1, 3, 3)
            term_tensors = compute_loss_terms(env, frame_idx, current_limbs, action, limbs)

            step_loss = torch.zeros((), device=device)
            for key in keys:
                if enabled.get(key, True):
                    weighted = coeffs[key] * term_tensors[key]
                    step_loss = step_loss + weighted
                    mean_terms[key] = mean_terms[key] + weighted

            if not torch.isfinite(step_loss) or not torch.isfinite(action).all():
                env.reset_poses()
                total_loss = None
                break

            total_loss = total_loss + step_loss
            actions_for_env.append(action.detach())

            previous_limbs = current_limbs
            current_limbs = limbs
            frame_idx = frame_idx + 1

        if total_loss is None:
            continue

        total_loss = total_loss / steps
        for key in keys:
            mean_terms[key] = mean_terms[key] / steps

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for action in actions_for_env:
                env.step(action)

        step += 1
        interval_loss += float(total_loss.item())
        for key in keys:
            interval_terms[key] += float(mean_terms[key].item())
        interval_steps += 1

        now = time.perf_counter()
        if now - last_checkpoint >= checkpoint_interval:
            n = max(interval_steps, 1)
            mean_loss = interval_loss / n
            logged_terms = {key: interval_terms[key] / n for key in keys}
            save_checkpoint(checkpoint_dir, **policy.numpy_weights(), step=step, loss=mean_loss)
            append_loss_point(checkpoint_dir, step=step, loss=mean_loss, terms=logged_terms)
            interval_loss = 0.0
            interval_terms = {key: 0.0 for key in keys}
            interval_steps = 0
            last_checkpoint = now

    if interval_steps > 0:
        n = interval_steps
        mean_loss = interval_loss / n
        logged_terms = {key: interval_terms[key] / n for key in keys}
        save_checkpoint(checkpoint_dir, **policy.numpy_weights(), step=step, loss=mean_loss)
        append_loss_point(checkpoint_dir, step=step, loss=mean_loss, terms=logged_terms)
