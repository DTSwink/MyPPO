"""Checkpoint save/load shared between training and visualizer."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

WEIGHT_KEYS = ("w1", "b1", "w2", "b2")


def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    if not tmp_path.is_file():
        raise FileNotFoundError(f"Staging file missing: {tmp_path}")
    os.replace(tmp_path, final_path)


def save_checkpoint(
    directory: Path,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    step: int,
    loss: float,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    staging_path = directory / "_latest_staging.npz"
    final_path = directory / "latest.npz"
    np.savez(
        staging_path,
        w1=w1.astype(np.float32),
        b1=b1.astype(np.float32),
        w2=w2.astype(np.float32),
        b2=b2.astype(np.float32),
        step=np.int64(step),
        loss=np.float32(loss),
    )
    _atomic_replace(staging_path, final_path)


def load_checkpoint(directory: Path) -> dict[str, np.ndarray | int | float] | None:
    path = directory / "latest.npz"
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        with np.load(path) as data:
            return {
                "w1": data["w1"],
                "b1": data["b1"],
                "w2": data["w2"],
                "b2": data["b2"],
                "step": int(data["step"]),
                "loss": float(data["loss"]),
            }
    except (OSError, ValueError, KeyError, EOFError):
        return None


def _load_metrics_file(metrics_path: Path) -> dict:
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        return {"points": []}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"points": []}


def append_loss_point(
    directory: Path,
    step: int,
    loss: float,
    terms: dict[str, float] | None = None,
) -> list[dict[str, float | int]]:
    directory.mkdir(parents=True, exist_ok=True)
    metrics_path = directory / "metrics.json"
    staging_path = directory / "_metrics_staging.json"
    history = _load_metrics_file(metrics_path)
    if "points" not in history:
        history = {"points": []}

    point: dict[str, float | int] = {"step": step, "loss": loss}
    if terms:
        point.update(terms)

    history["points"].append(point)
    staging_path.write_text(json.dumps(history), encoding="utf-8")
    _atomic_replace(staging_path, metrics_path)
    return history["points"]


def load_loss_history(directory: Path) -> list[dict[str, float | int]]:
    metrics_path = directory / "metrics.json"
    history = _load_metrics_file(metrics_path)
    return history.get("points", [])
