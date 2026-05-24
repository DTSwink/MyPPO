"""Persistent user settings shared across sessions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kinematic2d.loss_terms import (
    default_loss_coeffs,
    default_loss_terms_enabled,
    normalize_loss_coeffs,
    normalize_loss_terms_enabled,
)

SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"

DEFAULTS: dict[str, float] = {
    "checkpoint_refresh_sec": 10.0,
    "autoregressive_window": 1.0,
}

MIN_AUTOREGRESSIVE_WINDOW = 1
MAX_AUTOREGRESSIVE_WINDOW = 64

EXPERIMENT_DEFAULTS: dict[str, str | int | float] = {
    "checkpoint_dir": "checkpoints/live",
    "batch_size": 2048,
    "lr": 3e-3,
    "seed": 0,
}

MIN_CHECKPOINT_REFRESH_SEC = 1.0
MAX_CHECKPOINT_REFRESH_SEC = 120.0


@dataclass(frozen=True)
class ExperimentConfig:
    checkpoint_dir: Path
    batch_size: int
    lr: float
    seed: int
    loss_terms_enabled: dict[str, bool]


def _read_settings_file() -> dict:
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_settings_file(data: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _clamp_refresh(value: float) -> float:
    return max(MIN_CHECKPOINT_REFRESH_SEC, min(MAX_CHECKPOINT_REFRESH_SEC, value))


def _clamp_autoregressive_window(value: float) -> int:
    return int(max(MIN_AUTOREGRESSIVE_WINDOW, min(MAX_AUTOREGRESSIVE_WINDOW, round(value))))


def load_settings() -> dict[str, float]:
    data = _read_settings_file()
    merged = DEFAULTS.copy()
    if "checkpoint_refresh_sec" in data:
        merged["checkpoint_refresh_sec"] = _clamp_refresh(float(data["checkpoint_refresh_sec"]))
    if "autoregressive_window" in data:
        merged["autoregressive_window"] = float(
            _clamp_autoregressive_window(float(data["autoregressive_window"]))
        )
    return merged


def load_loss_terms_enabled() -> dict[str, bool]:
    data = _read_settings_file()
    raw = data.get("loss_terms")
    if isinstance(raw, dict):
        return normalize_loss_terms_enabled({str(k): bool(v) for k, v in raw.items()})
    return default_loss_terms_enabled()


def load_loss_coeffs() -> dict[str, float]:
    data = _read_settings_file()
    raw = data.get("loss_coeffs")
    if isinstance(raw, dict):
        return normalize_loss_coeffs({str(k): float(v) for k, v in raw.items()})
    return default_loss_coeffs()


def save_loss_coeffs(coeffs: dict[str, float]) -> dict[str, float]:
    data = _read_settings_file()
    normalized = normalize_loss_coeffs(coeffs)
    data["loss_coeffs"] = normalized
    _write_settings_file(data)
    return normalized


def save_loss_terms_enabled(enabled: dict[str, bool]) -> dict[str, bool]:
    data = _read_settings_file()
    normalized = normalize_loss_terms_enabled(enabled)
    data["loss_terms"] = normalized
    _write_settings_file(data)
    return normalized


def load_experiment_config() -> ExperimentConfig:
    data = _read_settings_file()
    raw = {**EXPERIMENT_DEFAULTS, **data.get("experiment", {})}
    return ExperimentConfig(
        checkpoint_dir=Path(str(raw["checkpoint_dir"])),
        batch_size=int(raw["batch_size"]),
        lr=float(raw["lr"]),
        seed=int(raw["seed"]),
        loss_terms_enabled=load_loss_terms_enabled(),
    )


def save_settings(settings: dict[str, float]) -> dict[str, float]:
    data = _read_settings_file()
    merged = load_settings()
    merged.update(settings)
    merged["checkpoint_refresh_sec"] = _clamp_refresh(float(merged["checkpoint_refresh_sec"]))
    merged["autoregressive_window"] = float(
        _clamp_autoregressive_window(float(merged.get("autoregressive_window", DEFAULTS["autoregressive_window"])))
    )
    data["checkpoint_refresh_sec"] = merged["checkpoint_refresh_sec"]
    data["autoregressive_window"] = int(merged["autoregressive_window"])
    _write_settings_file(data)
    return merged


def save_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    data = _read_settings_file()
    data["experiment"] = {
        "checkpoint_dir": str(config.checkpoint_dir).replace("\\", "/"),
        "batch_size": config.batch_size,
        "lr": config.lr,
        "seed": config.seed,
    }
    if "checkpoint_refresh_sec" not in data:
        data["checkpoint_refresh_sec"] = DEFAULTS["checkpoint_refresh_sec"]
    data["loss_terms"] = normalize_loss_terms_enabled(config.loss_terms_enabled)
    _write_settings_file(data)
    return config


class SettingsStore:
    """Cached settings reader that reloads when the file changes."""

    def __init__(self) -> None:
        self._cache = load_settings()
        self._experiment_cache = load_experiment_config()
        self._loss_terms_cache = load_loss_terms_enabled()
        self._loss_coeffs_cache = load_loss_coeffs()
        self._mtime = self._read_mtime()

    @staticmethod
    def _read_mtime() -> float:
        if SETTINGS_PATH.is_file():
            return SETTINGS_PATH.stat().st_mtime
        return 0.0

    def _reload_if_changed(self) -> None:
        mtime = self._read_mtime()
        if mtime != self._mtime:
            self._cache = load_settings()
            self._experiment_cache = load_experiment_config()
            self._loss_terms_cache = load_loss_terms_enabled()
            self._loss_coeffs_cache = load_loss_coeffs()
            self._mtime = mtime

    def get(self) -> dict[str, float]:
        self._reload_if_changed()
        return self._cache

    def get_experiment_config(self) -> ExperimentConfig:
        self._reload_if_changed()
        return self._experiment_cache

    def refresh(self) -> dict[str, float]:
        self._cache = load_settings()
        self._experiment_cache = load_experiment_config()
        self._loss_terms_cache = load_loss_terms_enabled()
        self._loss_coeffs_cache = load_loss_coeffs()
        self._mtime = self._read_mtime()
        return self._cache

    def loss_terms_enabled(self) -> dict[str, bool]:
        self._reload_if_changed()
        return self._loss_terms_cache.copy()

    def loss_coeffs(self) -> dict[str, float]:
        self._reload_if_changed()
        return self._loss_coeffs_cache.copy()

    def save(
        self,
        settings: dict[str, float],
        loss_terms: dict[str, bool] | None = None,
        loss_coeffs: dict[str, float] | None = None,
    ) -> dict[str, float]:
        self._cache = save_settings(settings)
        if loss_terms is not None:
            self._loss_terms_cache = save_loss_terms_enabled(loss_terms)
        if loss_coeffs is not None:
            self._loss_coeffs_cache = save_loss_coeffs(loss_coeffs)
        self._experiment_cache = load_experiment_config()
        self._mtime = self._read_mtime()
        return self._cache

    def save_experiment_config(self, config: ExperimentConfig) -> ExperimentConfig:
        self._experiment_cache = save_experiment_config(config)
        self._loss_terms_cache = load_loss_terms_enabled()
        self._loss_coeffs_cache = load_loss_coeffs()
        self._cache = load_settings()
        self._mtime = self._read_mtime()
        return self._experiment_cache

    def checkpoint_refresh_ms(self) -> int:
        return int(self.get()["checkpoint_refresh_sec"] * 1000)

    def checkpoint_refresh_sec(self) -> float:
        return self.get()["checkpoint_refresh_sec"]

    def autoregressive_window(self) -> int:
        return _clamp_autoregressive_window(float(self.get()["autoregressive_window"]))
