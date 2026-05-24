"""Registry of training loss terms (single source for trainer, viz, settings)."""

from __future__ import annotations

from dataclasses import dataclass

MIN_LOSS_COEFF = 1e-6
MAX_LOSS_COEFF = 1e6


@dataclass(frozen=True)
class LossTermSpec:
    key: str
    label: str
    default_enabled: bool = True
    default_coeff: float = 1.0


LOSS_TERMS: tuple[LossTermSpec, ...] = (
    LossTermSpec("pelvis_tilt", "Pelvis tilt", default_coeff=1.0),
    LossTermSpec("foot_pin", "Foot pin", default_coeff=1000.0),
    LossTermSpec("limb_radius", "Limb-root distance", default_coeff=100.0),
    LossTermSpec("foot_stride", "Unpinned stride", default_coeff=1000.0),
)


def loss_term_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in LOSS_TERMS)


def default_loss_terms_enabled() -> dict[str, bool]:
    return {spec.key: spec.default_enabled for spec in LOSS_TERMS}


def default_loss_coeffs() -> dict[str, float]:
    return {spec.key: spec.default_coeff for spec in LOSS_TERMS}


def normalize_loss_terms_enabled(raw: dict[str, bool] | None) -> dict[str, bool]:
    """Merge saved toggles with registry defaults (auto-adds new terms)."""
    merged = default_loss_terms_enabled()
    if raw:
        for spec in LOSS_TERMS:
            if spec.key in raw:
                merged[spec.key] = bool(raw[spec.key])
    return merged


def normalize_loss_coeffs(raw: dict[str, float] | None) -> dict[str, float]:
    """Merge saved coefficients with registry defaults (auto-adds new terms)."""
    merged = default_loss_coeffs()
    if raw:
        for spec in LOSS_TERMS:
            if spec.key in raw:
                merged[spec.key] = _clamp_coeff(float(raw[spec.key]))
    return merged


def _clamp_coeff(value: float) -> float:
    return max(MIN_LOSS_COEFF, min(MAX_LOSS_COEFF, value))
