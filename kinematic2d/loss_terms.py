"""Registry of training loss terms (single source for trainer, viz, settings)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossTermSpec:
    key: str
    label: str
    default_enabled: bool = True


LOSS_TERMS: tuple[LossTermSpec, ...] = (
    LossTermSpec("pelvis_tilt", "Pelvis tilt"),
    LossTermSpec("foot_pin", "Foot pin"),
    LossTermSpec("limb_radius", "Limb radius"),
)


def loss_term_keys() -> tuple[str, ...]:
    return tuple(spec.key for spec in LOSS_TERMS)


def default_loss_terms_enabled() -> dict[str, bool]:
    return {spec.key: spec.default_enabled for spec in LOSS_TERMS}


def normalize_loss_terms_enabled(raw: dict[str, bool] | None) -> dict[str, bool]:
    """Merge saved toggles with registry defaults (auto-adds new terms)."""
    merged = default_loss_terms_enabled()
    if raw:
        for spec in LOSS_TERMS:
            if spec.key in raw:
                merged[spec.key] = bool(raw[spec.key])
    return merged
