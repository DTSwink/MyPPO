"""Shared flag so the reload wrapper knows the user closed the window."""

from __future__ import annotations

from pathlib import Path

MARKER = Path(__file__).resolve().parent.parent / ".viz_user_closed"


def mark() -> None:
    MARKER.write_text("1", encoding="utf-8")


def clear() -> None:
    if MARKER.exists():
        MARKER.unlink()


def was_user_close() -> bool:
    return MARKER.exists()
