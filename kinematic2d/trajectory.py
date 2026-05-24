"""Predetermined root motion trajectories."""

from __future__ import annotations

import math

from kinematic2d.transforms import Transform2D


TRAJECTORY_COUNT = 5000
TRAJECTORY_SPEED = 0.025
TRAJECTORY_ANGLE = math.pi / 2


def forward_constant_trajectory(
    count: int = TRAJECTORY_COUNT,
    speed: float = TRAJECTORY_SPEED,
    start: tuple[float, float] = (0.0, 0.0),
) -> list[Transform2D]:
    """Root faces +Y and advances upward at constant speed."""
    x0, y0 = start
    return [
        Transform2D(x0, y0 + speed * i, TRAJECTORY_ANGLE)
        for i in range(count)
    ]
