"""2D transform utilities for root-relative kinematics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Transform2D:
    x: float
    y: float
    angle: float  # radians, 0 = +X forward

    def copy(self) -> Transform2D:
        return Transform2D(self.x, self.y, self.angle)

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.angle], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> Transform2D:
        return Transform2D(float(arr[0]), float(arr[1]), float(arr[2]))


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def local_to_global(root: Transform2D, local: Transform2D) -> Transform2D:
    rot = rotation_matrix(root.angle)
    pos = rot @ np.array([local.x, local.y])
    return Transform2D(
        root.x + float(pos[0]),
        root.y + float(pos[1]),
        root.angle + local.angle,
    )


def global_to_local(root: Transform2D, global_tf: Transform2D) -> Transform2D:
    rot = rotation_matrix(root.angle)
    delta = np.array([global_tf.x - root.x, global_tf.y - root.y])
    local_pos = rot.T @ delta
    return Transform2D(
        float(local_pos[0]),
        float(local_pos[1]),
        global_tf.angle - root.angle,
    )


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
