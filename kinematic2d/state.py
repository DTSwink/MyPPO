"""Simulation state: root trajectory + limb poses on a 2D plane."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kinematic2d.transforms import Transform2D, global_to_local, local_to_global


LIMB_NAMES = ("pelvis", "foot_left", "foot_right")


@dataclass
class LimbState:
    pelvis: Transform2D
    foot_left: Transform2D
    foot_right: Transform2D
    foot_left_height: float = 0.0
    foot_right_height: float = 0.0
    prev_foot_left_height: float = 0.0
    prev_foot_right_height: float = 0.0

    def limbs_global(self, root: Transform2D) -> dict[str, Transform2D]:
        return {
            "pelvis": local_to_global(root, self.pelvis),
            "foot_left": local_to_global(root, self.foot_left),
            "foot_right": local_to_global(root, self.foot_right),
        }

    def limbs_local_in_root(self) -> dict[str, Transform2D]:
        return {
            "pelvis": self.pelvis.copy(),
            "foot_left": self.foot_left.copy(),
            "foot_right": self.foot_right.copy(),
        }


def random_limb_state(rng: np.random.Generator | None = None) -> LimbState:
    rng = rng or np.random.default_rng()

    def rand_tf() -> Transform2D:
        return Transform2D(
            float(rng.uniform(-0.15, 0.15)),
            float(rng.uniform(-0.15, 0.15)),
            float(rng.uniform(-math.pi, math.pi)),
        )

    return LimbState(
        pelvis=rand_tf(),
        foot_left=rand_tf(),
        foot_right=rand_tf(),
        foot_left_height=float(rng.uniform(0.0, 0.1)),
        foot_right_height=float(rng.uniform(0.0, 0.1)),
        prev_foot_left_height=0.0,
        prev_foot_right_height=0.0,
    )


@dataclass
class Simulation:
    roots: list[Transform2D]
    frame_index: int = 0
    current_limbs: LimbState | None = None
    previous_limbs: LimbState | None = None
    future_horizon: int = 8
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self) -> None:
        if self.current_limbs is None:
            self.current_limbs = random_limb_state(self._rng)
        if self.previous_limbs is None:
            self.previous_limbs = random_limb_state(self._rng)

    @property
    def current_root(self) -> Transform2D:
        return self.roots[self.frame_index]

    @property
    def next_root(self) -> Transform2D | None:
        if self.frame_index + 1 >= len(self.roots):
            return None
        return self.roots[self.frame_index + 1]

    @property
    def finished(self) -> bool:
        return self.frame_index >= len(self.roots) - 1

    def reset(self) -> None:
        self.frame_index = 0
        self.current_limbs = random_limb_state(self._rng)
        self.previous_limbs = random_limb_state(self._rng)

    def build_agent_input(self) -> dict[str, np.ndarray]:
        root = self.current_root
        future_roots = []
        for i in range(1, self.future_horizon + 1):
            idx = min(self.frame_index + i, len(self.roots) - 1)
            future_roots.append(global_to_local(root, self.roots[idx]).to_array())
        future_roots_arr = np.stack(future_roots, axis=0)

        current_local = self.current_limbs.limbs_local_in_root()
        current_arr = np.stack([current_local[n].to_array() for n in LIMB_NAMES], axis=0)

        previous_local = self.previous_limbs.limbs_local_in_root()
        previous_arr = np.stack([previous_local[n].to_array() for n in LIMB_NAMES], axis=0)

        return {
            "future_roots": future_roots_arr,
            "current_limbs": current_arr,
            "previous_limbs": previous_arr,
        }

    def apply_agent_output(self, output: dict[str, np.ndarray]) -> bool:
        if self.finished:
            return False

        next_limbs = LimbState(
            pelvis=Transform2D.from_array(output["future_limbs"][0]),
            foot_left=Transform2D.from_array(output["future_limbs"][1]),
            foot_right=Transform2D.from_array(output["future_limbs"][2]),
            foot_left_height=float(output["foot_left_height"]),
            foot_right_height=float(output["foot_right_height"]),
            prev_foot_left_height=float(output["prev_foot_left_height"]),
            prev_foot_right_height=float(output["prev_foot_right_height"]),
        )

        self.previous_limbs = self.current_limbs
        self.current_limbs = next_limbs
        self.frame_index += 1
        return True
