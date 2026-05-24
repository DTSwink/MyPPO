"""Randomly initialized MLP policy for limb control (NumPy inference, fast startup)."""

from __future__ import annotations

import numpy as np

from kinematic2d.action_codec import sanitize_action_numpy
from kinematic2d.state import LIMB_NAMES

FUTURE_ROOTS_DIM = 8 * 3
CURRENT_LIMBS_DIM = 3 * 3
PREVIOUS_LIMBS_DIM = 3 * 3
INPUT_DIM = FUTURE_ROOTS_DIM + CURRENT_LIMBS_DIM + PREVIOUS_LIMBS_DIM
OUTPUT_LIMBS_DIM = 3 * 3
OUTPUT_HEIGHTS_DIM = 4
OUTPUT_DIM = OUTPUT_LIMBS_DIM + OUTPUT_HEIGHTS_DIM
HIDDEN_DIM = 128


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class MLPAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._init_weights(seed)

    def _init_weights(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / INPUT_DIM)
        scale2 = np.sqrt(2.0 / HIDDEN_DIM)
        self.w1 = rng.standard_normal((INPUT_DIM, HIDDEN_DIM)).astype(np.float32) * scale1
        self.b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
        self.w2 = rng.standard_normal((HIDDEN_DIM, OUTPUT_DIM)).astype(np.float32) * scale2
        self.b2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

    def reinit_weights(self, seed: int | None = None) -> None:
        self._init_weights(seed)

    def load_weights(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32)

    @staticmethod
    def flatten_input(agent_input: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [
                agent_input["future_roots"].reshape(-1),
                agent_input["current_limbs"].reshape(-1),
                agent_input["previous_limbs"].reshape(-1),
            ],
            dtype=np.float32,
        )

    def predict(self, agent_input: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = self.flatten_input(agent_input)
        hidden = _gelu(x @ self.w1 + self.b1)
        out = sanitize_action_numpy(hidden @ self.w2 + self.b2)

        future_limbs = out[:OUTPUT_LIMBS_DIM].reshape(len(LIMB_NAMES), 3)
        heights = out[OUTPUT_LIMBS_DIM:]

        return {
            "future_limbs": future_limbs.astype(np.float32),
            "foot_left_height": np.float32(heights[0]),
            "foot_right_height": np.float32(heights[1]),
            "prev_foot_left_height": np.float32(heights[2]),
            "prev_foot_right_height": np.float32(heights[3]),
        }

    @staticmethod
    def io_spec() -> dict[str, tuple[int, ...] | tuple]:
        return {
            "input_dim": (INPUT_DIM,),
            "future_roots": (8, 3),
            "current_limbs": (len(LIMB_NAMES), 3),
            "previous_limbs": (len(LIMB_NAMES), 3),
            "future_limbs": (len(LIMB_NAMES), 3),
            "foot_left_height": (),
            "foot_right_height": (),
            "prev_foot_left_height": (),
            "prev_foot_right_height": (),
        }
