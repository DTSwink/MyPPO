"""PyTorch MLP policy (same layout as NumPy inference agent)."""

from __future__ import annotations

import torch
import torch.nn as nn

from kinematic2d.action_codec import sanitize_action_torch
from kinematic2d.agent import HIDDEN_DIM, INPUT_DIM, OUTPUT_DIM


class TorchMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.fc1(x))
        return sanitize_action_torch(self.fc2(x))

    def numpy_weights(self) -> dict[str, "torch.Tensor"]:
        w1 = self.fc1.weight.detach().T.contiguous().cpu().numpy()
        b1 = self.fc1.bias.detach().cpu().numpy()
        w2 = self.fc2.weight.detach().T.contiguous().cpu().numpy()
        b2 = self.fc2.bias.detach().cpu().numpy()
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
