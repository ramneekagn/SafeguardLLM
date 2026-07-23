from abc import ABC, abstractmethod
from safeguard_llm.detectors.internal.internal_detector import InternalDetector
from torch import Tensor, nn
import torch

class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        probs = torch.sigmoid(self.forward(x))
        return (probs >= threshold).float()
