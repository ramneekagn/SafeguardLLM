from torch import Tensor, nn
import torch

class MLPProbe(nn.Module):
    """MLP probe to monitor activations

    Attributes:
        network: A 3-layer feedforward network producing a single binary output.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512):
        """ Initializes the MLPProbe model

        Args:
            in_dim: Dimensionality of the input vector for the model
            hidden_dim: Number of units inside the hidden layers. Defaults to 512.

        """
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
