import numpy as np
import torch


class TimeEncode(torch.nn.Module):
    """Trainable harmonic time encoding used by temporal graph models."""

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension))
            .float()
            .reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        return torch.cos(self.w(t.unsqueeze(dim=2)))


class FixedTimeEncode(torch.nn.Module):
    """Fixed harmonic time encoding with non-trainable frequencies."""

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.reset_parameters()

    def reset_parameters(self):
        self.w.weight = torch.nn.Parameter(
            torch.from_numpy(
                1 / 10 ** np.linspace(0, 9, self.dimension, dtype=np.float32)
            ).reshape(self.dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(self.dimension))
        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False

    @torch.no_grad()
    def forward(self, t):
        return torch.cos(self.w(t.unsqueeze(dim=2)))
