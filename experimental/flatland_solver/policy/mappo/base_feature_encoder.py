from __future__ import annotations

import torch
import torch.nn as nn


class BaseFeatureEncoder(nn.Module):
    """MAPPO base-feature encoder (22D -> hidden)."""

    def __init__(self, base_dim: int = 22, hidden: int = 64):
        super().__init__()
        self.base_dim = int(base_dim)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.base_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.LeakyReLU(0.01),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
