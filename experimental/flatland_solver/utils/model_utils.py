from __future__ import annotations

import torch
import torch.nn as nn


class DiscretePolicyNet(nn.Module):
    def __init__(self, obs_dim: int = 36, action_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int = 36, action_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, self.action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def split_obs_and_mask(observation) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.as_tensor(observation, dtype=torch.float32)
    features = t[:36]
    mask = t[36:41]
    if mask.numel() != 5:
        mask = torch.ones(5, dtype=torch.float32)
    # Guarantee at least one valid action so logits masking stays finite.
    if torch.sum(mask > 0.5).item() == 0:
        mask = torch.ones(5, dtype=torch.float32)
    return features, mask
