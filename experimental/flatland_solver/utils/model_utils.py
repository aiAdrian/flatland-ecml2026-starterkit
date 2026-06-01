from __future__ import annotations

import torch
import torch.nn as nn


def _unwrap_observation_vector(observation):
    # Legacy observations may come as tuples/lists where the first item is the base vector.
    if isinstance(observation, (tuple, list)) and len(observation) > 0:
        if torch.is_tensor(observation[0]) or hasattr(observation[0], "__array__"):
            return observation[0]
    return observation


def infer_obs_dim(observation, default: int = 36) -> int:
    vec = torch.as_tensor(_unwrap_observation_vector(observation), dtype=torch.float32).flatten()
    if vec.numel() <= 0:
        return int(default)
    if vec.numel() >= 41:
        # FastTree layout: 36 features + 5 mask.
        return 36
    return int(vec.numel())


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


def split_obs_and_mask(observation, obs_dim: int = 36) -> tuple[torch.Tensor, torch.Tensor]:
    obs_dim = int(obs_dim)
    t = torch.as_tensor(_unwrap_observation_vector(observation), dtype=torch.float32).flatten()
    features = t[:obs_dim]
    if features.numel() < obs_dim:
        pad = torch.zeros(obs_dim - features.numel(), dtype=torch.float32)
        features = torch.cat([features, pad], dim=0)

    mask = t[obs_dim : obs_dim + 5]
    if mask.numel() != 5:
        mask = torch.ones(5, dtype=torch.float32)
    # Guarantee at least one valid action so logits masking stays finite.
    if torch.sum(mask > 0.5).item() == 0:
        mask = torch.ones(5, dtype=torch.float32)
    return features, mask
