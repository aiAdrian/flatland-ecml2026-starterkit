from __future__ import annotations

import torch
import torch.nn as nn


class ActorCriticHead(nn.Module):
    """Legacy MAPPO actor+critic head with neighbor-pool critic input."""

    def __init__(self, hidden: int, action_size: int, base_dim: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_size),
        )
        self.neighbor_proj = nn.Sequential(
            nn.Linear(base_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def actor_logits(self, self_emb: torch.Tensor) -> torch.Tensor:
        return self.actor(self_emb)

    def value(self, self_emb: torch.Tensor, neighbor_pool: torch.Tensor) -> torch.Tensor:
        n_emb = self.neighbor_proj(neighbor_pool)
        x = torch.cat([self_emb, n_emb], dim=-1)
        return self.critic(x).squeeze(-1)
