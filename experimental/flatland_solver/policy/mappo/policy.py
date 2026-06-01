from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy

from utils.model_utils import ActorCriticNet, infer_obs_dim, split_obs_and_mask


class MAPPOPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    """Checkpoint-aware MAPPO-like actor policy (greedy inference)."""

    def __init__(self, seed: int = 42, checkpoint_path: str | None = None):
        super().__init__()
        self.seed = seed
        self.obs_dim = 36
        self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=5)
        self.model.eval()
        self.loaded = False
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.exists():
                payload = torch.load(path, map_location="cpu")
                self.obs_dim = int(payload.get("obs_dim", self.obs_dim))
                if self.obs_dim != self.model.obs_dim:
                    self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=5)
                self.model.load_state_dict(payload["model_state"])
                self.loaded = True

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        return {h: self.act(observations[idx]) for idx, h in enumerate(handles)}

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        if not self.loaded:
            inferred = infer_obs_dim(observation, default=self.obs_dim)
            if inferred != self.obs_dim:
                self.obs_dim = inferred
                self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=5)
                self.model.eval()

        features, mask = split_obs_and_mask(observation, obs_dim=self.obs_dim)
        with torch.no_grad():
            logits, _ = self.model(features)
            logits = logits.masked_fill(mask < 0.5, float("-inf"))
            action = int(torch.argmax(logits).item())
        return RailEnvActions(action)
