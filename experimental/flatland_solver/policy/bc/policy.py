from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy

from utils.model_utils import ActorCriticNet, DiscretePolicyNet, infer_obs_dim, split_obs_and_mask


class BCPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__()
        self.obs_dim = 36
        self.model = DiscretePolicyNet(obs_dim=self.obs_dim, action_dim=5)
        self._actor_critic_mode = False
        self.model.eval()
        self.loaded = False
        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.exists():
                payload = torch.load(path, map_location="cpu")
                self.obs_dim = int(payload.get("obs_dim", self.obs_dim))
                state = payload.get("model_state", {})
                if any(str(k).startswith("backbone.") for k in state.keys()):
                    self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=5)
                    self._actor_critic_mode = True
                else:
                    self.model = DiscretePolicyNet(obs_dim=self.obs_dim, action_dim=5)
                    self._actor_critic_mode = False
                self.model.load_state_dict(state, strict=False)
                self.model.eval()
                self.loaded = True

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        return {h: self.act(observations[idx]) for idx, h in enumerate(handles)}

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        if not self.loaded:
            inferred = infer_obs_dim(observation, default=self.obs_dim)
            if inferred != self.obs_dim:
                self.obs_dim = inferred
                if self._actor_critic_mode:
                    self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=5)
                else:
                    self.model = DiscretePolicyNet(obs_dim=self.obs_dim, action_dim=5)
                self.model.eval()

        features, mask = split_obs_and_mask(observation, obs_dim=self.obs_dim)
        with torch.no_grad():
            if self._actor_critic_mode:
                logits, _ = self.model(features)
            else:
                logits = self.model(features)
            logits = logits.masked_fill(mask < 0.5, float("-inf"))
            action = int(torch.argmax(logits).item())
        return RailEnvActions(action)
