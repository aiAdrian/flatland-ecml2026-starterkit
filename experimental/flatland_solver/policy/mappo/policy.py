from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy

from policy.mappo.actor_critic_head import ActorCriticHead
from policy.mappo.base_feature_encoder import BaseFeatureEncoder
from policy.mappo.tree_payload_encoder import TreePayloadEncoder
from utils.model_utils import ActorCriticNet, infer_obs_dim, split_obs_and_mask


class MAPPOPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    """Checkpoint-aware MAPPO actor policy.

    Supports both:
    - compact checkpoint format (`model_state` from ActorCriticNet)
    - legacy checkpoint format (`base_encoder`, `tree_encoder`, `fuse`, `head`)
    """

    def __init__(self, seed: int = 42, checkpoint_path: str | None = None):
        super().__init__()
        self.seed = seed
        self.action_size = 5

        # Compact path.
        self.obs_dim = 36
        self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=self.action_size)
        self.model.eval()

        # Legacy path.
        self.base_dim = 22
        self.hidden = 64
        self.base_encoder: BaseFeatureEncoder | None = None
        self.tree_encoder: TreePayloadEncoder | None = None
        self.fuse: nn.Module | None = None
        self.head: ActorCriticHead | None = None

        self.loaded = False
        self.mode = "compact"  # or "legacy"

        if checkpoint_path:
            path = Path(checkpoint_path)
            if path.exists():
                payload = torch.load(path, map_location="cpu")
                if "model_state" in payload:
                    self.obs_dim = int(payload.get("obs_dim", self.obs_dim))
                    if self.obs_dim != self.model.obs_dim:
                        self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=self.action_size)
                    self.model.load_state_dict(payload["model_state"], strict=False)
                    self.model.eval()
                    self.mode = "compact"
                    self.loaded = True
                elif all(k in payload for k in ["base_encoder", "tree_encoder", "fuse", "head"]):
                    self.hidden = int(payload.get("hidden", self.hidden))
                    self.base_dim = int(payload.get("base_dim", payload.get("obs_dim", self.base_dim)))
                    self.base_encoder = BaseFeatureEncoder(base_dim=self.base_dim, hidden=self.hidden)
                    self.tree_encoder = TreePayloadEncoder(hidden=self.hidden)
                    self.fuse = nn.Sequential(
                        nn.Linear(self.hidden * 2, self.hidden),
                        nn.LayerNorm(self.hidden),
                        nn.LeakyReLU(0.01),
                    )
                    self.head = ActorCriticHead(self.hidden, self.action_size, self.base_dim)

                    self.base_encoder.load_state_dict(payload["base_encoder"], strict=False)
                    self.tree_encoder.load_state_dict(payload["tree_encoder"], strict=False)
                    self.fuse.load_state_dict(payload["fuse"], strict=False)
                    self.head.load_state_dict(payload["head"], strict=False)

                    self.base_encoder.eval()
                    self.tree_encoder.eval()
                    self.fuse.eval()
                    self.head.eval()
                    self.mode = "legacy"
                    self.loaded = True

    @staticmethod
    def _unwrap_state(state) -> tuple[np.ndarray, list[np.ndarray], Dict[str, Any]]:
        if isinstance(state, list) and len(state) > 0 and isinstance(state[0], (tuple, list)):
            state = state[-1]
        if isinstance(state, (tuple, list)):
            if len(state) >= 3:
                obs, opps, payload = state[0], state[1], state[2]
                if not isinstance(opps, list):
                    opps = []
                if not isinstance(payload, dict):
                    payload = {}
                return np.asarray(obs, dtype=np.float32).flatten(), opps, payload
            if len(state) >= 1:
                return np.asarray(state[0], dtype=np.float32).flatten(), [], {}
        return np.asarray(state, dtype=np.float32).flatten(), [], {}

    @staticmethod
    def _neighbor_pool(opps: list[np.ndarray], base_dim: int) -> np.ndarray:
        if not opps:
            return np.zeros(base_dim, dtype=np.float32)
        arrs = []
        for o in opps:
            v = np.asarray(o, dtype=np.float32).flatten()
            if v.shape[0] >= base_dim:
                arrs.append(v[:base_dim])
        if not arrs:
            return np.zeros(base_dim, dtype=np.float32)
        return np.mean(np.stack(arrs, axis=0), axis=0)

    def _legal_action_mask_from_base_obs(self, base_obs: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        # Conservative fallback without agent object: keep STOP/DO_NOTHING legal.
        mask[int(RailEnvActions.DO_NOTHING.value)] = 1.0
        mask[int(RailEnvActions.STOP_MOVING.value)] = 1.0
        if base_obs.shape[0] >= 3:
            left_ok = float(base_obs[0]) > 0.5
            fwd_ok = float(base_obs[1]) > 0.5
            right_ok = float(base_obs[2]) > 0.5
            n_trans = int(left_ok) + int(fwd_ok) + int(right_ok)
            if n_trans == 1:
                mask[int(RailEnvActions.MOVE_FORWARD.value)] = 1.0
            elif n_trans > 1:
                mask[int(RailEnvActions.MOVE_LEFT.value)] = 1.0 if left_ok else 0.0
                mask[int(RailEnvActions.MOVE_FORWARD.value)] = 1.0 if fwd_ok else 0.0
                mask[int(RailEnvActions.MOVE_RIGHT.value)] = 1.0 if right_ok else 0.0
        else:
            mask[int(RailEnvActions.MOVE_FORWARD.value)] = 1.0
        if np.sum(mask > 0.5) <= 0:
            mask[:] = 1.0
        return mask

    def _legacy_forward_logits(self, observation: Any) -> torch.Tensor:
        assert self.base_encoder is not None and self.tree_encoder is not None and self.fuse is not None and self.head is not None
        base_obs, opps, payload = self._unwrap_state(observation)
        if base_obs.shape[0] < self.base_dim:
            pad = np.zeros(self.base_dim - base_obs.shape[0], dtype=np.float32)
            base_obs = np.concatenate([base_obs, pad], axis=0)
        base_obs = base_obs[: self.base_dim]
        n_pool = self._neighbor_pool(opps, self.base_dim)

        with torch.no_grad():
            base_t = torch.from_numpy(base_obs).float().unsqueeze(0)
            pool_t = torch.from_numpy(n_pool).float().unsqueeze(0)
            emb_base = self.base_encoder(base_t)
            emb_tree = self.tree_encoder.forward_batch([payload])
            emb = self.fuse(torch.cat([emb_base, emb_tree], dim=-1))
            logits = self.head.actor_logits(emb).squeeze(0)

            mask_np = self._legal_action_mask_from_base_obs(base_obs)
            mask_t = torch.from_numpy(mask_np).float()
            logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
            return logits

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        return {h: self.act(observations[idx]) for idx, h in enumerate(handles)}

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        if self.mode == "legacy" and self.loaded:
            logits = self._legacy_forward_logits(observation)
            action = int(torch.argmax(logits).item())
            return RailEnvActions(action)

        if not self.loaded:
            inferred = infer_obs_dim(observation, default=self.obs_dim)
            if inferred != self.obs_dim:
                self.obs_dim = inferred
                self.model = ActorCriticNet(obs_dim=self.obs_dim, action_dim=self.action_size)
                self.model.eval()

        features, mask = split_obs_and_mask(observation, obs_dim=self.obs_dim)
        with torch.no_grad():
            logits, _ = self.model(features)
            logits = logits.masked_fill(mask < 0.5, float("-inf"))
            action = int(torch.argmax(logits).item())
        return RailEnvActions(action)
