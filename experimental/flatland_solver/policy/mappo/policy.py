from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.envs.step_utils.states import TrainState

from utils.model_utils import ActorCriticNet, infer_obs_dim, split_obs_and_mask


class BaseFeatureEncoder(nn.Module):
    """Legacy MAPPO base-feature encoder (22D -> hidden)."""

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


class TreePayloadEncoder(nn.Module):
    """Legacy-style tree payload encoder (node/edge message passing)."""

    NODE_DIM = 12
    EDGE_DIM = 20
    MAX_NODES = 32

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = int(hidden)
        self.node_proj = nn.Sequential(
            nn.Linear(self.NODE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(self.EDGE_DIM, hidden // 2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden // 2, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.01),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Tanh(),
        )

    @staticmethod
    def _encode_node(n: Dict[str, Any]) -> np.ndarray:
        depth = int(n.get("depth", 0))
        is_branch = bool(n.get("is_branch", False) or bool(n.get("is_switch", False)))
        alt = int(n.get("alternative_routes_count", 0))
        feat = np.array([
            float(n.get("deadlock_risk", 0.0)),
            min(1.0, float(n.get("num_transitions", 1)) / 3.0),
            1.0 if n.get("has_oncoming", False) else 0.0,
            min(1.0, float(n.get("backward_inflow_count", 0)) / 2.0),
            min(1.0, float(max(depth, 0)) / 12.0),
            1.0 if n.get("has_agents_encountered", False) else 0.0,
            min(1.0, float(n.get("incoming_agent_count", 0)) / 2.0),
            1.0 if is_branch else 0.0,
            float(n.get("deadlock_distance_norm", 0.0)),
            float(n.get("deadlock_hard_distance_norm", 0.0)),
            1.0 if n.get("deadlock_exists_within_probe", False) else 0.0,
            min(1.0, float(alt) / 3.0),
        ], dtype=np.float32)
        np.clip(feat, 0.0, 1.0, out=feat)
        return feat

    @staticmethod
    def _encode_edge(e: Dict[str, Any]) -> np.ndarray:
        def _f(v, d=0.0):
            try:
                x = float(v)
                return x if np.isfinite(x) else d
            except (TypeError, ValueError):
                return d

        def _norm_d(v):
            d = _f(v, 1.0)
            return float(d) if d <= 1.0 else float(d / (d + 32.0))

        rel = int(e.get("rel_dir_bin", 1))
        a_l = _f(e.get("action_left", 1.0 if rel == 0 else 0.0))
        a_f = _f(e.get("action_forward", 1.0 if rel == 1 else 0.0))
        a_r = _f(e.get("action_right", 1.0 if rel == 2 else 0.0))
        n_agents_edge = int(e.get("agents_on_edge_count", 0))
        if n_agents_edge <= 0:
            agents_on = e.get("agents_on_edge", [])
            if isinstance(agents_on, list):
                n_agents_edge = len(agents_on)

        dl_delta = _f(e.get("deadlock_distance_delta", 0.0))
        dl_delta_unit = 0.5 * (max(-1.0, min(1.0, dl_delta)) + 1.0)
        feat = np.array([
            a_l, a_f, a_r,
            1.0 if n_agents_edge > 0 else 0.0,
            1.0 if e.get("has_oncoming_edge", False) else 0.0,
            min(1.0, _f(e.get("edge_len_cells", 1), 1.0) / 4.0),
            _norm_d(e.get("src_dist_to_target", 1.0)),
            _norm_d(e.get("dst_dist_to_target", 1.0)),
            _f(e.get("delta_from_root", 0.5), 0.5),
            _f(e.get("improves_over_current", 0.0)),
            1.0 if e.get("target_on_edge", False) else 0.0,
            _f(e.get("dst_deadlock_risk", 0.0)),
            _f(e.get("dst_deadlock_hard_block", 0.0)),
            _f(e.get("dst_deadlock_distance_norm", 0.0)),
            _f(e.get("dst_deadlock_hard_distance_norm", 0.0)),
            _f(e.get("src_deadlock_distance_norm", 0.0)),
            dl_delta_unit,
            _f(e.get("is_shortest_path_edge", 0.0)),
            _f(e.get("branch_choice_prob", 0.0)),
            min(1.0, _f(e.get("alternative_routes_count", 0)) / 3.0),
        ], dtype=np.float32)
        np.clip(feat, 0.0, 1.0, out=feat)
        return feat

    def _payload_to_graph(self, payload: Dict[str, Any], max_nodes: int):
        max_nodes = max(1, int(max_nodes))
        node_feats = np.zeros((max_nodes, self.NODE_DIM), dtype=np.float32)
        edge_list: List[tuple[int, int, np.ndarray]] = []
        if not isinstance(payload, dict):
            return node_feats, edge_list, 0

        nodes = payload.get("nodes", []) or []
        edges = payload.get("edges", []) or []

        idx_simple: Dict[tuple[int, int, int], int] = {}
        idx_pos: Dict[tuple[int, int], List[tuple[int, int]]] = defaultdict(list)

        n_valid = min(len(nodes), max_nodes)
        for i in range(n_valid):
            node = nodes[i]
            pos = node.get("pos", (0, 0))
            d = int(node.get("dir", 0))
            depth = int(node.get("depth", 0))
            node_feats[i] = self._encode_node(node)
            idx_simple[(int(pos[0]), int(pos[1]), d)] = i
            idx_pos[(int(pos[0]), int(pos[1]))].append((depth, i))

        for plist in idx_pos.values():
            plist.sort(key=lambda x: x[0])

        def _resolve(key3: tuple[int, int, int]) -> int | None:
            i = idx_simple.get(key3)
            if i is not None:
                return i
            pos_list = idx_pos.get((key3[0], key3[1]))
            if pos_list:
                return pos_list[0][1]
            return None

        for edge in edges:
            if "src" in edge and "dst" in edge:
                s_i = int(edge["src"])
                d_i = int(edge["dst"])
            else:
                s_pos = edge.get("src_pos", (0, 0))
                d_pos = edge.get("dst_pos", (0, 0))
                s_dir = int(edge.get("src_dir", 0))
                d_dir = int(edge.get("dst_dir", 0))
                s_i = _resolve((int(s_pos[0]), int(s_pos[1]), s_dir))
                d_i = _resolve((int(d_pos[0]), int(d_pos[1]), d_dir))
                if s_i is None or d_i is None:
                    continue
            if s_i < 0 or d_i < 0 or s_i >= max_nodes or d_i >= max_nodes:
                continue
            edge_list.append((s_i, d_i, self._encode_edge(edge)))

        return node_feats, edge_list, n_valid

    def forward_batch(self, payload_list: List[Dict[str, Any]]) -> torch.Tensor:
        device = next(self.parameters()).device
        bsz = len(payload_list)
        if bsz == 0:
            return torch.empty(0, self.hidden, device=device)

        max_nodes = 1
        for p in payload_list:
            if isinstance(p, dict):
                max_nodes = max(max_nodes, len(p.get("nodes", []) or []))
        max_nodes = min(self.MAX_NODES, max_nodes)

        node_arr = np.zeros((bsz, max_nodes, self.NODE_DIM), dtype=np.float32)
        node_mask = torch.zeros((bsz, max_nodes), dtype=torch.float32, device=device)
        edge_graphs = []

        for b, payload in enumerate(payload_list):
            n_feat, e_list, n_valid = self._payload_to_graph(payload, max_nodes)
            node_arr[b] = n_feat
            edge_graphs.append(e_list)
            if n_valid > 0:
                node_mask[b, :n_valid] = 1.0

        nodes = torch.as_tensor(node_arr, device=device)
        h_node = self.node_proj(nodes)
        messages = torch.zeros_like(h_node)

        for b, e_list in enumerate(edge_graphs):
            for s_i, d_i, e_feat_np in e_list:
                if s_i >= max_nodes or d_i >= max_nodes:
                    continue
                e_t = torch.as_tensor(e_feat_np, device=device)
                gate = torch.sigmoid(self.edge_gate(e_t)).squeeze(-1)
                messages[b, d_i] += gate * h_node[b, s_i]

        h_updated = self.update(torch.cat([h_node, messages], dim=-1))
        mask_exp = node_mask.unsqueeze(-1)
        denom = mask_exp.sum(dim=1).clamp(min=1.0)
        pooled = (h_updated * mask_exp).sum(dim=1) / denom
        return self.output(pooled)


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


class RolloutBuffer:
    """Legacy-compatible rollout buffer container for future trainer integration."""

    def __init__(self):
        self.data: Dict[int, List[tuple]] = {}

    def push(self, handle: int, transition: tuple):
        self.data.setdefault(int(handle), []).append(transition)

    def get(self, handle: int) -> List[tuple]:
        return self.data.get(int(handle), [])

    def reset(self):
        self.data = {}

    def all_transitions(self) -> List[tuple]:
        out = []
        for ts in self.data.values():
            out.extend(ts)
        return out

    def __len__(self):
        return sum(len(v) for v in self.data.values())


class MAPPOPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    """Checkpoint-aware MAPPO actor policy.

    Supports both:
    - current compact checkpoint format (`model_state` from ActorCriticNet)
    - legacy checkpoint format (`base_encoder`, `tree_encoder`, `fuse`, `head`)
    """

    def __init__(self, seed: int = 42, checkpoint_path: str | None = None):
        super().__init__()
        self.seed = seed
        self.action_size = 5

        # Current compact path.
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
