from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn


class TreePayloadEncoder(nn.Module):
    """Structured tree payload encoder (node/edge message passing)."""

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
