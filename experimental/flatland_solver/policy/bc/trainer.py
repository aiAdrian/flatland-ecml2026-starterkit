from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from policy.dla.observation import DLAFullEnvObservation
from policy.dla.policy import DLAPolicy
from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env
from utils.model_utils import DiscretePolicyNet, split_obs_and_mask


def train_bc(args, checkpoint_path: Path) -> Path:
    obs_builder = args.obs_builder
    cfg = SolverConfig(
        width=args.width,
        height=args.height,
        n_agents=args.n_agents,
        n_cities=args.n_cities,
        max_rails_between_cities=args.max_rails_between_cities,
        max_rail_pairs_in_city=args.max_rail_pairs_in_city,
        seed=args.seed,
    )
    env = build_env(cfg=cfg, obs_builder=obs_builder)

    expert = DLAPolicy(seed=args.seed)
    expert_obs_builder = DLAFullEnvObservation()
    del expert_obs_builder  # expert receives env directly in act_many

    model = DiscretePolicyNet(obs_dim=36, action_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.train_epochs):
        losses = []
        for ep in range(args.episodes):
            observations, info = env.reset(random_seed=args.seed + ep + epoch * 1000)
            del info
            done = {"__all__": False}
            steps = 0

            while not done.get("__all__", False) and steps < args.max_episode_steps:
                handles = list(range(env.get_num_agents()))
                # DLA expects RailEnv observations.
                expert_actions = expert.act_many(handles, [env for _ in handles])
                norm_expert_actions = normalize_actions(expert_actions)

                obs_batch = [observations[h] for h in handles]
                feats = []
                masks = []
                labels = []
                for h, obs in zip(handles, obs_batch):
                    f, m = split_obs_and_mask(obs)
                    feats.append(f)
                    masks.append(m)
                    labels.append(norm_expert_actions[h])

                feat_t = torch.stack(feats, dim=0)
                mask_t = torch.stack(masks, dim=0)
                row_has_valid = torch.sum(mask_t > 0.5, dim=1) > 0
                if torch.any(~row_has_valid):
                    mask_t[~row_has_valid] = 1.0
                label_t = torch.as_tensor(labels, dtype=torch.long)
                for i in range(mask_t.shape[0]):
                    mask_t[i, label_t[i]] = 1.0

                logits = model(feat_t)
                logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
                loss = F.cross_entropy(logits, label_t)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(float(loss.item()))
                observations, rewards, done, info = env.step(norm_expert_actions)
                del rewards, info
                steps += 1

        avg_loss = sum(losses) / max(1, len(losses))
        print(f"[train-bc] epoch={epoch + 1}/{args.train_epochs} avg_loss={avg_loss:.4f}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": 36,
            "action_dim": 5,
            "kind": "bc",
        },
        checkpoint_path,
    )
    print(f"[train-bc] checkpoint={checkpoint_path}")
    return checkpoint_path
