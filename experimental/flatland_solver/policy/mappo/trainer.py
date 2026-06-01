from __future__ import annotations

from pathlib import Path

import torch
from torch.distributions import Categorical

from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env
from utils.model_utils import ActorCriticNet, split_obs_and_mask


def _discounted_returns(rewards, gamma: float):
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return out


def train_mappo(args, checkpoint_path: Path) -> Path:
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

    model = ActorCriticNet(obs_dim=36, action_dim=5)
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        if "model_state" in payload:
            model.load_state_dict(payload["model_state"], strict=False)
            print(f"[train-mappo] warmstart={args.init_checkpoint}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.train_epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        n_batches = 0

        for ep in range(args.episodes):
            observations, info = env.reset(random_seed=args.seed + epoch * 1000 + ep)
            del info
            done = {"__all__": False}
            steps = 0

            step_logprobs = []
            step_values = []
            step_rewards = []

            while not done.get("__all__", False) and steps < args.max_episode_steps:
                handles = list(range(env.get_num_agents()))
                obs_batch = [observations[h] for h in handles]

                actions = {}
                logprobs = []
                values = []

                for h, obs in zip(handles, obs_batch):
                    feat, mask = split_obs_and_mask(obs)
                    logits, value = model(feat)
                    logits = logits.masked_fill(mask < 0.5, float("-inf"))
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    actions[h] = int(action.item())
                    logprobs.append(dist.log_prob(action))
                    values.append(value)

                observations, rewards, done, info = env.step(normalize_actions(actions))
                del info

                # Team reward signal (mean over agents)
                team_reward = sum(float(rewards[h]) for h in handles) / max(1, len(handles))
                step_rewards.append(team_reward)
                step_logprobs.append(torch.stack(logprobs).mean())
                step_values.append(torch.stack(values).mean())
                steps += 1

            if not step_rewards:
                continue

            returns = torch.as_tensor(_discounted_returns(step_rewards, args.gamma), dtype=torch.float32)
            values_t = torch.stack(step_values)
            logprob_t = torch.stack(step_logprobs)
            advantages = returns - values_t.detach()

            policy_loss = -(logprob_t * advantages).mean()
            value_loss = torch.mean((values_t - returns) ** 2)
            loss = policy_loss + 0.5 * value_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_policy_loss += float(policy_loss.item())
            epoch_value_loss += float(value_loss.item())
            n_batches += 1

        if n_batches == 0:
            print(f"[train-mappo] epoch={epoch + 1}/{args.train_epochs} no_batches")
        else:
            print(
                f"[train-mappo] epoch={epoch + 1}/{args.train_epochs} "
                f"policy_loss={epoch_policy_loss / n_batches:.4f} value_loss={epoch_value_loss / n_batches:.4f}"
            )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": 36,
            "action_dim": 5,
            "kind": "mappo",
        },
        checkpoint_path,
    )
    print(f"[train-mappo] checkpoint={checkpoint_path}")
    return checkpoint_path
