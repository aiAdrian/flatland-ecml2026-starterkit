from __future__ import annotations

from pathlib import Path

import torch
from torch.distributions import Categorical
from flatland.envs.step_utils.states import TrainState

from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env, build_env_from_pkl, list_pkl_dataset
from utils.model_utils import ActorCriticNet, infer_obs_dim, split_obs_and_mask
from utils.progress import RollingDoneRatio, format_console_row, format_fixed_metrics, make_progress_bar


def _discounted_returns(rewards, gamma: float):
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return out


def train_mappo(args, checkpoint_path: Path, tb_logger=None) -> Path:
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
    pkl_files = list_pkl_dataset(args.pkl_dir) if args.env_source == "pkl" else []
    if args.env_source == "pkl" and not pkl_files:
        raise ValueError(f"No PKL environments found in {args.pkl_dir}. Run with --prepare-pkls first.")
    if args.env_source == "generated":
        env = build_env(cfg=cfg, obs_builder=obs_builder)
    else:
        env = build_env_from_pkl(pkl_files[0], obs_builder=obs_builder)

    probe_obs, _probe_info = env.reset(random_seed=args.seed)
    del _probe_info
    if isinstance(probe_obs, dict):
        first_obs = probe_obs[0] if 0 in probe_obs else next(iter(probe_obs.values()))
    else:
        first_obs = probe_obs[0]
    obs_dim = infer_obs_dim(first_obs, default=36)

    model = ActorCriticNet(obs_dim=obs_dim, action_dim=5)
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        if "model_state" in payload:
            model.load_state_dict(payload["model_state"], strict=False)
            print(f"[train-mappo] warmstart={args.init_checkpoint}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_feature_sum = 0.0
    total_feature_sq_sum = 0.0
    total_feature_count = 0
    total_mask_active = 0.0
    total_mask_count = 0.0
    total_action_hist = torch.zeros(5, dtype=torch.long)

    for epoch in range(args.train_epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_approx_kl = 0.0
        n_batches = 0
        rolling_done = RollingDoneRatio(window_size=20)
        bar = make_progress_bar(total=args.episodes, desc=f"MAPPO[{epoch + 1}/{args.train_epochs}]")

        for ep in range(args.episodes):
            if args.env_source == "pkl":
                pkl_path = pkl_files[(epoch * max(1, args.episodes) + ep) % len(pkl_files)]
                env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)

            observations, info = env.reset(random_seed=args.seed + epoch * 1000 + ep)
            del info
            done = {"__all__": False}
            steps = 0

            step_logprobs = []
            step_values = []
            step_rewards = []
            transition_feats = []
            transition_masks = []
            transition_actions = []
            transition_old_logprobs = []

            while not done.get("__all__", False) and steps < args.max_episode_steps:
                handles = list(range(env.get_num_agents()))
                obs_batch = [observations[h] for h in handles]

                actions = {}
                logprobs = []
                values = []

                for h, obs in zip(handles, obs_batch):
                    feat, mask = split_obs_and_mask(obs, obs_dim=obs_dim)
                    logits, value = model(feat)
                    logits = logits.masked_fill(mask < 0.5, float("-inf"))
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    actions[h] = int(action.item())
                    logprobs.append(dist.log_prob(action))
                    values.append(value)
                    transition_feats.append(feat.detach())
                    transition_masks.append(mask.detach())
                    transition_actions.append(action.detach())
                    transition_old_logprobs.append(dist.log_prob(action).detach())
                    total_feature_sum += float(feat.sum().item())
                    total_feature_sq_sum += float((feat * feat).sum().item())
                    total_feature_count += int(feat.numel())
                    total_mask_active += float((mask > 0.5).sum().item())
                    total_mask_count += float(mask.numel())
                    total_action_hist[int(action.item())] += 1

                observations, rewards, done, info = env.step(normalize_actions(actions))
                del info

                # Team reward signal (mean over agents)
                team_reward = sum(float(rewards[h]) for h in handles) / max(1, len(handles))
                step_rewards.append(team_reward)
                step_logprobs.append(torch.stack(logprobs).mean())
                step_values.append(torch.stack(values).mean())
                steps += 1

            if not step_rewards:
                bar.set_postfix_str(f"steps={steps} no_steps")
                bar.update(1)
                print(
                    f"[train-mappo][epoch={epoch + 1}/{args.train_epochs}]"
                    f"[episode={ep + 1}/{args.episodes}] no_steps"
                )
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

            with torch.no_grad():
                feat_t = torch.stack(transition_feats, dim=0)
                mask_t = torch.stack(transition_masks, dim=0)
                action_t = torch.stack(transition_actions, dim=0)
                old_logp_t = torch.stack(transition_old_logprobs, dim=0)

                new_logits, _ = model(feat_t)
                new_logits = new_logits.masked_fill(mask_t < 0.5, float("-inf"))
                new_dist = Categorical(logits=new_logits)
                new_logp_t = new_dist.log_prob(action_t)
                approx_kl = torch.mean(old_logp_t - new_logp_t)
                entropy = torch.mean(new_dist.entropy())

            epoch_policy_loss += float(policy_loss.item())
            epoch_value_loss += float(value_loss.item())
            epoch_entropy += float(entropy.item())
            epoch_approx_kl += float(approx_kl.item())
            n_batches += 1

            ep_done = sum(a.state == TrainState.DONE for a in env.agents)
            rolling_done.update(ep_done, env.get_num_agents())
            bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())
            ep_reward = float(sum(step_rewards))
            bar.set_postfix_str(f"s={steps} rew={ep_reward:+.2f}")
            bar.update(1)
            print(format_console_row("train", "mappo", epoch=f"{epoch + 1}/{args.train_epochs}", ep=f"{ep + 1}/{args.episodes}", steps=steps, done=f"{ep_done}/{env.get_num_agents()}", rew=ep_reward, p_loss=float(policy_loss.item()), v_loss=float(value_loss.item())))

        if n_batches == 0:
            print(format_console_row("epoch", "mappo", epoch=f"{epoch + 1}/{args.train_epochs}", status="no_batches"))
        else:
            avg_p = epoch_policy_loss / n_batches
            avg_v = epoch_value_loss / n_batches
            avg_entropy = epoch_entropy / n_batches
            avg_kl = epoch_approx_kl / n_batches
            dbg = ""
            if getattr(args, "debug_checks", False):
                dbg = f" entropy={avg_entropy:.4f} approx_kl={avg_kl:.5f}"
            print(format_console_row("epoch", "mappo", epoch=f"{epoch + 1}/{args.train_epochs}", policy_loss=avg_p, value_loss=avg_v, debug=dbg.strip() if dbg else "-"))
            if tb_logger is not None:
                tb_logger.log_mappo_epoch(
                    epoch + 1,
                    policy_loss=avg_p,
                    value_loss=avg_v,
                    entropy=avg_entropy,
                    approx_kl=avg_kl,
                )

        bar.close()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": 5,
            "kind": "mappo",
        },
        checkpoint_path,
    )
    feature_mean = total_feature_sum / max(1, total_feature_count)
    feature_var = max(0.0, total_feature_sq_sum / max(1, total_feature_count) - feature_mean * feature_mean)
    feature_std = feature_var ** 0.5
    mask_active_ratio = total_mask_active / max(1.0, total_mask_count)
    action_total = int(total_action_hist.sum().item())
    action_hist_text = ", ".join(
        f"a{idx}={int(count)}" for idx, count in enumerate(total_action_hist.tolist())
    )
    print(format_console_row("summary", "mappo", obs_dim=obs_dim, feature_mean=feature_mean, feature_std=feature_std, mask_active_ratio=mask_active_ratio, actions_total=action_total, hist=action_hist_text))
    print(format_console_row("checkpoint", "mappo", path=str(checkpoint_path)))
    return checkpoint_path
