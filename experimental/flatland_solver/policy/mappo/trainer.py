from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.distributions import Categorical
from flatland.envs.step_utils.states import TrainState

from utils.action_utils import normalize_actions
from utils.env_factory import (
    PklEnvMeta,
    SolverConfig,
    build_env,
    build_env_from_pkl,
    list_pkl_dataset,
    list_pkl_dataset_meta,
)
from utils.model_utils import ActorCriticNet, infer_obs_dim, split_obs_and_mask
from utils.progress import RollingDoneRatio, format_console_row, make_progress_bar


def _discounted_returns(rewards, gamma: float):
    out = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    out.reverse()
    return out


def _collect_episode(env, model, obs_dim: int, max_steps: int, seed: int, gamma: float):
    observations, info = env.reset(random_seed=seed)
    del info
    done = {"__all__": False}
    steps = 0
    step_rewards: list[float] = []

    feats: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_logprobs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    transition_step_idx: list[int] = []

    while not done.get("__all__", False) and steps < max_steps:
        handles = list(range(env.get_num_agents()))
        obs_batch = [observations[h] for h in handles]
        step_actions = {}

        for h, obs in zip(handles, obs_batch):
            feat, mask = split_obs_and_mask(obs, obs_dim=obs_dim)
            logits, value = model(feat)
            logits = logits.masked_fill(mask < 0.5, float("-inf"))
            dist = Categorical(logits=logits)
            action = dist.sample()

            step_actions[h] = int(action.item())
            feats.append(feat.detach())
            masks.append(mask.detach())
            actions.append(action.detach())
            old_logprobs.append(dist.log_prob(action).detach())
            values.append(value.detach().reshape(()))
            transition_step_idx.append(steps)

        observations, rewards, done, info = env.step(normalize_actions(step_actions))
        del info
        team_reward = sum(float(rewards[h]) for h in handles) / max(1, len(handles))
        step_rewards.append(team_reward)
        steps += 1

    if not step_rewards:
        return None

    returns_step = torch.as_tensor(_discounted_returns(step_rewards, gamma), dtype=torch.float32)
    values_t = torch.stack(values).to(torch.float32)
    returns_t = torch.as_tensor([float(returns_step[i]) for i in transition_step_idx], dtype=torch.float32)
    advantages_t = returns_t - values_t

    ep_done = sum(a.state == TrainState.DONE for a in env.agents)
    active_or_blocked = sum(
        a.state in (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION)
        for a in env.agents
    )
    deadlock_rate = active_or_blocked / max(1, env.get_num_agents())

    return {
        "feats": torch.stack(feats, dim=0),
        "masks": torch.stack(masks, dim=0),
        "actions": torch.stack(actions, dim=0),
        "old_logprobs": torch.stack(old_logprobs, dim=0),
        "returns": returns_t,
        "advantages": advantages_t,
        "steps": steps,
        "ep_done": ep_done,
        "n_agents": env.get_num_agents(),
        "ep_reward": float(sum(step_rewards)),
        "deadlock_rate": deadlock_rate,
    }


def _ppo_update(
    model,
    opt,
    buffer,
    ppo_epochs: int,
    batch_size: int,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    clip_eps: float = 0.2,
    target_kl: float = 0.05,
    kl_stop_factor: float = 1.5,
):
    feats = torch.cat(buffer["feats"], dim=0)
    masks = torch.cat(buffer["masks"], dim=0)
    actions = torch.cat(buffer["actions"], dim=0)
    old_logprobs = torch.cat(buffer["old_logprobs"], dim=0)
    returns = torch.cat(buffer["returns"], dim=0)
    advantages = torch.cat(buffer["advantages"], dim=0)

    if feats.shape[0] == 0:
        return None

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    n = feats.shape[0]

    stats = {
        "p_loss": [],
        "v_loss": [],
        "entropy": [],
        "approx_kl": [],
        "ratio": [],
        "clip_frac": [],
        "early_stop_skips": [],
    }
    n_minibatches = 0
    kl_skip_threshold = float(target_kl) * float(kl_stop_factor)

    for _ in range(max(1, int(ppo_epochs))):
        perm = torch.randperm(n)
        for start in range(0, n, max(1, int(batch_size))):
            mb = perm[start:start + max(1, int(batch_size))]
            n_minibatches += 1

            b_feat = feats[mb]
            b_mask = masks[mb]
            b_act = actions[mb]
            b_old_lp = old_logprobs[mb]
            b_ret = returns[mb]
            b_adv = advantages[mb]

            logits, values = model(b_feat)
            logits = logits.masked_fill(b_mask < 0.5, float("-inf"))
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(b_act)
            ent = dist.entropy().mean()

            ratio = torch.exp(new_logp - b_old_lp)
            log_ratio = new_logp - b_old_lp
            with torch.no_grad():
                # Legacy-like KL guard BEFORE optimizer step.
                approx_kl_guard = torch.mean((ratio - 1.0) - log_ratio)
            if float(approx_kl_guard.item()) > kl_skip_threshold:
                if n_minibatches <= 2 or n_minibatches % 8 == 0:
                    print(
                        f"[PPO] KL early-stop at mb#{n_minibatches}: "
                        f"approx_kl={float(approx_kl_guard.item()):.4f} > {kl_skip_threshold:.4f} (skip update)"
                    )
                stats["approx_kl"].append(float(approx_kl_guard.item()))
                stats["ratio"].append(float(ratio.mean().item()))
                stats["early_stop_skips"].append(1.0)
                continue

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            p_loss = -torch.min(surr1, surr2).mean()

            values = values.view(-1)
            v_loss = torch.mean((values - b_ret) ** 2)
            loss = p_loss + float(value_coef) * v_loss - float(entropy_coef) * ent

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                approx_kl = torch.mean(b_old_lp - new_logp)
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).to(torch.float32))

            stats["p_loss"].append(float(p_loss.item()))
            stats["v_loss"].append(float(v_loss.item()))
            stats["entropy"].append(float(ent.item()))
            stats["approx_kl"].append(float(approx_kl.item()))
            stats["ratio"].append(float(ratio.mean().item()))
            stats["clip_frac"].append(float(clip_frac.item()))
            stats["early_stop_skips"].append(0.0)

    out = {k: (sum(v) / max(1, len(v))) for k, v in stats.items()}
    out["n_minibatches"] = n_minibatches
    out["n_samples"] = int(n)
    print(
        f"[PPO-DEBUG] DONE n_mb={n_minibatches} "
        f"KL={out.get('approx_kl', 0.0):+.6f} ratio={out.get('ratio', 0.0):.5f} "
        f"clip_frac={out.get('clip_frac', 0.0):.3f} skip={out.get('early_stop_skips', 0.0):.3f}"
    )
    return out


def _group_pkls_by_n_agents(metas: list[PklEnvMeta], curriculum: list[int]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    allowed = set(curriculum)
    for m in metas:
        if m.n_agents in allowed:
            grouped[m.n_agents].append(m.path)
    return grouped


def _eval_model(
    model,
    obs_builder,
    cfg: SolverConfig,
    env_source: str,
    pkl_files: list[Path],
    obs_dim: int,
    n_episodes: int,
    max_steps: int,
    seed_base: int,
    greedy: bool,
):
    if n_episodes <= 0:
        return {"done_rate": 0.0, "deadlock_rate": 0.0, "episode_len": 0.0, "total_reward": 0.0}

    if env_source == "generated":
        env = build_env(cfg=cfg, obs_builder=obs_builder)
    else:
        env = build_env_from_pkl(pkl_files[0], obs_builder=obs_builder)

    done_rates = []
    deadlock_rates = []
    lengths = []
    rewards_sum = []

    for ep in range(n_episodes):
        if env_source == "pkl":
            env = build_env_from_pkl(pkl_files[ep % len(pkl_files)], obs_builder=obs_builder)

        observations, info = env.reset(random_seed=seed_base + ep)
        del info
        done = {"__all__": False}
        steps = 0
        ep_reward = 0.0

        while not done.get("__all__", False) and steps < max_steps:
            handles = list(range(env.get_num_agents()))
            obs_batch = [observations[h] for h in handles]
            actions = {}
            for h, obs in zip(handles, obs_batch):
                feat, mask = split_obs_and_mask(obs, obs_dim=obs_dim)
                logits, _ = model(feat)
                logits = logits.masked_fill(mask < 0.5, float("-inf"))
                if greedy:
                    action = int(torch.argmax(logits).item())
                else:
                    action = int(Categorical(logits=logits).sample().item())
                actions[h] = action

            observations, rewards, done, info = env.step(normalize_actions(actions))
            del info
            ep_reward += float(sum(rewards.values())) if rewards else 0.0
            steps += 1

        ep_done = sum(a.state == TrainState.DONE for a in env.agents)
        active_or_blocked = sum(
            a.state in (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION)
            for a in env.agents
        )
        done_rates.append(ep_done / max(1, env.get_num_agents()))
        deadlock_rates.append(active_or_blocked / max(1, env.get_num_agents()))
        lengths.append(float(steps))
        rewards_sum.append(float(ep_reward))

    return {
        "done_rate": sum(done_rates) / max(1, len(done_rates)),
        "deadlock_rate": sum(deadlock_rates) / max(1, len(deadlock_rates)),
        "episode_len": sum(lengths) / max(1, len(lengths)),
        "total_reward": sum(rewards_sum) / max(1, len(rewards_sum)),
    }


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
    pkl_metas = list_pkl_dataset_meta(args.pkl_dir) if args.env_source == "pkl" else []
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
    done_history: list[float] = []

    curriculum_mode = bool(args.agent_curriculum)
    curriculum = [int(x) for x in (args.agent_curriculum or [])]
    grouped_pkls = _group_pkls_by_n_agents(pkl_metas, curriculum) if curriculum_mode else {}
    grouped_cursor = {k: 0 for k in grouped_pkls.keys()}
    global_pkl_cursor = 0
    global_update_idx = 0
    eval_log: list[dict] = []

    for epoch in range(args.train_epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_approx_kl = 0.0
        epoch_ratio = 0.0
        epoch_clip_frac = 0.0
        n_updates = 0

        rolling_done = RollingDoneRatio(window_size=20)
        bar = make_progress_bar(total=args.episodes, desc=f"MAPPO[{epoch + 1}/{args.train_epochs}]")

        ep = 0
        last_update_stats = {
            "p_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "ratio": 0.0,
            "clip_frac": 0.0,
        }

        while ep < args.episodes:
            if curriculum_mode:
                target_rollout_eps = min(args.episodes - ep, max(1, len(curriculum)))
            else:
                target_rollout_eps = min(args.episodes - ep, max(1, int(args.mappo_rollout_episodes)))

            buffer = {
                "feats": [],
                "masks": [],
                "actions": [],
                "old_logprobs": [],
                "returns": [],
                "advantages": [],
            }
            block_collected = 0

            for block_i in range(target_rollout_eps):
                if args.env_source == "pkl":
                    if curriculum_mode and grouped_pkls:
                        n_agents_target = curriculum[block_i % len(curriculum)]
                        bucket = grouped_pkls.get(n_agents_target, [])
                        if bucket:
                            idx = grouped_cursor.get(n_agents_target, 0) % len(bucket)
                            pkl_path = bucket[idx]
                            grouped_cursor[n_agents_target] = grouped_cursor.get(n_agents_target, 0) + 1
                        else:
                            pkl_path = pkl_files[global_pkl_cursor % len(pkl_files)]
                            global_pkl_cursor += 1
                    else:
                        pkl_path = pkl_files[global_pkl_cursor % len(pkl_files)]
                        global_pkl_cursor += 1
                    env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)
                else:
                    if curriculum_mode:
                        n_agents_target = curriculum[block_i % len(curriculum)]
                        sub_cfg = SolverConfig(
                            width=args.width,
                            height=args.height,
                            n_agents=n_agents_target,
                            n_cities=args.n_cities,
                            max_rails_between_cities=args.max_rails_between_cities,
                            max_rail_pairs_in_city=args.max_rail_pairs_in_city,
                            seed=args.seed,
                        )
                        env = build_env(cfg=sub_cfg, obs_builder=obs_builder)
                    else:
                        env = build_env(cfg=cfg, obs_builder=obs_builder)

                episode = _collect_episode(
                    env=env,
                    model=model,
                    obs_dim=obs_dim,
                    max_steps=args.max_episode_steps,
                    seed=args.seed + epoch * 1000 + ep,
                    gamma=args.gamma,
                )
                if episode is None:
                    continue

                buffer["feats"].append(episode["feats"])
                buffer["masks"].append(episode["masks"])
                buffer["actions"].append(episode["actions"])
                buffer["old_logprobs"].append(episode["old_logprobs"])
                buffer["returns"].append(episode["returns"])
                buffer["advantages"].append(episode["advantages"])
                block_collected += 1

                total_feature_sum += float(episode["feats"].sum().item())
                total_feature_sq_sum += float((episode["feats"] * episode["feats"]).sum().item())
                total_feature_count += int(episode["feats"].numel())
                total_mask_active += float((episode["masks"] > 0.5).sum().item())
                total_mask_count += float(episode["masks"].numel())
                total_action_hist += torch.bincount(episode["actions"].to(torch.long), minlength=5)

                ep_done = int(episode["ep_done"])
                n_agents_ep = int(episode["n_agents"])
                ep_reward = float(episode["ep_reward"])
                done_rate = ep_done / max(1, n_agents_ep)
                done_history.append(done_rate)
                done_window = max(1, int(getattr(args, "mappo_done_window", 50)))
                recent_done_50 = sum(done_history[-done_window:]) / max(1, len(done_history[-done_window:]))

                rolling_done.update(ep_done, n_agents_ep)
                bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())
                bar.set_postfix_str(
                    f"s={int(episode['steps'])} rew={ep_reward:+.2f} KL={last_update_stats['approx_kl']:+.4f} ent={last_update_stats['entropy']:.4f}"
                )
                bar.update(1)

                ep += 1
                print(
                    format_console_row(
                        "train",
                        "mappo",
                        epoch=f"{epoch + 1}/{args.train_epochs}",
                        ep=f"{ep}/{args.episodes}",
                        steps=int(episode["steps"]),
                        done=f"{ep_done}/{n_agents_ep}",
                        done50=recent_done_50,
                        rew=ep_reward,
                        p_loss=last_update_stats["p_loss"],
                        v_loss=last_update_stats["v_loss"],
                        entropy=last_update_stats["entropy"],
                        approx_kl=last_update_stats["approx_kl"],
                    )
                )

                if tb_logger is not None:
                    ep_idx = epoch * max(1, args.episodes) + ep
                    tb_logger.log_mappo_episode(
                        episode_idx=ep_idx,
                        done_rate=done_rate,
                        episode_len=float(episode["steps"]),
                        total_reward=ep_reward,
                        n_agents=n_agents_ep,
                        deadlock_rate=float(episode["deadlock_rate"]),
                        done_rolling=recent_done_50,
                        policy_loss=last_update_stats["p_loss"],
                        value_loss=last_update_stats["v_loss"],
                    )
                    tb_logger.log_scalar("env/n_agents", n_agents_ep, ep_idx)
                    tb_logger.log_scalar("env/done_count", ep_done, ep_idx)
                    ad = torch.bincount(episode["actions"].to(torch.long), minlength=5).to(torch.float32)
                    denom = max(1.0, float(ad.sum().item()))
                    tb_logger.log_scalar("actions/do_nothing", float(ad[0].item() / denom), ep_idx)
                    tb_logger.log_scalar("actions/move_left", float(ad[1].item() / denom), ep_idx)
                    tb_logger.log_scalar("actions/move_forward", float(ad[2].item() / denom), ep_idx)
                    tb_logger.log_scalar("actions/move_right", float(ad[3].item() / denom), ep_idx)
                    tb_logger.log_scalar("actions/stop", float(ad[4].item() / denom), ep_idx)

                mid_eval_every = int(getattr(args, "mappo_mid_eval_every", 0))
                if mid_eval_every > 0 and ep % mid_eval_every == 0:
                    print(f"\n[Train] Mid-training eval at episode {ep} ...")
                    mid_metrics = _eval_model(
                        model=model,
                        obs_builder=obs_builder,
                        cfg=cfg,
                        env_source=args.env_source,
                        pkl_files=pkl_files,
                        obs_dim=obs_dim,
                        n_episodes=int(getattr(args, "mappo_mid_eval_episodes", 10)),
                        max_steps=args.max_episode_steps,
                        seed_base=args.seed + epoch * 100000 + ep,
                        greedy=bool(getattr(args, "mappo_eval_greedy", False)),
                    )
                    eval_entry = {"kind": "mid", "episode": ep, **mid_metrics}
                    eval_log.append(eval_entry)
                    if tb_logger is not None:
                        tb_logger.log_scalar("train_eval/done_rate", float(mid_metrics["done_rate"]), ep)
                        tb_logger.log_scalar("train_eval/deadlock_rate", float(mid_metrics["deadlock_rate"]), ep)
                        tb_logger.log_scalar("train_eval/episode_len", float(mid_metrics["episode_len"]), ep)
                        tb_logger.log_scalar("train_eval/total_reward", float(mid_metrics["total_reward"]), ep)
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

            update_stats = _ppo_update(
                model=model,
                opt=opt,
                buffer=buffer,
                ppo_epochs=args.mappo_ppo_epochs,
                batch_size=args.mappo_batch_size,
                entropy_coef=args.mappo_entropy_coef,
                value_coef=args.mappo_value_coef,
                clip_eps=args.mappo_clip_eps,
                target_kl=args.mappo_target_kl,
                kl_stop_factor=args.mappo_kl_stop_factor,
            )
            if update_stats is None:
                continue

            global_update_idx += 1
            n_updates += 1
            epoch_policy_loss += float(update_stats["p_loss"])
            epoch_value_loss += float(update_stats["v_loss"])
            epoch_entropy += float(update_stats["entropy"])
            epoch_approx_kl += float(update_stats["approx_kl"])
            epoch_ratio += float(update_stats["ratio"])
            epoch_clip_frac += float(update_stats["clip_frac"])
            last_update_stats = {
                "p_loss": float(update_stats["p_loss"]),
                "v_loss": float(update_stats["v_loss"]),
                "entropy": float(update_stats["entropy"]),
                "approx_kl": float(update_stats["approx_kl"]),
                "ratio": float(update_stats["ratio"]),
                "clip_frac": float(update_stats["clip_frac"]),
            }

            print(
                format_console_row(
                    "update",
                    "mappo",
                    epoch=f"{epoch + 1}/{args.train_epochs}",
                    upd=global_update_idx,
                    n_rollout=block_collected,
                    n_samples=int(update_stats["n_samples"]),
                    k_epoch=args.mappo_ppo_epochs,
                    mb=int(update_stats["n_minibatches"]),
                    p_loss=float(update_stats["p_loss"]),
                    v_loss=float(update_stats["v_loss"]),
                    entropy=float(update_stats["entropy"]),
                    approx_kl=float(update_stats["approx_kl"]),
                    ratio=float(update_stats["ratio"]),
                    clip_frac=float(update_stats["clip_frac"]),
                    skip=float(update_stats.get("early_stop_skips", 0.0)),
                )
            )

            if tb_logger is not None:
                tb_logger.log_scalar("ppo_update/p_loss", float(update_stats["p_loss"]), global_update_idx)
                tb_logger.log_scalar("ppo_update/v_loss", float(update_stats["v_loss"]), global_update_idx)
                tb_logger.log_scalar("ppo_update/entropy", float(update_stats["entropy"]), global_update_idx)
                tb_logger.log_scalar("ppo_update/approx_kl", float(update_stats["approx_kl"]), global_update_idx)
                tb_logger.log_scalar("ppo_update/ratio", float(update_stats["ratio"]), global_update_idx)
                tb_logger.log_scalar("ppo_update/clip_frac", float(update_stats["clip_frac"]), global_update_idx)
                tb_logger.log_scalar("ppo_update/early_stop_skips", float(update_stats.get("early_stop_skips", 0.0)), global_update_idx)

        if n_updates == 0:
            print(format_console_row("epoch", "mappo", epoch=f"{epoch + 1}/{args.train_epochs}", status="no_batches"))
        else:
            avg_p = epoch_policy_loss / n_updates
            avg_v = epoch_value_loss / n_updates
            avg_entropy = epoch_entropy / n_updates
            avg_kl = epoch_approx_kl / n_updates
            avg_ratio = epoch_ratio / n_updates
            avg_clip_frac = epoch_clip_frac / n_updates
            print(
                format_console_row(
                    "epoch",
                    "mappo",
                    epoch=f"{epoch + 1}/{args.train_epochs}",
                    policy_loss=avg_p,
                    value_loss=avg_v,
                    entropy=avg_entropy,
                    approx_kl=avg_kl,
                    ratio=avg_ratio,
                    clip_frac=avg_clip_frac,
                )
            )
            if tb_logger is not None:
                tb_logger.log_mappo_epoch(
                    epoch + 1,
                    policy_loss=avg_p,
                    value_loss=avg_v,
                    entropy=avg_entropy,
                    approx_kl=avg_kl,
                    ratio=avg_ratio,
                    clip_frac=avg_clip_frac,
                )

        bar.close()

    print("\n[Train] Final eval ...")
    final_metrics = _eval_model(
        model=model,
        obs_builder=obs_builder,
        cfg=cfg,
        env_source=args.env_source,
        pkl_files=pkl_files,
        obs_dim=obs_dim,
        n_episodes=int(getattr(args, "mappo_mid_eval_episodes", 10)),
        max_steps=args.max_episode_steps,
        seed_base=args.seed + 999999,
        greedy=bool(getattr(args, "mappo_eval_greedy", False)),
    )
    eval_log.append({"kind": "final", "episode": args.episodes, **final_metrics})
    if tb_logger is not None:
        tb_logger.log_scalar("train_eval/done_rate", float(final_metrics["done_rate"]), args.episodes)
        tb_logger.log_scalar("train_eval/deadlock_rate", float(final_metrics["deadlock_rate"]), args.episodes)
        tb_logger.log_scalar("train_eval/episode_len", float(final_metrics["episode_len"]), args.episodes)
        tb_logger.log_scalar("train_eval/total_reward", float(final_metrics["total_reward"]), args.episodes)

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
    if tb_logger is not None:
        tb_logger.log_mappo_summary(
            obs_dim=obs_dim,
            feature_mean=feature_mean,
            feature_std=feature_std,
            mask_active_ratio=mask_active_ratio,
            actions_total=action_total,
            action_hist=[int(x) for x in total_action_hist.tolist()],
        )
    print(
        format_console_row(
            "summary",
            "mappo",
            obs_dim=obs_dim,
            feature_mean=feature_mean,
            feature_std=feature_std,
            mask_active_ratio=mask_active_ratio,
            actions_total=action_total,
            hist=action_hist_text,
        )
    )
    if eval_log:
        print("=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        for entry in eval_log:
            label = str(entry.get("kind", "mid")).upper()
            print(
                f"  {label:<5} ep {int(entry['episode']):5d}: "
                f"done={float(entry['done_rate']):.3f}  "
                f"deadlock={float(entry['deadlock_rate']):.3f}  "
                f"rew={float(entry['total_reward']):+.1f}"
            )
    print(format_console_row("checkpoint", "mappo", path=str(checkpoint_path)))
    return checkpoint_path
