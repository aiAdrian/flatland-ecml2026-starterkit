from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from policy.mappo.actor_critic_head import ActorCriticHead
from policy.mappo.base_feature_encoder import BaseFeatureEncoder
from policy.mappo.tree_payload_encoder import TreePayloadEncoder
from utils.action_utils import normalize_actions
from utils.env_factory import (
    PklEnvMeta,
    SolverConfig,
    build_env,
    build_env_from_pkl,
    list_pkl_dataset,
    list_pkl_dataset_meta,
)
from utils.model_utils import infer_obs_dim
from rewards.outcome_reward import build_outcome_reward
from utils.progress import RollingDoneRatio, format_console_row, format_episode_compact, format_ppo_update, make_progress_bar


def _group_pkls_by_n_agents(metas: list[PklEnvMeta], curriculum: list[int]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    allowed = set(curriculum)
    for m in metas:
        if m.n_agents in allowed:
            grouped[m.n_agents].append(m.path)
    return grouped


def _unwrap_state(state: Any, base_dim: int) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    if isinstance(state, list) and len(state) > 0 and isinstance(state[0], (tuple, list)):
        state = state[-1]

    if isinstance(state, (tuple, list)):
        if len(state) >= 3:
            obs, opps, payload = state[0], state[1], state[2]
            base = np.asarray(obs, dtype=np.float32).flatten()
            if not isinstance(opps, list):
                opps = []
            if not isinstance(payload, dict):
                payload = {}
        elif len(state) >= 1:
            base = np.asarray(state[0], dtype=np.float32).flatten()
            opps, payload = [], {}
        else:
            base = np.zeros(base_dim, dtype=np.float32)
            opps, payload = [], {}
    else:
        base = np.asarray(state, dtype=np.float32).flatten()
        opps, payload = [], {}

    if base.shape[0] < base_dim:
        base = np.concatenate([base, np.zeros(base_dim - base.shape[0], dtype=np.float32)], axis=0)
    return base[:base_dim], opps, payload


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


def _legal_action_mask(base_obs: np.ndarray, agent) -> np.ndarray:
    mask = np.zeros(5, dtype=np.float32)

    if agent.state == TrainState.DONE:
        mask[int(RailEnvActions.DO_NOTHING.value)] = 1.0
        return mask

    waiting_state = getattr(TrainState, "WAITING", None)
    if waiting_state is not None and agent.state == waiting_state:
        mask[int(RailEnvActions.DO_NOTHING.value)] = 1.0
        return mask

    if hasattr(agent.state, "is_off_map_state") and agent.state.is_off_map_state():
        mask[int(RailEnvActions.DO_NOTHING.value)] = 1.0
        mask[int(RailEnvActions.STOP_MOVING.value)] = 1.0
        mask[int(RailEnvActions.MOVE_FORWARD.value)] = 1.0
        return mask

    mask[int(RailEnvActions.STOP_MOVING.value)] = 1.0

    left_ok = float(base_obs[0]) > 0.5 if base_obs.shape[0] >= 1 else False
    fwd_ok = float(base_obs[1]) > 0.5 if base_obs.shape[0] >= 2 else False
    right_ok = float(base_obs[2]) > 0.5 if base_obs.shape[0] >= 3 else False
    n_trans = int(left_ok) + int(fwd_ok) + int(right_ok)

    if n_trans == 1:
        mask[int(RailEnvActions.MOVE_FORWARD.value)] = 1.0
    elif n_trans > 1:
        mask[int(RailEnvActions.MOVE_LEFT.value)] = 1.0 if left_ok else 0.0
        mask[int(RailEnvActions.MOVE_FORWARD.value)] = 1.0 if fwd_ok else 0.0
        mask[int(RailEnvActions.MOVE_RIGHT.value)] = 1.0 if right_ok else 0.0

    if np.sum(mask > 0.5) <= 0:
        mask[int(RailEnvActions.DO_NOTHING.value)] = 1.0

    return mask


def _forward_structured(
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: torch.nn.Module,
    head: ActorCriticHead,
    base_t: torch.Tensor,
    payloads: list[dict[str, Any]],
    pool_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    emb_base = base_encoder(base_t)
    emb_tree = tree_encoder.forward_batch(payloads)
    emb = fuse(torch.cat([emb_base, emb_tree], dim=-1))
    logits = head.actor_logits(emb)
    values = head.value(emb, pool_t)
    return logits, values


def _collect_episode(
    env,
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: torch.nn.Module,
    head: ActorCriticHead,
    base_dim: int,
    seed: int,
    max_steps: int,
    reward_shaper=None,
) -> dict[str, Any] | None:
    observations, info = env.reset(random_seed=seed)
    del info
    done = {"__all__": False}
    steps = 0

    traj: dict[int, list[tuple]] = defaultdict(list)
    action_hist = torch.zeros(5, dtype=torch.long)
    feat_sum = 0.0
    feat_sq_sum = 0.0
    feat_count = 0
    mask_active = 0.0
    mask_count = 0.0
    ep_reward = 0.0

    device = next(base_encoder.parameters()).device

    while not done.get("__all__", False) and steps < max_steps:
        handles = list(range(env.get_num_agents()))
        step_actions: dict[int, int] = {}
        pending: list[tuple] = []

        for h in handles:
            base_obs, opps, payload = _unwrap_state(observations[h], base_dim)
            n_pool = _neighbor_pool(opps, base_dim)
            mask_np = _legal_action_mask(base_obs, env.agents[h])

            base_t = torch.from_numpy(base_obs).float().unsqueeze(0).to(device)
            pool_t = torch.from_numpy(n_pool).float().unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value = _forward_structured(base_encoder, tree_encoder, fuse, head, base_t, [payload], pool_t)
                logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
                dist = Categorical(logits=logits)
                action_t = dist.sample()
                action = int(action_t.item())
                old_lp = float(dist.log_prob(action_t).item())
                val = float(value.item())

            step_actions[h] = action
            pending.append((h, base_obs, n_pool, payload, action, old_lp, val, mask_np))

            action_hist[action] += 1
            feat_sum += float(base_obs.sum())
            feat_sq_sum += float(np.square(base_obs).sum())
            feat_count += int(base_obs.size)
            mask_active += float(np.sum(mask_np > 0.5))
            mask_count += float(mask_np.size)

        observations, rewards, done, info = env.step(normalize_actions(step_actions))
        if reward_shaper is not None:
            rewards = reward_shaper(rewards, done, info, env, step_actions)
        del info
        ep_reward += float(sum(rewards.values())) if rewards else 0.0

        for h, base_obs, n_pool, payload, action, old_lp, val, mask_np in pending:
            reward_h = float(rewards.get(h, 0.0)) if isinstance(rewards, dict) else 0.0
            done_h = False
            if isinstance(done, dict):
                try:
                    done_h = bool(done[h])  # type: ignore[index]
                except Exception:
                    done_h = False
            finished_h = bool(env.agents[h].state == TrainState.DONE)
            traj[h].append((base_obs, n_pool, payload, action, reward_h, done_h, finished_h, old_lp, val, mask_np))

        steps += 1

    if len(traj) == 0:
        return None

    ep_done = sum(a.state == TrainState.DONE for a in env.agents)
    active_or_blocked = sum(
        a.state in (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION)
        for a in env.agents
    )

    return {
        "traj": traj,
        "steps": steps,
        "ep_done": ep_done,
        "n_agents": env.get_num_agents(),
        "ep_reward": ep_reward,
        "deadlock_rate": active_or_blocked / max(1, env.get_num_agents()),
        "action_hist": action_hist,
        "feat_sum": feat_sum,
        "feat_sq_sum": feat_sq_sum,
        "feat_count": feat_count,
        "mask_active": mask_active,
        "mask_count": mask_count,
    }


def _build_ppo_batch(
    episodes: list[dict[str, Any]],
    gamma: float,
    gae_lambda: float,
    base_dim: int,
    device: torch.device,
) -> dict[str, Any] | None:
    all_base, all_pool, all_payload = [], [], []
    all_act, all_old_lp = [], []
    all_adv, all_ret, all_mask = [], [], []

    for ep in episodes:
        for transitions in ep["traj"].values():
            if not transitions:
                continue

            t_len = len(transitions)
            base_arr = np.stack([t[0] for t in transitions], axis=0).astype(np.float32)
            pool_arr = np.stack([t[1] for t in transitions], axis=0).astype(np.float32)
            payloads = [t[2] for t in transitions]
            actions = np.asarray([int(t[3]) for t in transitions], dtype=np.int64)
            rewards = np.asarray([float(t[4]) for t in transitions], dtype=np.float32)
            dones = np.asarray([float(t[5]) for t in transitions], dtype=np.float32)
            finished = np.asarray([float(t[6]) for t in transitions], dtype=np.float32)
            old_lp = np.asarray([float(t[7]) for t in transitions], dtype=np.float32)
            values = np.asarray([float(t[8]) for t in transitions], dtype=np.float32)
            masks = np.stack([t[9] for t in transitions], axis=0).astype(np.float32)

            adv = np.zeros(t_len, dtype=np.float32)
            gae = 0.0
            next_value = 0.0 if (t_len > 0 and finished[-1] > 0.5) else float(values[-1])
            for i in reversed(range(t_len)):
                next_v = next_value if i == t_len - 1 else float(values[i + 1])
                delta = rewards[i] + gamma * next_v * (1.0 - finished[i]) - values[i]
                gae = delta + gamma * gae_lambda * gae * (1.0 - dones[i])
                adv[i] = gae
            ret = adv + values

            all_base.append(base_arr[:, :base_dim])
            all_pool.append(pool_arr[:, :base_dim])
            all_payload.extend(payloads)
            all_act.append(actions)
            all_old_lp.append(old_lp)
            all_adv.append(adv)
            all_ret.append(ret)
            all_mask.append(masks)

    if not all_base:
        return None

    base_arr = np.concatenate(all_base, axis=0)
    pool_arr = np.concatenate(all_pool, axis=0)
    act_arr = np.concatenate(all_act, axis=0)
    old_lp_arr = np.concatenate(all_old_lp, axis=0)
    adv_arr = np.concatenate(all_adv, axis=0)
    ret_arr = np.concatenate(all_ret, axis=0)
    mask_arr = np.concatenate(all_mask, axis=0)

    adv_mean = float(np.mean(adv_arr))
    adv_std = float(np.std(adv_arr) + 1e-8)
    adv_norm = np.clip((adv_arr - adv_mean) / adv_std, -5.0, 5.0)

    return {
        "base": torch.from_numpy(base_arr).float().to(device),
        "pool": torch.from_numpy(pool_arr).float().to(device),
        "payload": all_payload,
        "act": torch.from_numpy(act_arr).long().to(device),
        "old_lp": torch.from_numpy(old_lp_arr).float().to(device),
        "adv": torch.from_numpy(adv_norm).float().to(device),
        "ret": torch.from_numpy(ret_arr).float().to(device),
        "mask": torch.from_numpy(mask_arr).float().to(device),
    }


def _ppo_update(
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: torch.nn.Module,
    head: ActorCriticHead,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any] | None,
    ppo_epochs: int,
    batch_size: int,
    entropy_coef: float,
    value_coef: float,
    clip_eps: float,
    max_grad_norm: float,
    target_kl: float,
    kl_stop_factor: float,
    epoch: int = 0,
    total_epochs: int = 1,
) -> dict[str, float] | None:
    if batch is None:
        return None

    n = int(batch["act"].shape[0])
    if n <= 0:
        return None

    stats: dict[str, list[float]] = {
        "v_loss": [],
        "p_loss": [],
        "entropy": [],
        "approx_kl": [],
        "ratio": [],
        "clip_frac": [],
        "early_stop_skips": [],
    }
    n_minibatches = 0
    kl_skip_threshold = float(target_kl) * float(kl_stop_factor)

    with torch.no_grad():
        head_snapshot = [p.detach().clone() for p in head.parameters()]

    for _ in range(max(1, int(ppo_epochs))):
        idx = np.random.permutation(n)
        epoch_early_stop = False

        for start in range(0, n, max(1, int(batch_size))):
            mb = idx[start:start + max(1, int(batch_size))]
            if len(mb) < 8:
                continue
            n_minibatches += 1

            b_base = batch["base"][mb]
            b_pool = batch["pool"][mb]
            b_payload = [batch["payload"][i] for i in mb.tolist()]
            b_act = batch["act"][mb]
            b_old_lp = batch["old_lp"][mb]
            b_adv = batch["adv"][mb]
            b_ret = batch["ret"][mb]
            b_mask = batch["mask"][mb]

            logits, values = _forward_structured(base_encoder, tree_encoder, fuse, head, b_base, b_payload, b_pool)
            logits = logits.masked_fill(b_mask < 0.5, float("-inf"))
            dist = Categorical(logits=logits)
            lp = dist.log_prob(b_act)
            ent = dist.entropy().mean()

            log_ratio = lp - b_old_lp
            log_ratio = torch.clamp(log_ratio, min=-10.0, max=10.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = ((values - b_ret) ** 2).mean()
            loss = p_loss + float(value_coef) * v_loss - float(entropy_coef) * ent

            with torch.no_grad():
                approx_kl_guard = float(torch.mean((ratio - 1.0) - log_ratio).item())
            if approx_kl_guard > kl_skip_threshold:
                if n_minibatches <= 2 or n_minibatches % 8 == 0:
                    print(
                        f"[PPO] KL early-stop at mb#{n_minibatches}: "
                        f"approx_kl={approx_kl_guard:.4f} > {kl_skip_threshold:.4f} (skip update)"
                    )
                stats["approx_kl"].append(approx_kl_guard)
                stats["ratio"].append(float(ratio.mean().item()))
                stats["v_loss"].append(float(v_loss.item()))
                stats["p_loss"].append(float(p_loss.item()))
                stats["entropy"].append(float(ent.item()))
                stats["clip_frac"].append(0.0)
                stats["early_stop_skips"].append(1.0)
                epoch_early_stop = True
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(base_encoder.parameters())
                + list(tree_encoder.parameters())
                + list(fuse.parameters())
                + list(head.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = float(torch.mean(b_old_lp - lp).item())
                clip_frac = float(((ratio - 1.0).abs() > clip_eps).float().mean().item())

            stats["v_loss"].append(float(v_loss.item()))
            stats["p_loss"].append(float(p_loss.item()))
            stats["entropy"].append(float(ent.item()))
            stats["approx_kl"].append(approx_kl)
            stats["ratio"].append(float(ratio.mean().item()))
            stats["clip_frac"].append(clip_frac)
            stats["early_stop_skips"].append(0.0)

        if epoch_early_stop:
            continue

    with torch.no_grad():
        max_param_change = 0.0
        for p_old, p_new in zip(head_snapshot, head.parameters()):
            diff = float((p_new - p_old).abs().max().item())
            if diff > max_param_change:
                max_param_change = diff

    out = {k: (float(np.mean(v)) if v else 0.0) for k, v in stats.items()}
    out["n_minibatches"] = float(n_minibatches)
    out["n_samples"] = float(n)
    out["head_max_dparam"] = float(max_param_change)

    print(
        format_ppo_update(
            epoch=epoch,
            total_epochs=total_epochs,
            batch_num=n_minibatches,
            n_samples=n,
            approx_kl=out.get("approx_kl", 0.0),
            ratio=out.get("ratio", 0.0),
            p_loss=out.get("p_loss", 0.0),
            v_loss=out.get("v_loss", 0.0),
            clip_frac=out.get("clip_frac", 0.0),
            entropy=out.get("entropy", 0.0),
        )
    )

    return out


def _eval_model(
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: torch.nn.Module,
    head: ActorCriticHead,
    obs_builder,
    cfg: SolverConfig,
    env_source: str,
    pkl_files: list[Path],
    base_dim: int,
    n_episodes: int,
    max_steps: int,
    seed_base: int,
    greedy: bool,
    reward_shaper=None,
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

    device = next(base_encoder.parameters()).device

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
            actions = {}
            for h in handles:
                base_obs, opps, payload = _unwrap_state(observations[h], base_dim)
                n_pool = _neighbor_pool(opps, base_dim)
                mask_np = _legal_action_mask(base_obs, env.agents[h])
                base_t = torch.from_numpy(base_obs).float().unsqueeze(0).to(device)
                pool_t = torch.from_numpy(n_pool).float().unsqueeze(0).to(device)
                mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, _ = _forward_structured(base_encoder, tree_encoder, fuse, head, base_t, [payload], pool_t)
                    logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
                    if greedy:
                        action = int(torch.argmax(logits, dim=-1).item())
                    else:
                        action = int(Categorical(logits=logits).sample().item())
                actions[h] = action

            observations, rewards, done, info = env.step(normalize_actions(actions))
            if reward_shaper is not None:
                rewards = reward_shaper(rewards, done, info, env, actions)
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


def _save_checkpoint(
    checkpoint_path: Path,
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: torch.nn.Module,
    head: ActorCriticHead,
    optimizer: torch.optim.Optimizer,
    base_dim: int,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_encoder": base_encoder.state_dict(),
            "tree_encoder": tree_encoder.state_dict(),
            "fuse": fuse.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "hidden": 64,
            "base_dim": base_dim,
            "kind": "mappo_structured",
        },
        checkpoint_path,
    )


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

    probe_obs, _ = env.reset(random_seed=args.seed)
    if isinstance(probe_obs, dict):
        first_obs = probe_obs[0] if 0 in probe_obs else next(iter(probe_obs.values()))
    else:
        first_obs = probe_obs[0]
    base_dim = infer_obs_dim(first_obs, default=36)

    print(">> MAPPOPolicy")
    print(f">> MAPPOPolicy initialised with BASE_DIM={base_dim}")

    device = torch.device("cpu")
    base_encoder = BaseFeatureEncoder(base_dim=base_dim, hidden=64).to(device)
    tree_encoder = TreePayloadEncoder(hidden=64).to(device)
    fuse = torch.nn.Sequential(
        torch.nn.Linear(128, 64),
        torch.nn.LayerNorm(64),
        torch.nn.LeakyReLU(0.01),
    ).to(device)
    head = ActorCriticHead(hidden=64, action_size=5, base_dim=base_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(base_encoder.parameters())
        + list(tree_encoder.parameters())
        + list(fuse.parameters())
        + list(head.parameters()),
        lr=float(args.lr),
    )

    warmstart_used = False
    if args.init_checkpoint and Path(args.init_checkpoint).exists():
        payload = torch.load(args.init_checkpoint, map_location="cpu")
        if all(k in payload for k in ["base_encoder", "tree_encoder", "fuse", "head"]):
            base_encoder.load_state_dict(payload["base_encoder"], strict=False)
            tree_encoder.load_state_dict(payload["tree_encoder"], strict=False)
            fuse.load_state_dict(payload["fuse"], strict=False)
            head.load_state_dict(payload["head"], strict=False)
            warmstart_used = True
            print(f"[train-mappo] warmstart(structured)={args.init_checkpoint}")
        elif "model_state" in payload:
            print("[train-mappo] init-checkpoint is compact BC/ActorCritic format; structured warmstart skipped.")

    print(
        format_console_row(
            "config",
            "mappo",
            epochs=1,
            episodes=args.episodes,
            lr=args.lr,
            obs=args.obs_variant,
            env_source=args.env_source,
            pkl_dir=str(args.pkl_dir),
            rollout_eps=args.mappo_rollout_episodes,
            ppo_epochs=args.mappo_ppo_epochs,
            batch_size=args.mappo_batch_size,
            max_steps=args.max_episode_steps,
            warmstart=warmstart_used,
            requested_outer_epochs=int(getattr(args, "train_epochs", 1)),
        )
    )

    reward_shaper = build_outcome_reward(args)
    if reward_shaper is not None:
        print(reward_shaper.description())

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
    eval_log: list[dict[str, Any]] = []

    outer_passes = 1
    for epoch in range(outer_passes):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_approx_kl = 0.0
        epoch_ratio = 0.0
        epoch_clip_frac = 0.0
        n_updates = 0

        rolling_done = RollingDoneRatio(window_size=max(1, int(getattr(args, "mappo_done_window", 20))))
        bar = make_progress_bar(total=args.episodes, desc="MAPPO")

        ep = 0
        last_update_stats = {
            "p_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "ratio": 0.0,
            "clip_frac": 0.0,
        }

        prev_n_agents = None
        while ep < args.episodes:
            if curriculum_mode:
                target_rollout_eps = min(args.episodes - ep, max(1, len(curriculum)))
            else:
                target_rollout_eps = min(args.episodes - ep, max(1, int(args.mappo_rollout_episodes)))

            episodes_block: list[dict[str, Any]] = []
            block_collected = 0

            # Detect curriculum transitions.
            # `ep` is the global collected-episode counter and determines the
            # currently active curriculum slot for the progress marker.
            if curriculum_mode and curriculum:
                next_n_agents = curriculum[ep % len(curriculum)]
                if next_n_agents != prev_n_agents:
                    print(f"[CURRICULUM] n_agents={next_n_agents} @ episode {ep+1}/{args.episodes}")
                    prev_n_agents = next_n_agents

            for block_i in range(target_rollout_eps):
                if args.env_source == "pkl":
                    if curriculum_mode and grouped_pkls:
                        # Curriculum next/iteration resolution happens here.
                        # `block_i` advances inside the rollout block and is
                        # mapped to the curriculum array via modulo.
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
                        # Same curriculum index logic for on-the-fly generation.
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
                    base_encoder=base_encoder,
                    tree_encoder=tree_encoder,
                    fuse=fuse,
                    head=head,
                    base_dim=base_dim,
                    seed=args.seed + ep,
                    max_steps=args.max_episode_steps,
                    reward_shaper=reward_shaper,
                )
                if episode is None:
                    continue

                episodes_block.append(episode)
                block_collected += 1

                total_feature_sum += float(episode["feat_sum"])
                total_feature_sq_sum += float(episode["feat_sq_sum"])
                total_feature_count += int(episode["feat_count"])
                total_mask_active += float(episode["mask_active"])
                total_mask_count += float(episode["mask_count"])
                total_action_hist += episode["action_hist"]

                ep_done = int(episode["ep_done"])
                n_agents_ep = int(episode["n_agents"])
                ep_reward = float(episode["ep_reward"])
                done_rate = ep_done / max(1, n_agents_ep)
                done_history.append(done_rate)
                done_window = max(1, int(getattr(args, "mappo_done_window", 20)))
                recent_done_rolling_ration = sum(done_history[-done_window:]) / max(1, len(done_history[-done_window:]))

                rolling_done.update(ep_done, n_agents_ep)
                bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())
                bar.set_postfix_str(
                    f"s={int(episode['steps'])} rew={ep_reward:+.2f} KL={last_update_stats['approx_kl']:+.4f} ent={last_update_stats['entropy']:.4f}"
                )
                bar.update(1)

                ep += 1
                act_hist = {i: int(episode["action_hist"][i].item()) for i in range(5)}
                print(
                    format_episode_compact(
                        "TRAIN",
                        episode=ep,
                        total=args.episodes,
                        done=ep_done,
                        n_agents=n_agents_ep,
                        rew=ep_reward,
                        steps=int(episode["steps"]),
                        approx_kl=last_update_stats["approx_kl"],
                        entropy=last_update_stats["entropy"],
                        acts=act_hist,
                    )
                )

                if tb_logger is not None:
                    ep_idx = ep
                    tb_logger.log_mappo_episode(
                        episode_idx=ep_idx,
                        done_rate=done_rate,
                        episode_len=float(episode["steps"]),
                        total_reward=ep_reward,
                        n_agents=n_agents_ep,
                        deadlock_rate=float(episode["deadlock_rate"]),
                        done_rolling=recent_done_rolling_ration,
                        policy_loss=last_update_stats["p_loss"],
                        value_loss=last_update_stats["v_loss"],
                    )
                    tb_logger.log_scalar("env/n_agents", n_agents_ep, ep_idx)
                    tb_logger.log_scalar("env/done_count", ep_done, ep_idx)
                    ad = episode["action_hist"].to(torch.float32)
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
                        base_encoder=base_encoder,
                        tree_encoder=tree_encoder,
                        fuse=fuse,
                        head=head,
                        obs_builder=obs_builder,
                        cfg=cfg,
                        env_source=args.env_source,
                        pkl_files=pkl_files,
                        base_dim=base_dim,
                        n_episodes=int(getattr(args, "mappo_mid_eval_episodes", 10)),
                        max_steps=args.max_episode_steps,
                        seed_base=args.seed + ep,
                        greedy=bool(getattr(args, "mappo_eval_greedy", False)),
                        reward_shaper=reward_shaper,
                    )
                    eval_entry = {"kind": "mid", "episode": ep, **mid_metrics}
                    eval_log.append(eval_entry)
                    if tb_logger is not None:
                        tb_logger.log_scalar("train_eval/done_rate", float(mid_metrics["done_rate"]), ep)
                        tb_logger.log_scalar("train_eval/deadlock_rate", float(mid_metrics["deadlock_rate"]), ep)
                        tb_logger.log_scalar("train_eval/episode_len", float(mid_metrics["episode_len"]), ep)
                        tb_logger.log_scalar("train_eval/total_reward", float(mid_metrics["total_reward"]), ep)
                    _save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        base_encoder=base_encoder,
                        tree_encoder=tree_encoder,
                        fuse=fuse,
                        head=head,
                        optimizer=optimizer,
                        base_dim=base_dim,
                    )

            batch = _build_ppo_batch(
                episodes=episodes_block,
                gamma=float(args.gamma),
                gae_lambda=0.95,
                base_dim=base_dim,
                device=device,
            )
            update_stats = _ppo_update(
                base_encoder=base_encoder,
                tree_encoder=tree_encoder,
                fuse=fuse,
                head=head,
                optimizer=optimizer,
                batch=batch,
                ppo_epochs=int(args.mappo_ppo_epochs),
                batch_size=int(args.mappo_batch_size),
                entropy_coef=float(args.mappo_entropy_coef),
                value_coef=float(args.mappo_value_coef),
                clip_eps=float(args.mappo_clip_eps),
                max_grad_norm=5.0,
                target_kl=float(args.mappo_target_kl),
                kl_stop_factor=float(args.mappo_kl_stop_factor),
                epoch=1,
                total_epochs=1,
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
                    epoch="1/1",
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
            print(format_console_row("epoch", "mappo", epoch="1/1", status="no_batches"))
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
                    epoch="1/1",
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
                    1,
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
        base_encoder=base_encoder,
        tree_encoder=tree_encoder,
        fuse=fuse,
        head=head,
        obs_builder=obs_builder,
        cfg=cfg,
        env_source=args.env_source,
        pkl_files=pkl_files,
        base_dim=base_dim,
        n_episodes=int(getattr(args, "mappo_mid_eval_episodes", 10)),
        max_steps=args.max_episode_steps,
        seed_base=args.seed + 999999,
        greedy=bool(getattr(args, "mappo_eval_greedy", False)),
        reward_shaper=reward_shaper,
    )
    eval_log.append({"kind": "final", "episode": args.episodes, **final_metrics})
    if tb_logger is not None:
        tb_logger.log_scalar("train_eval/done_rate", float(final_metrics["done_rate"]), args.episodes)
        tb_logger.log_scalar("train_eval/deadlock_rate", float(final_metrics["deadlock_rate"]), args.episodes)
        tb_logger.log_scalar("train_eval/episode_len", float(final_metrics["episode_len"]), args.episodes)
        tb_logger.log_scalar("train_eval/total_reward", float(final_metrics["total_reward"]), args.episodes)

    _save_checkpoint(
        checkpoint_path=checkpoint_path,
        base_encoder=base_encoder,
        tree_encoder=tree_encoder,
        fuse=fuse,
        head=head,
        optimizer=optimizer,
        base_dim=base_dim,
    )

    feature_mean = total_feature_sum / max(1, total_feature_count)
    feature_var = max(0.0, total_feature_sq_sum / max(1, total_feature_count) - feature_mean * feature_mean)
    feature_std = feature_var ** 0.5
    mask_active_ratio = total_mask_active / max(1.0, total_mask_count)
    action_total = int(total_action_hist.sum().item())
    action_hist_text = ", ".join(f"a{idx}={int(count)}" for idx, count in enumerate(total_action_hist.tolist()))

    if tb_logger is not None:
        tb_logger.log_mappo_summary(
            obs_dim=base_dim,
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
            obs_dim=base_dim,
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
