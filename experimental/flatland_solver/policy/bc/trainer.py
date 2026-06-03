from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flatland.envs.step_utils.states import TrainState

from policy.dla.observation import DLAFullEnvObservation
from policy.dla.policy import DLAPolicy
from policy.mappo.actor_critic_head import ActorCriticHead
from policy.mappo.base_feature_encoder import BaseFeatureEncoder
from policy.mappo.tree_payload_encoder import TreePayloadEncoder
from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env, build_env_from_pkl, list_pkl_dataset
from utils.model_utils import infer_obs_dim, split_obs_and_mask
from utils.progress import RollingDoneRatio, format_console_row, format_episode_compact, make_progress_bar


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


def _forward_logits(
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: nn.Module,
    head: ActorCriticHead,
    base_t: torch.Tensor,
    payloads: list[dict[str, Any]],
) -> torch.Tensor:
    emb_base = base_encoder(base_t)
    emb_tree = tree_encoder.forward_batch(payloads)
    emb = fuse(torch.cat([emb_base, emb_tree], dim=-1))
    return head.actor_logits(emb)


def _build_legacy_bc_stack(base_dim: int):
    hidden = 64
    base_encoder = BaseFeatureEncoder(base_dim=base_dim, hidden=hidden)
    tree_encoder = TreePayloadEncoder(hidden=hidden)
    fuse = nn.Sequential(
        nn.Linear(hidden * 2, hidden),
        nn.LayerNorm(hidden),
        nn.LeakyReLU(0.01),
    )
    head = ActorCriticHead(hidden=hidden, action_size=5, base_dim=base_dim)
    return base_encoder, tree_encoder, fuse, head, hidden


def _maybe_resume_bc(
    checkpoint_path: Path,
    base_encoder: BaseFeatureEncoder,
    tree_encoder: TreePayloadEncoder,
    fuse: nn.Module,
    head: ActorCriticHead,
    optimizer: torch.optim.Optimizer,
) -> bool:
    if not checkpoint_path.exists():
        return False

    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return False

    if not all(k in payload for k in ["base_encoder", "tree_encoder", "fuse", "head"]):
        return False

    base_encoder.load_state_dict(payload["base_encoder"], strict=False)
    tree_encoder.load_state_dict(payload["tree_encoder"], strict=False)
    fuse.load_state_dict(payload["fuse"], strict=False)
    head.load_state_dict(payload["head"], strict=False)

    if "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
        except Exception:
            # Keep resumed model weights even if optimizer state is incompatible.
            pass

    return True


def train_bc(args, checkpoint_path: Path, tb_logger=None) -> Path:
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

    expert_obs_builder = DLAFullEnvObservation()
    del expert_obs_builder

    probe_obs, _probe_info = env.reset(random_seed=args.seed)
    del _probe_info
    if isinstance(probe_obs, dict):
        first_obs = probe_obs[0] if 0 in probe_obs else next(iter(probe_obs.values()))
    else:
        first_obs = probe_obs[0]

    obs_dim = infer_obs_dim(first_obs, default=36)
    base_obs, _, _ = _unwrap_state(first_obs, obs_dim)
    base_dim = int(base_obs.shape[0])

    base_encoder, tree_encoder, fuse, head, hidden = _build_legacy_bc_stack(base_dim=base_dim)
    optimizer = torch.optim.Adam(
        list(base_encoder.parameters())
        + list(tree_encoder.parameters())
        + list(fuse.parameters())
        + list(head.parameters()),
        lr=args.lr,
    )
    resumed = _maybe_resume_bc(checkpoint_path, base_encoder, tree_encoder, fuse, head, optimizer)

    print(
        format_console_row(
            "config",
            "bc-online",
            epochs=1,
            episodes=args.episodes,
            lr=args.lr,
            obs=args.obs_variant,
            env_source=args.env_source,
            pkl_dir=str(args.pkl_dir),
            max_steps=args.max_episode_steps,
            base_dim=base_dim,
            resumed=resumed,
            requested_outer_epochs=int(getattr(args, "train_epochs", 1)),
        )
    )

    online_passes = 1
    for epoch in range(online_passes):
        losses = []
        correct = 0
        total = 0
        rows_without_valid_before_fix = 0
        labels_forced_valid = 0
        rolling_done = RollingDoneRatio(window_size=20)
        bar = make_progress_bar(total=args.episodes, desc="BC")

        for ep in range(args.episodes):
            if args.env_source == "pkl":
                pkl_path = pkl_files[ep % len(pkl_files)]
                env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)

            expert = DLAPolicy(seed=args.seed + ep)
            expert.reset_env(env)

            observations, info = env.reset(random_seed=args.seed + ep)
            del info
            done = {"__all__": False}
            steps = 0
            ep_losses = []

            while not done.get("__all__", False) and steps < args.max_episode_steps:
                handles = list(range(env.get_num_agents()))
                expert_actions = expert.act_many(handles, [env for _ in handles])
                norm_expert_actions = normalize_actions(expert_actions)

                obs_batch = [observations[h] for h in handles]
                base_rows = []
                payload_rows = []
                masks = []
                labels = []

                for h, obs in zip(handles, obs_batch):
                    b_obs, opps, payload = _unwrap_state(obs, base_dim)
                    n_pool = _neighbor_pool(opps, base_dim)
                    # Preserve legacy behavior: average own + neighbor pool.
                    base_rows.append(0.5 * (b_obs + n_pool))
                    payload_rows.append(payload)

                    _, m = split_obs_and_mask(obs, obs_dim=obs_dim)
                    masks.append(m)
                    labels.append(norm_expert_actions[h])

                base_t = torch.as_tensor(np.stack(base_rows, axis=0), dtype=torch.float32)
                mask_t = torch.stack(masks, dim=0)
                row_has_valid = torch.sum(mask_t > 0.5, dim=1) > 0
                if torch.any(~row_has_valid):
                    rows_without_valid_before_fix += int(torch.sum(~row_has_valid).item())
                    mask_t[~row_has_valid] = 1.0

                label_t = torch.as_tensor(labels, dtype=torch.long)
                for i in range(mask_t.shape[0]):
                    if mask_t[i, label_t[i]].item() < 0.5:
                        labels_forced_valid += 1
                    mask_t[i, label_t[i]] = 1.0

                logits = _forward_logits(base_encoder, tree_encoder, fuse, head, base_t, payload_rows)
                logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
                loss = F.cross_entropy(logits, label_t)

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    correct += int(torch.sum(pred == label_t).item())
                    total += int(label_t.numel())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(float(loss.item()))
                ep_losses.append(float(loss.item()))
                observations, rewards, done, info = env.step(norm_expert_actions)
                del rewards, info
                steps += 1

            ep_done = sum(a.state == TrainState.DONE for a in env.agents)
            rolling_done.update(ep_done, env.get_num_agents())
            bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())
            ep_avg_loss = sum(ep_losses) / max(1, len(ep_losses))
            bar.set_postfix_str(f"s={steps} l={ep_avg_loss:.4f}")
            bar.update(1)
            print(
                format_console_row(
                    "train",
                    "bc",
                    epoch="1/1",
                    ep=f"{ep + 1}/{args.episodes}",
                    steps=steps,
                    done=f"{ep_done}/{env.get_num_agents()}",
                    avg_loss=ep_avg_loss,
                )
            )

        bar.close()

        avg_loss = sum(losses) / max(1, len(losses))
        acc = correct / max(1, total)
        dbg = ""
        if getattr(args, "debug_checks", False):
            dbg = f" mask_rows_fixed={rows_without_valid_before_fix} labels_forced_valid={labels_forced_valid}"
        print(format_console_row("epoch", "bc", epoch="1/1", avg_loss=avg_loss, acc=acc, debug=dbg.strip() if dbg else "-"))
        if tb_logger is not None:
            tb_logger.log_bc_epoch(1, avg_loss=avg_loss, accuracy=acc)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_encoder": base_encoder.state_dict(),
            "tree_encoder": tree_encoder.state_dict(),
            "fuse": fuse.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "base_dim": int(base_dim),
            "hidden": int(hidden),
            "obs_dim": int(obs_dim),
            "action_dim": 5,
            "kind": "bc",
        },
        checkpoint_path,
    )
    print(format_console_row("checkpoint", "bc", path=str(checkpoint_path)))
    return checkpoint_path


def train_bc_from_dataset(args, checkpoint_path: Path, tb_logger=None) -> Path:
    """Offline BC training from recorded dataset using MAPPO architecture."""
    dataset_path = args.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. "
            "Run with --mode record --policy bc first."
        )

    data = torch.load(dataset_path, map_location="cpu")
    feats: torch.Tensor = data["feats"]
    masks: torch.Tensor = data["masks"]
    labels: torch.Tensor = data["labels"]
    obs_dim: int = int(data.get("obs_dim", 36))
    obs_variant = str(data.get("obs_variant", getattr(args, "obs_variant", "unknown")))
    n_samples = feats.shape[0]

    base_dim = int(obs_dim)
    base_encoder, tree_encoder, fuse, head, hidden = _build_legacy_bc_stack(base_dim=base_dim)
    optimizer = torch.optim.Adam(
        list(base_encoder.parameters())
        + list(tree_encoder.parameters())
        + list(fuse.parameters())
        + list(head.parameters()),
        lr=args.lr,
    )
    resumed = _maybe_resume_bc(checkpoint_path, base_encoder, tree_encoder, fuse, head, optimizer)

    batch_size = getattr(args, "batch_size", 256)
    print(format_console_row("dataset", "bc", path=str(dataset_path), n_samples=n_samples, obs_dim=obs_dim))
    print(
        format_console_row(
            "config",
            "bc-offline",
            epochs=args.train_epochs,
            batch_size=batch_size,
            lr=args.lr,
            obs=obs_variant,
            resumed=resumed,
        )
    )

    legal_for_expert = masks[torch.arange(n_samples), labels] > 0.5
    keep = torch.nonzero(legal_for_expert, as_tuple=False).reshape(-1)
    if int(keep.numel()) < int(n_samples):
        print(f"[BC] Filtered {n_samples - int(keep.numel())}/{n_samples} demos with illegal expert actions.")
    if int(keep.numel()) == 0:
        print("[BC] No valid demos after filtering.")
        return checkpoint_path

    feats = feats[keep]
    masks = masks[keep]
    labels = labels[keep]
    n_samples = int(feats.shape[0])

    print(f"[BC] Training on {n_samples} demos for {args.train_epochs} epochs (batch={batch_size})...")

    for epoch in range(args.train_epochs):
        perm = torch.randperm(n_samples)
        feats_s = feats[perm]
        masks_s = masks[perm]
        labels_s = labels[perm]

        losses = []
        correct = 0
        total = 0

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            feat_t = feats_s[start:end]
            mask_t = masks_s[start:end]
            label_t = labels_s[start:end]
            if int(label_t.shape[0]) < 4:
                continue

            if feat_t.shape[1] < base_dim:
                pad = torch.zeros((feat_t.shape[0], base_dim - feat_t.shape[1]), dtype=feat_t.dtype)
                base_t = torch.cat([feat_t, pad], dim=1)
            else:
                base_t = feat_t[:, :base_dim]

            payloads = [{} for _ in range(int(base_t.shape[0]))]
            logits = _forward_logits(base_encoder, tree_encoder, fuse, head, base_t, payloads)
            logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
            loss = F.cross_entropy(logits, label_t)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += int(torch.sum(pred == label_t).item())
                total += int(label_t.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        avg_loss = sum(losses) / max(1, len(losses))
        acc = correct / max(1, total)
        print(
            format_episode_compact(
                "BC",
                episode=epoch + 1,
                total=args.train_epochs,
                loss=avg_loss,
                acc=acc,
            )
        )
        if tb_logger is not None:
            tb_logger.log_bc_epoch(epoch + 1, avg_loss=avg_loss, accuracy=acc)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "base_encoder": base_encoder.state_dict(),
            "tree_encoder": tree_encoder.state_dict(),
            "fuse": fuse.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "base_dim": int(base_dim),
            "hidden": int(hidden),
            "obs_dim": int(obs_dim),
            "action_dim": 5,
            "kind": "bc",
        },
        checkpoint_path,
    )
    print(format_console_row("checkpoint", "bc", path=str(checkpoint_path)))
    return checkpoint_path
