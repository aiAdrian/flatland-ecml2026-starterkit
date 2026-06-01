from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from flatland.envs.step_utils.states import TrainState

from policy.dla.observation import DLAFullEnvObservation
from policy.dla.policy import DLAPolicy
from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env, build_env_from_pkl, list_pkl_dataset
from utils.model_utils import DiscretePolicyNet, infer_obs_dim, split_obs_and_mask
from utils.progress import RollingDoneRatio, format_console_row, format_fixed_metrics, make_progress_bar


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
    del expert_obs_builder  # expert receives env directly in act_many

    probe_obs, _probe_info = env.reset(random_seed=args.seed)
    del _probe_info
    if isinstance(probe_obs, dict):
        first_obs = probe_obs[0] if 0 in probe_obs else next(iter(probe_obs.values()))
    else:
        first_obs = probe_obs[0]
    obs_dim = infer_obs_dim(first_obs, default=36)
    model = DiscretePolicyNet(obs_dim=obs_dim, action_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.train_epochs):
        losses = []
        correct = 0
        total = 0
        rows_without_valid_before_fix = 0
        labels_forced_valid = 0
        rolling_done = RollingDoneRatio(window_size=20)
        bar = make_progress_bar(total=args.episodes, desc=f"BC[{epoch + 1}/{args.train_epochs}]")
        for ep in range(args.episodes):
            if args.env_source == "pkl":
                pkl_path = pkl_files[(epoch * max(1, args.episodes) + ep) % len(pkl_files)]
                env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)

            expert = DLAPolicy(seed=args.seed + epoch * 1000 + ep)
            expert.reset_env(env)

            observations, info = env.reset(random_seed=args.seed + ep + epoch * 1000)
            del info
            done = {"__all__": False}
            steps = 0
            ep_losses = []

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
                    f, m = split_obs_and_mask(obs, obs_dim=obs_dim)
                    feats.append(f)
                    masks.append(m)
                    labels.append(norm_expert_actions[h])

                feat_t = torch.stack(feats, dim=0)
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

                logits = model(feat_t)
                logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
                loss = F.cross_entropy(logits, label_t)

                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    correct += int(torch.sum(pred == label_t).item())
                    total += int(label_t.numel())

                opt.zero_grad()
                loss.backward()
                opt.step()

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
            print(format_console_row("train", "bc", epoch=f"{epoch + 1}/{args.train_epochs}", ep=f"{ep + 1}/{args.episodes}", steps=steps, done=f"{ep_done}/{env.get_num_agents()}", avg_loss=ep_avg_loss))

        bar.close()

        avg_loss = sum(losses) / max(1, len(losses))
        acc = correct / max(1, total)
        dbg = ""
        if getattr(args, "debug_checks", False):
            dbg = (
                f" mask_rows_fixed={rows_without_valid_before_fix}"
                f" labels_forced_valid={labels_forced_valid}"
            )
        print(format_console_row("epoch", "bc", epoch=f"{epoch + 1}/{args.train_epochs}", avg_loss=avg_loss, acc=acc, debug=dbg.strip() if dbg else "-"))
        if tb_logger is not None:
            tb_logger.log_bc_epoch(epoch + 1, avg_loss=avg_loss, accuracy=acc)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": 5,
            "kind": "bc",
        },
        checkpoint_path,
    )
    print(format_console_row("checkpoint", "bc", path=str(checkpoint_path)))
    return checkpoint_path


def train_bc_from_dataset(args, checkpoint_path: Path, tb_logger=None) -> Path:
    """Offline BC training: loads a pre-recorded DLA dataset and trains for N epochs.

    Much faster than online BC since DLA runs once during recording, not during training.
    """
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
    n_samples = feats.shape[0]

    print(format_console_row("dataset", "bc", path=str(dataset_path),
                             n_samples=n_samples, obs_dim=obs_dim))

    model = DiscretePolicyNet(obs_dim=obs_dim, action_dim=5)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    batch_size = getattr(args, "batch_size", 256)
    indices = torch.arange(n_samples)

    for epoch in range(args.train_epochs):
        # Shuffle each epoch
        perm = torch.randperm(n_samples)
        feats_s = feats[perm]
        masks_s = masks[perm]
        labels_s = labels[perm]

        losses = []
        correct = 0
        total = 0
        bar = make_progress_bar(
            total=(n_samples + batch_size - 1) // batch_size,
            desc=f"BC-offline[{epoch + 1}/{args.train_epochs}]",
        )

        for start in range(0, n_samples, batch_size):
            feat_t = feats_s[start:start + batch_size]
            mask_t = masks_s[start:start + batch_size]
            label_t = labels_s[start:start + batch_size]

            logits = model(feat_t)
            logits = logits.masked_fill(mask_t < 0.5, float("-inf"))
            loss = F.cross_entropy(logits, label_t)

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += int(torch.sum(pred == label_t).item())
                total += int(label_t.numel())

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            bar.update(1)

        bar.close()
        avg_loss = sum(losses) / max(1, len(losses))
        acc = correct / max(1, total)
        print(format_console_row("epoch", "bc-offline", epoch=f"{epoch + 1}/{args.train_epochs}",
                                 avg_loss=avg_loss, acc=acc, n_samples=n_samples))
        if tb_logger is not None:
            tb_logger.log_bc_epoch(epoch + 1, avg_loss=avg_loss, accuracy=acc)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": 5,
            "kind": "bc",
        },
        checkpoint_path,
    )
    print(format_console_row("checkpoint", "bc", path=str(checkpoint_path)))
    return checkpoint_path
