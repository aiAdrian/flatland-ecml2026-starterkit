"""DLA Dataset Recorder

Runs DLA once over all PKL environments and saves (obs, mask, action) tuples
to a .pt dataset file for offline BC training.
"""
from __future__ import annotations

from pathlib import Path

import torch
from flatland.envs.step_utils.states import TrainState

from policy.bc.observation import BCObservationBuilder
from policy.dla.policy import DLAPolicy
from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env, build_env_from_pkl, list_pkl_dataset
from utils.model_utils import infer_obs_dim, split_obs_and_mask
from utils.progress import RollingDoneRatio, format_console_row, make_progress_bar


def record_dla_dataset(args, dataset_path: Path) -> Path:
    """Run DLA on all PKL environments (or generated envs) and save dataset."""
    obs_builder = BCObservationBuilder(obs_variant=args.obs_variant)
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

    # Probe obs_dim from first env (Flatland reset may return dict or list-like)
    probe_obs, _ = env.reset(random_seed=args.seed)
    if isinstance(probe_obs, dict):
        first_obs = probe_obs[0] if 0 in probe_obs else next(iter(probe_obs.values()))
    else:
        first_obs = probe_obs[0]
    obs_dim = infer_obs_dim(first_obs, default=36)

    all_feats: list[torch.Tensor] = []
    all_masks: list[torch.Tensor] = []
    all_labels: list[int] = []

    n_episodes = args.episodes
    rolling_done = RollingDoneRatio(window_size=20)
    bar = make_progress_bar(total=n_episodes, desc="Record[DLA]")

    for ep in range(n_episodes):
        if args.env_source == "pkl":
            pkl_path = pkl_files[ep % len(pkl_files)]
            env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)

        expert = DLAPolicy(seed=args.seed + ep)
        expert.reset_env(env)

        observations, _ = env.reset(random_seed=args.seed + ep)
        done = {"__all__": False}
        steps = 0
        ep_samples = 0

        while not done.get("__all__", False) and steps < args.max_episode_steps:
            handles = list(range(env.get_num_agents()))
            expert_actions = expert.act_many(handles, [env for _ in handles])
            norm_actions = normalize_actions(expert_actions)

            obs_batch = [observations[h] for h in handles]
            for h, obs in zip(handles, obs_batch):
                f, m = split_obs_and_mask(obs, obs_dim=obs_dim)
                label = norm_actions[h]
                # Ensure label is valid within mask; fall back to 0 (DO_NOTHING) if needed
                if m[label].item() < 0.5:
                    m[label] = 1.0
                all_feats.append(f)
                all_masks.append(m)
                all_labels.append(label)
                ep_samples += 1

            observations, _, done, _ = env.step(norm_actions)
            steps += 1

        ep_done = sum(a.state == TrainState.DONE for a in env.agents)
        rolling_done.update(ep_done, env.get_num_agents())
        bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())
        bar.set_postfix_str(f"s={steps} done={ep_done}/{env.get_num_agents()} samples={ep_samples}")
        bar.update(1)
        print(format_console_row("record", "dla", ep=f"{ep + 1}/{n_episodes}", steps=steps,
                                 done=f"{ep_done}/{env.get_num_agents()}", samples=ep_samples))

    bar.close()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "feats": torch.stack(all_feats),
        "masks": torch.stack(all_masks),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "obs_dim": obs_dim,
        "obs_variant": args.obs_variant,
        "n_samples": len(all_labels),
    }
    torch.save(dataset, dataset_path)
    print(format_console_row("record", "dla", dataset=str(dataset_path),
                             n_samples=len(all_labels), obs_dim=obs_dim))
    return dataset_path
