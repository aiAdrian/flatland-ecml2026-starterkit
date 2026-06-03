"""DLA Dataset Recorder

Runs DLA once over all PKL environments and saves (obs, mask, action) tuples
to a .pt dataset file for offline BC training.
"""
from __future__ import annotations

from pathlib import Path
import time

import torch
from flatland.envs.step_utils.states import TrainState

from observations.factory import build_observation_builder
from policy.dla.policy import DLAPolicy
from utils.action_utils import normalize_actions
from utils.env_factory import SolverConfig, build_env, build_env_from_pkl, list_pkl_dataset
from utils.model_utils import infer_obs_dim, split_obs_and_mask
from utils.progress import RollingDoneRatio, format_console_row, format_episode_compact, make_progress_bar


def record_dla_dataset(args, dataset_path: Path) -> Path:
    """Run DLA on all PKL environments (or generated envs) and save dataset."""
    obs_builder = build_observation_builder(
        obs_variant=args.obs_variant,
        debug=bool(getattr(args, "obs_debug", False)),
        search_depth=int(getattr(args, "obs_search_depth", 4)),
    )
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
        print(f"Load environments from disk. # {len(pkl_files):3d} loaded.")

    print(
        format_console_row(
            "config",
            "record",
            expert="dla",
            episodes=args.episodes,
            max_steps=args.max_episode_steps,
            obs=args.obs_variant,
            env_source=args.env_source,
            pkl_dir=str(args.pkl_dir),
        )
    )

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
    total_candidate_samples = 0
    kept_samples = 0

    decision_point_only = str(args.obs_variant).lower() in {"decision_point", "spawn_aware", "conflict_aware"}
    dp_utils = None
    if decision_point_only:
        try:
            from observations.decision_point_utils import DecisionPointUtils

            dp_utils = DecisionPointUtils
        except Exception:
            # Fallback: if legacy utility cannot be imported, keep all samples.
            decision_point_only = False

    n_episodes = args.episodes
    rolling_done = RollingDoneRatio(window_size=20)
    bar = make_progress_bar(total=n_episodes, desc="Record[DLA]")

    print(f"\n[DemoCollect] DLA x {n_episodes} episodes (decision-points only) ...")
    print(f"[DemoCollect] starting ({n_episodes} iterations)")
    t0 = time.perf_counter()

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
                total_candidate_samples += 1
                if decision_point_only and dp_utils is not None:
                    try:
                        agent = env.agents[h]
                        ctype = dp_utils.classify_cell_type(agent, env)
                        if ctype not in ("SWITCH", "MERGING", "PRE_M"):
                            continue
                    except Exception:
                        # Keep robust behavior under API mismatches.
                        pass

                f, m = split_obs_and_mask(obs, obs_dim=obs_dim)
                label = norm_actions[h]
                # Ensure label is valid within mask; fall back to 0 (DO_NOTHING) if needed
                if m[label].item() < 0.5:
                    m[label] = 1.0
                all_feats.append(f)
                all_masks.append(m)
                all_labels.append(label)
                ep_samples += 1
                kept_samples += 1

            observations, _, done, _ = env.step(norm_actions)
            steps += 1

        ep_done = sum(a.state == TrainState.DONE for a in env.agents)
        rolling_done.update(ep_done, env.get_num_agents())
        bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())
        bar.set_postfix_str(f"s={steps} done={ep_done}/{env.get_num_agents()} samples={ep_samples}")
        bar.update(1)
        print(
            format_episode_compact(
                "REC",
                episode=ep + 1,
                total=n_episodes,
                done=ep_done,
                n_agents=env.get_num_agents(),
                samples=ep_samples,
            )
        )

        # Legacy-like periodic progress line (print each episode for full trace).
        now = time.perf_counter()
        elapsed_s = int(now - t0)
        pct = 100.0 * float(ep + 1) / max(1, n_episodes)
        print(f"[DemoCollect] {ep + 1:5d}/{n_episodes:<3d} ({pct:5.1f}%)  demos={len(all_labels)}  ({elapsed_s}s)")

    bar.close()

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if len(all_labels) == 0:
        raise ValueError("No demos collected. Try increasing --episodes or disabling decision-point filtering.")
    dataset = {
        "feats": torch.stack(all_feats),
        "masks": torch.stack(all_masks),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "obs_dim": obs_dim,
        "obs_variant": args.obs_variant,
        "n_samples": len(all_labels),
    }
    torch.save(dataset, dataset_path)
    elapsed = time.perf_counter() - t0
    keep_rate = float(kept_samples) / max(1, int(total_candidate_samples))
    if decision_point_only:
        print(f"[DemoCollect] {len(all_labels)} demos in {elapsed:.1f}s (decision-point keep-rate={keep_rate:.1%})")
    else:
        print(f"[DemoCollect] {len(all_labels)} demos in {elapsed:.1f}s")
    print(format_console_row("record", "dla", dataset=str(dataset_path),
                             n_samples=len(all_labels), obs_dim=obs_dim))
    return dataset_path
