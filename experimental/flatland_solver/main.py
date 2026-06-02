import argparse
from dataclasses import dataclass
from pathlib import Path

from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool

from policy.bc.policy import BCPolicy
from policy.bc.recorder import record_dla_dataset
from policy.bc.trainer import train_bc, train_bc_from_dataset
from policy.dla.observation import DLAFullEnvObservation
from policy.dla.policy import DLAPolicy
from policy.mappo.policy import MAPPOPolicy
from policy.mappo.trainer import train_mappo
from policy.random.observation import RandomObservationBuilder
from policy.random.policy import RandomPolicy
from observations.factory import build_observation_builder
from utils.action_utils import normalize_actions
from utils.env_factory import (
    SolverConfig,
    build_env,
    build_env_from_pkl,
    ensure_pkl_dataset,
    list_pkl_dataset,
    write_pkl_metadata_index,
)
from utils.progress import RollingDoneRatio, format_console_row, format_episode_compact, make_progress_bar
from rewards.outcome_reward import build_outcome_reward
from utils.tb_logger import TBLogger, format_args_text


def parse_curriculum_spec(spec: str | None, mode: str = "auto") -> list[int] | None:
    """
    Parse curriculum specification.
    
    Formats:
    - "5" (repeat mode): return [5] (to be repeated by --curriculum-repeat)
    - "3x10,5x10,7x5" (sequence mode): return [3]*10 + [5]*10 + [7]*5
    - None: return None
    
    Args:
        spec: Curriculum spec string
        mode: "repeat", "sequence", or "auto" to detect
    
    Returns:
        List of agent counts or None
    """
    if spec is None:
        return None
    
    # Try to detect sequence mode (contains 'x' and ',')
    is_sequence = 'x' in spec and (',' in spec or spec.count('x') > 0)
    
    if mode == "auto":
        mode = "sequence" if is_sequence else "repeat"
    
    if mode == "repeat":
        # Just return the single value as list
        return [int(spec.strip())]
    
    if mode == "sequence":
        # Parsing keeps left-to-right order from the spec string.
        # Example: "3x2,5x2" -> [3, 3, 5, 5]
        result = []
        for part in spec.split(','):
            part = part.strip()
            if 'x' in part:
                count_str, repeat_str = part.split('x')
                count = int(count_str.strip())
                repeat = int(repeat_str.strip())
                result.extend([count] * repeat)
            else:
                # Plain number, treat as 1x
                result.append(int(part))
        return result
    
    raise ValueError(f"Unknown curriculum mode: {mode}")


@dataclass
class EvalStats:
    episodes: int
    total_agents: int
    done_agents: int
    total_steps: int


def make_policy_and_observation(args):
    if args.policy == "random":
        return RandomPolicy(seed=args.seed), RandomObservationBuilder()
    if args.policy == "dla":
        return DLAPolicy(seed=args.seed), DLAFullEnvObservation()
    if args.policy == "bc":
        return (
            BCPolicy(checkpoint_path=str(args.bc_checkpoint)),
            build_observation_builder(
                obs_variant=args.obs_variant,
                debug=bool(getattr(args, "obs_debug", False)),
                search_depth=int(getattr(args, "obs_search_depth", 4)),
            ),
        )
    if args.policy == "mappo":
        return (
            MAPPOPolicy(seed=args.seed, checkpoint_path=str(args.mappo_checkpoint)),
            build_observation_builder(
                obs_variant=args.obs_variant,
                debug=bool(getattr(args, "obs_debug", False)),
                search_depth=int(getattr(args, "obs_search_depth", 4)),
            ),
        )
    raise ValueError(f"Unsupported policy: {args.policy}")


def make_solver_config(args) -> SolverConfig:
    return SolverConfig(
        width=args.width,
        height=args.height,
        n_agents=args.n_agents,
        n_cities=args.n_cities,
        max_rails_between_cities=args.max_rails_between_cities,
        max_rail_pairs_in_city=args.max_rail_pairs_in_city,
        seed=args.seed,
    )


def maybe_prepare_pkl_dataset(args) -> None:
    if not args.prepare_pkls:
        return

    print(
        format_console_row(
            "config",
            "pkl-gen",
            dir=str(args.pkl_dir),
            width=args.width,
            height=args.height,
            seed=args.seed,
            overwrite=bool(args.pkl_overwrite),
        )
    )

    cfg = make_solver_config(args)

    if args.agent_curriculum:
        curriculum = [int(x) for x in args.agent_curriculum]
        per_agent_envs = int(args.pkl_num_envs_per_agent)
        expected_total = per_agent_envs * len(curriculum)

        all_pkls = list_pkl_dataset(args.pkl_dir)
        n_cached = 0
        for p in all_pkls:
            stem = p.stem
            prefix = stem.split("_")[0]
            parts = prefix.split("x")
            if len(parts) != 3 or not all(part.isdigit() for part in parts):
                continue
            width = int(parts[0])
            height = int(parts[1])
            n_agents = int(parts[2])
            if width == args.width and height == args.height and n_agents in curriculum:
                n_cached += 1

        written_total = 0
        if n_cached < expected_total or args.pkl_overwrite:
            print(
                f"[pkl] Generating up to {expected_total} envs "
                f"({per_agent_envs} per agent-count) to {args.pkl_dir} ..."
            )
            print(f"[pkl] Agent counts: {curriculum}")
            pkl_bar = make_progress_bar(total=len(curriculum), desc="PKL-Gen")
            # Generation order is deterministic:
            # 1) iterate agent counts in curriculum order,
            # 2) generate `per_agent_envs` files for each entry,
            # 3) seed_start is offset per curriculum position.
            for idx, n_agents in enumerate(curriculum):
                sub_cfg = SolverConfig(
                    width=args.width,
                    height=args.height,
                    n_agents=n_agents,
                    n_cities=args.n_cities,
                    max_rails_between_cities=args.max_rails_between_cities,
                    max_rail_pairs_in_city=args.max_rail_pairs_in_city,
                    seed=args.seed,
                )
                written = ensure_pkl_dataset(
                    cfg=sub_cfg,
                    out_dir=args.pkl_dir,
                    obs_builder_factory=RandomObservationBuilder,
                    count=per_agent_envs,
                    seed_start=args.pkl_seed_start + idx * per_agent_envs,
                    overwrite=args.pkl_overwrite,
                )
                written_total += len(written)
                pkl_bar.set_postfix_str(f"agents={n_agents} generated={written_total}/{expected_total}")
                pkl_bar.update(1)
            pkl_bar.close()
        else:
            print(f"[pkl] Using {n_cached} cached environments at {args.pkl_dir}")

        all_pkls_after = list_pkl_dataset(args.pkl_dir)
        index_path = write_pkl_metadata_index(args.pkl_dir)
        print(
            f"[pkl] dir={args.pkl_dir} generated={written_total} total={len(all_pkls_after)} "
            f"expected_total={expected_total}"
        )
        print(f"[pkl] metadata_index={index_path}")
        return

    written = ensure_pkl_dataset(
        cfg=cfg,
        out_dir=args.pkl_dir,
        obs_builder_factory=RandomObservationBuilder,
        count=args.pkl_count,
        seed_start=args.pkl_seed_start,
        overwrite=args.pkl_overwrite,
    )
    all_pkls = list_pkl_dataset(args.pkl_dir)
    index_path = write_pkl_metadata_index(args.pkl_dir)
    print(
        f"[pkl] dir={args.pkl_dir} generated={len(written)} total={len(all_pkls)} "
        f"seed_start={args.pkl_seed_start}"
    )
    print(f"[pkl] metadata_index={index_path}")


def run_eval(args) -> EvalStats:
    policy, obs_builder = make_policy_and_observation(args)
    cfg = make_solver_config(args)
    pkl_files = list_pkl_dataset(args.pkl_dir) if args.env_source == "pkl" else []
    if args.env_source == "pkl" and not pkl_files:
        raise ValueError(f"No PKL environments found in {args.pkl_dir}. Run with --prepare-pkls first.")

    if args.env_source == "generated":
        env = build_env(cfg=cfg, obs_builder=obs_builder)
    else:
        env = build_env_from_pkl(pkl_files[0], obs_builder=obs_builder)

    if hasattr(policy, "reset_env"):
        policy.reset_env(env)

    reward_shaper = build_outcome_reward(args)
    if reward_shaper is not None:
        print(reward_shaper.description())

    renderer = RenderTool(env, gl="PGL") if args.rendering and env is not None else None

    total_done_agents = 0
    total_agents = 0
    total_steps = 0
    total_reward_sum = 0.0
    total_deadlock_rate = 0.0
    done_rates = []
    deadlock_rates = []
    episode_lengths = []
    rewards_list = []
    rolling_done = RollingDoneRatio(window_size=20)
    bar = make_progress_bar(total=args.episodes, desc=f"Eval[{args.policy}]")

    for episode in range(args.episodes):
        if args.env_source == "pkl":
            pkl_path = pkl_files[episode % len(pkl_files)]
            env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)
            if renderer is not None:
                renderer.close_window()
            renderer = RenderTool(env, gl="PGL") if args.rendering else None

        if hasattr(policy, "reset_env"):
            policy.reset_env(env)

        observations, info = env.reset(random_seed=args.seed + episode)
        del info
        done = {"__all__": False}
        steps = 0
        episode_reward = 0.0

        while not done.get("__all__", False) and steps < args.max_episode_steps:
            handles = list(range(env.get_num_agents()))
            obs_batch = [observations[h] for h in handles]
            actions = policy.act_many(handles, obs_batch)
            observations, rewards, done, info = env.step(normalize_actions(actions))
            if reward_shaper is not None:
                rewards = reward_shaper(rewards, done, info, env, actions)
            if rewards:
                episode_reward += float(sum(rewards.values()))
            steps += 1

            if renderer is not None:
                renderer.render_env(
                    show=True,
                    show_agents=True,
                    show_inactive_agents=False,
                    show_observations=False,
                    show_predictions=False,
                    frames=False,
                )

        episode_done = sum(a.state == TrainState.DONE for a in env.agents)
        active_or_blocked = sum(
            a.state in (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION)
            for a in env.agents
        )
        deadlock_rate = active_or_blocked / max(1, env.get_num_agents())
        total_done_agents += episode_done
        total_agents += env.get_num_agents()
        total_steps += steps
        total_reward_sum += episode_reward
        total_deadlock_rate += deadlock_rate
        done_rates.append(episode_done / max(1, env.get_num_agents()))
        deadlock_rates.append(deadlock_rate)
        episode_lengths.append(float(steps))
        rewards_list.append(float(episode_reward))
        rolling_done.update(episode_done, env.get_num_agents())
        bar.set_secondary(rolling_done.window_ratio(), rolling_done.format_postfix())

        if getattr(args, "tb_logger", None) is not None:
            args.tb_logger.log_eval_episode(
                episode_idx=episode + 1,
                done_rate=episode_done / max(1, env.get_num_agents()),
                episode_len=float(steps),
                total_reward=episode_reward,
                deadlock_rate=deadlock_rate,
            )

        running_done = sum(done_rates) / max(1, len(done_rates))
        running_dlk = sum(deadlock_rates) / max(1, len(deadlock_rates))
        bar.set_postfix_str(f"agents={episode_done}/{env.get_num_agents()} done={running_done:>4.0%} dlk={running_dlk:>4.0%} s={steps} r={episode_reward:+.1f}")
        bar.update(1)

        print(format_console_row("eval", args.policy, ep=f"{episode + 1}/{args.episodes}", steps=steps, done=f"{episode_done}/{env.get_num_agents()}", deadlock=deadlock_rate, reward=episode_reward))
        if args.policy == "dla" and hasattr(policy, "get_debug_stats"):
            stats = policy.get_debug_stats()
            print(format_console_row("eval", "dla", calls_total=stats.get("calls_total", 0), calls_episode=stats.get("calls_episode", 0)))

    bar.close()

    if renderer is not None:
        renderer.close_window()

    if getattr(args, "tb_logger", None) is not None:
        args.tb_logger.log_eval_summary(
            episodes=args.episodes,
            success_rate=total_done_agents / max(1, total_agents),
            avg_steps=total_steps / max(1, args.episodes),
            avg_reward=total_reward_sum / max(1, args.episodes),
            avg_deadlock_rate=total_deadlock_rate / max(1, args.episodes),
        )

    # Legacy-like condensed eval result block.
    mean_done = sum(done_rates) / max(1, len(done_rates))
    mean_dl = sum(deadlock_rates) / max(1, len(deadlock_rates))
    mean_len = sum(episode_lengths) / max(1, len(episode_lengths))
    mean_rew = sum(rewards_list) / max(1, len(rewards_list))
    print("-" * 60)
    print(f"[Eval]  {args.policy}  RESULT")
    print(f"  done_rate     = {mean_done:.3f}")
    print(f"  deadlock_rate = {mean_dl:.3f}")
    print(f"  episode_len   = {mean_len:.1f}")
    print(f"  total_reward  = {mean_rew:+.2f}")
    print()

    return EvalStats(
        episodes=args.episodes,
        total_agents=total_agents,
        done_agents=total_done_agents,
        total_steps=total_steps,
    )


def run_record(args) -> None:
    """Phase 1: Run DLA once over all PKLs and save (obs, mask, action) dataset."""
    args.env_source = args.env_source  # already set
    record_dla_dataset(args, dataset_path=args.dataset_path)


def _ensure_bc_default_pkls(args) -> None:
    if args.env_source != "pkl":
        return
    pkl_files = list_pkl_dataset(args.pkl_dir)
    if pkl_files:
        print(f"[Env] Using {len(pkl_files)} cached environments at {args.pkl_dir}")
        return

    # Default BC curriculum: 5 envs per agent-count.
    if not args.agent_curriculum:
        args.agent_curriculum = [1, 2, 3, 4, 5, 7, 10, 15, 20]
    if int(args.pkl_num_envs_per_agent) <= 0:
        args.pkl_num_envs_per_agent = 5

    expected_total = int(args.pkl_num_envs_per_agent) * len(args.agent_curriculum)
    print(f"[Env] Generating {expected_total} envs ({args.pkl_num_envs_per_agent} per agent-count) to {args.pkl_dir} ...")
    print(f"[Env] Agent counts: {args.agent_curriculum}")

    args.prepare_pkls = True
    maybe_prepare_pkl_dataset(args)


def run_bc_mode(args) -> None:
    print("\n" + "=" * 60)
    print("  Flatland MARL - mode=bc")
    print("=" * 60 + "\n")
    print("=" * 70)
    print("BEHAVIOR CLONING from DLA expert (decision-points only)")
    print("=" * 70)

    # BC mode semantics: the BC pipeline runs against a PKL cache by default.
    args.policy = "bc"
    args.env_source = "pkl"
    _ensure_bc_default_pkls(args)

    record_args = argparse.Namespace(**vars(args))
    record_args.episodes = int(args.bc_demo_episodes)
    run_record(record_args)

    train_args = argparse.Namespace(**vars(args))
    train_args.episodes = int(args.bc_demo_episodes)
    train_args.train_epochs = int(args.bc_epochs)
    run_train(train_args)

    print(f"\n[BC] Evaluating BC-pretrained policy on {int(args.bc_eval_episodes)} episodes ...")

    eval_args = argparse.Namespace(**vars(args))
    eval_args.policy = "mappo"
    eval_args.mappo_checkpoint = Path(args.bc_checkpoint)
    eval_args.episodes = int(args.bc_eval_episodes)
    stats = run_eval(eval_args)
    done_rate = stats.done_agents / max(1, stats.total_agents)
    print(f"[BC] BC-policy done_rate = {done_rate:.3f}")


def run_train(args) -> None:
    if args.policy == "bc":
        args.obs_builder = build_observation_builder(
            obs_variant=args.obs_variant,
            debug=bool(getattr(args, "obs_debug", False)),
            search_depth=int(getattr(args, "obs_search_depth", 4)),
        )
        if getattr(args, "dataset_path", None) and args.dataset_path.exists():
            # Offline mode: train from pre-recorded DLA dataset (fast, no DLA overhead)
            train_bc_from_dataset(args, checkpoint_path=args.bc_checkpoint, tb_logger=getattr(args, "tb_logger", None))
        else:
            # Online mode: DLA runs live during training
            train_bc(args, checkpoint_path=args.bc_checkpoint, tb_logger=getattr(args, "tb_logger", None))
        return
    if args.policy == "mappo":
        args.obs_builder = build_observation_builder(
            obs_variant=args.obs_variant,
            debug=bool(getattr(args, "obs_debug", False)),
            search_depth=int(getattr(args, "obs_search_depth", 4)),
        )
        train_mappo(args, checkpoint_path=args.mappo_checkpoint, tb_logger=getattr(args, "tb_logger", None))
        return
    raise ValueError("Train mode currently supports --policy bc or --policy mappo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental Flatland Solver")
    parser.add_argument("--mode", choices=["eval", "train", "record", "bc"], default="eval")
    parser.add_argument("--policy", choices=["random", "dla", "bc", "mappo"], default="random")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--rendering", action="store_true")

    parser.add_argument("--n-agents", type=int, default=5)
    parser.add_argument("--width", type=int, default=35)
    parser.add_argument("--height", type=int, default=35)
    parser.add_argument("--n-cities", type=int, default=3)
    parser.add_argument("--max-rails-between-cities", type=int, default=2)
    parser.add_argument("--max-rail-pairs-in-city", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--obs-variant",
        choices=["fast_tree", "decision_point", "spawn_aware", "conflict_aware"],
        default="fast_tree",
    )
    parser.add_argument(
        "--obs",
        dest="obs_variant",
        choices=["fast_tree", "decision_point", "spawn_aware", "conflict_aware"],
        help="Legacy alias for --obs-variant",
    )
    parser.add_argument("--env-source", choices=["generated", "pkl"], default="generated")
    parser.add_argument("--pkl-dir", type=Path, default=Path("generated_envs"))
    parser.add_argument("--pkl-count", type=int, default=32)
    parser.add_argument("--pkl-seed-start", type=int, default=1000)
    parser.add_argument("--pkl-overwrite", action="store_true")
    parser.add_argument("--agent-curriculum", nargs="+", type=int, default=None)
    parser.add_argument(
        "--curriculum-spec",
        type=str,
        default=None,
        help="Curriculum specification: '5' (repeat once) or '3x10,5x10' (sequence: 10x3 agents, 10x5 agents)",
    )
    parser.add_argument(
        "--curriculum-mode",
        choices=["auto", "repeat", "sequence"],
        default="auto",
        help="How to interpret --curriculum-spec: auto=detect, repeat=single value, sequence=AxN,BxN format",
    )
    parser.add_argument(
        "--curriculum-repeat",
        type=int,
        default=None,
        help="When using --curriculum-spec in repeat mode, repeat this many times (e.g. --curriculum-spec 5 --curriculum-repeat 10 → [5]*10)",
    )
    parser.add_argument("--pkl-num-envs-per-agent", type=int, default=5)
    parser.add_argument("--prepare-pkls", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--debug-checks", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mappo-rollout-episodes", type=int, default=10,
                        help="Episodes to collect before one PPO update block (non-curriculum)")
    parser.add_argument("--mappo-ppo-epochs", type=int, default=4,
                        help="Number of PPO epochs per update block")
    parser.add_argument("--mappo-batch-size", type=int, default=256,
                        help="Mini-batch size for MAPPO PPO updates")
    parser.add_argument("--mappo-entropy-coef", type=float, default=0.02,
                        help="Entropy regularization coefficient for MAPPO PPO loss")
    parser.add_argument("--mappo-value-coef", type=float, default=0.5,
                        help="Value loss coefficient for MAPPO PPO loss")
    parser.add_argument("--mappo-clip-eps", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--mappo-target-kl", type=float, default=0.05,
                        help="Target KL for legacy-style early-stop guard")
    parser.add_argument("--mappo-kl-stop-factor", type=float, default=1.5,
                        help="Early-stop threshold factor applied to target KL")
    parser.add_argument("--mappo-done-window", type=int, default=50,
                        help="Rolling window size for done-rate metric in MAPPO train logs")
    parser.add_argument("--mappo-mid-eval-every", type=int, default=0,
                        help="Run MAPPO mid-training eval every N collected episodes (0 disables)")
    parser.add_argument("--mappo-mid-eval-episodes", type=int, default=10,
                        help="Number of episodes for each mid/final MAPPO eval run")
    parser.add_argument("--mappo-eval-greedy", action="store_true",
                        help="Use greedy argmax actions during MAPPO mid/final eval")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--mappo-checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path("datasets/dla_dataset.pt"),
                        help="Path for DLA-recorded dataset (--mode record writes, --mode train reads)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for offline BC training from dataset")
    parser.add_argument("--obs-debug", "--legacy-obs-debug", dest="obs_debug", action="store_true",
                        help="Enable verbose observation debug/performance reporting")
    parser.add_argument("--obs-search-depth", "--legacy-obs-search-depth", dest="obs_search_depth", type=int, default=4,
                        help="Search depth for observation variants")
    parser.add_argument("--disable-outcome-reward", action="store_true",
                        help="Disable legacy OutcomeBasedReward shaping for BC/MAPPO train/eval")
    parser.add_argument("--bc-demo-episodes", type=int, default=200,
                        help="Number of DLA demo episodes for --mode bc")
    parser.add_argument("--bc-epochs", type=int, default=10,
                        help="Number of offline BC epochs for --mode bc")
    parser.add_argument("--bc-eval-episodes", type=int, default=20,
                        help="Number of eval episodes after BC in --mode bc")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Parse curriculum specification
    if args.curriculum_spec:
        curriculum = parse_curriculum_spec(args.curriculum_spec, mode=args.curriculum_mode)
        if args.curriculum_mode == "repeat" or (args.curriculum_mode == "auto" and "x" not in args.curriculum_spec):
            # Repeat mode: expand by curriculum_repeat count (default 1)
            repeat_count = args.curriculum_repeat if args.curriculum_repeat else 1
            curriculum = curriculum * repeat_count
        args.agent_curriculum = curriculum

    maybe_prepare_pkl_dataset(args)
    if args.prepare_only:
        return

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.bc_checkpoint is None:
        args.bc_checkpoint = args.checkpoint_dir / "bc.pt"
    if args.mappo_checkpoint is None:
        args.mappo_checkpoint = args.checkpoint_dir / "mappo.pt"

    args.tb_logger = None
    if not args.no_tensorboard:
        run_dir = TBLogger.build_run_dir(
            runs_root=args.runs_dir,
            mode=args.mode,
            policy=args.policy,
            n_agents=args.n_agents,
            width=args.width,
            height=args.height,
            n_cities=args.n_cities,
            seed=args.seed,
        )
        args.tb_logger = TBLogger(run_dir)
        args.tb_logger.log_hparams_text(format_args_text(args))

    try:
        if args.mode == "eval":
            stats = run_eval(args)
            success_rate = stats.done_agents / max(1, stats.total_agents)
            avg_steps = stats.total_steps / max(1, stats.episodes)
            print(format_console_row("summary", args.policy, episodes=stats.episodes, success_rate=success_rate, avg_steps=avg_steps, env_source=args.env_source))
            return

        if args.mode == "record":
            run_record(args)
            return

        if args.mode == "bc":
            run_bc_mode(args)
            return

        run_train(args)
    finally:
        if args.tb_logger is not None:
            args.tb_logger.close()


if __name__ == "__main__":
    main()
