import argparse
from dataclasses import dataclass
from pathlib import Path

from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool

from policy.bc.observation import BCObservationBuilder
from policy.bc.policy import BCPolicy
from policy.bc.trainer import train_bc
from policy.dla.observation import DLAFullEnvObservation
from policy.dla.policy import DLAPolicy
from policy.mappo.observation import MAPPOObservationBuilder
from policy.mappo.policy import MAPPOPolicy
from policy.mappo.trainer import train_mappo
from policy.random.observation import RandomObservationBuilder
from policy.random.policy import RandomPolicy
from utils.action_utils import normalize_actions
from utils.env_factory import (
    SolverConfig,
    build_env,
    build_env_from_pkl,
    ensure_pkl_dataset,
    list_pkl_dataset,
)
from utils.tb_logger import TBLogger, format_args_text


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
            BCObservationBuilder(obs_variant=args.obs_variant),
        )
    if args.policy == "mappo":
        return (
            MAPPOPolicy(seed=args.seed, checkpoint_path=str(args.mappo_checkpoint)),
            MAPPOObservationBuilder(obs_variant=args.obs_variant),
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

    cfg = make_solver_config(args)
    written = ensure_pkl_dataset(
        cfg=cfg,
        out_dir=args.pkl_dir,
        obs_builder_factory=RandomObservationBuilder,
        count=args.pkl_count,
        seed_start=args.pkl_seed_start,
        overwrite=args.pkl_overwrite,
    )
    all_pkls = list_pkl_dataset(args.pkl_dir)
    print(
        f"[pkl] dir={args.pkl_dir} generated={len(written)} total={len(all_pkls)} "
        f"seed_start={args.pkl_seed_start}"
    )


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

    renderer = RenderTool(env, gl="PGL") if args.rendering and env is not None else None

    total_done_agents = 0
    total_agents = 0
    total_steps = 0
    total_reward_sum = 0.0
    total_deadlock_rate = 0.0

    for episode in range(args.episodes):
        if args.env_source == "pkl":
            pkl_path = pkl_files[episode % len(pkl_files)]
            env = build_env_from_pkl(pkl_path, obs_builder=obs_builder)
            if renderer is not None:
                renderer.close_window()
            renderer = RenderTool(env, gl="PGL") if args.rendering else None

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
            if rewards:
                episode_reward += float(sum(rewards.values()))
            deadlocks = float(info.get("deadlocks", 0.0)) if isinstance(info, dict) else 0.0
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

        if getattr(args, "tb_logger", None) is not None:
            args.tb_logger.log_eval_episode(
                episode_idx=episode + 1,
                done_rate=episode_done / max(1, env.get_num_agents()),
                episode_len=float(steps),
                total_reward=episode_reward,
                deadlock_rate=deadlock_rate,
            )

        print(
            f"[eval] episode={episode + 1}/{args.episodes} policy={args.policy} "
            f"steps={steps} done={episode_done}/{env.get_num_agents()}"
        )

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

    return EvalStats(
        episodes=args.episodes,
        total_agents=total_agents,
        done_agents=total_done_agents,
        total_steps=total_steps,
    )


def run_train(args) -> None:
    if args.policy == "bc":
        args.obs_builder = BCObservationBuilder(obs_variant=args.obs_variant)
        train_bc(args, checkpoint_path=args.bc_checkpoint, tb_logger=getattr(args, "tb_logger", None))
        return
    if args.policy == "mappo":
        args.obs_builder = MAPPOObservationBuilder(obs_variant=args.obs_variant)
        train_mappo(args, checkpoint_path=args.mappo_checkpoint, tb_logger=getattr(args, "tb_logger", None))
        return
    raise ValueError("Train mode currently supports --policy bc or --policy mappo")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental Flatland Solver")
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
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
    parser.add_argument("--env-source", choices=["generated", "pkl"], default="generated")
    parser.add_argument("--pkl-dir", type=Path, default=Path("pkl_envs"))
    parser.add_argument("--pkl-count", type=int, default=32)
    parser.add_argument("--pkl-seed-start", type=int, default=1000)
    parser.add_argument("--pkl-overwrite", action="store_true")
    parser.add_argument("--prepare-pkls", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--debug-checks", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--mappo-checkpoint", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
            print(
                f"[summary] episodes={stats.episodes} success_rate={success_rate:.3f} "
                f"avg_steps={avg_steps:.1f} env_source={args.env_source}"
            )
            return

        run_train(args)
    finally:
        if args.tb_logger is not None:
            args.tb_logger.close()


if __name__ == "__main__":
    main()
