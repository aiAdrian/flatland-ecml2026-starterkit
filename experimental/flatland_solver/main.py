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
from utils.env_factory import SolverConfig, build_env


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
        return BCPolicy(checkpoint_path=str(args.bc_checkpoint)), BCObservationBuilder()
    if args.policy == "mappo":
        return MAPPOPolicy(seed=args.seed, checkpoint_path=str(args.mappo_checkpoint)), MAPPOObservationBuilder()
    raise ValueError(f"Unsupported policy: {args.policy}")


def run_eval(args) -> EvalStats:
    policy, obs_builder = make_policy_and_observation(args)
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

    renderer = RenderTool(env, gl="PGL") if args.rendering else None

    total_done_agents = 0
    total_agents = 0
    total_steps = 0

    for episode in range(args.episodes):
        observations, info = env.reset(random_seed=args.seed + episode)
        del info
        done = {"__all__": False}
        steps = 0

        while not done.get("__all__", False) and steps < args.max_episode_steps:
            handles = list(range(env.get_num_agents()))
            obs_batch = [observations[h] for h in handles]
            actions = policy.act_many(handles, obs_batch)
            observations, rewards, done, info = env.step(normalize_actions(actions))
            del rewards, info
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

        episode_done = sum(1 for a in env.agents if a.state == TrainState.DONE)
        total_done_agents += episode_done
        total_agents += env.get_num_agents()
        total_steps += steps

        print(
            f"[eval] episode={episode + 1}/{args.episodes} policy={args.policy} "
            f"steps={steps} done={episode_done}/{env.get_num_agents()}"
        )

    if renderer is not None:
        renderer.close_window()

    return EvalStats(
        episodes=args.episodes,
        total_agents=total_agents,
        done_agents=total_done_agents,
        total_steps=total_steps,
    )


def run_train(args) -> None:
    if args.policy == "bc":
        args.obs_builder = BCObservationBuilder()
        train_bc(args, checkpoint_path=args.bc_checkpoint)
        return
    if args.policy == "mappo":
        args.obs_builder = MAPPOObservationBuilder()
        train_mappo(args, checkpoint_path=args.mappo_checkpoint)
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
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument("--mappo-checkpoint", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.bc_checkpoint is None:
        args.bc_checkpoint = args.checkpoint_dir / "bc.pt"
    if args.mappo_checkpoint is None:
        args.mappo_checkpoint = args.checkpoint_dir / "mappo.pt"

    if args.mode == "eval":
        stats = run_eval(args)
        success_rate = stats.done_agents / max(1, stats.total_agents)
        avg_steps = stats.total_steps / max(1, stats.episodes)
        print(
            f"[summary] episodes={stats.episodes} success_rate={success_rate:.3f} avg_steps={avg_steps:.1f}"
        )
        return

    run_train(args)


if __name__ == "__main__":
    main()
