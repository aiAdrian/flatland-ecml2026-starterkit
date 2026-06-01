from dataclasses import dataclass

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


@dataclass(frozen=True)
class SolverConfig:
    width: int = 35
    height: int = 35
    n_agents: int = 5
    n_cities: int = 3
    max_rails_between_cities: int = 2
    max_rail_pairs_in_city: int = 2
    seed: int = 42


def build_env(cfg: SolverConfig, obs_builder) -> RailEnv:
    return RailEnv(
        width=cfg.width,
        height=cfg.height,
        rail_generator=sparse_rail_generator(
            max_num_cities=cfg.n_cities,
            seed=cfg.seed,
            max_rails_between_cities=cfg.max_rails_between_cities,
            max_rail_pairs_in_city=cfg.max_rail_pairs_in_city,
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=cfg.n_agents,
        obs_builder_object=obs_builder,
    )
