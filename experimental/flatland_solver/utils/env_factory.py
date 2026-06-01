from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


def _load_persist_util_classes():
    util_path = Path(__file__).resolve().parents[2] / "utils" / "flatland_env_persist_util.py"
    spec = importlib.util.spec_from_file_location("flatland_env_persist_util_shared", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load persister util from {util_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EnvSpec, module.FlatlandEnvPersistUtil


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


def ensure_pkl_dataset(
    cfg: SolverConfig,
    out_dir: str | Path,
    obs_builder_factory,
    count: int,
    seed_start: int,
    overwrite: bool = False,
) -> list[Path]:
    if count <= 0:
        return []
    EnvSpec, FlatlandEnvPersistUtil = _load_persist_util_classes()
    spec = EnvSpec(
        width=cfg.width,
        height=cfg.height,
        n_agents=cfg.n_agents,
        n_cities=cfg.n_cities,
        max_rails_between_cities=cfg.max_rails_between_cities,
        max_rail_pairs_in_city=cfg.max_rail_pairs_in_city,
    )
    seeds = range(seed_start, seed_start + count)
    return FlatlandEnvPersistUtil.generate_and_persist(
        out_dir=out_dir,
        seeds=seeds,
        spec=spec,
        obs_builder_factory=obs_builder_factory,
        overwrite=overwrite,
    )


def list_pkl_dataset(path: str | Path) -> list[Path]:
    _EnvSpec, FlatlandEnvPersistUtil = _load_persist_util_classes()
    return FlatlandEnvPersistUtil.list_pickles(path)


def build_env_from_pkl(pkl_path: str | Path, obs_builder) -> RailEnv:
    _EnvSpec, FlatlandEnvPersistUtil = _load_persist_util_classes()
    env = FlatlandEnvPersistUtil.load_env(pkl_path)
    env.obs_builder = obs_builder
    if hasattr(obs_builder, "set_env"):
        obs_builder.set_env(env)
    return env
