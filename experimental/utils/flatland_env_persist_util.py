from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator


@dataclass(frozen=True)
class EnvSpec:
    width: int = 35
    height: int = 35
    n_agents: int = 5
    n_cities: int = 3
    max_rails_between_cities: int = 2
    max_rail_pairs_in_city: int = 2


class FlatlandEnvPersistUtil:
    """Static helper methods for generating, saving, and loading Flatland RailEnv pickles."""

    @staticmethod
    def build_filename(path: str | Path, spec: EnvSpec, seed: int) -> Path:
        base = Path(path)
        return base / f"{spec.width:04d}x{spec.height:04d}x{spec.n_agents:04d}_{seed:09d}.pkl"

    @staticmethod
    def list_pickles(path: str | Path) -> List[Path]:
        root = Path(path)
        if not root.exists():
            return []
        return sorted(root.glob("*.pkl"), key=lambda p: p.stat().st_mtime)

    @staticmethod
    def save_env(env: RailEnv, file_path: str | Path) -> Path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        RailEnvPersister.save(env, filename=str(file_path))
        return file_path

    @staticmethod
    def load_env(file_path: str | Path) -> RailEnv:
        # Start with a minimal valid env shell, then overwrite by persister load.
        env = RailEnv(
            width=10,
            height=10,
            rail_generator=sparse_rail_generator(max_num_cities=2),
            line_generator=sparse_line_generator(),
            number_of_agents=1,
        )
        RailEnvPersister.load(env, str(file_path))
        return env

    @staticmethod
    def make_env(
        spec: EnvSpec,
        seed: int,
        obs_builder_factory: Callable[[], ObservationBuilder],
    ) -> RailEnv:
        env = RailEnv(
            width=spec.width,
            height=spec.height,
            rail_generator=sparse_rail_generator(
                max_num_cities=spec.n_cities,
                seed=seed,
                max_rails_between_cities=spec.max_rails_between_cities,
                max_rail_pairs_in_city=spec.max_rail_pairs_in_city,
            ),
            line_generator=sparse_line_generator(),
            number_of_agents=spec.n_agents,
            obs_builder_object=obs_builder_factory(),
        )
        env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=seed)
        return env

    @staticmethod
    def generate_and_persist(
        out_dir: str | Path,
        seeds: Iterable[int],
        spec: EnvSpec,
        obs_builder_factory: Callable[[], ObservationBuilder],
        overwrite: bool = False,
    ) -> List[Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        written: List[Path] = []
        for seed in seeds:
            file_path = FlatlandEnvPersistUtil.build_filename(out_dir, spec, seed)
            if file_path.exists() and not overwrite:
                continue
            env = FlatlandEnvPersistUtil.make_env(spec=spec, seed=seed, obs_builder_factory=obs_builder_factory)
            FlatlandEnvPersistUtil.save_env(env, file_path)
            written.append(file_path)
        return written
