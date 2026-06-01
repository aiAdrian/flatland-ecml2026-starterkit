from pathlib import Path
import sys
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy


def _ensure_baseline_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    baselines_root = repo_root / "experimental" / "flatland-baselines"
    baselines_path = str(baselines_root)
    if baselines_root.exists() and baselines_path not in sys.path:
        sys.path.insert(0, baselines_path)


_ensure_baseline_on_path()

from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import (  # noqa: E402
    DeadLockAvoidancePolicy,
)


class DLAPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    def __init__(self, seed: int = 42):
        super().__init__()
        self._delegate = DeadLockAvoidancePolicy(
            min_free_cell=1,
            show_debug_plot=False,
            count_num_opp_agents_towards_min_free_cell=True,
            use_switches_heuristic=True,
            use_entering_prevention=False,
            use_alternative_at_first_intermediate_and_then_always_first_strategy=3,
            k_shortest_path_cutoff=500,
            seed=seed,
            verbose=False,
        )

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        return self._delegate.act_many(handles, observations, **kwargs)

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        raise NotImplementedError("DLAPolicy is intended for act_many(handles, observations).")
