from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy

from policy.dla.vendor.deadlock_avoidance_policy import DeadLockAvoidancePolicy


class DLAPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed = int(seed)
        self.calls_total = 0
        self.calls_episode = 0

        self._delegate_kwargs = dict(
            min_free_cell=1,
            show_debug_plot=False,
            count_num_opp_agents_towards_min_free_cell=True,
            use_switches_heuristic=True,
            use_entering_prevention=False,
            use_alternative_at_first_intermediate_and_then_always_first_strategy=3,
            k_shortest_path_cutoff=500,
            seed=self.seed,
            verbose=False,
        )
        self._delegate = DeadLockAvoidancePolicy(**self._delegate_kwargs)

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        self.calls_total += len(handles)
        self.calls_episode += len(handles)
        return self._delegate.act_many(handles, observations, **kwargs)

    def reset_env(self, env) -> None:
        # Vendor DLA policy does not expose a robust reset() for cross-env reuse.
        # Re-create delegate to guarantee clean state for each new env/episode.
        self._delegate = DeadLockAvoidancePolicy(**self._delegate_kwargs)
        self.calls_episode = 0

    def get_debug_stats(self) -> Dict[str, int]:
        return {"calls_total": self.calls_total, "calls_episode": self.calls_episode}

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        raise NotImplementedError("DLAPolicy is intended for act_many(handles, observations).")
