from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_policy import RailEnvPolicy
from flatland.utils.seeding import np_random


class RandomPolicy(RailEnvPolicy[Any, Any, RailEnvActions]):
    def __init__(self, action_size: int = 5, seed: int = 42):
        super().__init__()
        self.action_size = int(action_size)
        self.np_random, _ = np_random(seed=seed)

    def act_many(self, handles: List[int], observations: List[Any], **kwargs) -> Dict[int, RailEnvActions]:
        return {h: self.act(observations[idx]) for idx, h in enumerate(handles)}

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions(int(self.np_random.choice(self.action_size)))
