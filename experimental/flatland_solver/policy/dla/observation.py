from flatland.core.env_observation_builder import AgentHandle, ObservationBuilder, ObservationType
from flatland.envs.rail_env import RailEnv


class DLAFullEnvObservation(ObservationBuilder[RailEnv, RailEnv]):
    def get(self, handle: AgentHandle = 0) -> ObservationType:
        return self.env

    def reset(self):
        return None

    def set_env(self, env):
        self.env = env
