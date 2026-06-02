from __future__ import annotations

from typing import Any

from policy.mappo.policy import MAPPOPolicy


class BCPolicy(MAPPOPolicy):
    """BC policy reusing MAPPO network structure and checkpoint formats."""

    def __init__(self, checkpoint_path: str | None = None, seed: int = 42):
        super().__init__(seed=seed, checkpoint_path=checkpoint_path)

    def act(self, observation: Any, **kwargs):
        return super().act(observation, **kwargs)
