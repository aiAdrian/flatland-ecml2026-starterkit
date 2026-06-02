from __future__ import annotations

from typing import Dict, List


class RolloutBuffer:
    """Legacy-compatible rollout buffer container."""

    def __init__(self):
        self.data: Dict[int, List[tuple]] = {}

    def push(self, handle: int, transition: tuple):
        self.data.setdefault(int(handle), []).append(transition)

    def get(self, handle: int) -> List[tuple]:
        return self.data.get(int(handle), [])

    def reset(self):
        self.data = {}

    def all_transitions(self) -> List[tuple]:
        out = []
        for ts in self.data.values():
            out.extend(ts)
        return out

    def __len__(self):
        return sum(len(v) for v in self.data.values())
