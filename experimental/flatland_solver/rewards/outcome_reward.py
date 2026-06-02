from __future__ import annotations

from typing import Any

import numpy as np

from flatland.envs.step_utils.states import TrainState

from policy.dla.policy import DLAPolicy

try:
    from observations.decision_point_utils import DecisionPointUtils
except Exception:
    DecisionPointUtils = None


class OutcomeBasedReward:
    def __init__(
        self,
        step_penalty: float = 1.0,
        deadlock_penalty: float = 5.0,
        done_bonus: float = 50.0,
        time_saved_factor: float = 2.0,
        all_done_bonus: float = 100.0,
        fail_penalty: float = 200.0,
        match_bonus: float = 0.5,
        reward_scale: float = 0.01,
        seed: int = 42,
    ):
        self.step_penalty = float(step_penalty)
        self.deadlock_penalty = float(deadlock_penalty)
        self.done_bonus = float(done_bonus)
        self.time_saved_factor = float(time_saved_factor)
        self.all_done_bonus = float(all_done_bonus)
        self.fail_penalty = float(fail_penalty)
        self.match_bonus = float(match_bonus)
        self.reward_scale = float(reward_scale)

        self._dla = DLAPolicy(seed=seed)
        self._last_episode_step = -1
        self._terminal_given = False
        self._rewarded_done: dict[int, bool] = {}

    @staticmethod
    def description() -> str:
        return "[reward] Using OutcomeBasedReward (step=-1, deadlock=-5, done=+50+saved, all_done=+100, fail=-200, scale=0.01)"

    def _build_agent_map(self, env) -> np.ndarray:
        raw = env
        agent_map = np.full((raw.height, raw.width), -1, dtype=np.int32)
        for a in raw.agents:
            if a.position is not None:
                agent_map[a.position] = int(a.handle)
        return agent_map

    def _is_decision_point(self, raw_env, agent) -> bool:
        if DecisionPointUtils is None:
            return False
        if agent.position is None or agent.state == TrainState.DONE:
            return False
        try:
            ctype = DecisionPointUtils.classify_cell_type(agent, raw_env)
            return ctype in ("SWITCH", "MERGING", "PRE_M")
        except Exception:
            return False

    def _reset_episode(self, env) -> None:
        self._terminal_given = False
        self._rewarded_done = {i: False for i in range(len(env.agents))}
        self._dla.reset_env(env)

    def __call__(self, rewards, done, info, env, actions: dict[int, int] | None = None):
        del info
        raw = env
        cur_step = int(getattr(raw, "_elapsed_steps", 0))
        max_steps = int(getattr(raw, "_max_episode_steps", 1))

        if self._last_episode_step < 0 or cur_step <= 1 or cur_step < self._last_episode_step:
            self._reset_episode(raw)
        self._last_episode_step = cur_step

        handles = [int(a.handle) for a in raw.agents]
        shaped = {h: 0.0 for h in handles}
        agent_map = self._build_agent_map(raw)

        dla_actions = None
        if self.match_bonus > 0.0 and actions is not None:
            try:
                dla_actions = self._dla.act_many(handles, [raw for _ in handles])
            except Exception:
                dla_actions = None

        def _action_to_int(a) -> int:
            v = getattr(a, "value", a)
            return int(v)

        for h in handles:
            agent = raw.agents[h]

            if agent.state == TrainState.DONE:
                if not self._rewarded_done.get(h, False):
                    elapsed = cur_step
                    time_saved = max(0, max_steps - elapsed)
                    shaped[h] += self.done_bonus + self.time_saved_factor * float(time_saved)
                    self._rewarded_done[h] = True
                continue

            if agent.state <= TrainState.WAITING:
                continue

            shaped[h] -= self.step_penalty

            if DecisionPointUtils is not None and agent.position is not None:
                try:
                    if DecisionPointUtils.is_local_deadlock(raw, agent, agent_map):
                        shaped[h] -= self.deadlock_penalty
                except Exception:
                    pass

            if (
                self.match_bonus > 0.0
                and actions is not None
                and dla_actions is not None
                and self._is_decision_point(raw, agent)
                and h in actions
                and h in dla_actions
                and _action_to_int(actions[h]) == _action_to_int(dla_actions[h])
            ):
                shaped[h] += self.match_bonus

        all_term = bool(done.get("__all__", False)) if isinstance(done, dict) else bool(done)
        is_final_step = cur_step >= max_steps - 1
        if (all_term or is_final_step) and not self._terminal_given:
            self._terminal_given = True
            n_agents = len(raw.agents)
            done_count = sum(a.state == TrainState.DONE for a in raw.agents)

            for ag in raw.agents:
                if ag.state != TrainState.DONE:
                    shaped[int(ag.handle)] -= self.fail_penalty

            if done_count == n_agents:
                for ag in raw.agents:
                    shaped[int(ag.handle)] += self.all_done_bonus

        if self.reward_scale != 1.0:
            for h in shaped:
                shaped[h] *= self.reward_scale

        return shaped


def build_outcome_reward(args: Any) -> OutcomeBasedReward | None:
    if bool(getattr(args, "disable_outcome_reward", False)):
        return None
    return OutcomeBasedReward(seed=int(getattr(args, "seed", 42)))
