"""Backward-compatible import shim for shared reward module."""

from rewards.outcome_reward import OutcomeBasedReward, build_outcome_reward

__all__ = ["OutcomeBasedReward", "build_outcome_reward"]
