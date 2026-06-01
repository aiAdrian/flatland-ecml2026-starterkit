from pathlib import Path
import sys


def MAPPOObservationBuilder():
    repo_root = Path(__file__).resolve().parents[4]
    rl_dir = repo_root / "reinforcement-learning"
    rl_path = str(rl_dir)
    if rl_dir.exists() and rl_path not in sys.path:
        sys.path.insert(0, rl_path)
    from my_observation_builder import FastTreeObsBuilder

    return FastTreeObsBuilder(max_depth=3, with_action_mask=True)
