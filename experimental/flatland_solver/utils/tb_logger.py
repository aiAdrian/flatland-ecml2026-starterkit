from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter


class TBLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    @staticmethod
    def build_run_dir(
        runs_root: Path,
        mode: str,
        policy: str,
        n_agents: int,
        width: int,
        height: int,
        n_cities: int,
        seed: int,
    ) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = (
            f"{ts}_{mode}_{policy}_"
            f"a{n_agents}_w{width}h{height}_c{n_cities}_s{seed}"
        )
        return Path(runs_root) / run_name

    def log_hparams_text(self, params_text: str):
        self.writer.add_text("run/params", params_text, global_step=0)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(str(tag), float(value), int(step))

    def log_eval_episode(
        self,
        episode_idx: int,
        done_rate: float,
        episode_len: float,
        total_reward: float,
        deadlock_rate: float,
    ):
        self.writer.add_scalar("eval/done_rate", float(done_rate), episode_idx)
        self.writer.add_scalar("eval/episode_len", float(episode_len), episode_idx)
        self.writer.add_scalar("eval/total_reward", float(total_reward), episode_idx)
        self.writer.add_scalar("eval/deadlock_rate", float(deadlock_rate), episode_idx)

    def log_eval_summary(
        self,
        episodes: int,
        success_rate: float,
        avg_steps: float,
        avg_reward: float,
        avg_deadlock_rate: float,
    ):
        self.writer.add_scalar("eval_summary/success_rate", float(success_rate), episodes)
        self.writer.add_scalar("eval_summary/avg_steps", float(avg_steps), episodes)
        self.writer.add_scalar("eval_summary/avg_reward", float(avg_reward), episodes)
        self.writer.add_scalar("eval_summary/avg_deadlock_rate", float(avg_deadlock_rate), episodes)

    def log_bc_epoch(self, epoch_idx: int, avg_loss: float, accuracy: float):
        self.writer.add_scalar("bc/loss", float(avg_loss), epoch_idx)
        self.writer.add_scalar("bc/accuracy", float(accuracy), epoch_idx)

    def log_mappo_epoch(
        self,
        epoch_idx: int,
        policy_loss: float,
        value_loss: float,
        entropy: Optional[float] = None,
        approx_kl: Optional[float] = None,
        ratio: Optional[float] = None,
        clip_frac: Optional[float] = None,
    ):
        self.writer.add_scalar("ppo/p_loss", float(policy_loss), epoch_idx)
        self.writer.add_scalar("ppo/v_loss", float(value_loss), epoch_idx)
        if entropy is not None:
            self.writer.add_scalar("ppo/entropy", float(entropy), epoch_idx)
        if approx_kl is not None:
            self.writer.add_scalar("ppo/approx_kl", float(approx_kl), epoch_idx)
        if ratio is not None:
            self.writer.add_scalar("ppo/ratio", float(ratio), epoch_idx)
        if clip_frac is not None:
            self.writer.add_scalar("ppo/clip_frac", float(clip_frac), epoch_idx)

    def log_mappo_episode(
        self,
        episode_idx: int,
        done_rate: float,
        episode_len: float,
        total_reward: float,
        n_agents: int,
        deadlock_rate: Optional[float] = None,
        done_rolling: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
    ):
        self.writer.add_scalar("ppo_episode/done_rate", float(done_rate), episode_idx)
        self.writer.add_scalar("ppo_episode/episode_len", float(episode_len), episode_idx)
        self.writer.add_scalar("ppo_episode/total_reward", float(total_reward), episode_idx)
        self.writer.add_scalar("ppo_episode/n_agents", float(n_agents), episode_idx)
        if deadlock_rate is not None:
            self.writer.add_scalar("ppo_episode/deadlock_rate", float(deadlock_rate), episode_idx)
        if done_rolling is not None:
            self.writer.add_scalar("ppo_episode/done_rolling", float(done_rolling), episode_idx)
        if policy_loss is not None:
            self.writer.add_scalar("ppo_episode/p_loss", float(policy_loss), episode_idx)
        if value_loss is not None:
            self.writer.add_scalar("ppo_episode/v_loss", float(value_loss), episode_idx)

    def log_mappo_summary(
        self,
        obs_dim: int,
        feature_mean: float,
        feature_std: float,
        mask_active_ratio: float,
        actions_total: int,
        action_hist: list[int],
    ):
        self.writer.add_scalar("ppo_summary/obs_dim", float(obs_dim), 0)
        self.writer.add_scalar("ppo_summary/feature_mean", float(feature_mean), 0)
        self.writer.add_scalar("ppo_summary/feature_std", float(feature_std), 0)
        self.writer.add_scalar("ppo_summary/mask_active_ratio", float(mask_active_ratio), 0)
        self.writer.add_scalar("ppo_summary/actions_total", float(actions_total), 0)
        for i, count in enumerate(action_hist):
            self.writer.add_scalar(f"ppo_summary/action_hist/a{i}", float(count), 0)

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


def format_args_text(args) -> str:
    items = sorted(vars(args).items(), key=lambda x: x[0])
    lines = [f"- {k}: {v}" for k, v in items]
    return "\n".join(lines)
