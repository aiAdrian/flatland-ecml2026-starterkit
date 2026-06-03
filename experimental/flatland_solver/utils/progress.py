from __future__ import annotations

from collections import deque
import time


class FallbackProgressBar:
    def __init__(self, total: int, desc: str = ""):
        self.total = max(0, int(total))
        self.desc = desc
        self.n = 0
        self._t0 = time.perf_counter()
        self._last_print = 0.0
        self._postfix = ""
        print(f"[{self.desc}] starting ({self.total} iterations)")

    def set_postfix_str(self, postfix: str):
        self._postfix = str(postfix)

    def update(self, n: int = 1):
        self.n += int(n)
        now = time.perf_counter()
        if now - self._last_print >= 2.0 or self.n >= self.total:
            elapsed = max(1e-9, now - self._t0)
            rate = self.n / elapsed
            remaining = max(0, self.total - self.n)
            eta = remaining / max(1e-9, rate)
            suffix = f"  {self._postfix}" if self._postfix else ""
            print(
                f"[{self.desc}] {self.n}/{self.total} "
                f"({rate:.2f} it/s, eta={eta:.0f}s){suffix}"
            )
            self._last_print = now

    def close(self):
        pass


class DualProgressBar:
    def __init__(self, 
                 total: int, 
                 desc: str = "",
                   width: int = 15, 
                   secondary_width: int = 10,
                   disable_bar_1: bool = True,
                   disable_bar_2: bool = False):
        self.total = max(1, int(total))
        self.desc = desc
        self.width = max(4, int(width))
        self.secondary_width = max(4, int(secondary_width))
        self.disable_bar_1 = disable_bar_1
        self.disable_bar_2 = disable_bar_2
        self.n = 0
        self._t0 = time.perf_counter()
        self._last_print = 0.0
        self._postfix = ""
        self._secondary_ratio = 0.0
        self._secondary_text = ""

    def set_postfix_str(self, postfix: str):
        self._postfix = str(postfix)

    def set_secondary(self, ratio: float, text: str = ""):
        self._secondary_ratio = max(0.0, min(1.0, float(ratio)))
        self._secondary_text = str(text)

    def update(self, n: int = 1):
        self.n = min(self.total, self.n + int(n))
        self._render()

    def close(self):
        self._render(force=True)

    def _bar(self, ratio: float, width: int) -> str:
        filled = int(round(max(0.0, min(1.0, ratio)) * width))
        return "█" * filled + "░" * max(0, width - filled)

    def _render(self, force: bool = False):
        now = time.perf_counter()
        if not force and now - self._last_print < 0.08:
            return
        progress_ratio = self.n / self.total
        elapsed = max(1e-9, now - self._t0)
        rate = self.n / elapsed
        remaining = max(0, self.total - self.n)
        eta = remaining / rate if rate > 0 else None
        primary = self._bar(progress_ratio, self.width)
        secondary = self._bar(self._secondary_ratio, self.secondary_width)
        progress_pct = int(round(progress_ratio * 100))
        secondary_pct = int(round(self._secondary_ratio * 100))
        postfix = f" {self._postfix}" if self._postfix else ""
        secondary_text = f" {self._secondary_text}" if self._secondary_text else ""
        eta_text = f"eta={eta:.0f}s" if eta is not None else "eta=--"
        
        parts = []
        parts.append(f"{self.desc}\t {self.n}/{self.total}\t")
        if not self.disable_bar_1:
            parts.append(f"[{primary}] {progress_pct:3d}%")

        if not self.disable_bar_2:
            parts.append(f"[{secondary}] done={secondary_pct:3d}%{secondary_text}")

        parts.append(f"[{rate:.2f} it/s {eta_text}]")
        line = " ".join(parts) + postfix
        print(line)
        self._last_print = now


class RollingDoneRatio:
    def __init__(self, window_size: int = 20):
        self.window_size = max(1, int(window_size))
        self._done = deque(maxlen=self.window_size)
        self._total = deque(maxlen=self.window_size)

    def is_ready(self) -> bool:
        return len(self._done) >= self.window_size

    def window_ratio(self) -> float:
        if not self.is_ready():
            return 0.0
        done_sum = sum(self._done)
        total_sum = sum(self._total)
        return done_sum / max(1, total_sum)

    def window_text(self) -> str:
        if not self.is_ready():
            return f"w{self.window_size}=----- r=nan"
        done_sum = sum(self._done)
        total_sum = sum(self._total)
        rate = done_sum / max(1, total_sum)
        return f"w{self.window_size}={done_sum:>3}/{total_sum:<3} r={rate:.2f}"

    def update(self, done_count: int, total_count: int) -> None:
        self._done.append(int(done_count))
        self._total.append(max(1, int(total_count)))

    def format_postfix(self) -> str:
        return self.window_text()

    def ratio(self) -> float:
        return self.window_ratio()


def format_fixed_metrics(**fields) -> str:
    parts = []
    for key, value in fields.items():
        if isinstance(value, int):
            parts.append(f"{key}={value:>4}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:>7.4f}")
        else:
            parts.append(f"{key}={str(value):<8}")
    return " ".join(parts)


def format_console_row(stage: str, policy: str, **fields) -> str:
    base = [f"stage={stage:<5}", f"policy={policy:<6}"]
    metrics = []
    for key, value in fields.items():
        if isinstance(value, int):
            metrics.append(f"{key}={value:>4}")
        elif isinstance(value, float):
            metrics.append(f"{key}={value:>7.4f}")
        else:
            metrics.append(f"{key}={str(value):<8}")
    return " ".join(base + metrics)


def make_progress_bar(total: int, desc: str = ""):
    return DualProgressBar(total=total, desc=desc, width=15, secondary_width=30)


def format_episode_compact(
    kind: str,
    episode: int,
    total: int,
    **metrics,
) -> str:
    """Compact single-line episode output (max ~210 chars).
    
    Examples:
      [TRAIN] ep=100/2000 done=85% rew=+5.2 KL=-0.001 H=0.32 acts[F:60% R:20% S:20%]
      [EVAL]  ep=10/20   done=100% dlk=0% rew=+8.5
      [BC]    ep=5/10    loss=0.123 acc=85%
      [REC]   ep=50/200  samples=156 done=60% dlk=20%
    """ 
    pct = 100.0 * float(episode) / max(1, int(total))
    prefix = f"[{kind:<6}] ep={episode:>4}/{total:<4} ({pct:>5.1f}%)"
    
    parts = []
    
    # done_rate
    if "done_rate" in metrics:
        v = float(metrics["done_rate"]) * 100.0
        parts.append(f"done={v:>3.0f}%")
    elif "done" in metrics:
        v = int(metrics["done"])
        n = int(metrics.get("n_agents", 1))
        p = 100.0 * v / max(1, n)
        parts.append(f"done={p:>3.0f}%")
    
    # deadlock_rate
    if "deadlock_rate" in metrics:
        v = float(metrics["deadlock_rate"]) * 100.0
        parts.append(f"dlk={v:>3.0f}%")
    
    # reward
    if "total_reward" in metrics:
        v = float(metrics["total_reward"])
        parts.append(f"rew={v:+.1f}")
    elif "rew" in metrics:
        v = float(metrics["rew"])
        parts.append(f"rew={v:+.1f}")
    elif "reward" in metrics:
        v = float(metrics["reward"])
        parts.append(f"rew={v:+.1f}")
    
    # loss (BC)
    if "loss" in metrics:
        v = float(metrics["loss"])
        parts.append(f"loss={v:.4f}")
    
    # accuracy (BC)
    if "acc" in metrics:
        v = float(metrics["acc"]) * 100.0
        parts.append(f"acc={v:.1f}%")
    
    # KL (PPO)
    if "approx_kl" in metrics:
        v = float(metrics["approx_kl"])
        parts.append(f"KL={v:+.6f}")
    
    # entropy (PPO)
    if "entropy" in metrics:
        v = float(metrics["entropy"])
        parts.append(f"H={v:.4f}")

    # epsilon exploration (optional)
    if "eps" in metrics:
        v = float(metrics["eps"])
        parts.append(f"eps={v:.3f}")
    
    # PPO stats (compact)
    ppo_parts = []
    if "n_minibatches" in metrics:
        ppo_parts.append(f"mb={int(metrics['n_minibatches'])}")
    if "p_loss" in metrics:
        ppo_parts.append(f"pl={float(metrics['p_loss']):+.4f}")
    if "v_loss" in metrics:
        ppo_parts.append(f"vl={float(metrics['v_loss']):+.4f}")
    if "clip_frac" in metrics:
        ppo_parts.append(f"cl={float(metrics['clip_frac']):.3f}")
    if ppo_parts:
        parts.append(f"ppo[{' '.join(ppo_parts)}]")
    
    # samples (recording)
    if "samples" in metrics:
        parts.append(f"samples={int(metrics['samples'])}")
    
    # steps (episode length)
    if "steps" in metrics:
        parts.append(f"s={int(metrics['steps'])}")
    
    # actions (compact histogram)
    if "acts" in metrics:
        acts_dict = metrics["acts"]
        if isinstance(acts_dict, dict):
            act_names = ["N", "L", "F", "R", "S"]
            act_pcts = []
            total_acts = sum(acts_dict.values())
            for name, idx in zip(act_names, range(5)):
                if total_acts > 0:
                    pct_val = 100.0 * acts_dict.get(idx, 0) / total_acts
                    if pct_val > 0:
                        act_pcts.append(f"{name}:{pct_val:.0f}%")
            if act_pcts:
                parts.append(f"acts[{' '.join(act_pcts)}]")
    
    line = prefix + " " + " ".join(parts)
    
    # Truncate if over 210 chars
    if len(line) > 210:
        line = line[:207] + "..."
    
    return line


def format_ppo_update(
    epoch: int,
    total_epochs: int,
    batch_num: int,
    n_samples: int,
    **stats,
) -> str:
    """Compact PPO update stats (one line, max ~210 chars).
    
    Example:
      [PPO-UPD] epoch=1/5 batch=16 samples=256 KL=+0.003 ratio=0.998 p_loss=-0.008 v_loss=0.002 clip=0.05 ent=0.32
    """
    prefix = f"[PPO-UPD] epoch={epoch}/{total_epochs} batch={batch_num} samples={n_samples}"
    parts = []
    
    if "approx_kl" in stats:
        parts.append(f"KL={float(stats['approx_kl']):+.6f}")
    if "ratio" in stats:
        parts.append(f"ratio={float(stats['ratio']):.5f}")
    if "p_loss" in stats:
        parts.append(f"p_loss={float(stats['p_loss']):+.6f}")
    if "v_loss" in stats:
        parts.append(f"v_loss={float(stats['v_loss']):+.6f}")
    if "clip_frac" in stats:
        parts.append(f"clip={float(stats['clip_frac']):.3f}")
    if "entropy" in stats:
        parts.append(f"ent={float(stats['entropy']):.4f}")
    if "early_stop_skips" in stats:
        parts.append(f"skip={float(stats['early_stop_skips']):.3f}")
    
    line = prefix + " " + " ".join(parts)
    if len(line) > 210:
        line = line[:207] + "..."
    return line