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
    def __init__(self, total: int, desc: str = "", width: int = 15, secondary_width: int = 10):
        self.total = max(1, int(total))
        self.desc = desc
        self.width = max(4, int(width))
        self.secondary_width = max(4, int(secondary_width))
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
        line = (
            f"{self.desc}: [{primary}] {self.n}/{self.total} {progress_pct:3d}% "
            f"[{secondary}] done={secondary_pct:3d}%"
            f"{secondary_text}{postfix}  [{rate:.2f} it/s {eta_text}]"
        )
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