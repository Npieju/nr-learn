from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Callable


Logger = Callable[[str], None]


def _default_logger(message: str) -> None:
    print(message, flush=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--"

    total_seconds = max(int(seconds), 0)
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def render_bar(current: int, total: int, width: int = 24) -> str:
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    ratio = current / total
    filled = min(int(round(ratio * width)), width)
    return "#" * filled + "-" * (width - filled)


@dataclass
class ProgressBar:
    total: int
    prefix: str
    logger: Logger | None = None
    width: int = 24
    min_interval_sec: float = 1.0
    current: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    _last_emitted_at: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 1)
        if self.logger is None:
            self.logger = _default_logger

    def start(self, message: str | None = None) -> None:
        self._emit(message=message, force=True)

    def update(
        self,
        step: int = 1,
        *,
        current: int | None = None,
        message: str | None = None,
        force: bool = False,
    ) -> None:
        if current is None:
            self.current = min(self.total, self.current + int(step))
        else:
            self.current = min(self.total, max(int(current), 0))
        self._emit(message=message, force=force or self.current >= self.total)

    def complete(self, message: str | None = None) -> None:
        self.current = self.total
        self._emit(message=message, force=True)

    def _emit(self, message: str | None, force: bool) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_emitted_at) < max(float(self.min_interval_sec), 0.0):
            return

        self._last_emitted_at = now
        ratio = self.current / self.total
        elapsed = now - self.started_at
        rate = (self.current / elapsed) if elapsed > 0 else 0.0
        remaining = self.total - self.current
        eta = (remaining / rate) if rate > 0 else None
        bar = render_bar(self.current, self.total, width=self.width)
        suffix = f" {message}" if message else ""
        assert self.logger is not None
        self.logger(
            f"{self.prefix} [{bar}] {self.current}/{self.total} ({ratio:.0%}) "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}{suffix}"
        )


class Heartbeat:
    def __init__(
        self,
        prefix: str,
        label: str,
        *,
        logger: Logger | None = None,
        interval_sec: float = 10.0,
    ) -> None:
        self.prefix = prefix
        self.label = label
        self.logger = logger or _default_logger
        self.interval_sec = max(float(interval_sec), 0.0)
        self.started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Heartbeat":
        self.started_at = time.perf_counter()
        self.logger(f"{self.prefix} {self.label} started")
        if self.interval_sec > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec, 0.1))

        elapsed = time.perf_counter() - self.started_at
        status = "done" if exc is None else "failed"
        self.logger(f"{self.prefix} {self.label} {status} in {format_duration(elapsed)}")

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            elapsed = time.perf_counter() - self.started_at
            self.logger(f"{self.prefix} {self.label} running... elapsed={format_duration(elapsed)}")