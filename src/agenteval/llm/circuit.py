"""熔断器：LLM 连续失败触发 OPEN，冷却后 HALF_OPEN 试探，成功即恢复。"""

from __future__ import annotations

import threading
import time


class CircuitBreaker:
    """状态机: CLOSED(正常) -> OPEN(熔断,直接兜底) -> HALF_OPEN(试探) -> CLOSED/OPEN"""

    def __init__(self, fail_threshold: int = 5, cooldown_sec: float = 60.0):
        self.fail_threshold = fail_threshold
        self.cooldown_sec = cooldown_sec
        self._failures = 0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "OPEN" and time.monotonic() - self._opened_at >= self.cooldown_sec:
                self._state = "HALF_OPEN"
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == "OPEN"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.fail_threshold:
                self._state = "OPEN"
                self._opened_at = time.monotonic()

    def reset(self) -> None:
        with self._lock:
            self._failures = 0
            self._state = "CLOSED"

    def __repr__(self) -> str:  # pragma: no cover
        return f"CircuitBreaker(state={self.state}, failures={self._failures}, threshold={self.fail_threshold})"
