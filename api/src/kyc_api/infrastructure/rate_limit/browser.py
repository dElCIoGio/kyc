"""Bounded fixed-window rate limiting for active browser credentials."""

from __future__ import annotations

from threading import RLock
from time import monotonic


class BrowserCredentialRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("browser rate-limit values must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._buckets: dict[str, tuple[float, int]] = {}
        self._lock = RLock()

    def allow(self, credential_id: str) -> int | None:
        with self._lock:
            now = monotonic()
            started, count = self._buckets.get(credential_id, (now, 0))
            elapsed = now - started
            if elapsed >= self._window_seconds:
                started, count, elapsed = now, 0, 0.0
            if count >= self._max_requests:
                return max(1, int(self._window_seconds - elapsed) + 1)
            self._buckets[credential_id] = (started, count + 1)
            self._prune_locked(now)
            return None

    def _prune_locked(self, now: float) -> None:
        if len(self._buckets) <= 256:
            return
        self._buckets = {
            key: bucket
            for key, bucket in self._buckets.items()
            if now - bucket[0] < self._window_seconds
        }
