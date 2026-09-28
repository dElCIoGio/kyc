from __future__ import annotations

from threading import RLock
from time import monotonic


class ApiKeyRateLimiter:
    """A fixed-window limiter for the current single valid-key deployment."""

    def __init__(self, *, max_requests: int, window_seconds: int) -> None:
        if max_requests <= 0 or window_seconds <= 0:
            raise ValueError("Rate-limit values must be positive")
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._window_started = monotonic()
        self._requests = 0
        self._lock = RLock()

    def allow(self) -> int | None:
        """Return retry seconds when limited, otherwise allow the request."""
        with self._lock:
            now = monotonic()
            elapsed = now - self._window_started
            if elapsed >= self._window_seconds:
                self._window_started = now
                self._requests = 0
                elapsed = 0.0
            if self._requests >= self._max_requests:
                remaining = self._window_seconds - elapsed
                return max(1, int(remaining) + 1)
            self._requests += 1
            return None
