from collections import defaultdict
from threading import Lock
import time


class RateLimiter:
    """
    In-memory sliding window rate limiter.
    Per-IP, per-endpoint.
    """

    def __init__(self):
        self._windows: dict = defaultdict(list)
        self._lock = Lock()

    def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> bool:
        now = time.time()
        with self._lock:
            self._windows[key] = [
                t for t in self._windows[key]
                if now - t < window_seconds
            ]
            if len(self._windows[key]) >= limit:
                return False
            self._windows[key].append(now)
            return True


rate_limiter = RateLimiter()
