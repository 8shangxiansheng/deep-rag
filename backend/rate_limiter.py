from collections import defaultdict, deque
from time import monotonic


class InMemoryRateLimiter:
    """Simple sliding-window in-memory rate limiter."""

    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)

    def allow(self, scope: str, key: str, max_requests: int, window_seconds: int) -> bool:
        bucket = self._events[(scope, key)]
        now = monotonic()
        cutoff = now - window_seconds

        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= max_requests:
            return False

        bucket.append(now)
        return True
