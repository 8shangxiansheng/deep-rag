import unittest

from backend.rate_limiter import InMemoryRateLimiter


class InMemoryRateLimiterTest(unittest.TestCase):
    def test_block_after_limit(self) -> None:
        limiter = InMemoryRateLimiter()
        self.assertTrue(limiter.allow("chat", "127.0.0.1", max_requests=2, window_seconds=60))
        self.assertTrue(limiter.allow("chat", "127.0.0.1", max_requests=2, window_seconds=60))
        self.assertFalse(limiter.allow("chat", "127.0.0.1", max_requests=2, window_seconds=60))

    def test_scope_and_key_are_isolated(self) -> None:
        limiter = InMemoryRateLimiter()
        self.assertTrue(limiter.allow("chat", "client-a", max_requests=1, window_seconds=60))
        self.assertFalse(limiter.allow("chat", "client-a", max_requests=1, window_seconds=60))

        self.assertTrue(limiter.allow("chat", "client-b", max_requests=1, window_seconds=60))
        self.assertTrue(limiter.allow("retrieve", "client-a", max_requests=1, window_seconds=60))


if __name__ == "__main__":
    unittest.main()
