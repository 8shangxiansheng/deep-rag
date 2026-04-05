import importlib.util
import unittest


HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(HAS_FASTAPI, "fastapi is required for API error contract test")
class ErrorContractTest(unittest.TestCase):
    def test_api_error_has_code_and_message(self) -> None:
        from backend.main import _api_error

        exc = _api_error(429, "RATE_LIMIT_EXCEEDED", "Too many requests")
        self.assertEqual(exc.status_code, 429)
        self.assertIsInstance(exc.detail, dict)
        self.assertEqual(exc.detail.get("code"), "RATE_LIMIT_EXCEEDED")
        self.assertEqual(exc.detail.get("message"), "Too many requests")


if __name__ == "__main__":
    unittest.main()
