"""Send-retry classification tests (G1 + G6)."""

import unittest

from nomorals.core.errors import RateLimited, TimeoutError_
from nomorals.core.retry import BackoffPolicy, JitterStyle, RetryBudget
from nomorals.social.chat.base import SendResult
from nomorals.social.chat.send_retry import (
    classify_send_error,
    get_retry_budget,
    reset_retry_budgets,
    send_with_retry,
)


def _fail(error: str) -> SendResult:
    return SendResult(ok=False, platform="test", error=error)


def _ok() -> SendResult:
    return SendResult(ok=True, platform="test", message_id="m1")


_NO_SLEEP = BackoffPolicy(base=0.0, factor=2.0, cap=0.0,
                          jitter=False, jitter_style=JitterStyle.NONE)


class TestClassify(unittest.TestCase):
    def test_transient_patterns(self) -> None:
        for err in ("429 Too Many Requests", "503 Service Unavailable",
                    "500 Internal Server Error", "timeout",
                    "connection reset by peer", "RateLimited: slow down",
                    "FloodWaitError: 30s"):
            self.assertEqual(classify_send_error(err), "transient", err)

    def test_terminal_patterns(self) -> None:
        for err in ("400 Bad Request", "chat not found",
                    "bot was blocked by the user", "403 Forbidden",
                    "invalid token", "twilio not configured",
                    "media file not found"):
            self.assertEqual(classify_send_error(err), "terminal", err)

    def test_unknown_patterns(self) -> None:
        self.assertEqual(classify_send_error("something weird"), "unknown")
        self.assertEqual(classify_send_error(""), "unknown")
        self.assertEqual(classify_send_error(None), "unknown")

    def test_exceptions(self) -> None:
        self.assertEqual(classify_send_error(TimeoutError("boom")), "transient")
        self.assertEqual(classify_send_error(ConnectionError("reset")), "transient")
        self.assertEqual(classify_send_error(RateLimited("429")), "transient")
        self.assertEqual(classify_send_error(TimeoutError_("timed out")), "transient")
        self.assertEqual(classify_send_error(ValueError("bad request")), "terminal")


class TestSendWithRetry(unittest.TestCase):
    def setUp(self) -> None:
        reset_retry_budgets()

    def tearDown(self) -> None:
        reset_retry_budgets()

    def test_transient_retried_then_succeeds(self) -> None:
        calls = {"n": 0}

        def flaky() -> SendResult:
            calls["n"] += 1
            if calls["n"] < 3:
                return _fail("503 Service Unavailable")
            return _ok()

        result, attempts = send_with_retry(flaky, platform="t1", policy=_NO_SLEEP)
        self.assertTrue(result.ok)
        self.assertEqual(attempts, 3)

    def test_terminal_never_retried(self) -> None:
        calls = {"n": 0}

        def doomed() -> SendResult:
            calls["n"] += 1
            return _fail("400 chat not found")

        result, attempts = send_with_retry(doomed, platform="t1", policy=_NO_SLEEP)
        self.assertFalse(result.ok)
        self.assertEqual(attempts, 1)
        self.assertEqual(calls["n"], 1)

    def test_unknown_retried_twice_then_terminal(self) -> None:
        calls = {"n": 0}

        def weird() -> SendResult:
            calls["n"] += 1
            return _fail("something weird happened")

        result, attempts = send_with_retry(weird, platform="t1", policy=_NO_SLEEP)
        self.assertFalse(result.ok)
        self.assertEqual(attempts, 3)
        self.assertEqual(calls["n"], 3)

    def test_transient_exhaustion_stops(self) -> None:
        calls = {"n": 0}

        def always_down() -> SendResult:
            calls["n"] += 1
            return _fail("503 Service Unavailable")

        result, attempts = send_with_retry(
            always_down, platform="t1", policy=_NO_SLEEP,
            max_transient_attempts=5)
        self.assertFalse(result.ok)
        self.assertEqual(attempts, 5)

    def test_backoff_schedule_uses_policy(self) -> None:
        sleeps: list[float] = []
        calls = {"n": 0}
        policy = BackoffPolicy(base=0.5, factor=2.0, cap=30.0,
                               jitter=False, jitter_style=JitterStyle.NONE)

        def flaky() -> SendResult:
            calls["n"] += 1
            if calls["n"] < 3:
                return _fail("503 Service Unavailable")
            return _ok()

        send_with_retry(flaky, platform="t1", policy=policy,
                        sleep=sleeps.append)
        self.assertEqual(sleeps, [0.5, 1.0])

    def test_budget_spent_stops_retries(self) -> None:
        budget = RetryBudget(retry_ratio=0.1, min_tokens=0.0)
        calls = {"n": 0}

        def flaky() -> SendResult:
            calls["n"] += 1
            return _fail("503 Service Unavailable")

        result, attempts = send_with_retry(
            flaky, platform="t1", policy=_NO_SLEEP, budget=budget)
        self.assertFalse(result.ok)
        self.assertEqual(attempts, 1)

    def test_exception_from_send_fn_is_classified(self) -> None:
        def raising() -> SendResult:
            raise TimeoutError("boom")

        result, attempts = send_with_retry(raising, platform="t1",
                                           policy=_NO_SLEEP,
                                           max_transient_attempts=2)
        self.assertFalse(result.ok)
        self.assertEqual(attempts, 2)

    def test_budgets_are_per_platform(self) -> None:
        self.assertIsNot(get_retry_budget("alpha"), get_retry_budget("beta"))
        self.assertIs(get_retry_budget("alpha"), get_retry_budget("alpha"))


if __name__ == "__main__":
    unittest.main()
