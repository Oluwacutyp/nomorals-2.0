"""L4 — social: parallel fan-out, failure isolation, rate limits, scheduling.

No network. Every test drives a fake adapter, so the assertions are about the
manager's behaviour — ordering, isolation, rate limiting, persistence — rather
than about a live platform's mood.

The parallelism test measures wall time. Posting to four platforms sequentially
takes four sleeps; fanned out it must take about one.
"""

from __future__ import annotations

import tempfile
import time
import unittest

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.core.errors import NotFound, ValidationError
from nomorals.social import Account, PlatformAdapter, PostResult, PostStatus, SocialError, SocialManager


class FakeAdapter(PlatformAdapter):
    """Records calls; optionally slow or failing."""

    def __init__(self, name: str, *, delay: float = 0.0, fail: bool = False, max_chars: int = 500):
        self.name = name
        self.max_chars = max_chars
        self.delay = delay
        self.fail = fail
        self.calls: list[str] = []

    def post(self, account, content, **kwargs):
        self.calls.append(content)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            return PostResult(platform=self.name, ok=False, error="synthetic platform error")
        return PostResult(
            platform=self.name, ok=True, external_id=f"{self.name}-1",
            url=f"https://{self.name}.test/1", status_code=200,
        )


def _context(home: str):
    ctx = build_context(Settings(home=home))
    ctx.__enter__()
    return ctx


class AccountTests(unittest.TestCase):
    def test_platform_and_handle_are_required(self):
        with self.assertRaises(ValidationError):
            Account(platform="", handle="x")
        with self.assertRaises(ValidationError):
            Account(platform="mastodon", handle="")

    def test_credential_reference_is_resolved_lazily(self):
        import os

        account = Account(platform="mastodon", handle="@a@b.test", credentials="env:NM_TEST_TOKEN")
        os.environ.pop("NM_TEST_TOKEN", None)
        self.assertEqual(account.resolve_token(), "")
        os.environ["NM_TEST_TOKEN"] = "secret"
        try:
            self.assertEqual(account.resolve_token(), "secret")
        finally:
            os.environ.pop("NM_TEST_TOKEN", None)

    def test_the_resolved_secret_never_reaches_the_row(self):
        """This database is backed up to a git repo. Tokens must not be in it."""
        import os

        os.environ["NM_TEST_TOKEN"] = "super-secret-token"
        try:
            account = Account(
                platform="mastodon", handle="@a@b.test", credentials="env:NM_TEST_TOKEN"
            )
            account.resolve_token()
            serialized = str(account.to_row())
            self.assertNotIn("super-secret-token", serialized)
            self.assertIn("env:NM_TEST_TOKEN", serialized)
        finally:
            os.environ.pop("NM_TEST_TOKEN", None)

    def test_limits_have_conservative_defaults(self):
        account = Account(platform="mastodon", handle="@a@b.test")
        self.assertEqual(account.posts_per_day, 50)
        self.assertEqual(account.min_interval, 30.0)
        self.assertEqual(account.max_chars, 500)

    def test_row_round_trip(self):
        original = Account(
            id="a1", platform="bluesky", handle="me.bsky.social",
            limits={"max_chars": 300}, metadata={"note": "x"},
        )
        restored = Account.from_row(original.to_row())
        self.assertEqual(restored.platform, "bluesky")
        self.assertEqual(restored.limits["max_chars"], 300)
        self.assertEqual(restored.metadata["note"], "x")

    def test_from_row_tolerates_undecoded_json(self):
        row = Account(id="a", platform="p", handle="h").to_row()
        row["limits"] = '{"max_chars": 280}'
        row["metadata"] = "garbage"
        restored = Account.from_row(row)
        self.assertEqual(restored.limits["max_chars"], 280)
        self.assertEqual(restored.metadata, {})


class AdapterValidationTests(unittest.TestCase):
    def test_empty_content_is_rejected(self):
        with self.assertRaises(ValidationError):
            FakeAdapter("x").validate("   ")

    def test_overlong_content_is_rejected_before_spending_a_request(self):
        adapter = FakeAdapter("x", max_chars=10)
        with self.assertRaises(ValidationError):
            adapter.validate("this is far too long for the limit")

    def test_content_is_trimmed(self):
        self.assertEqual(FakeAdapter("x").validate("  hi  "), "hi")

    def test_health_defaults_to_having_a_token(self):
        self.assertFalse(FakeAdapter("x").health(Account(platform="p", handle="h")))
        self.assertTrue(
            FakeAdapter("x").health(Account(platform="p", handle="h", credentials="lit"))
        )


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-social-")
        self.context = _context(self.home)
        self.social = SocialManager(self.context)
        self.alpha = FakeAdapter("alpha")
        self.beta = FakeAdapter("beta")
        self.social.register_adapter(self.alpha)
        self.social.register_adapter(self.beta)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def _token(self) -> str:
        """Multi-platform posting is a confirmable capability by design."""
        return self.context.policy.issue_confirmation("social.bulk", ttl=60.0)

    def test_bulk_publish_without_a_confirmation_token_is_refused(self):
        """A prompt-injected agent must not be able to blast every account."""
        self.social.connect("alpha", "@a@alpha.test")
        self.social.connect("beta", "@b@beta.test")
        with self.assertRaises(SocialError) as caught:
            self.social.publish("no authority")
        self.assertIn("confirmation", str(caught.exception))
        self.assertEqual(self.alpha.calls, [])

    def test_single_platform_post_needs_no_token(self):
        self.social.connect("alpha", "@a@alpha.test")
        self.assertTrue(self.social.publish("just one", platforms=["alpha"]).ok)

    def test_connect_and_list(self):
        self.social.connect("alpha", "@me@alpha.test")
        accounts = self.social.list_accounts()
        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0].handle, "@me@alpha.test")

    def test_connect_inherits_the_adapter_character_limit(self):
        self.social.register_adapter(FakeAdapter("short", max_chars=140))
        account = self.social.connect("short", "@me@short.test")
        self.assertEqual(account.limits["max_chars"], 140)

    def test_duplicate_connection_is_refused(self):
        self.social.connect("alpha", "@me@alpha.test")
        with self.assertRaises(SocialError):
            self.social.connect("alpha", "@me@alpha.test")

    def test_disconnect_removes_the_account(self):
        self.social.connect("alpha", "@me@alpha.test")
        self.assertEqual(self.social.disconnect("alpha", "@me@alpha.test"), 1)
        self.assertEqual(self.social.list_accounts(), [])

    def test_disconnecting_an_unknown_account_raises(self):
        with self.assertRaises(NotFound):
            self.social.disconnect("alpha", "@ghost@alpha.test")

    def test_publish_reaches_every_targeted_platform(self):
        self.social.connect("alpha", "@a@alpha.test")
        self.social.connect("beta", "@b@beta.test")
        outcome = self.social.publish("hello both", confirmation=self._token())
        self.assertTrue(outcome.ok)
        self.assertEqual(len(outcome.posted), 2)
        self.assertEqual(self.alpha.calls, ["hello both"])
        self.assertEqual(self.beta.calls, ["hello both"])

    def test_publish_to_a_subset_of_platforms(self):
        self.social.connect("alpha", "@a@alpha.test")
        self.social.connect("beta", "@b@beta.test")
        outcome = self.social.publish("only alpha", platforms=["alpha"])
        self.assertEqual(len(outcome.posted), 1)
        self.assertEqual(self.beta.calls, [])

    def test_publish_with_no_connected_accounts_raises(self):
        with self.assertRaises(SocialError):
            self.social.publish("nobody home")

    def test_empty_content_is_rejected(self):
        self.social.connect("alpha", "@a@alpha.test")
        with self.assertRaises(ValidationError):
            self.social.publish("   ")

    def test_one_platform_failing_does_not_stop_the_others(self):
        """The core isolation guarantee."""
        self.social.register_adapter(FakeAdapter("broken", fail=True))
        self.social.connect("alpha", "@a@alpha.test")
        self.social.connect("broken", "@b@broken.test")
        outcome = self.social.publish("partial failure", confirmation=self._token())
        self.assertFalse(outcome.ok)
        self.assertEqual(len(outcome.posted), 1)
        self.assertEqual(len(outcome.failed), 1)
        self.assertEqual(outcome.failed[0].platform, "broken")
        self.assertEqual(self.alpha.calls, ["partial failure"])

    def test_parallel_fan_out_is_actually_parallel(self):
        """Four 0.15s platforms must take about 0.15s, not 0.6s."""
        for index in range(4):
            self.social.register_adapter(FakeAdapter(f"slow{index}", delay=0.15))
            self.social.connect(f"slow{index}", f"@me@slow{index}.test")
        started = time.perf_counter()
        outcome = self.social.publish("parallel", parallel=True, confirmation=self._token())
        elapsed = time.perf_counter() - started
        self.assertEqual(len(outcome.posted), 4)
        self.assertLess(elapsed, 0.45, f"fan-out ran serially: {elapsed:.2f}s")

    def test_serial_mode_is_offered_and_slower(self):
        for index in range(3):
            self.social.register_adapter(FakeAdapter(f"seq{index}", delay=0.1))
            self.social.connect(f"seq{index}", f"@me@seq{index}.test")
        started = time.perf_counter()
        self.social.publish("serial", parallel=False, confirmation=self._token())
        self.assertGreater(time.perf_counter() - started, 0.25)

    def test_every_attempt_is_persisted(self):
        self.social.connect("alpha", "@a@alpha.test")
        self.social.publish("recorded")
        history = self.social.history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], PostStatus.POSTED)
        self.assertEqual(history[0]["content"], "recorded")

    def test_failures_are_persisted_with_their_error(self):
        self.social.register_adapter(FakeAdapter("broken", fail=True))
        self.social.connect("broken", "@b@broken.test")
        self.social.publish("will fail")
        history = self.social.history(status=PostStatus.FAILED)
        self.assertEqual(len(history), 1)
        self.assertIn("synthetic", history[0]["error"])

    def test_unregistered_platform_is_reported_not_raised(self):
        """A missing adapter is a per-platform failure, not an exception."""
        self.social.connect("gamma", "@g@gamma.test")
        outcome = self.social.publish("no adapter")
        self.assertFalse(outcome.ok)
        self.assertIn("no adapter", outcome.failed[0].error)

    def test_inactive_accounts_are_skipped(self):
        account = self.social.connect("alpha", "@a@alpha.test")
        self.social.accounts.update(account.id, {"active": 0})
        outcome = self.social.publish("inactive")
        self.assertFalse(outcome.ok)
        self.assertIn("inactive", outcome.failed[0].error)
        self.assertEqual(self.alpha.calls, [])

    def test_overlong_content_fails_without_calling_the_platform(self):
        self.social.register_adapter(FakeAdapter("tiny", max_chars=5))
        self.social.connect("tiny", "@t@tiny.test")
        outcome = self.social.publish("this is much longer than five")
        self.assertFalse(outcome.ok)
        adapter = self.social.adapters["tiny"]
        self.assertEqual(adapter.calls, [], "platform was called despite invalid content")


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-rate-")
        self.context = _context(self.home)
        self.social = SocialManager(self.context)
        self.adapter = FakeAdapter("alpha")
        self.social.register_adapter(self.adapter)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_second_post_within_the_interval_is_refused_locally(self):
        account = self.social.connect("alpha", "@a@alpha.test", limits={"min_interval_seconds": 60})
        self.assertTrue(self.social.publish("first").ok)
        outcome = self.social.publish("too soon")
        self.assertFalse(outcome.ok)
        self.assertIn("rate limited", outcome.failed[0].error)
        self.assertEqual(len(self.adapter.calls), 1, "platform was hit despite the local limit")
        self.assertEqual(self.social.stats["rate_limited"], 1)

    def test_posting_is_allowed_once_the_interval_elapses(self):
        self.social.connect("alpha", "@a@alpha.test", limits={"min_interval_seconds": 0.05})
        self.social.publish("first")
        time.sleep(0.06)
        self.assertTrue(self.social.publish("second").ok)
        self.assertEqual(len(self.adapter.calls), 2)

    def test_posts_today_counts_only_published_posts(self):
        account = self.social.connect("alpha", "@a@alpha.test")
        self.assertEqual(self.social.posts_today(account), 0)
        self.social.publish("one")
        self.social.publish("two")  # rate limited, so not counted
        self.assertEqual(self.social.posts_today(account), 1)


class SchedulingTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-sched-")
        self.context = _context(self.home)
        self.social = SocialManager(self.context)
        self.adapter = FakeAdapter("alpha")
        self.social.register_adapter(self.adapter)
        self.social.connect("alpha", "@a@alpha.test", limits={"min_interval_seconds": 0})

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_schedule_creates_a_scheduled_row(self):
        post_id = self.social.schedule("later", "alpha", "@a@alpha.test", at=time.time() + 3600)
        rows = self.social.history(status=PostStatus.SCHEDULED)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], post_id)

    def test_scheduling_for_an_unknown_account_raises(self):
        with self.assertRaises(NotFound):
            self.social.schedule("x", "alpha", "@ghost@alpha.test", at=time.time())

    def test_due_returns_only_posts_whose_time_has_come(self):
        now = time.time()
        self.social.schedule("past", "alpha", "@a@alpha.test", at=now - 100)
        self.social.schedule("future", "alpha", "@a@alpha.test", at=now + 3600)
        due = self.social.due(now=now)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["content"], "past")

    def test_run_due_publishes_and_clears_the_queue(self):
        self.social.schedule("go", "alpha", "@a@alpha.test", at=time.time() - 10)
        results = self.social.run_due()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok)
        self.assertEqual(self.adapter.calls, ["go"])
        self.assertEqual(self.social.due(), [])

    def test_run_due_marks_orphaned_posts_failed(self):
        post_id = self.social.schedule("orphan", "alpha", "@a@alpha.test", at=time.time() - 10)
        account = self.social.list_accounts()[0]
        self.social.accounts.delete(account.id)
        self.social.run_due()
        row = self.social.posts.get(post_id)
        self.assertEqual(row["status"], PostStatus.FAILED)
        self.assertIn("no longer connected", row["error"])


class BuiltinAdapterTests(unittest.TestCase):
    """Adapter wiring only — no network calls."""

    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-adapters-")
        self.context = _context(self.home)

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_builtins_register_under_their_platform_names(self):
        social = SocialManager(self.context).register_builtins()
        self.assertIn("mastodon", social.adapters)
        self.assertIn("bluesky", social.adapters)

    def test_platform_character_limits_are_the_real_ones(self):
        social = SocialManager(self.context).register_builtins()
        self.assertEqual(social.adapters["mastodon"].max_chars, 500)
        self.assertEqual(social.adapters["bluesky"].max_chars, 300)

    def test_mastodon_instance_is_derived_from_the_handle_domain(self):
        from nomorals.social.adapters.mastodon import Adapter

        adapter = Adapter()
        self.assertEqual(
            adapter._base(Account(platform="mastodon", handle="@me@example.social")),
            "https://example.social",
        )

    def test_mastodon_base_url_override_wins(self):
        from nomorals.social.adapters.mastodon import Adapter

        account = Account(
            platform="mastodon", handle="@me@x.test",
            limits={"base_url": "https://custom.instance/"},
        )
        self.assertEqual(Adapter()._base(account), "https://custom.instance")

    def test_mastodon_without_a_domain_raises(self):
        from nomorals.social.adapters.mastodon import Adapter

        with self.assertRaises(ValueError):
            Adapter()._base(Account(platform="mastodon", handle="nohandle"))

    def test_bluesky_permalink_is_built_from_the_at_uri(self):
        from nomorals.social.adapters.bluesky import _post_url

        url = _post_url("@me.bsky.social", "at://did:plc:abc/app.bsky.feed.post/rkey1")
        self.assertEqual(url, "https://bsky.app/profile/me.bsky.social/post/rkey1")

    def test_bluesky_without_an_app_password_fails_cleanly(self):
        from nomorals.social.adapters.bluesky import Adapter

        result = Adapter().post(Account(platform="bluesky", handle="me.bsky.social"), "hi")
        self.assertFalse(result.ok)
        self.assertIn("app password", result.error)

    def test_health_is_false_without_credentials(self):
        social = SocialManager(self.context).register_builtins()
        self.assertFalse(social.adapters["mastodon"].health(Account(platform="mastodon", handle="@a@b.test")))
        self.assertFalse(social.adapters["bluesky"].health(Account(platform="bluesky", handle="a.bsky.social")))


class StatsTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-stats-")
        self.context = _context(self.home)
        self.social = SocialManager(self.context)
        self.social.register_adapter(FakeAdapter("alpha"))

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_stats_report_accounts_statuses_and_adapters(self):
        self.social.connect("alpha", "@a@alpha.test", limits={"min_interval_seconds": 0})
        self.social.publish("one")
        self.social.publish("two")
        stats = self.social.stats_snapshot()
        self.assertEqual(stats["accounts"], 1)
        self.assertEqual(stats["published"], 2)
        self.assertEqual(stats["by_status"][PostStatus.POSTED], 2)
        self.assertIn("alpha", stats["adapters"])


if __name__ == "__main__":
    unittest.main()
