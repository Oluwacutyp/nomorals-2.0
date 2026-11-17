"""Breach organ: k-anonymity password check, HIBP account check, watcher.

Failing-first: the organ (nomorals/security/breach.py) does not exist
on the base — this file errors at collection there.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.security.breach import (
    BreachRecord,
    BreachWatchStore,
    check_account,
    check_password,
    ensure_breach_watch_job,
    format_breach_reply,
    run_breach_watch_tick_once,
)

# SHA-1("password") = 5BAA61E4C9B93F3F0682250B6CF8331B7EE68FD8
_PWNED_PREFIX = "5BAA6"
_PWNED_SUFFIX = "1E4C9B93F3F0682250B6CF8331B7EE68FD8"


def _range_body_with_pwned() -> str:
    return (
        "0018A45C4D1DEF81644B54AB7F969B88D65:1\n"
        f"{_PWNED_SUFFIX}:9543753\n"
        "00D4F6E8FA6EECAD2A3AA415EEC418D38:2\n"
    )


def _account_body() -> str:
    return json.dumps([
        {
            "Name": "Adobe",
            "Title": "Adobe",
            "BreachDate": "2013-10-04",
            "DataClasses": ["Email addresses", "Password hints",
                            "Passwords", "Usernames"],
        },
        {
            "Name": "LinkedIn",
            "Title": "LinkedIn",
            "BreachDate": "2012-05-05",
            "DataClasses": ["Email addresses", "Passwords"],
        },
    ])


class _RecordingFetcher:
    """A fetcher that records every URL it is asked for."""

    def __init__(self, routes: dict[str, tuple[int, str]]):
        self.routes = routes
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str] | None = None):
        self.urls.append(url)
        self.headers.append(dict(headers or {}))
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                return payload
        return (404, "")


class PasswordCheckTests(unittest.TestCase):
    def test_pwned_password_flags_with_count(self):
        fetcher = _RecordingFetcher({
            "https://api.pwnedpasswords.com/range/": (200, _range_body_with_pwned()),
        })
        out = check_password("password", fetcher=fetcher)
        self.assertTrue(out["ok"])
        self.assertTrue(out["pwned"])
        self.assertEqual(out["count"], 9543753)
        self.assertEqual(out["prefix"], _PWNED_PREFIX)

    def test_only_five_hash_chars_ever_leave_the_box(self):
        fetcher = _RecordingFetcher({
            "https://api.pwnedpasswords.com/range/": (200, _range_body_with_pwned()),
        })
        check_password("password", fetcher=fetcher)
        self.assertEqual(len(fetcher.urls), 1)
        url = fetcher.urls[0]
        self.assertEqual(
            url, "https://api.pwnedpasswords.com/range/5BAA6")
        full = hashlib.sha1(b"password").hexdigest().upper()
        # "password" is a substring of the service hostname itself;
        # the guarantee is the full hash and its suffix never travel.
        self.assertNotIn(full, url)
        self.assertNotIn(full[5:], url)

    def test_random_long_password_clears(self):
        fetcher = _RecordingFetcher({
            "https://api.pwnedpasswords.com/range/": (200, _range_body_with_pwned()),
        })
        out = check_password("X7q!zR9vLm2#kP4wT8sDfGhJkL0pOaS5dFgHjKl",
                             fetcher=fetcher)
        self.assertTrue(out["ok"])
        self.assertFalse(out["pwned"])
        self.assertEqual(out["count"], 0)

    def test_empty_password_is_an_honest_refusal(self):
        out = check_password("", fetcher=_RecordingFetcher({}))
        self.assertFalse(out["ok"])
        self.assertIn("password", out["error"].lower())

    def test_dead_network_is_ok_false_never_raises(self):
        def boom(url, headers=None):
            raise ConnectionError("no route")

        out = check_password("password", fetcher=boom)
        self.assertFalse(out["ok"])
        self.assertIn("error", out)


class AccountCheckTests(unittest.TestCase):
    def test_breached_account_names_breaches_with_dates_and_classes(self):
        fetcher = _RecordingFetcher({
            "https://haveibeenpwned.com/api/v3/breachedaccount/": (
                200, _account_body()),
        })
        out = check_account("ada@example.com", key="k", fetcher=fetcher)
        self.assertTrue(out["ok"])
        self.assertTrue(out["pwned"])
        self.assertEqual(out["count"], 2)
        first = out["breaches"][0]
        self.assertEqual(first["name"], "Adobe")
        self.assertEqual(first["date"], "2013-10-04")
        self.assertIn("Passwords", first["data_classes"])
        self.assertIn("Email addresses", first["data_classes"])

    def test_404_means_clean_not_an_error(self):
        fetcher = _RecordingFetcher({})
        out = check_account("nobody@example.com", key="k", fetcher=fetcher)
        self.assertTrue(out["ok"])
        self.assertFalse(out["pwned"])
        self.assertEqual(out["breaches"], [])

    def test_no_key_is_an_honest_refusal_naming_the_key(self):
        out = check_account("ada@example.com", key="",
                            fetcher=_RecordingFetcher({}))
        self.assertFalse(out["ok"])
        self.assertIn("NM_OSINT_HIBP_KEY", out["error"])

    def test_domain_uses_the_domain_endpoint(self):
        fetcher = _RecordingFetcher({
            "https://haveibeenpwned.com/api/v3/breacheddomain/": (
                200, _account_body()),
        })
        out = check_account("example.com", key="k", fetcher=fetcher)
        self.assertTrue(out["ok"])
        self.assertTrue(out["pwned"])
        self.assertTrue(fetcher.urls[0].startswith(
            "https://haveibeenpwned.com/api/v3/breacheddomain/example.com"))

    def test_api_key_rides_the_header_not_the_url(self):
        fetcher = _RecordingFetcher({
            "https://haveibeenpwned.com/api/v3/breachedaccount/": (
                200, _account_body()),
        })
        check_account("ada@example.com", key="secret-key", fetcher=fetcher)
        self.assertEqual(fetcher.headers[0].get("hibp-api-key"), "secret-key")
        self.assertNotIn("secret-key", fetcher.urls[0])

    def test_record_round_trip(self):
        rec = BreachRecord(name="Adobe", date="2013-10-04",
                           data_classes=("Passwords",),
                           source="haveibeenpwned")
        self.assertEqual(BreachRecord.from_dict(rec.to_dict()), rec)


class WatchStoreTests(unittest.TestCase):
    def _store(self, tmp: Path) -> BreachWatchStore:
        return BreachWatchStore(tmp / "breach_watch.json")

    def test_add_list_remove_round_trip(self):
        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/breach-verify") as d:
            store = self._store(Path(d))
            self.assertEqual(store.list_watched(), [])
            self.assertTrue(store.add("ada@example.com"))
            self.assertFalse(store.add("ada@example.com"))
            self.assertEqual(store.list_watched(), ["ada@example.com"])
            self.assertTrue(store.remove("ada@example.com"))
            self.assertFalse(store.remove("ada@example.com"))
            self.assertEqual(store.list_watched(), [])

    def test_seen_diff_names_only_new_breaches(self):
        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/breach-verify") as d:
            store = self._store(Path(d))
            store.add("ada@example.com")
            recs = [BreachRecord(name="Adobe", date="2013-10-04",
                                 data_classes=["Passwords"]),
                    BreachRecord(name="LinkedIn", date="2012-05-05",
                                 data_classes=["Passwords"])]
            self.assertEqual(
                sorted(r.name for r in store.new_breaches("ada@example.com", recs)),
                ["Adobe", "LinkedIn"])
            store.record("ada@example.com", recs)
            self.assertEqual(
                store.new_breaches("ada@example.com", recs), [])
            recs2 = recs + [BreachRecord(name="Dropbox", date="2012-07-01",
                                         data_classes=["Email addresses"])]
            self.assertEqual(
                [r.name for r in store.new_breaches("ada@example.com", recs2)],
                ["Dropbox"])

    def test_state_survives_reload(self):
        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/breach-verify") as d:
            store = self._store(Path(d))
            store.add("ada@example.com")
            store.record("ada@example.com", [BreachRecord(
                name="Adobe", date="2013-10-04", data_classes=["Passwords"])])
            again = self._store(Path(d))
            self.assertEqual(again.list_watched(), ["ada@example.com"])
            self.assertEqual(again.seen_names("ada@example.com"), ["Adobe"])


class WatchTickTests(unittest.TestCase):
    def _ctx(self, tmp: Path, key: str = "k") -> SimpleNamespace:
        home = Path(tmp)
        return SimpleNamespace(
            settings=SimpleNamespace(
                home_path=home,
                osint=SimpleNamespace(hibp_key=key)))

    def test_tick_publishes_once_on_a_new_breach_and_stays_silent_after(self):
        fetcher = _RecordingFetcher({
            "https://haveibeenpwned.com/api/v3/breachedaccount/": (
                200, _account_body()),
        })
        published: list[dict[str, str]] = []

        class FakeNotifier:
            def publish(self, kind, title=None, body=None, **kw):
                published.append({"kind": kind, "title": title or "",
                                  "body": body or ""})

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/breach-verify") as d:
            ctx = self._ctx(Path(d))
            store = BreachWatchStore(Path(d) / "data" / "breach_watch.json")
            store.add("ada@example.com")
            out = run_breach_watch_tick_once(
                context=ctx, notifier=FakeNotifier(), fetcher=fetcher,
                state_path=Path(d) / "data" / "breach_watch.json")
            self.assertTrue(out["ok"])
            self.assertEqual(out["new"], ["Adobe", "LinkedIn"])
            self.assertEqual(len(published), 1)
            self.assertIn("Adobe", published[0]["body"])
            self.assertIn("LinkedIn", published[0]["body"])
            published.clear()
            out2 = run_breach_watch_tick_once(
                context=ctx, notifier=FakeNotifier(), fetcher=fetcher,
                state_path=Path(d) / "data" / "breach_watch.json")
            self.assertTrue(out2["ok"])
            self.assertEqual(out2["new"], [])
            self.assertEqual(published, [])

    def test_tick_without_a_key_fails_honestly_and_publishes_nothing(self):
        published: list[dict[str, str]] = []

        class FakeNotifier:
            def publish(self, kind, title=None, body=None, **kw):
                published.append({"kind": kind})

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/breach-verify") as d:
            ctx = self._ctx(Path(d), key="")
            store = BreachWatchStore(Path(d) / "data" / "breach_watch.json")
            store.add("ada@example.com")
            out = run_breach_watch_tick_once(
                context=ctx, notifier=FakeNotifier(),
                fetcher=_RecordingFetcher({}),
                state_path=Path(d) / "data" / "breach_watch.json")
            self.assertFalse(out["ok"])
            self.assertEqual(published, [])

    def test_tick_never_raises(self):
        def boom(url, headers=None):
            raise RuntimeError("network down")

        with tempfile.TemporaryDirectory(dir="/home/hatch/workspace/breach-verify") as d:
            ctx = self._ctx(Path(d))
            store = BreachWatchStore(Path(d) / "data" / "breach_watch.json")
            store.add("ada@example.com")
            out = run_breach_watch_tick_once(
                context=ctx, notifier=None, fetcher=boom,
                state_path=Path(d) / "data" / "breach_watch.json")
            self.assertIsInstance(out, dict)
            self.assertIn("ok", out)


class ReplyTests(unittest.TestCase):
    def test_account_reply_names_breaches_dates_and_classes(self):
        out = {"ok": True, "target": "ada@example.com", "pwned": True,
               "count": 1, "breaches": [BreachRecord(
                   name="Adobe", date="2013-10-04",
                   data_classes=["Email addresses", "Passwords"]).to_dict()]}
        text = format_breach_reply(out)
        self.assertIn("Adobe", text)
        self.assertIn("2013-10-04", text)
        self.assertIn("Passwords", text)

    def test_clean_account_reply_says_clean(self):
        text = format_breach_reply({"ok": True, "target": "ada@example.com",
                                    "pwned": False, "count": 0,
                                    "breaches": []})
        self.assertIn("clean", text.lower())

    def test_pwned_password_reply_warns(self):
        text = format_breach_reply({"ok": True, "pwned": True,
                                    "count": 9543753, "kind": "password"})
        self.assertIn("9,543,753", text)
        self.assertIn("do not use", text.lower())

    def test_failed_check_reply_names_the_error(self):
        text = format_breach_reply({"ok": False,
                                    "error": "no HIBP API key set"})
        self.assertIn("no HIBP API key set", text)


class EnsureJobTests(unittest.TestCase):
    def test_ensure_registers_a_daily_job_idempotently(self):
        class FakeScheduler:
            def __init__(self):
                self.jobs: list[dict[str, Any]] = []

            def list_jobs(self):
                return list(self.jobs)

            def add(self, name, spec, kind, payload, routine=False):
                job = {"name": name, "spec": spec, "kind": kind,
                       "payload": payload, "routine": routine,
                       "id": f"job-{len(self.jobs)}"}
                self.jobs.append(job)
                return job

            def remove(self, job_id):
                self.jobs = [j for j in self.jobs if j["id"] != job_id]

        sched = FakeScheduler()
        job1 = ensure_breach_watch_job(sched)
        job2 = ensure_breach_watch_job(sched)
        self.assertIsNotNone(job1)
        self.assertEqual(job1, job2)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(job1["payload"]["tool"], "__breach_watch_tick__")

    def test_ensure_with_no_scheduler_returns_none(self):
        self.assertIsNone(ensure_breach_watch_job(None))


class OsintDelegationTests(unittest.TestCase):
    def test_breach_check_now_carries_data_classes(self):
        from nomorals.tools import osint_people

        fetcher = _RecordingFetcher({
            "https://haveibeenpwned.com/api/v3/breachedaccount/": (
                200, _account_body()),
        })
        ctx = SimpleNamespace(settings=SimpleNamespace(
            osint=SimpleNamespace(hibp_key="k")))
        with mock.patch("nomorals.security.breach._default_fetcher",
                        new=fetcher):
            out = osint_people.breach_check(ctx, "ada@example.com")
        self.assertTrue(out["checked"])
        self.assertTrue(out["pwned"])
        self.assertIn("Passwords",
                      out["breaches"][0].get("data_classes", []))


if __name__ == "__main__":
    unittest.main()
