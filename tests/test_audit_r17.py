"""R17 audit tests.

- trial SMS watches persist across restarts like assist runs: the watch
  row pins the exact temp number grabbed; a restart resumes watches
  still inside their deadline and closes expired ones as ``timeout``,
  reporting once via the durable notifier; ``/trial status`` shows
  watches; old terminal watches are pruned.
- (music test-hygiene and flaky-test hardening are fixes inside the
  existing test files, covered by their own runs.)
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.storage.db import Database


def _ctx(db):
    return build_context(db=db, with_tools=True, with_router=False,
                         with_memory=False)


_NUMBER_A = {
    "status": "ok", "number": "+15550101", "masked": "+1555••101",
    "country": "us", "country_name": "United States",
    "provider": "simcodes", "inbox_id": "inbox-A",
}
_NUMBER_B = {
    "status": "ok", "number": "+15550202", "masked": "+1555••202",
    "country": "us", "country_name": "United States",
    "provider": "simcodes", "inbox_id": "inbox-B",
}


def _seed_watch(db, watch_id, state="watching", age_s=0, deadline_in_s=120,
                info=None, chat_key="telegram:123", timeout=180):
    now = time.time()
    db.execute(
        "INSERT OR REPLACE INTO trial_sms_watches"
        " (watch_id, number, number_info, chat_key, started, deadline,"
        "  timeout, state, code, note, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (watch_id, (info or _NUMBER_A)["masked"],
         json.dumps(info or _NUMBER_A), chat_key, now - age_s,
         now + deadline_in_s, timeout, state, "", "", now - age_s),
    )


class SmsWatchPersistenceTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.trial.flow import TrialFlow
        TrialFlow._sms_recovery_done = True  # tests drive recovery explicitly
        # file-backed: the watch thread gets its own sqlite connection,
        # and :memory: DBs are per-connection (invisible across threads).
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "t.db")
        self.db.migrate()  # brings trial_sms_watches (migration 72)

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _flow(self):
        from nomorals.agents.trial.flow import TrialFlow
        return TrialFlow(_ctx(self.db))

    def _stash(self, info):
        self.db.execute(
            "INSERT OR REPLACE INTO kv_store (key, value, kind, updated_at)"
            " VALUES ('trial.temp_sms', ?, 'json', ?)",
            (json.dumps(info), time.time()),
        )

    def _wait_state(self, watch_id, want, timeout_s=15):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            row = self.db.query_one(
                "SELECT state FROM trial_sms_watches WHERE watch_id=?",
                (watch_id,))
            if row and row["state"] in want:
                return row["state"]
            time.sleep(0.05)
        self.fail(f"watch {watch_id} never reached {want}")

    def test_async_persists_watch_and_finishes(self):
        from nomorals.agents.trial.flow import TrialFlow
        flow = self._flow()
        self._stash(_NUMBER_A)
        with mock.patch("nomorals.accounts.temp_sms.wait_code",
                        return_value="") as wc:
            out = flow.temp_sms_code_async("telegram:123", timeout=30)
        self.assertIn("watching", out)
        self.assertIn("+1555••101", out)
        rows = self.db.query("SELECT * FROM trial_sms_watches")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["chat_key"], "telegram:123")
        self.assertEqual(json.loads(row["number_info"])["inbox_id"],
                         "inbox-A")
        # the background thread polls with the pinned info, then closes
        # the row as timeout (no code arrived).
        self._wait_state(row["watch_id"], {"timeout"})
        final = self.db.query_one(
            "SELECT state, note FROM trial_sms_watches WHERE watch_id=?",
            (row["watch_id"],))
        self.assertEqual(final["state"], "timeout")
        self.assertIn("no code arrived", final["note"])
        wc.assert_called_once()
        self.assertEqual(wc.call_args[0][0]["inbox_id"], "inbox-A")

    def test_watch_pins_number_against_newer_grab(self):
        # grabbing a second number mid-watch must not hijack the poll.
        flow = self._flow()
        self._stash(_NUMBER_A)
        seen = []

        def slow_wait(info, timeout=180):
            seen.append(dict(info))
            time.sleep(0.4)
            return ""

        with mock.patch("nomorals.accounts.temp_sms.wait_code",
                        side_effect=slow_wait):
            flow.temp_sms_code_async("telegram:123", timeout=30)
            self._stash(_NUMBER_B)  # owner grabs another number meanwhile
        wid = self.db.query_one(
            "SELECT watch_id FROM trial_sms_watches")["watch_id"]
        self._wait_state(wid, {"timeout"})
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["inbox_id"], "inbox-A")
        row = self.db.query_one(
            "SELECT number_info FROM trial_sms_watches WHERE watch_id=?",
            (wid,))
        self.assertEqual(json.loads(row["number_info"])["inbox_id"],
                         "inbox-A")

    def test_recovery_closes_expired_watch_and_reports(self):
        from nomorals.agents.trial.flow import TrialFlow
        _seed_watch(self.db, "w-expired", state="watching",
                    deadline_in_s=-10)
        _seed_watch(self.db, "w-done", state="done", deadline_in_s=-10)
        recovered = TrialFlow.recover_interrupted_watches(_ctx(self.db))
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["watch_id"], "w-expired")
        self.assertEqual(recovered[0]["state"], "timeout")
        row = self.db.query_one(
            "SELECT state, note FROM trial_sms_watches WHERE watch_id=?",
            ("w-expired",))
        self.assertEqual(row["state"], "timeout")
        self.assertIn("expired", row["note"])
        # terminal row untouched
        self.assertEqual(
            self.db.query_one("SELECT state FROM trial_sms_watches"
                              " WHERE watch_id='w-done'")["state"], "done")
        # the report went through the durable notifier (persisted row)
        notes = self.db.query(
            "SELECT title, body FROM notifications WHERE kind='trial'")
        self.assertTrue(any("expired" in (n.get("title") or "")
                            for n in notes),
                        f"no trial expiry notification persisted: {notes}")
        # second recovery is a no-op — nothing re-reported
        self.assertEqual(
            TrialFlow.recover_interrupted_watches(_ctx(self.db)), [])

    def test_recovery_rearms_live_watch(self):
        from nomorals.agents.trial.flow import TrialFlow
        _seed_watch(self.db, "w-live", state="watching", deadline_in_s=60,
                    info=_NUMBER_A, timeout=60)
        with mock.patch("nomorals.accounts.temp_sms.wait_code",
                        return_value="424242") as wc:
            recovered = TrialFlow.recover_interrupted_watches(_ctx(self.db))
            self.assertEqual(len(recovered), 1)
            self.assertEqual(recovered[0]["state"], "watching")
            # the re-armed thread polls with the pinned info for the
            # remaining time and lands the code.
            self._wait_state("w-live", {"done"})
        wc.assert_called_once()
        self.assertEqual(wc.call_args[0][0]["inbox_id"], "inbox-A")
        row = self.db.query_one(
            "SELECT state, code FROM trial_sms_watches WHERE watch_id=?",
            ("w-live",))
        self.assertEqual(row["state"], "done")
        self.assertEqual(row["code"], "424242")
        notes = self.db.query(
            "SELECT body FROM notifications WHERE kind='trial'")
        self.assertTrue(any("424242" in (n.get("body") or "")
                            for n in notes),
                        f"code never reported: {notes}")

    def test_recovery_marks_failed_poll_honestly(self):
        from nomorals.agents.trial.flow import TrialFlow
        _seed_watch(self.db, "w-live2", state="watching", deadline_in_s=60,
                    info=_NUMBER_A, timeout=60)
        with mock.patch("nomorals.accounts.temp_sms.wait_code",
                        side_effect=RuntimeError("provider down")):
            TrialFlow.recover_interrupted_watches(_ctx(self.db))
            self._wait_state("w-live2", {"timeout"})
        row = self.db.query_one(
            "SELECT state, note FROM trial_sms_watches WHERE watch_id=?",
            ("w-live2",))
        self.assertEqual(row["state"], "timeout")
        self.assertIn("provider down", row["note"])

    def test_status_shows_watches(self):
        flow = self._flow()
        _seed_watch(self.db, "w-s1", state="done", info=_NUMBER_A)
        # reload the in-memory view the way a fresh process would
        flow._load_sms_watches()
        status = flow.assist_status()
        self.assertIn("sms code watches", status)
        self.assertIn("+1555••101", status)
        self.assertIn("done", status)

    def test_prune_drops_old_terminal_watches(self):
        flow = self._flow()
        _seed_watch(self.db, "w-old", state="timeout",
                    age_s=8 * 24 * 3600, deadline_in_s=-8 * 24 * 3600)
        _seed_watch(self.db, "w-fresh", state="done")
        flow._load_sms_watches()  # pruning happens on load
        remaining = {r["watch_id"]
                     for r in self.db.query("SELECT watch_id"
                                             " FROM trial_sms_watches")}
        self.assertNotIn("w-old", remaining)
        self.assertIn("w-fresh", remaining)


if __name__ == "__main__":
    unittest.main()
