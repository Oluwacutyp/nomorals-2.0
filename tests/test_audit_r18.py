"""R18 audit tests.

- codebeast small-run rebuild: ``build_500k.py --target <small>`` scales
  every source cap proportionally so the mix stays faithful to the 500K
  recipe (no first-source fill bias); the v3 notebook's rebuild fallback
  passes ``--target {TARGET_ROWS}`` instead of hardcoding 500000.
- trial SMS watch recovery digest: a restart with several live watches
  publishes ONE owner message (digest), not one ping per watch; a single
  watch keeps the exact per-watch message it always had.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.storage.db import Database

_REPO = Path(__file__).resolve().parents[1]


def _load_build_500k():
    path = _REPO / "docs" / "codebeast" / "build_500k.py"
    spec = importlib.util.spec_from_file_location("cb_build_500k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
_NUMBER_C = {
    "status": "ok", "number": "+15550303", "masked": "+1555••303",
    "country": "us", "country_name": "United States",
    "provider": "simcodes", "inbox_id": "inbox-C",
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


class CodebeastSmallTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_build_500k()

    def test_scale_keeps_full_mix_at_500k(self):
        scaled = self.mod.scale_sources_for_target(500000)
        self.assertEqual([c for _, _, _, c, _ in scaled],
                         [c for _, _, _, c, _ in self.mod.SOURCES])

    def test_scale_proportional_and_no_source_drops_out(self):
        scaled = self.mod.scale_sources_for_target(6000)
        self.assertEqual(len(scaled), 5)
        caps = [c for _, _, _, c, _ in scaled]
        # every source still present (the bug: OpenHermes alone would fill
        # a 6000-row budget, starving the other four sources)
        self.assertTrue(all(c >= 1 for c in caps), caps)
        # proportional to the original caps (~1.2%)
        for (orig, new) in zip(
                [c for _, _, _, c, _ in self.mod.SOURCES], caps):
            self.assertAlmostEqual(new / orig, 6000 / 500000, delta=0.01)
        # total lands near the target (source caps are an upper bound)
        self.assertLessEqual(sum(caps), 6000 * 1.2)

    def test_scale_tiny_target_still_covers_all_sources(self):
        scaled = self.mod.scale_sources_for_target(100)
        caps = [c for _, _, _, c, _ in scaled]
        self.assertTrue(all(c >= 1 for c in caps), caps)

    def test_notebook_rebuild_fallback_uses_target_rows(self):
        nb = json.loads(
            (_REPO / "docs" / "codebeast" / "codebeast_v3.ipynb").read_text(
                encoding="utf-8"))
        cells = ["".join(c.get("source", [])) for c in nb["cells"]]
        load_cells = [s for s in cells if "run_line_magic" in s
                      and "build_500k.py" in s]
        self.assertEqual(len(load_cells), 1)
        src = load_cells[0]
        self.assertIn("--target {TARGET_ROWS}", src)
        self.assertNotIn("--target 500000", src)


class SmsWatchRecoveryDigestTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.trial.flow import TrialFlow
        TrialFlow._sms_recovery_done = True  # tests drive recovery explicitly
        # file-backed: the watch thread gets its own sqlite connection,
        # like production.
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self._tmp.name) / "t.db")
        self.db.migrate()  # brings trial_sms_watches (migration 72)

    def tearDown(self):
        try:
            self.db.close()
        finally:
            self._tmp.cleanup()

    def _wait_state(self, watch_id, want, timeout_s=15):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            row = self.db.query_one(
                "SELECT state FROM trial_sms_watches WHERE watch_id=?",
                (watch_id,))
            if row and row["state"] in want:
                return
            time.sleep(0.05)
        self.fail(f"watch {watch_id} never reached {want}")

    def _recovery_titles(self):
        rows = self.db.query(
            "SELECT title FROM notifications WHERE kind='trial'")
        return [r.get("title") or "" for r in rows]

    def test_multi_watch_recovery_publishes_single_digest(self):
        from nomorals.agents.trial.flow import TrialFlow
        _seed_watch(self.db, "w-exp-1", state="watching", deadline_in_s=-10,
                    info=_NUMBER_A)
        _seed_watch(self.db, "w-exp-2", state="watching", deadline_in_s=-5,
                    info=_NUMBER_B)
        _seed_watch(self.db, "w-live", state="watching", deadline_in_s=3600,
                    info=_NUMBER_C, timeout=3600)
        with mock.patch("nomorals.accounts.temp_sms.wait_code",
                        return_value="424242"):
            recovered = TrialFlow.recover_interrupted_watches(_ctx(self.db))
            self.assertEqual(len(recovered), 3)
            # the re-armed thread lands the code on its own message later
            self._wait_state("w-live", {"done"})
        titles = self._recovery_titles()
        digests = [t for t in titles if "recovered" in t]
        self.assertEqual(len(digests), 1,
                         f"expected one recovery digest, got: {titles}")
        self.assertIn("1 resumed", digests[0])
        self.assertIn("2 expired", digests[0])
        # no per-watch resume/expired pings alongside the digest
        self.assertFalse(
            any(t.startswith("sms code watch resumed —") or
                t.startswith("sms code watch expired —") for t in titles),
            f"per-watch pings leaked alongside digest: {titles}")
        body = self.db.query_one(
            "SELECT body FROM notifications WHERE title LIKE '%recovered%'"
        )["body"]
        self.assertIn("+1555••101", body)
        self.assertIn("+1555••202", body)
        self.assertIn("+1555••303", body)

    def test_single_watch_keeps_exact_per_watch_message(self):
        from nomorals.agents.trial.flow import TrialFlow
        _seed_watch(self.db, "w-solo", state="watching", deadline_in_s=-10,
                    info=_NUMBER_A)
        recovered = TrialFlow.recover_interrupted_watches(_ctx(self.db))
        self.assertEqual(len(recovered), 1)
        titles = self._recovery_titles()
        self.assertEqual(len(titles), 1)
        self.assertEqual(titles[0], "sms code watch expired — +1555••101")
        self.assertFalse(any("recovered" in t for t in titles))

    def test_no_watches_publishes_nothing(self):
        from nomorals.agents.trial.flow import TrialFlow
        recovered = TrialFlow.recover_interrupted_watches(_ctx(self.db))
        self.assertEqual(recovered, [])
        self.assertEqual(self._recovery_titles(), [])


if __name__ == "__main__":
    unittest.main()
