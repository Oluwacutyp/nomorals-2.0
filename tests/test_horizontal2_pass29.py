"""Pass #29 — horizontal modules wave 2 (life/household/shows/scam/
focus/preferences/reviews/commitments/identity/people).

Verified findings pinned here:

1. reviews.build_weekly_review counted kept/broken across ALL TIME
   while the focus builder it composes scopes wins to the week — a
   resolution from months ago inflated this week's reliability, and
   build_monthly_review's per-week trend counted the same resolution
   in every week.
2. commitments.parse_deadline: a clause that is ONLY a time of day
   ("by 5pm", "by noon", "before midnight") yielded NO deadline,
   despite the module docstring listing those forms as supported; and
   "by 20 Oct" misread the day-of-month as an hour (20:00 / 05:00
   instead of the 17:00 default).
3. commitments extractor hedge inconsistency: "I'll check the figures
   by Friday" is hedged (nothing extracted) but the identical
   uncontracted "I will check the figures by Friday" extracted a
   commitment — "i will check" was missing from _HEDGES while
   "i'll check" (and both forms of see/try/attempt/think) were there.
4. preferences.PreferenceStore.from_dict dropped the signal log that
   to_dict exports — weights round-tripped, signals vanished.

Plus durability pins (people store + household shared layer survive
reopen with identical contents).
"""

from __future__ import annotations

import datetime as _dt
import tempfile
import unittest
from pathlib import Path

from nomorals.commitments.ledger import (
    CommitmentLedger,
    extract_commitments,
    parse_deadline,
)
from nomorals.family.registry import family_registry
from nomorals.household.household import Household
from nomorals.people.store import PersonStore
from nomorals.preferences.feedback import record_signal
from nomorals.preferences.store import PreferenceStore
from nomorals.reviews import build_monthly_review, build_weekly_review

UTC = _dt.timezone.utc


def _ts(year: int, month: int, day: int, hour: int = 12,
        minute: int = 0) -> float:
    return _dt.datetime(year, month, day, hour, minute,
                        tzinfo=UTC).timestamp()


FRI = _ts(2026, 10, 16, 12)        # Friday 2026-10-16 12:00 UTC
NOW = _ts(2026, 10, 14)            # Wednesday of week Mon 2026-10-12
MON = _ts(2026, 10, 12, 0)


def _msg(text: str, ts: float = FRI, **extra):
    base = {"text": text, "sender": "owner", "timestamp": ts, "id": "m1"}
    base.update(extra)
    return base


# ── 1. reviews reliability is week-scoped ────────────────────────────

class TestReviewsReliabilityWeekScoped(unittest.TestCase):
    def _ledger_with_old_and_new(self) -> CommitmentLedger:
        ledger = CommitmentLedger()
        old = ledger.add(promisee="Ada", what="old promise",
                         ts=NOW - 90 * 86400)
        ledger.mark_kept(old.id, "done long ago", ts=NOW - 89 * 86400)
        fresh = ledger.add(promisee="Ada", what="fresh promise", ts=MON)
        ledger.mark_broken(fresh.id, "missed", ts=MON + 7200)
        return ledger

    def test_old_resolution_does_not_inflate_this_week(self):
        review = build_weekly_review(
            commitments=self._ledger_with_old_and_new(), now=NOW)
        # Only the promise broken THIS week counts.
        self.assertEqual(review.kept, 0)
        self.assertEqual(review.broken, 1)
        self.assertEqual(review.reliability, 0.0)

    def test_resolution_counts_in_its_own_week(self):
        ledger = self._ledger_with_old_and_new()
        own_week_now = NOW - 89 * 86400 + 3600  # inside the old week
        review = build_weekly_review(commitments=ledger, now=own_week_now)
        self.assertEqual(review.kept, 1)
        self.assertEqual(review.broken, 0)
        self.assertEqual(review.reliability, 1.0)

    def test_monthly_trend_does_not_repeat_one_resolution(self):
        ledger = CommitmentLedger()
        only = ledger.add(promisee="Ada", what="only promise",
                          ts=NOW - 90 * 86400)
        ledger.mark_kept(only.id, "done", ts=NOW - 89 * 86400)
        # Three recent weeks (none contains the resolution) plus the
        # resolution's own week: across all four trend entries the
        # single resolution must be counted exactly once. Pre-fix
        # all-time counting reported it in every week (4).
        nows = [NOW - i * 7 * 86400 for i in range(3)]
        nows.append(NOW - 89 * 86400 + 3600)
        weeks = [build_weekly_review(commitments=ledger, now=n)
                 for n in nows]
        monthly = build_monthly_review(weeks=weeks)
        total_kept = sum(t["kept"] for t in monthly.reliability_trend)
        self.assertEqual(total_kept, 1)


# ── 2. deadline parser: pure-time clauses + day-of-month ─────────────

class TestParseDeadlineTimeOfDay(unittest.TestCase):
    def test_pure_time_clauses_resolve_same_day(self):
        self.assertEqual(parse_deadline("by 5pm", FRI),
                         _ts(2026, 10, 16, 17))
        self.assertEqual(parse_deadline("by noon", _ts(2026, 10, 16, 9)),
                         _ts(2026, 10, 16, 12))
        self.assertEqual(parse_deadline("by 17:00", FRI),
                         _ts(2026, 10, 16, 17))
        self.assertEqual(parse_deadline("before noon", _ts(2026, 10, 16, 9)),
                         _ts(2026, 10, 16, 12))

    def test_pure_time_in_the_past_rolls_to_next_day(self):
        # Said at 18:00, "by 5pm" can only mean tomorrow's 5pm.
        self.assertEqual(parse_deadline("by 5pm", _ts(2026, 10, 16, 18)),
                         _ts(2026, 10, 17, 17))
        # "By midnight" means the end of the day, never 00:00 today.
        self.assertEqual(parse_deadline("by midnight", _ts(2026, 10, 16, 9)),
                         _ts(2026, 10, 17, 0))

    def test_day_of_month_is_not_an_hour(self):
        self.assertEqual(parse_deadline("by 20 Oct", FRI),
                         _ts(2026, 10, 20, 17))
        # Past date rolls a year forward — at 17:00, not 05:00.
        self.assertEqual(parse_deadline("by 5 Oct", FRI),
                         _ts(2027, 10, 5, 17))
        # Control: the "on" form was already correct.
        self.assertEqual(parse_deadline("on 20 Oct", FRI),
                         _ts(2026, 10, 20, 17))

    def test_extraction_uses_time_clause_and_cleans_what(self):
        out = extract_commitments(_msg("I will send it by 5pm"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].deadline, _ts(2026, 10, 16, 17))
        self.assertNotIn("5pm", out[0].what)


# ── 3. hedge contraction consistency ─────────────────────────────────

class TestCheckHedgeConsistency(unittest.TestCase):
    def test_check_is_a_hedge_in_both_forms(self):
        self.assertEqual(
            extract_commitments(_msg("I'll check the figures by Friday")),
            [])
        self.assertEqual(
            extract_commitments(_msg("I will check the figures by Friday")),
            [])
        self.assertEqual(extract_commitments(_msg("I will check")), [])

    def test_real_promise_still_extracts(self):
        out = extract_commitments(
            _msg("I will send you the figures by Friday"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].deadline, _ts(2026, 10, 16, 17))


# ── 4. preferences from_dict restores the signal log ─────────────────

class TestPreferencesFromDictSignals(unittest.TestCase):
    def test_round_trip_keeps_signals(self):
        store = PreferenceStore()
        record_signal(store, "music", {"energy": "high"}, True,
                      source="unit")
        record_signal(store, "music", {"energy": "low"}, False,
                      source="unit")
        raw = store.to_dict()
        self.assertEqual(len(raw["signals"]), 2)
        restored = PreferenceStore.from_dict(raw)
        again = restored.to_dict()
        self.assertEqual(again["weights"], raw["weights"])
        self.assertEqual(len(again["signals"]), 2)
        first = again["signals"][0]
        self.assertEqual(first["domain"], "music")
        self.assertEqual(first["approve"], True)
        self.assertEqual(first["delta"], raw["signals"][0]["delta"])
        self.assertEqual(first["source"], "unit")
        self.assertEqual(first["created_at"],
                         raw["signals"][0]["created_at"])
        self.assertEqual(first["item"], {"energy": "high"})


# ── durability pins ──────────────────────────────────────────────────

class TestPeopleStoreRestart(unittest.TestCase):
    def test_reopen_preserves_everything(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "people.db")
            store = PersonStore(path)
            store.ensure("ada", "Ada", at=1000.0)
            store.ensure("bola", "Bola", at=1000.0)
            store.record_inbound("ada", topic="rent", at=2000.0)
            store.add_note("ada", "likes tea")
            store.add_link("ada", "bola", "friend")
            store.add_gift("ada", "Book", "given", value_minor=500,
                           currency="NGN", date="2026-10-01")
            store.link_obligation("ada", "cm-1", "send the deck",
                                  deadline="2026-10-20", status="open")
            before = store.get("ada")
            store.close()
            reopened = PersonStore(path)
            try:
                self.assertEqual(reopened.get("ada"), before)
                self.assertEqual(len(reopened.gifts_for("ada")), 1)
                self.assertEqual(len(reopened.obligations_for("ada")), 1)
                self.assertEqual(len(reopened.links_for("ada")), 1)
            finally:
                reopened.close()


class TestHouseholdRestart(unittest.TestCase):
    def test_shared_records_survive_reopen(self):
        with tempfile.TemporaryDirectory() as td:
            registry = family_registry(td)
            registry.call(
                "family_person_add", actor_id="owner1", actor_role="owner",
                person_id="owner1", display_name="Owner", role="owner")
            registry.call(
                "family_person_add", actor_id="owner1", actor_role="owner",
                person_id="adult1", display_name="Adult", role="adult")
            household = Household(registry=registry)
            entry = household.add_shared_commitment(
                "owner1", "Pay the rent", owner_person="adult1",
                due=1_800_000_000.0)
            household.add_shared_asset("owner1", "Family car")
            household.close()
            reopened = Household(registry=family_registry(td))
            try:
                rows = reopened.list_shared_commitments("adult1")
                self.assertEqual([r["id"] for r in rows], [entry["id"]])
                self.assertEqual(
                    [a["name"] for a in reopened.list_shared_assets("adult1")],
                    ["Family car"])
            finally:
                reopened.close()


if __name__ == "__main__":
    unittest.main()
