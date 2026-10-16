"""Tests for MORE_TO_ADD gaps 1–3: commitment ledger, decision
journal, focus sessions + weekly review.

Invariants under test:

* A commitment is extracted ONLY from explicit promise text — hedges,
  questions, third-person statements, past tense, and "I'll think
  about it" yield nothing.
* Reliability is computed from the ledger; no resolved data → rate is
  ``None``, never 0%.
* Decision calibration is honest: no data → accuracy/brier/sentence
  are ``None``, never 0%.
* The weekly review includes open commitments + decisions, and never
  invents health/learn/finance content when no summarizer is wired.
"""

from __future__ import annotations

import ast
import datetime as _dt
import tempfile
import unittest
from pathlib import Path

from nomorals.commitments.ledger import (
    CommitmentLedger,
    extract_commitments,
    format_deadline,
    nudge_text,
    parse_deadline,
)
from nomorals.decisions.journal import DecisionJournal
from nomorals.focus.sessions import (
    FocusTimer,
    build_weekly_review,
    format_duration,
    week_bounds,
)

UTC = _dt.timezone.utc


def _ts(year: int, month: int, day: int, hour: int = 12,
        minute: int = 0) -> float:
    return _dt.datetime(year, month, day, hour, minute,
                        tzinfo=UTC).timestamp()


# Friday 2026-10-16 12:00 UTC — the fixed "now" for deadline math.
FRI = _ts(2026, 10, 16, 12)
MON = _ts(2026, 10, 12, 9)   # Monday of the same week, 09:00
WED = _ts(2026, 10, 14, 15)  # Wednesday of the same week
THU = _ts(2026, 10, 15, 10)  # Thursday of the same week
SAT = _ts(2026, 10, 17, 10)  # Saturday of the same week


def _msg(text: str, ts: float = FRI, **extra):
    base = {"text": text, "sender": "owner", "timestamp": ts, "id": "m1"}
    base.update(extra)
    return base


class TestCommitmentExtraction(unittest.TestCase):
    def test_explicit_promise_extracted(self):
        out = extract_commitments(_msg(
            "I will send you the deck by Tuesday", recipient="Chidi"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].promisee, "Chidi")
        self.assertIn("send", out[0].what)
        self.assertEqual(
            out[0].deadline, _ts(2026, 10, 20, 17))  # next Tuesday 17:00

    def test_nudge_text_exact(self):
        out = extract_commitments(_msg(
            "I will send you the deck by Tuesday", recipient="Chidi"))
        self.assertEqual(
            nudge_text(out[0], FRI),
            "⏰ You told Chidi you'd send the deck by Tuesday.")

    def test_contraction_extracted(self):
        out = extract_commitments(_msg(
            "I'll call you tomorrow", recipient="Ada"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].deadline, _ts(2026, 10, 17, 17))

    def test_hedges_yield_nothing(self):
        hedged = [
            "I might send you the deck",
            "I could send you the deck",
            "I'll try to send you the deck",
            "I will try to send you the deck",
            "I should probably send you the deck",
            "I wish I could send you the deck",
            "I will send it if I have time",
            "Maybe I'll send you the deck",
            "I will think about it",
            "I plan to send you the deck",
        ]
        for text in hedged:
            with self.subTest(text=text):
                self.assertEqual(extract_commitments(_msg(text)), [])

    def test_questions_third_person_past_yield_nothing(self):
        for text in [
            "Will you send me the deck?",
            "He will send you the deck",
            "She will call you tomorrow",
            "I sent you the deck",
            "I already sent you the deck",
        ]:
            with self.subTest(text=text):
                self.assertEqual(extract_commitments(_msg(text)), [])

    def test_weekday_noun_is_not_a_deadline(self):
        out = extract_commitments(_msg(
            "I will send you the Tuesday report", recipient="Chidi"))
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].deadline)

    def test_by_clause_beats_noun_weekday(self):
        out = extract_commitments(_msg(
            "I will send you the Tuesday report by Friday"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].deadline, _ts(2026, 10, 16, 17))

    def test_next_weekday_and_relative(self):
        out = extract_commitments(_msg("I will send it by next Tuesday"))
        self.assertEqual(out[0].deadline, _ts(2026, 10, 27, 17))
        out = extract_commitments(_msg("I will finish it in 3 days"))
        self.assertEqual(out[0].deadline, _ts(2026, 10, 19, 17))
        out = extract_commitments(_msg("I will pay you on 20 Oct 2026"))
        self.assertEqual(out[0].deadline, _ts(2026, 10, 20, 17))

    def test_pidgin_and_french_extracted(self):
        out = extract_commitments(_msg(
            "I go send you the money tomorrow", recipient="Tunde"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].deadline, _ts(2026, 10, 17, 17))
        out = extract_commitments(_msg(
            "Je promets de t'envoyer le rapport demain"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].language, "fr")
        self.assertEqual(out[0].deadline, _ts(2026, 10, 17, 17))

    def test_owe_you_extracted(self):
        out = extract_commitments(_msg(
            "I owe you $50 by Friday", recipient="Chidi"))
        self.assertEqual(len(out), 1)
        self.assertEqual(nudge_text(out[0], FRI),
                         "⏰ You told Chidi you'd give $50 by Friday.")

    def test_deadline_parser_standalone(self):
        self.assertEqual(parse_deadline("by Tuesday", FRI),
                         _ts(2026, 10, 20, 17))
        self.assertIsNone(parse_deadline("the Tuesday report", FRI))
        self.assertIsNone(parse_deadline("no date here", FRI))

    def test_format_deadline(self):
        self.assertEqual(format_deadline(_ts(2026, 10, 20, 17), FRI),
                         "Tuesday")
        self.assertEqual(format_deadline(_ts(2026, 11, 3, 17), FRI),
                         "Tuesday, 3 Nov")
        self.assertEqual(format_deadline(None), "no deadline set")


class TestCommitmentLedger(unittest.TestCase):
    def test_extract_and_record_idempotent(self):
        ledger = CommitmentLedger()
        msg = _msg("I will send you the deck by Tuesday",
                   recipient="Chidi")
        first = ledger.extract_and_record([msg])
        second = ledger.extract_and_record([msg])
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(len(ledger.commitments), 1)

    def test_lifecycle_and_append_only(self):
        ledger = CommitmentLedger()
        commitment = ledger.add(promisee="Chidi", what="send the deck",
                                deadline=_ts(2026, 10, 20, 17))
        ledger.mark_kept(commitment.id, evidence="sent via email")
        self.assertEqual(ledger.commitments[commitment.id].status, "kept")
        with self.assertRaises(ValueError):
            ledger.mark_kept(commitment.id)
        with self.assertRaises(KeyError):
            ledger.mark_kept("cm-nope")
        # The created entry is still the original one — never rewritten.
        created = [e for e in ledger.entries if e.kind == "created"]
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0].payload["status"], "open")
        self.assertTrue(ledger.verify()["ok"])

    def test_correction_appends_never_rewrites(self):
        ledger = CommitmentLedger()
        commitment = ledger.add(promisee="Chidi", what="send the deck",
                                deadline=_ts(2026, 10, 20, 17))
        ledger.correct(commitment.id,
                       fields={"deadline": _ts(2026, 10, 21, 17)},
                       reason="deadline was Wednesday, not Tuesday")
        self.assertEqual(ledger.commitments[commitment.id].deadline,
                         _ts(2026, 10, 21, 17))
        kinds = [e.kind for e in ledger.entries]
        self.assertEqual(kinds, ["created", "correction"])
        self.assertTrue(ledger.verify()["ok"])

    def test_supersede_keeps_old(self):
        ledger = CommitmentLedger()
        old = ledger.add(promisee="Chidi", what="send the deck",
                         deadline=_ts(2026, 10, 20, 17))
        new = ledger.supersede(old.id, what="send the deck",
                               deadline=_ts(2026, 10, 22, 17),
                               reason="Chidi asked for Thursday")
        self.assertEqual(ledger.commitments[old.id].status, "superseded")
        self.assertEqual(new.status, "open")
        self.assertTrue(ledger.verify()["ok"])

    def test_reliability_from_ledger(self):
        ledger = CommitmentLedger()
        self.assertIsNone(ledger.reliability()["rate"])  # no data ≠ 0%
        self.assertIsNone(ledger.reliability_sentence())
        kept1 = ledger.add(promisee="Chidi", what="send the deck")
        kept2 = ledger.add(promisee="Chidi", what="call back")
        broken = ledger.add(promisee="Ada", what="pay the invoice")
        open_c = ledger.add(promisee="Chidi", what="review the draft")
        ledger.mark_kept(kept1.id)
        ledger.mark_kept(kept2.id)
        ledger.mark_broken(broken.id, evidence="missed it")
        stats = ledger.reliability()
        self.assertEqual((stats["kept"], stats["broken"]), (2, 1))
        self.assertAlmostEqual(stats["rate"], 2 / 3)
        self.assertEqual(stats["open"], 1)
        chidi = ledger.reliability("Chidi")
        self.assertEqual((chidi["kept"], chidi["broken"]), (2, 0))
        self.assertEqual(chidi["rate"], 1.0)
        self.assertEqual(ledger.reliability_sentence("Chidi"),
                         "You've kept 2 of 2 promises to Chidi (100%).")
        self.assertEqual(open_c.status, "open")

    def test_reliability_by_person(self):
        ledger = CommitmentLedger()
        a = ledger.add(promisee="Chidi", what="one")
        b = ledger.add(promisee="Ada", what="two")
        ledger.mark_kept(a.id)
        ledger.mark_broken(b.id)
        by_person = ledger.reliability_by_person()
        self.assertEqual(by_person["Chidi"]["rate"], 1.0)
        self.assertEqual(by_person["Ada"]["rate"], 0.0)

    def test_overdue_due_soon_nudges(self):
        ledger = CommitmentLedger()
        soon = ledger.add(promisee="Ada", what="call back",
                          deadline=_ts(2026, 10, 17, 17))
        late = ledger.add(promisee="Chidi", what="send the deck",
                          deadline=_ts(2026, 10, 13, 17))
        far = ledger.add(promisee="Ada", what="review the draft",
                         deadline=_ts(2026, 11, 3, 17))
        self.assertEqual([c.id for c in ledger.overdue(SAT)], [late.id])
        nudged = ledger.nudges(within_seconds=48 * 3600, now=SAT)
        self.assertEqual({c.id for c, _ in nudged}, {soon.id, late.id})
        overdue_text = dict((c.id, t) for c, t in nudged)[late.id]
        self.assertIn("Overdue", overdue_text)
        self.assertIn("3 days late", overdue_text)
        # Nudging is audited: a reminded entry per nudge.
        reminded = [e for e in ledger.entries if e.kind == "reminded"]
        self.assertEqual(len(reminded), 2)
        self.assertEqual(far.status, "open")

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ledger.json")
            ledger = CommitmentLedger(path)
            commitment = ledger.add(promisee="Chidi", what="send the deck")
            ledger.mark_kept(commitment.id, evidence="sent")
            reloaded = CommitmentLedger(path)
            self.assertEqual(reloaded.commitments[commitment.id].status,
                             "kept")
            self.assertTrue(reloaded.verify()["ok"])


class TestDecisionJournal(unittest.TestCase):
    def test_no_data_is_none_not_zero(self):
        journal = DecisionJournal()
        stats = journal.calibration()
        self.assertIsNone(stats["accuracy"])
        self.assertIsNone(stats["mean_confidence"])
        self.assertIsNone(stats["brier"])
        self.assertIsNone(stats["calibration_gap"])
        self.assertIsNone(stats["verdict"])
        self.assertIsNone(journal.calibration_sentence())

    def test_calibration_accuracy_and_brier(self):
        journal = DecisionJournal()
        decisions = [
            ("Ship now", 0.9, True),
            ("Paper-trade", 0.7, True),
            ("Buy the dip", 0.8, False),
        ]
        for title, confidence, success in decisions:
            entry = journal.record(decision=title, chosen="go",
                                   confidence=confidence, ts=FRI)
            journal.resolve(entry.id, outcome="done", success=success,
                            ts=SAT)
        stats = journal.calibration()
        self.assertAlmostEqual(stats["accuracy"], 2 / 3)
        self.assertAlmostEqual(stats["mean_confidence"], 0.8)
        self.assertAlmostEqual(
            stats["brier"],
            ((0.9 - 1) ** 2 + (0.7 - 1) ** 2 + (0.8 - 0) ** 2) / 3)
        self.assertEqual(stats["verdict"], "overconfident")
        sentence = journal.calibration_sentence()
        self.assertIn("67% accurate", sentence)
        self.assertIn("3 resolved decisions", sentence)

    def test_buckets_show_where_judgement_is_off(self):
        journal = DecisionJournal()
        entry = journal.record(decision="High confidence call",
                               chosen="yes", confidence=0.95, ts=FRI)
        journal.resolve(entry.id, outcome="wrong", success=False, ts=SAT)
        buckets = {b["bucket"]: b for b in
                   journal.calibration()["buckets"]}
        self.assertEqual(buckets["90–100%"]["n"], 1)
        self.assertEqual(buckets["90–100%"]["hit_rate"], 0.0)
        self.assertIsNone(buckets["50–60%"]["hit_rate"])

    def test_outcome_revisions_append_only(self):
        journal = DecisionJournal()
        entry = journal.record(decision="Launch", chosen="go", ts=FRI)
        journal.resolve(entry.id, outcome="looks good", success=True,
                        ts=SAT)
        journal.resolve(entry.id, outcome="actually it flopped",
                        success=False, note="corrected", ts=SAT + 3600)
        self.assertEqual(len(entry.history), 2)
        self.assertEqual(entry.outcome, "actually it flopped")
        self.assertIs(entry.outcome_success, False)

    def test_resolve_without_verdict_excluded_from_accuracy(self):
        journal = DecisionJournal()
        entry = journal.record(decision="Try a new venue", chosen="yes",
                               confidence=0.6, ts=FRI)
        journal.resolve(entry.id, outcome="too early to tell",
                        success=None, ts=SAT)
        stats = journal.calibration()
        self.assertEqual(stats["n_resolved"], 1)
        self.assertEqual(stats["n_with_verdict"], 0)
        self.assertIsNone(stats["accuracy"])
        self.assertIsNone(journal.calibration_sentence())

    def test_invalid_confidence_rejected(self):
        journal = DecisionJournal()
        with self.assertRaises(ValueError):
            journal.record(decision="Bad", confidence=1.5)
        with self.assertRaises(KeyError):
            journal.resolve("dc-nope", outcome="x")

    def test_review_weekly(self):
        journal = DecisionJournal()
        old = journal.record(decision="Old call", chosen="yes",
                             confidence=0.8, ts=_ts(2026, 9, 20))
        journal.resolve(old.id, outcome="won", success=True,
                        ts=_ts(2026, 9, 21))
        recent = journal.record(decision="This week call", chosen="no",
                                confidence=0.6, ts=THU)
        journal.resolve(recent.id, outcome="lost", success=False,
                        ts=FRI - 3600)
        pending = journal.record(decision="Still open", chosen="maybe",
                                 ts=FRI)
        review = journal.review("weekly", now=FRI)
        self.assertEqual(len(review["resolved_this_period"]), 1)
        self.assertEqual(review["resolved_this_period"][0]["decision"],
                         "This week call")
        self.assertEqual(len(review["open_decisions"]), 1)
        self.assertEqual(review["open_decisions"][0]["decision"],
                         "Still open")
        self.assertIn("0% accurate", review["calibration_sentence"])
        monthly = journal.review("monthly", now=FRI)
        self.assertEqual(len(monthly["resolved_this_period"]), 2)

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "journal.json")
            journal = DecisionJournal(path)
            entry = journal.record(decision="Launch", chosen="go",
                                   confidence=0.7, ts=FRI)
            journal.resolve(entry.id, outcome="won", success=True, ts=SAT)
            reloaded = DecisionJournal(path)
            self.assertEqual(reloaded.get(entry.id).status, "resolved")
            self.assertIs(reloaded.get(entry.id).outcome_success, True)


class TestFocusSessions(unittest.TestCase):
    def test_pause_math(self):
        timer = FocusTimer()
        session = timer.start("codebeast training", ts=MON)
        timer.pause(ts=MON + 600)    # 10 min in
        timer.resume(ts=MON + 900)   # 5 min pause
        timer.stop(ts=MON + 1500, note="smoke test green")
        self.assertEqual(session.status, "done")
        self.assertEqual(session.elapsed(), 1200)  # 25 min − 5 paused
        self.assertEqual(session.notes, ["smoke test green"])

    def test_interruption_logged(self):
        timer = FocusTimer()
        session = timer.start("writing", ts=MON)
        timer.interrupt(note="phone call")
        timer.stop(ts=MON + 1800)
        self.assertEqual(session.interruptions, 1)
        self.assertIn("interruption: phone call", session.notes)

    def test_double_start_and_bad_transitions_fail_fast(self):
        timer = FocusTimer()
        timer.start("one", ts=MON)
        with self.assertRaises(ValueError):
            timer.start("two", ts=MON + 10)
        timer.stop(ts=MON + 60)
        with self.assertRaises(ValueError):
            timer.pause()
        with self.assertRaises(ValueError):
            timer.resume()
        with self.assertRaises(ValueError):
            timer.stop()

    def test_summary_window(self):
        timer = FocusTimer()
        session = timer.start("deep work", ts=MON)
        timer.stop(ts=MON + 3600)
        start, end = week_bounds(FRI)
        summary = timer.summary(start, end, now=FRI)
        self.assertEqual(summary["sessions"], 1)
        self.assertEqual(summary["total_seconds"], 3600)
        self.assertEqual(summary["by_label"][0]["label"], "deep work")
        # A window before the session sees nothing.
        empty = timer.summary(_ts(2026, 10, 1), _ts(2026, 10, 5),
                              now=FRI)
        self.assertEqual(empty["sessions"], 0)

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "focus.json")
            timer = FocusTimer(path)
            session = timer.start("deep work", ts=MON)
            timer.stop(ts=MON + 3600)
            reloaded = FocusTimer(path)
            self.assertEqual(reloaded.sessions[session.id].status, "done")
            self.assertEqual(reloaded.sessions[session.id].elapsed(), 3600)

    def test_format_duration(self):
        self.assertEqual(format_duration(7380), "2h 03m")
        self.assertEqual(format_duration(900), "15m")
        self.assertEqual(format_duration(45), "45s")


class TestWeeklyReview(unittest.TestCase):
    def _build(self):
        ledger = CommitmentLedger()
        kept = ledger.add(promisee="Chidi", what="send the deck",
                          deadline=_ts(2026, 10, 20, 17), ts=MON)
        ledger.mark_kept(kept.id, evidence="sent", ts=WED)
        ledger.add(promisee="Ada", what="review the draft",
                   deadline=_ts(2026, 10, 21, 17), ts=FRI)
        journal = DecisionJournal()
        won = journal.record(decision="Paper-trade first", chosen="yes",
                             confidence=0.7, ts=THU)
        journal.resolve(won.id, outcome="found two bugs", success=True,
                        ts=SAT)
        journal.record(decision="Hire or wait", chosen="undecided", ts=FRI)
        timer = FocusTimer()
        session = timer.start("codebeast training", ts=MON)
        timer.stop(ts=MON + 3600)
        return ledger, journal, timer

    def test_review_includes_open_commitments_and_decisions(self):
        ledger, journal, timer = self._build()
        review = build_weekly_review(
            commitments=ledger, decisions=journal, focus=timer, now=FRI,
            summarizers={
                "health": lambda: "Ran 5km, slept 7h",
                "learn": lambda: {"topic": "Yoruba", "minutes": 30},
                "finance": lambda: None,
            })
        # Wins: kept promise + successful decision, both this week.
        self.assertEqual(len(review["wins"]), 2)
        self.assertTrue(any("send the deck" in w for w in review["wins"]))
        self.assertTrue(any("Paper-trade" in w for w in review["wins"]))
        # Open commitments + open decisions are the review's payload.
        self.assertEqual(len(review["open_commitments"]), 1)
        self.assertEqual(review["open_commitments"][0]["what"],
                         "review the draft")
        self.assertEqual(len(review["decisions_open"]), 1)
        self.assertEqual(review["decisions_open"][0]["decision"],
                         "Hire or wait")
        self.assertEqual(len(review["decisions_resolved"]), 1)
        self.assertIn("100% accurate",
                      review["calibration_sentence"])
        # Focus totals come from the timer.
        self.assertEqual(review["focus_summary"]["sessions"], 1)
        self.assertEqual(review["focus_summary"]["total_seconds"], 3600)
        # Summarizers: str, dict, and None (honest "no data").
        self.assertEqual(review["health_summary"], "Ran 5km, slept 7h")
        self.assertIn("Yoruba", review["learn_summary"])
        self.assertIn("No data", review["finance_summary"])
        # Next week carries the open items + continues the top label.
        plan = review["next_week_plan"]
        self.assertTrue(any("Keep: review the draft" in p for p in plan))
        self.assertTrue(any("Decide: Hire or wait" in p for p in plan))
        self.assertTrue(any("codebeast training" in p for p in plan))
        # Rendered markdown contains every section.
        rendered = review.render_markdown()
        for section in ("## Wins", "## Open commitments", "## Decisions",
                        "## Focus", "## Health", "## Learn", "## Finance",
                        "## Next week"):
            self.assertIn(section, rendered)
        self.assertIn("Ran 5km", rendered)

    def test_review_honest_when_empty(self):
        review = build_weekly_review(now=FRI)
        self.assertEqual(review["wins"], [])
        self.assertEqual(review["open_commitments"], [])
        self.assertEqual(review["decisions_open"], [])
        self.assertIsNone(review["calibration_sentence"])
        self.assertIn("not connected", review["health_summary"])
        self.assertIn("not connected", review["learn_summary"])
        self.assertIn("not connected", review["finance_summary"])
        rendered = review.render_markdown()
        self.assertIn("No recorded wins", rendered)
        self.assertIn("Nothing owed to anyone", rendered)

    def test_review_accepts_plain_lists_and_dicts(self):
        review = build_weekly_review(
            commitments=[{"id": "c1", "promisee": "Chidi",
                          "what": "send the deck", "deadline": None,
                          "status": "open", "created_at": FRI,
                          "closed_at": None}],
            decisions=[{"id": "d1", "decision": "Ship or wait",
                        "chosen": "wait", "outcome": "", "success": None,
                        "confidence": None, "status": "open",
                        "created_at": FRI, "resolved_at": None}],
            focus=[],
            now=FRI)
        self.assertEqual(len(review["open_commitments"]), 1)
        self.assertEqual(len(review["decisions_open"]), 1)

    def test_review_summarizer_failure_is_honest(self):
        def _boom():
            raise RuntimeError("provider down")
        review = build_weekly_review(now=FRI,
                                     summarizers={"health": _boom})
        self.assertIn("unavailable", review["health_summary"])

    def test_review_overdue_flagged(self):
        ledger = CommitmentLedger()
        ledger.add(promisee="Chidi", what="send the deck",
                   deadline=_ts(2026, 10, 13, 17), ts=MON)
        review = build_weekly_review(commitments=ledger, now=FRI)
        self.assertIn("OVERDUE",
                      review["open_commitments"][0]["deadline_label"])
        self.assertTrue(any("OVERDUE" in p
                            for p in review["next_week_plan"]))


class TestNewModulesAreStdlibOnly(unittest.TestCase):
    """Layering: the new packages sit at layer 0, so they must not
    import ``nomorals`` at all — stdlib only."""

    def test_no_nomorals_imports(self):
        root = Path(__file__).resolve().parent.parent / "nomorals"
        for rel in ("commitments/ledger.py", "decisions/journal.py",
                    "focus/sessions.py", "commitments/__init__.py",
                    "decisions/__init__.py", "focus/__init__.py"):
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name.startswith("nomorals"), rel)
                elif isinstance(node, ast.ImportFrom):
                    self.assertTrue(
                        node.level == 0 and not (node.module or "").startswith("nomorals")
                        or node.level > 0, rel)


if __name__ == "__main__":
    unittest.main()
