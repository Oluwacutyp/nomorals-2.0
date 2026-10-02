"""Phase 4 tests: PracticeGuide, session scripts, safety framing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.wisdom import (PracticeGuide, PracticeError, SAFETY_TEXT,
                             WisdomError, WisdomKeeper)


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-practice-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


class FakeClock:
    """Injectable clock: records sleeps, never actually waits."""

    def __init__(self, now=1700000000.0):
        self._now = now
        self.slept = 0.0
        self.ticks: list[float] = []

    def sleep(self, seconds):
        self.slept += float(seconds)
        self.ticks.append(float(seconds))

    def now(self):
        return self._now


def _run(guide, session_id, **kw):
    lines: list[str] = []
    clock = kw.pop("clock", None) or FakeClock()
    result = guide.run(session_id, clock=clock, out=lines.append, **kw)
    return result, lines, clock


EXPECTED_IDS = {
    "four-seven-eight", "box-breathing", "nadi-shodhana",
    "kapalabhati", "astral-prep", "void-state",
}


class SessionDataTests(unittest.TestCase):
    def test_all_six_sessions_load(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        self.assertEqual({s["id"] for s in guide.list_sessions()}, EXPECTED_IDS)

    def test_list_sessions_shape_and_totals(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        by_id = {s["id"]: s for s in guide.list_sessions()}
        for s in by_id.values():
            self.assertTrue(s["name"])
            self.assertTrue(s["description"])
            self.assertIsInstance(s["beginner"], bool)
            self.assertGreater(s["total_seconds"], 0)
        # 4-7-8: (4+7+8)s x 4 cycles = 76s
        self.assertEqual(by_id["four-seven-eight"]["total_seconds"], 76)
        # box: 16s x 5 rounds = 80s
        self.assertEqual(by_id["box-breathing"]["total_seconds"], 80)

    def test_beginner_flags(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        by_id = {s["id"]: s for s in guide.list_sessions()}
        # the owner's entry point
        self.assertTrue(by_id["four-seven-eight"]["beginner"])
        self.assertTrue(by_id["box-breathing"]["beginner"])
        self.assertTrue(by_id["nadi-shodhana"]["beginner"])
        # kapalabhati is vigorous: NOT beginner
        self.assertFalse(by_id["kapalabhati"]["beginner"])

    def test_astral_prep_framed_as_relaxation_not_guarantee(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        desc = [s for s in guide.list_sessions()
                if s["id"] == "astral-prep"][0]["description"]
        self.assertIn("does not guarantee", desc)
        self.assertIn("relaxation", desc)

    def test_unknown_session_raises_listing_available(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        with self.assertRaises(PracticeError) as cm:
            guide.list_sessions() and guide.run("nope", clock=FakeClock())
        msg = str(cm.exception)
        self.assertIn("unknown session 'nope'", msg)
        for sid in EXPECTED_IDS:
            self.assertIn(sid, msg)

    def test_malformed_session_json_rejected(self):
        tmp = tempfile.mkdtemp(prefix="wisdom-bad-sessions-")
        Path(tmp, "broken.json").write_text("{not valid json",
                                            encoding="utf-8")
        ctx, _ = _ctx()
        with self.assertRaises(PracticeError):
            PracticeGuide(ctx, sessions_dir=tmp)

    def test_missing_key_in_session_rejected(self):
        tmp = tempfile.mkdtemp(prefix="wisdom-bad-sessions-")
        Path(tmp, "incomplete.json").write_text(
            json.dumps({"id": "x", "name": "X"}), encoding="utf-8")
        ctx, _ = _ctx()
        with self.assertRaises(PracticeError):
            PracticeGuide(ctx, sessions_dir=tmp)

    def test_practice_error_is_wisdom_error(self):
        self.assertTrue(issubclass(PracticeError, WisdomError))

    def test_phase_instructions_short_and_chat_ready(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        for sid in EXPECTED_IDS:
            for msg in guide.phases_for_chat(sid):
                self.assertLessEqual(len(msg), 200, msg)
                self.assertTrue(msg.strip())


class RunTests(unittest.TestCase):
    def test_run_sequences_phases_with_fake_clock(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        result, lines, clock = _run(guide, "four-seven-eight")
        self.assertTrue(result["completed"])
        self.assertEqual(result["session_id"], "four-seven-eight")
        # 3 phases x repeat 4 = 12 phase executions
        self.assertEqual(result["phases_done"], 12)
        # (4+7+8)s x 4 cycles of counted time
        self.assertEqual(clock.slept, 76)
        phase_lines = [l for l in lines if l.startswith("[")]
        self.assertEqual(len(phase_lines), 12)
        self.assertIn("[1/12] Inhale", phase_lines[0])
        self.assertIn("[5/12] Hold", phase_lines[4])
        self.assertIn("[9/12] Exhale", phase_lines[8])
        self.assertIn("  4…", lines)  # countdown tick printed

    def test_run_prints_safety_text_first(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        _, lines, _ = _run(guide, "box-breathing")
        self.assertEqual(lines[0], "=== Box Breathing 4-4-4-4 ===")
        self.assertEqual(lines[1], SAFETY_TEXT)

    def test_safety_text_has_all_five_qualifications(self):
        for phrase in (
                "parasympathetic", "does NOT guarantee",
                "astral projection", "kundalini awakening",
                "lightheaded, dizzy", "clinician",
                "not medical or mental-health care"):
            self.assertIn(phrase, SAFETY_TEXT)
        ctx, _ = _ctx()
        self.assertEqual(PracticeGuide(ctx).safety_text(), SAFETY_TEXT)

    def test_run_logs_completion_to_jsonl(self):
        ctx, tmp = _ctx()
        guide = PracticeGuide(ctx)
        result, _, _ = _run(guide, "nadi-shodhana")
        log = Path(tmp, "wisdom", "practice_log.jsonl")
        self.assertTrue(log.is_file())
        entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        self.assertEqual(entry["session_id"], "nadi-shodhana")
        self.assertTrue(entry["completed"])
        self.assertEqual(entry["notes"], "")
        self.assertEqual(entry["started_at"], result["started_at"])

    def test_run_rounds_param(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        result, _, clock = _run(guide, "box-breathing", rounds=2)
        # 4 phases x 5 repeats x 2 rounds
        self.assertEqual(result["phases_done"], 40)
        self.assertEqual(clock.slept, 160)

    def test_run_bad_rounds_rejected(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        for bad in (0, -1, "2", 1.5, True):
            with self.assertRaises(PracticeError, msg=f"rounds={bad!r}"):
                guide.run("box-breathing", clock=FakeClock(),
                          out=lambda l: None, rounds=bad)

    def test_no_raw_time_sleep_in_session_logic(self):
        src = Path(__file__).resolve().parent.parent / "nomorals" / "wisdom" \
            / "practice.py"
        text = src.read_text(encoding="utf-8")
        self.assertEqual(text.count("time.sleep("), 1,
                         "time.sleep may only appear in RealClock.sleep")

    def test_phases_for_chat_box(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        msgs = guide.phases_for_chat("box-breathing")
        self.assertEqual(len(msgs), 20)  # 4 phases x 5 repeats
        self.assertIn("Inhale (4s):", msgs[0])
        for m in msgs:
            self.assertLessEqual(len(m), 200)
        with self.assertRaises(PracticeError):
            guide.phases_for_chat("nope")


class JournalHistoryTests(unittest.TestCase):
    def test_journal_stores_notes(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        _run(guide, "four-seven-eight")
        entry = guide.journal("four-seven-eight",
                              "Felt calmer after round 3.")
        self.assertEqual(entry["type"], "journal")
        self.assertEqual(entry["session_id"], "four-seven-eight")
        self.assertEqual(entry["notes"], "Felt calmer after round 3.")
        hist = guide.history()
        self.assertEqual(hist[0]["type"], "journal")
        self.assertIn("calmer", hist[0]["notes"])

    def test_journal_empty_notes_rejected(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        for bad in ("", "   ", None):
            with self.assertRaises(PracticeError, msg=f"notes={bad!r}"):
                guide.journal("box-breathing", bad)
        # nothing stored
        self.assertEqual(guide.history(), [])

    def test_journal_unknown_session_rejected(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        with self.assertRaises(PracticeError):
            guide.journal("nope", "some notes")

    def test_history_recent_newest_first_with_limit(self):
        ctx, _ = _ctx()
        guide = PracticeGuide(ctx)
        _run(guide, "four-seven-eight")
        _run(guide, "box-breathing")
        guide.journal("box-breathing", "steady")
        hist = guide.history(limit=2)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["type"], "journal")
        self.assertEqual(hist[1]["session_id"], "box-breathing")
        full = guide.history()
        self.assertEqual(len(full), 3)
        with self.assertRaises(PracticeError):
            guide.history(limit=0)

    def test_history_empty_when_no_log(self):
        ctx, _ = _ctx()
        self.assertEqual(PracticeGuide(ctx).history(), [])


class KeeperPracticeTests(unittest.TestCase):
    def test_keeper_lazy_practice_property(self):
        ctx, _ = _ctx()
        keeper = WisdomKeeper(ctx)
        self.assertIsInstance(keeper.practice, PracticeGuide)
        # same object on second access
        self.assertIs(keeper.practice, keeper.practice)
        self.assertEqual(len(keeper.practice.list_sessions()), 6)


if __name__ == "__main__":
    unittest.main()
