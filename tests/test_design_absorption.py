"""Design absorption #1 + #4 (2026-11-09) — chat presentation.

#1 — the long-job status grammar in the shared live-progress
organ (``social/chat/progress.ProgressReporter``): a TITLED
reporter renders its one live message as glyph + title + total
elapsed, a step trace (completed steps fold to ✓ lines, the
running step carries the working glyph and its OWN elapsed), and
finish/fail COLLAPSE the trace into one terminal line.  An
untitled reporter keeps the historical verbatim behaviour
(pinned here and in test_hub).

#4 — confirm-with-visible-expiry for the staged confirm gate:
the prompt is born carrying a countdown rendered from the
pending's REAL ``expires_at`` (the gate's CONFIRM_TTL_S store —
no second timer), the send channel registers the delivered
prompt, and the runtime's countdown tick edits the countdown
down as it burns, settles the message on resolution, and
finalizes expiry (pending popped, "did not run", no buttons).

All clocks are injected; the ticker THREAD is exercised only
through its start/stop lifecycle, never through sleeps.
"""

from __future__ import annotations

import time
import unittest
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat import confirm_live
from nomorals.social.chat.base import SendResult
from nomorals.social.chat.control import CONFIRM_TTL_S
from nomorals.social.chat.progress import (
    GLYPH_DONE,
    GLYPH_FAILED,
    MAX_TRACE_STEPS,
    SPIN_FRAMES,
    ProgressReporter,
    format_elapsed,
)
from nomorals.social.chat.progress_throttle import ProgressThrottler
from nomorals.storage.db import Database

CHAT = "telegram:1"


def _titled(seen, clock, title="🌐 Proxy lab"):
    return ProgressReporter(
        seen.append, title=title, clock=lambda: clock[0],
        throttler=ProgressThrottler(min_interval_s=0.0,
                                    clock=lambda: clock[0]))


# ── #1: the status grammar ──────────────────────────────────────

class ElapsedFormatTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(format_elapsed(0), "0s")
        self.assertEqual(format_elapsed(7), "7s")
        self.assertEqual(format_elapsed(59.9), "59s")
        self.assertEqual(format_elapsed(60), "1m00s")
        self.assertEqual(format_elapsed(65), "1m05s")
        self.assertEqual(format_elapsed(3599), "59m59s")
        self.assertEqual(format_elapsed(3600), "1h00m")
        self.assertEqual(format_elapsed(3661), "1h01m")
        self.assertEqual(format_elapsed(-3), "0s")  # never negative


class LegacyProgressTests(unittest.TestCase):
    def test_untitled_stays_verbatim(self):
        seen: list[str] = []
        rep = ProgressReporter(
            seen.append,
            throttler=ProgressThrottler(min_interval_s=0.0))
        rep.milestone("step one")
        rep.count(3, 12, "testing")
        rep.fail("it broke")
        self.assertEqual(seen, ["step one",
                                "testing: 3/12 ▓▓▓░░░░░░░",
                                "it broke"])


class GrammarRenderTests(unittest.TestCase):
    def test_header_glyph_title_elapsed(self):
        clock = [100.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.milestone("scraping free lists…")
        lines = seen[-1].split("\n")
        self.assertEqual(
            lines[0], f"{SPIN_FRAMES[0]} 🌐 Proxy lab · 0s")
        self.assertEqual(
            lines[1], f"{SPIN_FRAMES[0]} scraping free lists… · 0s")

    def test_steps_fold_and_current_carries_own_elapsed(self):
        clock = [100.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.milestone("step one")
        clock[0] = 104.0
        rep.milestone("step two")
        clock[0] = 111.0
        rep.tick()
        lines = seen[-1].split("\n")
        self.assertIn("🌐 Proxy lab · 11s", lines[0])
        self.assertIn(f"{GLYPH_DONE} step one", lines)
        # The running step shows ITS elapsed (7s), not the job's.
        self.assertTrue(lines[-1].endswith("step two · 7s"), lines)

    def test_spinner_advances_per_render(self):
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.milestone("a")
        rep.milestone("b")
        rep.milestone("c")
        frames = [s.split(" ", 1)[0] for s in seen]
        self.assertEqual(frames, list(SPIN_FRAMES[:3]))

    def test_count_updates_in_place_then_graduates(self):
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.count(45, 200, "testing")
        clock[0] = 5.0
        rep.count(200, 200, "testing")
        current_lines = [ln for ln in seen[-1].split("\n")
                         if "testing" in ln]
        # One current line only — the count never duplicated.
        self.assertEqual(len(current_lines), 1)
        self.assertIn("200/200", current_lines[0])
        rep.milestone("packing")
        folded = [ln for ln in seen[-1].split("\n")
                  if "testing" in ln]
        self.assertEqual(folded,
                         [f"{GLYPH_DONE} testing: 200/200 "
                          "▓▓▓▓▓▓▓▓▓▓"])

    def test_count_without_total_is_label_only(self):
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.count(3, 0, "scraping")
        self.assertNotIn("▓", seen[-1])
        self.assertIn("scraping", seen[-1])

    def test_tick_respects_throttle(self):
        clock = [0.0]
        seen: list[str] = []
        rep = ProgressReporter(
            seen.append, title="🌐 Proxy lab", clock=lambda: clock[0],
            throttler=ProgressThrottler(min_interval_s=30.0,
                                        clock=lambda: clock[0]))
        rep.milestone("working")
        self.assertEqual(len(seen), 1)
        clock[0] = 10.0
        rep.tick()  # inside the edit interval → suppressed
        self.assertEqual(len(seen), 1)
        clock[0] = 31.0
        rep.tick()  # interval passed → the elapsed refresh lands
        self.assertEqual(len(seen), 2)
        self.assertIn("· 31s", seen[-1].split("\n")[0])

    def test_trace_cap_elides_oldest(self):
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        total = MAX_TRACE_STEPS + 5
        for i in range(total):
            rep.milestone(f"step {i}")
        rep.milestone("current")
        lines = seen[-1].split("\n")
        folded = [ln for ln in lines if ln.startswith(GLYPH_DONE)]
        self.assertEqual(len(folded), MAX_TRACE_STEPS)
        self.assertIn(f"… (+{total - MAX_TRACE_STEPS} earlier steps)",
                      lines)
        # The NEWEST steps are the visible ones.
        self.assertIn(f"{GLYPH_DONE} step {total - 1}", lines)
        self.assertNotIn(f"{GLYPH_DONE} step 0", lines)


class GrammarFinishTests(unittest.TestCase):
    def test_finish_collapses_to_one_line(self):
        clock = [100.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.milestone("step one")
        clock[0] = 104.0
        rep.milestone("step two")
        clock[0] = 131.0
        rep.finish("done")
        final = seen[-1]
        self.assertNotIn("\n", final)  # ONE line — the fold
        self.assertEqual(
            final, f"{GLYPH_DONE} 🌐 Proxy lab — done · 31s · 2 steps")

    def test_fail_collapses_with_failed_glyph(self):
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock, title="📸 shot")
        rep.milestone("opening page")
        rep.fail("couldn't shoot https://x.example")
        final = seen[-1]
        self.assertTrue(final.startswith(GLYPH_FAILED), final)
        self.assertIn("couldn't shoot", final)
        self.assertIn("· 1 step", final)

    def test_finish_strips_baked_title_and_glyph(self):
        # Legacy-shaped finish text (title prefix + ✅, the shape
        # the pre-grammar call sites produced) still renders ONE
        # clean grammar line.
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.milestone("working")
        rep.finish("🌐 Proxy lab — ✅ done")
        self.assertEqual(
            seen[-1],
            f"{GLYPH_DONE} 🌐 Proxy lab — done · 0s · 1 step")

    def test_terminal_state_is_final(self):
        clock = [0.0]
        seen: list[str] = []
        rep = _titled(seen, clock)
        rep.milestone("working")
        rep.finish("done")
        n = len(seen)
        rep.milestone("late step")
        rep.count(1, 2, "late count")
        rep.tick()
        rep.finish("again")
        rep.fail("also again")
        self.assertEqual(len(seen), n)

    def test_ticker_thread_lifecycle(self):
        seen: list[str] = []
        rep = ProgressReporter(seen.append, title="job",
                               tick_s=0.01)
        rep.milestone("working")
        self.assertIsNotNone(rep._ticker)
        self.assertTrue(rep._ticker.is_alive())
        rep.finish("done")
        rep._ticker.join(timeout=2.0)
        self.assertFalse(rep._ticker.is_alive())


# ── #4: the countdown grammar (pure) ────────────────────────────

class CountdownTextTests(unittest.TestCase):
    def test_remaining_shapes(self):
        self.assertEqual(confirm_live.format_remaining(300), "5:00")
        self.assertEqual(confirm_live.format_remaining(299.2), "5:00")
        self.assertEqual(confirm_live.format_remaining(61), "1:01")
        self.assertEqual(confirm_live.format_remaining(60), "1:00")
        self.assertEqual(confirm_live.format_remaining(10), "0:10")
        self.assertEqual(confirm_live.format_remaining(9), "9s")
        self.assertEqual(confirm_live.format_remaining(0.2), "1s")
        self.assertEqual(confirm_live.format_remaining(0), "0s")
        self.assertEqual(confirm_live.format_remaining(-5), "0s")

    def test_prompt_derives_from_the_real_expiry(self):
        now = 1_700_000_000.0
        expires = now + CONFIRM_TTL_S  # the gate's own store value
        text = confirm_live.prompt_text(
            "quit", "shut Devon down", expires, now)
        self.assertEqual(
            text,
            "⚠️ confirm: /quit — this will shut Devon down.\n"
            f"⏳ {confirm_live.format_remaining(CONFIRM_TTL_S)} "
            "left to confirm")
        # Same expiry, later "now" → less time: ONE source, the
        # stored expires_at — no independent timer anywhere.
        later = confirm_live.prompt_text(
            "quit", "shut Devon down", expires, now + 240)
        self.assertIn("⏳ 1:00 left to confirm", later)

    def test_with_countdown_replaces_line_keeps_rest(self):
        sent = ("⚠️ confirm: /quit — this will shut Devon down.\n"
                "⏳ 5:00 left to confirm\n"
                "\n<i>Powered by peacethefirst</i>")
        out = confirm_live.with_countdown(sent, "quit", 1000.0, 760.0)
        lines = out.split("\n")
        self.assertEqual(lines[1], "⏳ 4:00 left to confirm")
        self.assertEqual(lines[0], sent.split("\n")[0])
        self.assertEqual(lines[-1], "<i>Powered by peacethefirst</i>")

    def test_with_countdown_appends_when_missing(self):
        out = confirm_live.with_countdown(
            "⚠️ confirm: /quit — this will shut Devon down.",
            "quit", 1000.0, 940.0)
        self.assertTrue(out.endswith("⏳ 1:00 left to confirm"))

    def test_terminal_texts(self):
        expired = confirm_live.expired_text("forget")
        self.assertIn("did not run", expired)
        self.assertIn("/forget", expired)
        confirmed = confirm_live.settled_text("quit", "confirmed")
        self.assertIn("went ahead", confirmed)
        cancelled = confirm_live.settled_text("quit", "cancelled")
        self.assertIn("did not run", cancelled)

    def test_is_confirm_prompt(self):
        text = confirm_live.prompt_text("quit", "x", 100.0, 0.0)
        self.assertEqual(confirm_live.is_confirm_prompt(text), "quit")
        self.assertEqual(
            confirm_live.is_confirm_prompt(text, "quit"), "quit")
        self.assertEqual(
            confirm_live.is_confirm_prompt(text, "forget"), "")
        self.assertEqual(confirm_live.is_confirm_prompt("hello"), "")


# ── #4: the runtime wiring ──────────────────────────────────────

class _FakeAdapter:
    def __init__(self, fail: bool = False):
        self.edits: list = []
        self.fail = fail

    def edit_message(self, chat, message_id, text, buttons=None,
                     parse_mode=""):
        self.edits.append((message_id, text, buttons))
        if self.fail:
            return None  # the bot adapter's failure shape
        return {"ok": True}


class _FakeGateway:
    def __init__(self, adapter, platform="telegram"):
        self.adapters = {platform: adapter}
        self.sent: list = []

    def send(self, platform, chat, text, **kw):
        self.sent.append(text)
        return SendResult(ok=True, platform=platform, message_id="777")


def _live_runtime(adapter=None, platform="telegram"):
    """A bare runtime wired like test_confirm_flow's, plus the
    send-channel seams (gateway, settings, typing) the funnel
    needs — dispatch intercepted so no real quit/forget runs."""
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=Database(":memory:"))
    rt.settings = SimpleNamespace(
        partner=SimpleNamespace(typing_in_groups=False))
    rt._typing_enabled = lambda chat: False
    rt._owner_chats = {CHAT}
    rt.calls: list = []
    rt.stop = lambda: rt.calls.append(("stop",))
    rt._control_forget = lambda tail: (
        rt.calls.append(("forget", tail)) or f"forgot {tail}")
    rt.gateway = _FakeGateway(adapter or _FakeAdapter(), platform)
    return rt


def _chat(platform="telegram"):
    return SimpleNamespace(key=CHAT, platform=platform)


def _msg(chat, text):
    return SimpleNamespace(chat=chat, text=text, meta={})


def _stage_and_send(rt, command="/quit"):
    chat = _chat()
    msg = _msg(chat, command)
    reply = rt.handle_control(command, CHAT, message=msg)
    rt._send_control_reply(msg, reply)
    return chat


class ConfirmGateCountdownTests(unittest.TestCase):
    def test_gate_prompt_carries_countdown_from_stored_expiry(self):
        rt = _live_runtime()
        reply = rt.handle_control("/quit", CHAT,
                                  message=_msg(_chat(), "/quit"))
        kind, _tail, expires_at = rt._pending_confirms[CHAT]
        self.assertEqual(kind, "quit")
        # The stored expiry IS the engine's: now + CONFIRM_TTL_S.
        self.assertAlmostEqual(
            expires_at - time.time(), CONFIRM_TTL_S, delta=2.0)
        # And the prompt's countdown is that same value's face.
        self.assertIn(
            f"⏳ {confirm_live.format_remaining(CONFIRM_TTL_S)} "
            "left to confirm", reply.text)


class ConfirmLiveTickTests(unittest.TestCase):
    def test_registration_and_burn_down(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter)
        _stage_and_send(rt)
        entry = rt._confirm_live[CHAT]
        self.assertEqual(entry["kind"], "quit")
        self.assertEqual(entry["message_id"], "777")
        expires_at = entry["expires_at"]
        rt._confirm_live_tick(now=expires_at - 240)
        self.assertIn("⏳ 4:00 left to confirm",
                      adapter.edits[-1][1])
        # Buttons ride the countdown edits (the prompt stays live).
        self.assertTrue(adapter.edits[-1][2])
        rt._confirm_live_tick(now=expires_at - 61)
        self.assertIn("⏳ 1:01 left to confirm",
                      adapter.edits[-1][1])
        # A tick whose DISPLAYED remainder is unchanged (60.4s
        # rounds to the same "1:01") sends nothing.
        n = len(adapter.edits)
        rt._confirm_live_tick(now=expires_at - 60.4)
        self.assertEqual(len(adapter.edits), n)
        # Crossing into the next displayed second edits again.
        rt._confirm_live_tick(now=expires_at - 59.6)
        self.assertEqual(len(adapter.edits), n + 1)
        self.assertIn("⏳ 1:00 left to confirm",
                      adapter.edits[-1][1])

    def test_expiry_finalizes_pending_and_message(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter)
        _stage_and_send(rt)
        expires_at = rt._pending_confirms[CHAT][2]
        rt._confirm_live_tick(now=expires_at + 1)
        text = adapter.edits[-1][1]
        self.assertIn("expired", text)
        self.assertIn("did not run", text)
        self.assertEqual(adapter.edits[-1][2], [])  # no live buttons
        self.assertNotIn(CHAT, rt._pending_confirms)
        self.assertNotIn(CHAT, rt._confirm_live)
        self.assertEqual(rt.calls, [])  # the action never ran

    def test_cancel_settles_prompt(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter)
        _stage_and_send(rt)
        rt.handle_control("/quit cancel", CHAT,
                          message=_msg(_chat(), "/quit cancel"))
        rt._confirm_live_tick()
        text = adapter.edits[-1][1]
        self.assertIn("cancelled", text)
        self.assertIn("did not run", text)
        self.assertEqual(adapter.edits[-1][2], [])
        self.assertNotIn(CHAT, rt._confirm_live)
        self.assertEqual(rt.calls, [])

    def test_confirm_settles_prompt_and_action_ran(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter)
        _stage_and_send(rt)
        rt.handle_control("/quit confirm", CHAT,
                          message=_msg(_chat(), "/quit confirm"))
        self.assertEqual(rt.calls, [("stop",)])
        rt._confirm_live_tick()
        self.assertIn("✓ confirmed", adapter.edits[-1][1])
        self.assertEqual(adapter.edits[-1][2], [])
        self.assertNotIn(CHAT, rt._confirm_live)

    def test_gate_late_confirm_marks_prompt_expired(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter)
        _stage_and_send(rt)
        kind, tail, _exp = rt._pending_confirms[CHAT]
        past = time.time() - 1
        rt._pending_confirms[CHAT] = (kind, tail, past)
        # The entry's expiry must match for the mark to bind —
        # in production both are the one staged float.
        rt._confirm_live[CHAT]["expires_at"] = past
        out = rt.handle_control("/quit confirm", CHAT,
                                message=_msg(_chat(), "/quit confirm"))
        self.assertIn("expired", out)
        rt._confirm_live_tick()
        self.assertIn("expired", adapter.edits[-1][1])
        self.assertEqual(rt.calls, [])

    def test_no_registration_on_uneditable_surface(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter, platform="sms")
        chat = _chat(platform="sms")
        msg = _msg(chat, "/quit")
        reply = rt.handle_control("/quit", CHAT, message=msg)
        rt._send_control_reply(msg, reply)
        # The prompt still went out (with its birth countdown);
        # it simply never registers for edits on SMS.
        self.assertTrue(rt.gateway.sent)
        self.assertNotIn(CHAT, getattr(rt, "_confirm_live", {}))

    def test_failed_edit_drops_registration(self):
        adapter = _FakeAdapter(fail=True)
        rt = _live_runtime(adapter)
        _stage_and_send(rt)
        expires_at = rt._pending_confirms[CHAT][2]
        rt._confirm_live_tick(now=expires_at - 240)
        self.assertNotIn(CHAT, rt._confirm_live)
        # The pending itself is untouched — typed confirm works.
        self.assertIn(CHAT, rt._pending_confirms)

    def test_superseded_prompt_stops_ticking(self):
        adapter = _FakeAdapter()
        rt = _live_runtime(adapter)
        _stage_and_send(rt, "/quit")
        first_expiry = rt._confirm_live[CHAT]["expires_at"]
        # A different kind stages over the same chat: the old
        # prompt's registration is replaced by the new send, and
        # the OLD expiry no longer drives anything.
        _stage_and_send(rt, "/forget abc")
        entry = rt._confirm_live[CHAT]
        self.assertEqual(entry["kind"], "forget")
        self.assertNotEqual(entry["expires_at"], first_expiry)
        # A tick inside the forget window serves the FORGET
        # entry only — the quit window drives nothing anymore.
        rt._confirm_live_tick(now=entry["expires_at"] - 10)
        self.assertIn(CHAT, rt._confirm_live)
        for _mid, text, _btn in adapter.edits:
            self.assertNotIn("expired", text)
            self.assertNotIn("/quit", text)


if __name__ == "__main__":
    unittest.main()
