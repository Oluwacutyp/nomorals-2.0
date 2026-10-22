"""Pass M1 — Teachings ledger + Motivation class.

Pins the design's guarantees:
* served-marks are shared across consumers and both cooldowns
  (30-day item, 48h theme) hold for every surface;
* the window re-roll lands inside 05:00–06:00 / 19:00–21:00 Lagos
  across many simulated days, anchored on Africa/Lagos regardless of
  the host clock (Denver host simulated);
* catch-up fires after a simulated overnight outage and never after
  the day-part cutoff;
* the backstop re-arms a broken one-shot chain and dispatches through
  the real scheduler;
* generation degrades to a real Library passage card when synthesis
  is unavailable (never canned text) and renders through the gateway.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import nomorals.agents.motivation as mot
import nomorals.agents.teachings as teach
import nomorals.llm.brain as brain_mod
from nomorals.agents.scheduler import Scheduler, parse_schedule_spec
from nomorals.agents.teachings import TeachingsLedger
from nomorals.pulses.timezone import PulseAnchor
from nomorals.storage.db import Database

LAGOS = ZoneInfo("Africa/Lagos")


def _ts(year, month, day, hour, minute=0) -> float:
    return datetime(year, month, day, hour, minute,
                    tzinfo=LAGOS).timestamp()


def _ctx(test: unittest.TestCase, **over: object) -> SimpleNamespace:
    tmp = tempfile.mkdtemp(prefix="m1-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram-bot:111", proactive_enabled=True,
        proactive_briefing=True, proactive_watchers=True,
        quiet_start=22, quiet_end=8, timezone="UTC")
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner,
        resolve=lambda rel: os.path.join(tmp, rel))
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    for key, val in over.items():
        setattr(ctx, key, val)
    return ctx


def _item(*, theme="discipline", source_type="wisdom",
          ref="meditations", label="Meditations — WisdomKeeper",
          passage="The impediment to action advances action; "
                  "what stands in the way becomes the way.",
          tradition="", title="Meditations") -> dict:
    return {
        "id": f"teach_{ref}_{abs(hash(passage)) % 10**8}",
        "kind": "teaching", "title": title, "body": "", "theme": theme,
        "tradition": tradition, "weight": 1.0,
        "source": {"type": source_type, "ref": ref, "label": label,
                   "passage": passage},
        "created_at": time.time(), "served": [],
    }


class _FakeNotifier:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title, "body": body,
                           **kw})
        return {"delivered": True, "persisted_only": False}


class _Brain:
    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages):
        return SimpleNamespace(text=self._text)


def _stub_producers(test: unittest.TestCase, items: list[dict]) -> None:
    """Stock the ledger deterministically from a fixed item list."""
    lib = [it for it in items
           if it["source"]["type"] in ("wisdom", "book")]
    net = [it for it in items
           if it["source"]["type"] in ("web", "news")]
    test.addCleanup(setattr, teach, "library_candidates",
                    teach.library_candidates)
    test.addCleanup(setattr, teach, "net_candidates",
                    teach.net_candidates)
    test.addCleanup(setattr, teach, "memory_candidates",
                    teach.memory_candidates)
    teach.library_candidates = lambda ctx, themes: list(lib)
    teach.net_candidates = lambda ctx, themes: list(net)
    teach.memory_candidates = lambda ctx: []


def _stub_brain(test: unittest.TestCase, brain) -> None:
    test.addCleanup(setattr, brain_mod, "brain_for", brain_mod.brain_for)
    brain_mod.brain_for = lambda ctx: brain


WORDS_150 = " ".join(["word"] * 150)


# ── ledger guarantees ────────────────────────────────────────────────

class LedgerCooldownTests(unittest.TestCase):
    def test_served_marks_shared_across_consumers(self):
        ctx = _ctx(self)
        ledger = TeachingsLedger(ctx)
        now = _ts(2026, 10, 23, 7, 0)
        item = _item()
        ledger._store_candidates([item], now)
        picked = ledger.pick(surface="motivation-morning", now=now,
                             rng=random.Random(1))
        self.assertIsNotNone(picked)
        ledger.mark_served(picked, "motivation-morning", body="piece",
                           now=now)
        # the item is gone from the pool — for EVERY surface
        fresh = TeachingsLedger(ctx)  # second consumer, same store
        self.assertEqual(fresh.candidates(now=now), [])
        self.assertIsNone(fresh.pick(surface="pulse-realtalk", now=now,
                                     rng=random.Random(2)))
        self.assertIsNone(fresh.pick(surface="trivia", now=now,
                                     rng=random.Random(3)))

    def test_theme_cooldown_shared_across_consumers(self):
        ctx = _ctx(self)
        ledger = TeachingsLedger(ctx)
        now = _ts(2026, 10, 23, 7, 0)
        first = _item(theme="discipline", ref="meditations",
                      passage="Passage one on discipline, long enough "
                              "to be real material for the ledger.")
        second = _item(theme="discipline", ref="epictetus",
                       passage="Passage two on discipline, also real "
                               "material drawn from the same theme.")
        ledger._store_candidates([first, second], now)
        picked = ledger.pick(surface="motivation-morning",
                             theme="discipline", now=now,
                             rng=random.Random(1))
        self.assertIsNotNone(picked)
        ledger.mark_served(picked, "motivation-morning", now=now)
        # 48h theme cooldown: even a DIFFERENT item on the theme is
        # unservable, on any surface, until the cooldown lapses
        self.assertIsNone(ledger.pick(surface="trivia",
                                      theme="discipline",
                                      now=now + 3600,
                                      rng=random.Random(2)))
        # after the cooldown the surviving item serves — restocked
        # fresh, as the daily producers do (the pool itself is a 24h
        # cache; the served item stays filtered by the 30-day rule)
        other = second if picked["id"] == first["id"] else first
        ledger._store_candidates([other], now + 49 * 3600)
        later = ledger.pick(surface="trivia", theme="discipline",
                            now=now + 49 * 3600, rng=random.Random(2))
        self.assertIsNotNone(later)
        self.assertEqual(later["id"], other["id"])

    def test_item_cooldown_expires_after_30_days(self):
        ctx = _ctx(self)
        ledger = TeachingsLedger(ctx)
        now = _ts(2026, 10, 23, 7, 0)
        item = _item()
        ledger._store_candidates([item], now - 31 * 86400)
        ledger.mark_served(item, "motivation-morning",
                           now=now - 31 * 86400)
        # restock fresh (pool TTL is 24h) and confirm visibility again
        ledger._store_candidates([item], now)
        pool = ledger.candidates(now=now)
        self.assertEqual(len(pool), 1)

    def test_mix_floor_prefers_net_when_owed(self):
        ctx = _ctx(self)
        ledger = TeachingsLedger(ctx)
        now = _ts(2026, 10, 23, 7, 0)
        # three wisdom serves earlier this week (older than the 48h
        # theme cooldown for their own themes)
        for i, theme in enumerate(("mortality", "money", "wonder")):
            ledger.mark_served(
                _item(theme=theme, ref=f"old-{i}",
                      passage=f"Earlier teaching {i} on {theme}, "
                              "served days ago this week."),
                "motivation-morning", now=now - 3 * 86400)
        wisdom = _item(theme="courage", ref="seneca",
                       passage="A wisdom passage on courage stocked "
                               "fresh from the Library today.")
        web = _item(theme="courage", source_type="web",
                    ref="https://example.test/courage",
                    label="A hunted piece on courage",
                    passage="A web passage on courage stocked fresh "
                            "from the live net today.")
        ledger._store_candidates([wisdom, web], now)
        picked = ledger.pick(surface="motivation-morning", now=now,
                             rng=random.Random(7))
        self.assertIsNotNone(picked)
        self.assertEqual(picked["source"]["type"], "web")


# ── window rolls + Lagos anchoring ───────────────────────────────────

class WindowRollTests(unittest.TestCase):
    def test_rolls_stay_inside_windows_across_days(self):
        anchor = PulseAnchor("Africa/Lagos")
        prefs = mot._prefs(SimpleNamespace(settings=SimpleNamespace()))
        base = datetime(2026, 10, 23, tzinfo=LAGOS).date()
        for name, lo, hi in (("morning", 5 * 60, 6 * 60),
                             ("evening", 19 * 60, 21 * 60)):
            cfg = mot._window_cfg(prefs, name)
            minutes = set()
            for seed in range(12):
                rng = random.Random(seed)
                for day_off in range(30):
                    instant = mot.roll_instant(
                        anchor, cfg, base + timedelta(days=day_off), rng)
                    lagos = instant.astimezone(LAGOS)
                    self.assertEqual(lagos.utcoffset(),
                                     timedelta(hours=1))
                    mod = lagos.hour * 60 + lagos.minute
                    self.assertGreaterEqual(mod, lo, (name, lagos))
                    self.assertLessEqual(mod, hi, (name, lagos))
                    self.assertEqual(
                        lagos.date(), base + timedelta(days=day_off))
                    minutes.add(mod)
            # the roll actually varies — not a fixed daily instant
            self.assertGreater(len(minutes), 20, name)

    def test_lagos_anchoring_regardless_of_host_tz(self):
        anchor = PulseAnchor("Africa/Lagos")
        prefs = mot._prefs(SimpleNamespace(settings=SimpleNamespace()))
        cfg = mot._window_cfg(prefs, "morning")
        now_ts = _ts(2026, 10, 23, 3, 0)  # 03:00 Lagos, before window
        old_tz = os.environ.get("TZ")
        try:
            for host_tz in ("America/Denver", "UTC", "Asia/Tokyo"):
                os.environ["TZ"] = host_tz
                if hasattr(time, "tzset"):
                    time.tzset()
                instant = mot.next_fire(anchor, cfg, now_ts=now_ts,
                                        rng=random.Random(4))
                lagos = instant.astimezone(LAGOS)
                self.assertEqual(lagos.date().isoformat(), "2026-10-23")
                self.assertTrue(5 * 60 <= lagos.hour * 60 + lagos.minute
                                <= 6 * 60, (host_tz, lagos))
                # the scheduler-facing spec round-trips to the same
                # instant on this host's clock (minute grain)
                kind, detail = parse_schedule_spec(mot._at_spec(instant))
                self.assertEqual(kind, "at")
                self.assertLessEqual(
                    abs(detail - instant.timestamp()), 60.0, host_tz)
        finally:
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            if hasattr(time, "tzset"):
                time.tzset()


# ── the run: delivery, degradation, re-arm ───────────────────────────

class RunTests(unittest.TestCase):
    def _wired_ctx(self, brain):
        ctx = _ctx(self)
        ctx.notifier = _FakeNotifier()
        _stub_producers(self, [_item()])
        _stub_brain(self, brain)
        return ctx

    def test_synthesis_path_delivers_and_marks_served(self):
        ctx = self._wired_ctx(_Brain(WORDS_150))
        now = _ts(2026, 10, 23, 7, 5)  # morning day-part, post-window
        res = mot.run_motivation(ctx, "morning", now=now,
                                 rng=random.Random(3))
        self.assertTrue(res["delivered"], res)
        self.assertFalse(res["degraded_card"])
        self.assertEqual(len(ctx.notifier.calls), 1)
        call = ctx.notifier.calls[0]
        self.assertEqual(call["kind"], "motivation")
        self.assertIn("From her library", call["body"])
        self.assertIn("Meditations", call["body"])
        # served in the ledger, visible to other consumers
        served = TeachingsLedger(ctx).served_items(
            surface_prefix="motivation")
        self.assertEqual(len(served), 1)
        self.assertEqual(served[0]["kind"], "motivation")
        # and the one-shot chain re-armed itself for the next day
        job = mot._pending_shot(Scheduler(ctx), mot.MORNING_JOB, now)
        self.assertIsNotNone(job)

    def test_degraded_card_when_synthesis_unavailable(self):
        ctx = self._wired_ctx(None)  # brain_for -> None
        item = _item()
        now = _ts(2026, 10, 23, 7, 5)
        res = mot.run_motivation(ctx, "morning", now=now,
                                 rng=random.Random(3))
        self.assertTrue(res["delivered"], res)
        self.assertTrue(res["degraded_card"])
        body = ctx.notifier.calls[0]["body"]
        # the card IS the real passage + provenance — never canned
        self.assertIn(item["source"]["passage"], body)
        self.assertIn("Meditations — WisdomKeeper", body)

    def test_run_skips_after_cutoff(self):
        ctx = self._wired_ctx(_Brain(WORDS_150))
        now = _ts(2026, 10, 23, 15, 0)  # 3 PM: worse than none
        res = mot.run_motivation(ctx, "morning", now=now,
                                 rng=random.Random(3))
        self.assertFalse(res["delivered"])
        self.assertEqual(ctx.notifier.calls, [])

    def test_delivery_renders_through_the_gateway(self):
        from nomorals.social.chat.gateway import ChatGateway
        from nomorals.social.chat.base import ChatAdapter, SendResult

        class _RecordingAdapter(ChatAdapter):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.calls = []

            def run(self, handler):  # pragma: no cover - unused
                pass

            def send(self, chat, text, **kw):
                self.calls.append((text, dict(kw)))
                return SendResult(ok=True, platform=self.name,
                                  message_id="m1")

        adapter = _RecordingAdapter("telegram-bot")
        gateway = ChatGateway({"telegram-bot": adapter})
        ctx = _ctx(self, gateway=gateway)
        ctx.extras["gateway"] = gateway
        passage = ("Hold **fast** to what is yours to control; "
                   "the rest is weather, and weather passes.")
        _stub_producers(self, [_item(passage=passage)])
        _stub_brain(self, None)  # card path: body carries the dialect
        now = _ts(2026, 10, 23, 19, 30)  # inside the evening window
        res = mot.run_motivation(ctx, "evening", now=now,
                                 rng=random.Random(5))
        self.assertTrue(res["delivered"], res)
        self.assertEqual(len(adapter.calls), 1)
        text, kw = adapter.calls[0]
        self.assertEqual(kw.get("parse_mode"), "HTML")
        self.assertIn("<b>🌙 Evening motivation", text)
        self.assertIn("<b>fast</b>", text)


# ── catch-up + backstop ──────────────────────────────────────────────

class CatchupBackstopTests(unittest.TestCase):
    def test_catchup_fires_after_overnight_outage(self):
        ctx = _ctx(self)
        ctx.notifier = _FakeNotifier()
        _stub_producers(self, [_item()])
        _stub_brain(self, _Brain(WORDS_150))
        # host slept through the 05:00–06:00 window; wakes 07:30 WAT
        now = _ts(2026, 10, 23, 7, 30)
        out = mot.check_motivation_catchup(ctx, now=now,
                                           rng=random.Random(2))
        self.assertTrue(out["morning"]["delivered"], out)
        self.assertTrue(out["morning"]["catchup"])
        self.assertFalse(out["evening"].get("delivered", False))
        self.assertEqual(len(ctx.notifier.calls), 1)
        # second wake the same morning: no double delivery
        out2 = mot.check_motivation_catchup(ctx, now=now + 600,
                                            rng=random.Random(2))
        self.assertFalse(out2["morning"].get("delivered", False))
        self.assertEqual(len(ctx.notifier.calls), 1)

    def test_catchup_silent_after_daypart(self):
        ctx = _ctx(self)
        ctx.notifier = _FakeNotifier()
        _stub_producers(self, [_item()])
        _stub_brain(self, _Brain(WORDS_150))
        now = _ts(2026, 10, 23, 15, 0)
        out = mot.check_motivation_catchup(ctx, now=now,
                                           rng=random.Random(2))
        self.assertFalse(out["morning"]["catchup"])
        self.assertEqual(ctx.notifier.calls, [])

    def test_backstop_rearms_broken_chain(self):
        ctx = _ctx(self)
        now = _ts(2026, 10, 23, 3, 0)  # inert day-part for catch-up
        armed = mot.ensure_motivation_jobs(ctx, rng=random.Random(1),
                                           now=now)
        self.assertTrue(armed["windows"]["morning"]["scheduled"])
        sched = Scheduler(ctx)
        # break the chain: delete both one-shots (a runner that died
        # before re-arming, a wiped row — the failure shape is moot)
        for job in list(sched.list_jobs()):
            if job.get("name") in (mot.MORNING_JOB, mot.EVENING_JOB):
                sched.remove(job["id"])
        self.assertIsNone(mot._pending_shot(sched, mot.MORNING_JOB, now))
        mot.motivation_backstop(ctx, now=now, rng=random.Random(1))
        self.assertIsNotNone(mot._pending_shot(sched, mot.MORNING_JOB,
                                               now))
        self.assertIsNotNone(mot._pending_shot(sched, mot.EVENING_JOB,
                                               now))

    def test_backstop_dispatches_through_the_scheduler(self):
        ctx = _ctx(self)
        now = time.time()
        calls: list[dict] = []

        def _tools_call(tool, capabilities=None, **args):
            calls.append({"tool": tool, **args})
            if tool == "motivation" and args.get("action") == "backstop":
                value = mot.motivation_backstop(
                    ctx, now=_ts(2026, 10, 23, 3, 0),
                    rng=random.Random(1))
                return SimpleNamespace(ok=True, value=value, error=None)
            return SimpleNamespace(ok=True, value={}, error=None)

        ctx.tools = SimpleNamespace(call=_tools_call)
        sched = Scheduler(ctx)
        sched.add("motivation backstop (test)", "in 1 seconds", "tool",
                  {"tool": "motivation", "args": {"action": "backstop"}},
                  routine=True)
        deadline = time.time() + 15
        while time.time() < deadline and not calls:
            sched.tick()
            time.sleep(0.1)
        self.assertTrue(calls, "backstop job never dispatched")
        # the firing was journaled as a schedule run (ok=1)
        rows = []
        deadline = time.time() + 10
        while time.time() < deadline:
            rows = ctx.db.query(
                "SELECT ok FROM schedule_runs "
                "ORDER BY started_at DESC LIMIT 1")
            if rows:
                break
            time.sleep(0.1)
        self.assertTrue(rows and rows[0]["ok"] == 1, rows)
        # and the chain it armed is real
        self.assertIsNotNone(mot._pending_shot(sched, mot.MORNING_JOB,
                                               now))


# ── tool surface ─────────────────────────────────────────────────────

class ToolSurfaceTests(unittest.TestCase):
    def test_register_and_actions(self):
        ctx = _ctx(self)
        ctx.notifier = _FakeNotifier()
        _stub_producers(self, [_item()])
        _stub_brain(self, _Brain(WORDS_150))
        captured: dict = {}

        class _Registry:
            context = ctx

            def register(self, name, description="", capability=None,
                         parameters=None):
                def deco(fn):
                    captured[name] = fn
                    return fn
                return deco

        mot.register(_Registry())
        self.assertIn("motivation", captured)
        fn = captured["motivation"]
        status = fn(action="status")
        self.assertTrue(status["ok"])
        self.assertIn("morning", status["windows"])
        forced = fn(action="now", window="morning")
        self.assertTrue(forced["delivered"], forced)
        history = fn(action="history")
        self.assertEqual(len(history["history"]), 1)
        self.assertEqual(history["history"][0]["theme"], "discipline")

    def test_ensure_is_idempotent(self):
        ctx = _ctx(self)
        now = _ts(2026, 10, 23, 3, 0)
        first = mot.ensure_motivation_jobs(ctx, rng=random.Random(9),
                                           now=now)
        second = mot.ensure_motivation_jobs(ctx, rng=random.Random(9),
                                            now=now)
        self.assertTrue(
            second["windows"]["morning"]["already_scheduled"])
        self.assertEqual(first["windows"]["morning"]["job_id"],
                         second["windows"]["morning"]["job_id"])
        sched = Scheduler(ctx)
        names = [j["name"] for j in sched.list_jobs() if j["enabled"]]
        self.assertEqual(names.count(mot.MORNING_JOB), 1)
        self.assertEqual(names.count(mot.EVENING_JOB), 1)
        self.assertEqual(names.count(mot.BACKSTOP_JOB), 1)


if __name__ == "__main__":
    unittest.main()
