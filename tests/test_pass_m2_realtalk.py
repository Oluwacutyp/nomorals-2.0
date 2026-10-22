"""Pass M2 — pulse real-talk segments.

Owner's program (2026-10-22): every pulse gains a real talk —
trending topics AND beyond-trending finds AND Library passages,
co-equal, drawn from and marked in the one Teachings ledger (M1).

Liveness first (the design's census warning): afternoon/evening
exist TWICE — slot composers inside morning_pulse.py and standalone
agents/afternoon_pulse.py + evening_pulse.py modules.  These tests
pin which path actually fires:

* boot (partner/runtime.py) calls morning_pulse.ensure_pulse_jobs —
  the standalone ensure_* functions have NO production caller;
* the jobs that machinery registers carry payload
  {"action": "run", "slot": <slot>}, which the pulse tool dispatches
  to run_slot → the slot composers;
* the standalone modules' payloads ({"action": "run_afternoon"}) hit
  a tool that does not know that action — a dead path even if a stale
  job with their name existed.

Then the talk itself: lands in every slot's delivered text, marks
the shared ledger (invisible to later slots and to motivation),
keeps the mix discipline over a simulated week (≤1 trending-led
slot per rolling 24h; ≥ half beyond-trending/Library across the
week), grounds trending in the EXTRACTED page (fail-closed when
extraction fails — never in the feed summary), degrades to the real
passage card when the brain is down, and is honestly absent when
there is no material.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nomorals.agents.morning_pulse as mp
import nomorals.agents.realtalk as rt
import nomorals.agents.teachings as teach
import nomorals.llm.brain as brain_mod
from nomorals.agents import morning_briefing as mb
from nomorals.agents.news import NewsAgent
from nomorals.agents.teachings import TeachingsLedger
from nomorals.core.ids import new_id
from nomorals.storage.db import Database


# ── fixtures (F6/M1 patterns) ────────────────────────────────────────

class _SendResult:
    def __init__(self, ok=True, message_id="m1", error=""):
        self.ok = ok
        self.message_id = message_id
        self.error = error


class FakeGateway:
    """Records every send (the F6 fake, verbatim shape)."""

    def __init__(self, fail_on=()):
        self.sent = []
        self.fail_on = set(fail_on)
        self.calls = 0

    def status(self):
        return {"telegram-bot": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, **kwargs):
        self.calls += 1
        if self.calls in self.fail_on:
            return _SendResult(ok=False, error="boom (fake)")
        self.sent.append((platform, chat_ref.chat_id, text))
        return _SendResult(ok=True)

    def platforms_for_chat_id(self, chat_id):
        return []


def make_ctx(test, gateway=None):
    tmp = tempfile.mkdtemp(prefix="m2-realtalk-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats="telegram:111", proactive_enabled=True,
        proactive_briefing=True, proactive_watchers=True,
        quiet_start=22, quiet_end=8, timezone="UTC")
    settings = SimpleNamespace(workspace_dir=tmp, partner=partner,
                               pulse=None)
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    if gateway is not None:
        ctx.gateway = gateway
    return ctx, tmp


def _item(*, theme="discipline", ref="meditations",
          label="Meditations — WisdomKeeper",
          passage="The impediment to action advances action; "
                  "what stands in the way becomes the way.",
          title="Meditations", source_type="wisdom") -> dict:
    return {
        "id": f"teach_{ref}", "kind": "teaching", "title": title,
        "body": "", "theme": theme, "tradition": "", "weight": 1.0,
        "source": {"type": source_type, "ref": ref, "label": label,
                   "passage": passage},
        "created_at": time.time(), "served": [],
    }


TALK_TEXT = " ".join(["TALKMARK"] + ["word"] * 160)   # 161 words
ARTICLE_TEXT = ("GROUNDING-PHRASE-42 " + " ".join(
    f"Article sentence {i} carries the real substance of the piece."
    for i in range(30)))
WHEEL_TEXT = ("WHEEL-GROUNDING-7 " + " ".join(
    f"Wheel sentence {i} explains the idea with real depth."
    for i in range(30)))


class _Brain:
    """Records prompts; answers with a fixed on-size talk."""

    def __init__(self, text=TALK_TEXT):
        self._text = text
        self.prompts: list[str] = []

    def chat(self, messages):
        self.prompts.append(messages[0]["content"] if messages else "")
        return SimpleNamespace(text=self._text)


def _stub_brain(test, brain):
    test.addCleanup(setattr, brain_mod, "brain_for", brain_mod.brain_for)
    brain_mod.brain_for = lambda ctx: brain


def _stub_producers(test, lib_items, net_items=()):
    test.addCleanup(setattr, teach, "library_candidates",
                    teach.library_candidates)
    test.addCleanup(setattr, teach, "net_candidates",
                    teach.net_candidates)
    test.addCleanup(setattr, teach, "memory_candidates",
                    teach.memory_candidates)
    teach.library_candidates = lambda ctx, themes: list(lib_items)
    teach.net_candidates = lambda ctx, themes: list(net_items)
    teach.memory_candidates = lambda ctx: []


def _stub_hunt(test, results):
    test.addCleanup(setattr, rt, "_hunt", rt._hunt)
    rt._hunt = lambda ctx, query, limit=5: list(results)


def _stub_extract(test, fn):
    test.addCleanup(setattr, rt, "_extract", rt._extract)
    rt._extract = fn


def seed_story(ctx, url, *, title="Big story", interest=None,
               summary="sum", published=1000.0):
    NewsAgent(ctx)  # ensures the delivery columns exist
    ctx.db.execute(
        "INSERT INTO news_items (id, source, publisher, title, url, "
        "summary, published, created_at, delivery_state, interest) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (new_id(), "testfeed", "TestWire", title, url, summary,
         published, time.time(), "undelivered", interest))


def sent_text(gateway):
    return "\n".join(t for _, _, t in gateway.sent)


def served_talks(ctx, **kw):
    return [it for it in TeachingsLedger(ctx).served_items(**kw)
            if it.get("kind") == "realtalk"]


class _FakeRegistry:
    def __init__(self, ctx):
        self.context = ctx
        self.tools = {}

    def register(self, name, **kw):
        def deco(fn):
            self.tools[name] = fn
            return fn
        return deco


# ── liveness: which path fires ───────────────────────────────────────

class LivenessTests(unittest.TestCase):

    def test_boot_registers_slot_jobs_that_dispatch_to_run_slot(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self, gw)
        _stub_producers(self, [_item()])
        _stub_brain(self, _Brain())
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: "")
        mp.ensure_pulse_jobs(ctx)
        # list_jobs() strips payloads; the durable row IS what fires,
        # so read the registered payloads straight from the table.
        rows = ctx.db.query(
            "SELECT name, payload FROM schedule_jobs") or []
        payloads = {}
        for row in rows:
            try:
                payloads[row["name"]] = json.loads(row["payload"] or "{}")
            except (TypeError, ValueError):
                pass
        for slot in ("afternoon", "evening", "midnight"):
            payload = payloads[f"{slot}-pulse"]
            self.assertEqual(payload["tool"], "pulse")
            self.assertEqual(payload["args"],
                             {"action": "run", "slot": slot})
        # and the registered payload really delivers a talk end-to-end
        reg = _FakeRegistry(ctx)
        mp.register(reg)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = reg.tools["pulse"](action="run", slot="afternoon")
        self.assertEqual(r["slot"], "afternoon")
        self.assertIn("Real talk", r["text"])
        self.assertIn("TALKMARK", sent_text(gw))

    def test_standalone_payload_actions_are_dead(self):
        ctx, _ = make_ctx(self)
        reg = _FakeRegistry(ctx)
        mp.register(reg)
        for action in ("run_afternoon", "run_evening"):
            out = reg.tools["pulse"](action=action)
            self.assertIn("error", out)
            self.assertIn("unsupported pulse action", out["error"])

    def test_standalone_ensure_has_no_production_caller(self):
        # Census pin: the ONLY ensure_* the boot path calls is the
        # slot machinery's; the standalone modules are imported by
        # tests alone.  If someone wires them in later, this pin is
        # the deliberate place that decision gets recorded.
        root = Path(mp.__file__).resolve().parent.parent  # nomorals/
        callers = []
        for path in root.rglob("*.py"):
            if path.name in ("afternoon_pulse.py", "evening_pulse.py"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "ensure_afternoon_pulse_job(" in text or \
               "ensure_evening_pulse_job(" in text:
                callers.append(str(path))
        self.assertEqual(callers, [])
        runtime_src = (root / "agents" / "partner" / "runtime.py") \
            .read_text(encoding="utf-8")
        self.assertIn("ensure_pulse_jobs", runtime_src)


# ── the talk in every slot ───────────────────────────────────────────

class SlotTalkTests(unittest.TestCase):

    def _prep(self, gw):
        ctx, _ = make_ctx(self, gw)
        items = [
            _item(theme="discipline", ref="meditations"),
            _item(theme="courage", ref="letters",
                  label="Letters — WisdomKeeper",
                  passage="We suffer more in imagination than in "
                          "reality, and the antidote is attention "
                          "to the hour in front of us."),
            _item(theme="craft", ref="enchiridion",
                  label="Enchiridion — WisdomKeeper",
                  passage="If you want things to be good, do not "
                          "wish them easy; wish yourself equal to "
                          "them, and practice daily."),
        ]
        _stub_producers(self, items)
        _stub_brain(self, _Brain())
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: "")
        return ctx

    def test_all_slots_carry_a_talk_and_never_repeat(self):
        gw = FakeGateway()
        ctx = self._prep(gw)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r_aft = mp.run_slot(ctx, "afternoon")
            r_eve = mp.run_slot(ctx, "evening")
            r_mid = mp.run_slot(ctx, "midnight")
        for r in (r_aft, r_eve, r_mid):
            self.assertIn("🗣️ *Real talk*", r["text"])
            self.assertIn("TALKMARK", r["text"])
            self.assertIn("— From her library:", r["text"])
        self.assertIn("TALKMARK", sent_text(gw))
        talks = served_talks(ctx)
        self.assertEqual(len(talks), 3)
        self.assertEqual(len({t["id"] for t in talks}), 3)
        self.assertEqual({t["served"][0]["surface"] for t in talks},
                         {"pulse-afternoon", "pulse-evening",
                          "pulse-midnight"})

    def test_morning_pulse_text_carries_the_talk(self):
        gw = FakeGateway()
        ctx = self._prep(gw)
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_synthesize", return_value=""):
            r = mp.run_pulse(ctx)
        self.assertIn("Real talk", r["text"])
        self.assertIn("TALKMARK", r["text"])
        talks = served_talks(ctx)
        self.assertEqual(len(talks), 1)
        self.assertEqual(talks[0]["served"][0]["surface"],
                         "pulse-morning")

    def test_morning_composer_registers_realtalk_after_news(self):
        ctx, _ = make_ctx(self)
        _stub_producers(self, [_item()])
        _stub_brain(self, _Brain())
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: "")
        seed_story(ctx, "https://news.test/composer-story")
        b = mb.BriefingComposer().compose(ctx, "2026-10-23")
        names = [s.name for s in b.sections]
        self.assertIn("realtalk", names)
        sec = next(s for s in b.sections if s.name == "realtalk")
        self.assertEqual(sec.title, "🗣️ Real talk")
        self.assertIn("TALKMARK", sec.render_text())
        if "news" in names:  # design: after News, before FromYourPast
            self.assertLess(names.index("news"), names.index("realtalk"))


# ── ledger sharing: pulses vs motivation ─────────────────────────────

class LedgerSharingTests(unittest.TestCase):

    def test_talk_spends_item_and_theme_for_every_surface(self):
        ctx, _ = make_ctx(self)
        a = _item(theme="discipline", ref="meditations")
        b = _item(theme="courage", ref="letters",
                  label="Letters — WisdomKeeper",
                  passage="We suffer more in imagination than in "
                          "reality, and the antidote is attention.")
        _stub_producers(self, [a])
        _stub_brain(self, _Brain())
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: "")
        t0 = time.time()
        first = rt.compose_realtalk(ctx, "morning", now=t0,
                                    rng=random.Random(1))
        self.assertIsNotNone(first)
        self.assertEqual(first["talk_kind"], "library")
        # theme cooldown is shared: motivation feels it too
        ledger = TeachingsLedger(ctx)
        self.assertTrue(ledger.theme_in_cooldown("discipline",
                                                 now=t0 + 3600))
        # a second item stocks; the evening talk must take IT, not A
        _stub_producers(self, [a, b])
        second = rt.compose_realtalk(ctx, "evening", now=t0 + 7200,
                                     rng=random.Random(2))
        self.assertIsNotNone(second)
        self.assertNotEqual(second["id"], first["id"])
        # both spent: a motivation pick finds nothing left to give
        picked = ledger.pick(surface="motivation-morning",
                             now=t0 + 7200, rng=random.Random(3))
        self.assertIsNone(picked)


# ── mix discipline over a simulated week ─────────────────────────────

class MixTests(unittest.TestCase):

    def test_mix_over_simulated_week(self):
        ctx, _ = make_ctx(self)
        counter = {"n": 0}

        def fresh_lib(ctx_, themes):
            out = []
            for _ in range(3):
                counter["n"] += 1
                n = counter["n"]
                out.append(_item(
                    theme=f"libtheme-{n}", ref=f"work-{n}",
                    label=f"Work {n} — WisdomKeeper",
                    passage=f"Passage {n}: a teaching with enough "
                            f"substance to carry a real talk today."))
            return out

        self.addCleanup(setattr, teach, "library_candidates",
                        teach.library_candidates)
        self.addCleanup(setattr, teach, "net_candidates",
                        teach.net_candidates)
        self.addCleanup(setattr, teach, "memory_candidates",
                        teach.memory_candidates)
        teach.library_candidates = fresh_lib
        teach.net_candidates = lambda ctx_, themes: []
        teach.memory_candidates = lambda ctx_: []
        _stub_brain(self, _Brain())
        _stub_hunt(self, [{"title": "Wheel piece",
                           "url": "https://wheel.test/p",
                           "snippet": "snippet"}])

        def extract(url):
            if "wheel.test" in url:
                return WHEEL_TEXT
            if "news.test" in url:
                return ARTICLE_TEXT
            return ""
        _stub_extract(self, extract)

        t0 = time.time() - 30 * 86400  # simulated past; spans a week
        served = []  # (now, talk_kind)
        for day in range(7):
            base = t0 + day * 25 * 3600
            seed_story(ctx, f"https://news.test/day{day}",
                       title=f"Day {day} story", interest=0.9,
                       published=base)
            for slot, off in (("morning", 0), ("afternoon", 4 * 3600),
                              ("evening", 9 * 3600)):
                item = rt.compose_realtalk(ctx, slot, now=base + off,
                                           rng=random.Random(day))
                self.assertIsNotNone(item, f"day {day} {slot}")
                served.append((base + off, item["talk_kind"]))
        self.assertEqual(len(served), 21)
        trending_at = [t for t, k in served if k == "trending"]
        for i, t1 in enumerate(trending_at):
            for t2 in trending_at[i + 1:]:
                self.assertGreaterEqual(
                    t2 - t1, 24 * 3600,
                    "two trending-led talks inside one rolling day")
        beyond = sum(1 for _, k in served if k in ("wheel", "library"))
        self.assertGreaterEqual(
            beyond * 2, len(served),
            "beyond-trending/Library must be ~half the week's talks")


# ── grounding: extraction or nothing ─────────────────────────────────

class GroundingTests(unittest.TestCase):

    def test_trending_talk_is_grounded_in_the_extracted_page(self):
        ctx, _ = make_ctx(self)
        _stub_producers(self, [])
        brain = _Brain()
        _stub_brain(self, brain)
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: ARTICLE_TEXT)
        seed_story(ctx, "https://news.test/big-story", interest=0.9,
                   summary="SHORT-SUMMARY-MARKER")
        item = rt.compose_realtalk(ctx, "morning", rng=random.Random(0))
        self.assertIsNotNone(item)
        self.assertEqual(item["talk_kind"], "trending")
        # the brain was fed the extracted page, NOT the feed summary
        self.assertIn("GROUNDING-PHRASE-42", brain.prompts[0])
        self.assertNotIn("SHORT-SUMMARY-MARKER", brain.prompts[0])
        # and the ledger's served record carries the extracted passage
        talks = served_talks(ctx)
        self.assertEqual(len(talks), 1)
        self.assertIn("GROUNDING-PHRASE-42",
                      talks[0]["source"]["passage"])
        self.assertIn("https://news.test/big-story", item["text"])

    def test_extraction_failure_is_fail_closed(self):
        ctx, _ = make_ctx(self)
        _stub_producers(self, [])
        _stub_brain(self, _Brain())
        _stub_hunt(self, [{"title": "Wheel piece",
                           "url": "https://wheel.test/p",
                           "snippet": "snippet"}])
        _stub_extract(self, lambda url: "")  # every fetch walls/fails
        seed_story(ctx, "https://news.test/walled", interest=0.9,
                   summary="A summary that must never ground a talk.")
        item = rt.compose_realtalk(ctx, "morning", rng=random.Random(0))
        self.assertIsNone(item)
        self.assertEqual(served_talks(ctx), [])

    def test_walled_story_falls_through_to_wheel_not_summary(self):
        ctx, _ = make_ctx(self)
        _stub_producers(self, [])
        _stub_brain(self, _Brain())
        _stub_hunt(self, [{"title": "Wheel piece",
                           "url": "https://wheel.test/p",
                           "snippet": "snippet"}])

        def extract(url):
            return WHEEL_TEXT if "wheel.test" in url else ""
        _stub_extract(self, extract)
        seed_story(ctx, "https://news.test/walled", interest=0.9,
                   summary="WALLED-SUMMARY-MARKER")
        item = rt.compose_realtalk(ctx, "morning", rng=random.Random(0))
        self.assertIsNotNone(item)
        self.assertEqual(item["talk_kind"], "wheel")
        self.assertNotIn("WALLED-SUMMARY-MARKER",
                         item["source"]["passage"])


# ── degraded + absent honesty ────────────────────────────────────────

class DegradedAndAbsentTests(unittest.TestCase):

    def test_dead_brain_degrades_to_the_real_passage_card(self):
        ctx, _ = make_ctx(self)
        a = _item(passage="PASSCARD the impediment to action advances "
                          "action; what stands in the way becomes "
                          "the way, every single time it is met.")
        _stub_producers(self, [a])
        _stub_brain(self, None)  # brain_for returns None
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: "")
        item = rt.compose_realtalk(ctx, "morning", rng=random.Random(0))
        self.assertIsNotNone(item)
        self.assertTrue(item["degraded"])
        self.assertIn("PASSCARD", item["text"])
        self.assertIn("— From her library: Meditations — WisdomKeeper",
                      item["text"])
        self.assertEqual(len(served_talks(ctx)), 1)

    def test_no_material_means_no_segment_and_no_invention(self):
        gw = FakeGateway()
        ctx, _ = make_ctx(self, gw)
        _stub_producers(self, [])
        _stub_brain(self, _Brain())
        _stub_hunt(self, [])
        _stub_extract(self, lambda url: "")
        self.assertIsNone(rt.compose_realtalk(ctx, "afternoon",
                                              rng=random.Random(0)))
        with patch.object(NewsAgent, "poll", lambda self: {"ok": True}), \
             patch.object(mp, "_afternoon_markets", return_value=("", [])):
            r = mp.run_slot(ctx, "afternoon")
        self.assertNotIn("Real talk", r["text"])
        self.assertIn("Quiet afternoon", r["text"])
        self.assertEqual(served_talks(ctx), [])


if __name__ == "__main__":
    unittest.main()
