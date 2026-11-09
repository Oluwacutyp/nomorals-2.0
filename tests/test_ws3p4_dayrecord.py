"""WS3 P4 — the DayRecord + evening derivation.

Pins:

* (a) fragments are append-only and never overwritten, keyed to
  the Nigeria-anchored day (23:30 UTC is already tomorrow in WAT;
  22:30 UTC is still today);
* (b) /day tell → record → derive → drafts land pending_review
  with the source fragments as evidence in their metadata,
  through the REAL derivation path — directly with a fake brain
  seam, and end-to-end through the partner runtime's /day surface;
* (c) a draft's numbers all trace to the record: a planted section
  value (goal progress) may appear; a number the fake brain
  invents gets that draft challenged and blocked — it never lands;
* (d) the GS section reads the newest bundle VERIFIED (the day
  window differenced from the reader's own journal stats; STATE.md
  prose never becomes a number) and is honestly absent without one;
* (e) the briefing bookend renders yesterday + the pending count
  shape and is absent without a yesterday record;
* (f) nothing in the flow posts anything (the publisher fake
  records zero publishes across the whole runtime flow);
* the evening job: registered durably at 19:00 owner-local, and
  the automatic run derives a day with fragments, does nothing
  for a skipped day or a fragment-less day, and never derives a
  day twice;
* the catalog: ``day`` registered everywhere a command must be,
  owner-only, never public.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents import dayrecord
from nomorals.agents.dayrecord import (
    DayStore,
    build_sections,
    derive_day,
    ensure_day_derivation_job,
    run_evening_derivation,
)
from nomorals.agents.delegation import DelegationStore
from nomorals.agents.scheduler import Scheduler
from nomorals.goals import GoalTracker, surface
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    parse_control,
)
from nomorals.social.drafts import (
    PENDING_REVIEW,
    DraftQueue,
    reviewable_drafts,
)
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.tools.agents import AGENT_TOOL_MODULES

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

DAY = "2026-11-09"
UID = "owner"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


class _DayCase(unittest.TestCase):
    """File-backed context db, owner tz pinned to Africa/Lagos."""

    def setUp(self) -> None:
        self._env = mock.patch.dict(
            os.environ, {"NM_OWNER_TZ": "Africa/Lagos"})
        self._env.start()
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.home = Path(self._tmp.name)
        self.db = Database(self.home / "context.db")
        self.db.migrate()
        self.context = SimpleNamespace(db=self.db)
        self.store = DayStore(self.db)
        self.start, self.end = dayrecord.day_window(DAY)
        # An untrained voice profile, deterministically: drafting
        # must not depend on the box's real profile.
        self._voice = mock.patch(
            "nomorals.social.voice.VoiceProfile.load",
            classmethod(lambda cls: SimpleNamespace(
                n_posts=0, describe=lambda: "plain test voice")))
        self._voice.start()

    def tearDown(self) -> None:
        self._voice.stop()
        self._env.stop()
        self._tmp.cleanup()

    def at(self, hour: float) -> float:
        return self.start + hour * 3600.0

    def queue(self) -> DraftQueue:
        return DraftQueue(db_path=str(self.home / "drafts.db"))


# ── (a) fragments: append-only, Nigeria-anchored ─────────────────────

class FragmentTests(_DayCase):

    def test_append_only_and_ordered(self) -> None:
        first = self.store.add_fragment(
            text="shipped the day record", ts=self.at(9), day=DAY)
        second = self.store.add_fragment(
            text="paired the GS bundle", ts=self.at(10), day=DAY,
            source="voice")
        frags = self.store.fragments(DAY)
        self.assertEqual([f["id"] for f in frags],
                         [first["id"], second["id"]])
        self.assertEqual(frags[1]["source"], "voice")
        # Telling the same words again APPENDS — nothing overwrites.
        self.store.add_fragment(
            text="shipped the day record", ts=self.at(11), day=DAY)
        frags = self.store.fragments(DAY)
        self.assertEqual(len(frags), 3)
        self.assertEqual(frags[0]["text"], "shipped the day record")
        # The store deliberately grows no edit/delete verbs.
        for verb in ("update_fragment", "delete_fragment",
                     "edit_fragment", "remove_fragment"):
            self.assertFalse(hasattr(self.store, verb), verb)

    def test_day_key_is_nigeria_anchored(self) -> None:
        # 23:30 UTC = 00:30 WAT the NEXT day; 22:30 UTC = 23:30 WAT.
        late = datetime(2026, 11, 9, 23, 30, tzinfo=UTC).timestamp()
        early = datetime(2026, 11, 9, 22, 30, tzinfo=UTC).timestamp()
        self.assertEqual(dayrecord.day_key(late), "2026-11-10")
        self.assertEqual(dayrecord.day_key(early), "2026-11-09")
        frag = self.store.add_fragment(text="past midnight",
                                       ts=late)
        self.assertEqual(frag["day"], "2026-11-10")
        self.assertEqual(self.store.fragments("2026-11-09"), [])

    def test_skip_flag_records_and_clears(self) -> None:
        self.assertFalse(self.store.skipped(DAY))
        self.store.set_skip(DAY, skip=True, note="owner's word")
        self.assertTrue(self.store.skipped(DAY))
        self.assertEqual(self.store.meta(DAY)["skip_note"],
                         "owner's word")
        self.store.set_skip(DAY, skip=False)
        self.assertFalse(self.store.skipped(DAY))


# ── (b)+(c) the derivation cell over a fake brain ────────────────────

_LINKEDIN_OK = ("Shipped the day-record build today. The tracker"
                " goal sits at 40% and the work goes on.")
_X_OK = "Day-record build shipped; the tracker goal sits at 40%."
_WA_OK = "today: shipped the day-record build, tracker at 40%"
_LINKEDIN_BAD = ("Huge day: engagement grew 900% overnight and"
                 " the tracker goal sits at 40%.")
_CLEAN = {
    "linkedin": "Shipped the day-record build today, and the"
                " work goes on.",
    "x": "Day-record build shipped.",
    "whatsapp": "today: shipped the day-record build",
}


def _fake_llm(linkedin_text: str = "", x_text: str = "",
              wa_text: str = ""):
    """A fact-stable fake brain: drafts from the angle lines in
    the drafter prompt; hands text straight back on rewrite
    prompts (tone/voice), the way a faithful rewrite would.
    Default texts carry NO numbers — a draft may only state a
    number the test planted as a record receipt."""

    def fn(prompt: str) -> str:
        if "build-in-public" in prompt:
            return linkedin_text or _CLEAN["linkedin"]
        if "single sharpest moment" in prompt:
            return x_text or _CLEAN["x"]
        if "day in a breath" in prompt:
            return wa_text or _CLEAN["whatsapp"]
        if "\n\n" in prompt:
            return prompt.rsplit("\n\n", 1)[-1]
        return prompt

    return fn


class DeriveFlowTests(_DayCase):

    def _seed_day(self) -> list[dict[str, Any]]:
        frags = [
            self.store.add_fragment(
                text="shipped the day-record build and wired the"
                     " evening derivation",
                ts=self.at(9), day=DAY),
            self.store.add_fragment(
                text="read the verified GS bundle end to end",
                ts=self.at(15), day=DAY, source="voice"),
        ]
        tracker = GoalTracker(self.db)
        goal = surface.run_sync(tracker.create(
            UID, "Ship the day record", goal_class="build"))
        surface.run_sync(tracker.update_progress(goal.goal_id, 40.0))
        return frags

    def test_derive_lands_pending_review_with_evidence(self) -> None:
        frags = self._seed_day()
        queue = self.queue()
        try:
            result = derive_day(
                self.context, DAY, queue=queue,
                llm_fn=_fake_llm(linkedin_text=_LINKEDIN_OK,
                                 x_text=_X_OK, wa_text=_WA_OK))
            self.assertEqual([d["platform"] for d in result.drafts],
                             ["linkedin", "x", "whatsapp"])
            self.assertEqual(result.blocked, [])
            review = reviewable_drafts(queue)
            self.assertEqual(len(review), 3)
            for draft in review:
                self.assertEqual(draft.status, PENDING_REVIEW)
                meta = draft.metadata
                self.assertEqual(meta["origin"], "day-derivation")
                self.assertEqual(meta["day"], DAY)
                self.assertEqual(meta["source_fragments"],
                                 [f["id"] for f in frags])
                self.assertEqual(meta["contract_id"],
                                 result.contract_id)
                self.assertEqual(meta["drafter"], "brain")
            # The cell ran on the real delegation substrate, and
            # the contract completed through its own gate.
            contracts = DelegationStore(self.db)
            contract = contracts.get_contract(result.contract_id)
            assert contract is not None
            self.assertEqual(contract.status, "completed")
            # Evidence kinds ride the ledger as claim scopes (the
            # substrate's shape); the row kinds stay its own.
            scopes = {c.scope for c in contracts.claims(contract.id)}
            self.assertIn("day-sections", scopes)
            self.assertIn("draft", scopes)
            # A derivation receipt is on the record.
            receipts = self.store.derivations(DAY)
            self.assertEqual(len(receipts), 1)
            self.assertEqual(len(receipts[0]["drafts"]), 3)
        finally:
            queue.close()

    def test_template_fallback_is_marked_not_disguised(self) -> None:
        self._seed_day()
        queue = self.queue()
        try:
            result = derive_day(self.context, DAY, queue=queue,
                                llm_fn=None)
            self.assertEqual(len(result.drafts), 3)
            for draft in reviewable_drafts(queue):
                self.assertEqual(draft.metadata["drafter"],
                                 "template")
        finally:
            queue.close()

    def test_invented_number_is_blocked_and_never_lands(self) -> None:
        self._seed_day()
        queue = self.queue()
        try:
            result = derive_day(
                self.context, DAY, queue=queue,
                llm_fn=_fake_llm(linkedin_text=_LINKEDIN_BAD,
                                 x_text=_X_OK, wa_text=_WA_OK))
            self.assertEqual([d["platform"] for d in result.drafts],
                             ["x", "whatsapp"])
            self.assertEqual(len(result.blocked), 1)
            blocked = result.blocked[0]
            self.assertEqual(blocked["platform"], "linkedin")
            self.assertIn(900.0, blocked["numbers"])
            landed = reviewable_drafts(queue)
            self.assertEqual(len(landed), 2)
            for draft in landed:
                self.assertNotIn("900", draft.content)
            # The block is a permanent ledger fact: a contradicted
            # challenge against the linkedin draft claim.
            contracts = DelegationStore(self.db)
            claims = contracts.claims(result.contract_id)
            contra = [c for c in claims
                      if c.verdict == "contradicted"]
            self.assertTrue(contra)
            # …and the contract still completed on the drafts that
            # survived (the delivery claim is the latest word).
            contract = contracts.get_contract(result.contract_id)
            assert contract is not None
            self.assertEqual(contract.status, "completed")
        finally:
            queue.close()

    def test_empty_day_derives_nothing(self) -> None:
        queue = self.queue()
        try:
            result = derive_day(self.context, DAY, queue=queue,
                                llm_fn=_fake_llm())
            self.assertEqual(result.drafts, [])
            self.assertIn("nothing recorded", result.note)
            self.assertEqual(reviewable_drafts(queue), [])
        finally:
            queue.close()


# ── (d) the GS section through the bundle contract ───────────────────

def _gs_bundle(journal_lines: list[Any], *,
               created_epoch: float) -> bytes:
    """A GS-shaped state bundle (the P7 fixture's shape): family
    manifest + STATE.md prose + journal; the WS1 contract manifest
    written last, recording every member."""
    journal = ("\n".join(
        line if isinstance(line, str) else json.dumps(line)
        for line in journal_lines) + "\n").encode("utf-8")
    members: dict[str, bytes] = {
        "trade_journal.jsonl": journal,
        "STATE.md": (
            b"# GoldenSwing state export\n\nThe desk made"
            b" 999999.99 today, probably. Prose is not evidence.\n"),
    }
    family = {
        "bundle": "goldenswing state export",
        "exported_utc": _iso(created_epoch),
        "files": [
            {"arcname": name, "bytes": len(data),
             "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()],
    }
    members["manifest.json"] = json.dumps(
        family, indent=2, ensure_ascii=False).encode("utf-8")
    contract = {
        "format": "nomorals-bundle", "version": 1,
        "created_by": "goldenswing", "created_at": created_epoch,
        "kind": "journal-state",
        "members": [
            {"name": name, "size": len(data),
             "sha256": hashlib.sha256(data).hexdigest(),
             "is_dir": False}
            for name, data in members.items()],
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        zf.writestr("bundle.manifest.json", json.dumps(
            contract, indent=2, ensure_ascii=False).encode("utf-8"))
    return buf.getvalue()


class TradingSectionTests(_DayCase):

    def _register_bundle(self, journal: list[Any]) -> None:
        store = ArtifactStore(
            self.db, BlobStore(self.db, self.home / "blobs"))
        data = _gs_bundle(journal, created_epoch=self.end + 3600.0)
        store.put(data, type="file", creator="goldenswing",
                  mime="application/zip")

    def test_present_with_verified_bundle(self) -> None:
        self._register_bundle([
            {"ts": _iso(self.start - 3600), "type": "close",
             "pnl_money": 999.0},                      # yesterday
            {"ts": _iso(self.at(10)), "type": "close",
             "pnl_money": 200.0},                      # the day
            {"ts": _iso(self.at(11)), "type": "open"},
        ])
        sections = {s.name: s for s in build_sections(self.context, DAY)}
        trading = sections["trading"]
        self.assertTrue(trading.present)
        self.assertEqual(trading.evidence["closed_trades_day"], 1)
        self.assertEqual(trading.evidence["realized_pnl_day"], 200.0)
        self.assertTrue(trading.evidence["bundle_verified"])
        # STATE.md's prose figure is never a reading.
        self.assertNotIn(999999.99, trading.numbers)
        self.assertNotIn(999.0, trading.numbers)

    def test_honestly_absent_without_a_bundle(self) -> None:
        sections = {s.name: s for s in build_sections(self.context, DAY)}
        trading = sections["trading"]
        self.assertFalse(trading.present)
        self.assertIn("no GoldenSwing state bundle", trading.note)


class SectionTests(_DayCase):

    def test_artifacts_and_missions_sections(self) -> None:
        store = ArtifactStore(
            self.db, BlobStore(self.db, self.home / "blobs"))
        art = store.put(b"hello", type="file", creator="chat_inbound",
                        metadata={"origin": "chat_inbound",
                                  "filename": "notes.txt"})
        self.db.execute(
            "UPDATE artifacts SET created_at = ? WHERE id = ?",
            (self.at(12), art.id))
        sections = {s.name: s for s in build_sections(
            self.context, DAY, artifact_store=store)}
        arrived = sections["artifacts"]
        self.assertTrue(arrived.present)
        self.assertIn("notes.txt", arrived.lines[0])
        self.assertIn("sent in chat", arrived.lines[0])
        missions = sections["missions"]
        self.assertFalse(missions.present)
        self.assertIn("no missions finished", missions.note)
        goals = sections["goals"]
        self.assertFalse(goals.present)

    def test_goals_section_carries_basis_verbatim(self) -> None:
        tracker = GoalTracker(self.db)
        goal = surface.run_sync(tracker.create(
            UID, "Ship the day record", goal_class="build"))
        surface.run_sync(tracker.update_progress(goal.goal_id, 40.0))
        sections = {s.name: s for s in build_sections(
            self.context, DAY, tracker=tracker)}
        goals = sections["goals"]
        self.assertTrue(goals.present)
        self.assertIn("40% (self-reported)", goals.lines[0])
        self.assertIn(40.0, goals.numbers)


# ── (e) the morning bookend ──────────────────────────────────────────

class BookendTests(_DayCase):

    def test_renders_yesterday_and_absent_without(self) -> None:
        from nomorals.agents.dayrecord import DayBookendProvider

        provider = DayBookendProvider()
        self.assertIsNone(provider.collect(self.context, 0.0))
        yday = dayrecord.yesterday_key()
        start, _end = dayrecord.day_window(yday)
        self.store.add_fragment(
            text="closed the parity program and shipped the"
                 " bookend fix",
            ts=start + 3600.0, day=yday)
        section = provider.collect(self.context, 0.0)
        assert section is not None
        self.assertEqual(section.title,
                         "🌅 Yesterday, in your words")
        self.assertIn("closed the parity program",
                      section.lines[0])
        self.assertIn("1 note(s) yesterday", section.lines[0])


# ── the evening job ──────────────────────────────────────────────────

class EveningJobTests(_DayCase):

    def test_ensure_registers_the_durable_job(self) -> None:
        scheduler = Scheduler(self.context)
        out = ensure_day_derivation_job(scheduler)
        self.assertTrue(out[0]["scheduled"])
        jobs = [j for j in scheduler.list_jobs()
                if j.get("name") == "day derivation"]
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        # The scheduler canonicalises "daily 19:00" + timezone into
        # the stored spec string; the payload body rides the row.
        self.assertIn("19:00", job["spec"])
        self.assertEqual(job["timezone"], "Africa/Lagos")
        self.assertEqual(job["payload_kind"], "tool")
        self.assertEqual(job["missed_fire_policy"], "fire_now")
        self.assertEqual(job["overlap_policy"], "skip")
        row = self.db.query(
            "SELECT payload FROM schedule_jobs WHERE name = ?",
            ("day derivation",))[0]
        payload = json.loads(row["payload"])
        self.assertEqual(payload["tool"], "day")
        self.assertEqual(payload["args"]["action"], "evening")
        again = ensure_day_derivation_job(scheduler)
        self.assertTrue(again[0]["already_scheduled"])

    def test_evening_run_derives_a_narrated_day(self) -> None:
        now = self.at(19)
        self.store.add_fragment(text="shipped the evening run",
                                ts=self.at(9))
        queue = self.queue()
        try:
            out = run_evening_derivation(
                self.context, now=now, queue=queue,
                llm_fn=_fake_llm())
            self.assertEqual(out["drafts"], 3)
            self.assertEqual(len(reviewable_drafts(queue)), 3)
            # Idempotent: a catch-up refire derives nothing twice.
            again = run_evening_derivation(
                self.context, now=now, queue=queue,
                llm_fn=_fake_llm())
            self.assertTrue(again["already_derived"])
            self.assertEqual(len(reviewable_drafts(queue)), 3)
        finally:
            queue.close()

    def test_evening_run_honours_skip_and_empty(self) -> None:
        now = self.at(19)
        today = dayrecord.today_key(now)
        queue = self.queue()
        try:
            empty = run_evening_derivation(
                self.context, now=now, queue=queue,
                llm_fn=_fake_llm())
            self.assertTrue(empty["empty"])
            self.store.add_fragment(text="a quiet day", ts=self.at(9))
            self.store.set_skip(today, skip=True)
            skipped = run_evening_derivation(
                self.context, now=now, queue=queue,
                llm_fn=_fake_llm())
            self.assertTrue(skipped["skipped"])
            self.assertEqual(reviewable_drafts(queue), [])
            # His explicit word afterwards still derives the day.
            result = derive_day(self.context, today, queue=queue,
                                llm_fn=_fake_llm())
            self.assertEqual(len(result.drafts), 3)
        finally:
            queue.close()


# ── the brain tool ───────────────────────────────────────────────────

class _FakeRegistry:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.tools: dict[str, Any] = {}

    def register(self, name: str, **kwargs: Any):
        def deco(fn: Any) -> Any:
            self.tools[name] = fn
            return fn
        return deco


class ToolRouteTests(_DayCase):

    def test_day_tool_tell_show_skip(self) -> None:
        registry = _FakeRegistry(self.context)
        dayrecord.register(registry)
        day = registry.tools["day"]
        out = day("tell", text="narrated the day from chat",
                  source="voice")
        self.assertTrue(out["ok"])
        self.assertEqual(out["fragments"], 1)
        shown = day("show")
        self.assertIn("narrated the day from chat", shown["text"])
        skipped = day("skip")
        self.assertTrue(skipped["skipped"])
        self.assertTrue(self.store.skipped(dayrecord.today_key()))
        self.assertIn("dayrecord", AGENT_TOOL_MODULES)


# ── catalog pins ─────────────────────────────────────────────────────

class CatalogTests(unittest.TestCase):

    def test_day_registered_everywhere(self) -> None:
        self.assertIn("day", CONTROL_COMMANDS)
        self.assertIn("day", COMMAND_DETAILS)
        self.assertIn("day", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("day"), 1)
        self.assertIn("/day", COMMAND_DETAILS["day"]["usage"])
        self.assertTrue(COMMAND_DETAILS["day"]["what"])
        self.assertIn("/day", detailed_help("day"))

    def test_day_is_owner_only(self) -> None:
        self.assertNotIn("day", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("day", False))
        self.assertTrue(caller_may_run("day", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/day tell shipped the thing")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "day")
        self.assertIsNotNone(parse_control("/day"))


if __name__ == "__main__":
    unittest.main()


# ── (b2)+(f) the /day surface end-to-end; nothing posts ─────────────

from test_artifact_command import (  # noqa: E402
    _FakeTools,
    _msg,
    _RecordingGateway,
)

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social import drafts as drafts_mod  # noqa: E402

_REAL_QUEUE = DraftQueue


class _FakeSocialManager:
    """The publisher seam: any publish anywhere would land here."""

    instances: list[_FakeSocialManager] = []

    def __init__(self, context: Any) -> None:
        self.context = context
        self.published: list[tuple[str, Any]] = []

    def register_builtins(self) -> _FakeSocialManager:
        _FakeSocialManager.instances.append(self)
        return self

    def publish(self, content: str, platforms: Any = None,
                actor: str = "user", confirmation: str = "") -> Any:
        self.published.append((content, platforms))
        return SimpleNamespace(
            posted=["x"], failed=[], error="",
            results=[{"platform": "x", "ok": True}])


class RuntimeDayTests(unittest.TestCase):
    """/day through the REAL partner-runtime dispatch: tell keeps
    the fragment, make derives drafts into the review queue, a
    non-owner is refused, and no publisher is ever touched."""

    def setUp(self) -> None:
        self._env = mock.patch.dict(
            os.environ, {"NM_OWNER_TZ": "Africa/Lagos"})
        self._env.start()
        self.tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        _FakeSocialManager.instances = []
        self._patchers = [
            mock.patch.object(
                drafts_mod, "DraftQueue",
                lambda *a, **k: _REAL_QUEUE(db_path=self.db_path)),
            mock.patch("nomorals.social.SocialManager",
                       _FakeSocialManager),
            mock.patch(
                "nomorals.social.voice.VoiceProfile.load",
                classmethod(lambda cls: SimpleNamespace(
                    n_posts=0, describe=lambda: "plain test voice"))),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in self._patchers:
            patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self._env.stop()
        self.tmp.cleanup()

    def _run(self, text: str, *, owner: bool = True) -> Any:
        return self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999", owner=owner))

    @staticmethod
    def _text(reply: Any) -> str:
        return reply.text if hasattr(reply, "text") else str(reply)

    def test_tell_make_and_nothing_posts(self) -> None:
        reply = self._run("/day tell shipped the day-record build"
                          " and read the bundle end to end")
        self.assertIn("kept in today's record", self._text(reply))
        store = DayStore(self.context.db)
        today = dayrecord.today_key()
        frags = store.fragments(today)
        self.assertEqual(len(frags), 1)
        self.assertEqual(frags[0]["source"], "chat")

        reply = self._run("/day make")
        text = self._text(reply)
        self.assertIn("/drafts", text)
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            review = reviewable_drafts(queue)
            self.assertEqual(len(review), 3)
            for draft in review:
                self.assertEqual(draft.status, PENDING_REVIEW)
                self.assertEqual(
                    draft.metadata["origin"], "day-derivation")
                self.assertEqual(
                    draft.metadata["source_fragments"],
                    [frags[0]["id"]])
        finally:
            queue.close()
        # (f) the whole flow published nothing, anywhere.
        for manager in _FakeSocialManager.instances:
            self.assertEqual(manager.published, [])

    def test_day_is_refused_for_non_owners(self) -> None:
        reply = self._run("/day tell a stranger's note", owner=False)
        self.assertNotIn("kept in today's record",
                         self._text(reply))
        store = DayStore(self.context.db)
        self.assertEqual(
            store.fragments(dayrecord.today_key()), [])
