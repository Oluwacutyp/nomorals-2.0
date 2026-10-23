"""Formatting F7 — the sweep + polish that closes the formatting program.

Pinned here, by scope letter:

- A: every Notifier publisher family carries honest alert_policy
  classification (hints + subject) — decision replays through the
  REAL Notifier/AlertPolicy journal for the newly classified
  families; the immune family and the Pass-7 fixtures are untouched
  (that suite is the pin).
- B: DIGEST-tier rows fold into the next briefing exactly once —
  a section in the composed briefing, marked folded only after a
  CONFIRMED delivery; a failed delivery leaves the digest intact.
- C: the jobs views re-present a job row's last outcome from its
  data (never a mid-word slice of the stored contract string);
  chat tool results render through the generic Doc treatment;
  scheduler command reports publish as Docs.
- D: to_telegram converts single-* italic / ~~strike~~ / __underline__
  with conservative escaping boundaries; send_file captions ride
  the presentation layer with an explicit parse mode + the plain
  retry; the Telegram CONNECTOR aligns with the F1 transport (no
  double conversion of pre-formatted HTML).
- E: run_briefing marks its news highlights delivered through the
  same undelivered-ledger the pulses use — the same story is never
  delivered twice in one morning, in either order.
- F: Doc rows persist as doc_json and redeliver WITH structure;
  the migration is additive + idempotent against a pre-F7-shaped
  database; document-sized overflow rides as a file attachment.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.notifier import Notifier
from nomorals.core.ids import new_id
from nomorals.social.chat.platforms import to_plain, to_telegram
from nomorals.social.chat.present import (
    Doc,
    KV,
    Section,
    body_text,
    canonical_text,
    doc_from_json,
    doc_to_json,
)


# ── fixtures ─────────────────────────────────────────────────────────

class _SendResult:
    def __init__(self, ok=True, message_id="m1", error=""):
        self.ok = ok
        self.message_id = message_id
        self.error = error


class RecordingGateway:
    """Owner channel live on telegram-bot; records sends + files."""

    def __init__(self, fail=False):
        self.sent = []        # (platform, chat_id, text, kwargs)
        self.files = []       # (platform, chat, path, caption)
        self.fail = fail

    def status(self):
        return {"telegram-bot": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, **kwargs):
        if self.fail:
            return _SendResult(ok=False, error="boom (fake)")
        self.sent.append((platform, chat_ref.chat_id, text, kwargs))
        return _SendResult(ok=True)

    def send_file(self, platform, chat, path, caption=""):
        if self.fail:
            return _SendResult(ok=False, error="boom (fake)")
        with open(path, "rb") as fh:  # the temp file lives one pass
            content = fh.read()
        self.files.append((platform, chat.chat_id, path, caption, content))
        return _SendResult(ok=True)

    def platforms_for_chat_id(self, chat_id):
        return []


class BareGateway:
    """Live owner channel but NO send_file (overflow stays inline)."""

    def __init__(self):
        self.sent = []

    def status(self):
        return {"telegram-bot": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, **kwargs):
        self.sent.append((platform, chat_ref.chat_id, text, kwargs))
        return _SendResult(ok=True)

    def platforms_for_chat_id(self, chat_id):
        return []


def make_partner(**kw):
    base = dict(
        owner_chats="telegram:111",
        proactive_enabled=True,
        proactive_briefing=True,
        proactive_watchers=True,
        quiet_start=0,
        quiet_end=0,
        timezone="UTC",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def make_ctx(test, gateway=None, partner=None):
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="f7-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner or make_partner(),
        resolve=lambda rel: os.path.join(tmp, rel))
    ctx = SimpleNamespace(db=db, settings=settings, extras={})
    if gateway is not None:
        ctx.gateway = gateway
    return ctx, tmp


def journal(ctx):
    return Notifier(ctx).alert_journal()


def journal_tiers(ctx, kind):
    """Chronological (subject, tier, reason) for one kind.

    The journal reads newest-first; reasons are full sentences —
    assertions pin their opening clause via startswith.
    """
    rows = [e for e in journal(ctx) if e.get("kind") == kind]
    return [(e.get("subject"), e.get("tier"),
             str(e.get("reason") or ""))
            for e in reversed(rows)]


# ── D1: to_telegram dialect completion ───────────────────────────────

class ToTelegramDialectTests(unittest.TestCase):
    def test_italic_strike_underline_convert(self):
        self.assertEqual(to_telegram("a *little* test"),
                         "a <i>little</i> test")
        self.assertEqual(to_telegram("this is ~~gone~~ now"),
                         "this is <s>gone</s> now")
        self.assertEqual(to_telegram("an __under__ word"),
                         "an <u>under</u> word")
        self.assertEqual(
            to_telegram("**bold** and *italic* and ~~gone~~ and __under__"),
            "<b>bold</b> and <i>italic</i> and <s>gone</s> and <u>under</u>")

    def test_literal_asterisks_stay_literal(self):
        # multiplication / spaced stars are prose, not emphasis
        self.assertEqual(to_telegram("2 * 3 * 4 = 24"), "2 * 3 * 4 = 24")
        self.assertEqual(to_telegram("cost is 5 * 4 dollars"),
                         "cost is 5 * 4 dollars")
        self.assertEqual(to_telegram("*unpaired"), "*unpaired")
        self.assertEqual(to_telegram("trailing *"), "trailing *")

    def test_underscore_words_never_convert(self):
        self.assertEqual(to_telegram("the __agency_shift__ tool"),
                         "the __agency_shift__ tool")
        self.assertEqual(to_telegram("call a__b__c(x)"), "call a__b__c(x)")

    def test_spaced_markers_stay_literal(self):
        self.assertEqual(to_telegram("~~ spaced ~~"), "~~ spaced ~~")

    def test_code_spans_untouched(self):
        self.assertEqual(to_telegram("`__init__` and `*star*`"),
                         "<code>__init__</code> and <code>*star*</code>")

    def test_bold_wins_over_inner_singles(self):
        self.assertEqual(to_telegram("**bold *nested* end**"),
                         "<b>bold *nested* end</b>")

    def test_producer_html_passthrough_plus_italic(self):
        self.assertEqual(to_telegram("<b>pre</b> *it*"),
                         "<b>pre</b> <i>it</i>")

    def test_escaping_still_holds(self):
        self.assertEqual(to_telegram("Tom & Jerry *cartoon*"),
                         "Tom &amp; Jerry <i>cartoon</i>")
        self.assertEqual(to_telegram("a < b and *c < d*"),
                         "a &lt; b and <i>c &lt; d</i>")

    def test_converted_output_conforms(self):
        from nomorals.social.format_conformance import check_message

        for text in ("a *little* test", "this is ~~gone~~ now",
                     "an __under__ word",
                     "**b** *i* ~~s~~ __u__ and `code`"):
            self.assertEqual(check_message("telegram", to_telegram(text)),
                             [], text)

    def test_to_plain_strips_new_markers(self):
        self.assertIn("gone", to_plain("this is ~~gone~~ now"))
        self.assertNotIn("~~", to_plain("this is ~~gone~~ now"))
        self.assertNotIn("__", to_plain("an __under__ word"))


# ── A: classification replays ────────────────────────────────────────

class AlertClassificationReplayTests(unittest.TestCase):
    """Each newly classified family, replayed through the real
    Notifier + AlertPolicy with the family's own call shape."""

    def publish(self, ctx, kind, title, body="body", **kw):
        return Notifier(ctx).publish(kind, title, body, **kw)

    def test_ops_failure_pages_chronic_digests(self):
        hints = {"failed": True, "subject": "mission: abc"}
        gw = RecordingGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        res = self.publish(ctx, "ops", "Stuck mission needs attention",
                           alert_hints=hints, alert_subject="mission: abc")
        self.assertTrue(res["delivered"])
        sent = "\n".join(t for *_xy, t, _kw in gw.sent)
        self.assertIn("alert policy: page", sent)
        tiers = journal_tiers(ctx, "ops")
        self.assertEqual(tiers[0][:2], ("mission: abc", "page"))
        self.assertTrue(tiers[0][2].startswith("failure"))
        res2 = self.publish(
            ctx, "ops", "Role chronically failing",
            alert_hints={"notice": True, "subject": "role: researcher"},
            alert_subject="role: researcher")
        self.assertEqual(res2["delivery_state"], "digest")
        tiers = journal_tiers(ctx, "ops")
        self.assertEqual(tiers[-1][:2], ("role: researcher", "digest"))
        self.assertTrue(tiers[-1][2].startswith("noteworthy"))

    def test_monitor_watch_down_pages(self):
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        self.publish(ctx, "monitor", "Watch down: api",
                     alert_hints={"failed": True,
                                  "subject": "https://api.test"},
                     alert_subject="https://api.test")
        tiers = journal_tiers(ctx, "monitor")
        self.assertEqual(tiers[0][:2], ("https://api.test", "page"))
        self.assertTrue(tiers[0][2].startswith("failure"))

    def test_upgrade_proposal_pages_merge_digests(self):
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        self.publish(ctx, "upgrade_proposal", "Upgrade proposed: X",
                     alert_hints={"actionable": True, "subject": "X"},
                     alert_subject="X")
        self.publish(ctx, "upgrade_proposal", "Upgrade merged: X",
                     alert_hints={"notice": True, "subject": "X"},
                     alert_subject="X")
        tiers = journal_tiers(ctx, "upgrade_proposal")
        self.assertEqual(tiers[0][:2], ("X", "page"))
        self.assertTrue(tiers[0][2].startswith("owner action"))
        self.assertEqual(tiers[1][:2], ("X", "digest"))
        self.assertTrue(tiers[1][2].startswith("noteworthy"))

    def test_backup_failure_pages_queued_logs(self):
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        self.publish(ctx, "backup", "Backup failed",
                     alert_hints={"failed": True, "subject": "nightly"},
                     alert_subject="nightly")
        res = self.publish(
            ctx, "backup", "Backup queued",
            alert_hints={"self_resolving": True, "subject": "nightly"},
            alert_subject="nightly")
        self.assertEqual(res["delivery_state"], "logged")
        tiers = journal_tiers(ctx, "backup")
        self.assertEqual(tiers[0][:2], ("nightly", "page"))
        self.assertTrue(tiers[0][2].startswith("failure"))
        self.assertEqual(tiers[1][:2], ("nightly", "log"))
        self.assertTrue(tiers[1][2].startswith("self-resolving"))

    def test_scheduler_failure_pages_under_job_name(self):
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        self.publish(ctx, "schedule", "Job failed: nightly-report",
                     alert_hints={"failed": True,
                                  "subject": "nightly-report"},
                     alert_subject="nightly-report")
        tiers = journal_tiers(ctx, "schedule")
        self.assertEqual(tiers[0][:2], ("nightly-report", "page"))
        self.assertTrue(tiers[0][2].startswith("failure"))

    def test_trial_assist_outcomes(self):
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        self.publish(ctx, "trial", "Trial assist needs you",
                     alert_hints={"actionable": True,
                                  "subject": "trial assist spotify"},
                     alert_subject="trial assist spotify")
        tiers = journal_tiers(ctx, "trial")
        self.assertEqual(tiers[0][:2], ("trial assist spotify", "page"))
        self.assertTrue(tiers[0][2].startswith("owner action"))

    def test_recurrence_escalates_a_digest_family(self):
        # three notices for the same subject inside the window: the
        # third pages, named as recurrence (Pass 7's frozen rule,
        # exercised through a newly classified family).
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        for i in range(3):
            self.publish(ctx, "backup", f"Drill note {i}",
                         alert_hints={"notice": True, "subject": "drill"},
                         alert_subject="drill")
        tiers = journal_tiers(ctx, "backup")
        self.assertEqual([t[1] for t in tiers],
                         ["digest", "digest", "page"])
        self.assertTrue(tiers[-1][2].startswith("recurrence"))

    def test_unclassified_publish_unchanged(self):
        # standing content (desks, owner-asked reports) carries no
        # hints and journals nothing — pre-F7 behavior exactly.
        ctx, _ = make_ctx(self, gateway=RecordingGateway())
        res = self.publish(ctx, "price_alert", "XAUUSD crossed 2400")
        self.assertTrue(res["delivered"])
        self.assertEqual(Notifier(ctx).alert_journal(), [])


class ProducerHintShapeTests(unittest.TestCase):
    """The producers' own hint wiring (captured at their call sites)."""

    def test_missions_stalled_pages_terminal_failed_pages(self):
        from nomorals.missions.progress import MissionMilestones

        calls = []

        class FakeNotifier:
            def publish(self, kind, title, body, **kw):
                calls.append((kind, title, kw))
                return {"delivered": True, "delivery_state": "sent"}

        class FakeStore:
            def detail(self, mission_id):
                return {}

            def save(self, mission):
                pass

        ctx = SimpleNamespace(db=None, settings=None, extras={})
        mm = MissionMilestones(ctx, store=FakeStore())
        mm._notifier = FakeNotifier()
        mission = SimpleNamespace(
            name="deep research", id="m1", goal="find things",
            state={"stall": {"code": "waiting_input",
                             "message": "needs a key"}},
            budget_wall=0.0, budget_tokens=0)

        mm.on_stalled(mission)
        self.assertTrue(calls, "on_stalled published nothing")
        kind, _title, kw = calls[-1]
        self.assertEqual(kind, "mission")
        self.assertTrue(kw["alert_hints"]["actionable"])
        self.assertEqual(kw["alert_hints"]["subject"],
                         "mission deep research")

        calls.clear()
        mm.on_terminal(mission, "failed", error="boom")
        _kind, _title, kw = calls[-1]
        self.assertTrue(kw["alert_hints"]["failed"])
        self.assertEqual(kw["alert_hints"]["subject"],
                         "mission deep research")

        calls.clear()
        mm.on_terminal(mission, "done")
        _kind, _title, kw = calls[-1]
        self.assertIsNone(kw.get("alert_hints"))

    def test_checkpoints_request_pages_as_actionable(self):
        from nomorals.connectors.checkpoints import (
            CheckpointKind,
            HumanCheckpointPending,
            request_human_action,
        )

        gw = RecordingGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        with self.assertRaises(HumanCheckpointPending):
            request_human_action(
                "spotify", CheckpointKind.CAPTCHA, "Confirm the captcha",
                "Solve the wall at step 2.", db=ctx.db, context=ctx)
        tiers = journal_tiers(ctx, "checkpoint")
        self.assertTrue(tiers, "no checkpoint decision journaled")
        self.assertEqual(tiers[0][:2],
                         ("connector checkpoint spotify", "page"))
        self.assertTrue(tiers[0][2].startswith("owner action"))


# ── B: digest folded into briefings ──────────────────────────────────

def seed_digest(ctx, titles=("Disk almost full", "Drill finished")):
    n = Notifier(ctx)
    for t in titles:
        n.publish("ops", t, "detail body",
                  alert_hints={"notice": True, "subject": t},
                  alert_subject=t)
    return n


class DigestFoldTests(unittest.TestCase):
    def test_digest_becomes_a_briefing_section(self):
        from nomorals.agents.morning_briefing import BriefingComposer

        ctx, _ = make_ctx(self)  # no gateway: rows sit at digest
        seed_digest(ctx)
        briefing = BriefingComposer().compose(ctx, "2026-10-23")
        names = [s.name for s in briefing.sections]
        self.assertIn("alert_digest", names)
        sec = next(s for s in briefing.sections if s.name == "alert_digest")
        self.assertEqual(sec.title, "🧾 Alert digest")
        lines = [i["title"] for i in sec.items]
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("Disk almost full" in ln for ln in lines))
        text = briefing.render_text()
        self.assertIn("🧾 Alert digest", text)

    def test_empty_digest_no_section(self):
        from nomorals.agents.morning_briefing import BriefingComposer

        ctx, _ = make_ctx(self)
        briefing = BriefingComposer().compose(ctx, "2026-10-23")
        self.assertNotIn("alert_digest",
                         [s.name for s in briefing.sections])

    def test_delivered_briefing_folds_exactly_once(self):
        from nomorals.agents import morning_briefing as mb

        gw = RecordingGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        seed_digest(ctx)
        res = mb.run_briefing(ctx)
        self.assertTrue(res["delivered"])
        self.assertEqual(Notifier(ctx).digest_unfolded(), [])
        # the digest rows remain stored (visible on /notify history)
        states = [r["delivery_state"] for r in Notifier(ctx).recent(10)]
        self.assertIn("digest", states)
        # exactly once: the section rode the delivered briefing, and
        # a freshly composed briefing has no digest section anymore
        sent = "\n".join(t for *_xy, t, _kw in gw.sent)
        self.assertEqual(sent.count("🧾 Alert digest"), 1)
        from nomorals.agents.morning_briefing import BriefingComposer

        briefing = BriefingComposer().compose(ctx, "2026-10-24")
        self.assertNotIn("alert_digest",
                         [s.name for s in briefing.sections])

    def test_failed_delivery_does_not_consume(self):
        from nomorals.agents import morning_briefing as mb

        ctx, _ = make_ctx(self)  # no gateway at all → persist-only
        seed_digest(ctx)
        res = mb.run_briefing(ctx)
        self.assertFalse(res["delivered"])
        self.assertEqual(len(Notifier(ctx).digest_unfolded()), 2)

    def test_digest_lines_shape(self):
        from nomorals.agents.morning_briefing import alert_digest_lines

        rows = [{"title": "Backup drill done", "kind": "backup"},
                {"title": "  spaced   title ", "kind": ""},
                {"title": "", "kind": "ops"}]
        self.assertEqual(alert_digest_lines(rows),
                         ["• Backup drill done (backup)",
                          "• spaced title"])


# ── E: run_briefing shares the news ledger ───────────────────────────

def seed_stories(ctx, n, start=0):
    from nomorals.agents.news import NewsAgent

    NewsAgent(ctx)
    urls = []
    for i in range(start, start + n):
        url = f"https://news.test/s{i}/full"
        ctx.db.execute(
            "INSERT INTO news_items (id, source, publisher, title, url, "
            "summary, published, created_at, delivery_state) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (new_id(), "testfeed", "TestWire", f"Story {i} headline",
             url, "", 1000.0 + i, time.time(), "undelivered"))
        urls.append(url)
    return urls


class BriefingNewsLedgerTests(unittest.TestCase):
    def test_run_briefing_marks_its_highlights(self):
        from nomorals.agents import morning_briefing as mb
        from nomorals.agents.news import NewsAgent

        gw = RecordingGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        # the highlights pick caps at MAX_PER_SOURCE=3 per publisher;
        # seed exactly the cap so every seeded story is shown — the
        # pin is that SHOWN stories are marked, by construction.
        urls = seed_stories(ctx, 3)
        res = mb.run_briefing(ctx)
        self.assertTrue(res["delivered"])
        remaining = {str(r.get("url")) for r in NewsAgent(ctx).undelivered()}
        self.assertTrue(remaining.isdisjoint(set(urls)),
                        f"briefing stories still undelivered: {remaining}")

    def test_failed_briefing_marks_nothing(self):
        from nomorals.agents import morning_briefing as mb
        from nomorals.agents.news import NewsAgent

        ctx, _ = make_ctx(self)  # no gateway → delivery cannot confirm
        urls = seed_stories(ctx, 3)
        res = mb.run_briefing(ctx)
        self.assertFalse(res["delivered"])
        remaining = {str(r.get("url")) for r in NewsAgent(ctx).undelivered()}
        self.assertEqual(remaining, set(urls))

    def test_pulse_marked_stories_skip_the_briefing(self):
        # reverse order: the pulse delivered them first — the
        # briefing's highlights must not repeat them.
        from nomorals.agents import morning_briefing as mb
        from nomorals.agents.news import NewsAgent

        gw = RecordingGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        urls = seed_stories(ctx, 3)
        NewsAgent(ctx).mark_delivered(urls, channel="pulse-morning")
        res = mb.run_briefing(ctx)
        self.assertTrue(res["delivered"])
        sent = "\n".join(t for *_xy, t, _kw in gw.sent)
        self.assertNotIn("Story 0 headline", sent)


# ── F1: doc_json storage + redelivery ────────────────────────────────

def sample_doc():
    return Doc(
        title="Nightly backup", tone="ok", summary="Backup shipped.",
        sections=[Section("Facts", rows=[
            KV("Size", "1.2 GB"), KV("When", "tonight")]),
            Section("Detail", quote=["manifest verified",
                                     "release pinned"],
                    expandable=True)],
        footer="backup • nightly", footer_unix=1760000000)


class DocJsonTests(unittest.TestCase):
    def test_round_trip_all_field_kinds(self):
        doc = Doc(
            title="T", tone="warn", summary="S",
            sections=[Section("Sec", rows=[
                KV("plain", "v"), KV("num", 42), KV("mono", "x",
                                                     mono=True),
                KV("link", "label", url="https://x.test")])],
            footer="f", footer_unix=1760000000,
            overflow_text="long tail")
        back = doc_from_json(doc_to_json(doc))
        self.assertIsNotNone(back)
        self.assertEqual(canonical_text(back), canonical_text(doc))
        self.assertEqual(back.footer_unix, 1760000000)
        self.assertEqual(back.overflow_text, "long tail")

    def test_corrupt_json_is_none_never_raise(self):
        for bad in ("", "not json", "[]", "{}", '{"format": 99}',
                    '{"format": 1, "title": 5}'):
            self.assertIsNone(doc_from_json(bad), bad)
        self.assertIsNone(doc_from_json(None))

    def test_doc_row_stores_json_and_redelivers_with_structure(self):
        gw = RecordingGateway()
        ctx, _ = make_ctx(self)  # publish with NO gateway → pending
        n = Notifier(ctx)
        res = n.publish("backup", "Nightly backup", "string body",
                        doc=sample_doc())
        self.assertEqual(res["delivery_state"], "pending")
        row = ctx.db.query_one(
            "SELECT * FROM notifications WHERE id=?", (res["id"],))
        self.assertTrue(row["doc_json"])
        # redeliver through a delivering gateway: the Doc path runs
        ctx.gateway = gw
        sent_count = Notifier(ctx).redeliver()
        self.assertGreaterEqual(sent_count, 1)
        self.assertEqual(len(gw.sent), 1)
        _platform, _chat, text, kwargs = gw.sent[0]
        self.assertEqual(kwargs.get("parse_mode"), "HTML")
        self.assertIn("<blockquote expandable>", text)
        self.assertIn("<tg-time", text)

    def test_string_row_redelivers_as_before(self):
        gw = RecordingGateway()
        ctx, _ = make_ctx(self)
        n = Notifier(ctx)
        res = n.publish("backup", "Plain note", "plain body text")
        row = ctx.db.query_one(
            "SELECT * FROM notifications WHERE id=?", (res["id"],))
        self.assertFalse(row["doc_json"])
        ctx.gateway = gw
        Notifier(ctx).redeliver()
        self.assertEqual(len(gw.sent), 1)
        _platform, _chat, text, kwargs = gw.sent[0]
        self.assertNotIn("parse_mode", kwargs)
        self.assertIn("plain body text", text)


class MigrationFixtureTests(unittest.TestCase):
    def test_pre_f7_database_migrates_additively(self):
        from nomorals.storage.db import Database

        tmp = tempfile.mkdtemp(prefix="f7-mig-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "old.db")
        # Build a genuine pre-F7 database: run the full chain, then
        # remove exactly F7's footprint (the two columns + the
        # migration-88 record).  Everything else is production shape.
        db = Database(path)
        db.migrate()
        raw = sqlite3.connect(path)
        raw.execute("DELETE FROM schema_migrations WHERE version=88")
        raw.execute("ALTER TABLE notifications DROP COLUMN doc_json")
        raw.execute(
            "ALTER TABLE notifications DROP COLUMN digest_folded_at")
        raw.execute(
            "INSERT INTO notifications (id, kind, title, body, "
            "created_at, delivered) VALUES ('old1', 'ops', 'Old alert', "
            "'old body', 1700000000, 0)")
        raw.commit()
        raw.close()

        db2 = Database(path)
        db2.migrate()
        cols = {r["name"]
                for r in db2.query("PRAGMA table_info(notifications)")}
        self.assertIn("doc_json", cols)
        self.assertIn("digest_folded_at", cols)
        old = db2.query_one("SELECT * FROM notifications WHERE id='old1'")
        self.assertIsNone(old["doc_json"])
        self.assertIsNone(old["digest_folded_at"])
        self.assertEqual(old["body"], "old body")
        # idempotent: a second migrate applies nothing new
        summary = db2.migrate()
        self.assertEqual(summary.applied, [])
        # new writes against the migrated DB carry the Doc
        ctx, _tmp2 = make_ctx(self)
        ctx.db = db2
        res = Notifier(ctx).publish("ops", "Fresh", "new body",
                                    doc=sample_doc())
        row = db2.query_one(
            "SELECT * FROM notifications WHERE id=?", (res["id"],))
        self.assertTrue(row["doc_json"])


# ── F2: overflow → document route ────────────────────────────────────

class OverflowRouteTests(unittest.TestCase):
    BIG = "detail line\n" * 6000  # ≈72k chars — document territory

    def big_doc(self):
        return Doc(title="Huge report", summary="Too big for a quote.",
                   overflow_text=self.BIG)

    def test_overflow_rides_as_file_when_supported(self):
        gw = RecordingGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        res = Notifier(ctx).publish("report", "Huge report", "body",
                                    doc=self.big_doc())
        self.assertTrue(res["delivered"])
        self.assertEqual(len(gw.files), 1)
        _platform, _chat, _path, caption, content = gw.files[0]
        self.assertIn("long output", caption)
        self.assertEqual(content.decode(), self.BIG)
        sent = "\n".join(t for *_xy, t, _kw in gw.sent)
        self.assertIn("full detail attached as a document", sent)
        self.assertNotIn("detail line\ndetail line", sent)

    def test_overflow_inline_without_file_support(self):
        gw = BareGateway()
        ctx, _ = make_ctx(self, gateway=gw)
        res = Notifier(ctx).publish("report", "Huge report", "body",
                                    doc=self.big_doc())
        self.assertTrue(res["delivered"])
        sent = "\n".join(t for *_xy, t, _kw in gw.sent)
        self.assertIn("detail line", sent)


# ── C: job views + tool results + command reports ────────────────────

class SummarizeLastResultTests(unittest.TestCase):
    def test_tool_header_folds_to_first_fact(self):
        from nomorals.agents.job_reports import summarize_last_result

        job = {"last_result":
               "web_search:\nok: yes\nResults: 5\n- first hit"}
        self.assertEqual(summarize_last_result(job), "✅ Results: 5")

    def test_failure_marks_and_names(self):
        from nomorals.agents.job_reports import summarize_last_result

        job = {"last_result": "job failed: boom\n(timed out)"}
        self.assertTrue(summarize_last_result(job).startswith("❌ job failed"))

    def test_command_shape_and_plain(self):
        from nomorals.agents.job_reports import summarize_last_result

        self.assertEqual(
            summarize_last_result({"last_result": "exit=0\nall good"}),
            "✅ exit=0")
        self.assertEqual(
            summarize_last_result({"last_result": "plain one liner"}),
            "✅ plain one liner")

    def test_empty_and_markless(self):
        from nomorals.agents.job_reports import summarize_last_result

        self.assertEqual(summarize_last_result({}), "")
        self.assertEqual(summarize_last_result({"last_result": "  "}), "")
        self.assertEqual(
            summarize_last_result({"last_result": "exit=0\nall good"},
                                  with_mark=False),
            "exit=0")


class ToolResultDocTests(unittest.TestCase):
    def test_generic_mapping_becomes_rows_not_a_dump(self):
        from nomorals.agents.job_reports import (
            job_report_doc,
            tool_result_doc,
        )

        value = {"ok": True, "query": "gold price", "results": [1, 2, 3],
                 "took_ms": 41}
        text = canonical_text(tool_result_doc("web_search", value))
        # structured rows with humanized labels — never the raw
        # render_tool_value dump (no python-dict keys, no braces)
        self.assertIn("gold price", text)
        self.assertIn("**Query**: gold price", text)
        self.assertNotIn("took_ms", text)
        self.assertNotIn("'query':", text)
        # and it is EXACTLY the F3 generic treatment (the scheduled
        # report for the same value), modulo the title — the C2
        # convergence claim, pinned as equality
        job_text = canonical_text(job_report_doc("web_search", value))
        self.assertEqual(
            text.replace("**web_search**", "**scheduled: web_search**"),
            job_text)

    def test_string_and_scalar_values(self):
        from nomorals.agents.job_reports import tool_result_doc

        self.assertEqual(
            tool_result_doc("ping", "pong").summary, "pong")
        self.assertEqual(tool_result_doc("count", 7).summary, "7")
        doc = tool_result_doc("empty", None)
        self.assertEqual(doc.summary, "No output.")


class CommandReportTests(unittest.TestCase):
    def make_scheduler(self, value=None, ok=True):
        from nomorals.agents.scheduler import Scheduler

        outcome = SimpleNamespace(
            ok=ok, value=value or {"exit_code": 0,
                                    "stdout": "line one\nline two",
                                    "stderr": ""},
            error=None)
        tools = SimpleNamespace(call=lambda name, **kw: outcome)
        ctx = SimpleNamespace(tools=tools, db=None, settings=None)
        return Scheduler(ctx)

    def test_command_report_doc_and_stored_string(self):
        sched = self.make_scheduler()
        text, doc = sched._run_command_report(
            {"command": "echo hi"}, job_name="greeter")
        self.assertEqual(text, "exit=0\nline one\nline two")
        self.assertIsNotNone(doc)
        rendered = canonical_text(doc)
        self.assertIn("scheduled: greeter", rendered)
        self.assertIn("Output", rendered)
        self.assertIn("line one", rendered)
        self.assertIn("echo hi", rendered)

    def test_run_command_keeps_the_stored_string(self):
        sched = self.make_scheduler()
        self.assertEqual(sched._run_command({"command": "echo hi"}),
                         "exit=0\nline one\nline two")

    def test_failing_command_still_declares(self):
        from nomorals.agents.scheduler import _DeclaredFailure

        sched = self.make_scheduler(
            value={"exit_code": 3, "stdout": "", "stderr": "bad flag"})
        with self.assertRaises(_DeclaredFailure):
            sched._run_command_report({"command": "ls --nope"})

    def test_command_failure_doc_names_the_failure(self):
        from nomorals.agents.job_reports import command_failure_doc

        doc = command_failure_doc("greeter", "ls --nope",
                                  "command failed: exit=3: bad flag")
        self.assertEqual(doc.tone, "error")
        text = canonical_text(doc)
        self.assertIn("command failed: exit=3: bad flag", text)
        self.assertIn("ls --nope", text)


class ProducerDataBoundaryTests(unittest.TestCase):
    """The layer's data contract, pinned (F7 scope F4 residue).

    KV values are DATA, shown literally on the plain/markdown
    dialects (the F2 contract): a value that happens to be
    tag-shaped (``<prices>``) is the owner's data, not leaked
    markup — it renders escaped on HTML, and producers holding
    code-shaped values mark them ``mono`` (backticks/code make the
    value inert on every dialect).  The conformance gate therefore
    iterates producer fixtures whose values are prose-shaped; this
    pin documents the boundary instead of hiding it.
    """

    def test_tag_shaped_value_is_data_not_markup(self):
        from nomorals.social.chat.present import render

        doc = Doc(title="Search", sections=[Section(rows=[
            KV("Query", "gold <prices> today")])])
        html = render(doc, "telegram-bot").text
        self.assertIn("gold &lt;prices&gt; today", html)
        plain = render(doc, "local").text
        self.assertIn("gold <prices> today", plain)
        mono = Doc(title="Search", sections=[Section(rows=[
            KV("Query", "gold <prices> today", mono=True)])])
        md = render(mono, "discord").text
        self.assertIn("`gold <prices> today`", md)


# ── D2: send_file caption plumbing ───────────────────────────────────

class _CaptionAdapter:
    """A ChatAdapter double recording media calls verbatim."""

    def __init__(self, name="telegram-bot", with_parse_mode=True):
        self.name = name
        self.media_calls = []
        self.with_parse_mode = with_parse_mode

    def status(self):
        return {"running_in_session": True}


def _caption_adapter_class():
    from nomorals.social.chat.base import ChatAdapter, SendResult

    class Adapter(ChatAdapter):
        def __init__(self, name="telegram-bot"):
            super().__init__()
            self.name = name
            self.media_calls = []
            self.text_calls = []

        def run(self, handler):  # pragma: no cover - unused
            pass

        def send(self, chat, text, **kw):
            self.text_calls.append((text, dict(kw)))
            return SendResult(ok=True, platform=self.name, message_id="t1")

        def send_media(self, chat, media, *, caption="", parse_mode=""):
            self.media_calls.append((media.path, caption, parse_mode))
            return SendResult(ok=True, platform=self.name, message_id="m1")

    return Adapter


class CaptionGatewayTests(unittest.TestCase):
    def _file(self):
        tmp = tempfile.mkdtemp(prefix="f7-cap-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "pic.bin")
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        return path

    def test_dialect_caption_converted_with_parse_mode(self):
        from nomorals.social.chat.base import ChatRef
        from nomorals.social.chat.gateway import ChatGateway

        adapter = _caption_adapter_class()()
        gw = ChatGateway({"telegram-bot": adapter})
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = gw.send_file("telegram-bot", chat, self._file(),
                           caption="Deploy **done** — see *notes*")
        self.assertTrue(res.ok)
        self.assertEqual(len(adapter.media_calls), 1)
        _path, caption, mode = adapter.media_calls[0]
        self.assertEqual(mode, "HTML")
        self.assertIn("<b>done</b>", caption)
        self.assertIn("<i>notes</i>", caption)
        self.assertNotIn("**", caption)

    def test_plain_caption_untouched(self):
        from nomorals.social.chat.base import ChatRef
        from nomorals.social.chat.gateway import ChatGateway

        adapter = _caption_adapter_class()()
        gw = ChatGateway({"telegram-bot": adapter})
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = gw.send_file("telegram-bot", chat, self._file(),
                           caption="a plain caption")
        self.assertTrue(res.ok)
        _path, caption, mode = adapter.media_calls[0]
        self.assertEqual(caption, "a plain caption")
        self.assertEqual(mode, "")


class BotAdapterCaptionTests(unittest.TestCase):
    def _adapter(self, fail_first=False):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        ad = TelegramBotAdapter(token="<redacted>", session=object())
        calls = []

        def fake_upload(method, data, field, path):
            calls.append((method, dict(data)))
            if fail_first and len(calls) == 1:
                raise Exception(
                    "Bad Request: can't parse entities: Unsupported "
                    "start tag \"x\" at byte offset 3")
            return {"message_id": 7}

        ad._api_upload = fake_upload
        return ad, calls

    def _media(self):
        from nomorals.social.chat.base import MediaRef

        tmp = tempfile.mkdtemp(prefix="f7-media-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "doc.txt")
        with open(path, "w") as fh:
            fh.write("hello")
        return MediaRef(path=path, mime="text/plain", kind="document",
                        name="doc.txt")

    def test_parse_mode_forwarded_for_caption(self):
        from nomorals.social.chat.base import ChatRef

        ad, calls = self._adapter()
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = ad.send_media(chat, self._media(), caption="<b>hi</b>",
                            parse_mode="HTML")
        self.assertTrue(res.ok)
        self.assertEqual(calls[0][1].get("parse_mode"), "HTML")
        self.assertEqual(calls[0][1].get("caption"), "<b>hi</b>")

    def test_entity_rejection_retries_plain(self):
        from nomorals.social.chat.base import ChatRef

        ad, calls = self._adapter(fail_first=True)
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = ad.send_media(chat, self._media(), caption="<b>hi</b> there",
                            parse_mode="HTML")
        self.assertTrue(res.ok)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("parse_mode", calls[1][1])
        self.assertEqual(calls[1][1].get("caption"), "hi there")

    def test_no_parse_mode_no_retry_magic(self):
        from nomorals.social.chat.base import ChatRef

        ad, calls = self._adapter(fail_first=True)
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = ad.send_media(chat, self._media(), caption="plain")
        self.assertFalse(res.ok)
        self.assertEqual(len(calls), 1)


# ── D3: telegram connector alignment ─────────────────────────────────

class ConnectorAlignTests(unittest.TestCase):
    def _connector(self):
        from nomorals.accounts.vault import CredentialVault
        from nomorals.connectors.telegram import TelegramConnector
        from nomorals.storage.db import Database

        vault = CredentialVault(Database(":memory:"),
                                master_passphrase="test")
        conn = TelegramConnector(vault, http=SimpleNamespace())
        calls = []

        def fake_api(method, params=None, **kw):
            calls.append((method, dict(params or {})))
            return {"message_id": 1}

        conn._api = fake_api
        return conn, calls

    def test_dialect_text_converted(self):
        conn, calls = self._connector()
        conn.send_message(111, "Deploy **done** at *last*")
        _method, params = calls[0]
        self.assertEqual(params["parse_mode"], "HTML")
        self.assertIn("<b>done</b>", params["text"])
        self.assertIn("<i>last</i>", params["text"])

    def test_preformatted_html_not_double_converted(self):
        conn, calls = self._connector()
        # Producer HTML spans return verbatim; a bare & in the prose
        # between them is escaped exactly once — the same treatment
        # the chat gateway gives pre-formatted text (F1).  The
        # conversion must never re-convert the spans themselves
        # (&lt;b&gt;) or double-escape (&amp;amp;).
        conn.send_message(111, "<b>already</b> formatted & ready")
        _method, params = calls[0]
        self.assertEqual(params["text"],
                         "<b>already</b> formatted &amp; ready")
        self.assertEqual(params["parse_mode"], "HTML")

    def test_plain_text_untouched_no_parse_mode(self):
        conn, calls = self._connector()
        conn.send_message(111, "just a plain note")
        _method, params = calls[0]
        self.assertEqual(params["text"], "just a plain note")
        self.assertNotIn("parse_mode", params)

    def test_explicit_parse_mode_wins(self):
        conn, calls = self._connector()
        conn.send_message(111, "**raw** stays raw",
                          parse_mode="MarkdownV2")
        _method, params = calls[0]
        self.assertEqual(params["text"], "**raw** stays raw")
        self.assertEqual(params["parse_mode"], "MarkdownV2")


if __name__ == "__main__":
    unittest.main()
