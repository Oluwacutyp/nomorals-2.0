"""Formatting Pass F2 — the presentation layer core.

Pins (Phase-0 survey §E.1 / §E.2 row F2):

- one canonical document model (``present.Doc`` / ``Section`` / ``KV``)
  rendered per transport by five renderers fed by the same Doc —
  telegram-bot and telegram-personal produce IDENTICAL HTML strings
  (the F1 asymmetry cannot return through the Doc path);
- the conformance net: no literal ``**`` in telegram output, no HTML
  tags in WhatsApp/terminal output, and every chunk of a long Doc —
  split with ``format_conformance.chunk`` — is balanced and within
  the platform cap, verified with ``format_conformance.check_message``;
- schema-driven dict rendering (``from_dict`` + ``Field``): human
  labels, empty/zero noise suppressed — the finance-alert raw dump
  becomes a report-shaped document;
- Notifier adoption: ``publish(..., doc=...)`` stores title/body
  derived from the Doc (dedupe, /notify, redelivery keep working on
  strings) while delivery renders per target platform with the parse
  mode explicit and the Doc's actions as buttons — and string
  producers are byte-unaffected;
- end-to-end through the real gateway: a long Doc arrives as
  multiple chunks, EVERY chunk carrying ``parse_mode="HTML"`` and
  independently balanced.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.present import (
    Doc, Field, KV, Section, actions_rows, body_text, chunks, from_dict,
    humanize_key, render, render_text)
from nomorals.social.format_conformance import check_message, max_text_for


def _rich_doc() -> Doc:
    return Doc(
        title="Finance alerts", icon="⏰", tone="ok",
        summary="3 price alerts checked — **none** fired. "
                "See `run 42` for the [full log](https://x.test/log).",
        sections=[
            Section("Results", rows=[
                KV("Checked", 3), KV("Fired", 0),
                KV("Run id", "sched<42>", mono=True),
                KV("Report", "finance.pdf",
                   url="https://x.test/finance.pdf"),
            ]),
            Section("Detail", quote=[
                "leg 1: XAU > 2400 & rising",
                "leg 2: quiet — nothing to do",
            ], expandable=True),
            Section("Notes", lines=["first note", "second **note**"]),
        ],
        footer="next run", footer_unix=1792905600.0,
        actions=[("Jobs", "schedule list"), ("Re-run", "schedule run fin")],
        overflow_text="full archive line 1\nfull archive line 2",
    )


# ── the document model ───────────────────────────────────────────────

class ModelTests(unittest.TestCase):
    def test_unknown_tone_fails_fast(self):
        with self.assertRaises(ValueError):
            Doc(title="x", tone="loud")

    def test_fail_alias_normalizes_to_error(self):
        doc = Doc(title="x", tone="fail")
        self.assertEqual(doc.tone, "error")
        self.assertEqual(doc.resolved_icon, "❌")

    def test_explicit_icon_wins_over_tone(self):
        doc = Doc(title="x", icon="⏰", tone="ok")
        self.assertEqual(doc.resolved_icon, "⏰")
        self.assertEqual(Doc(title="x", tone="warn").resolved_icon, "⚠️")

    def test_actions_flat_pairs_become_one_row(self):
        doc = Doc(title="x", actions=[("A", "a"), ("B", "b")])
        self.assertEqual(actions_rows(doc), [[("A", "a"), ("B", "b")]])

    def test_actions_rows_preserved(self):
        doc = Doc(title="x", actions=[[("A", "a")], [("B", "b")]])
        self.assertEqual(actions_rows(doc), [[("A", "a")], [("B", "b")]])

    def test_section_rows_accept_pairs(self):
        sec = Section("S", rows=[("Checked", "3"), KV("Fired", 0)])
        self.assertEqual(sec.rows[0], KV("Checked", "3"))
        self.assertEqual(sec.rows[1].label, "Fired")


# ── the renderers ────────────────────────────────────────────────────

class RendererTests(unittest.TestCase):
    def test_telegram_bot_and_personal_identical(self):
        doc = _rich_doc()
        bot = render(doc, "telegram-bot")
        personal = render(doc, "telegram-personal")
        userbot = render(doc, "telegram")
        self.assertEqual(bot.text, personal.text)
        self.assertEqual(bot.text, userbot.text)
        self.assertEqual(bot.parse_mode, "HTML")
        self.assertEqual(personal.parse_mode, "HTML")

    def test_telegram_html_shape(self):
        text = render_text(_rich_doc(), "telegram-bot")
        self.assertIn("⏰ <b>Finance alerts</b>", text)
        # summary is first, dialect converted
        self.assertLess(text.index("price alerts checked"),
                        text.index("<b>Results</b>"))
        self.assertIn("<b>none</b>", text)
        self.assertIn("<code>run 42</code>", text)
        self.assertIn('<a href="https://x.test/log">full log</a>', text)
        # KV rows: bold labels, mono value escaped, url value linked
        self.assertIn("<b>Checked</b>: 3", text)
        self.assertIn("<code>sched&lt;42&gt;</code>", text)
        self.assertIn(
            '<a href="https://x.test/finance.pdf">finance.pdf</a>', text)
        # expandable detail quote + tg-time footer
        self.assertIn("<blockquote expandable>", text)
        self.assertIn("XAU &gt; 2400 &amp; rising", text)
        self.assertIn('<tg-time unix="1792905600"', text)
        self.assertIn("<i>next run</i>", text)
        # overflow rides a trailing expandable quote
        self.assertIn("full archive line 1", text)

    def test_structural_text_cannot_inject(self):
        doc = Doc(title='Bad <script>alert(1)</script>',
                  sections=[Section("S<b>", rows=[KV("a<b", "v<i>")])])
        text = render_text(doc, "telegram-bot")
        self.assertNotIn("<script>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertEqual(check_message("telegram", text), [])

    def test_whatsapp_native_no_html(self):
        text = render_text(_rich_doc(), "whatsapp")
        self.assertIn("⏰ *Finance alerts*", text)
        self.assertIn("*Checked*: 3", text)
        self.assertIn("```sched<42>```", text)
        self.assertNotIn("<b>", text)
        self.assertNotIn("<blockquote", text)
        self.assertEqual(check_message("whatsapp", text), [])

    def test_discord_markdown(self):
        text = render_text(_rich_doc(), "discord")
        self.assertIn("⏰ **Finance alerts**", text)
        self.assertIn("**Checked**: 3", text)
        self.assertIn("`sched<42>`", text)
        self.assertIn("> leg 1: XAU > 2400 & rising", text)
        self.assertEqual(check_message("discord", text), [])
        # web shares the markdown renderer
        self.assertEqual(render_text(_rich_doc(), "web"), text)

    def test_terminal_plain_output_style_look(self):
        text = render_text(_rich_doc(), "local")
        lines = text.splitlines()
        self.assertEqual(lines[0], "⏰ Finance alerts")
        self.assertEqual(lines[1], "━━━━━━━━━━━━━━━")
        self.assertIn("  Checked: 3", lines)
        self.assertIn("  Run id: sched<42>", lines)
        self.assertNotIn("**", text)
        self.assertNotIn("<b>", text)
        self.assertEqual(check_message("local", text), [])

    def test_unknown_platform_degrades_to_plain(self):
        r = render(_rich_doc(), "smoke-signal")
        self.assertEqual(r.parse_mode, "")
        self.assertEqual(r.text, render_text(_rich_doc(), "local"))

    def test_body_text_is_canonical_without_title(self):
        body = body_text(_rich_doc())
        self.assertNotIn("**Finance alerts**", body)
        self.assertIn("**Checked**: 3", body)
        self.assertIn("`sched<42>`", body)


# ── the conformance net ──────────────────────────────────────────────

class ConformanceTests(unittest.TestCase):
    #: platform → the name check_message/max_text_for reason about
    PLATFORMS = {
        "telegram-bot": "telegram", "telegram-personal": "telegram",
        "whatsapp": "whatsapp", "discord": "discord", "web": "web",
        "local": "local",
    }

    def test_every_renderer_chunk_conforms(self):
        doc = _rich_doc()
        for platform, check_as in self.PLATFORMS.items():
            with self.subTest(platform=platform):
                parts = chunks(doc, platform)
                self.assertTrue(parts)
                for part in parts:
                    self.assertEqual(
                        check_message(check_as, part.text), [],
                        f"{platform} chunk violates: {part.text[:80]!r}")
                    self.assertLessEqual(
                        len(part.text), max_text_for(check_as))

    def test_no_literal_stars_in_telegram_output(self):
        doc = _rich_doc()
        for part in chunks(doc, "telegram-bot"):
            self.assertNotIn("**", part.text)

    def test_long_doc_chunks_balanced_and_complete(self):
        lines = [f"ledger line {i}: moved **{i}** units & settled"
                 for i in range(300)]
        doc = Doc(title="Daily ledger", tone="info",
                  summary="300 movements today.",
                  sections=[Section("Movements", quote=lines,
                                    expandable=True)])
        parts = chunks(doc, "telegram-bot")
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertEqual(part.parse_mode, "HTML")
            self.assertEqual(check_message("telegram", part.text), [])
            self.assertLessEqual(len(part.text), 4096)
        joined = "\n".join(p.text for p in parts)
        self.assertIn("ledger line 0", joined)
        self.assertIn("ledger line 299", joined)


# ── schema-driven dict rendering ─────────────────────────────────────

class FromDictTests(unittest.TestCase):
    #: the owner's screenshot, verbatim shape
    DUMP = {"finance_alert": "", "ok": True, "delivered": True,
            "messages": [], "checked": 0, "fired": 0, "details": [],
            "skipped_snoozed": 0}

    def test_dump_becomes_report(self):
        doc = from_dict(
            {"ok": True, "delivered": True, "messages": [],
             "checked": 3, "fired": 0, "details": [],
             "skipped_snoozed": 0},
            [Field("checked"), Field("fired"),
             Field("skipped_snoozed", hide_zero=True)],
            title="Finance alerts", tone="ok")
        text = render_text(doc, "local")
        self.assertIn("Checked: 3", text)
        self.assertIn("Fired: 0", text)       # shown: not hide_zero
        self.assertIn("Ok: yes", text)
        self.assertNotIn("Messages", text)    # empty list suppressed
        self.assertNotIn("Details", text)     # empty list suppressed
        self.assertNotIn("Skipped snoozed", text)   # zero suppressed
        self.assertNotIn("skipped_snoozed", text)   # no raw keys

    def test_humanize_key(self):
        self.assertEqual(humanize_key("skipped_snoozed"), "Skipped snoozed")
        self.assertEqual(humanize_key("delivery_state"), "Delivery state")
        self.assertEqual(humanize_key("ok"), "Ok")

    def test_fmt_and_unit(self):
        doc = from_dict({"size": 12, "price": 2400.5},
                        [Field("size", label="Size", unit="MB"),
                         Field("price", fmt=lambda v: f"${v:,.2f}")],
                        title="T")
        text = render_text(doc, "local")
        self.assertIn("Size: 12 MB", text)
        self.assertIn("Price: $2,400.50", text)

    def test_broken_fmt_falls_back(self):
        def boom(_v):
            raise RuntimeError("producer bug")
        doc = from_dict({"n": 7}, [Field("n", fmt=boom)], title="T")
        self.assertIn("N: 7", render_text(doc, "local"))

    def test_nested_dict_becomes_section(self):
        doc = from_dict({"name": "desk",
                         "stats": {"wins": 2, "losses": 1, "open": []}},
                        title="T")
        text = render_text(doc, "local")
        self.assertIn("Name: desk", text)
        self.assertIn("Stats", text)
        self.assertIn("Wins: 2", text)
        self.assertNotIn("Open", text)

    def test_scalar_list_and_dict_list(self):
        doc = from_dict({"tags": ["a", "b"],
                         "legs": [{"side": "long", "qty": 2},
                                  {"side": "short", "qty": 1}]},
                        title="T")
        text = render_text(doc, "local")
        self.assertIn("Tags: a, b", text)
        self.assertIn("Side: long · Qty: 2", text)
        self.assertIn("Side: short · Qty: 1", text)

    def test_unlisted_keys_toggle(self):
        data = {"named": 1, "extra_key": 2}
        doc = from_dict(data, [Field("named")], title="T")
        self.assertIn("Extra key: 2", render_text(doc, "local"))
        doc = from_dict(data, [Field("named")], title="T",
                        include_unlisted=False)
        self.assertNotIn("Extra key", render_text(doc, "local"))

    def test_non_mapping_raises(self):
        with self.assertRaises(TypeError):
            from_dict([("a", 1)])  # type: ignore[arg-type]


# ── Notifier adoption ────────────────────────────────────────────────

class _SendResult:
    def __init__(self, ok=True):
        self.ok = ok
        self.message_id = "m1"


class _DocFakeGateway:
    """Records every send with its kwargs; telegram + whatsapp live."""

    def __init__(self):
        self.sent = []  # (platform, chat_id, text, kwargs)

    def status(self):
        return {"telegram": {"running_in_session": True},
                "whatsapp": {"running_in_session": True}}

    def send(self, platform, chat_ref, text, buttons=None,
             parse_mode=""):
        self.sent.append((platform, chat_ref.chat_id, text,
                          {"buttons": buttons, "parse_mode": parse_mode}))
        return _SendResult(ok=True)


class _ButtonsOnlyGateway(_DocFakeGateway):
    """A gateway double whose send() predates the parse_mode kwarg."""

    def send(self, platform, chat_ref, text, buttons=None):  # noqa: ARG002
        self.sent.append((platform, chat_ref.chat_id, text,
                          {"buttons": buttons, "parse_mode": "<absent>"}))
        return _SendResult(ok=True)


def _make_ctx(test, gateway, owner_chats="telegram:111,whatsapp:222"):
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="f2-notify-")
    test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    partner = SimpleNamespace(
        owner_chats=owner_chats, proactive_enabled=True,
        proactive_briefing=True, proactive_watchers=True,
        quiet_start=22, quiet_end=8, timezone="UTC")
    settings = SimpleNamespace(
        workspace_dir=tmp, partner=partner,
        resolve=lambda rel: os.path.join(tmp, rel))
    return SimpleNamespace(db=db, settings=settings,
                           extras={"gateway": gateway})


class NotifierDocTests(unittest.TestCase):
    def test_publish_doc_renders_per_platform(self):
        from nomorals.agents.notifier import Notifier

        gw = _DocFakeGateway()
        ctx = _make_ctx(self, gw)
        doc = _rich_doc()
        out = Notifier(ctx).publish_doc("finance", doc)
        self.assertTrue(out["delivered"], out)
        by_platform = {p: (t, kw) for p, _c, t, kw in gw.sent}
        tg_text, tg_kw = by_platform["telegram"]
        self.assertIn("<b>Finance alerts</b>", tg_text)
        self.assertEqual(tg_kw["parse_mode"], "HTML")
        wa_text, wa_kw = by_platform["whatsapp"]
        self.assertIn("*Finance alerts*", wa_text)
        self.assertNotIn("<b>", wa_text)
        self.assertEqual(wa_kw["parse_mode"], "")
        # the Doc's actions are the buttons on every channel
        expected = [[("Jobs", "schedule list"),
                     ("Re-run", "schedule run fin")]]
        self.assertEqual(tg_kw["buttons"], expected)
        self.assertEqual(wa_kw["buttons"], expected)
        # stored row: title/body derived from the Doc
        row = ctx.db.query_one(
            "SELECT title, body FROM notifications WHERE id = ?",
            (out["id"],))
        self.assertEqual(row["title"], "Finance alerts")
        self.assertNotIn("Finance alerts", row["body"])
        self.assertIn("**Checked**: 3", row["body"])

    def test_string_publish_byte_identical(self):
        from nomorals.agents.notifier import Notifier
        from nomorals.agents.render import banner

        gw = _DocFakeGateway()
        ctx = _make_ctx(self, gw, owner_chats="telegram:111")
        out = Notifier(ctx).publish("immune", "gate **ENFORCED**",
                                    "sample: spawned 89")
        self.assertTrue(out["delivered"], out)
        _p, _c, text, kw = gw.sent[0]
        self.assertEqual(
            text, banner("gate **ENFORCED**", "🔔") + "\nsample: spawned 89")
        self.assertEqual(kw["parse_mode"], "")   # F1 path untouched

    def test_doc_send_degrades_on_old_gateway(self):
        from nomorals.agents.notifier import Notifier

        gw = _ButtonsOnlyGateway()
        ctx = _make_ctx(self, gw, owner_chats="telegram:111")
        out = Notifier(ctx).publish_doc("finance", _rich_doc())
        self.assertTrue(out["delivered"], out)
        _p, _c, text, _kw = gw.sent[0]
        # parse_mode kwarg dropped AND the text swapped to the plain
        # rendering — never raw HTML without a parse mode.
        self.assertNotIn("<b>", text)
        self.assertIn("Finance alerts", text)
        self.assertIn("Checked: 3", text)

    def test_non_doc_doc_is_ignored_not_lost(self):
        from nomorals.agents.notifier import Notifier

        gw = _DocFakeGateway()
        ctx = _make_ctx(self, gw, owner_chats="telegram:111")
        out = Notifier(ctx).publish("k", "Plain title", "plain body",
                                    doc={"not": "a doc"})
        self.assertTrue(out["delivered"], out)
        self.assertEqual(gw.sent[0][2],
                         "🔔 **Plain title**\nplain body")

    def test_notify_doc_helper(self):
        from nomorals.agents.notifier import notify_doc

        gw = _DocFakeGateway()
        ctx = _make_ctx(self, gw, owner_chats="telegram:111")
        out = notify_doc(ctx, "finance", _rich_doc())
        self.assertTrue(out["delivered"], out)
        self.assertIn("<b>Finance alerts</b>", gw.sent[0][2])

    def test_policy_page_line_rides_doc_footer(self):
        from nomorals.agents.alert_policy import AlertHints
        from nomorals.agents.notifier import Notifier

        gw = _DocFakeGateway()
        ctx = _make_ctx(self, gw, owner_chats="telegram:111")
        doc = Doc(title="Immune gate", tone="error",
                  summary="containment failed")
        out = Notifier(ctx).publish(
            "immune", doc=doc, alert_subject="immune sampler",
            alert_hints=AlertHints(subject="immune sampler",
                                   failed=True))
        self.assertTrue(out["delivered"], out)
        text = gw.sent[0][2]
        self.assertIn("alert policy: page", text)
        self.assertIn("immune sampler", text)


# ── end-to-end through the real gateway ──────────────────────────────

class _RecordingAdapter(ChatAdapter):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.calls = []

    def run(self, handler):  # pragma: no cover - unused
        pass

    def send(self, chat, text, **kw):
        self.calls.append((text, dict(kw)))
        return SendResult(ok=True, platform=self.name, message_id="m1")

    def send_media(self, chat, media, *, caption=""):
        return SendResult(ok=True, platform=self.name, message_id="mm1")


class GatewayDocEndToEndTests(unittest.TestCase):
    def test_long_doc_every_chunk_html_and_balanced(self):
        from nomorals.agents.notifier import Notifier

        adapter = _RecordingAdapter("telegram-bot")
        gateway = ChatGateway({"telegram-bot": adapter})
        ctx = _make_ctx(self, gateway, owner_chats="telegram-bot:111")
        lines = [f"ledger line {i}: moved **{i}** units & settled"
                 for i in range(300)]
        doc = Doc(title="Daily ledger", tone="info",
                  summary="300 movements today.",
                  sections=[Section("Movements", quote=lines,
                                    expandable=True)])
        out = Notifier(ctx, gateway).publish_doc("ledger", doc)
        self.assertTrue(out["delivered"], out)
        self.assertGreater(len(adapter.calls), 1)
        for text, kw in adapter.calls:
            self.assertEqual(kw.get("parse_mode"), "HTML")
            self.assertEqual(check_message("telegram", text), [],
                             f"unbalanced chunk: {text[:80]!r}")
            self.assertNotIn("**", text)
        joined = "\n".join(t for t, _kw in adapter.calls)
        self.assertIn("ledger line 0", joined)
        self.assertIn("ledger line 299", joined)


if __name__ == "__main__":
    unittest.main()
