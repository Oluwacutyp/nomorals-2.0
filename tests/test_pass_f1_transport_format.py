"""Formatting Pass F1 — transport truth fix.

The bug (Phase-0 survey, §A): producers write one markdown-ish dialect
(``agents/render.py::banner`` wraps every notification title in
``**…**``) but the push path — Notifier → gateway → adapter — never
passed ``parse_mode`` and never converted, so the Bot API printed the
asterisks literally in the bot chat while Telethon's markdown default
rendered the same bytes bold in Saved Messages. Two more traps: the
gateway applied a caller's parse mode ONLY to the last chunk, and
nothing anywhere converted the dialect.

Pinned here:

- the converter (``platforms.to_telegram``) upgrades the house dialect
  to Bot API HTML and escapes everything else — user-derived content
  with ``<``, ``>``, ``&`` or stray asterisks can neither break the
  entity parse nor inject tags;
- the gateway presents dialect text for the Telegram family (bot AND
  personal) with ``parse_mode="HTML"`` — zero producer changes;
- EVERY chunk of a multi-chunk formatted send carries the parse mode,
  and every chunk is independently balanced HTML;
- unformatted callers are byte-unaffected (plain text, no parse mode);
- an explicit caller parse mode is never second-guessed;
- the Bot API rejecting the markup ("can't parse entities") retries
  the message as PLAIN text — a formatting failure never eats an
  alert; other failures are not masked by the fallback;
- Pass 7 alert tiers are untouched: LOG/DIGEST rows are stored, never
  delivered, even through the real gateway.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
from nomorals.social.chat.chunking import strip_label, utf16_len
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.platforms import to_telegram
from nomorals.social.chat.telegram import TelegramAdapter, TelegramBotAdapter


# ── converter ────────────────────────────────────────────────────────

class ConverterTests(unittest.TestCase):
    def test_banner_becomes_html_bold(self):
        self.assertEqual(to_telegram("🔔 **immune gate ENFORCED**"),
                         "🔔 <b>immune gate ENFORCED</b>")

    def test_plain_text_unchanged(self):
        self.assertEqual(to_telegram("deploy finished, rc=0"),
                         "deploy finished, rc=0")

    def test_user_content_escaped(self):
        out = to_telegram("temp 5 < 6 & 7 > 2")
        self.assertEqual(out, "temp 5 &lt; 6 &amp; 7 &gt; 2")

    def test_escaping_inside_spans(self):
        self.assertEqual(to_telegram("**a < b & c**"),
                         "<b>a &lt; b &amp; c</b>")
        self.assertEqual(to_telegram("run `rm <x> & go`"),
                         "run <code>rm &lt;x&gt; &amp; go</code>")

    def test_no_tag_injection_from_content(self):
        out = to_telegram("look: <script>alert(1)</script> **x**")
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertIn("<b>x</b>", out)

    def test_stray_asterisks_stay_literal(self):
        self.assertEqual(to_telegram("stray ** here"), "stray ** here")
        self.assertEqual(to_telegram("2 * 3 * 4"), "2 * 3 * 4")

    def test_snake_case_underscores_untouched(self):
        self.assertEqual(to_telegram("__immune_sampler__ tool"),
                         "__immune_sampler__ tool")

    def test_link_conversion_escapes_url(self):
        out = to_telegram("[docs](https://x.test/?a=1&b=2)")
        self.assertEqual(
            out, '<a href="https://x.test/?a=1&amp;b=2">docs</a>')

    def test_producer_html_passes_through(self):
        self.assertEqual(to_telegram("pre <b>authored</b> text"),
                         "pre <b>authored</b> text")
        out = to_telegram("pre <b>authored</b> and **md**")
        self.assertEqual(out, "pre <b>authored</b> and <b>md</b>")

    def test_code_inside_bold_nests_correctly(self):
        self.assertEqual(
            to_telegram("**bold with `a<b` inside**"),
            "<b>bold with <code>a&lt;b</code> inside</b>")

    def test_fence_becomes_pre_escaped(self):
        self.assertEqual(to_telegram("```\nx < 1\n```"),
                         "<pre>x &lt; 1\n</pre>")


# ── gateway presentation ─────────────────────────────────────────────

class _RecordingAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.calls: list[tuple[str, dict]] = []

    def run(self, handler):  # pragma: no cover - unused
        pass

    def send(self, chat, text, **kw):
        self.calls.append((text, dict(kw)))
        return SendResult(ok=True, platform=self.name, message_id="m1")

    def send_media(self, chat, media, *, caption=""):
        return SendResult(ok=True, platform=self.name, message_id="mm1")


def _gateway(name: str = "telegram-bot", **kw):
    adapter = _RecordingAdapter(name)
    return ChatGateway({name: adapter}, **kw), adapter


class GatewayPresentationTests(unittest.TestCase):
    def test_dialect_converted_with_parse_mode_bot(self):
        gw, ad = _gateway("telegram-bot")
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = gw.send("telegram-bot", chat, "🔔 **Alert** body `x<y`")
        self.assertTrue(res.ok)
        self.assertEqual(len(ad.calls), 1)
        text, kw = ad.calls[0]
        self.assertEqual(text, "🔔 <b>Alert</b> body <code>x&lt;y</code>")
        self.assertEqual(kw.get("parse_mode"), "HTML")

    def test_dialect_converted_with_parse_mode_personal(self):
        gw, ad = _gateway("telegram")
        chat = ChatRef(platform="telegram", chat_id="111")
        gw.send("telegram", chat, "**Saved** message")
        text, kw = ad.calls[0]
        self.assertEqual(text, "<b>Saved</b> message")
        # Explicit — the personal adapter must not fall back to
        # Telethon's implicit markdown default ever again.
        self.assertEqual(kw.get("parse_mode"), "HTML")

    def test_plain_caller_unaffected(self):
        gw, ad = _gateway("telegram-bot")
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        plain = "deploy finished rc=0 — all good, 2 * 3 = 6"
        gw.send("telegram-bot", chat, plain)
        text, kw = ad.calls[0]
        self.assertEqual(text, plain)
        self.assertFalse(kw.get("parse_mode"))

    def test_explicit_parse_mode_never_second_guessed(self):
        gw, ad = _gateway("telegram-bot")
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        gw.send("telegram-bot", chat, "**raw** stays raw",
                parse_mode="HTML")
        text, kw = ad.calls[0]
        self.assertEqual(text, "**raw** stays raw")
        self.assertEqual(kw.get("parse_mode"), "HTML")

    def test_non_html_platforms_untouched(self):
        for name in ("discord", "local"):
            gw, ad = _gateway(name)
            chat = ChatRef(platform=name, chat_id="111")
            gw.send(name, chat, "**md** dialect")
            text, kw = ad.calls[0]
            self.assertEqual(text, "**md** dialect", name)
            self.assertFalse(kw.get("parse_mode"), name)

    def test_every_chunk_carries_parse_mode_and_is_balanced(self):
        gw, ad = _gateway("telegram-bot")
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        para = "**Section** " + ("body text with `code` and **bold** "
                                 "spans. " * 12) + "\n\n"
        text = para * 30  # ~29k chars → several chunks
        res = gw.send("telegram-bot", chat, text)
        self.assertTrue(res.ok)
        self.assertGreater(len(ad.calls), 2)
        for chunk_text, kw in ad.calls:
            self.assertEqual(kw.get("parse_mode"), "HTML")
            body = strip_label(chunk_text)
            self.assertLessEqual(utf16_len(chunk_text), 4096)
            for tag in ("b", "code"):
                self.assertEqual(body.count(f"<{tag}>"),
                                 body.count(f"</{tag}>"),
                                 f"unbalanced <{tag}> in chunk")
            self.assertNotIn("**", body)

    def test_durable_path_carries_parse_mode_per_chunk(self):
        from nomorals.storage.db import Database

        tmp = tempfile.mkdtemp(prefix="f1-durable-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = Database(os.path.join(tmp, "t.db"))
        db.migrate()
        gw, ad = _gateway("telegram-bot", db=db, durable_outbox=True)
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = gw.send("telegram-bot", chat, "**Durable** alert")
        self.assertTrue(res.ok)
        self.assertEqual(len(ad.calls), 1)
        text, kw = ad.calls[0]
        self.assertEqual(text, "<b>Durable</b> alert")
        self.assertEqual(kw.get("parse_mode"), "HTML")


# ── bot adapter: parse mode + entity fallback ────────────────────────

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeBotSession:
    """Records Bot API calls; rejects formatted sends on demand."""

    def __init__(self, reject: str = "") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.reject = reject  # "entities" | "chat" | ""

    def post_json(self, url, params, timeout=None):
        self.calls.append((url, dict(params)))
        if self.reject == "entities" and params.get("parse_mode"):
            return _FakeResp({
                "ok": False, "error_code": 400,
                "description": "Bad Request: can't parse entities: "
                               "Unsupported start tag \"x\" at byte "
                               "offset 3"})
        if self.reject == "chat":
            return _FakeResp({"ok": False, "error_code": 400,
                              "description": "Bad Request: chat not found"})
        return _FakeResp({"ok": True, "result": {"message_id": 42}})

    def get(self, url, timeout=None):  # pragma: no cover - unused
        return _FakeResp({"ok": True, "result": True})


def _bot_adapter(session) -> TelegramBotAdapter:
    return TelegramBotAdapter(token="test-token-123", session=session)


class BotAdapterFormatTests(unittest.TestCase):
    def test_parse_mode_forwarded_to_api(self):
        sess = _FakeBotSession()
        ad = _bot_adapter(sess)
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = ad.send(chat, "<b>hi</b> there", buttons=[],
                      parse_mode="HTML")
        self.assertTrue(res.ok)
        sends = [p for u, p in sess.calls if u.endswith("sendMessage")]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].get("parse_mode"), "HTML")
        self.assertEqual(sends[0].get("text"), "<b>hi</b> there")

    def test_entity_rejection_retries_plain_and_delivers(self):
        sess = _FakeBotSession(reject="entities")
        ad = _bot_adapter(sess)
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = ad.send(chat, "<b>hi</b>", buttons=[], parse_mode="HTML")
        self.assertTrue(res.ok)  # the message was NOT eaten
        sends = [p for u, p in sess.calls if u.endswith("sendMessage")]
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[0].get("parse_mode"), "HTML")
        self.assertNotIn("parse_mode", sends[1])
        self.assertEqual(sends[1].get("text"), "<b>hi</b>")

    def test_other_failures_not_masked_by_fallback(self):
        sess = _FakeBotSession(reject="chat")
        ad = _bot_adapter(sess)
        chat = ChatRef(platform="telegram-bot", chat_id="111")
        res = ad.send(chat, "<b>hi</b>", buttons=[], parse_mode="HTML")
        self.assertFalse(res.ok)
        sends = [p for u, p in sess.calls if u.endswith("sendMessage")]
        self.assertEqual(len(sends), 1)  # no plain retry for real errors


# ── personal adapter: explicit parse mode + fallback ─────────────────

class _FakeTelethonClient:
    def __init__(self, reject_parse: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.reject_parse = reject_parse

    async def send_message(self, entity, text, **kwargs):
        self.calls.append((text, dict(kwargs)))
        if self.reject_parse and "parse_mode" in kwargs:
            raise ValueError("can't parse entities: bad tag at offset 2")
        return SimpleNamespace(id=99)


class PersonalAdapterFormatTests(unittest.TestCase):
    def _adapter(self, tmp: str) -> TelegramAdapter:
        ad = TelegramAdapter(api_id=123, api_hash="deadbeef",
                             session_path=os.path.join(tmp, "s.session"),
                             media_dir=os.path.join(tmp, "media"))
        ad._input_entity_cache.set("111", "ENTITY")
        return ad

    def test_explicit_parse_mode_on_every_chunk(self):
        tmp = tempfile.mkdtemp(prefix="f1-personal-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ad = self._adapter(tmp)
        client = _FakeTelethonClient()
        chat = ChatRef(platform="telegram", chat_id="111")
        text = "<b>head</b> " + ("body " * 1500)  # >4096 → 2 chunks
        result = asyncio.run(
            ad._send_async(client, chat, text, "", "HTML"))
        self.assertEqual(getattr(result, "id", None), 99)
        self.assertGreaterEqual(len(client.calls), 2)
        for _text, kw in client.calls:
            self.assertEqual(kw.get("parse_mode"), "HTML")

    def test_parse_rejection_retries_plain(self):
        tmp = tempfile.mkdtemp(prefix="f1-personal-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ad = self._adapter(tmp)
        client = _FakeTelethonClient(reject_parse=True)
        chat = ChatRef(platform="telegram", chat_id="111")
        result = asyncio.run(
            ad._send_async(client, chat, "<b>hi</b>", "", "HTML"))
        self.assertEqual(getattr(result, "id", None), 99)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0][1].get("parse_mode"), "HTML")
        self.assertNotIn("parse_mode", client.calls[1][1])
        self.assertEqual(client.calls[1][0], "hi")  # tags stripped


# ── Notifier end-to-end + Pass 7 tiers through the real gateway ──────

def _make_ctx(test: unittest.TestCase, gateway) -> SimpleNamespace:
    from nomorals.storage.db import Database

    tmp = tempfile.mkdtemp(prefix="f1-notifier-")
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
    return SimpleNamespace(db=db, settings=settings,
                           extras={"gateway": gateway}, gateway=gateway)


class NotifierEndToEndTests(unittest.TestCase):
    def test_banner_arrives_as_html_bold_on_bot_adapter(self):
        sess = _FakeBotSession()
        bot = _bot_adapter(sess)
        gw = ChatGateway({"telegram-bot": bot})
        ctx = _make_ctx(self, gw)
        from nomorals.agents.notifier import Notifier

        res = Notifier(ctx).publish(
            "news", "Finance alerts", "checked **0** alerts & <none> fired")
        self.assertTrue(res.get("delivered"))
        sends = [p for u, p in sess.calls if u.endswith("sendMessage")]
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].get("parse_mode"), "HTML")
        self.assertEqual(
            sends[0].get("text"),
            "🔔 <b>Finance alerts</b>\n"
            "checked <b>0</b> alerts &amp; &lt;none&gt; fired")

    def test_log_and_digest_tiers_never_delivered(self):
        gw, ad = _gateway("telegram-bot")
        ctx = _make_ctx(self, gw)
        from nomorals.agents.notifier import Notifier

        n = Notifier(ctx)
        subject = "host process population (adaptive cancer gate)"
        res = n.publish(
            "immune", "🛡️ immune gate ENFORCED", "gate enforced: x",
            alert_subject=subject,
            alert_hints={"subject": subject, "self_resolving": True,
                         "incident_key": "adaptive-gate"})
        self.assertEqual(res.get("tier"), "log")
        res2 = n.publish(
            "immune", "🛡️ immune baseline ready", "30 samples learned",
            alert_hints={"subject": "immune baseline", "notice": True})
        self.assertEqual(res2.get("tier"), "digest")
        self.assertEqual(ad.calls, [])  # nothing reached the adapter
        self.assertEqual(n.pending(), [])  # and nothing is redelivered
        self.assertEqual(len(n.digest_pending()), 1)

    def test_page_tier_delivers_formatted_and_names_subject(self):
        gw, ad = _gateway("telegram-bot")
        ctx = _make_ctx(self, gw)
        from nomorals.agents.notifier import Notifier

        res = Notifier(ctx).publish(
            "immune", "❌ immune sampler error", "tick failed: boom",
            alert_hints={"subject": "immune sampler", "failed": True})
        self.assertTrue(res.get("delivered"))
        self.assertEqual(len(ad.calls), 1)
        text, kw = ad.calls[0]
        self.assertEqual(kw.get("parse_mode"), "HTML")
        self.assertIn("<b>❌ immune sampler error</b>", text)
        self.assertIn("immune sampler", text)  # subject named (Pass 7)


if __name__ == "__main__":
    unittest.main()
