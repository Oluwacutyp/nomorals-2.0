"""Expression repair F3: /proposals inspection + the stamps.

Part 1 — the owner could not inspect a proposal: /proposals
listed one-line rows and the only verbs were approve/deny by id.
These tests pin the inspection view (`/proposals show <id>`),
its honesty edges (unknown / decided / ambiguous ids), and the
Telegram list buttons whose taps run the REAL commands through
the shared engine (approve actually sends, deny actually
closes, inspect actually opens the detail).

Part 2 — the stamps. The image watermark is a RENDERED mark
(typeset from text, sized by rule, contrast-adaptive, burned
into the delivered pixels — owner order 2026-11-11), and the
presented families that bypassed every stamp choke point (the
Notifier's pushes, the experience packs' home cards) now carry
the signature by construction. A plain chat reply does NOT.
"""
from __future__ import annotations

import io
import tempfile
import unittest
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.agents.notifier import Notifier
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.agents.upgrade_queue import UpgradeQueue
from nomorals.core.config import load_settings
from nomorals.storage.db import Database

STAMP = "Powered by peacethefirst"


# ── fixtures (the test_proposals_unified idiom) ─────────────────

def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None):
    return SimpleNamespace(db=db or _db(), settings=SimpleNamespace(
        partner=SimpleNamespace(proactive_enabled=True, quiet_start=22,
                                quiet_end=8)))


def _runtime(ctx, sent=None):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = ctx
    rt._autonomy = None

    def _send(platform, chat, text, **kw):
        if sent is not None:
            sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, error=None, message_id="m1")

    rt.gateway = SimpleNamespace(send=_send)
    return rt


def _proactive_row(db, pid="prop-1", kind="dm",
                   content="hello?", reason=""):
    db.execute(
        "INSERT INTO proactive_log (id, kind, platform, chat_id, content,"
        " status, reason, decided_at, acted_at)"
        " VALUES (?, ?, 'telegram', '123', ?, 'pending', ?, 0, 0)",
        (pid, kind, content, reason))


def _upgrade(ctx, **kw):
    args = dict(
        title="cache compiled regexes in the router hot path",
        rationale="the router recompiles on every message " * 3,
        patch_plan={"steps": ["hoist the compile", "pin a test"]},
        files=["nomorals/agents/router.py"], tests=["tests/test_router.py"],
        claim_ids=["claim-7"], source="research_loop")
    args.update(kw)
    return UpgradeQueue(ctx).propose(**args)


# ── Part 1: inspection ──────────────────────────────────────────

class ProposalDetailTests(unittest.TestCase):
    def test_show_proactive_send_full_body_and_consequences(self):
        ctx = _ctx()
        body = "morning boss — the gold desk moved overnight. " * 4
        _proactive_row(ctx.db, content=body.strip(),
                       reason="scheduled briefing held for approval")
        out = _runtime(ctx).proposal_detail("prop-1")
        self.assertIn("prop-1", out)
        self.assertIn(body.strip(), out)          # FULL body, no [:70] cut
        self.assertIn("telegram:123", out)
        self.assertIn("scheduled briefing held for approval", out)
        self.assertIn("sent to telegram:123 exactly as written", out)
        self.assertIn("nothing is sent", out)

    def test_show_initiative_runs_not_sends(self):
        ctx = _ctx()
        _proactive_row(ctx.db, pid="init-1", kind="initiative",
                       content="restart the research loop with the new seed",
                       reason="autonomous initiative from modes.tick")
        out = _runtime(ctx).proposal_detail("init-1")
        self.assertIn("initiative", out)
        self.assertIn("autonomous initiative from modes.tick", out)
        self.assertIn("RUNS", out)
        self.assertIn("never runs", out)

    def test_show_upgrade_full_ticket(self):
        ctx = _ctx()
        pid = _upgrade(ctx)
        out = _runtime(ctx).proposal_detail(pid)
        self.assertIn(pid, out)
        self.assertIn("research upgrade ticket", out)
        self.assertIn("the router recompiles on every message", out)
        self.assertIn("hoist the compile", out)        # plan steps
        self.assertIn("nomorals/agents/router.py", out)
        self.assertIn("claim-7", out)                  # evidence claims
        self.assertIn("research_loop", out)            # filed by
        self.assertIn("patch plan above is applied", out)
        self.assertIn("a reason is required", out)

    def test_show_unknown_id_is_honest(self):
        ctx = _ctx()
        out = _runtime(ctx).proposal_detail("nope-404")
        self.assertIn("no pending proposal 'nope-404'", out)

    def test_show_decided_id_says_decided(self):
        ctx = _ctx()
        _proactive_row(ctx.db)
        rt = _runtime(ctx)
        self.assertTrue(rt.deny("prop-1").get("ok"))
        out = rt.proposal_detail("prop-1")
        self.assertIn("not pending", out)
        self.assertIn("already denied", out)

    def test_show_ambiguous_prefix_names_candidates(self):
        ctx = _ctx()
        _proactive_row(ctx.db, pid="prop-aa")
        _proactive_row(ctx.db, pid="prop-ab")
        out = _runtime(ctx).proposal_detail("prop-a")
        self.assertIn("more than one pending", out)
        self.assertIn("prop-aa", out)
        self.assertIn("prop-ab", out)

    def test_show_unique_prefix_resolves(self):
        ctx = _ctx()
        _proactive_row(ctx.db, pid="prop-aa")
        _proactive_row(ctx.db, pid="prop-bb")
        out = _runtime(ctx).proposal_detail("prop-b")
        self.assertIn("prop-bb", out)
        self.assertIn("full proposal:", out)

    def test_show_routes_through_subcommand_handler(self):
        ctx = _ctx()
        _proactive_row(ctx.db, content="a held hello for the group")
        rt = _runtime(ctx)
        out = rt._proposals_command("show prop-1")
        self.assertIn("a held hello for the group", out)
        self.assertIsNone(rt._proposals_command("frobnicate prop-1"))
        self.assertEqual(rt._proposals_command("show"),
                         "usage: /proposals show <proposal_id>")


class ProposalButtonTests(unittest.TestCase):
    """The full runtime: the Telegram list carries buttons, and a
    tap (the adapter re-running the button's command through this
    same dispatch) performs the real action."""

    def _full_runtime(self, tmp, sent):
        settings = load_settings(overrides={
            "home": tmp, "partner.platforms": "local",
            "chat.local_enabled": "true"})
        ctx = build_context(settings, with_executor=False,
                            with_tools=False)

        class _Gateway:
            def send(self, platform, chat, text, **kw):
                sent.append((platform, chat, text))
                return SimpleNamespace(ok=True, error=None,
                                       message_id="m1")

        return PartnerRuntime(ctx, gateway=_Gateway()), ctx

    def test_telegram_list_carries_working_buttons(self):
        sent: list = []
        with tempfile.TemporaryDirectory(prefix="nm-expf3-") as tmp:
            rt, ctx = self._full_runtime(tmp, sent)
            _proactive_row(ctx.db, content="held: post the market note")
            out = rt.handle_control("/proposals", "telegram:123")
            self.assertTrue(hasattr(out, "text"))
            self.assertIn("prop-1", out.text)
            self.assertIn("pending proposals (1):", out.text)
            flat = [pair for row in out.buttons for pair in row]
            commands = {data for _label, data in flat}
            self.assertIn("proposals show prop-1", commands)
            self.assertIn("approve prop-1", commands)
            self.assertIn("deny prop-1", commands)
            # INSPECT tap → the detail view opens.
            detail = rt.handle_control("/proposals show prop-1",
                                       "telegram:123")
            self.assertIn("held: post the market note", str(detail))
            # APPROVE tap → the message is really sent + row decided.
            verdict = rt.handle_control("/approve prop-1", "telegram:123")
            self.assertIn("approved prop-1", str(verdict))
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0][2], "held: post the market note")
            row = ctx.db.query_one(
                "SELECT status FROM proactive_log WHERE id = 'prop-1'")
            self.assertEqual(row["status"], "sent")
            # The list is now empty — the decision really landed.
            self.assertIn("no pending proposals",
                          str(rt.handle_control("/proposals",
                                                "telegram:123")))

    def test_telegram_list_deny_button_closes(self):
        sent: list = []
        with tempfile.TemporaryDirectory(prefix="nm-expf3-") as tmp:
            rt, ctx = self._full_runtime(tmp, sent)
            _proactive_row(ctx.db, content="held: a note never to send")
            rt.handle_control("/proposals", "telegram:123")
            verdict = rt.handle_control("/deny prop-1", "telegram:123")
            self.assertIn("denied", str(verdict))
            self.assertEqual(sent, [])
            row = ctx.db.query_one(
                "SELECT status FROM proactive_log WHERE id = 'prop-1'")
            self.assertEqual(row["status"], "denied")

    def test_local_list_stays_plain_text(self):
        sent: list = []
        with tempfile.TemporaryDirectory(prefix="nm-expf3-") as tmp:
            rt, ctx = self._full_runtime(tmp, sent)
            _proactive_row(ctx.db)
            out = rt.handle_control("/proposals", "local:console")
            self.assertIsInstance(out, str)
            self.assertIn("prop-1", out)
            self.assertIn("/proposals show <id>", out)


# ── Part 2a: the rendered image watermark ───────────────────────

def _png_bytes(color, size=(800, 600)):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _corner_extrema(data):
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("L")
    w, h = img.size
    return img.crop((w - 320, h - 70, w, h)).getextrema()


class WatermarkRenderTests(unittest.TestCase):
    def test_dark_ink_on_bright_photo(self):
        from nomorals.social.chat.brandmark import burn_brand_mark

        marked = burn_brand_mark(_png_bytes((245, 245, 245)))
        lo, _hi = _corner_extrema(marked)
        # dark ink present in the corner band (raw corner min is 245)
        self.assertLess(lo, 120)

    def test_light_ink_on_dark_photo(self):
        from nomorals.social.chat.brandmark import burn_brand_mark

        marked = burn_brand_mark(_png_bytes((12, 12, 16)))
        _lo, hi = _corner_extrema(marked)
        # light ink present in the corner band (raw corner max is ~16)
        self.assertGreater(hi, 150)

    def test_center_untouched_and_format_kept(self):
        from PIL import Image

        from nomorals.social.chat.brandmark import burn_brand_mark

        raw = _png_bytes((120, 90, 60))
        marked = burn_brand_mark(raw)
        self.assertEqual(Image.open(io.BytesIO(marked)).format, "PNG")
        img = Image.open(io.BytesIO(marked)).convert("RGB")
        self.assertEqual(img.getpixel((400, 300)), (120, 90, 60))

    def test_short_banner_mark_stays_subordinate(self):
        from PIL import Image

        from nomorals.social.chat.brandmark import MAX_HEIGHT_FRACTION, burn_brand_mark

        raw = _png_bytes((30, 34, 44), size=(920, 182))
        marked = burn_brand_mark(raw)
        img = Image.open(io.BytesIO(marked)).convert("L")
        # rows changed by the mark: within the height cap's band
        base = Image.open(io.BytesIO(raw)).convert("L")
        changed = [y for y in range(182)
                   if any(img.getpixel((x, y)) != base.getpixel((x, y))
                          for x in range(460, 920, 7))]
        self.assertTrue(changed)
        self.assertLessEqual(max(changed) - min(changed),
                             int(182 * MAX_HEIGHT_FRACTION) + 14)

    def test_rejects_non_images_and_tiny(self):
        from nomorals.social.chat.brandmark import BrandMarkError, burn_brand_mark

        with self.assertRaises(BrandMarkError):
            burn_brand_mark(b"this is not an image")
        with self.assertRaises(BrandMarkError):
            burn_brand_mark(_png_bytes((0, 0, 0), size=(40, 40)))


# ── Part 2b: Notifier pushes carry the stamp ────────────────────

def _notifier(sent):
    partner = SimpleNamespace(owner_chats="telegram:123")
    ctx = SimpleNamespace(
        db=_db(),
        settings=SimpleNamespace(partner=partner),
        extras={})
    gw = SimpleNamespace(
        status=lambda: {"telegram": {"running_in_session": True}},
        send=lambda platform, chat, text, **kw: (
            sent.append((platform, text)),
            SimpleNamespace(ok=True, message_id="m1"))[1],
        send_file=lambda *a, **k: SimpleNamespace(ok=False))
    return Notifier(ctx, gateway=gw)


class NotifierStampTests(unittest.TestCase):
    def test_plain_push_closes_with_stamp(self):
        sent: list = []
        out = _notifier(sent)._deliver("briefing", "Morning briefing",
                                       "gold held the range overnight.")
        self.assertTrue(out.delivered)
        self.assertEqual(len(sent), 1)
        self.assertTrue(sent[0][1].rstrip().endswith(STAMP)
                        or f"<i>{STAMP}</i>" in sent[0][1])

    def test_doc_push_closes_with_stamp(self):
        from nomorals.social.chat.present import Doc

        sent: list = []
        doc = Doc(title="Desk report", summary="positions flat, "
                                               "risk inside limits.")
        out = _notifier(sent)._deliver("report", "Desk report", "",
                                       doc=doc)
        self.assertTrue(out.delivered)
        self.assertEqual(len(sent), 1)
        self.assertIn(STAMP, sent[0][1])


# ── Part 2c: pack home cards carry the stamp; chat does not ─────

class PackAndChatStampTests(unittest.TestCase):
    def test_pack_family_resolves_presented_class(self):
        from nomorals.social.chat import packs  # noqa: F401 - registers
        from nomorals.social.chat.living_cards import family_surface_class
        from nomorals.social.chat.typography import STAMP_CLASSES

        self.assertIn(family_surface_class("music-deck"), STAMP_CLASSES)
        self.assertEqual(family_surface_class("generic"), "chat")

    def test_pack_home_card_text_closes_with_stamp(self):
        from nomorals.social.chat import packs  # noqa: F401 - registers
        from nomorals.social.chat.living_cards import Card as LivingCard
        from nomorals.social.chat.living_cards import render_card

        card = LivingCard(card_id="proof-1", family="music-deck",
                          title="Music deck",
                          body="your queue, your decks")
        rendered = render_card(card, "telegram")
        self.assertIn(STAMP, rendered.text)

    def test_plain_chat_card_has_no_stamp(self):
        from nomorals.social.chat.typography import Card, render_card

        rendered = render_card(Card("chat", title="hi"),
                               platform="telegram-bot")
        self.assertNotIn(STAMP, rendered.text)

    def test_register_family_class_rejects_unknown(self):
        from nomorals.social.chat.living_cards import register_family_class

        with self.assertRaises(ValueError):
            register_family_class("some-family", "not-a-class")


if __name__ == "__main__":
    unittest.main()
