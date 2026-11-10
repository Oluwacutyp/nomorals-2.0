"""Public-surface Fix 3 — the CARDS programme (2026-11-10).

Pins:
* the eight platform designs (facebook · x · tiktok · reddit ·
  instagram · mastodon · tinder · pof) each render a DISTINCT
  layout — fingerprints, regions and marker pixels, not just
  "an image came out";
* comment sections render multi-speaker, nested where the
  platform nests (reddit/mastodon/facebook indent by depth)
  and flat where it doesn't (x/tiktok);
* avatars: handed-over bytes land on the card; absent ones
  get the design-coloured initials placeholder;
* /quote delivers a rendered PNG card through the real
  command path (not bare text), from real quote sources —
  the dummy provider is NOT in the production chain — and
  text returns only as the NAMED degradation when the
  renderer itself is down;
* motivation renders + delivers its PNG card through the
  real run path, with the same named fallback;
* teachings hygiene: a stocked/read passage never ends
  mid-sentence (the raw [:1200] slice is dead).
"""

from __future__ import annotations

import io
import os
import random
import shutil
import tempfile
import time
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from PIL import Image

import nomorals.agents.motivation as mot
import nomorals.agents.teachings as teach
import nomorals.llm.brain as brain_mod
from nomorals.agents.teachings import TeachingsLedger, clean_passage
from nomorals.pulses.timezone import PulseAnchor  # noqa: F401  (parity import)
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.fun import (
    LOCAL_QUOTES,
    DummyQuotesProvider,
    FunService,
    LocalQuotesProvider,
    QuotableProvider,
    ZenQuotesProvider,
    default_fun_chains,
)
from nomorals.social.chat.makers import (
    CARD_DESIGNS,
    FEATURE_DESIGNS,
    PLATFORM_DESIGNS,
    MakerError,
    MakerService,
    parse_card_request,
    render_card_png,
)
from nomorals.storage.db import Database

ENTRIES = [
    {"name": "Ada Lovelace",
     "text": "We ship at dawn. The build is green, the tests "
             "sing, and nobody is going back to sleep.",
     "likes": "42"},
    {"name": "Bola Ahmed", "text": "This is the way.",
     "likes": "7"},
    {"name": "Ada Lovelace", "text": "Replying right here.",
     "depth": 1},
    {"name": "Chidi O", "text": "Deeper still.", "depth": 2},
]


def _pixel(data: bytes, xy: tuple[int, int]) -> tuple:
    img = Image.open(io.BytesIO(data))
    return img.convert("RGB").getpixel(xy)


# ── the platform designs ──────────────────────────────────────

class PlatformDesignTests(unittest.TestCase):
    def test_registry_carries_the_eight_platforms(self):
        self.assertEqual(
            set(PLATFORM_DESIGNS),
            {"facebook", "x", "tiktok", "reddit", "instagram",
             "mastodon", "tinder", "pof"})
        for key in PLATFORM_DESIGNS:
            self.assertIn(key, CARD_DESIGNS)
        self.assertEqual(set(FEATURE_DESIGNS),
                         {"bigquote", "motivation"})

    def test_each_design_renders_a_distinct_layout(self):
        fingerprints = {}
        for key in PLATFORM_DESIGNS:
            data, drawn = render_card_png(key, ENTRIES)
            self.assertTrue(data.startswith(b"\x89PNG"), key)
            self.assertEqual(drawn["design"], key)
            self.assertEqual(len(drawn["names"]), 4, key)
            fingerprints[key] = tuple(sorted(
                (k, str(v)) for k, v in
                drawn["fingerprint"].items()))
        # all eight fingerprints pairwise distinct
        self.assertEqual(len(set(fingerprints.values())), 8)

    def test_marker_regions_differ_per_design(self):
        _d, fb = render_card_png("facebook", ENTRIES)
        _d, xx = render_card_png("x", ENTRIES)
        _d, tt = render_card_png("tiktok", ENTRIES)
        _d, rd = render_card_png("reddit", ENTRIES)
        _d, td = render_card_png("tinder", ENTRIES)
        # header geometry: bar (64) vs minimal (52) vs none (0)
        # vs band (120)
        self.assertEqual(fb["regions"]["header"][3], 64)
        self.assertEqual(xx["regions"]["header"][3], 52)
        self.assertNotIn("header", tt["regions"])
        self.assertEqual(td["regions"]["header"][3], 120)
        # tiktok carries a right action rail; reddit a vote rail
        self.assertIn("rail", tt["regions"])
        self.assertIn("vote_rail", rd["regions"])
        self.assertNotIn("rail", fb["regions"])
        # marker pixels: tinder's gradient band opens pink,
        # mastodon's chrome is its dark surface, x is black
        data_t, _ = render_card_png("tinder", ENTRIES)
        self.assertEqual(_pixel(data_t, (5, 0)), (0xFD, 0x26, 0x6D))
        data_m, _ = render_card_png("mastodon", ENTRIES)
        self.assertEqual(_pixel(data_m, (5, 5)), (0x28, 0x2C, 0x37))
        data_x, _ = render_card_png("x", ENTRIES)
        self.assertEqual(_pixel(data_x, (5, 5)), (0, 0, 0))
        data_r, _ = render_card_png("reddit", ENTRIES)
        self.assertEqual(_pixel(data_r, (5, 100)),
                         (0xDA, 0xE0, 0xE6))

    def test_comment_sections_render_all_speakers(self):
        for key in PLATFORM_DESIGNS:
            _data, drawn = render_card_png(key, ENTRIES)
            self.assertEqual(
                len(drawn["regions"]["comments"]), 3, key)
            self.assertEqual(
                len(drawn["regions"]["avatars"]), 4, key)

    def test_nesting_indents_where_the_platform_nests(self):
        for key in ("reddit", "mastodon", "facebook"):
            _data, drawn = render_card_png(key, ENTRIES)
            xs = [box[0] for box in drawn["regions"]["comments"]]
            self.assertLess(xs[0], xs[1], key)
            self.assertLess(xs[1], xs[2], key)
        for key in ("x", "tiktok", "instagram", "pof"):
            _data, drawn = render_card_png(key, ENTRIES)
            xs = [box[0] for box in drawn["regions"]["comments"]]
            self.assertEqual(len(set(xs)), 1, key)

    def test_avatar_bytes_land_and_placeholder_stands_in(self):
        red = Image.new("RGB", (96, 96), (0xE0, 0x20, 0x20))
        buf = io.BytesIO()
        red.save(buf, format="PNG")
        entries = [dict(ENTRIES[0], avatar=buf.getvalue()),
                   *ENTRIES[1:]]
        data, drawn = render_card_png("facebook", entries)
        ax, ay, aw, _ah = drawn["regions"]["avatars"][0]
        self.assertEqual(_pixel(data, (ax + aw // 2, ay + aw // 2)),
                         (0xE0, 0x20, 0x20))
        # no avatar handed over → the design's placeholder disc
        data2, drawn2 = render_card_png("facebook", ENTRIES)
        ax, ay, aw, _ah = drawn2["regions"]["avatars"][0]
        bg = PLATFORM_DESIGNS["facebook"]["avatar_bg"]
        self.assertEqual(_pixel(data2, (ax + 20, ay + 12)), bg)
        # instagram's ring rides outside the avatar box
        data3, drawn3 = render_card_png("instagram", ENTRIES)
        ax, ay, aw, _ah = drawn3["regions"]["avatars"][0]
        self.assertEqual(_pixel(data3, (ax - 3, ay + aw // 2)),
                         PLATFORM_DESIGNS["instagram"]["avatar_ring"])

    def test_feature_designs_render_square_backdrop_cards(self):
        data, drawn = render_card_png(
            "bigquote", [{"name": "Seneca",
                          "text": "Luck is preparation meeting "
                                  "opportunity."}])
        self.assertEqual(drawn["size"], (1080, 1080))
        self.assertEqual(drawn["backdrop"], "gradient")
        art = Image.new("RGB", (800, 600), (0x30, 0x60, 0x90))
        data2, drawn2 = render_card_png(
            "motivation",
            [{"name": "Meditations — WisdomKeeper",
              "text": "Begin again today.", "title": "Morning",
              "meta": "evening"}],
            backdrop=art)
        self.assertEqual(drawn2["backdrop"], "image")
        self.assertEqual(drawn2["size"], (1080, 1080))
        self.assertTrue(data2.startswith(b"\x89PNG"))


class ParseGrammarTests(unittest.TestCase):
    def test_design_first_word_and_comment_lines(self):
        style, entries = parse_card_request(
            "qcard",
            "facebook Ada | hello world\n"
            "> Bola | nice one\n"
            ">> Ada | thanks — nested")
        self.assertEqual(style, "facebook")
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[1]["name"], "Bola")
        self.assertEqual(entries[1]["depth"], 0)
        self.assertEqual(entries[2]["depth"], 1)

    def test_comments_refused_on_legacy_styles(self):
        with self.assertRaises(MakerError):
            parse_card_request(
                "qcard", "quote Ada | hi\n> Bola | a comment")

    def test_legacy_grammar_unchanged(self):
        style, entries = parse_card_request("qcard", "Ada | hi")
        self.assertEqual((style, entries[0]["name"]),
                         ("quote", "Ada"))
        style, entries = parse_card_request("fakettc", "Ada | hi")
        self.assertEqual(style, "ttc")


# ── the quote sources + card ──────────────────────────────────

class _Send:
    def __init__(self):
        self.sent = []

    def __call__(self, ref, caption):
        self.sent.append((ref, caption))
        return SimpleNamespace(ok=True)


class QuoteSourceTests(unittest.TestCase):
    def test_production_chain_has_no_dummy(self):
        chain = default_fun_chains()["quote"]
        names = [p.name for p in chain]
        self.assertEqual(names, ["quotable", "zenquotes",
                                 "her quote store"])
        self.assertFalse(any(isinstance(p, DummyQuotesProvider)
                             for p in chain))

    def test_providers_map_their_real_shapes(self):
        quotable = QuotableProvider(fetcher=lambda url: {
            "content": "Stay hungry.", "author": "Steve Jobs"})
        out = quotable.quote()
        self.assertEqual(out["quote_text"], "Stay hungry.")
        self.assertEqual(out["author"], "Steve Jobs")
        zen = ZenQuotesProvider(fetcher=lambda url: [
            {"q": "Be here now.", "a": "Ram Dass"}])
        out = zen.quote()
        self.assertEqual(out["quote_text"], "Be here now.")
        local = LocalQuotesProvider(rng=random.Random(1))
        out = local.quote()
        self.assertIn((out["quote_text"], out["author"]),
                      set(LOCAL_QUOTES))

    def test_quote_card_sends_a_png_photo(self):
        send = _Send()
        service = FunService(
            media_dir=self._tmp(),
            chains={"quote": [LocalQuotesProvider(
                rng=random.Random(3))]})
        out = service.quote_card(
            send=send, backdrop_fn=lambda seed="": None)
        self.assertTrue(out["sent"], out)
        ref, caption = send.sent[0]
        self.assertEqual(ref.kind, "image")
        with open(ref.path, "rb") as fh:
            head = fh.read(8)
        self.assertEqual(head, b"\x89PNG\r\n\x1a\n")
        self.assertIn("—", caption)  # the author rides the caption

    def test_quote_card_renderer_down_is_a_named_degradation(self):
        def _boom(*a, **kw):
            raise RuntimeError("Pillow is gone")

        service = FunService(
            media_dir=self._tmp(),
            chains={"quote": [LocalQuotesProvider(
                rng=random.Random(3))]})
        out = service.quote_card(
            send=_Send(), renderer=_boom,
            backdrop_fn=lambda seed="": None)
        self.assertFalse(out["sent"])
        self.assertIn("card_error", out)
        self.assertTrue(out["text"])  # the words survive, named

    def _tmp(self):
        tmp = tempfile.mkdtemp(prefix="cards-q-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp


# ── the real command paths ────────────────────────────────────

def _runtime_harnesses():
    from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin
    from nomorals.agents.partner.runtime_fun import RuntimeFunMixin
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
    from nomorals.agents.partner.runtime_makers import RuntimeMakersMixin
    return (RuntimeFunMixin, RuntimeMakersMixin,
            RuntimeExpressionMixin, RuntimeGroupMixin)


class _Chat:
    def __init__(self, platform, key, kind=ChatKind.GROUP):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = kind


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.media = kw.get("media") or []


class _Adapter:
    def __init__(self):
        self.media_sent = []
        self._bot_id = 4242
        self.bot_status = "administrator"

    def get_me(self):
        return {"id": self._bot_id, "username": "Nomorals_bot"}

    def admin_member(self, chat, user_id):
        status = self.bot_status if str(user_id) == str(
            self._bot_id) else "member"
        return {"ok": True, "result": {"status": status}}

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append((ref, caption))
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class CommandPathTests(unittest.TestCase):
    def _runtime(self, tmp):
        fun_m, makers_m, expr_m, group_m = _runtime_harnesses()

        class _Runtime(fun_m, makers_m, expr_m, group_m):
            def __init__(self):
                self.sent = []
                self.gateway = _Gateway(
                    {"telegram-bot": _Adapter()})
                self.operators = {"99"}
                self.context = SimpleNamespace(
                    settings=SimpleNamespace(
                        workspace_dir=str(tmp)))

            def _send(self, chat, text, **kw):
                self.sent.append(text)
                return SimpleNamespace(ok=True, message_id="1")

            def _is_operator(self, message):
                return str(message.sender_id or "") in \
                    self.operators

        return _Runtime()

    def test_quote_command_delivers_the_card(self):
        tmp = tempfile.mkdtemp(prefix="cards-cmd-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rt = self._runtime(tmp)
        service = FunService(
            media_dir=os.path.join(tmp, "fun_media"),
            chains={"quote": [LocalQuotesProvider(
                rng=random.Random(5))]})
        rt._fun_service_obj = service
        chat = _Chat("telegram-bot", "telegram-bot:-100123")
        out = rt._control_fun("quote", "", _Msg(chat))
        self.assertEqual(out, "")  # the card WAS the reply
        adapter = rt.gateway.adapters["telegram-bot"]
        self.assertEqual(len(adapter.media_sent), 1)
        ref, _caption = adapter.media_sent[0]
        self.assertEqual(Image.open(ref.path).format, "PNG")

    def test_makers_command_renders_a_platform_card(self):
        tmp = tempfile.mkdtemp(prefix="cards-cmd-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        rt = self._runtime(tmp)
        rt._makers_service_obj = MakerService(
            media_dir=os.path.join(tmp, "maker_media"))
        chat = _Chat("telegram-bot", "telegram-bot:-100123")
        out = rt._control_makers(
            "qcard",
            "reddit Ada | we shipped\n> Bola | nice\n>> Ada | ty",
            _Msg(chat))
        self.assertEqual(out, "")
        adapter = rt.gateway.adapters["telegram-bot"]
        self.assertEqual(len(adapter.media_sent), 1)
        ref, caption = adapter.media_sent[0]
        self.assertEqual(Image.open(ref.path).format, "PNG")
        self.assertIn("Reddit card", caption)


# ── motivation cards ──────────────────────────────────────────

LAGOS = ZoneInfo("Africa/Lagos")


def _ts(year, month, day, hour, minute=0) -> float:
    return datetime(year, month, day, hour, minute,
                    tzinfo=LAGOS).timestamp()


class _FakeNotifier:
    def __init__(self):
        self.calls = []

    def publish(self, kind, title, body="", **kw):
        self.calls.append({"kind": kind, "title": title,
                           "body": body, **kw})
        return {"delivered": True, "persisted_only": False}


class _CardGateway:
    """The gateway shape _deliver_card drives: status +
    send_file, owner on telegram-bot."""

    dry_run = False

    def __init__(self):
        self.files = []

    def status(self):
        return {"telegram-bot": {"running_in_session": True}}

    def platforms_for_chat_id(self, chat_id):
        return ["telegram-bot"]

    def send_file(self, platform, chat, path, **kw):
        self.files.append((platform, chat, path, kw))
        return SimpleNamespace(ok=True, message_id="7")


def _mot_ctx(test, gateway=None):
    tmp = tempfile.mkdtemp(prefix="cards-mot-")
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
    ctx = SimpleNamespace(db=db, settings=settings,
                          extras={})
    if gateway is not None:
        ctx.extras["gateway"] = gateway
    ctx.notifier = _FakeNotifier()
    return ctx


def _mot_item(passage=("The impediment to action advances "
                       "action; what stands in the way becomes "
                       "the way.")):
    return {
        "id": "teach_meditations_1", "kind": "teaching",
        "title": "Meditations", "body": "", "theme": "discipline",
        "tradition": "", "weight": 1.0,
        "source": {"type": "wisdom", "ref": "meditations",
                   "label": "Meditations — WisdomKeeper",
                   "passage": passage},
        "created_at": time.time(), "served": [],
    }


def _stub_mot(test, gateway=None):
    ctx = _mot_ctx(test, gateway=gateway)
    item = _mot_item()
    test.addCleanup(setattr, teach, "library_candidates",
                    teach.library_candidates)
    test.addCleanup(setattr, teach, "net_candidates",
                    teach.net_candidates)
    test.addCleanup(setattr, teach, "memory_candidates",
                    teach.memory_candidates)
    teach.library_candidates = lambda c, themes: [item]
    teach.net_candidates = lambda c, themes: []
    teach.memory_candidates = lambda c: []
    test.addCleanup(setattr, brain_mod, "brain_for",
                    brain_mod.brain_for)
    brain_mod.brain_for = lambda c: None  # degraded: passage card
    return ctx


class MotivationCardTests(unittest.TestCase):
    def test_run_delivers_the_png_card_as_a_photo(self):
        gateway = _CardGateway()
        ctx = _stub_mot(self, gateway=gateway)
        now = _ts(2026, 10, 23, 7, 5)
        res = mot.run_motivation(ctx, "morning", now=now,
                                 rng=random.Random(3))
        self.assertTrue(res["delivered"], res)
        self.assertTrue(res["card_rendered"], res)
        self.assertTrue(res["card_sent"], res)
        self.assertEqual(len(gateway.files), 1)
        _plat, _chat, path, kw = gateway.files[0]
        self.assertEqual(kw.get("kind"), "image")
        with open(path, "rb") as fh:
            self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
        # the notifier record still carries the words
        self.assertEqual(len(ctx.notifier.calls), 1)
        self.assertIn("impediment", ctx.notifier.calls[0]["body"])

    def test_renderer_down_names_the_degradation(self):
        ctx = _stub_mot(self)
        original = mot._render_piece_card

        def _boom(*a, **kw):
            raise RuntimeError("no Pillow on this box")

        mot._render_piece_card = _boom
        self.addCleanup(setattr, mot, "_render_piece_card",
                        original)
        now = _ts(2026, 10, 23, 7, 5)
        res = mot.run_motivation(ctx, "morning", now=now,
                                 rng=random.Random(3))
        self.assertTrue(res["delivered"], res)
        self.assertFalse(res["card_rendered"])
        self.assertIn("card_error", res)
        body = ctx.notifier.calls[0]["body"]
        self.assertIn("couldn't render", body)
        self.assertIn("impediment", body)  # words still land


# ── teachings hygiene ─────────────────────────────────────────

class TeachingsHygieneTests(unittest.TestCase):
    LONG = ("Discipline is the bridge between goals and "
            "accomplishment. " * 24) + \
           "And then a final fragment trails off mid thought"

    def test_clean_passage_never_ends_mid_sentence(self):
        cleaned = clean_passage(self.LONG)
        self.assertLessEqual(len(cleaned), 1200)
        self.assertTrue(cleaned.endswith("."), cleaned[-40:])
        self.assertNotIn("fragment", cleaned)
        self.assertEqual(clean_passage(cleaned), cleaned)

    def test_short_passage_passes_whole(self):
        text = "Begin again today. The way is in the work."
        self.assertEqual(clean_passage(text), text)

    def test_read_side_cleans_a_prestoked_raw_slice(self):
        ctx = SimpleNamespace(db=None, settings=SimpleNamespace())
        ledger = TeachingsLedger(ctx)
        raw_slice = self.LONG[:1200]  # the old producers' shape
        self.assertFalse(raw_slice.endswith("."))
        item = {
            "id": "teach_raw_1", "kind": "teaching",
            "title": "Old stock", "body": "", "theme": "",
            "tradition": "", "weight": 1.0,
            "source": {"type": "wisdom", "ref": "old",
                       "label": "Old stock",
                       "passage": raw_slice},
            "created_at": time.time(), "served": [],
        }
        ledger._store_candidates([item], time.time())
        pool = ledger.candidates(now=time.time())
        self.assertEqual(len(pool), 1)
        passage = pool[0]["source"]["passage"]
        self.assertTrue(passage.endswith("."), passage[-40:])
        picked = ledger.pick(surface="motivation-morning",
                             now=time.time(),
                             rng=random.Random(1))
        self.assertTrue(
            picked["source"]["passage"].endswith("."))


if __name__ == "__main__":
    unittest.main()
