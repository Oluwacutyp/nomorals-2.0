"""Animated emoji (owner, 2026-11-11, live: "Telegram emojis
still whack… Telegram has a way where emojis are animated…
Still using still emoji's").

The pins for the fix, in four parts:

A. the TGS generator (social/chat/emoji_anim.py) — her studio
   designs re-expressed as gzipped Lottie: 100×100, 60 fps,
   ≤3 s, REAL keyframed motion of the design itself, the
   64 KB cap enforced by simplification, deterministic bytes.
B. the adapter + registry — pack registration carries
   format "animated" through BOTH the create and the add
   payloads, and her registry records the format that
   actually landed (a static-era entry claims nothing).
C. the lone-emoji shape — one found emoji on Telegram goes
   out as its own single-emoji message (the only shape
   clients animate); a multi-hit answer is never split; a
   sentence is never split; WhatsApp keeps the combined text.
   Plus is_big on the celebration-reaction path, degrading
   to a normal reaction when refused.
D. the entity-refusal fallback — when Telegram refuses the
   custom-emoji entity send, the creation still lands (as a
   sticker, journaled), never a crash, never a fake claim.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat import emoji_anim, emoji_studio
from nomorals.social.chat.expression import (
    CustomEmojiRegistry,
    lone_emoji_char,
)

_DESIGNS = [
    "gold heart with sparkles",
    "a purple laughing cat",
    "a gold rocket",
    "letter d",
    "blue moon",
    "orange flame",
    "a sleepy face",
    "silver star",
    "a pink flower badge",
]


def _subject(doc):
    return next(layer for layer in doc["layers"]
                if layer["nm"] == "subject")


def _animated_props(doc):
    """The subject layer's keyframed properties whose values
    actually CHANGE across the loop — motion, not metadata."""
    moved = []
    for prop, value in _subject(doc)["ks"].items():
        if isinstance(value, dict) and value.get("a") == 1:
            frames = value["k"]
            values = {tuple(f["s"]) for f in frames}
            if len(frames) >= 3 and len(values) > 1:
                moved.append(prop)
    return moved


# ══ A · the TGS generator ════════════════════════════════════


@pytest.mark.parametrize("description", _DESIGNS)
def test_tgs_is_valid_telegram_animated_emoji_shape(description):
    spec = emoji_studio.parse_design(description)
    blob = emoji_anim.tgs_bytes(spec)
    assert len(blob) <= emoji_anim.MAX_TGS_BYTES
    doc = json.loads(gzip.decompress(blob))  # gunzip → Lottie JSON
    assert doc["w"] == 100 and doc["h"] == 100  # the canvas
    assert doc["fr"] == 60
    assert doc["op"] / doc["fr"] <= 3  # Telegram's duration cap
    assert doc["ip"] == 0 and doc["op"] == emoji_anim.DURATION_FRAMES
    # the loop closes: every animated property ends where it
    # began (rotation's 360 is its 0), so replay never jumps
    for value in _subject(doc)["ks"].values():
        if isinstance(value, dict) and value.get("a") == 1:
            first, last = value["k"][0]["s"], value["k"][-1]["s"]
            assert all(abs(a - b) % 360 == 0
                       for a, b in zip(first, last, strict=True))


@pytest.mark.parametrize("description", _DESIGNS)
def test_tgs_really_animates_the_design(description):
    spec = emoji_studio.parse_design(description)
    doc = json.loads(gzip.decompress(emoji_anim.tgs_bytes(spec)))
    assert _animated_props(doc), description


def test_tgs_is_deterministic_and_design_specific():
    heart = emoji_studio.parse_design("gold heart")
    rocket = emoji_studio.parse_design("a gold rocket")
    assert emoji_anim.tgs_bytes(heart) == emoji_anim.tgs_bytes(heart)
    assert emoji_anim.tgs_bytes(heart) != emoji_anim.tgs_bytes(rocket)


def test_design_motion_vocabulary():
    assert emoji_anim.design_motion(
        emoji_studio.parse_design("gold heart")) == "pulse"
    assert emoji_anim.design_motion(
        emoji_studio.parse_design("a gold rocket")) == "bounce"
    assert emoji_anim.design_motion(
        emoji_studio.parse_design("silver star")) == "spin"
    assert emoji_anim.design_motion(
        emoji_studio.parse_design("orange flame")) == "wiggle"
    # an unlisted shape still moves — one of the four, by seed
    assert emoji_anim.design_motion(
        {"shape": "blob", "palette": "teal"}) in emoji_anim.MOTIONS


def test_letter_design_is_drawn_as_shapes_not_a_text_layer():
    spec = emoji_studio.parse_design("letter d")
    doc = json.loads(gzip.decompress(emoji_anim.tgs_bytes(spec)))
    kinds = set()

    def _walk(node):
        if isinstance(node, dict):
            if "ty" in node:
                kinds.add(node["ty"])
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)

    _walk(doc)
    assert "rc" in kinds  # the glyph is rect shapes…
    assert "tm" not in kinds and "text" not in kinds  # …never a
    # font-dependent text layer (Lottie text needs font assets)


def test_sparkles_present_full_detail_shed_when_simplifying():
    spec = emoji_studio.parse_design("gold heart with sparkles")
    full = emoji_anim.lottie_for_design(spec, detail=0)
    lean = emoji_anim.lottie_for_design(spec, detail=1)
    assert any(layer["nm"].startswith("sparkle")
               for layer in full["layers"])
    assert not any(layer["nm"].startswith("sparkle")
                   for layer in lean["layers"])


def test_size_cap_is_enforced_honestly():
    spec = emoji_studio.parse_design("gold heart with sparkles")
    with pytest.raises(emoji_anim.EmojiAnimTooLarge):
        emoji_anim.tgs_bytes(spec, max_bytes=64)


def test_animated_emoji_tgs_writes_a_real_file(tmp_path):
    spec = emoji_studio.parse_design("a gold rocket")
    path = emoji_anim.animated_emoji_tgs(spec, tmp_path, name="rocket")
    assert path.suffix == ".tgs" and path.exists()
    assert path.stat().st_size <= emoji_anim.MAX_TGS_BYTES
    doc = json.loads(gzip.decompress(path.read_bytes()))
    assert _animated_props(doc)


# ══ B · the adapter payloads + the registry's truth ══════════


def _bot_adapter():
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter(
        token="123:fake", session=SimpleNamespace(),
        owner_sender_ids={"7"})
    adapter._bot_username = "Devon_test_bot"
    return adapter


def _pack_rig(adapter, *, pack_exists=False):
    """Fake the Bot API under a real adapter; returns the
    (api calls, uploads) record."""
    calls, uploads = [], []
    state = {"exists": pack_exists}

    def fake_api(method, **params):
        calls.append((method, params))
        if method == "getStickerSet":
            if not state["exists"]:
                raise RuntimeError("Bad Request: STICKERSET_INVALID")
            return {"stickers": [
                {"emoji": "🚀", "custom_emoji_id": "CEID1"}]}
        raise AssertionError(method)

    def fake_upload(method, fields, file_field, path, **kw):
        uploads.append((method, fields, Path(path)))
        state["exists"] = True
        return True

    adapter._api = fake_api
    adapter._api_upload = fake_upload
    return calls, uploads


def test_pack_create_payload_is_animated_by_default(tmp_path):
    adapter = _bot_adapter()
    _calls, uploads = _pack_rig(adapter)
    tgs = emoji_anim.animated_emoji_tgs(
        emoji_studio.parse_design("a gold rocket"), tmp_path,
        name="rocket")
    made = adapter.create_or_extend_custom_emoji_pack(
        "7", "rocket", "Devon — rocket", str(tgs), "🚀")
    assert made["created"] is True
    assert made["format"] == "animated"
    assert made["custom_emoji_id"] == "CEID1"
    method, fields, path = uploads[0]
    assert method == "createNewStickerSet"
    assert fields["sticker_type"] == "custom_emoji"
    stickers = json.loads(fields["stickers"])
    assert stickers[0]["format"] == "animated"
    assert path.suffix == ".tgs"  # the ANIMATED file rode up


def test_pack_add_payload_is_animated_and_static_still_works(
        tmp_path):
    adapter = _bot_adapter()
    _calls, uploads = _pack_rig(adapter, pack_exists=True)
    tgs = emoji_anim.animated_emoji_tgs(
        emoji_studio.parse_design("gold heart"), tmp_path,
        name="heart")
    made = adapter.create_or_extend_custom_emoji_pack(
        "7", "heart", "Devon — heart", str(tgs), "❤️")
    assert made["created"] is False
    assert made["format"] == "animated"
    method, fields, _path = uploads[0]
    assert method == "addStickerToSet"
    assert json.loads(fields["sticker"])["format"] == "animated"
    # the static rung is still available when a caller means it
    png = tmp_path / "e.png"
    png.write_bytes(b"png")
    made = adapter.create_or_extend_custom_emoji_pack(
        "7", "heart", "Devon — heart", str(png), "❤️",
        format="static")
    assert made["format"] == "static"
    assert json.loads(uploads[-1][1]["sticker"])["format"] == "static"


def test_pack_format_is_validated_before_any_traffic(tmp_path):
    adapter = _bot_adapter()
    _pack_rig(adapter)
    with pytest.raises(ValueError, match="unknown custom-emoji"):
        adapter.create_or_extend_custom_emoji_pack(
            "7", "x", "t", str(tmp_path / "e.tgs"), "✨",
            format="hologram")


def test_registry_records_the_format_that_landed():
    registry = CustomEmojiRegistry()
    registry.add("old", unicode="⭐")
    # a static-era entry claims nothing about format
    assert "format" not in registry.get("old")
    registry.add("new", unicode="🚀", format="animated")
    assert registry.get("new")["format"] == "animated"
    # a later unrelated update neither invents nor erases it
    registry.add("new", unicode="🛸")
    assert registry.get("new")["format"] == "animated"
    registry.add("flat", unicode="❤️", format="static")
    assert registry.get("flat")["format"] == "static"


# ══ C · lone-emoji sends + big reactions ═════════════════════


def test_lone_emoji_char_shape():
    assert lone_emoji_char("🚀") == "🚀"
    assert lone_emoji_char("❤️") == "❤️"
    assert lone_emoji_char("👍🏽") == "👍🏽"  # toned = one grapheme
    assert lone_emoji_char("😂😂") == ""  # two is not a moment
    assert lone_emoji_char("i ❤️ this") == ""  # a sentence…
    assert lone_emoji_char("rocket 🚀") == ""   # …is never split
    assert lone_emoji_char("hello") == ""
    assert lone_emoji_char("") == ""


class _Chat:
    def __init__(self, platform, key):
        from nomorals.social.chat.base import ChatKind

        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = ChatKind.DM


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


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _TgAdapter:
    """A Telegram-shaped adapter fake: pack verbs, a plain
    send, media, custom-emoji sends and reactions."""

    def __init__(self):
        self.texts = []
        self.media_sent = []
        self.custom_sent = []
        self.pack_calls = []
        self.reactions = []
        self.owner_sender_ids = {"7"}
        self._bot_id = 4242

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "administrator"}}

    def send(self, chat, text, **kw):
        self.texts.append(text)
        return SimpleNamespace(ok=True, message_id="3", error="")

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")

    def create_or_extend_custom_emoji_pack(
            self, user_id, slug, title, art_path, fallback,
            format="animated"):
        assert Path(art_path).exists()
        self.pack_calls.append((format, art_path))
        return {"pack": f"devon_{slug}_by_devonbot",
                "custom_emoji_id": "CEID9", "created": True,
                "format": format}

    def send_custom_emoji(self, chat, fallback, custom_emoji_id):
        self.custom_sent.append((fallback, custom_emoji_id))
        return SimpleNamespace(ok=True, message_id="10", error="")

    def react(self, chat, message_id, emoji, big=False):
        self.reactions.append((emoji, big))
        return SimpleNamespace(ok=True, message_id=message_id,
                               error="")


def _runtime(tmp_path, adapters, operators=("7",)):
    from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
        def __init__(self):
            self.gateway = _Gateway(adapters)
            self.operators = set(operators)
            self.context = SimpleNamespace(
                settings=SimpleNamespace(
                    workspace_dir=str(tmp_path)))

        def _send(self, chat, text, **kw):
            return SimpleNamespace(ok=True, message_id="1")

        def _is_operator(self, message):
            return str(message.sender_id or "") in self.operators

    return _Runtime()


def test_single_found_emoji_goes_out_alone_on_telegram(tmp_path):
    adapter = _TgAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    out = str(rt._control_emoji("unicorn", _Msg(chat)))
    # exactly one message, holding exactly the one emoji —
    # the only shape Telegram clients animate
    assert adapter.texts == ["🦄"]
    assert out == "🦄 unicorn face"  # the name line is the reply


def test_multi_hit_answer_is_never_split(tmp_path):
    adapter = _TgAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    out = str(rt._control_emoji("party popper", _Msg(chat)))
    assert adapter.texts == []  # nothing peeled off as a moment
    assert "🎉" in out and "🥳" in out and "\n" in out


def test_single_found_emoji_on_whatsapp_stays_combined(tmp_path):
    adapter = _TgAdapter()
    rt = _runtime(tmp_path, {"whatsapp": adapter})
    chat = _Chat("whatsapp", "whatsapp:5")
    out = str(rt._control_emoji("unicorn", _Msg(chat)))
    assert adapter.texts == []  # WhatsApp has no animated form
    assert out.startswith("🦄")  # the combined answer, as before


def test_creation_registers_animated_end_to_end(tmp_path):
    adapter = _TgAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    out = str(rt._control_emoji("make a gold rocket", _Msg(chat)))
    assert len(adapter.pack_calls) == 1
    fmt, art_path = adapter.pack_calls[0]
    assert fmt == "animated"
    assert art_path.endswith(".tgs")  # the animated file, not the PNG
    registry = rt._expression_library().custom_emoji
    entry = registry.get(registry.names()[0])
    assert entry["format"] == "animated"  # recorded truthfully
    assert entry["refs"]["telegram-bot"] == "CEID9"
    assert adapter.custom_sent == [("🚀", "CEID9")]
    assert "MOVES" in out and "animated" in out


def test_celebration_reaction_lands_big_on_telegram(tmp_path):
    adapter = _TgAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    rt._react_quietly(_Msg(chat), "celebration")
    assert len(adapter.reactions) == 1
    _emoji, big = adapter.reactions[0]
    assert big is True


def test_celebration_reaction_plain_where_no_big_form(tmp_path):
    class _OldAdapter(_TgAdapter):
        def react(self, chat, message_id, emoji):  # no big form
            self.reactions.append((emoji, False))
            return SimpleNamespace(ok=True, message_id=message_id,
                                   error="")

    adapter = _OldAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    rt._react_quietly(_Msg(chat), "celebration")
    assert len(adapter.reactions) == 1  # TypeError → plain retry


def test_adapter_react_big_and_refusal_degrades():
    from nomorals.social.chat.base import ChatRef

    adapter = _bot_adapter()
    calls = []

    def fake_api(method, **params):
        calls.append(params)
        if params.get("is_big"):
            raise RuntimeError("Bad Request: IS_BIG_NOT_ALLOWED")
        return True

    adapter._api = fake_api
    chat = ChatRef(platform="telegram-bot", chat_id="5")
    result = adapter.react(chat, "55", "🎉", big=True)
    assert result.ok  # refused big → journaled → normal landed
    assert calls[0]["is_big"] is True
    assert "is_big" not in calls[1]
    assert json.loads(calls[0]["reaction"])[0]["emoji"] == "🎉"
    calls.clear()
    result = adapter.react(chat, "55", "🎉")
    assert result.ok and "is_big" not in calls[0]  # plain stays plain


# ══ D · the entity-refusal fallback ══════════════════════════


def test_adapter_send_custom_emoji_refusal_is_a_result_not_a_raise():
    from nomorals.social.chat.base import ChatRef

    adapter = _bot_adapter()

    def fake_api(method, **params):
        raise RuntimeError(
            "Bad Request: CUSTOM_EMOJI not allowed for bots")

    adapter._api = fake_api
    chat = ChatRef(platform="telegram-bot", chat_id="5")
    result = adapter.send_custom_emoji(chat, "🚀", "CEID1")
    assert result.ok is False
    assert "CUSTOM_EMOJI" in result.error


def test_creation_survives_an_entity_refusal(tmp_path):
    class _RefusingAdapter(_TgAdapter):
        def send_custom_emoji(self, chat, fallback, custom_id):
            return SimpleNamespace(
                ok=False, message_id="",
                error="Bad Request: CUSTOM_EMOJI not allowed")

    adapter = _RefusingAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    out = str(rt._control_emoji("make a gold rocket", _Msg(chat)))
    # registered for real (animated), the send refused — the
    # art still lands as a sticker and the reply says so
    registry = rt._expression_library().custom_emoji
    entry = registry.get(registry.names()[0])
    assert entry["format"] == "animated"
    assert len(adapter.media_sent) == 1
    assert "custom set" in out
