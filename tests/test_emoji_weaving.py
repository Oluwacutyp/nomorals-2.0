"""In-text custom-emoji weaving (owner, 2026-11-11, with a
screen recording of the Monarq Delivery bot as the reference:
"Not sent as single emojis bro. I'm talking about real emoji
animation even in texts").

The pins, in five parts:

A. the WEAVER (expression.weave_custom_emoji_entities) —
   registered glyphs wrapped mid-sentence with UTF-16
   offsets, ZWJ/tone clusters as single spans, formatting
   entities preserved around them, the cap keeping the
   FIRST matches, unregistered glyphs untouched.
B. the SEND HOOK (TelegramBotAdapter.send) — plain chunks
   ride woven; markup sends keep their parse path; no
   provider, no weaving.
C. the REFUSAL STATE — an eligibility refusal lands the
   message plain, journals ONCE per boot per chat-type,
   and disables weaving for that scope (never a retry
   storm); a parse-class failure falls back WITHOUT the
   refusal state.
D. PACK IMPORT — /customemoji import registers a whole
   pack's custom-emoji stickers with truthful formats,
   deduped against hers (created/animated preferred);
   /customemoji status reports the three states plainly.
E. WhatsApp is never woven — the wiring only stamps
   adapters that expose the weaving seam.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from nomorals.social.chat import telegram as telegram_mod
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.expression import (
    CustomEmojiRegistry,
    weave_custom_emoji_entities,
)
from nomorals.social.chat.telegram import TelegramBotAdapter


def _map(**glyphs):
    """A glyph map in the registry's own output shape."""
    return {glyph: {"id": cid, "format": "animated", "name": glyph}
            for glyph, cid in glyphs.items()}


# ══ A · the weaver ═══════════════════════════════════════════


def test_weave_offsets_are_utf16_units_not_codepoints():
    # 🚀 is astral: 2 UTF-16 units.  A naive code-point count
    # puts 💰 at offset 5; the Bot API reads UTF-16 — 6.
    entities, woven = weave_custom_emoji_entities(
        "🚀 go 💰!", (), _map(**{"💰": "M1"}))
    assert woven == 1
    assert entities == [{"type": "custom_emoji", "offset": 6,
                         "length": 2, "custom_emoji_id": "M1"}]


def test_weave_zwj_sequence_is_one_span():
    entities, woven = weave_custom_emoji_entities(
        "a 👨‍👩‍👧 b", (), _map(**{"👨‍👩‍👧": "FAM1"}))
    assert woven == 1
    # 👨(2) ZWJ(1) 👩(2) ZWJ(1) 👧(2) = 8 units, after "a " = 2
    assert entities[0]["offset"] == 2
    assert entities[0]["length"] == 8


def test_weave_skin_tone_cluster_is_one_span():
    entities, woven = weave_custom_emoji_entities(
        "nice 👍🏽!", (), _map(**{"👍🏽": "T1"}))
    assert woven == 1
    assert entities[0]["offset"] == 5  # "nice " in units
    assert entities[0]["length"] == 4  # 👍(2) + tone(2)


def test_weave_fe0f_variants_match_both_ways():
    registry = CustomEmojiRegistry()
    registry.add("heart", unicode="❤️", platform="telegram",
                 ref="H1", format="animated")
    entities, woven = weave_custom_emoji_entities(
        "love ❤ ok", (), registry.glyph_map())
    assert woven == 1  # the bare typed form still weaves
    registry2 = CustomEmojiRegistry()
    registry2.add("heart", unicode="❤", platform="telegram",
                  ref="H2", format="animated")
    entities, woven = weave_custom_emoji_entities(
        "love ❤️ ok", (), registry2.glyph_map())
    assert woven == 1
    assert entities[0]["custom_emoji_id"] == "H2"


def test_weave_merges_inside_a_bold_span():
    text = "win 💰 now"
    bold = {"type": "bold", "offset": 0, "length": 10}
    entities, woven = weave_custom_emoji_entities(
        text, [bold], _map(**{"💰": "M1"}))
    assert woven == 1
    assert {"type": "bold", "offset": 0, "length": 10} in entities
    custom = next(e for e in entities
                  if e["type"] == "custom_emoji")
    # the custom entity wraps ONLY the glyph, inside the bold
    assert custom["offset"] == 4 and custom["length"] == 2
    assert bold not in (custom,)  # existing entity untouched


def test_weave_cap_keeps_the_first_matches():
    text = " ".join(["💰"] * 30)
    entities, woven = weave_custom_emoji_entities(
        text, (), _map(**{"💰": "M1"}))
    assert woven == 25  # WEAVE_MAX_CUSTOM_EMOJI, the first 25
    offsets = [e["offset"] for e in entities]
    assert offsets == sorted(offsets)
    _e, woven2 = weave_custom_emoji_entities(
        text, (), _map(**{"💰": "M1"}), cap=2)
    assert woven2 == 2


def test_weave_unregistered_glyphs_stay_plain():
    existing = [{"type": "italic", "offset": 0, "length": 3}]
    entities, woven = weave_custom_emoji_entities(
        "hey 🦄 yo", existing, _map(**{"💰": "M1"}))
    assert woven == 0
    assert entities == existing


def test_weave_never_duplicates_an_existing_custom_entity():
    existing = [{"type": "custom_emoji", "offset": 0, "length": 2,
                 "custom_emoji_id": "OTHER"}]
    entities, woven = weave_custom_emoji_entities(
        "💰!", existing, _map(**{"💰": "M1"}))
    assert woven == 0
    assert entities == existing


def test_glyph_map_prefers_animated_and_needs_a_ref():
    registry = CustomEmojiRegistry()
    registry.add("still", unicode="💰", platform="telegram",
                 ref="S1", format="static")
    registry.add("live", unicode="💰", platform="telegram",
                 ref="A1", format="animated")
    registry.add("faceless", unicode="🚀")  # no telegram ref
    glyph_map = registry.glyph_map()
    assert glyph_map["💰"]["id"] == "A1"
    assert "🚀" not in glyph_map


# ══ B/C · the send hook + the refusal state ══════════════════


def _bot():
    adapter = TelegramBotAdapter(
        token="123:fake", session=SimpleNamespace(),
        owner_sender_ids={"7"})
    adapter._bot_username = "Devon_test_bot"
    return adapter


def _chat(kind=ChatKind.DM, chat_id="42"):
    return ChatRef(platform="telegram-bot", chat_id=chat_id,
                   kind=kind)


def _send_rig(adapter, *, refuse=None, parse_fail=False):
    """Fake the Bot API under a real adapter; returns the
    sendMessage call list.  ``refuse`` is the error text the
    WOVEN attempt dies with; the plain send always lands."""
    calls = []

    def fake_api(method, **params):
        assert method == "sendMessage"
        calls.append(params)
        if "entities" in params and refuse is not None:
            raise RuntimeError(refuse)
        if "entities" in params and parse_fail:
            raise RuntimeError(
                "Bad Request: can't parse entities: "
                "unexpected end of entity")
        return {"message_id": 100 + len(calls)}

    adapter._api = fake_api
    return calls


def _wired(adapter):
    registry = CustomEmojiRegistry()
    registry.add("bag", unicode="💰", platform="telegram",
                 ref="M1", format="animated")
    adapter.custom_emoji_weave_source = registry.glyph_map
    return adapter


def test_send_weaves_registered_glyphs_mid_sentence():
    adapter = _wired(_bot())
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "paid 💰 today")
    assert out.ok
    assert len(calls) == 1
    entities = json.loads(calls[0]["entities"])
    assert entities == [{"type": "custom_emoji", "offset": 5,
                         "length": 2, "custom_emoji_id": "M1"}]
    assert calls[0]["text"] == "paid 💰 today"


def test_send_without_a_provider_is_plain_as_ever():
    adapter = _bot()
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "paid 💰 today")
    assert out.ok and len(calls) == 1
    assert "entities" not in calls[0]


def test_send_with_an_empty_registry_is_plain():
    adapter = _bot()
    adapter.custom_emoji_weave_source = \
        CustomEmojiRegistry().glyph_map
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "paid 💰 today")
    assert out.ok and "entities" not in calls[0]


def test_markup_sends_keep_their_parse_path():
    adapter = _wired(_bot())
    calls = _send_rig(adapter)
    out = adapter.send(_chat(), "<b>paid 💰</b>",
                       parse_mode="HTML")
    assert out.ok and len(calls) == 1
    assert "entities" not in calls[0]
    assert calls[0]["parse_mode"] == "HTML"


def test_refusal_lands_plain_journals_once_and_stops(
        monkeypatch):
    journaled = []
    monkeypatch.setattr(
        telegram_mod, "record_failure",
        lambda name, exc: journaled.append((name, str(exc))))
    adapter = _wired(_bot())
    calls = _send_rig(
        adapter,
        refuse="telegram bot api: sendMessage failed: "
               "Bad Request: custom emoji is not available")
    out = adapter.send(_chat(), "paid 💰 today")
    assert out.ok  # the message still went out — plain
    assert len(calls) == 2
    assert "entities" in calls[0] and "entities" not in calls[1]
    assert adapter._weave_refused.keys() == {"dm"}
    assert "custom emoji is not available" \
        in adapter._weave_refused["dm"]
    # a second DM send doesn't even attempt entities…
    out = adapter.send(_chat(chat_id="42"), "more 💰 here")
    assert out.ok and len(calls) == 3
    assert "entities" not in calls[2]
    # …and the journal heard about the refusal exactly once
    assert [n for n, _e in journaled
            if n == "telegram.weave.refused"] == [
                "telegram.weave.refused"]
    # a GROUP send is a different scope: it still tries, is
    # refused on its own, and is journaled for its own scope
    out = adapter.send(_chat(ChatKind.GROUP, "-100"), "g 💰")
    assert out.ok
    assert adapter._weave_refused.keys() == {"dm", "group"}
    assert [n for n, _e in journaled
            if n == "telegram.weave.refused"] == [
                "telegram.weave.refused"] * 2
    status = adapter.custom_emoji_weaving_status()
    assert status["wired"] is True
    assert set(status["refused"]) == {"dm", "group"}


def test_parse_failure_falls_back_without_refusal_state(
        monkeypatch):
    journaled = []
    monkeypatch.setattr(
        telegram_mod, "record_failure",
        lambda name, exc: journaled.append(name))
    adapter = _wired(_bot())
    calls = _send_rig(adapter, parse_fail=True)
    out = adapter.send(_chat(), "paid 💰 today")
    assert out.ok and len(calls) == 2
    assert "entities" not in calls[1]
    assert adapter._weave_refused == {}  # NOT a refusal
    assert "telegram.weave.entities" in journaled
    # the next send attempts weaving again — no scope ban
    out = adapter.send(_chat(), "again 💰")
    assert out.ok and "entities" in calls[2]


# ══ D · pack import (adapter + registry) ═════════════════════


def test_get_custom_emoji_pack_reads_truthful_formats():
    adapter = _bot()
    seen = {}

    def fake_api(method, **params):
        seen["method"] = method
        seen["params"] = params
        return {"stickers": [
            {"type": "regular", "emoji": "😀",
             "custom_emoji_id": "NO"},
            {"type": "custom_emoji", "emoji": "💰",
             "custom_emoji_id": "A1", "is_animated": True},
            {"type": "custom_emoji", "emoji": "🚀",
             "custom_emoji_id": "V1", "is_video": True},
            {"type": "custom_emoji", "emoji": "🛡",
             "custom_emoji_id": "S1"},
            {"type": "custom_emoji", "emoji": "👻"},
        ]}

    adapter._api = fake_api
    rows = adapter.get_custom_emoji_pack("Rich_by_x")
    assert seen == {"method": "getStickerSet",
                    "params": {"name": "Rich_by_x"}}
    assert rows == [
        {"glyph": "💰", "custom_emoji_id": "A1",
         "format": "animated"},
        {"glyph": "🚀", "custom_emoji_id": "V1",
         "format": "video"},
        {"glyph": "🛡", "custom_emoji_id": "S1",
         "format": "static"},
    ]


def test_verify_custom_emoji_ids_confirms_and_degrades():
    adapter = _bot()
    adapter._api = lambda method, **params: [
        {"custom_emoji_id": "A1"}, {"custom_emoji_id": "V1"}]
    assert adapter.verify_custom_emoji_ids(["A1", "V1", "S1"]) \
        == {"A1", "V1"}
    assert adapter.verify_custom_emoji_ids([]) == set()

    def boom(method, **params):
        raise RuntimeError("network down")

    adapter._api = boom
    assert adapter.verify_custom_emoji_ids(["A1"]) is None


def test_import_stickers_dedupe_rules():
    registry = CustomEmojiRegistry()
    # hers: a created ANIMATED 💰, and a unicode-only 🚀
    registry.add("mybag", unicode="💰", platform="telegram",
                 ref="MINE", format="animated")
    registry.add("rocket_face", unicode="🚀")
    result = registry.import_stickers([
        {"glyph": "💰", "custom_emoji_id": "IMP1",
         "format": "static"},     # hers is better → kept
        {"glyph": "🚀", "custom_emoji_id": "IMP2",
         "format": "animated"},   # no ref yet → upgraded
        {"glyph": "🛡", "custom_emoji_id": "IMP3",
         "format": "animated"},   # new → added
    ], pack="Rich_by_x")
    assert result["added"] == 1 and result["upgraded"] == 1 \
        and result["kept"] == 1
    bag = registry.get("mybag")
    assert bag["refs"]["telegram"] == "MINE"
    assert bag["format"] == "animated"
    rocket = registry.get("rocket_face")
    assert rocket["refs"]["telegram"] == "IMP2"
    assert rocket["format"] == "animated"
    shield_name = result["names"][0]
    shield = registry.get(shield_name)
    assert shield["unicode"] == "🛡"
    assert shield["refs"]["telegram"] == "IMP3"
    assert shield["imported_from"] == "Rich_by_x"


# ══ D · the /customemoji surface (runtime) ═══════════════════


class _CChat:
    def __init__(self, platform, key):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = ChatKind.GROUP


class _CMsg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = {}
        self.sender_id = kw.get("sender_id", "99")
        self.sender = "Owner"
        self.sender_username = "owner"
        self.text = ""
        self.message_id = "55"
        self.reply_to = ""
        self.media = []


class _PackAdapter:
    """A telegram-ish adapter with the pack surface + a
    weaving seam to stamp."""

    def __init__(self, rows=(), *, verify="all", refused=None):
        self.rows = list(rows)
        self.verify_mode = verify
        self.refused = dict(refused or {})
        self.custom_emoji_weave_source = None

    def get_custom_emoji_pack(self, name):
        if name == "Missing_by_x":
            raise RuntimeError("Bad Request: STICKERSET_INVALID")
        return list(self.rows)

    def verify_custom_emoji_ids(self, ids):
        if self.verify_mode == "fail":
            return None
        if self.verify_mode == "subset":
            return set(ids[:1])
        return set(ids)

    def custom_emoji_weaving_status(self):
        return {"wired": self.custom_emoji_weave_source
                is not None,
                "refused": dict(self.refused)}


class _CGateway:
    def __init__(self, adapters):
        self.adapters = adapters


def _runtime(tmp_path, adapter):
    from nomorals.agents.partner.runtime_expression import (
        RuntimeExpressionMixin,
    )
    from nomorals.agents.partner.runtime_group import (
        RuntimeGroupMixin,
    )

    class _RT(RuntimeExpressionMixin, RuntimeGroupMixin):
        def __init__(self):
            self.gateway = _CGateway({"telegram-bot": adapter})
            self.context = SimpleNamespace(
                settings=SimpleNamespace(
                    workspace_dir=str(tmp_path)))

        def _is_operator(self, message):
            return str(message.sender_id or "") == "99"

    return _RT()


_ROWS = [
    {"glyph": "💰", "custom_emoji_id": "A1",
     "format": "animated"},
    {"glyph": "🚀", "custom_emoji_id": "V1",
     "format": "video"},
]


def test_customemoji_import_command_registers_the_pack(
        tmp_path):
    adapter = _PackAdapter(_ROWS)
    rt = _runtime(tmp_path, adapter)
    chat = _CChat("telegram-bot", "telegram-bot:-1001")
    out = rt._control_customemoji("import Rich_by_x",
                                  _CMsg(chat))
    assert "2 new" in str(out) and "0 upgraded" in str(out)
    registry = rt._expression_library().custom_emoji
    glyph_map = registry.glyph_map()
    assert glyph_map["💰"]["id"] == "A1"
    assert glyph_map["🚀"]["format"] == "video"
    # persisted: a fresh library over the same file sees them
    from nomorals.social.chat.expression import ExpressionLibrary

    root = tmp_path / "data" / "social"
    again = ExpressionLibrary(root / "expression_library.json")
    assert again.custom_emoji.glyph_map()["💰"]["id"] == "A1"


def test_customemoji_import_unverified_and_missing(tmp_path):
    adapter = _PackAdapter(_ROWS, verify="subset")
    rt = _runtime(tmp_path, adapter)
    chat = _CChat("telegram-bot", "telegram-bot:-1001")
    out = rt._control_customemoji("import Rich_by_x",
                                  _CMsg(chat))
    assert "1 new" in str(out) and "wouldn't verify" in str(out)
    out = rt._control_customemoji("import Missing_by_x",
                                  _CMsg(chat))
    assert "couldn't read" in str(out)


def test_customemoji_status_reports_the_three_states(tmp_path):
    adapter = _PackAdapter(_ROWS)
    rt = _runtime(tmp_path, adapter)
    chat = _CChat("telegram-bot", "telegram-bot:-1001")
    msg = _CMsg(chat)
    out = str(rt._control_customemoji("status", msg))
    assert "0 named" in out and "not wired" in out
    rt._control_customemoji("import Rich_by_x", msg)
    rt._wire_custom_emoji_weaving()
    out = str(rt._control_customemoji("status", msg))
    assert "2 glyphs carry a Telegram id" in out
    assert "2 of them move" in out
    assert "on — registered glyphs" in out
    adapter.refused = {"dm": "telegram bot api: sendMessage "
                             "failed: Bad Request: custom "
                             "emoji is not available"}
    out = str(rt._control_customemoji("status", msg))
    assert "REFUSED by Telegram" in out
    assert "custom emoji is not available" in out
    assert "owner Premium" in out


# ══ E · WhatsApp is never woven ══════════════════════════════


def test_whatsapp_adapter_has_no_weaving_seam(tmp_path):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    wa = WhatsAppAdapter(media_dir=str(tmp_path / "wa"),
                         outbox_dir=str(tmp_path / "out"),
                         outbox_enabled=False)
    assert not hasattr(wa, "custom_emoji_weave_source")
    adapter = _PackAdapter(_ROWS)
    rt = _runtime(tmp_path, adapter)
    rt.gateway.adapters["whatsapp"] = wa
    rt._wire_custom_emoji_weaving()
    # the telegram-ish adapter got the provider; WhatsApp was
    # not stamped and still has no seam to stamp
    assert adapter.custom_emoji_weave_source is not None
    assert not hasattr(wa, "custom_emoji_weave_source")
