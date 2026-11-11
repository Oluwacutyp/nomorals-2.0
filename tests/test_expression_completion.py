"""EXPRESSION COMPLETION (owner orders, 2026-11-07) — the
pins for the pass that closed the media-dead surfaces:

1. status react ROTATES through a pool (her /customemoji
   set ∪ the owner's additions ∪ a built-in pool), never
   repeating the last face twice; /statusreact pins, lists
   and edits the pool, and turns rotation off/on.
2. TRUE emoji creation: her studio authors 100×100 custom
   emoji locally (deterministic, zero-config); on a Telegram
   bot surface the emoji registers in HER pack (Bot API
   custom_emoji set) and is sent AS an emoji through the
   entity channel; WhatsApp names its wall (no custom emoji
   exists there) and carries the art as a sticker.
3. The emoji-art organ: braille dot art with U+2800 blank
   padding, ASCII, emoji mosaics, drawn subjects, remix
   verbs, the fancy styler, her own combo library.
4. Media in messages: an empty library + dead providers no
   longer means silence — she creates the moment herself
   (provenance ``self:render``); the menu home declares a
   picture head that rides the send as photo+caption+keys
   and edits by caption afterwards.
5. The per-command exercise: every expression-family kind
   driven through the real composed handle_control — no
   exceptions, honest outcomes ("" only when media was
   delivered).

Network-free throughout: providers, backends and the Bot
API are fakes with the real shapes.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.social.chat.expression import (
    REACTION_GIF_COMMANDS,
    ExpressionLibrary,
    ExpressionService,
    ProviderHealth,
)

GIF_BYTES = b"GIF89a" + bytes(range(32))
BOT_KEY = "telegram-bot:-100123"


def _png_bytes(color=(200, 40, 40), size=(120, 90)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# ══ 1 · status react rotation ════════════════════════════════


def _personal(tmp_path, **seams):
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault, **seams)
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked",
        created_at=1.0))
    return personal, registry, vault, rec


def test_rotation_pool_composition_and_walk():
    from nomorals.social.pairing import AccountRecord, status_react_pick, status_react_pool_for

    rec = AccountRecord(
        id="whatsapp:1", platform="whatsapp", identity="1",
        status="linked", created_at=1.0)
    pool = status_react_pool_for(rec, ["🥷", "❤️"])
    assert pool[0] == "🥷"          # her custom set leads
    assert pool.count("❤️") == 1    # deduped across sources
    assert len(pool) >= 10          # the built-in pool is real
    # the walk: consecutive picks never repeat, and the pool
    # cycles (cursor advances only through the PEEK contract)
    seen = []
    cursor = 0
    last = ""
    for _ in range(len(pool)):
        probe = AccountRecord(
            id="whatsapp:1", platform="whatsapp", identity="1",
            status="linked", created_at=1.0,
            status_react_cursor=cursor, status_react_last=last)
        emoji, cursor = status_react_pick(probe, ["🥷", "❤️"])
        assert emoji != last
        seen.append(emoji)
        last = emoji
    assert set(seen) == set(pool)  # a full cycle visits all


def test_rotation_pinned_emoji_wins_when_off():
    from nomorals.social.pairing import AccountRecord, status_react_pick

    rec = AccountRecord(
        id="whatsapp:1", platform="whatsapp", identity="1",
        status="linked", created_at=1.0,
        status_react_emoji="🔥", status_react_rotate=False)
    emoji, _cursor = status_react_pick(rec, ["🥷"])
    assert emoji == "🔥"


def test_status_reactions_rotate_and_journal(tmp_path):
    calls = []

    def reactor(rec, item):
        calls.append(item)
        return True

    personal, registry, vault, rec = _personal(
        tmp_path, status_reactor=reactor,
        status_react_pool=lambda r: ["🥷", "🚀"])
    registry.set_toggles(rec.id, status_react=True)
    faces = []
    for i in range(5):
        personal.observe_status_post(
            "whatsapp", poster_id="222@s.whatsapp.net",
            poster_name="Bola", text="hi", media=[],
            source_id=f"st{i}", ts=10.0 + i)
    faces = [c["emoji"] for c in calls]
    assert len(faces) == 5
    assert all(faces[i] != faces[i - 1] for i in range(1, 5))
    assert set(faces) <= {"🥷", "🚀", "❤️", "🔥", "🙌", "💯",
                          "😍", "🤩", "✨", "👏", "💪", "🎉",
                          "😂", "🫶", "😮", "👍", "🥰", "🙏"}
    assert {"🥷", "🚀"} <= set(faces)  # her custom set rotates in
    journaled = [r for r in vault.journal()
                 if r["kind"] == "status_react"]
    assert [r["note"] for r in journaled] == [
        f"reacted {face}" for face in faces]  # the pick is journaled


def test_status_react_rotation_controls(tmp_path):
    personal, registry, _vault, rec = _personal(tmp_path)
    personal.set_status_react_rotation(rec.id, rotate=False)
    assert registry.get(rec.id).status_react_rotate is False
    personal.set_status_react_rotation(
        rec.id, add=("🌚",), rotate=True)
    rec = registry.get(rec.id)
    assert rec.status_react_rotate is True
    assert "🌚" in rec.status_react_pool
    personal.set_status_react_rotation(rec.id, remove=("🌚",))
    assert "🌚" not in registry.get(rec.id).status_react_pool
    # unknown account: the setter family's plain ValueError
    with pytest.raises(ValueError, match="no paired account"):
        personal.set_status_react_rotation("whatsapp:nope",
                                           rotate=True)


# ══ 2 · true emoji creation ══════════════════════════════════


def test_studio_renderer_is_deterministic_and_sized(tmp_path):
    from nomorals.social.chat import emoji_studio

    a = emoji_studio.emoji_png(
        emoji_studio.parse_design("a purple laughing cat"),
        tmp_path, name="one")
    b = emoji_studio.emoji_png(
        emoji_studio.parse_design("a purple laughing cat"),
        tmp_path, name="two")
    assert a.read_bytes() == b.read_bytes()  # deterministic
    with Image.open(a) as img:
        assert img.size == (100, 100)  # the custom-emoji format
        assert img.mode == "RGBA"
    other = emoji_studio.emoji_png(
        emoji_studio.parse_design("a gold rocket"),
        tmp_path, name="three")
    assert other.read_bytes() != a.read_bytes()


def test_bot_api_custom_emoji_pack_verbs(tmp_path):
    from nomorals.social.chat.base import ChatRef
    from nomorals.social.chat.telegram import TelegramBotAdapter

    png = tmp_path / "e.png"
    png.write_bytes(_png_bytes())
    adapter = TelegramBotAdapter(
        token="123:fake", session=SimpleNamespace(),
        owner_sender_ids={"7"})
    adapter._bot_username = "Devon_test_bot"
    calls = []
    state = {"created": False}

    def fake_api(method, **params):
        calls.append((method, params))
        if method == "getStickerSet":
            if not state["created"]:
                raise RuntimeError("Bad Request: STICKERSET_INVALID")
            return {"stickers": [
                {"emoji": "🚀", "custom_emoji_id": "CEID1"}]}
        if method == "sendMessage":
            return {"message_id": 42}
        raise AssertionError(method)

    uploads = []

    def fake_upload(method, fields, file_field, path, **kw):
        uploads.append((method, fields, file_field))
        state["created"] = True
        return True

    adapter._api = fake_api
    adapter._api_upload = fake_upload
    made = adapter.create_or_extend_custom_emoji_pack(
        "7", "rocket", "Devon — rocket", str(png), "🚀")
    assert made["created"] is True
    assert made["custom_emoji_id"] == "CEID1"
    assert made["pack"] == "devon_rocket_by_Devon_test_bot"
    method, fields, file_field = uploads[0]
    assert method == "createNewStickerSet"
    assert fields["sticker_type"] == "custom_emoji"
    chat = ChatRef(platform="telegram-bot", chat_id="5")
    sent = adapter.send_custom_emoji(chat, "🚀", "CEID1")
    assert sent.ok and sent.message_id == "42"
    params = dict(calls[-1][1])
    assert params["text"] == "🚀"
    import json as _json

    entities = _json.loads(params["entities"])
    assert entities[0]["type"] == "custom_emoji"
    assert entities[0]["custom_emoji_id"] == "CEID1"
    assert entities[0]["length"] == 2  # UTF-16 units of 🚀


# ── runtime rigs (the test_emoji_creation shapes) ────────────


class _Chat:
    def __init__(self, platform, key, kind=None):
        from nomorals.social.chat.base import ChatKind

        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = kind or ChatKind.DM


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


class _PackAdapter:
    """A Telegram bot adapter fake WITH the pack verbs."""

    def __init__(self):
        self.media_sent = []
        self.custom_sent = []
        self.pack_calls = []
        self.owner_sender_ids = {"7"}
        self._bot_id = 4242

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "administrator"}}

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="TGFILE1", error="")

    def create_or_extend_custom_emoji_pack(
            self, user_id, slug, title, art_path, fallback,
            format="animated"):
        assert Path(art_path).exists()
        self.pack_calls.append(format)
        return {"pack": f"devon_{slug}_by_devonbot",
                "custom_emoji_id": "CEID9", "created": True,
                "format": format}

    def send_custom_emoji(self, chat, fallback, custom_emoji_id):
        self.custom_sent.append((fallback, custom_emoji_id))
        return SimpleNamespace(ok=True, message_id="10", error="")


class _PlainAdapter:
    """No pack verbs, no custom sends (the WhatsApp shape)."""

    def __init__(self):
        self.media_sent = []
        self._bot_id = 4242

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


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


def test_make_on_telegram_registers_and_sends_as_emoji(tmp_path):
    adapter = _PackAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    out = str(rt._control_emoji("make a gold rocket",
                                 _Msg(chat)))
    registry = rt._expression_library().custom_emoji
    names = registry.names()
    assert len(names) == 1
    entry = registry.get(names[0])
    # registered: the telegram ref IS the pack's custom_emoji_id
    assert entry["refs"]["telegram-bot"] == "CEID9"
    assert entry["unicode"] == "🚀"
    # sent AS an emoji through the entity channel — and NOT
    # as a sticker file
    assert adapter.custom_sent == [("🚀", "CEID9")]
    assert adapter.media_sent == []
    assert "is live" in out and "real custom emoji" in out


def test_make_on_whatsapp_names_the_wall(tmp_path):
    adapter = _PlainAdapter()
    rt = _runtime(tmp_path, {"whatsapp": adapter})
    chat = _Chat("whatsapp", "whatsapp:5")
    out = str(rt._control_emoji("make a gold rocket",
                                 _Msg(chat)))
    assert "WhatsApp has no custom emoji" in out  # the wall, named
    assert "travels as a sticker" in out
    assert len(adapter.media_sent) == 1  # the art still lands
    registry = rt._expression_library().custom_emoji
    assert len(registry.names()) == 1  # and joins her set


def test_customemoji_show_sends_the_emoji_itself(tmp_path):
    adapter = _PackAdapter()
    rt = _runtime(tmp_path, {"telegram-bot": adapter})
    chat = _Chat("telegram-bot", "telegram-bot:5")
    rt._control_emoji("make a gold rocket", _Msg(chat))
    name = rt._expression_library().custom_emoji.names()[0]
    adapter.custom_sent.clear()
    out = str(rt._control_customemoji(f"show {name}",
                                      _Msg(chat)))
    assert adapter.custom_sent == [("🚀", "CEID9")]
    assert "live as an emoji" in out


# ══ 3 · the emoji-art organ ══════════════════════════════════


def test_braille_art_shape_padding_and_roundtrip():
    from nomorals.social.chat import emoji_art

    art = emoji_art.image_to_braille(_png_bytes(), width=20)
    lines = art.splitlines()
    assert lines and all(len(ln) == 20 for ln in lines)
    # padding is the braille blank, never a plain space —
    # chat renderers collapse spaces and would eat the grid
    assert " " not in art
    matrix = emoji_art.piece_to_matrix(art)
    assert emoji_art.matrix_to_braille(matrix) == art


def _half_bytes() -> bytes:
    import io

    img = Image.new("RGB", (120, 90), "white")
    for x in range(60):
        for y in range(90):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_art_transforms_and_styles():
    from nomorals.social.chat import emoji_art

    art = emoji_art.image_to_braille(_half_bytes(), width=16)
    assert emoji_art.invert_piece(art) != art
    assert emoji_art.mirror_piece(art) != art
    small = emoji_art.scale_piece(art, 0.5)
    assert len(small.splitlines()) < len(art.splitlines())
    assert "⬛" in emoji_art.restyle_piece(art, "mosaic")
    assert emoji_art.fancy_text("devon", "bold") == "𝐝𝐞𝐯𝐨𝐧"
    assert emoji_art.fancy_text("hi", "sparkle") == "✨ hi ✨"
    assert emoji_art.fancy_text("hi", "nope") is None
    assert "bold" in emoji_art.FANCY_STYLES


def test_combo_library_seeds_search_and_growth(tmp_path):
    from nomorals.social.chat.emoji_art import ComboLibrary

    lib = ComboLibrary(tmp_path / "combos.json")
    assert len(lib) >= 15  # her seed shelf (drawn, not copied)
    assert all(p["source"] == "seed:self" for p in lib.pieces())
    hits = lib.search("cat")
    assert any("cat" in p["name"] for p in hits)
    lib.add(name="my wave", text="〜〜", kind="combo",
            tags=["wave"], source="owner")
    assert len(lib) == len(ComboLibrary(tmp_path / "combos.json"))


def test_describe_art_library_first_then_drawn(tmp_path):
    from nomorals.social.chat.emoji_art import ComboLibrary, describe_art

    lib = ComboLibrary(tmp_path / "combos.json")
    art, how = describe_art("rocket", lib)
    assert how == "her library" and art
    art2, how2 = describe_art("a sleepy dragon", lib)
    assert art2 and "drawn" in how2


def test_emojiart_command_draw_fancy_combo_remix(tmp_path):
    adapter = _PlainAdapter()
    rt = _runtime(tmp_path, {"whatsapp": adapter})
    chat = _Chat("whatsapp", "whatsapp:5")
    out = str(rt._control_emojiart("draw a gold heart",
                                   _Msg(chat)))
    assert "drawn for you" in out or "her library" in out
    out = str(rt._control_emojiart("fancy bold devon",
                                   _Msg(chat)))
    assert out == "𝐝𝐞𝐯𝐨𝐧"
    out = str(rt._control_emojiart("combo cat", _Msg(chat)))
    assert "combo library" in out
    before = rt._emojiart_last_for(_Msg(chat))
    out = str(rt._control_emojiart("remix invert", _Msg(chat)))
    assert out != before  # the remix actually reshaped the piece
    assert any("⠀" <= ch <= "⣿" for ch in out)  # braille landed
    out = str(rt._control_emojiart("save my heart", _Msg(chat)))
    assert "in my combo library" in out


# ══ 4 · media in messages ════════════════════════════════════


def _ok_send(sent):
    def _send(ref):
        sent.append(ref)
        return SimpleNamespace(ok=True, message_id="1",
                               media_file_id="", error="")
    return _send


def test_moment_self_creates_when_library_and_providers_empty(
        tmp_path):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    svc = ExpressionService(lib, providers=[],
                            media_dir=tmp_path / "m", cooldown_s=0)
    sent = []
    out = svc.fire("welcome", room="r", platform="whatsapp",
                   settings={}, send=_ok_send(sent))
    assert out["sent"] is True
    assert len(sent) == 1
    assert lib.items[0]["source"] == "self:render"  # provenance
    # and the honest line survives where creation is
    # impossible (no media home at all)
    svc2 = ExpressionService(
        ExpressionLibrary(tmp_path / "lib2.json"), providers=[],
        media_dir=None, cooldown_s=0)
    out2 = svc2.fire("welcome", room="r", platform="whatsapp",
                     settings={}, send=_ok_send([]))
    assert out2["sent"] is False
    assert out2["reason"] == "no media available — text only"


def test_reaction_honest_line_when_no_media_home(tmp_path):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    svc = ExpressionService(lib, providers=[], media_dir=None,
                            cooldown_s=0)
    out = svc.reaction_gif("hug", room="r", platform="whatsapp",
                           send=_ok_send([]))
    assert out["sent"] is False
    assert out["reason"] == (
        "no /hug GIF reached me — every source is down and my "
        "library has none yet")


def test_menu_banner_item_renders_once_and_shelves(tmp_path):
    lib = ExpressionLibrary(tmp_path / "lib.json")
    svc = ExpressionService(lib, providers=[],
                            media_dir=tmp_path / "m", cooldown_s=0)
    item = svc.menu_banner_item()
    assert item is not None and Path(item["path"]).exists()
    with Image.open(item["path"]) as img:
        assert img.size == (1024, 512)
    assert item["source"] == "self:render"
    assert svc.menu_banner_item()["id"] == item["id"]  # shelved


def test_menu_render_carries_the_media_head():
    from nomorals.social.chat.control import menu_session_render

    media = {"path": "/x/banner.png", "kind": "image"}
    render = menu_session_render(
        platform="telegram-bot", public_only=True, home=True,
        media=media)
    assert render.media == media
    plain = menu_session_render(
        platform="telegram-bot", public_only=True, home=True)
    assert plain.media is None


# ══ composed runtime: menu media flow + the exercise ═════════


def _composed_runtime():
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-exprc-", dir=os.environ.get("NM_TEST_TMPDIR")
        or None)
    settings = load_settings(overrides={
        "home": tmp.name, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false"})
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context), tmp


class _MenuAdapter:
    """Telegram bot adapter fake with the media+buttons verb
    and the caption edit verb."""

    name = "telegram-bot"

    def __init__(self):
        self.media_sent = []
        self.caption_edits = []
        self.owner_sender_ids = {"7"}

    def send_media(self, chat, ref, caption="", parse_mode="",
                   buttons=None):
        self.media_sent.append(
            {"ref": ref, "caption": caption, "buttons": buttons})
        return SimpleNamespace(ok=True, message_id="77",
                               media_file_id="", error="")

    def edit_caption(self, chat, message_id, caption, buttons=None,
                     parse_mode=""):
        self.caption_edits.append(
            {"message_id": message_id, "caption": caption,
             "buttons": buttons})
        return {"message_id": int(message_id)}


def test_menu_home_rides_a_picture_head_then_edits_caption():
    from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
    from nomorals.social.chat.control import parse_control

    rt, tmp = _composed_runtime()
    try:
        adapter = _MenuAdapter()
        rt.gateway.adapters["telegram-bot"] = adapter
        chat = ChatRef(platform="telegram-bot", chat_id="5",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True, text="/menu",
                          sender="Ada", sender_id="7",
                          message_id="1")
        render = rt._menu_session_reply(msg, "telegram-bot:5",
                                        parse_control("/menu"))
        assert render is not None and render.media is not None
        result = rt._send_control_reply(msg, render)
        assert getattr(result, "ok", False)
        # ONE media message: photo head + the menu as caption
        # + the packed category keyboard riding it
        assert len(adapter.media_sent) == 1
        head = adapter.media_sent[0]
        assert head["ref"].kind == "image"
        assert "command menu" in head["caption"]
        assert head["buttons"] and len(head["buttons"]) >= 3
        registry = rt._live_session_registry()
        assert registry.is_media("telegram-bot:5", "menu") is True
        # a page render now edits the CAPTION of that same
        # message — still one live message, never a second post
        msg2 = ChatMessage(chat=chat, incoming=True,
                           text="/list media", sender="Ada",
                           sender_id="7", message_id="2")
        render2 = rt._menu_session_reply(
            msg2, "telegram-bot:5", parse_control("/list media"))
        assert render2 is not None
        rt._send_control_reply(msg2, render2)
        assert len(adapter.media_sent) == 1  # no second post
        assert len(adapter.caption_edits) == 1
        assert adapter.caption_edits[0]["message_id"] == "77"
    finally:
        tmp.cleanup()


def test_couplepp_self_art_when_pools_down(tmp_path):
    from nomorals.social.chat.makers import MakerService

    class DeadPool:
        name = "deadpool"

        def image_url(self):
            raise OSError("pool down")

    svc = MakerService(media_dir=tmp_path / "mk",
                       chains={"couplepp": [DeadPool()]},
                       bytes_fetcher=lambda url: b"")
    sent = []
    out = svc.couplepp(send=lambda ref, caption: sent.append(
        (ref, caption)) or SimpleNamespace(ok=True))
    assert out["sent"] is True
    assert out["source"] == "self:render"  # her own artwork
    assert "my own artwork" in sent[0][1]  # captioned honestly


class _FakeReactionProvider:
    def __init__(self, name):
        self.name = name

    def fetch(self, category):
        return {"url": f"https://x.test/{category}.gif",
                "provider": self.name}


def _exercise_runtime(tmp_path_home):
    """The composed runtime with every network seam faked:
    reaction providers return GIF bytes, the couplepp pool
    returns a real picture, the anime pools report down."""
    rt, tmp = _composed_runtime()
    library = ExpressionLibrary(Path(tmp.name) / "rx.json")
    dance = Path(tmp.name) / "dance.gif"
    dance.write_bytes(GIF_BYTES)
    library.add_item(kind="gif", path=str(dance), tags=["dance"],
                     source="test")
    rt._expression_service_obj = ExpressionService(
        library, media_dir=Path(tmp.name) / "rx_media",
        providers=[],
        reaction_providers=[_FakeReactionProvider("waifu.pics"),
                            _FakeReactionProvider("nekos.best")],
        bytes_fetcher=lambda url: GIF_BYTES,
        health=ProviderHealth(), rng=random.Random(0))
    from nomorals.social.chat.makers import MakerService

    class Pool:
        name = "fakepool"

        def image_url(self):
            return "https://x.test/pair.png"

    rt._makers_service_obj = MakerService(
        media_dir=Path(tmp.name) / "mk",
        chains={"couplepp": [Pool()]},
        bytes_fetcher=lambda url: _png_bytes())
    from nomorals.anime.service import Outcome

    class AnimeDown:
        def picture(self, kind):
            return Outcome(ok=False, title="Anime",
                           error="the picture pools are down")

    rt._anime_service_obj = AnimeDown()
    return rt, tmp


def test_per_command_exercise_table():
    """Item 5: every expression-family command through the
    REAL composed handle_control on a DM surface.  Verdicts:
    "" is a pass ONLY where media delivery is the contract;
    anything else must be a non-empty honest string; an
    exception anywhere fails the whole table."""
    from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef, MediaRef

    rt, tmp = _exercise_runtime(None)
    try:
        chat = ChatRef(platform="local", chat_id="room1",
                       kind=ChatKind.DM)
        pic = Path(tmp.name) / "pic.png"
        pic.write_bytes(_png_bytes())
        gif = Path(tmp.name) / "anim.gif"
        gif.write_bytes(GIF_BYTES)

        def drive(text, media=None, reply_to=""):
            msg = ChatMessage(
                chat=chat, incoming=True, text=text, sender="Ada",
                sender_id="7", message_id="55",
                media=media or [], reply_to=reply_to)
            return rt.handle_control(text, "local:room1",
                                     message=msg)

        verdicts = {}
        # the media family
        verdicts["steal"] = drive(
            "/steal party", media=[MediaRef(
                path=str(gif), kind="animation")])
        verdicts["sticker"] = drive("/sticker hello there")
        verdicts["convert"] = drive(
            "/convert image-to-sticker",
            media=[MediaRef(path=str(pic), kind="image")])
        verdicts["gif"] = drive("/gif dance")
        verdicts["emoji"] = drive("/emoji rocket")
        verdicts["emoji make"] = drive("/emoji make a moon fox")
        verdicts["emoji sticker"] = drive(
            "/emoji sticker a gold star")
        verdicts["customemoji"] = drive("/customemoji list")
        verdicts["emojiart draw"] = drive(
            "/emojiart draw a gold heart")
        verdicts["emojiart dot"] = drive(
            "/emojiart dot", media=[MediaRef(
                path=str(pic), kind="image")])
        verdicts["emojiart fancy"] = drive(
            "/emojiart fancy script devon")
        verdicts["react"] = drive("/react 🔥", reply_to="44")
        # every per-emotion reaction command
        for kind in sorted(REACTION_GIF_COMMANDS):
            verdicts[kind] = drive(f"/{kind}")
        # the owner's named set
        verdicts["couplepp"] = drive("/couplepp")
        verdicts["pp"] = drive("/pp")
        for kind in ("waifu", "neko", "kitsune", "husbando",
                     "charpic"):
            verdicts[kind] = drive(f"/{kind}")
        media_delivered = {
            "steal", "sticker", "convert", "gif", "couplepp",
            "pp", "emoji sticker"} | set(REACTION_GIF_COMMANDS)
        problems = []
        for kind, reply in verdicts.items():
            if not isinstance(reply, str):
                reply = str(getattr(reply, "text", "") or "")
            if kind in media_delivered:
                if reply.strip():
                    problems.append(
                        f"{kind}: expected media delivery (''), "
                        f"got text: {reply[:80]!r}")
            elif not reply.strip():
                problems.append(f"{kind}: silent (no reply)")
        assert problems == [], "\n".join(problems)
    finally:
        tmp.cleanup()
