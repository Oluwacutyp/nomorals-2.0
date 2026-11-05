"""FEI gap wave C — the per-emotion reaction GIF commands
(/hug, /pat, /bonk, …) at every seam:

- the category table: every shipped command maps to REAL
  categories of its sources (the canonical waifu.pics SFW /
  nekos.best v2 lists, pinned below) — a command added
  without a mapping fails here, and the runtime's literal
  dispatch set can never drift from the table;
- the providers' real response shapes (fixtures shaped from
  the published APIs), the health-checked failover order,
  the sniff (a still is never sent), and the library floor;
- the runtime handler: target resolution (reply → @name →
  solo), caption forms, the §33 gate, and the animation
  delivery shape (adapter send_media + caption, one message);
- the composed control path: a delivered reaction (and a
  delivered /gif) returns "" — MEDIA_DELIVERED — and never
  the never-empty law's "produced no reply" fallback.

Network-free throughout: every fetcher is a fake.
"""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_expression import (
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.expression import (
    MEDIA_DELIVERED_KINDS,
    REACTION_GIF_COMMANDS,
    REACTION_GIFS,
    ExpressionLibrary,
    ExpressionService,
    NekosBestProvider,
    ProviderHealth,
    WaifuPicsProvider,
    reaction_caption,
)

BOT_KEY = "telegram-bot:-100123"
USER_KEY = "telegram:-100999"

#: The sources' canonical category lists (their published
#: SFW / v2 sets) — the floor the table is checked against.
WAIFU_PICS_SFW = frozenset([
    "waifu", "neko", "shinobu", "megumin", "bully", "cuddle",
    "cry", "hug", "awoo", "kiss", "lick", "pat", "smug", "bonk",
    "yeet", "blush", "smile", "wave", "handhold", "highfive",
    "nom", "bite", "glomp", "slap", "kill", "kick", "happy",
    "wink", "poke", "dance", "cringe"])
NEKOS_BEST_V2 = frozenset([
    "neko", "waifu", "husbando", "kitsune", "baka", "bite",
    "blush", "bored", "cry", "cuddle", "dance", "facepalm",
    "feed", "happy", "highfive", "hug", "kick", "kiss", "laugh",
    "nom", "pat", "poke", "pout", "shrug", "slap", "sleep",
    "smile", "smug", "stare", "thumbsup", "tickle", "wave",
    "wink", "yeet", "handshake", "handhold"])
CANONICAL = {"waifu.pics": WAIFU_PICS_SFW, "nekos.best": NEKOS_BEST_V2}

GIF_BYTES = b"GIF89a" + bytes(range(32))
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + bytes(range(32))
WEBP_BYTES = b"RIFF\x20\x00\x00\x00WEBPVP8 " + bytes(range(16))


# ── the table + the registry ─────────────────────────────────


def test_every_command_maps_to_real_source_categories():
    assert frozenset(REACTION_GIFS) == REACTION_GIF_COMMANDS
    for name, entry in REACTION_GIFS.items():
        sources = entry["sources"]
        assert sources, f"/{name} has no source chain"
        providers = [p for p, _c in sources]
        # the seed order is waifu.pics first, nekos.best second
        assert providers == sorted(
            providers, key=("waifu.pics", "nekos.best").index), name
        for provider, category in sources:
            assert category in CANONICAL[provider], (
                f"/{name} maps to {provider}:{category}, which is "
                "not a real category of that source")
        assert entry["obj"] and "{obj}" in entry["obj"], name
        if entry["transitive"]:
            assert entry["obj"], name
        else:
            assert entry["plain"], f"/{name} needs a solo phrase"


def test_owned_names_are_never_hijacked():
    # /kick is member moderation and /kill is the conflict
    # system in Devon — the FEI reactions of those names
    # stay unshipped rather than steal a working command.
    assert "kick" not in REACTION_GIFS
    assert "kill" not in REACTION_GIFS


def test_dispatch_literal_set_matches_the_table():
    src = Path("nomorals/agents/partner/runtime.py").read_text()
    start = src.index('"awoo", "baka"')
    block = src[start:src.index("}:", start)]
    kinds = set(re.findall(r'"([a-z]+)"', block))
    assert kinds == set(REACTION_GIF_COMMANDS)


def test_registry_presence_for_every_command():
    from nomorals.social.chat import control

    for name in REACTION_GIF_COMMANDS:
        assert name in control.CONTROL_COMMANDS, name
        assert name in control.PUBLIC_COMMANDS, name
        assert name in control.COMMAND_DETAILS, name
        assert name in control.LIST_ONELINERS, name
        memberships = sum(
            name in members for _g, members in control.LIST_GROUPS)
        assert memberships == 1, (name, memberships)
        assert control.detailed_help(name), name
    menu = control.help_menu("whatsapp")
    for name in REACTION_GIF_COMMANDS:
        assert f"/{name}" in menu, name
    assert frozenset(
        {"steal", "sticker", "convert", "gif", "react"}) \
        == MEDIA_DELIVERED_KINDS


# ── captions ──────────────────────────────────────────────────


def test_caption_forms():
    assert reaction_caption("hug", actor="Ada", target="Bola") \
        == "Ada hugs Bola 💞"
    assert reaction_caption("hug", actor="Ada", target="@bola") \
        == "Ada hugs @bola 💞"
    # solo: the owner summoning on himself gets "you"
    assert reaction_caption("hug", actor="Peace", owner=True) \
        == "Devon hugs you 💞"
    # solo: a member's transitive reaction names them
    assert reaction_caption("hug", actor="Ada") == "Devon hugs Ada 💞"
    # solo: a one-person reaction is just Devon performing it
    assert reaction_caption("dance", actor="Ada") == "Devon dances 💃"
    assert reaction_caption("dance", actor="Ada", target="Bola") \
        == "Ada dances with Bola 💃"
    assert reaction_caption("baka", actor="Ada", target="Bola") \
        == "Ada calls Bola a baka 💢"
    assert reaction_caption("nosuch", actor="Ada") == ""


# ── providers (real response shapes) ─────────────────────────


def test_waifu_pics_provider_shape():
    seen = []

    def fetcher(url):
        seen.append(url)
        return {"url": "https://i.waifu.pics/abc.gif"}

    hit = WaifuPicsProvider(fetcher=fetcher).fetch("hug")
    assert seen == ["https://api.waifu.pics/sfw/hug"]
    assert hit["url"] == "https://i.waifu.pics/abc.gif"
    assert WaifuPicsProvider(fetcher=fetcher).search("hug") == []


def test_nekos_best_provider_shape():
    seen = []

    def fetcher(url):
        seen.append(url)
        return {"results": [{"url": "https://cdn.nekos.best/x.gif",
                             "anime_name": "Some Anime"}]}

    hit = NekosBestProvider(fetcher=fetcher).fetch("pat")
    assert seen == ["https://nekos.best/api/v2/pat"]
    assert hit["url"] == "https://cdn.nekos.best/x.gif"
    assert hit["title"] == "Some Anime"
    empty = NekosBestProvider(fetcher=lambda url: {"results": []})
    assert empty.fetch("pat")["url"] == ""


# ── the service: chain, sniff, floor ─────────────────────────


class _FakeProvider:
    def __init__(self, name, *, url="", error=None):
        self.name = name
        self.url = url
        self.error = error
        self.calls = []

    def fetch(self, category):
        self.calls.append(category)
        if self.error is not None:
            raise self.error
        return {"url": self.url, "title": category, "kind": "gif"}


def _service(tmp_path, providers, bytes_by_url):
    library = ExpressionLibrary(tmp_path / "library.json")
    return ExpressionService(
        library, media_dir=tmp_path / "media",
        reaction_providers=providers,
        bytes_fetcher=lambda url: bytes_by_url[url],
        health=ProviderHealth(), rng=random.Random(0))


def _recording_send(box):
    def send(ref):
        box.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="TGFILE1", error="")
    return send


def test_service_fetches_first_seed_and_sends_animation(tmp_path):
    wp = _FakeProvider("waifu.pics", url="https://x/hug.gif")
    nb = _FakeProvider("nekos.best", url="https://y/hug.gif")
    svc = _service(tmp_path, [wp, nb], {"https://x/hug.gif": GIF_BYTES})
    sent = []
    out = svc.reaction_gif("hug", room=BOT_KEY, platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is True
    assert wp.calls == ["hug"] and nb.calls == []  # first seed wins
    assert len(sent) == 1
    assert sent[0].kind == "animation"  # a GIF rides sendAnimation
    assert sent[0].path.endswith(".gif")
    # the fetch joined her library, tagged with the command
    items = svc.library.search("hug")
    assert items and items[0]["tags"] == ["hug"]


def test_service_failover_order_and_health(tmp_path):
    wp = _FakeProvider("waifu.pics", error=RuntimeError("down"))
    nb = _FakeProvider("nekos.best", url="https://y/pat.gif")
    svc = _service(tmp_path, [wp, nb], {"https://y/pat.gif": GIF_BYTES})
    sent = []
    out = svc.reaction_gif("pat", room=BOT_KEY, platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is True
    assert wp.calls == ["pat"] and nb.calls == ["pat"]
    assert svc.health.status("waifu.pics")["fails"] == 1
    assert svc.health.status("nekos.best").get("fails", 0) == 0


def test_service_never_sends_a_still(tmp_path):
    wp = _FakeProvider("waifu.pics", url="https://x/blush.png")
    nb = _FakeProvider("nekos.best", url="https://y/blush.png")
    svc = _service(tmp_path, [wp, nb],
                   {"https://x/blush.png": PNG_BYTES,
                    "https://y/blush.png": PNG_BYTES})
    sent = []
    out = svc.reaction_gif("blush", room=BOT_KEY,
                           platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is False
    assert sent == []  # nothing was ever sent
    assert "every source is down" in out["reason"]


def test_service_still_falls_through_to_a_real_gif(tmp_path):
    wp = _FakeProvider("waifu.pics", url="https://x/wave.png")
    nb = _FakeProvider("nekos.best", url="https://y/wave.gif")
    svc = _service(tmp_path, [wp, nb],
                   {"https://x/wave.png": PNG_BYTES,
                    "https://y/wave.gif": GIF_BYTES})
    sent = []
    out = svc.reaction_gif("wave", room=BOT_KEY,
                           platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is True and len(sent) == 1


def test_service_library_floor_when_sources_are_down(tmp_path):
    wp = _FakeProvider("waifu.pics", error=RuntimeError("down"))
    nb = _FakeProvider("nekos.best", error=RuntimeError("down"))
    svc = _service(tmp_path, [wp, nb], {})
    floor = tmp_path / "floor.gif"
    floor.write_bytes(GIF_BYTES)
    svc.library.add_item(kind="gif", path=str(floor), tags=["hug"],
                         source="test")
    sent = []
    out = svc.reaction_gif("hug", room=BOT_KEY, platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is True
    assert sent[0].path == str(floor)


def test_service_all_down_empty_library_is_one_honest_line(tmp_path):
    wp = _FakeProvider("waifu.pics", error=RuntimeError("down"))
    nb = _FakeProvider("nekos.best", error=RuntimeError("down"))
    svc = _service(tmp_path, [wp, nb], {})
    sent = []
    out = svc.reaction_gif("hug", room=BOT_KEY, platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is False
    assert sent == []
    assert out["reason"] == (
        "no /hug GIF reached me — every source is down and my "
        "library has none yet")


def test_service_sniffed_kinds_ride_the_right_send_kind(tmp_path):
    wp = _FakeProvider("waifu.pics", url="https://x/wink.webp")
    svc = _service(tmp_path, [wp], {"https://x/wink.webp": WEBP_BYTES})
    sent = []
    out = svc.reaction_gif("wink", room=BOT_KEY,
                           platform="telegram-bot",
                           send=_recording_send(sent))
    assert out["sent"] is True
    assert sent[0].path.endswith(".webp")
    assert sent[0].kind == "sticker"  # animated WebP rides as a sticker

    wp2 = _FakeProvider("waifu.pics", url="https://x/smile.mp4")
    svc2 = _service(tmp_path, [wp2], {"https://x/smile.mp4": MP4_BYTES})
    sent2 = []
    out2 = svc2.reaction_gif("smile", room=BOT_KEY,
                             platform="telegram-bot",
                             send=_recording_send(sent2))
    assert out2["sent"] is True
    assert sent2[0].path.endswith(".mp4")
    assert sent2[0].kind == "animation"


# ── the runtime handler ───────────────────────────────────────


class _Chat:
    def __init__(self, platform, key, title="Builders"):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = title
        self.kind = ChatKind.GROUP


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
    def __init__(self, tmp_path, admin_ids=(),
                 bot_status="administrator"):
        self.media_sent = []  # (ref, caption)
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status
        self.tmp_path = tmp_path

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append((ref, caption))
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="TGFILE1", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
    def __init__(self, tmp_path, adapters, operators=()):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators

    def turn(self, msg):
        return self._group_surface_turn(msg)


def _rig(tmp_path, platform="telegram-bot", key=BOT_KEY, **kw):
    adapter = _Adapter(tmp_path, **kw)
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    chat = _Chat(platform, key)
    # a fake-provider expression service: the handler must
    # never touch the network from a test
    wp = _FakeProvider("waifu.pics", url="https://x/reaction.gif")
    nb = _FakeProvider("nekos.best", url="https://y/reaction.gif")
    library = ExpressionLibrary(tmp_path / "rx_library.json")
    rt._expression_service_obj = ExpressionService(
        library, media_dir=tmp_path / "rx_media",
        reaction_providers=[wp, nb],
        bytes_fetcher=lambda url: GIF_BYTES,
        health=ProviderHealth(), rng=random.Random(0))
    return rt, adapter, chat


def _join(chat, user_id="8", name="Bola", username="bola"):
    return _Msg(chat, meta={"service": "join", "new_members": [
        {"id": user_id, "username": username, "name": name}]},
        sender_id=user_id, sender=name, sender_username=username)


def test_handler_reply_target_sends_one_captioned_animation(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    msg = _Msg(chat, reply_to="44", meta={"reply_to_sender": {
        "id": "8", "username": "bola", "name": "Bola"}})
    out = rt._control_reaction("hug", "", msg)
    assert out == ""  # the media WAS the reply
    assert len(adapter.media_sent) == 1
    ref, caption = adapter.media_sent[0]
    assert ref.kind == "animation"
    assert caption == "Ada hugs Bola 💞"


def test_handler_name_target_resolves_through_the_ledger(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    assert rt.turn(_join(chat)) is True  # Bola joins; ledger learns her
    adapter.media_sent.clear()
    out = rt._control_reaction("pat", "@bola", _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == 1
    _ref, caption = adapter.media_sent[0]
    assert caption == "Ada pats Bola 🫳"


def test_handler_solo_member_and_owner_forms(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_reaction("hug", "", _Msg(chat))
    assert out == ""
    _ref, caption = adapter.media_sent[-1]
    assert caption == "Devon hugs Ada 💞"
    owner = _Msg(chat, sender_id="99", sender="Peace",
                 sender_username="peace")
    out = rt._control_reaction("hug", "", owner)
    assert out == ""
    _ref, caption = adapter.media_sent[-1]
    assert caption == "Devon hugs you 💞"
    out = rt._control_reaction("dance", "", _Msg(chat))
    assert out == ""
    _ref, caption = adapter.media_sent[-1]
    assert caption == "Devon dances 💃"


def test_handler_unresolvable_name_is_honest_not_solo(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_reaction("hug", "@ghostperson", _Msg(chat))
    assert "couldn't find" in str(out)
    assert adapter.media_sent == []


def test_handler_unclaimed_room_is_gated(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    out = rt._control_reaction("hug", "", _Msg(chat))
    assert "isn't claimed" in str(out)
    assert adapter.media_sent == []


def test_handler_no_message_is_usage(tmp_path):
    rt, _adapter, _chat = _rig(tmp_path)
    out = rt._control_reaction("hug", "", None)
    assert out.startswith("usage: /hug")


# ── the composed control path (MEDIA_DELIVERED) ──────────────


def _composed_runtime(tmp_path):
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-wavec-", dir=os.environ.get("NM_TEST_TMPDIR")
        or None)
    settings = load_settings(overrides={
        "home": tmp.name, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false"})
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context), tmp


def test_composed_reaction_delivery_is_not_a_no_reply(tmp_path):
    from nomorals.social.chat.base import ChatMessage, ChatRef

    rt, tmp = _composed_runtime(tmp_path)
    try:
        wp = _FakeProvider("waifu.pics", url="https://x/hug.gif")
        library = ExpressionLibrary(Path(tmp.name) / "rx.json")
        rt._expression_service_obj = ExpressionService(
            library, media_dir=Path(tmp.name) / "rx_media",
            reaction_providers=[wp],
            bytes_fetcher=lambda url: GIF_BYTES,
            health=ProviderHealth(), rng=random.Random(0))
        chat = ChatRef(platform="local", chat_id="room1",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True, text="/hug",
                          sender="Ada", sender_id="7",
                          message_id="55")
        reply = rt.handle_control("/hug", "local:room1", message=msg)
        # the GIF was the reply: "" — and crucially NOT the
        # never-empty fallback text after a successful send
        assert reply == ""
        assert library.search("hug")  # the fetch really landed
    finally:
        tmp.cleanup()


def test_composed_gif_delivery_is_not_a_no_reply(tmp_path):
    from nomorals.social.chat.base import ChatMessage, ChatRef

    rt, tmp = _composed_runtime(tmp_path)
    try:
        gif = Path(tmp.name) / "dance.gif"
        gif.write_bytes(GIF_BYTES)
        rt._expression_library().add_item(
            kind="gif", path=str(gif), tags=["dance"], source="test")
        chat = ChatRef(platform="local", chat_id="room1",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True,
                          text="/gif dance", sender="Ada",
                          sender_id="7", message_id="55")
        reply = rt.handle_control("/gif dance", "local:room1",
                                  message=msg)
        assert reply == ""
    finally:
        tmp.cleanup()
