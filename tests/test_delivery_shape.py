"""DELIVERY-SHAPE PASS (owner screenshots + orders, 2026-11-07).

The defect class: features that DESCRIBE the deliverable
instead of delivering it in the form it's used.

1. /couplepp + /pp sent ONE combined diptych captioned "the
   left half for one of you, the right half for the other".
   A couple pfp is TWO separate profile-ready pictures.
2. /husbando arrived as a raw FILE (a hash-named document)
   instead of an inline photo — the gateway's file send
   hardcoded kind "file", so EVERY picture sent through it
   landed as a document.
3. GIFs must arrive as animations, never stills or files —
   including through the gateway fallback path.
4. More sources, not anime-only: the pools diversify
   (nekosapi through its safe-rated door, with the rating
   re-checked in code; picsum general photography in the
   couplepp chain), sources remaining seeds, never a
   closed list.
5. Buttons: media that offers a follow-up choice carries
   it — Another one on the anime pictures, Another pair on
   the couplepp pair's second photo.
6. Usage strings: WhatsApp's presentation strips
   tag-shaped text, so "<title>" placeholders reached the
   owner's phone eaten.  The presentation layer now shields
   non-tag spans, and the personal/chatops/pair/anime/
   expression usage strings are written bracket-free.

Exercised, not audited: every verdict below is a recording
of what a composed handle_control (or the real gateway /
Bot API adapter) actually put on the wire.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

GIF_BYTES = b"GIF89a" + bytes(range(32))


def _png_bytes(color=(200, 40, 40), size=(120, 90)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg_bytes(color=(40, 120, 200), size=(120, 90)) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


# ══ recording adapter (telegram-bot shaped) ══════════════════


class _RecordingBotAdapter:
    """Records every media send: the ref (kind/name/path),
    the caption and any buttons offered with it."""

    name = "telegram-bot"

    def __init__(self):
        self.media: list[dict] = []
        self.owner_sender_ids = {"7"}

    def send_media(self, chat, ref, caption="", parse_mode="",
                   buttons=None):
        from nomorals.social.chat.base import SendResult

        self.media.append({"ref": ref, "caption": caption,
                           "buttons": buttons})
        return SendResult(ok=True, platform="telegram-bot",
                          message_id=str(len(self.media)))


def _composed_runtime():
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-dshape-", dir=os.environ.get("NM_TEST_TMPDIR")
        or None)
    settings = load_settings(overrides={
        "home": tmp.name, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false"})
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context), tmp


def _dm(text, sender_id="7"):
    from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

    chat = ChatRef(platform="telegram-bot", chat_id="5",
                   kind=ChatKind.DM)
    return ChatMessage(chat=chat, incoming=True, text=text,
                       sender="Ada", sender_id=sender_id,
                       message_id="55")


# ══ 1 · the gateway kind contract ════════════════════════════


def test_gateway_send_file_default_keeps_document_shape(tmp_path):
    from nomorals.social.chat.base import ChatRef
    from nomorals.social.chat.gateway import ChatGateway

    rec = _RecordingBotAdapter()
    gw = ChatGateway({"telegram-bot": rec})
    pic = tmp_path / "x.jpg"
    pic.write_bytes(_jpg_bytes())
    chat = ChatRef(platform="telegram-bot", chat_id="5")
    out = gw.send_file("telegram-bot", chat, str(pic),
                       caption="a genuine file send")
    assert out.ok
    # no kind declared → the historical document shape stands
    assert rec.media[0]["ref"].kind == "file"


def test_gateway_send_file_declared_kind_rides_the_wire(tmp_path):
    from nomorals.social.chat.base import ChatRef
    from nomorals.social.chat.gateway import ChatGateway

    rec = _RecordingBotAdapter()
    gw = ChatGateway({"telegram-bot": rec})
    pic = tmp_path / "abc123hash.jpg"
    pic.write_bytes(_jpg_bytes())
    chat = ChatRef(platform="telegram-bot", chat_id="5")
    out = gw.send_file("telegram-bot", chat, str(pic),
                       caption="a picture", kind="image",
                       name="husbando.jpg",
                       buttons=[[("🔁 Another one", "husbando")]])
    assert out.ok
    ref = rec.media[0]["ref"]
    assert ref.kind == "image"          # a photo, not a document
    assert ref.name == "husbando.jpg"   # never the hash name
    assert rec.media[0]["buttons"] == [[("🔁 Another one",
                                         "husbando")]]


def test_bot_api_method_mapping_by_kind(tmp_path):
    """The real Bot API adapter: kind decides the verb —
    image → sendPhoto, gif/animation → sendAnimation,
    sticker → sendSticker, file → sendDocument (the shape
    the defect wore)."""
    from nomorals.social.chat.base import ChatRef, MediaRef
    from nomorals.social.chat.telegram import TelegramBotAdapter

    adapter = TelegramBotAdapter(
        token="123:fake", session=SimpleNamespace(),
        owner_sender_ids={"7"})
    methods = []

    def fake_upload(method, fields, file_field, path, **kw):
        methods.append(method)
        return {"message_id": 1}

    adapter._api_upload = fake_upload
    chat = ChatRef(platform="telegram-bot", chat_id="5")
    path = tmp_path / "m.bin"
    path.write_bytes(_jpg_bytes())
    for kind in ("image", "gif", "animation", "sticker", "file"):
        out = adapter.send_media(chat, MediaRef(
            path=str(path), kind=kind, name="m"))
        assert out.ok, kind
    assert methods == ["sendPhoto", "sendAnimation",
                       "sendAnimation", "sendSticker",
                       "sendDocument"]


# ══ 2 · the anime picture family, composed ═══════════════════


class _PoolTransport:
    """The pool APIs' real shapes + the image bytes behind
    them, over the anime transport protocol."""

    def __init__(self, img: bytes):
        self.img = img

    def request(self, method, url, **kw):
        from nomorals.anime.transport import HttpResult

        if "img.test" in url:
            return HttpResult(200, content=self.img)
        if "api.waifu.pics" in url:
            return HttpResult(
                200, data={"url": "https://img.test/waifu.jpg"})
        if "nekos.best" in url:
            return HttpResult(200, data={"results": [
                {"url": "https://img.test/neko.png"}]})
        if "nekosapi" in url:
            return HttpResult(200, data=[
                {"url": "https://img.test/api.webp",
                 "rating": "safe"}])
        return HttpResult(404)


def _anime_runtime(tmp_path_home=None):
    from nomorals.anime import (
        AniListClient,
        ImagePoolClient,
        JikanClient,
        QuoteClient,
        TraceMoeClient,
    )
    from nomorals.anime.favorites import FavoritesStore
    from nomorals.anime.service import AnimeService

    rt, tmp = _composed_runtime()
    transport = _PoolTransport(_jpg_bytes())
    rt._anime_service_obj = AnimeService(
        anilist=AniListClient(transport),
        jikan=JikanClient(transport),
        quotes=QuoteClient(transport),
        images=ImagePoolClient(transport),
        trace=TraceMoeClient(transport),
        favorites=FavoritesStore(None))
    extras = getattr(rt.context, "extras", None)
    if isinstance(extras, dict):
        extras["anime_transport"] = transport
    rec = _RecordingBotAdapter()
    rt.gateway.adapters["telegram-bot"] = rec
    return rt, tmp, rec


@pytest.mark.parametrize("kind", ["waifu", "neko", "kitsune",
                                  "husbando", "charpic"])
def test_picture_kinds_deliver_inline_photos(kind):
    rt, tmp, rec = _anime_runtime()
    try:
        reply = rt.handle_control(
            f"/{kind}", "telegram-bot:5", message=_dm(f"/{kind}"))
        assert "sent it here" in str(
            getattr(reply, "text", reply))
        assert len(rec.media) == 1
        ref = rec.media[0]["ref"]
        # THE defect: this used to be kind "file" (a document)
        # wearing a content-hash name
        assert ref.kind == "image"
        assert kind in ref.name
        assert ref.name.endswith((".jpg", ".png"))
        # the follow-up choice rides the photo
        assert rec.media[0]["buttons"] == [
            [("🔁 Another one", kind)]]
    finally:
        tmp.cleanup()


def test_anime_usage_line_is_bracket_free():
    rt, tmp, rec = _anime_runtime()
    try:
        reply = rt.handle_control(
            "/anime", "telegram-bot:5", message=_dm("/anime"))
        text = str(getattr(reply, "text", reply))
        assert text == "usage: /anime title — e.g. /anime frieren"
        assert "<" not in text and ">" not in text
    finally:
        tmp.cleanup()


# ══ 3 · couplepp: two photos, both rungs ═════════════════════


class _PairPool:
    name = "fakepool"

    def image_url(self):
        return "https://x.test/pair.png"


class _DeadPool:
    name = "deadpool"

    def image_url(self):
        raise OSError("pool down")


def _pair_png() -> bytes:
    """A two-tone artwork: the halves genuinely differ."""
    import io

    img = Image.new("RGB", (800, 600), (230, 220, 200))
    for x in range(400):
        for y in range(600):
            img.putpixel((x, y), (30, 30, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _makers_runtime(chain):
    from nomorals.social.chat.makers import MakerService

    rt, tmp = _composed_runtime()
    rt._makers_service_obj = MakerService(
        media_dir=Path(tmp.name) / "mk", chains={"couplepp": chain},
        bytes_fetcher=lambda url: _pair_png())
    rec = _RecordingBotAdapter()
    rt.gateway.adapters["telegram-bot"] = rec
    return rt, tmp, rec


@pytest.mark.parametrize("command", ["couplepp", "pp"])
def test_couplepp_delivers_two_separate_photos(command):
    rt, tmp, rec = _makers_runtime([_PairPool()])
    try:
        reply = rt.handle_control(
            f"/{command}", "telegram-bot:5",
            message=_dm(f"/{command}"))
        assert str(getattr(reply, "text", reply) or "") == ""
        # TWO sends — the pair as two pictures, not one poster
        assert len(rec.media) == 2
        first, second = rec.media
        for send in (first, second):
            assert send["ref"].kind == "image"
            with Image.open(send["ref"].path) as img:
                assert img.size == (480, 640)  # profile-ready
        assert first["ref"].path != second["ref"].path
        assert "left half" in first["caption"]
        assert "right half" in second["caption"]
        # the follow-up choice rides the second photo
        assert second["buttons"] == [
            [("🔁 Another pair", "couplepp")]]
    finally:
        tmp.cleanup()


def test_couplepp_self_render_rung_also_splits():
    rt, tmp, rec = _makers_runtime([_DeadPool()])
    try:
        reply = rt.handle_control(
            "/couplepp", "telegram-bot:5", message=_dm("/couplepp"))
        assert str(getattr(reply, "text", reply) or "") == ""
        assert len(rec.media) == 2
        assert "my own artwork" in rec.media[0]["caption"]
        for send in rec.media:
            with Image.open(send["ref"].path) as img:
                assert img.size == (480, 640)
    finally:
        tmp.cleanup()


def test_split_diptych_halves_are_the_pair():
    import io

    from nomorals.social.chat.makers import make_diptych, split_diptych

    img = Image.new("RGB", (800, 600), "white")
    for x in range(400):
        for y in range(600):
            img.putpixel((x, y), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    left, right, info = split_diptych(buf.getvalue())
    with Image.open(io.BytesIO(left)) as li, \
            Image.open(io.BytesIO(right)) as ri:
        assert li.size == ri.size == (480, 640)
        # the left crop is the dark side, the right the light
        assert li.convert("L").getpixel((240, 320)) < 60
        assert ri.convert("L").getpixel((240, 320)) > 190
    assert info["half"] == (480, 640)
    # the composed preview shape still exists for its callers
    png, dinfo = make_diptych(buf.getvalue())
    with Image.open(io.BytesIO(png)) as di:
        assert di.size == (968, 640)
    assert dinfo["half"] == (480, 640)


def test_couplepp_half_landed_pair_is_named_incomplete(tmp_path):
    from nomorals.social.chat.makers import MakerService

    svc = MakerService(
        media_dir=tmp_path / "mk",
        chains={"couplepp": [_PairPool()]},
        bytes_fetcher=lambda url: _png_bytes(size=(800, 600)))
    calls = []

    def flaky_send(ref, caption):
        calls.append(ref)
        return SimpleNamespace(ok=len(calls) == 1, message_id="1",
                               media_file_id="", error="boom")

    out = svc.couplepp(send=flaky_send)
    assert out["sent"] is False
    assert out["parts"] == 1
    assert "incomplete" in out["reason"]


# ══ 4 · GIFs land as animations — direct AND fallback ════════


class _FakeReactionProvider:
    def __init__(self, name):
        self.name = name

    def fetch(self, category):
        return {"url": f"https://x.test/{category}.gif",
                "provider": self.name}


def _expression_rig(rt, tmp):
    from nomorals.social.chat.expression import ExpressionLibrary, ExpressionService, ProviderHealth

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
    return library


def test_reaction_and_gif_deliver_as_animations():
    rt, tmp = _composed_runtime()
    try:
        _expression_rig(rt, tmp)
        rec = _RecordingBotAdapter()
        rt.gateway.adapters["telegram-bot"] = rec
        reply = rt.handle_control(
            "/hug", "telegram-bot:5", message=_dm("/hug"))
        assert str(getattr(reply, "text", reply) or "") == ""
        reply = rt.handle_control(
            "/gif dance", "telegram-bot:5", message=_dm("/gif dance"))
        assert str(getattr(reply, "text", reply) or "") == ""
        kinds = [s["ref"].kind for s in rec.media]
        assert kinds == ["animation", "animation"]
    finally:
        tmp.cleanup()


class _BareGateway:
    """A gateway double with ONLY send_file — the shape that
    forces _expression_send onto its fallback path."""

    def __init__(self):
        self.file_calls = []

    def send_file(self, platform, chat, path, **kwargs):
        self.file_calls.append(
            {"platform": platform, "path": path, **kwargs})
        return SimpleNamespace(ok=True, message_id="1",
                               media_file_id="", error="")


def _fallback_runtime(tmp_path, gateway):
    from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin):
        def __init__(self):
            self.gateway = gateway
            self.operators = {"7"}
            self.context = SimpleNamespace(
                settings=SimpleNamespace(
                    workspace_dir=str(tmp_path)))

        def _group_adapter(self, platform):
            return None  # no adapter → the gateway fallback

        def _is_operator(self, message):
            return True

    return _Runtime()


def test_expression_fallback_declares_the_refs_kind(tmp_path):
    gateway = _BareGateway()
    rt = _fallback_runtime(tmp_path, gateway)
    from nomorals.social.chat.expression import ExpressionLibrary, ExpressionService, ProviderHealth

    library = ExpressionLibrary(tmp_path / "rx.json")
    dance = tmp_path / "dance.gif"
    dance.write_bytes(GIF_BYTES)
    library.add_item(kind="gif", path=str(dance), tags=["dance"],
                     source="test")
    rt._expression_service_obj = ExpressionService(
        library, media_dir=tmp_path / "rx_media", providers=[],
        reaction_providers=[], health=ProviderHealth(),
        rng=random.Random(0))
    chat = SimpleNamespace(platform="telegram-bot", key="telegram-bot:5")
    msg = SimpleNamespace(chat=chat, sender_id="7", sender="Ada",
                          message_id="55", reply_to="", meta={},
                          media=[], text="/gif dance")
    out = rt._control_gif("dance", message=msg)
    assert out == ""
    assert len(gateway.file_calls) == 1
    call = gateway.file_calls[0]
    # through the fallback the GIF still lands AS an animation
    # with its friendly name — not kind "file", not a hash name
    assert call["kind"] == "animation"
    assert call["name"] == "expression gif"


# ══ 5 · fun one-shots ride the photo path ════════════════════


def test_dog_delivers_a_photo():
    from nomorals.social.chat.fun import DogCeoProvider, FunService

    rt, tmp = _composed_runtime()
    try:
        rt._fun_service_obj = FunService(
            media_dir=Path(tmp.name) / "fun",
            chains={"dog": [DogCeoProvider(fetcher=lambda url: {
                "status": "success",
                "message": "https://img.test/dog.jpg"})]},
            bytes_fetcher=lambda url: _jpg_bytes())
        rec = _RecordingBotAdapter()
        rt.gateway.adapters["telegram-bot"] = rec
        reply = rt.handle_control(
            "/dog", "telegram-bot:5", message=_dm("/dog"))
        assert str(getattr(reply, "text", reply) or "") == ""
        assert len(rec.media) == 1
        assert rec.media[0]["ref"].kind == "image"
        assert "dog.ceo" in rec.media[0]["caption"]
    finally:
        tmp.cleanup()


# ══ 6 · source diversification (verified shapes only) ════════


class _RoutingTransport:
    """Routes the pool JSON endpoints; selected hosts fail."""

    def __init__(self, data_by_host):
        self.data_by_host = data_by_host

    def request(self, method, url, **kw):
        from nomorals.anime.transport import HttpResult

        for host, data in self.data_by_host.items():
            if host in url:
                if data is None:
                    return HttpResult(503)
                return HttpResult(200, data=data)
        return HttpResult(404)


def test_nekosapi_pool_row_must_be_safe_rated():
    from nomorals.anime import ImagePoolClient

    transport = _RoutingTransport({
        "api.waifu.pics": None,
        "nekos.best": None,
        "nekosapi": [
            {"url": "https://img.test/explicit.webp",
             "rating": "explicit"},
            {"url": "https://img.test/suggestive.webp",
             "rating": "suggestive"},
            {"url": "https://img.test/safe.webp",
             "rating": "safe"},
        ],
    })
    client = ImagePoolClient(transport)
    url, source = client.fetch_url("waifu", rng=random.Random(0))
    assert source == "nekosapi"
    assert url == "https://img.test/safe.webp"


def test_nekosapi_pool_with_no_safe_row_fails_honestly():
    from nomorals.anime import ImagePoolClient
    from nomorals.anime.sources import SourceFailure

    transport = _RoutingTransport({
        "api.waifu.pics": None,
        "nekos.best": None,
        "nekosapi": [
            {"url": "https://img.test/x.webp", "rating": "explicit"},
        ],
    })
    client = ImagePoolClient(transport)
    with pytest.raises(SourceFailure):
        client.fetch_url("waifu", rng=random.Random(0))


def test_couplepp_chain_is_diverse_and_not_anime_only():
    from nomorals.social.chat.makers import (
        NekosApiPoolProvider,
        PicsumPoolProvider,
        default_maker_chains,
    )

    names = [p.name for p in default_maker_chains()["couplepp"]]
    assert "picsum" in names      # general photography
    assert "nekosapi" in names    # a third anime pool
    assert "waifu.pics" in names and "nekos.best" in names
    picsum = PicsumPoolProvider(rng=random.Random(1))
    assert picsum.image_url().startswith(
        "https://picsum.photos/seed/")
    nekosapi = NekosApiPoolProvider(fetcher=lambda url: [
        {"url": "https://img.test/no.webp", "rating": "suggestive"},
        {"url": "https://img.test/yes.webp", "rating": "safe"}])
    assert nekosapi.image_url() == "https://img.test/yes.webp"


# ══ 7 · downloads land in their consumable kind ══════════════


class _KindGateway:
    def __init__(self):
        self.calls = []

    def send_file(self, platform, chat, path, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(ok=True)


def test_platform_file_send_declares_its_kind(tmp_path):
    rt, tmp = _composed_runtime()
    try:
        gateway = _KindGateway()
        rt.gateway = gateway
        chat = SimpleNamespace(platform="telegram-bot",
                               chat_id="5")
        for name, want in (("clip.mp4", "video"),
                           ("song.mp3", "audio"),
                           ("cover.png", "image"),
                           ("bundle.zip", "")):
            gateway.calls.clear()
            path = tmp_path / name
            path.write_bytes(b"x")
            _note, sent = rt._send_platform_file(
                chat, str(path), "cap", 10)
            assert sent is True
            assert gateway.calls[0].get("kind") == want, name
    finally:
        tmp.cleanup()


# ══ 8 · usage strings survive WhatsApp presentation ══════════


def test_whatsapp_presentation_keeps_placeholders():
    from nomorals.social.chat.platforms import to_whatsapp

    line = "usage: /anime <title> — e.g. /anime frieren"
    assert to_whatsapp(line) == line
    assert to_whatsapp(
        "usage: /disappear <off|24h|7d|90d> [chat name]") == \
        "usage: /disappear <off|24h|7d|90d> [chat name]"
    assert to_whatsapp("usage: /gif <what you're feeling>") == \
        "usage: /gif <what you're feeling>"
    # real markup still converts/strips exactly as before
    assert to_whatsapp("<b>bold</b>") == "*bold*"
    assert to_whatsapp("<i>x</i> and <u>gone</u>") == "_x_ and gone"
    assert to_whatsapp("2 < 3 and 5 > 4") == "2 < 3 and 5 > 4"


def test_usage_families_are_bracket_free():
    from nomorals.agents.partner.runtime_anime import _USAGE

    for line in _USAGE.values():
        assert "<" not in line and ">" not in line

    rt, tmp = _composed_runtime()
    try:
        # /gif with no chat: the usage line itself
        assert rt._control_gif("", message=None) == \
            "usage: /gif what you're feeling"
        # /sticker with no chat
        assert rt._control_sticker("", message=None) == \
            "usage: /sticker text — or reply to an image/video."
        # /disappear with no duration, through the chatops door
        out = rt._control_chatops("disappear", "",
                                  message=_dm("/disappear"))
        assert out.startswith("usage: /disappear off, 24h, 7d")
        assert "<" not in out
    finally:
        tmp.cleanup()
