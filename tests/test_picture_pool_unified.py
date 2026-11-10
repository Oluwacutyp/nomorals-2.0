"""Unified picture pool + kind-true delivery (2026-11-10).

The picture-command family (/waifu /neko /kitsune /husbando
/charpic /couplepp /pp + the fun image one-shots) used to draw
from anime-only pools behind a stills-only send: three anime
sources for the pool kinds, nekos.best alone for two of them,
a duplicate endpoint implementation inside makers for
/couplepp, and a suffix derived from the URL so a GIF would
have been written out as a .jpg photo.  This suite pins the
fix: ONE pool organ (nomorals.anime.sources) whose table
spans source classes, a weighted walk with her own studio as
a rung and the floor, and delivery that sniffs the BYTES —
animated payloads ride as animations, stills as photos —
through every surface of the family.  Network-free: every
transport/fetcher is a fixture with the real APIs' shapes.
"""

from __future__ import annotations

import io
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from nomorals.anime.sources import (
    IMAGE_POOLS,
    ImagePoolClient,
    SourceFailure,
    pool_classes,
)
from nomorals.anime.transport import HttpResult, TransportError
from nomorals.social.chat.fun import (
    CataasGifProvider,
    FunService,
    picture_shape,
)
from nomorals.social.chat.makers import (
    MakerService,
    NekosPoolProvider,
    WaifuPoolProvider,
)

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
GIF = b"GIF89a" + b"\x01\x00\x01\x00" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


def _webp(anim_flag: bool, vp8x: bool = True) -> bytes:
    if vp8x:
        return (b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"VP8X"
                + b"\x0a\x00\x00\x00"
                + bytes([0x02 if anim_flag else 0x00])
                + b"\x00" * 16)
    return b"RIFF" + b"\x1e\x00\x00\x00" + b"WEBP" + b"VP8 " \
        + b"\x00" * 16


def _art_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), (90, 60, 160)).save(
        buf, format="PNG")
    return buf.getvalue()


ART_PNG = _art_png()


# ── the kind truth: picture_shape ────────────────────────────

def test_picture_shape_reads_the_bytes_not_the_claim():
    assert picture_shape(JPEG) == (".jpg", "image/jpeg", "image")
    assert picture_shape(PNG) == (".png", "image/png", "image")
    assert picture_shape(GIF) == (".gif", "image/gif", "animation")
    assert picture_shape(MP4) == (".mp4", "video/mp4", "animation")
    assert picture_shape(_webp(True)) == (
        ".webp", "image/webp", "animation")
    # a still WebP is a still — VP8X without the ANIM flag,
    # and a VP8 WebP with no VP8X chunk at all
    assert picture_shape(_webp(False)) == (
        ".webp", "image/webp", "image")
    assert picture_shape(_webp(False, vp8x=False)) == (
        ".webp", "image/webp", "image")
    assert picture_shape(b"<html>not a picture</html>") is None
    assert picture_shape(b"") is None


# ── a fixture transport over the real APIs' shapes ──────────

class _Transport:
    """Routes by host substring.  A route value is an
    HttpResult, an HTTP status int, or "down" (TransportError).
    Hosts with no route answer 404."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def request(self, method: str, url: str, **kw: Any) -> HttpResult:
        self.calls.append(url)
        for host, resp in self.routes.items():
            if host in url:
                if resp == "down":
                    raise TransportError(f"{host} unreachable")
                if isinstance(resp, int):
                    return HttpResult(resp, url=url)
                if isinstance(resp, HttpResult) and not resp.url:
                    resp.url = url
                return resp
        return HttpResult(404, url=url)


def _json(data: Any) -> HttpResult:
    return HttpResult(200, data=data)


def _all_up_transport() -> _Transport:
    return _Transport({
        "api.waifu.pics": _json({"url": "https://i.waifu.pics/a.jpg"}),
        "nekos.best": _json({"results": [
            {"url": "https://nekos.best/b.png"}]}),
        "nekosapi": _json([{"url": "https://img.nekosapi/c.webp",
                            "rating": "safe"}]),
        "picsum.photos": HttpResult(
            200, content=JPEG, url="https://fastly.picsum/d.jpg"),
        "cataas.com/cat/gif": _json(
            {"url": "https://cataas.com/cat/gif1?position=center",
             "mimetype": "image/gif"}),
        "cataas.com": _json({"url": "https://cataas.com/cat/still1"}),
        "randomfox.ca": _json({"image": "https://randomfox.ca/f.jpg"}),
    })


class _ScriptedRng:
    """An rng whose random() replays a script (randrange and
    choice stay honest over the values they are handed)."""

    def __init__(self, script: list[float]) -> None:
        self._script = list(script)

    def random(self) -> float:
        return self._script.pop(0) if self._script else 0.5

    def randrange(self, *args: Any) -> int:
        return 424242

    def choice(self, seq: Any) -> Any:
        return seq[0]

    def shuffle(self, seq: Any) -> None:
        pass


# ── diversity: the table spans classes, anime never alone ───

POOL_KINDS = ("waifu", "neko", "kitsune", "husbando", "charpic",
              "couplepp")


def test_every_pool_kind_spans_at_least_three_classes():
    for kind in POOL_KINDS:
        classes = pool_classes(kind)
        assert len(classes) >= 3, (kind, classes)
        assert "anime" in classes
        assert classes - {"anime"}, kind


def test_anime_is_the_home_class_by_weight():
    # the kind's anime entries carry the top weight in its
    # chain — anime-first by construction, not by walk luck
    for kind in POOL_KINDS:
        entries = IMAGE_POOLS[kind]
        anime_weights = [e.weight for e in entries
                         if e.source_class == "anime"]
        other_weights = [e.weight for e in entries
                         if e.source_class != "anime"]
        assert anime_weights and max(anime_weights) > max(
            other_weights), kind


def test_no_keys_richness_with_and_without_her_studio():
    bare = ImagePoolClient(_all_up_transport())
    assert bare.available_classes("waifu") == {
        "anime", "photography"}
    assert bare.available_classes("neko") == {"anime", "animals"}
    assert bare.available_classes("kitsune") == {
        "anime", "animals"}
    wired = ImagePoolClient(
        _all_up_transport(), generator=lambda kind: "/x.gif")
    for kind in POOL_KINDS:
        classes = wired.available_classes(kind)
        assert len(classes) >= 3, (kind, classes)
        assert "studio" in classes
    keyed = ImagePoolClient(_all_up_transport(), tenor_key="k")
    assert "general" in keyed.available_classes("waifu")
    assert "general" not in bare.available_classes("waifu")


def test_rotation_varies_and_anime_stays_plurality():
    client = ImagePoolClient(
        _all_up_transport(), generator=lambda kind: "/x.gif")
    rng = random.Random(20261110)
    hits = [client.fetch_media("waifu", rng=rng) for _ in range(80)]
    classes = [h.source_class for h in hits]
    assert len(set(classes)) >= 3, set(classes)
    assert classes.count("anime") > len(classes) // 2
    assert "photography" in classes
    assert "studio" in classes


# ── the walk: fallthrough in order, studio floor, keyed ─────

def _one_source_pool(entries: list) -> dict[str, list]:
    return {"test": entries}


def test_walk_falls_through_in_order():
    from nomorals.anime.sources import _PoolSource

    entries = [
        _PoolSource("first-down", "https://down.test/x",
                    "waifu.pics"),
        _PoolSource("second-shapeless", "https://empty.test/x",
                    "waifu.pics"),
        _PoolSource("third-good", "https://good.test/x",
                    "waifu.pics"),
    ]
    transport = _Transport({
        "down.test": "down",
        "empty.test": _json({"nope": True}),
        "good.test": _json({"url": "https://good.test/pic.jpg"}),
    })
    client = ImagePoolClient(transport, pools=_one_source_pool(entries))
    hit = client.fetch_media(
        "test", rng=_ScriptedRng([0.95, 0.6, 0.2]))
    assert hit.url == "https://good.test/pic.jpg"
    assert hit.source == "third-good"
    assert transport.calls == [
        "https://down.test/x", "https://empty.test/x",
        "https://good.test/x"]


def test_all_sources_down_is_a_named_failure():
    from nomorals.anime.sources import _PoolSource

    entries = [_PoolSource("only", "https://down.test/x",
                           "waifu.pics")]
    client = ImagePoolClient(
        _Transport({"down.test": "down"}),
        pools=_one_source_pool(entries))
    with pytest.raises(SourceFailure) as err:
        client.fetch_media("test", rng=_ScriptedRng([0.5]))
    assert "only" in str(err.value)


def test_studio_is_the_floor_when_the_network_fails():
    from nomorals.anime.sources import _PoolSource

    entries = [
        _PoolSource("net", "https://down.test/x", "waifu.pics"),
        _PoolSource("devon-studio", "", "studio",
                    source_class="studio"),
    ]
    calls: list[str] = []

    def generate(kind: str) -> str:
        calls.append(f"generate:{kind}")
        return "/renders/her.gif"

    transport = _Transport({"down.test": "down"})
    client = ImagePoolClient(
        transport, pools=_one_source_pool(entries),
        generator=generate)
    # order draw 0.5 for the one network entry; share draw
    # 0.99 > the studio's share — the network walks first,
    # the studio floors
    hit = client.fetch_media(
        "test", rng=_ScriptedRng([0.5, 0.99]))
    assert hit.local_path == "/renders/her.gif"
    assert hit.source_class == "studio"
    assert transport.calls == ["https://down.test/x"]
    assert calls == ["generate:test"]


def test_studio_takes_its_rotation_share():
    from nomorals.anime.sources import _PoolSource

    entries = [
        _PoolSource("net", "https://good.test/x", "waifu.pics"),
        _PoolSource("devon-studio", "", "studio",
                    source_class="studio"),
    ]
    transport = _Transport({
        "good.test": _json({"url": "https://good.test/p.jpg"})})
    client = ImagePoolClient(
        transport, pools=_one_source_pool(entries),
        generator=lambda kind: "/renders/her.gif")
    # share draw 0.1 < the studio's 1/2 share — her hand
    # takes this draw and the network is never asked
    hit = client.fetch_media(
        "test", rng=_ScriptedRng([0.5, 0.1]))
    assert hit.local_path == "/renders/her.gif"
    assert transport.calls == []


def test_studio_without_a_generator_never_enters():
    from nomorals.anime.sources import _PoolSource

    entries = [
        _PoolSource("net", "https://down.test/x", "waifu.pics"),
        _PoolSource("devon-studio", "", "studio",
                    source_class="studio"),
    ]
    client = ImagePoolClient(
        _Transport({"down.test": "down"}),
        pools=_one_source_pool(entries))
    with pytest.raises(SourceFailure):
        client.fetch_media("test", rng=_ScriptedRng([0.5]))
    # and the URL-only walk sits the studio out even with a
    # generator wired, while fetch_media serves it
    wired = ImagePoolClient(
        _Transport({"down.test": "down"}),
        pools=_one_source_pool(entries),
        generator=lambda kind: "/renders/her.gif")
    with pytest.raises(SourceFailure):
        wired.fetch_url("test", rng=_ScriptedRng([0.5]))
    hit = wired.fetch_media("test", rng=_ScriptedRng([0.5, 0.99]))
    assert hit.local_path == "/renders/her.gif"


def test_keyed_rungs_enter_only_with_their_key():
    from nomorals.anime.sources import _PoolSource

    entries = [
        _PoolSource("tenor", "https://tenor.test/search", "tenor",
                    source_class="general", animated=True,
                    key="tenor", query="cat"),
    ]
    transport = _Transport({
        "tenor.test": _json({"results": [
            {"media_formats": {"gif": {"url": "https://t/x.gif"}}},
        ]}),
    })
    bare = ImagePoolClient(transport, pools=_one_source_pool(entries))
    with pytest.raises(SourceFailure):
        bare.fetch_media("test", rng=_ScriptedRng([0.5]))
    assert transport.calls == []
    keyed = ImagePoolClient(
        transport, pools=_one_source_pool(entries),
        tenor_key="secret")
    hit = keyed.fetch_media("test", rng=_ScriptedRng([0.5]))
    assert hit.url == "https://t/x.gif"
    assert hit.source == "tenor"
    assert hit.source_class == "general"


def test_picsum_draw_is_proven_live_before_it_is_handed_out():
    client = ImagePoolClient(_Transport({
        "picsum.photos": HttpResult(
            200, content=JPEG,
            url="https://fastly.picsum.photos/id/10/1200/900.jpg"),
    }))
    hit = client.fetch_media("waifu", rng=random.Random(3))
    # every other waifu source 404s in this rig, so the walk
    # must land on picsum — with the post-redirect URL
    assert hit.source == "picsum"
    assert hit.url.endswith("1200/900.jpg")
    assert hit.source_class == "photography"


# ── service level: the studio outcome shape ──────────────────

def test_service_picture_names_her_studio_draw():
    from nomorals.anime.favorites import FavoritesStore
    from nomorals.anime.service import AnimeService
    from nomorals.anime.sources import AniListClient, JikanClient, QuoteClient, TraceMoeClient
    from nomorals.storage.db import Database

    transport = _Transport({})  # everything 404s
    svc = AnimeService(
        anilist=AniListClient(transport), jikan=JikanClient(transport),
        quotes=QuoteClient(transport),
        images=ImagePoolClient(
            transport, generator=lambda kind: "/renders/her.gif"),
        trace=TraceMoeClient(transport),
        favorites=FavoritesStore(Database(":memory:")),
        rng=random.Random(5))
    out = svc.picture("husbando")
    assert out.ok
    assert out.image_path == "/renders/her.gif"
    assert "studio" in out.image_caption


# ── runtime level: kind preservation end to end ──────────────

from nomorals.agents.context import build_context  # noqa: E402
from nomorals.agents.partner_runtime import PartnerRuntime  # noqa: E402
from nomorals.core.config import load_settings  # noqa: E402
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef  # noqa: E402

OWNER_ID = "5478650254"


class _KindGateway:
    def __init__(self) -> None:
        self.files: list[dict[str, Any]] = []

    def send_file(self, platform: str, chat: Any, path: str,
                  **kwargs: Any) -> Any:
        self.files.append({"platform": platform, "chat": chat,
                           "path": path, **kwargs})
        return SimpleNamespace(ok=True)


def _runtime(transport: _Transport):
    scratch = Path.home() / "workspace" / ".test-tmp-pspics"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-pspics-",
                                      dir=str(scratch))
    settings = load_settings(overrides={
        "home": tmp.name,
        "partner.platforms": "telegram",
        "partner.owner_chats": OWNER_ID,
        "partner.owner_sender_ids": OWNER_ID,
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    context.extras["anime_transport"] = transport
    gateway = _KindGateway()
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, gateway, tmp


def _msg(chat_key: str) -> ChatMessage:
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform or "telegram",
                     chat_id=chat_id or "1", kind=ChatKind.DM),
        incoming=True, text="", sender="owner",
        sender_id=OWNER_ID, meta={"is_owner": True}, media=[])


def _text(reply: Any) -> str:
    return str(getattr(reply, "text", reply) or "")


def test_runtime_waifu_gif_lands_as_an_animation():
    transport = _Transport({
        "api.waifu.pics": _json(
            {"url": "https://i.waifu.pics/dance.gif"}),
        "i.waifu.pics": HttpResult(200, content=GIF),
        "nekos.best": 500,
    })
    rt, gw, tmp = _runtime(transport)
    try:
        reply = _text(rt.handle_control(
            "/waifu", "telegram:1", message=_msg("telegram:1")))
        assert "sent it here" in reply
        assert len(gw.files) == 1
        sent = gw.files[0]
        assert sent["kind"] == "animation"
        assert sent["name"].endswith(".gif")
        assert Path(sent["path"]).read_bytes() == GIF
    finally:
        tmp.cleanup()


def test_runtime_waifu_still_lands_as_a_photo():
    transport = _Transport({
        "api.waifu.pics": _json(
            {"url": "https://i.waifu.pics/portrait.jpg"}),
        "i.waifu.pics": HttpResult(200, content=JPEG),
        "nekos.best": 500,
    })
    rt, gw, tmp = _runtime(transport)
    try:
        reply = _text(rt.handle_control(
            "/waifu", "telegram:1", message=_msg("telegram:1")))
        assert "sent it here" in reply
        sent = gw.files[0]
        assert sent["kind"] == "image"
        assert sent["name"].endswith(".jpg")
    finally:
        tmp.cleanup()


def test_runtime_waifu_all_network_down_her_studio_answers():
    transport = _Transport({})  # every source 404s / unreachable
    rt, gw, tmp = _runtime(transport)
    try:
        reply = _text(rt.handle_control(
            "/waifu", "telegram:1", message=_msg("telegram:1")))
        assert "sent it here" in reply
        assert len(gw.files) == 1
        sent = gw.files[0]
        # her studio draws an animated GIF — it lands as one,
        # captioned as hers
        assert sent["kind"] == "animation"
        assert sent["name"].endswith(".gif")
        assert Path(sent["path"]).read_bytes()[:6] == b"GIF89a"
        assert "studio" in str(sent.get("caption") or "")
    finally:
        tmp.cleanup()


# ── fun one-shots: kind truth + the cat GIF rung ────────────

class _FakeImageProvider:
    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self._url = url

    def image(self, arg: str = "") -> dict[str, Any]:
        return {"image_url": self._url,
                "caption": f"via {self.name}"}


class _Send:
    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, ref: Any, caption: str) -> Any:
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=True)


def test_fun_image_gif_bytes_ride_as_animation(tmp_path):
    svc = FunService(
        media_dir=tmp_path / "fun_media",
        chains={"meme": [_FakeImageProvider("meme-api",
                                            "https://x/m.gif")]},
        bytes_fetcher=lambda url: GIF)
    send = _Send()
    out = svc.image("meme", "", send=send)
    assert out["sent"] is True
    ref, _caption = send.calls[0]
    assert ref.kind == "animation"
    assert ref.path.endswith(".gif")


def test_fun_image_jpeg_bytes_ride_as_photo(tmp_path):
    svc = FunService(
        media_dir=tmp_path / "fun_media",
        chains={"dog": [_FakeImageProvider("dog.ceo",
                                           "https://x/d.jpg")]},
        bytes_fetcher=lambda url: JPEG)
    send = _Send()
    out = svc.image("dog", "", send=send)
    assert out["sent"] is True
    ref, _caption = send.calls[0]
    assert ref.kind == "image"
    assert ref.path.endswith(".jpg")


def test_fun_cat_chain_rotates_across_its_sources(tmp_path):
    providers = [
        _FakeImageProvider("thecatapi", "https://x/1.jpg"),
        _FakeImageProvider("cataas", "https://x/2.jpg"),
        _FakeImageProvider("cataas-gif", "https://x/3.gif"),
    ]
    svc = FunService(
        media_dir=tmp_path / "fun_media",
        chains={"cat": providers},
        bytes_fetcher=lambda url: GIF if url.endswith(".gif")
        else JPEG)
    seen = []
    kinds = []
    for _ in range(3):
        send = _Send()
        out = svc.image("cat", "", send=send)
        assert out["sent"] is True
        seen.append(out["source"])
        kinds.append(send.calls[0][0].kind)
    assert seen == ["thecatapi", "cataas", "cataas-gif"]
    assert kinds == ["image", "image", "animation"]


def test_cataas_gif_provider_shape():
    provider = CataasGifProvider(
        fetcher=lambda url: {"url": "/cat/abc?position=center",
                             "mimetype": "image/gif"})
    out = provider.image()
    assert out["image_url"] == \
        "https://cataas.com/cat/abc?position=center"


def test_default_cat_chain_carries_the_gif_rung():
    from nomorals.social.chat.fun import default_fun_chains

    names = [p.name for p in default_fun_chains()["cat"]]
    assert names == ["thecatapi", "cataas", "cataas-gif"]


# ── couplepp: unified pool underneath, pair law on top ──────

class _SendPair:
    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, ref: Any, caption: str, buttons=None) -> Any:
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=True)


def test_couplepp_default_chain_delegates_to_the_pool():
    from nomorals.social.chat.makers import default_maker_chains

    names = [p.name for p in default_maker_chains()["couplepp"]]
    assert names == ["waifu.pics", "nekos.best", "nekosapi",
                     "picsum"]


def test_couplepp_pair_lands_from_the_unified_pool(tmp_path):
    transport = _all_up_transport()
    svc = MakerService(
        media_dir=tmp_path / "maker_media",
        chains={"couplepp": [
            WaifuPoolProvider(transport=transport),
            NekosPoolProvider(transport=transport)]},
        bytes_fetcher=lambda url: ART_PNG)
    send = _SendPair()
    out = svc.couplepp(send=send)
    assert out["sent"] is True and out["parts"] == 2
    assert out["source"] in ("waifu.pics", "nekos.best")
    assert len(send.calls) == 2
    for ref, _caption in send.calls:
        assert ref.kind == "image"
        assert Image.open(ref.path).size == (480, 640)


def test_couplepp_redraws_past_an_animated_draw(tmp_path):
    class _FlipBytes:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, url: str) -> bytes:
            self.n += 1
            return GIF if self.n == 1 else ART_PNG

    class _Poolish:
        name = "waifu.pics"

        def image_url(self) -> str:
            return "https://x/p"

    flips = _FlipBytes()
    svc = MakerService(
        media_dir=tmp_path / "maker_media",
        chains={"couplepp": [_Poolish()]},
        bytes_fetcher=flips)
    send = _SendPair()
    out = svc.couplepp(send=send)
    assert out["sent"] is True and out["parts"] == 2
    assert out["source"] == "waifu.pics"
    # the first (animated) draw was re-drawn, not flattened
    assert flips.n == 2


def test_couplepp_all_animated_draws_fall_to_her_studio(tmp_path):
    class _Poolish:
        name = "waifu.pics"

        def image_url(self) -> str:
            return "https://x/anim.gif"

    svc = MakerService(
        media_dir=tmp_path / "maker_media",
        chains={"couplepp": [_Poolish()]},
        bytes_fetcher=lambda url: GIF)
    send = _SendPair()
    out = svc.couplepp(send=send)
    # every pool draw animated → her studio floor draws the
    # pair (the pair law holds: two separate stills)
    assert out["sent"] is True and out["parts"] == 2
    assert out["source"] == "self:render"
    # the animated draws cost the source no health: it was
    # healthy — the asset's shape was wrong for a pair
    assert svc.health.status("waifu.pics").get("fails", 0) == 0
