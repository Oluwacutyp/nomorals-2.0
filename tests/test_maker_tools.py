"""FEI gap wave E — makers & tools: the quote-card maker and
its census skins, /removebg on the segmentation organ, /qr
both ways, /shorturl + /unshorten, the /bible lookup and
/couplepp.

Network-free throughout: provider fixtures carry the REAL
response shapes (bible-api.com and bolls.life payloads were
read off the live APIs — KJV John 3 with its Strong's-number
markup intact; the is.gd plain-text short link; the waifu
pool envelopes the anime suite pins), image fixtures are
tiny generated PNGs, and every provisioning/model seam is
stubbed — no pip install, no model download, no network
ever runs in here.
"""
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_makers import RuntimeMakersMixin
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    whatsapp_menu,
)
from nomorals.social.chat.expression import (
    MEDIA_DELIVERED_KINDS,
    ProviderHealth,
)
from nomorals.social.chat.makers import (
    CARD_KINDS,
    MAKER_COMMANDS,
    MEDIA_KINDS,
    BibleApiProvider,
    BollsProvider,
    IsGdProvider,
    MakerError,
    MakerService,
    TinyUrlProvider,
    VGdProvider,
    _ensure_package,
    make_diptych,
    parse_bible_tail,
    parse_card_request,
    pick_rembg_model,
    read_qr_codes,
    remove_background_bytes,
    render_card_png,
    resolve_redirect_chain,
    validate_long_url,
)

# ── fixture bytes ────────────────────────────────────────────

def _png_bytes(color=(200, 30, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _png_bytes()
ART_BYTES = _png_bytes(color=(90, 60, 160), size=(800, 600))


def _service(tmp_path, **kw) -> MakerService:
    kw.setdefault("media_dir", tmp_path / "maker_media")
    kw.setdefault("health", ProviderHealth())
    return MakerService(**kw)


class _Send:
    """The expression-send seam: records (ref, caption)."""

    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, message_id="9",
                               media_file_id="", error="")


# ── registry ─────────────────────────────────────────────────

def test_registry_presence_all_tables():
    flat = [k for _, ks in LIST_GROUPS for k in ks]
    for kind in MAKER_COMMANDS:
        assert kind in CONTROL_COMMANDS, kind
        assert kind in PUBLIC_COMMANDS, kind
        assert kind in COMMAND_DETAILS, kind
        assert kind in LIST_ONELINERS, kind
        assert flat.count(kind) == 1, kind


def test_makers_have_their_own_list_group():
    groups = dict(LIST_GROUPS)
    assert set(groups["makers & tools — cards · qr · links · "
                      "bible"]) == set(MAKER_COMMANDS)


def test_media_kinds_ride_the_delivered_set():
    assert MEDIA_KINDS <= MEDIA_DELIVERED_KINDS
    # text kinds must NOT be in it — their "" would be a
    # genuine no-reply defect, not a delivery
    for kind in ("shorturl", "unshorten", "bible"):
        assert kind not in MEDIA_DELIVERED_KINDS
    # the FULL set, pinned here (the hand-off wave D took
    # from wave C and wave E takes from wave D): the base
    # expression kinds + wave D's image one-shots + wave E's
    # media makers + the download family (joined 2026-11-05,
    # unpinned until now) + "emoji" (creation rung, 2026-11-06:
    # a created emoji answers by sending its sticker) —
    # nothing more, nothing less.
    assert frozenset({
        "steal", "sticker", "convert", "gif", "react", "emoji",
        "dog", "cat", "fox", "meme", "nasa", "pokemon",
        "qcard", "canvasquote", "iqc", "wndwqute", "meme2",
        "fakemsg", "fakettc", "nokiamsg", "fakeff",
        "fakeffduo", "ffgirl", "removebg", "qr", "couplepp",
        "pp",
        "play", "download", "tiktok", "ttdl", "instagram",
        "spotify", "aio"}) == MEDIA_DELIVERED_KINDS


def test_dispatch_literal_set_equals_maker_commands():
    src = Path("nomorals/agents/partner/runtime.py").read_text(
        encoding="utf-8")
    for match in re.finditer(r"if kind in \{([^}]*)\}:", src):
        following = src[match.end():match.end() + 900]
        if "_control_makers" in following:
            literals = set(re.findall(r'"([a-z0-9]+)"',
                                      match.group(1)))
            assert literals == set(MAKER_COMMANDS)
            return
    raise AssertionError("wave-E dispatch branch not found")


def test_whatsapp_menu_carries_every_maker():
    menu = whatsapp_menu()
    for kind in MAKER_COMMANDS:
        assert f"/{kind}" in menu, kind
    assert len(menu) < 65_536


def test_quote_stays_wave_ds_command():
    # /quote is the fun suite's text quote; the card maker is
    # /qcard — the registry never let the maker take the name.
    assert "qcard" in CARD_KINDS and "quote" not in CARD_KINDS


# ── the card renderer ────────────────────────────────────────

@pytest.mark.parametrize("style", ["quote", "ttc", "chat",
                                   "ff", "meme"])
def test_each_single_style_renders_a_real_png(style):
    data, drawn = render_card_png(
        style, [{"name": "Ada Lovelace",
                 "text": "we ship at dawn, no excuses"}])
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG"
    assert img.size[0] == 920
    assert 120 <= img.size[1] <= 900
    assert drawn["style"] == style
    assert drawn["names"] == ["Ada Lovelace"]
    assert drawn["texts"] == ["we ship at dawn, no excuses"]


def test_duo_renders_two_speakers():
    data, drawn = render_card_png(
        "duo", [{"name": "Ada", "text": "push mid"},
                {"name": "Bola", "text": "on my way"}])
    assert Image.open(io.BytesIO(data)).format == "PNG"
    assert drawn["names"] == ["Ada", "Bola"]
    assert drawn["texts"] == ["push mid", "on my way"]


def test_card_rejects_unknown_style_and_empty_words():
    with pytest.raises(MakerError):
        render_card_png(" hologram",
                        [{"name": "Ada", "text": "hi"}])
    with pytest.raises(MakerError):
        render_card_png("quote", [{"name": "Ada", "text": "  "}])
    with pytest.raises(MakerError):
        render_card_png("duo", [{"name": "Ada", "text": "solo"}])


def test_long_text_wraps_instead_of_overflowing():
    text = " ".join(["word"] * 120)
    _data, drawn = render_card_png(
        "quote", [{"name": "Ada", "text": text}])
    assert drawn["texts"] == [text]
    img = Image.open(io.BytesIO(_data))
    assert img.size[1] > 230  # it grew rows, not width


# ── card request parsing ─────────────────────────────────────

def test_parse_explicit_and_style_first_word():
    style, entries = parse_card_request(
        "qcard", "Ada | hello there", sender_name="Sam")
    assert style == "quote"
    assert entries == [{"name": "Ada", "text": "hello there"}]
    style, entries = parse_card_request(
        "qcard", "ttc Ada | hello", sender_name="Sam")
    assert style == "ttc"
    style, entries = parse_card_request(
        "fakemsg", "chat Bola | hey", sender_name="Sam")
    assert style == "chat"
    assert entries[0]["name"] == "Bola"


def test_parse_skin_kinds_imply_their_style():
    assert parse_card_request("fakettc", "Ada | hi")[0] == "ttc"
    assert parse_card_request("nokiamsg", "Ada | hi")[0] == "chat"
    assert parse_card_request("fakeff", "Ada | hi")[0] == "ff"
    assert parse_card_request("ffgirl", "Ada | hi")[0] == "ff"
    assert parse_card_request("meme2", "Ada | hi")[0] == "meme"
    assert parse_card_request("canvasquote", "Ada | hi")[0] \
        == "quote"


def test_parse_reply_form_and_sender_form():
    style, entries = parse_card_request(
        "qcard", "", reply_name="Ada", reply_text="her words",
        sender_name="Sam")
    assert entries == [{"name": "Ada", "text": "her words"}]
    # bare words, no pipe: the sender quotes THEMSELVES
    style, entries = parse_card_request(
        "qcard", "my own words", sender_name="Sam")
    assert entries == [{"name": "Sam", "text": "my own words"}]


def test_parse_duo_needs_four_parts():
    style, entries = parse_card_request(
        "fakeffduo", "Ada | push mid | Bola | on my way")
    assert style == "duo" and len(entries) == 2
    with pytest.raises(MakerError) as err:
        parse_card_request("fakeffduo", "Ada | only one side")
    assert "two speakers" in str(err.value)
    with pytest.raises(MakerError):
        parse_card_request("qcard", "")


def test_service_card_delivers_through_send(tmp_path):
    send = _Send()
    out = _service(tmp_path).card(
        "qcard", "Ada | we ship at dawn", send=send)
    assert out["sent"] is True and out["style"] == "quote"
    ref, caption = send.calls[0]
    assert ref.mime == "image/png"
    assert Path(ref.path).exists()
    assert "Ada" in caption
    # and a usage miss sends nothing, says why
    send2 = _Send()
    out = _service(tmp_path).card("qcard", "", send=send2)
    assert out["sent"] is False
    assert out["reason"].startswith("usage:")
    assert send2.calls == []


# ── /removebg: provisioning + the model seam ─────────────────

def _fake_remover(img, *, model=""):
    out = img.convert("RGBA")
    px = out.load()
    px[0, 0] = (255, 0, 0, 0)  # a transparent corner
    return out


def test_removebg_stubbed_rung_yields_alpha_png(tmp_path):
    png, model = remove_background_bytes(
        PNG_BYTES, remover=_fake_remover, model="u2netp")
    img = Image.open(io.BytesIO(png))
    assert img.mode == "RGBA"
    assert img.getpixel((0, 0))[3] == 0
    assert model == "u2netp"


def test_removebg_service_sends_the_cutout(tmp_path):
    send = _Send()
    out = _service(tmp_path, remover=_fake_remover).removebg(
        PNG_BYTES, send=send)
    assert out["sent"] is True
    ref, caption = send.calls[0]
    img = Image.open(ref.path)
    assert img.mode == "RGBA"
    assert "background removed" in caption


def test_removebg_unprovisioned_is_one_honest_line(tmp_path):
    # this box has no rembg; the ensure seam refuses — the
    # answer names the rung and the failed self-install
    svc = _service(tmp_path, ensure_rembg_fn=lambda: False)
    out = svc.removebg(PNG_BYTES, send=_Send())
    assert out["sent"] is False
    assert "rembg" in out["reason"]
    assert "installing it myself" in out["reason"]


def test_removebg_non_image_is_an_honest_ask(tmp_path):
    svc = _service(tmp_path, remover=_fake_remover)
    out = svc.removebg(b"definitely not an image", send=_Send())
    assert out["sent"] is False
    assert "isn't an image" in out["reason"]


def test_pick_rembg_model_bounds_the_first_run():
    never = lambda model: False  # noqa: E731
    always = lambda model: True  # noqa: E731
    # birefnet-general (800MB) uncached → the 4.7MB rung
    assert pick_rembg_model(cached=never,
                            default="birefnet-general") == "u2netp"
    # …but once its weights are on disk, the default stands
    assert pick_rembg_model(cached=always,
                            default="birefnet-general") \
        == "birefnet-general"
    # a light default is never swapped
    assert pick_rembg_model(cached=never, default="u2netp") \
        == "u2netp"
    assert pick_rembg_model(cached=never, default="u2net") \
        == "u2net"


def test_ensure_package_attempts_once_and_rechecks():
    import nomorals.social.chat.makers as makers

    makers._PROVISION_ATTEMPTED.discard("fakepkg")
    state = {"installed": False, "runs": 0}

    def runner():
        state["runs"] += 1
        state["installed"] = True
        return True

    ok = _ensure_package(
        "fakepkg", "fakepkg",
        available_fn=lambda: state["installed"], runner=runner)
    assert ok is True and state["runs"] == 1
    # a failed install is not retried within the process
    makers._PROVISION_ATTEMPTED.discard("fakepkg2")
    state2 = {"runs": 0}

    def bad_runner():
        state2["runs"] += 1
        return False

    assert _ensure_package("fakepkg2", "fakepkg2",
                           available_fn=lambda: False,
                           runner=bad_runner) is False
    assert _ensure_package("fakepkg2", "fakepkg2",
                           available_fn=lambda: False,
                           runner=bad_runner) is False
    assert state2["runs"] == 1


# ── /qr: the library seam + the vision decoder ───────────────

class _FakeQrLib:
    """The qrcode seam: make() returns a PIL image, like the
    real library's QrImage.get_image()."""

    def __init__(self):
        self.made = []

    def make(self, text):
        self.made.append(text)
        img = Image.new("RGB", (120, 120), (255, 255, 255))
        return SimpleNamespace(get_image=lambda: img)


def test_qr_make_delivers_a_png(tmp_path):
    lib = _FakeQrLib()
    send = _Send()
    out = _service(tmp_path, qr_lib=lib).qr_make(
        "https://example.com/join", send=send)
    assert out["sent"] is True
    assert lib.made == ["https://example.com/join"]
    ref, caption = send.calls[0]
    assert Image.open(ref.path).format == "PNG"
    assert caption == "https://example.com/join"


def test_qr_make_unprovisioned_line(tmp_path):
    svc = _service(tmp_path, ensure_qrcode_fn=lambda: False)
    out = svc.qr_make("hello", send=_Send())
    assert out["sent"] is False
    assert "qrcode library" in out["reason"]


def test_qr_read_maps_the_vision_shape():
    def reader(data):
        assert data == PNG_BYTES
        return {"count": 1, "method": "pyzbar-zbar",
                "codes": [{"type": "QRCODE",
                           "data": "https://example.com/x",
                           "bbox_1000": {"x": 1, "y": 2,
                                         "w": 3, "h": 4}}]}

    codes = read_qr_codes(PNG_BYTES, reader=reader)
    assert codes == [{"type": "QRCODE",
                      "data": "https://example.com/x"}]
    assert read_qr_codes(PNG_BYTES,
                         reader=lambda data: {"count": 0,
                                              "codes": []}) == []


def test_qr_read_surfaces_the_unavailable_rung():
    from nomorals.vision.native import NativeUnavailable

    def reader(data):
        raise NativeUnavailable(
            "QR/barcode reading needs pyzbar + the zbar "
            "system library (pip install pyzbar; Termux: "
            "pkg install zbar)")

    with pytest.raises(MakerError) as err:
        read_qr_codes(PNG_BYTES, reader=reader)
    assert "pyzbar" in str(err.value)


# ── /shorturl + /unshorten ───────────────────────────────────

class _TextSeed:
    """A shortener seed with a scripted plain-text answer."""

    def __init__(self, provider_cls, answer=None, error=None):
        self.provider = provider_cls(
            fetcher=lambda url: self._answer(url))
        self.answer = answer
        self.error = error
        self.urls = []

    def _answer(self, url):
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return self.answer


def test_shortener_providers_read_the_plain_text_shape():
    isgd = IsGdProvider(fetcher=lambda url: "https://is.gd/q0jCDH")
    assert isgd.shorten("https://example.com/a/long/path") == \
        {"short": "https://is.gd/q0jCDH"}
    vgd = VGdProvider(fetcher=lambda url: "https://v.gd/Ab12Cd")
    assert vgd.shorten("https://example.com")["short"] \
        == "https://v.gd/Ab12Cd"
    tiny = TinyUrlProvider(
        fetcher=lambda url: "https://tinyurl.com/2p9xe")
    assert tiny.shorten("https://example.com")["short"] \
        == "https://tinyurl.com/2p9xe"
    # is.gd answers refusals as "Error: …" TEXT, not a status
    bad = IsGdProvider(
        fetcher=lambda url: "Error: Please enter a valid URL")
    with pytest.raises(MakerError):
        bad.shorten("https://example.com")


def test_shorten_walks_the_chain_and_names_its_source(tmp_path):
    down = _TextSeed(IsGdProvider, error=RuntimeError("down"))
    live = _TextSeed(VGdProvider, answer="https://v.gd/Ab12Cd")
    svc = _service(tmp_path, chains={
        "shorturl": [down.provider, live.provider]})
    out = svc.shorten("https://example.com/a/very/long/path")
    assert out == {"ok": True, "short": "https://v.gd/Ab12Cd",
                   "url": "https://example.com/a/very/long/path",
                   "source": "v.gd"}
    # the seed received the URL-encoded long URL
    assert "example.com%2Fa%2Fvery%2Flong%2Fpath" in live.urls[0]


def test_shorten_refuses_non_links_and_dead_chains(tmp_path):
    svc = _service(tmp_path)
    for bad in ("", "notaurl", "ftp://example.com/x",
                "https://", "just some words"):
        out = svc.shorten(bad)
        assert out["ok"] is False, bad
        assert "isn't a web link" in out["reason"]
    with pytest.raises(MakerError):
        validate_long_url("example.com/no-scheme")
    dead = _service(tmp_path, chains={"shorturl": [
        _TextSeed(IsGdProvider, error=RuntimeError("x")).provider]})
    out = dead.shorten("https://example.com/x")
    assert out["ok"] is False
    assert "every shortener seed is down" in out["reason"]


def test_unshorten_walks_hops_without_a_body():
    hops = {
        "https://is.gd/q0jCDH": (301, "https://t.co/abc"),
        "https://t.co/abc": (302, "/landing"),
        "https://t.co/landing": (200, ""),
    }

    def hop(url):
        return hops[url]

    out = resolve_redirect_chain("https://is.gd/q0jCDH", hop=hop)
    assert out == {"final": "https://t.co/landing", "hops": 2,
                   "status": 200}
    still = resolve_redirect_chain(
        "https://example.com/direct",
        hop=lambda url: (200, ""))
    assert still["hops"] == 0
    assert still["final"] == "https://example.com/direct"


def test_unshorten_stops_a_loop_and_a_marathon():
    loop = {"https://a.example/x": (301, "https://b.example/y"),
            "https://b.example/y": (301, "https://a.example/x")}
    with pytest.raises(MakerError) as err:
        resolve_redirect_chain("https://a.example/x",
                               hop=lambda url: loop[url])
    assert "loops back on itself" in str(err.value)
    with pytest.raises(MakerError):
        resolve_redirect_chain(
            "https://a.example/0",
            hop=lambda url: (301, url + "x"), max_hops=3)


def test_unshorten_service_shapes(tmp_path):
    svc = _service(tmp_path, hop_fetcher=lambda url: (200, ""))
    out = svc.unshorten("https://example.com/plain")
    assert out["ok"] is True and out["hops"] == 0
    out = svc.unshorten("not a url")
    assert out["ok"] is False


# ── /bible ───────────────────────────────────────────────────

#: The live bible-api.com KJV shape for John 3:16 (the verse
#: text is the KJV's own words; the payload wraps it in \n).
BIBLEAPI_JOHN_3_16 = {
    "reference": "John 3:16",
    "verses": [{
        "book_id": "JHN", "book_name": "John", "chapter": 3,
        "verse": 16,
        "text": "\nFor God so loved the world, that he gave "
                "his only begotten Son, that whosoever "
                "believeth in him should not perish, but have "
                "everlasting life.\n"}],
    "text": "\nFor God so loved the world…\n",
    "translation_id": "kjv",
    "translation_name": "King James Version",
    "translation_note": "Public Domain",
}

KJV_3_16 = ("For God so loved the world, that he gave his "
            "only begotten Son, that whosoever believeth in "
            "him should not perish, but have everlasting life.")

#: The live bolls.life KJV chapter shape — Strong's markup
#: (~~n~~) rides inside the text and must be stripped, the
#: words themselves quoted exactly.
BOLLS_JOHN_3 = [
    {"pk": 2626311, "verse": 15,
     "text": "That~~3363~~ whosoever~~3956~~ believeth~~4100~~ "
             "in~~1519~~ him~~846~~ should~~622~~ not~~3363~~ "
             "perish~~622~~, but~~235~~ have~~2192~~ "
             "eternal~~166~~ life~~2222~~."},
    {"pk": 2626312, "verse": 16,
     "text": "For~~1063~~ God~~2316~~ so~~3779~~ loved~~25~~ "
             "the world~~2889~~, that~~5620~~ he gave~~1325~~ "
             "his~~846~~ only begotten~~3439~~ Son~~5207~~, "
             "that~~2443~~ whosoever~~3956~~ believeth~~4100~~ "
             "in~~1519~~ him~~846~~ should~~622~~ not~~3361~~ "
             "perish~~622~~, but~~235~~ have~~2192~~ "
             "everlasting~~166~~ life~~2222~~."},
]

#: The live bolls search shape: hits carry ==highlight==
#: marks, Strong's tags, and numeric book ids (62 = 1 John).
BOLLS_SEARCH_LOVE = [
    {"pk": 2630786, "translation": "KJV", "book": 62,
     "chapter": 4, "verse": 8,
     "text": "He that ==love==th~~25~~ not~~3361~~ "
             "knoweth~~1097~~ not~~3756~~ God~~2316~~; "
             "for~~1063~~ God~~2316~~ is~~2076~~ "
             "==love==~~26~~."},
    {"pk": 2616666, "translation": "KJV", "book": 20,
     "chapter": 4, "verse": 6,
     "text": "Forsake~~5800~~ her not, and she shall "
             "preserve~~8104~~ thee: ==love==~~157~~ her, "
             "and she shall keep~~5341~~ thee."},
    {"pk": 1, "translation": "KJV", "book": 999, "chapter": 1,
     "verse": 1, "text": "an unnameable book hit"},
]


def test_bible_parse_references():
    mode, ref = parse_bible_tail("John 3:16")
    assert mode == "passage"
    assert (ref.book, ref.number, ref.chapter, ref.verse_from,
            ref.verse_to, ref.version) == ("John", 43, 3, 16, 0,
                                           "kjv")
    assert ref.label() == "John 3:16"
    mode, ref = parse_bible_tail("1 Samuel 1:1")
    assert (ref.book, ref.number) == ("1 Samuel", 9)
    mode, ref = parse_bible_tail("Psalm 23")
    assert (ref.book, ref.chapter, ref.verse_from) \
        == ("Psalms", 23, 0)
    assert ref.label() == "Psalms 23"
    mode, ref = parse_bible_tail("Song of Solomon 2:7-9 web")
    assert (ref.book, ref.verse_from, ref.verse_to,
            ref.version) == ("Song of Solomon", 7, 9, "web")
    assert ref.label() == "Song of Solomon 2:7-9"
    mode, parsed = parse_bible_tail("search love")
    assert mode == "search" and parsed == ("love", "kjv")


def test_bible_parse_refusals_name_the_miss():
    for bad in ("", "John", "Narnia 1:1", "3:16",
                "John chapter three"):
        with pytest.raises(MakerError):
            parse_bible_tail(bad)
    with pytest.raises(MakerError) as err:
        parse_bible_tail("Narnia 1:1")
    assert "isn't a book I know" in str(err.value)


def test_bibleapi_fixture_maps_exactly():
    seen = []

    def fetch(url):
        seen.append(url)
        return BIBLEAPI_JOHN_3_16

    _mode, ref = parse_bible_tail("John 3:16")
    out = BibleApiProvider(fetcher=fetch).passage(ref)
    assert out["reference"] == "John 3:16"
    assert out["translation"] == "King James Version"
    assert out["verses"] == [{"verse": 16, "text": KJV_3_16}]
    assert "john%203%3a16" in seen[0].lower()
    assert "translation=kjv" in seen[0]


def test_bibleapi_unknown_reference_is_named():
    import urllib.error

    def fetch(url):
        raise urllib.error.HTTPError(url, 404, "not found",
                                     None, None)

    _mode, ref = parse_bible_tail("John 99:1")
    with pytest.raises(MakerError) as err:
        BibleApiProvider(fetcher=fetch).passage(ref)
    assert "no such passage" in str(err.value)


def test_bolls_passage_cleans_markup_and_filters_the_span():
    _mode, ref = parse_bible_tail("John 3:16")
    out = BollsProvider(fetcher=lambda url: BOLLS_JOHN_3
                        ).passage(ref)
    assert out["verses"] == [{"verse": 16, "text": KJV_3_16}]
    assert out["translation"] == "KJV"
    _mode, chapter = parse_bible_tail("John 3")
    out = BollsProvider(fetcher=lambda url: BOLLS_JOHN_3
                        ).passage(chapter)
    assert [v["verse"] for v in out["verses"]] == [15, 16]


def test_bolls_search_cleans_and_names_books():
    out = BollsProvider(fetcher=lambda url: BOLLS_SEARCH_LOVE
                        ).search("love", "kjv")
    assert [h["reference"] for h in out["hits"]] == [
        "1 John 4:8", "Proverbs 4:6"]  # book 999 skipped
    assert out["hits"][0]["text"] == (
        "He that loveth not knoweth not God; for God is love.")
    assert "==" not in out["hits"][1]["text"]
    assert "~~" not in out["hits"][1]["text"]


def test_bible_service_and_formatting(tmp_path):
    svc = _service(tmp_path, chains={
        "bible": [BibleApiProvider(
            fetcher=lambda url: BIBLEAPI_JOHN_3_16)]})
    rt = RuntimeMakersMixin()
    rt._makers_service_obj = svc
    out = rt._makers_bible("John 3:16")
    assert out.startswith("📖 John 3:16 — King James Version "
                          "(via bible-api.com)")
    assert f"16 {KJV_3_16}" in out
    bad = rt._makers_bible("Narnia 1:1")
    assert "isn't a book I know" in bad


def test_bible_chapter_reply_caps_its_verses(tmp_path):
    verses = [{"book_id": "JHN", "book_name": "John",
               "chapter": 3, "verse": n,
               "text": f"verse {n} text"}
              for n in range(1, 46)]
    payload = {"reference": "John 3", "verses": verses,
               "translation_name": "King James Version"}
    svc = _service(tmp_path, chains={
        "bible": [BibleApiProvider(fetcher=lambda url: payload)]})
    rt = RuntimeMakersMixin()
    rt._makers_service_obj = svc
    out = rt._makers_bible("John 3")
    lines = out.splitlines()
    numbered = [ln for ln in lines if ln.split(" ", 1)[0]
                .isdigit()]
    assert len(numbered) == 40
    assert "…and 5 more verses" in out


def test_bible_search_formatting(tmp_path):
    svc = _service(tmp_path, chains={
        "bible_search": [BollsProvider(
            fetcher=lambda url: BOLLS_SEARCH_LOVE)]})
    rt = RuntimeMakersMixin()
    rt._makers_service_obj = svc
    out = rt._makers_bible("search love")
    assert "bible search — “love” (KJV via bolls.life)" in out
    assert "• 1 John 4:8 — He that loveth not" in out


# ── /couplepp ────────────────────────────────────────────────

class _Pool:
    def __init__(self, name, url=None, error=None):
        self.name = name
        self.url = url
        self.error = error

    def image_url(self):
        if self.error is not None:
            raise self.error
        return self.url


def test_diptych_splits_one_artwork():
    png, info = make_diptych(ART_BYTES)
    img = Image.open(io.BytesIO(png))
    assert img.size == (968, 640)
    assert info["half"] == (480, 640)
    with pytest.raises(MakerError):
        make_diptych(b"not an image")


def test_couplepp_delivers_the_labelled_pair(tmp_path):
    pool = _Pool("waifu.pics", url="https://i.waifu.pics/x.jpg")
    send = _Send()
    svc = _service(tmp_path, chains={"couplepp": [pool]},
                   bytes_fetcher=lambda url: ART_BYTES)
    out = svc.couplepp(send=send)
    assert out["sent"] is True and out["source"] == "waifu.pics"
    # Delivery-shape law (owner, 2026-11-07): the pair lands
    # as TWO separate profile-ready photos — the left half,
    # then the right half — never one combined poster the
    # couple must crop themselves.
    assert out["parts"] == 2 and len(send.calls) == 2
    (ref1, cap1), (ref2, cap2) = send.calls
    assert Image.open(ref1.path).size == (480, 640)
    assert Image.open(ref2.path).size == (480, 640)
    assert "one artwork via waifu.pics" in cap1
    assert "left half" in cap1 and "right half" in cap2


def test_couplepp_sheds_a_bad_pool_and_dies_honestly(tmp_path):
    html = _Pool("waifu.pics", url="https://i.waifu.pics/x.jpg")
    good = _Pool("nekos.best", url="https://nekos.best/y.png")
    svc = _service(
        tmp_path, chains={"couplepp": [html, good]},
        bytes_fetcher=lambda url: b"<html>error</html>"
        if "waifu" in url else ART_BYTES)
    out = svc.couplepp(send=_Send())
    assert out["sent"] is True and out["source"] == "nekos.best"
    down = _service(tmp_path, chains={"couplepp": [
        _Pool("waifu.pics", error=RuntimeError("down"))]})
    out = down.couplepp(send=_Send())
    assert out["sent"] is False
    assert "pools are down" in out["reason"]


# ── the runtime handler (mixin rig, wave-D pattern) ──────────

from nomorals.agents.partner.runtime_expression import (  # noqa: E402
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_group import (  # noqa: E402
    RuntimeGroupMixin,
)

BOT_KEY = "telegram-bot:-100123"
USER_KEY = "telegram:-100123"


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
    def __init__(self, admin_ids=(), bot_status="administrator"):
        self.media_sent = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status

    def get_me(self):
        return {"id": self._bot_id, "username": "Nomorals_bot"}

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
                               media_file_id="", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeMakersMixin, RuntimeExpressionMixin,
               RuntimeGroupMixin):
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


def _rig(tmp_path, service=None, platform="telegram-bot",
         key=BOT_KEY):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {platform: adapter},
                  operators={"99"})
    chat = _Chat(platform, key)
    rt._makers_service_obj = service or _service(tmp_path)
    return rt, adapter, chat


def test_handler_card_from_a_reply(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    msg = _Msg(chat, meta={
        "reply_to_sender": {"id": "8", "username": "bola",
                            "name": "Bola"},
        "reply_to_text": "we ship at dawn"})
    out = rt._control_makers("qcard", "", msg)
    assert out == ""  # the card WAS the reply
    assert len(adapter.media_sent) == 1
    ref, caption = adapter.media_sent[0]
    assert Image.open(ref.path).format == "PNG"
    assert "Bola" in caption


def test_handler_card_explicit_and_usage(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    out = rt._control_makers("fakettc", "Ada | first!",
                             _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == 1
    out = rt._control_makers("qcard", "", _Msg(chat))
    assert out.startswith("usage:")


def test_handler_unclaimed_room_is_gated(tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    out = rt._control_makers("bible", "John 3:16", _Msg(chat))
    assert "isn't claimed" in str(out)
    out = rt._control_makers("qcard", "Ada | hi", _Msg(chat))
    assert "isn't claimed" in str(out)
    assert adapter.media_sent == []


def test_handler_removebg_round_trip(tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(PNG_BYTES)
    svc = _service(tmp_path, remover=_fake_remover)
    rt, adapter, chat = _rig(tmp_path, service=svc)
    msg = _Msg(chat, media=[SimpleNamespace(
        path=str(photo), kind="image", file_id="")])
    out = rt._control_makers("removebg", "", msg)
    assert out == ""
    ref, _caption = adapter.media_sent[0]
    assert Image.open(ref.path).mode == "RGBA"
    # and with nothing to cut, the ask is honest
    out = rt._control_makers("removebg", "", _Msg(chat))
    assert "reply to a photo" in out


def test_handler_qr_make_and_read(tmp_path):
    lib = _FakeQrLib()

    def reader(data):
        return {"count": 1, "codes": [
            {"type": "QRCODE", "data": "https://example.com/x"}]}

    svc = _service(tmp_path, qr_lib=lib, qr_reader=reader)
    rt, adapter, chat = _rig(tmp_path, service=svc)
    out = rt._control_makers("qr", "https://example.com/join",
                             _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == 1
    photo = tmp_path / "code.png"
    photo.write_bytes(PNG_BYTES)
    msg = _Msg(chat, media=[SimpleNamespace(
        path=str(photo), kind="image", file_id="")])
    out = rt._control_makers("qr", "read", msg)
    assert out == ("that picture's code says:\n"
                   "https://example.com/x")
    out = rt._control_makers("qr", "", _Msg(chat))
    assert out.startswith("usage: /qr")


def test_handler_shorturl_reply_names_the_destination(tmp_path):
    svc = _service(tmp_path, chains={"shorturl": [
        IsGdProvider(fetcher=lambda url: "https://is.gd/q0jCDH")]})
    rt, _adapter, chat = _rig(tmp_path, service=svc)
    out = rt._control_makers(
        "shorturl", "https://example.com/a/very/long/path",
        _Msg(chat))
    assert out == ("🔗 https://is.gd/q0jCDH\n"
                   "full destination: "
                   "https://example.com/a/very/long/path\n"
                   "— via is.gd")
    rt._makers_service_obj = _service(tmp_path, chains={
        "couplepp": [_Pool("waifu.pics",
                           error=RuntimeError("down"))]})
    out = rt._control_makers("couplepp", "", _Msg(chat))
    assert "pools are down" in out


def test_handler_couplepp_sends_the_pair(tmp_path):
    pool = _Pool("waifu.pics", url="https://i.waifu.pics/x.jpg")
    svc = _service(tmp_path, chains={"couplepp": [pool]},
                   bytes_fetcher=lambda url: ART_BYTES)
    rt, adapter, chat = _rig(tmp_path, service=svc)
    out = rt._control_makers("pp", "", _Msg(chat))
    assert out == ""
    # the pair is TWO sends (delivery-shape law) — the left
    # half and the right half as separate photos
    assert len(adapter.media_sent) == 2
    for ref, _caption in adapter.media_sent:
        assert Image.open(ref.path).size == (480, 640)


# ── the composed control path ────────────────────────────────

def _composed_runtime():
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-wavee-", dir=os.environ.get("NM_TEST_TMPDIR")
        or None)
    settings = load_settings(overrides={
        "home": tmp.name, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false"})
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context), tmp


def test_composed_bible_and_shorturl_through_handle_control():
    rt, tmp = _composed_runtime()
    try:
        svc = MakerService(
            media_dir=Path(tmp.name) / "maker_media",
            health=ProviderHealth(),
            chains={
                "bible": [BibleApiProvider(
                    fetcher=lambda url: BIBLEAPI_JOHN_3_16)],
                "shorturl": [IsGdProvider(
                    fetcher=lambda url: "https://is.gd/q0jCDH")]})
        rt._makers_service_obj = svc
        out = rt.handle_control("/bible John 3:16", "local:room1")
        assert f"16 {KJV_3_16}" in out
        out = rt.handle_control(
            "/shorturl https://example.com/a/very/long/path",
            "local:room1")
        assert "https://is.gd/q0jCDH" in out
        assert "full destination:" in out
    finally:
        tmp.cleanup()


def test_composed_card_delivery_is_not_a_no_reply():
    from nomorals.social.chat.base import ChatMessage, ChatRef

    rt, tmp = _composed_runtime()
    try:
        rt._makers_service_obj = MakerService(
            media_dir=Path(tmp.name) / "maker_media",
            health=ProviderHealth())
        chat = ChatRef(platform="local", chat_id="room1",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True,
                          text="/qcard Ada | we ship at dawn",
                          sender="Ada", sender_id="7",
                          message_id="55")
        reply = rt.handle_control("/qcard Ada | we ship at dawn",
                                  "local:room1", message=msg)
        # the card was the reply: "" — and crucially NOT the
        # never-empty fallback text after a successful send
        assert reply == ""
        landed = list((Path(tmp.name) / "maker_media")
                      .glob("*.png"))
        assert landed
    finally:
        tmp.cleanup()
