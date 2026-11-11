"""Fix 4 audience tails — the classification remainders.

Fix 4 put MediaRef.audience at the SEND seam (a public ref
burns the corner pill, a personal ref never does, personal
wins ties) but four producers still shipped unclassified or
substitute-stamped output.  This suite pins their closure:

- couplepp: BOTH pair halves carry the serving surface's
  audience (service level + the runtime handler, group →
  public with the pill burned at the seam, DM → personal
  with bytes identical to the unmarked control).
- /removebg + /qr: the output inherits the requesting
  surface's audience, the same rule, the same seam.
- /webshot + /webrec: the §19 caption-text stamp is GONE —
  captions are captions again.  A marked group webshot
  carries the designed treatment instead: audience="public"
  at the gateway seam, where the pill is burned.  A PDF
  capture keeps its §19 caption signature (a captured print
  PDF is not a Devon-rendered document).
- render_pdf_embedded: the shipped PDF watermark policy
  applies with render_pdf parity — same resolver, same
  geometry/alpha/tiling on every page, same opt-outs — and
  documents/convert.to_pdf_embedded resolves the mark from
  the manifest exactly as to_pdf does.

Network-free: the maker seams are stubbed exactly as
test_maker_tools stubs them, the webshot capture rung is
faked exactly as test_webshot_owner fakes it, and every
store path is a per-test tmp path.
"""
from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.agents.partner.runtime_expression import (  # noqa: E402
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_group import (  # noqa: E402
    RuntimeGroupMixin,
)
from nomorals.agents.partner.runtime_makers import (  # noqa: E402
    RuntimeMakersMixin,
)
from nomorals.core import branding  # noqa: E402
from nomorals.core.branding import resolve_brand_mark  # noqa: E402
from nomorals.core.pdf import (  # noqa: E402
    PdfError,
    _collect_pages,
    _decompress,
    _parse_objects,
    _stream_bytes,
    read_pdf_text,
    render_pdf,
    render_pdf_embedded,
)
from nomorals.social.chat import webshot  # noqa: E402
from nomorals.social.chat.base import ChatKind, MediaRef  # noqa: E402
from nomorals.social.chat.expression import ProviderHealth  # noqa: E402
from nomorals.social.chat.makers import MakerService  # noqa: E402
from nomorals.social.chat.typography import (  # noqa: E402
    DEFAULT_STAMP_TEXT,
)
from nomorals.social.chat.webshot import CaptureResult  # noqa: E402

BRAND = branding.BRAND_TEXT
GROUP_KEY = "telegram-bot:-1001"

# ── image fixtures (the maker-suite shapes) ──────────────────


def _png_bytes(color=(200, 30, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _png_bytes()
ART_BYTES = _png_bytes(color=(90, 60, 160), size=(800, 600))


def _corner_and_center(data: bytes):
    """(bottom-right pill-region pixels, centre pixel) — the
    designed-watermark suite's probe: a burned pill moves the
    corner and never the centre."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = img.size
    corner = [img.getpixel(p) for p in
              ((w - 4, h - 4), (w - 12, h - 12),
               (w - 30, h - 30), (w - 60, h - 8))]
    return corner, img.getpixel((w // 2, h // 2))


def _service(tmp_path, **kw) -> MakerService:
    kw.setdefault("media_dir", tmp_path / "maker_media")
    kw.setdefault("health", ProviderHealth())
    return MakerService(**kw)


class _Send:
    """The expression-send seam: records (ref, caption)."""

    def __init__(self):
        self.calls = []

    def __call__(self, ref, caption="", **kw):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")


def _fake_remover(img, *, model=""):
    out = img.convert("RGBA")
    px = out.load()
    px[0, 0] = (255, 0, 0, 0)  # a transparent corner
    return out


class _FakeQrLib:
    def __init__(self):
        self.made = []

    def make(self, text):
        self.made.append(text)
        img = Image.new("RGB", (240, 240), (255, 255, 255))
        return SimpleNamespace(get_image=lambda: img)


class _Pool:
    def __init__(self, name, url=None, error=None):
        self.name = name
        self.url = url
        self.error = error

    def image_url(self):
        if self.error is not None:
            raise self.error
        return self.url


def _pair_service(tmp_path) -> MakerService:
    pool = _Pool("waifu.pics", url="https://i.waifu.pics/x.jpg")
    return _service(tmp_path, chains={"couplepp": [pool]},
                    bytes_fetcher=lambda url: ART_BYTES)


# ── tail 2 (service level): removebg + qr carry the audience ─


def test_removebg_service_carries_the_given_audience(tmp_path):
    svc = _service(tmp_path, remover=_fake_remover)
    for audience in ("public", "personal"):
        send = _Send()
        out = svc.removebg(PNG_BYTES, send=send, audience=audience)
        assert out["sent"] is True
        ref, _caption = send.calls[0]
        assert ref.audience == audience
    # no classification named: unclassified, exactly as the
    # seam has always read it ("" — personal at every seam)
    send = _Send()
    svc.removebg(PNG_BYTES, send=send)
    assert send.calls[0][0].audience == ""


def test_qr_service_carries_the_given_audience(tmp_path):
    svc = _service(tmp_path, qr_lib=_FakeQrLib())
    for audience in ("public", "personal"):
        send = _Send()
        out = svc.qr_make("https://example.com/join", send=send,
                          audience=audience)
        assert out["sent"] is True
        assert send.calls[0][0].audience == audience
    send = _Send()
    svc.qr_make("https://example.com/join", send=send)
    assert send.calls[0][0].audience == ""


# ── tail 1 (service level): both pair halves classified ──────


def test_couplepp_service_classifies_both_halves(tmp_path):
    svc = _pair_service(tmp_path)
    for audience in ("public", "personal"):
        send = _Send()
        out = svc.couplepp(send=send, audience=audience)
        assert out["sent"] is True and out["parts"] == 2
        assert [ref.audience for ref, _ in send.calls] == \
            [audience, audience]
    send = _Send()
    svc.couplepp(send=send)
    assert [ref.audience for ref, _ in send.calls] == ["", ""]


# ── the seam itself: public burns, personal is untouched ─────


def test_seam_burns_public_and_never_personal(tmp_path):
    src = tmp_path / "art.png"
    src.write_bytes(ART_BYTES)
    seam = RuntimeExpressionMixin()
    public = seam._brand_marked_ref(MediaRef(
        path=str(src), mime="image/png", kind="image",
        audience="public"))
    assert public.path != str(src)
    assert Path(public.path).name.endswith(".brandmarked.png")
    # the caller's file is never mutated by the burn
    assert src.read_bytes() == ART_BYTES
    marked = Path(public.path).read_bytes()
    corner, center = _corner_and_center(marked)
    plain_corner, plain_center = _corner_and_center(ART_BYTES)
    assert corner != plain_corner       # the pill is there
    assert center == plain_center       # and only there
    for audience in ("personal", ""):
        ref = MediaRef(path=str(src), mime="image/png",
                       kind="image", audience=audience)
        out = seam._brand_marked_ref(ref)
        assert out is ref               # not even a copy


# ── tails 1+2 (runtime): the serving surface decides ─────────


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

    def get_me(self):
        return {"id": self._bot_id, "username": "Nomorals_bot"}

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "administrator"}}

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append((ref, caption))
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")


class _GatewayStub:
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
        self.gateway = _GatewayStub(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators


def _rig(tmp_path, service, *, kind=ChatKind.GROUP,
         key="telegram-bot:-100123"):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {"telegram-bot": adapter},
                  operators={"99"})
    rt._makers_service_obj = service
    chat = _Chat("telegram-bot", key, kind=kind)
    return rt, adapter, chat


def test_runtime_couplepp_group_public_dm_personal(tmp_path):
    # group: both halves classified public, and the seam the
    # handler sends through burns the pill into each half
    rt, adapter, chat = _rig(tmp_path, _pair_service(tmp_path))
    out = rt._control_makers("couplepp", "", _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == 2
    for ref, _caption in adapter.media_sent:
        assert ref.audience == "public"
        assert Path(ref.path).name.endswith(".brandmarked.png")
        corner, _center = _corner_and_center(
            Path(ref.path).read_bytes())
        plain_corner, _ = _corner_and_center(ART_BYTES)
        assert corner != plain_corner
    # DM: both halves personal, original files, bytes untouched
    rt, adapter, chat = _rig(
        tmp_path, _pair_service(tmp_path), kind=ChatKind.DM,
        key="telegram:555")
    out = rt._control_makers("couplepp", "", _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == 2
    control = _Send()
    _pair_service(tmp_path).couplepp(send=control)
    for (ref, _c), (want, _c2) in zip(adapter.media_sent,
                                      control.calls,
                                      strict=True):
        assert ref.audience == "personal"
        assert Path(ref.path).read_bytes() == \
            Path(want.path).read_bytes()


def test_runtime_removebg_group_public_dm_personal(tmp_path):
    from nomorals.social.chat.makers import remove_background_bytes
    photo = tmp_path / "photo.png"
    photo.write_bytes(ART_BYTES)
    media = [SimpleNamespace(path=str(photo), kind="image",
                             file_id="")]
    want, _model = remove_background_bytes(
        ART_BYTES, remover=_fake_remover)
    rt, adapter, chat = _rig(
        tmp_path, _service(tmp_path, remover=_fake_remover))
    out = rt._control_makers("removebg", "", _Msg(chat,
                                                  media=media))
    assert out == ""
    ref, _caption = adapter.media_sent[0]
    assert ref.audience == "public"
    assert Path(ref.path).name.endswith(".brandmarked.png")
    corner, _ = _corner_and_center(Path(ref.path).read_bytes())
    plain_corner, _ = _corner_and_center(want)
    assert corner != plain_corner
    # DM: personal — the delivered file IS the cutout, byte for
    # byte what the service produced unmarked
    rt, adapter, chat = _rig(
        tmp_path, _service(tmp_path, remover=_fake_remover),
        kind=ChatKind.DM, key="telegram:555")
    out = rt._control_makers("removebg", "", _Msg(chat,
                                                  media=media))
    assert out == ""
    ref, _caption = adapter.media_sent[0]
    assert ref.audience == "personal"
    assert ".brandmarked" not in Path(ref.path).name
    assert Path(ref.path).read_bytes() == want


def test_runtime_qr_group_public_dm_personal(tmp_path):
    rt, adapter, chat = _rig(
        tmp_path, _service(tmp_path, qr_lib=_FakeQrLib()))
    out = rt._control_makers("qr", "https://example.com/join",
                             _Msg(chat))
    assert out == ""
    ref, _caption = adapter.media_sent[0]
    assert ref.audience == "public"
    assert Path(ref.path).name.endswith(".brandmarked.png")

    rt, adapter, chat = _rig(
        tmp_path, _service(tmp_path, qr_lib=_FakeQrLib()),
        kind=ChatKind.DM, key="telegram:555")
    out = rt._control_makers("qr", "https://example.com/join",
                             _Msg(chat))
    assert out == ""
    ref, _caption = adapter.media_sent[0]
    assert ref.audience == "personal"
    assert ".brandmarked" not in Path(ref.path).name


# ── tail 3: webshot/webrec captions are captions again ───────


class _FileGateway:
    """Records send_file calls WITH their keyword arguments —
    the audience kwarg is the fact under test."""

    def __init__(self):
        self.calls = []

    def send_file(self, platform, chat_key, path, **kw):
        self.calls.append({"platform": platform,
                           "chat_key": chat_key, "path": path,
                           **kw})
        return SimpleNamespace(ok=True)


def _group_rig(tmp_path):
    rig = RuntimeGroupMixin()
    rig.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)),
        db=None)
    rig.gateway = _FileGateway()
    rig._owner_sender_ids = {"42"}
    rig._is_operator = lambda message: str(
        message.sender_id) == "42"
    return rig


def _gmsg(*, chat_key="telegram:555", platform="telegram",
          kind=ChatKind.DM, sender_id="42"):
    return SimpleNamespace(
        chat=SimpleNamespace(platform=platform, key=chat_key,
                             chat_id=chat_key.split(":")[-1],
                             kind=kind, title=""),
        sender_id=sender_id, sender="Sender",
        sender_username="oluwacutyp", text="",
        meta={}, message_id="m1", reply_to="")


def _fake_shot(url, opts, **kw):
    if opts.pdf:
        return CaptureResult(data=b"%PDF-1.4 fake print",
                             title="Ex", final_url=url, pdf=True,
                             rung="playwright")
    return CaptureResult(data=_png_bytes((255, 255, 255),
                                         (1440, 900)),
                         title="Ex", final_url=url,
                         rung="playwright", width=1440, height=900)


@pytest.fixture()
def _shot_rig(tmp_path, monkeypatch):
    monkeypatch.setattr(webshot, "capture", _fake_shot)
    monkeypatch.setattr(webshot, "_host_is_private",
                        lambda host: False)
    return _group_rig(tmp_path)


def test_webshot_dm_caption_has_no_stamp_no_audience(_shot_rig):
    rig = _shot_rig
    out = rig._control_webshot("https://shots.invalid/page",
                               _gmsg())
    assert "via playwright" in out
    assert len(rig.gateway.calls) == 1
    call = rig.gateway.calls[0]
    # the §19 caption stamp is gone even though the house
    # defaults resolve one — a personal surface is unmarked
    assert DEFAULT_STAMP_TEXT not in call["caption"]
    assert "audience" not in call


def test_webshot_group_image_public_and_caption_clean(_shot_rig):
    rig = _shot_rig
    group = _gmsg(chat_key=GROUP_KEY, platform="telegram-bot",
                  kind=ChatKind.GROUP)
    out = rig._control_webshot("https://shots.invalid/page",
                               group)
    assert "via playwright" in out
    assert len(rig.gateway.calls) == 1
    call = rig.gateway.calls[0]
    # the designed treatment rides the seam classification…
    assert call.get("audience") == "public"
    # …and the caption carries NO stamp text in its place
    assert call["caption"] == \
        "🌐 Ex\nhttps://shots.invalid/page"
    # the journal names the treatment honestly
    entries = rig._webshot_book().journal()
    assert any(e.get("rung") == "watermark"
               and "brand pill" in str(e.get("options", ""))
               for e in entries)


def test_webshot_group_pdf_keeps_its_caption_signature(_shot_rig):
    rig = _shot_rig
    group = _gmsg(chat_key=GROUP_KEY, platform="telegram-bot",
                  kind=ChatKind.GROUP)
    out = rig._control_webshot("pdf https://shots.invalid/page",
                               group)
    assert "print PDF" in out
    call = rig.gateway.calls[0]
    # a captured print PDF is not a Devon-rendered document:
    # its §19 caption behavior is exactly what it was
    assert DEFAULT_STAMP_TEXT in call["caption"]
    assert "audience" not in call


def test_webrec_caption_has_no_stamp(tmp_path, monkeypatch):
    rig = _group_rig(tmp_path)
    captured = {}

    def fake_record(url, opts, **kw):
        path = Path(kw["out_dir"]) / "rec.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 64)
        return SimpleNamespace(
            path=path, fmt="mp4", duration_s=5.0,
            includes_clearing=False,
            nav=SimpleNamespace(
                title="Rec Title", gates=[], wall="",
                presentations_tried=["p"], presentation=""),
            plan=None)

    monkeypatch.setattr(webshot, "record_video", fake_record)
    monkeypatch.setattr(
        rig, "_webrec_deliver",
        lambda message, path, *, caption, name:
            captured.update(caption=caption, name=name))
    out = rig._control_webrec("https://shots.invalid/clip",
                              _gmsg())
    assert out.startswith("🎬 Rec Title")
    assert captured["caption"] == \
        "🎬 Rec Title\nhttps://shots.invalid/clip"
    assert DEFAULT_STAMP_TEXT not in captured["caption"]


# ── tail 4: the embedded-font PDF carries the policy mark ────

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
requires_font = pytest.mark.skipif(
    not Path(_FONT).is_file(), reason="DejaVu font not present")

_BODY = "\n".join(
    f"Embedded watermark body line {i} with readable text."
    for i in range(120))


def _font() -> bytes:
    return Path(_FONT).read_bytes()


def _page_streams(data: bytes) -> list[bytes]:
    objects = _parse_objects(data)
    pages = _collect_pages(objects, objects[1])
    streams = []
    for page in pages:
        m = re.search(rb"/Contents\s+(\d+)\s+0\s+R", page)
        assert m is not None
        streams.append(_decompress(
            _stream_bytes(objects[int(m.group(1))])))
    return streams


def _tm_prefixes(stream: bytes) -> list[bytes]:
    """The watermark rows' geometry prefix (font, size, alpha,
    rotation matrix, start point) — everything up to the text
    operand, which legitimately differs between writers (a
    WinAnsi literal vs embedded glyph ids)."""
    return re.findall(rb"BT /F1 34 Tf .*? Tm", stream)


@requires_font
def test_embedded_marked_every_page_with_legacy_parity():
    mark = resolve_brand_mark()
    data = render_pdf_embedded(_BODY, font_data=_font(),
                               title="Marked", watermark=mark)
    streams = _page_streams(data)
    assert len(streams) >= 2
    for stream in streams:
        # the mark opens the page stream, under the content,
        # tiled three times — the designed suite's assertions,
        # carried to the embedded writer
        assert stream.startswith(b"q /GSW gs")
        assert stream.count(b"/F1 34 Tf") == 3
    assert b"/ExtGState" in data
    assert b"/ca 0.15" in data
    # the mark extracts as text through the ToUnicode CMap,
    # once per tile per page — it IS the brand text
    text = read_pdf_text(data)
    assert text.count(BRAND) == 3 * len(streams)
    # placement parity with render_pdf: the same mark on the
    # same page size yields the same geometry, row for row
    legacy = render_pdf(_BODY, title="Marked", watermark=mark)
    legacy_first = _page_streams(legacy)[0]
    assert _tm_prefixes(streams[0]) == _tm_prefixes(legacy_first)


@requires_font
def test_embedded_unmarked_and_policy_opt_outs_stay_clean():
    # the resolver's one opt-out is "off" — the embedded path
    # honors the same resolution the legacy path does
    for data in (
            render_pdf_embedded(_BODY, font_data=_font(),
                                title="Plain"),
            render_pdf_embedded(
                _BODY, font_data=_font(), title="Off",
                watermark=resolve_brand_mark("off")),
    ):
        assert b"/GSW" not in data
        assert b"/ExtGState" not in data
        assert BRAND not in read_pdf_text(data)
        for stream in _page_streams(data):
            assert not stream.startswith(b"q /GSW gs")


@requires_font
def test_embedded_watermark_coercion_matches_legacy():
    # a plain string is the mark's text, as on the legacy path
    data = render_pdf_embedded("Short body.", font_data=_font(),
                               watermark="Acme Corp")
    assert b"/GSW" in data
    assert "Acme Corp" in read_pdf_text(data)
    # a BrandMark object rides as itself
    data = render_pdf_embedded(
        "Short body.", font_data=_font(),
        watermark=branding.BrandMark(text="Zulu Ltd"))
    assert "Zulu Ltd" in read_pdf_text(data)
    # anything else is refused, never silently dropped
    with pytest.raises(PdfError):
        render_pdf_embedded("Short body.", font_data=_font(),
                            watermark=42)


@requires_font
def test_convert_embedded_pdf_carries_the_house_mark():
    from nomorals.documents import Document, Section, to_pdf, to_pdf_embedded

    doc = Document(
        format="test", title="Embedded Fixture",
        sections=[Section(level=1, heading="Intro",
                          text="Hello Αθήνα, café €50.")])
    blob = to_pdf_embedded(doc, _FONT)
    assert b"/GSW" in blob and b"/ExtGState" in blob
    assert BRAND in read_pdf_text(blob)
    # parity with the sibling producer: same document, same
    # house mark, on the WinAnsi path too
    assert BRAND in read_pdf_text(to_pdf(doc))
