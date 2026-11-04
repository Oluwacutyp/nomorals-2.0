"""DESIGN §21, the unbuilt remainder: personal sticker packs
(the stealer's other half) and the conversion chain made real.

Packs: a steal lands in the THIEF'S pack — pack-owned bytes
for WebP/WebM, by-reference (with the reason named) when the
platform never hands over bytes, .tgs by reference always.
Conversions are verified as REAL files — format signatures,
ffprobe streams and durations — including the two shapes the
shipped sticker→MP4 got wrong (WebM: ffmpeg rejects -loop;
animated WebP: -loop hangs and this ffmpeg cannot decode it
at all, so the PIL frame rung carries it), the no-audio
MP4→MP3 refusal in plain words, and the ffmpeg-absent path.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.runtime_expression import (
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat import media_convert, sticker_packs
from nomorals.social.chat.base import ChatKind

BOT_KEY = "telegram-bot:-100123"
USER_KEY = "telegram:-100999"

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe absent on this machine")


# ── the rig (the test_group_expression pattern) ─────────────


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
    """download_file serves REAL sticker bytes registered per
    file_id, or nothing ("" path) when ``serve`` maps the id
    to None — the by-reference platform shape."""

    def __init__(self, tmp_path, serve=None, admin_ids=(),
                 bot_status="administrator"):
        self.calls = []
        self.media_sent = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status
        self.tmp_path = tmp_path
        self.serve = serve or {}

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}

    def send_media(self, chat, ref):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="TGFILE1", error="")

    def download_file(self, file_id, fname="media"):
        payload = self.serve.get(file_id)
        if payload is None:
            return ""
        dest = self.tmp_path / f"dl_{file_id}_{fname}"
        dest.write_bytes(payload)
        return str(dest)


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin,
               RuntimeGamesMixin):
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


def _rig(tmp_path, serve=None, platform="telegram-bot",
         key=BOT_KEY, kind=ChatKind.GROUP, **kw):
    adapter = _Adapter(tmp_path, serve=serve, **kw)
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    return rt, adapter, _Chat(platform, key, kind=kind)


def _webp_bytes(tmp_path, animated=False):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    if animated:
        frames = [Image.new("RGBA", (128, 128),
                            (i * 60, 100, 200, 255))
                  for i in range(4)]
        path = tmp_path / "fixture_anim.webp"
        frames[0].save(path, "WEBP", save_all=True,
                       append_images=frames[1:], duration=120,
                       loop=0)
    else:
        path = tmp_path / "fixture_static.webp"
        Image.new("RGBA", (512, 512), (255, 0, 0, 255)).save(
            path, "WEBP", lossless=True)
    return path.read_bytes()


def _sticker_reply(chat, file_id="FID9", name="sticker.webp",
                   sender_id="7"):
    return _Msg(chat, reply_to="42", sender_id=sender_id,
                meta={"replied_media": {
                    "kind": "sticker", "file_id": file_id,
                    "file_unique_id": "U9", "name": name,
                    "set_name": "", "emoji": ""}})


def _pack(rt, sender_platform_id="telegram:7"):
    return rt._sticker_pack_store().pack(sender_platform_id)


# ── packs: capture ──────────────────────────────────────────


def test_steal_webp_lands_in_thiefs_pack_and_resends(tmp_path):
    webp = _webp_bytes(tmp_path)
    rt, adapter, chat = _rig(tmp_path, serve={"FID9": webp},
                             admin_ids={"7"})
    out = rt._control_steal("", _sticker_reply(chat))
    assert out == ""  # the re-sent sticker IS the reply
    pack = _pack(rt)
    assert len(pack) == 1
    item = pack[0]
    assert item["format"] == "webp"
    assert item["by_reference"] is False
    owned = Path(item["path"])
    assert owned.exists() and owned.read_bytes() == webp
    assert "packs" in owned.parts  # the pack owns its copy
    # her library still gets its capture (existing behavior)
    assert len(rt._expression_library()) == 1
    # list + send from the pack
    listing = rt._control_sticker("pack", _Msg(chat))
    assert "your sticker pack (1)" in str(listing)
    sent_before = len(adapter.media_sent)
    out = rt._control_sticker("pack 1", _Msg(chat))
    assert out == ""
    assert len(adapter.media_sent) == sent_before + 1
    ref = adapter.media_sent[-1]
    assert ref.kind == "sticker" and ref.path == str(owned)


def test_steal_webm_lands_in_pack_as_webm(tmp_path):
    webm = b"\x1a\x45\xdf\xa3" + b"\x00" * 64
    rt, adapter, chat = _rig(tmp_path, serve={"FIDV": webm},
                             admin_ids={"7"})
    msg = _sticker_reply(chat, file_id="FIDV",
                         name="sticker.webm")
    assert rt._control_steal("", msg) == ""
    item = _pack(rt)[0]
    assert item["format"] == "webm"
    assert Path(item["path"]).read_bytes() == webm


def test_steal_tgs_is_by_reference_with_the_reason(tmp_path):
    tgs = gzip.compress(b'{"v":"5.5.2","fr":30}')
    rt, adapter, chat = _rig(tmp_path, serve={"FIDT": tgs},
                             admin_ids={"7"})
    msg = _sticker_reply(chat, file_id="FIDT",
                         name="sticker.webp")
    assert rt._control_steal("", msg) == ""
    item = _pack(rt)[0]
    assert item["by_reference"] is True
    assert item["path"] == ""
    assert ".tgs" in item["note"]
    assert item["file_ids"]["telegram-bot"] == "FIDT"


def test_steal_by_reference_when_platform_gives_no_bytes(
        tmp_path):
    rt, adapter, chat = _rig(tmp_path, serve={}, admin_ids={"7"})
    assert rt._control_steal("", _sticker_reply(chat)) == ""
    item = _pack(rt)[0]
    assert item["by_reference"] is True
    assert "platform reference" in item["note"]
    # and it re-sends by handle from the pack
    out = rt._control_sticker("pack 1", _Msg(chat))
    assert out == ""
    ref = adapter.media_sent[-1]
    assert ref.path == "" and ref.file_id == "FID9"


def test_steal_dedupes_and_packs_are_per_person(tmp_path):
    webp = _webp_bytes(tmp_path)
    rt, adapter, chat = _rig(tmp_path, serve={"FID9": webp},
                             admin_ids={"7"})
    rt._control_steal("", _sticker_reply(chat))
    rt._control_steal("", _sticker_reply(chat))
    assert len(_pack(rt)) == 1  # never the same sticker twice
    # someone else's pack is their own
    out = rt._control_sticker("pack",
                              _Msg(chat, sender_id="8"))
    assert "empty" in str(out)
    assert _pack(rt, "telegram:8") == []


def test_sticker_steal_alias_routes_to_the_stealer(tmp_path):
    webp = _webp_bytes(tmp_path)
    rt, adapter, chat = _rig(tmp_path, serve={"FID9": webp},
                             admin_ids={"7"})
    out = rt._control_sticker("steal", _sticker_reply(chat))
    assert out == ""
    assert len(_pack(rt)) == 1
    assert len(rt._expression_library()) == 1


def test_pack_and_steal_refused_in_unclaimed_room(tmp_path):
    webp = _webp_bytes(tmp_path)
    rt, adapter, chat = _rig(tmp_path, serve={"FID9": webp},
                             platform="telegram", key=USER_KEY)
    out = rt._control_steal("", _sticker_reply(chat))
    assert "isn't claimed" in str(out)
    assert _pack(rt) == []
    assert len(rt._expression_library()) == 0
    out = rt._control_sticker("pack", _Msg(chat))
    assert "isn't claimed" in str(out)
    assert adapter.media_sent == []


def test_pack_works_in_dm(tmp_path):
    webp = _webp_bytes(tmp_path)
    rt, adapter, chat = _rig(tmp_path, serve={"FID9": webp},
                             key="telegram-bot:7",
                             kind=ChatKind.DM)
    assert rt._control_steal("", _sticker_reply(chat)) == ""
    assert len(_pack(rt)) == 1
    out = rt._control_sticker("pack 1", _Msg(chat))
    assert out == "" and adapter.media_sent


# ── the pack store itself ───────────────────────────────────


def test_pack_store_quarantines_a_corrupt_file(tmp_path):
    path = tmp_path / "sticker_packs.json"
    path.write_text("{not json", encoding="utf-8")
    store = sticker_packs.StickerPackStore(path)
    assert store.recovered is True
    assert store.pack("telegram:7") == []
    assert list(tmp_path.glob("sticker_packs.corrupt-*"))


def test_pack_full_is_a_plain_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(sticker_packs, "MAX_PACK_ITEMS", 1)
    store = sticker_packs.StickerPackStore(
        tmp_path / "sticker_packs.json")
    src = tmp_path / "one.webp"
    src.write_bytes(_webp_bytes(tmp_path))
    sticker_packs.capture_sticker(
        store, "telegram:7", path=str(src), kind="sticker",
        media_dir=tmp_path)
    src2 = tmp_path / "two.webp"
    src2.write_bytes(b"RIFF" + b"\x00" * 4 + b"WEBP"
                     + b"different bytes entirely")
    with pytest.raises(sticker_packs.PackError) as err:
        sticker_packs.capture_sticker(
            store, "telegram:7", path=str(src2), kind="sticker",
            media_dir=tmp_path)
    assert "full" in str(err.value)
    assert len(store.pack("telegram:7")) == 1


def test_capture_with_nothing_is_an_honest_refusal(tmp_path):
    store = sticker_packs.StickerPackStore(
        tmp_path / "sticker_packs.json")
    with pytest.raises(sticker_packs.PackError):
        sticker_packs.capture_sticker(
            store, "telegram:7", kind="sticker",
            media_dir=tmp_path)


# ── conversions: real files or honest failures ──────────────


def _make_media(tmp_path):
    """Real fixtures via ffmpeg itself + PIL: a WebM sticker,
    an MP4 with audio, a silent MP4, an MP3."""
    def run(*args):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel",
                        "error", "-y", *args], check=True,
                       timeout=60)

    webm = tmp_path / "clip.webm"
    run("-f", "lavfi", "-i",
        "testsrc=size=256x256:rate=10:duration=1",
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
        "-b:v", "200k", str(webm))
    sound = tmp_path / "sound.mp4"
    run("-f", "lavfi", "-i",
        "testsrc=size=320x240:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(sound))
    silent = tmp_path / "silent.mp4"
    run("-f", "lavfi", "-i",
        "testsrc=size=320x240:rate=10:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent))
    return webm, sound, silent


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


def _is_mp4(path):
    return Path(path).open("rb").read(16)[4:8] == b"ftyp"


@needs_ffmpeg
def test_webm_sticker_becomes_a_real_mp4(tmp_path):
    webm, _sound, _silent = _make_media(tmp_path)
    out = media_convert.sticker_to_mp4(webm, out_dir=tmp_path)
    assert _is_mp4(out)
    probe = _probe(out)
    stream = probe["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["width"] % 2 == 0 and stream["height"] % 2 == 0
    assert 0.7 < float(probe["format"]["duration"]) < 1.5


@needs_ffmpeg
def test_static_webp_becomes_a_short_mp4(tmp_path):
    src = tmp_path / "still.webp"
    src.write_bytes(_webp_bytes(tmp_path))
    out = media_convert.sticker_to_mp4(src, out_dir=tmp_path)
    assert _is_mp4(out)
    probe = _probe(out)
    assert probe["streams"][0]["codec_name"] == "h264"
    assert 2.5 < float(probe["format"]["duration"]) < 3.5


@needs_ffmpeg
def test_animated_webp_becomes_a_real_mp4_via_frames(tmp_path):
    src = tmp_path / "anim.webp"
    src.write_bytes(_webp_bytes(tmp_path, animated=True))
    before = set(Path(tempfile.gettempdir()).glob("webp_frames_*"))
    out = media_convert.sticker_to_mp4(src, out_dir=tmp_path)
    assert _is_mp4(out)
    probe = _probe(out)
    assert probe["streams"][0]["codec_name"] == "h264"
    # 4 frames x 120ms = 0.48s of animation, really carried
    assert 0.2 < float(probe["format"]["duration"]) < 1.5
    after = set(Path(tempfile.gettempdir()).glob("webp_frames_*"))
    assert after <= before  # the temp frame dir was cleaned


@needs_ffmpeg
def test_mp4_to_mp3_extracts_a_real_mp3(tmp_path):
    _webm, sound, _silent = _make_media(tmp_path)
    out = media_convert.mp4_to_mp3(sound, out_dir=tmp_path)
    head = out.open("rb").read(3)
    assert head == b"ID3" or head[:2] == b"\xff\xfb" \
        or head[:2] == b"\xff\xf3"
    assert _probe(out)["streams"][0]["codec_name"] == "mp3"


@needs_ffmpeg
def test_mp4_without_audio_fails_with_the_plain_reason(tmp_path):
    _webm, _sound, silent = _make_media(tmp_path)
    with pytest.raises(media_convert.ConversionError) as err:
        media_convert.mp4_to_mp3(silent, out_dir=tmp_path)
    assert "no audio track in that video" in str(err.value)
    # and no fake artifact lingers
    assert not (tmp_path / "silent.mp3").exists()


@needs_ffmpeg
def test_mp3_to_video_carries_the_full_duration(tmp_path):
    _webm, sound, _silent = _make_media(tmp_path)
    mp3 = media_convert.mp4_to_mp3(sound, out_dir=tmp_path)
    out = media_convert.mp3_to_video(mp3, out_dir=tmp_path)
    assert _is_mp4(out)
    probe = _probe(out)
    kinds = {(s["codec_type"], s["codec_name"])
             for s in probe["streams"]}
    assert ("video", "h264") in kinds
    assert ("audio", "aac") in kinds
    audio_dur = float(_probe(mp3)["format"]["duration"])
    assert abs(float(probe["format"]["duration"])
               - audio_dur) < 0.5


def test_ffmpeg_absent_is_an_honest_named_failure(
        tmp_path, monkeypatch):
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))
    src = tmp_path / "still.webp"
    src.write_bytes(_webp_bytes(tmp_path))
    with pytest.raises(media_convert.ConversionError) as err:
        media_convert.sticker_to_mp4(src, out_dir=tmp_path)
    assert "ffmpeg is not installed" in str(err.value)
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    with pytest.raises(media_convert.ConversionError) as err:
        media_convert.mp4_to_mp3(vid, out_dir=tmp_path)
    assert "ffmpeg is not installed" in str(err.value)


def test_oversize_input_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(media_convert, "MAX_INPUT_BYTES", 16)
    src = tmp_path / "still.webp"
    src.write_bytes(_webp_bytes(tmp_path))
    with pytest.raises(media_convert.ConversionError) as err:
        media_convert.sticker_to_mp4(src, out_dir=tmp_path)
    assert "cap inputs" in str(err.value)


def test_convert_names_never_collide_with_the_currency_guard():
    guard = RuntimeGroupMixin._convert_tail_is_currency
    for name in media_convert.CONVERSIONS:
        assert guard(name) is False, name
    assert guard("200") is True
    assert guard("sparks 50") is True
    assert guard("all") is True
