"""WhatsApp sound stickers: the maker, the adapter verb, and /sticker sound.

The owner asked for sound on WhatsApp stickers. The wire shape is
pinned from @aunucore/baileys 2.7.0's published source (MIT):
the audio travels as a SECOND media upload nested on StickerMessage
field 26, and the WebP EXIF carries {"sound-sha256": "<hex>"} so the
receiving client pairs sticker and sound. Limits are the test builds'
(10s photo-origin / 5s video-origin, OGG Opus audio).

Pinned here:
- build_sound_sha256_exif: the little-endian TIFF layout (tag 0x5741
  'AW', type undefined, JSON value) — byte-identical to the Aunu
  builder it is pinned from.
- add_sound_sha256_to_webp: drops any old EXIF, sets the VP8X EXIF
  flag, synthesises a VP8X header for plain VP8/VP8L files, rebuilds
  the RIFF size. One correction against the fork it is pinned from:
  the fork sets 0x20 (the ICC-profile bit) with an "EXIF flag"
  comment; per RFC 9649 the EXIF bit is 0x08, and that is what Devon
  writes.
- video_to_sticker / image_to_sticker gain an audio input: the video's
  own track ("original"), or an audio file. The sound form trims to
  5s (video) / 10s (photo), transcodes to OGG Opus, and returns a
  SoundSticker (sticker + audio + seconds + sha256). No audio means
  exactly today's behaviour (a Path, same argv).
- The adapter's send_media carries the audio payload on kind sticker;
  the outbox round-trips it; inbound stickers report their sound flag.
- /sticker sound: a replied video keeps its audio, a replied photo
  plus a voice note (captioned /sticker sound) takes that audio.
  Plain /sticker is byte-identical, and "/sticker sound" with no
  media still makes the text sticker that says "sound".
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatRef, MediaRef

# ── minimal WebP fixtures (no encoder needed) ────────────────────────


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    out = fourcc + struct.pack("<I", len(data)) + data
    if len(data) % 2:
        out += b"\x00"
    return out


def _webp_bytes(chunks: list[tuple[bytes, bytes]]) -> bytes:
    body = b"WEBP" + b"".join(_chunk(c, d) for c, d in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _vp8x_data(flags: int = 0x02, width: int = 512,
               height: int = 512) -> bytes:
    data = bytearray(10)
    data[0] = flags
    data[4:7] = (width - 1).to_bytes(3, "little")
    data[7:10] = (height - 1).to_bytes(3, "little")
    return bytes(data)


def _vp8l_data() -> bytes:
    # Lossless bitstream: signature 0x2f, then width-1 / height-1
    # as 14-bit LE fields (512x512).
    bits = (511) | (511 << 14)
    return b"\x2f" + bits.to_bytes(4, "little")


def _exif_chunk_for(hex_digest: str) -> bytes:
    payload = json.dumps({"sound-sha256": hex_digest}).encode()
    exif = bytearray(26 + len(payload))
    exif[0:2] = b"II"
    struct.pack_into("<H", exif, 2, 0x2A)
    struct.pack_into("<I", exif, 4, 8)
    struct.pack_into("<H", exif, 8, 1)
    struct.pack_into("<H", exif, 10, 0x5741)
    struct.pack_into("<H", exif, 12, 7)
    struct.pack_into("<I", exif, 14, len(payload))
    struct.pack_into("<I", exif, 18, 26)
    struct.pack_into("<I", exif, 22, 0)
    exif[26:] = payload
    return bytes(exif)


def _read_chunks(raw: bytes) -> list[tuple[str, bytes]]:
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8
    out = []
    off = 12
    while off + 8 <= len(raw):
        fourcc = raw[off:off + 4].decode("ascii")
        size = struct.unpack("<I", raw[off + 4:off + 8])[0]
        out.append((fourcc, raw[off + 8:off + 8 + size]))
        off += 8 + size + (size % 2)
    assert off == len(raw), "trailing bytes after the last chunk"
    return out


def _exif_json(raw: bytes) -> dict:
    for fourcc, data in _read_chunks(raw):
        if fourcc == "EXIF":
            assert data[:2] == b"II"
            assert struct.unpack("<H", data[2:4])[0] == 0x2A
            assert struct.unpack("<I", data[4:8])[0] == 8
            assert struct.unpack("<H", data[8:10])[0] == 1
            assert struct.unpack("<H", data[10:12])[0] == 0x5741
            assert struct.unpack("<H", data[12:14])[0] == 7
            count = struct.unpack("<I", data[14:18])[0]
            assert struct.unpack("<I", data[18:22])[0] == 26
            assert struct.unpack("<I", data[22:26])[0] == 0
            return json.loads(data[26:26 + count])
    raise AssertionError("no EXIF chunk in the WebP")


# ── EXIF builder + injection ─────────────────────────────────────────


def test_build_sound_sha256_exif_layout():
    from nomorals.social.chat import media_convert

    hex_digest = "ab" * 32
    exif = media_convert.build_sound_sha256_exif(hex_digest)
    payload = json.dumps({"sound-sha256": hex_digest}).encode()
    assert len(exif) == 26 + len(payload)
    assert exif[:2] == b"II"
    assert struct.unpack("<H", exif[2:4])[0] == 0x2A
    assert struct.unpack("<I", exif[4:8])[0] == 8
    assert struct.unpack("<H", exif[8:10])[0] == 1
    assert struct.unpack("<H", exif[10:12])[0] == 0x5741
    assert struct.unpack("<H", exif[12:14])[0] == 7
    assert struct.unpack("<I", exif[14:18])[0] == len(payload)
    assert struct.unpack("<I", exif[18:22])[0] == 26
    assert struct.unpack("<I", exif[22:26])[0] == 0
    assert json.loads(exif[26:]) == {"sound-sha256": hex_digest}


def test_build_sound_sha256_exif_rejects_bad_hex():
    from nomorals.social.chat import media_convert

    with pytest.raises(media_convert.ConversionError):
        media_convert.build_sound_sha256_exif("not-a-sha256")


def test_exif_injection_into_plain_vp8l_synthesises_vp8x(tmp_path):
    from nomorals.social.chat import media_convert

    hex_digest = "cd" * 32
    src = tmp_path / "still.webp"
    src.write_bytes(_webp_bytes([(b"VP8L", _vp8l_data())]))
    out = media_convert.add_sound_sha256_to_webp(src, hex_digest)
    assert out == src
    raw = out.read_bytes()
    chunks = dict(_read_chunks(raw))
    assert _exif_json(raw) == {"sound-sha256": hex_digest}
    # The synthesised VP8X carries the SPEC EXIF bit (0x08) — not
    # the fork's 0x20, which is the ICC-profile bit (RFC 9649).
    assert chunks["VP8X"][0] & 0x08
    assert not chunks["VP8X"][0] & 0x20
    assert struct.unpack("<I", chunks["VP8X"][4:7] + b"\x00")[0] == 511
    assert struct.unpack("<I", chunks["VP8X"][7:10] + b"\x00")[0] == 511
    assert chunks["VP8L"] == _vp8l_data()


def test_exif_injection_keeps_vp8x_and_replaces_old_exif(tmp_path):
    from nomorals.social.chat import media_convert

    old_hex, new_hex = "11" * 32, "22" * 32
    src = tmp_path / "anim.webp"
    src.write_bytes(_webp_bytes([
        (b"VP8X", _vp8x_data(flags=0x02)),  # animated, no EXIF yet
        (b"VP8L", _vp8l_data()),
        (b"EXIF", _exif_chunk_for(old_hex)),
    ]))
    media_convert.add_sound_sha256_to_webp(src, new_hex)
    raw = src.read_bytes()
    chunks = _read_chunks(raw)
    assert [c for c, _ in chunks].count("EXIF") == 1
    assert _exif_json(raw) == {"sound-sha256": new_hex}
    vp8x = dict(chunks)["VP8X"]
    assert vp8x[0] & 0x02  # animation flag preserved
    assert vp8x[0] & 0x08  # EXIF flag set
    assert not vp8x[0] & 0x20  # no ICC claim
    assert dict(chunks)["VP8L"] == _vp8l_data()


def test_exif_injection_rejects_non_webp_and_malformed(tmp_path):
    from nomorals.social.chat import media_convert

    bad = tmp_path / "bad.webp"
    bad.write_bytes(b"not a webp at all")
    with pytest.raises(media_convert.ConversionError):
        media_convert.add_sound_sha256_to_webp(bad, "ab" * 32)
    overrun = tmp_path / "overrun.webp"
    overrun.write_bytes(_webp_bytes([(b"VP8L", _vp8l_data())])[:-2]
                        + b"\x00\x00")
    # A chunk header claiming more bytes than the file holds.
    raw = bytearray(_webp_bytes([(b"VP8L", _vp8l_data())]))
    struct.pack_into("<I", raw, 16, 9999)
    overrun.write_bytes(bytes(raw))
    with pytest.raises(media_convert.ConversionError):
        media_convert.add_sound_sha256_to_webp(overrun, "ab" * 32)
    good = tmp_path / "good.webp"
    good.write_bytes(_webp_bytes([(b"VP8L", _vp8l_data())]))
    with pytest.raises(media_convert.ConversionError):
        media_convert.add_sound_sha256_to_webp(good, "xyz")


# ── maker: fake-runner argv shape + SoundSticker ─────────────────────


def _fake_runner(calls, webp=None, ogg=b"OggS" + b"\x00" * 100):
    def run(args, **kw):
        calls.append(list(args))
        dest = Path(args[-1])
        if dest.suffix == ".webp":
            dest.write_bytes(webp if webp is not None else
                             _webp_bytes([(b"VP8L", _vp8l_data())]))
        elif dest.suffix == ".ogg":
            dest.write_bytes(ogg)
        else:
            dest.write_bytes(b"converted")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return run


def _probe_seconds(monkeypatch, seconds):
    monkeypatch.setattr(
        "nomorals.media_edit.videos.video_probe",
        lambda p: {"duration": seconds,
                   "streams": [{"type": "audio"}]})


def test_video_to_sticker_sound_keeps_video_audio(tmp_path, monkeypatch):
    from nomorals.social.chat import media_convert

    _probe_seconds(monkeypatch, 4.5)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"mp4")
    calls: list = []
    out = media_convert.video_to_sticker(
        src, out_dir=tmp_path, runner=_fake_runner(calls),
        format="webp", audio="original")
    assert isinstance(out, media_convert.SoundSticker)
    assert out.sticker.suffix == ".webp"
    assert out.audio.suffix == ".ogg"
    assert out.audio_seconds == 4.5
    assert out.audio_sha256 == hashlib.sha256(
        out.audio.read_bytes()).hexdigest()
    assert _exif_json(out.sticker.read_bytes()) == {
        "sound-sha256": out.audio_sha256}
    # The sticker video runs the sound window (5s), not the
    # silent 3s; the audio is OGG Opus, video dropped.
    sticker_argv = calls[0]
    assert "-t" in sticker_argv and "5" in sticker_argv
    assert "3" not in sticker_argv[sticker_argv.index("-t") + 1:
                                   sticker_argv.index("-t") + 2]
    audio_argv = calls[1]
    assert "-vn" in audio_argv and "libopus" in audio_argv
    assert "-t" in audio_argv and "5" in audio_argv
    assert audio_argv[-1].endswith(".ogg")


def test_video_to_sticker_sound_rejects_webm_and_silent_audio(tmp_path):
    from nomorals.social.chat import media_convert

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"mp4")
    with pytest.raises(media_convert.ConversionError):
        media_convert.video_to_sticker(
            src, out_dir=tmp_path, runner=_fake_runner([]),
            format="webm", audio="original")


def test_video_to_sticker_no_audio_is_still_a_plain_path(tmp_path):
    from nomorals.social.chat import media_convert

    src = tmp_path / "clip.mp4"
    src.write_bytes(b"mp4")
    calls: list = []
    out = media_convert.video_to_sticker(
        src, out_dir=tmp_path, runner=_fake_runner(calls),
        format="webp")
    assert isinstance(out, Path) and out.suffix == ".webp"
    assert "-t" in calls[0] and "3" in calls[0]
    assert "libopus" not in calls[0]


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg binary absent on this machine")
def test_image_to_sticker_sound_trims_to_ten_seconds(tmp_path):
    from nomorals.media_edit.videos import video_probe
    from nomorals.social.chat import media_convert

    voice = tmp_path / "note.ogg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:a", "libopus", str(voice)],
        check=True, timeout=60)
    img = tmp_path / "photo.png"
    from PIL import Image
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(img, "PNG")
    out = media_convert.image_to_sticker(
        img, out_dir=tmp_path, audio=voice)
    assert isinstance(out, media_convert.SoundSticker)
    assert out.sticker.suffix == ".webp"
    # A 12s voice note is trimmed to the photo-origin limit.
    assert out.audio_seconds <= 10.05
    assert video_probe(out.audio)["duration"] <= 10.05
    assert _exif_json(out.sticker.read_bytes()) == {
        "sound-sha256": out.audio_sha256}


def test_image_to_sticker_sound_rejects_original_audio(tmp_path):
    from nomorals.social.chat import media_convert

    img = tmp_path / "photo.png"
    from PIL import Image
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(img, "PNG")
    with pytest.raises(media_convert.ConversionError):
        media_convert.image_to_sticker(
            img, out_dir=tmp_path, audio="original")


# ── maker: real ffmpeg end-to-end ─────────────────────────────────────


def _make_test_video_with_audio(tmp_path: Path, seconds: float) -> Path:
    """A real video WITH an audio track (testsrc + sine)."""
    vid = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i",
         f"testsrc=duration={seconds}:size=320x240:rate=10",
         "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(vid)],
        check=True, timeout=60)
    return vid


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg binary absent on this machine")
def test_video_to_sticker_sound_real_trim_to_five(tmp_path):
    from nomorals.media_edit.videos import video_probe
    from nomorals.social.chat import media_convert

    vid = _make_test_video_with_audio(tmp_path, 8)
    out = media_convert.video_to_sticker(
        vid, out_dir=tmp_path, format="webp", audio="original")
    assert isinstance(out, media_convert.SoundSticker)
    raw = out.sticker.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    assert b"ANIM" in raw, "the sound sticker keeps the animation"
    assert out.audio_seconds <= 5.05
    assert video_probe(out.audio)["duration"] <= 5.05
    assert video_probe(out.audio)["format"] in (
        "ogg", "application/ogg", "opus")
    assert out.audio_sha256 == hashlib.sha256(
        out.audio.read_bytes()).hexdigest()
    assert _exif_json(raw) == {"sound-sha256": out.audio_sha256}


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg binary absent on this machine")
def test_video_to_sticker_sound_refuses_silent_video(tmp_path):
    from nomorals.social.chat import media_convert

    vid = tmp_path / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)],
        check=True, timeout=60)
    with pytest.raises(media_convert.ConversionError,
                       match="no audio track"):
        media_convert.video_to_sticker(
            vid, out_dir=tmp_path, format="webp", audio="original")


# ── adapter: the send verb carries the audio ─────────────────────────


def _make_adapter(tmp_path, connected=True):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter(outbox_dir=str(tmp_path / "outbox"))
    if connected:
        adapter.connected.set()
    return adapter


def _chat():
    return ChatRef(platform="whatsapp", chat_id="2348012345678",
                   kind=ChatKind.DM)


def test_adapter_send_media_carries_sound_audio(tmp_path):
    adapter = _make_adapter(tmp_path)
    seen: list = []
    adapter._send_cmd = lambda cmd, timeout=60: seen.append(cmd) or {
        "ok": True, "message_id": "9"}
    ref = MediaRef(path="/tmp/sticker.webp", kind="sticker",
                   name="sticker.webp",
                   audio_path="/tmp/sound.ogg", audio_seconds=4.5)
    result = adapter.send_media(_chat(), ref)
    assert result.ok
    assert len(seen) == 1
    cmd = seen[0]
    assert cmd["cmd"] == "send_media"
    assert cmd["media_kind"] == "sticker"
    assert cmd["audio_path"] == "/tmp/sound.ogg"
    assert cmd["audio_seconds"] == 4.5


def test_adapter_send_media_silent_sticker_sends_no_audio(tmp_path):
    adapter = _make_adapter(tmp_path)
    seen: list = []
    adapter._send_cmd = lambda cmd, timeout=60: seen.append(cmd) or {
        "ok": True, "message_id": "9"}
    ref = MediaRef(path="/tmp/sticker.webp", kind="sticker",
                   name="sticker.webp")
    assert adapter.send_media(_chat(), ref).ok
    assert "audio_path" not in seen[0]
    assert "audio_seconds" not in seen[0]


def test_adapter_send_media_refuses_sound_on_non_sticker(tmp_path):
    adapter = _make_adapter(tmp_path)
    ref = MediaRef(path="/tmp/pic.jpg", kind="image",
                   audio_path="/tmp/sound.ogg", audio_seconds=4.5)
    result = adapter.send_media(_chat(), ref)
    assert not result.ok
    assert "sticker" in (result.error or "")


def test_adapter_outbox_round_trips_sound_audio(tmp_path):
    adapter = _make_adapter(tmp_path, connected=False)
    ref = MediaRef(path="/tmp/sticker.webp", kind="sticker",
                   name="sticker.webp",
                   audio_path="/tmp/sound.ogg", audio_seconds=4.5)
    result = adapter.send_media(_chat(), ref)
    assert not result.ok and "queued" in (result.error or "")
    queued = list((tmp_path / "outbox").glob("*.jsonl"))
    assert queued, "the outbox file was not written"
    entries = [json.loads(line) for line in
               queued[0].read_text().splitlines() if line.strip()]
    assert entries and entries[0].get("kind") == "media"
    assert entries[0].get("audio_path") == "/tmp/sound.ogg"
    assert entries[0].get("audio_seconds") == 4.5
    # The replay rebuilds the ref with the audio attached.
    seen: list = []
    adapter.connected.set()
    adapter._send_cmd = lambda cmd, timeout=60: seen.append(cmd) or {
        "ok": True, "message_id": "9"}
    assert adapter._deliver_entry(entries[0])
    assert seen[0].get("audio_path") == "/tmp/sound.ogg"


def test_inbound_sticker_reports_sound_flag():
    from nomorals.social.chat.whatsapp import _wa_envelope

    obj = {"id": "msg1", "media": [{
        "kind": "sticker", "mime": "image/webp",
        "size": 1234, "animated": True, "sound": True}]}
    env = _wa_envelope(obj, "")
    assert env["sticker"]["sound"] is True
    obj["media"][0].pop("sound")
    env = _wa_envelope(obj, "")
    assert env["sticker"]["sound"] is False


# ── command: /sticker sound ─────────────────────────────────────────


class _Chat:
    def __init__(self, platform, key, kind=ChatKind.DM):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Tlayo"
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

    def send_media(self, chat, ref, **kw):
        self.media_sent.append(ref)
        return SimpleNamespace(ok=True, message_id="9", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters

    def send(self, platform, chat, text, **kw):
        return SimpleNamespace(ok=True, message_id="1")


def _rig(tmp_path, platform):
    from nomorals.agents.partner.runtime_expression import RuntimeExpressionMixin
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin

    class _Runtime(RuntimeExpressionMixin, RuntimeGroupMixin,
                   RuntimeGamesMixin):
        def __init__(self, tmp_path, adapters, operators=()):
            self.gateway = _Gateway(adapters)
            self.operators = set(operators)
            self.context = SimpleNamespace(settings=SimpleNamespace(
                workspace_dir=str(tmp_path)))

        def _is_operator(self, message):
            return str(message.sender_id or "") in self.operators

    adapter = _Adapter()
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    chat = _Chat(platform, f"{platform}:123")
    return rt, adapter, chat


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg binary absent on this machine")
def test_control_sticker_sound_video_keeps_audio(tmp_path):
    rt, adapter, chat = _rig(tmp_path, "whatsapp")
    vid = _make_test_video_with_audio(tmp_path, 2)
    msg = _Msg(chat, meta={"replied_media_local": {
        "path": str(vid), "kind": "video"}})
    assert rt._control_sticker("sound", msg) == ""
    assert len(adapter.media_sent) == 1
    ref = adapter.media_sent[0]
    assert ref.kind == "sticker"
    assert ref.path.endswith(".webp")
    assert ref.audio_path.endswith(".ogg")
    assert 0 < ref.audio_seconds <= 2.05
    assert _exif_json(Path(ref.path).read_bytes()) == {
        "sound-sha256": hashlib.sha256(
            Path(ref.audio_path).read_bytes()).hexdigest()}


@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg binary absent on this machine")
def test_control_sticker_sound_photo_takes_voice_note(tmp_path):
    rt, adapter, chat = _rig(tmp_path, "whatsapp")
    voice = tmp_path / "note.ogg"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:a", "libopus", str(voice)],
        check=True, timeout=60)
    img = tmp_path / "photo.png"
    from PIL import Image
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(img, "PNG")
    msg = _Msg(
        chat,
        meta={"replied_media_local": {
            "path": str(img), "kind": "image"}},
        media=[SimpleNamespace(path=str(voice), kind="voice",
                               file_id="")])
    assert rt._control_sticker("sound", msg) == ""
    assert len(adapter.media_sent) == 1
    ref = adapter.media_sent[0]
    assert ref.kind == "sticker"
    assert ref.audio_path.endswith(".ogg")
    assert 0 < ref.audio_seconds <= 3.05


def test_control_sticker_sound_photo_without_audio_is_honest(tmp_path):
    rt, adapter, chat = _rig(tmp_path, "whatsapp")
    img = tmp_path / "photo.png"
    from PIL import Image
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(img, "PNG")
    msg = _Msg(chat, meta={"replied_media_local": {
        "path": str(img), "kind": "image"}})
    result = rt._control_sticker("sound", msg)
    assert adapter.media_sent == []
    assert "voice note" in result


def test_control_sticker_sound_refuses_non_whatsapp(tmp_path):
    rt, adapter, chat = _rig(tmp_path, "telegram")
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"mp4")
    msg = _Msg(chat, meta={"replied_media_local": {
        "path": str(vid), "kind": "video"}})
    result = rt._control_sticker("sound", msg)
    assert adapter.media_sent == []
    assert "WhatsApp" in result


def test_control_sticker_sound_without_media_is_a_text_sticker(
        tmp_path, monkeypatch):
    from nomorals.social.chat import media_convert

    rt, adapter, chat = _rig(tmp_path, "whatsapp")
    raster = tmp_path / "sound.webp"
    raster.write_bytes(b"RIFF")
    monkeypatch.setattr(
        media_convert, "make_text_sticker",
        lambda text, out_dir="": {"spec": SimpleNamespace(id="s1"),
                                  "raster": str(raster), "note": ""})
    msg = _Msg(chat)
    assert rt._control_sticker("sound", msg) == ""
    assert len(adapter.media_sent) == 1
    assert adapter.media_sent[0].path.endswith(".webp")
