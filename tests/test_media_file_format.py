"""Downloaded files land in their real format — never ``.bin``.

The owner's live report (2026-11-15): TikTok videos arrive on his phone
as ``.bin`` files. Root cause: the download chain's ``_safe_name``
appended ``.bin`` to every extensionless media URL (TikTok CDN URLs
routinely carry no extension), the shared magic sniffer could not see
MP4 at all (``ftyp`` sits at offset 4, not 0), and the send path
trusted the on-disk name. These tests pin the fix at every layer:
magic sniffing, safe naming, the landing normalizer, the send seam,
and the artifact export path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.decoder import (
    extension_for_magic,
    identify_magic,
)
from nomorals.media.downloader import (
    MediaDownloader,
    _safe_name,
)
from nomorals.media.platforms import (
    ensure_media_extension,
    sniff_media_ext,
)

# ── synthetic media bytes ─────────────────────────────────────────────

MP4_BYTES = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 100
M4A_BYTES = b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00M4A mp42" + b"\x00" * 100
MOV_BYTES = b"\x00\x00\x00\x18ftypqt  \x00\x00\x00\x00qt  " + b"\x00" * 100
WEBM_BYTES = b"\x1a\x45\xdf\xa3" + b"\x00" * 60
AVI_BYTES = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 40
WAV_BYTES = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40
TS_BYTES = b"\x47" + b"\x00" * 187 + b"\x47" + b"\x00" * 100
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
MP3_BYTES = b"ID3\x04\x00\x00" + b"\x00" * 40
OGG_BYTES = b"OggS" + b"\x00" * 40
FLAC_BYTES = b"fLaC" + b"\x00" * 40
UNKNOWN_BYTES = b"\x12\x34\x56\x78" * 32

#: TikTok CDN media URLs with no extension in the path — the shape
#: that used to mint ``.bin`` names.
TIKTOK_CDN_NO_EXT = (
    "https://v16-webapp-prime.tiktok.com/video/tos/useast2a/"
    "tos-useast2a-pv-0037/abc123def456/?a=1&b=2",
    "https://v19-webapp-prime.tiktok.com/video/tos/useast2a/xyz789",
    "https://api16-normal-c-useast2a.tiktokv.com/aweme/v1/play/"
    "?video_id=v123&line=0",
)


# ── 1 · the core sniffer sees the containers ─────────────────────────


class TestIdentifyMagic:
    def test_mp4_family_by_brand(self):
        assert identify_magic(MP4_BYTES) == "mp4"
        assert identify_magic(M4A_BYTES) == "m4a"
        assert identify_magic(MOV_BYTES) == "mov"

    def test_ebml_avi_wav_ts(self):
        assert identify_magic(WEBM_BYTES) == "webm/mkv"
        assert identify_magic(AVI_BYTES) == "riff/avi"
        assert identify_magic(WAV_BYTES) == "riff/wav"
        assert identify_magic(TS_BYTES) == "mpegts"

    def test_known_signatures_unchanged(self):
        assert identify_magic(PNG_BYTES) == "png"
        assert identify_magic(MP3_BYTES) == "mp3-id3"
        assert identify_magic(OGG_BYTES) == "ogg"
        assert identify_magic(FLAC_BYTES) == "flac"

    def test_unknown_and_empty(self):
        assert identify_magic(UNKNOWN_BYTES) == ""
        assert identify_magic(b"") == ""
        assert identify_magic(b"\x00" * 32) == ""


class TestExtensionForMagic:
    def test_media_labels(self):
        assert extension_for_magic("mp4") == ".mp4"
        assert extension_for_magic("m4a") == ".m4a"
        assert extension_for_magic("mov") == ".mov"
        assert extension_for_magic("riff/avi") == ".avi"
        assert extension_for_magic("riff/wav") == ".wav"
        assert extension_for_magic("mpegts") == ".ts"
        assert extension_for_magic("webm/mkv") == ".webm"
        assert extension_for_magic("mp3-id3") == ".mp3"

    def test_unknown_is_empty_not_bin(self):
        assert extension_for_magic("nope") == ""
        assert extension_for_magic("") == ""


# ── 2 · safe naming never invents .bin ──────────────────────────────


class TestSafeName:
    def test_extensionless_tiktok_cdn_urls_never_bin(self):
        for url in TIKTOK_CDN_NO_EXT:
            name = _safe_name(url, False)
            assert not name.endswith(".bin"), url
            assert Path(name).suffix == "", url

    def test_audio_only_also_never_bin(self):
        for url in TIKTOK_CDN_NO_EXT:
            assert not _safe_name(url, True).endswith(".bin")

    def test_url_extension_kept(self):
        url = ("https://v16-webapp-prime.tiktok.com/video/tos/useast2a/"
               "tos-useast2a-pv-0037/clip.mp4?x=1")
        assert _safe_name(url, False) == "clip.mp4"

    def test_bare_host_url_never_bin(self):
        assert not _safe_name("https://host/", False).endswith(".bin")


# ── 3 · sniff + repair on real files ────────────────────────────────


class TestSniffMediaExt:
    def test_sniffs_real_formats(self, tmp_path):
        cases = ((MP4_BYTES, ".mp4"), (M4A_BYTES, ".m4a"),
                 (MOV_BYTES, ".mov"), (WEBM_BYTES, ".webm"),
                 (MP3_BYTES, ".mp3"), (PNG_BYTES, ".png"),
                 (UNKNOWN_BYTES, ""))
        for i, (data, want) in enumerate(cases):
            f = tmp_path / f"f{i}.bin"
            f.write_bytes(data)
            assert sniff_media_ext(f) == want


class TestEnsureMediaExtension:
    def test_bin_mp4_renamed_bytes_intact(self, tmp_path):
        f = tmp_path / "clip.bin"
        f.write_bytes(MP4_BYTES)
        fixed = ensure_media_extension(f)
        assert fixed == str(tmp_path / "clip.mp4")
        assert not f.exists()
        assert Path(fixed).read_bytes() == MP4_BYTES

    def test_extensionless_mp3_renamed(self, tmp_path):
        f = tmp_path / "song"
        f.write_bytes(MP3_BYTES)
        assert ensure_media_extension(f) == str(tmp_path / "song.mp3")

    def test_unknown_bytes_kind_fallback(self, tmp_path):
        f = tmp_path / "v.bin"
        f.write_bytes(UNKNOWN_BYTES)
        assert ensure_media_extension(f) == str(tmp_path / "v.mp4")
        f2 = tmp_path / "a.bin"
        f2.write_bytes(UNKNOWN_BYTES)
        assert ensure_media_extension(f2, audio_only=True) == \
            str(tmp_path / "a.mp3")

    def test_unknown_bytes_no_fallback_stays_put(self, tmp_path):
        f = tmp_path / "mystery.bin"
        f.write_bytes(UNKNOWN_BYTES)
        assert ensure_media_extension(f, allow_kind_fallback=False) == \
            str(f)
        assert f.exists()

    def test_known_suffix_never_touched(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(UNKNOWN_BYTES)
        assert ensure_media_extension(f) == str(f)

    def test_missing_file_returned_unchanged(self, tmp_path):
        missing = tmp_path / "nope.bin"
        assert ensure_media_extension(missing) == str(missing)

    def test_collision_gets_numbered_suffix(self, tmp_path):
        taken = tmp_path / "clip.mp4"
        taken.write_bytes(MP4_BYTES)
        f = tmp_path / "clip.bin"
        f.write_bytes(MP4_BYTES)
        fixed = ensure_media_extension(f)
        assert fixed == str(tmp_path / "clip-2.mp4")


# ── 4 · the landing normalizer: full download() chain ───────────────


class _FakeRegistry:
    def for_url(self, url):
        return None

    def note_candidate(self, url, via=""):
        return None

    def learned_for_url(self, url):
        return None

    def note_rule_failure(self, desc):
        return None

    def record_outcome(self, desc, ok, latency_s=0.0, error_class=""):
        return None

    def learn_rule(self, desc, **kw):
        return None

    def bind_store(self, store):
        return None


def _stub_direct(md, data, dest_name="clip.bin"):
    """Stage stub: land *data* under a .bin name, claim success."""
    def _stage(report, url, target_dir, audio_only, format_spec,
               timeout, budget):
        dest = target_dir / dest_name
        dest.write_bytes(data)
        MediaDownloader._fill_success(
            report, {"path": str(dest), "bytes": len(data),
                     "title": "t", "extractor": "direct"})
        return True

    return _stage


class TestDownloadLandingNormalizer:
    def test_tiktok_video_lands_mp4(self, tmp_path, monkeypatch):
        md = MediaDownloader(context=None,
                             source_registry=_FakeRegistry())
        monkeypatch.setattr(md, "_stage_direct",
                            _stub_direct(md, MP4_BYTES))
        report = md.download("https://www.tiktok.com/@x/video/123",
                             dest_dir=str(tmp_path))
        assert report.ok is True
        assert report.path.endswith(".mp4"), report.path
        assert Path(report.path).read_bytes() == MP4_BYTES

    def test_unidentifiable_video_falls_back_mp4(self, tmp_path,
                                                 monkeypatch):
        md = MediaDownloader(context=None,
                             source_registry=_FakeRegistry())
        monkeypatch.setattr(md, "_stage_direct",
                            _stub_direct(md, UNKNOWN_BYTES))
        report = md.download("https://www.tiktok.com/@x/video/123",
                             dest_dir=str(tmp_path))
        assert report.ok is True
        assert report.path.endswith(".mp4"), report.path

    def test_unidentifiable_audio_falls_back_mp3(self, tmp_path,
                                                 monkeypatch):
        md = MediaDownloader(context=None,
                             source_registry=_FakeRegistry())
        monkeypatch.setattr(md, "_stage_direct",
                            _stub_direct(md, UNKNOWN_BYTES))
        report = md.download("https://www.tiktok.com/@x/video/123",
                             dest_dir=str(tmp_path), audio_only=True)
        assert report.ok is True
        assert report.path.endswith(".mp3"), report.path


# ── 5 · the send seam repairs a .bin before it goes out ─────────────


def _composed_runtime():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-tbin-", dir=os.environ.get("NM_TEST_TMPDIR") or None)
    settings = load_settings(overrides={
        "home": tmp.name, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false"})
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context), tmp


class _KindGateway:
    def __init__(self):
        self.calls = []

    def send_file(self, platform, chat, path, **kwargs):
        self.calls.append((path, kwargs))
        return SimpleNamespace(ok=True)


class TestSendPlatformFile:
    def test_bin_video_sent_as_mp4_video(self, tmp_path):
        rt, tmp = _composed_runtime()
        try:
            gateway = _KindGateway()
            rt.gateway = gateway
            chat = SimpleNamespace(platform="telegram-bot", chat_id="5")
            path = tmp_path / "clip.bin"
            path.write_bytes(MP4_BYTES)
            _note, sent = rt._send_platform_file(
                chat, str(path), "cap", 10)
            assert sent is True
            sent_path, kwargs = gateway.calls[0]
            assert sent_path.endswith(".mp4"), sent_path
            assert kwargs.get("kind") == "video"
            assert Path(sent_path).read_bytes() == MP4_BYTES
        finally:
            tmp.cleanup()

    def test_unidentifiable_bin_stays_a_document(self, tmp_path):
        # The send seam never guesses: an unknown file is still a
        # document, not a mislabelled video.
        rt, tmp = _composed_runtime()
        try:
            gateway = _KindGateway()
            rt.gateway = gateway
            chat = SimpleNamespace(platform="telegram-bot", chat_id="5")
            path = tmp_path / "mystery.bin"
            path.write_bytes(UNKNOWN_BYTES)
            _note, sent = rt._send_platform_file(
                chat, str(path), "cap", 10)
            assert sent is True
            sent_path, kwargs = gateway.calls[0]
            assert sent_path == str(path)
            assert kwargs.get("kind") == ""
        finally:
            tmp.cleanup()


# ── 6 · artifact export never mints .bin ────────────────────────────


class TestArtifactExportPath:
    def _runtime(self):
        rt, tmp = _composed_runtime()
        return rt, tmp

    def test_octet_stream_mime_gets_no_bin(self):
        rt, tmp = self._runtime()
        try:
            art = SimpleNamespace(type="file", id="abc123",
                                  metadata={"filename": "clip"},
                                  mime="application/octet-stream")
            path = rt._artifact_export_path(None, art)
            assert not str(path).endswith(".bin"), path
            assert Path(path).name == "clip"
        finally:
            tmp.cleanup()

    def test_known_mime_keeps_its_extension(self):
        rt, tmp = self._runtime()
        try:
            art = SimpleNamespace(type="file", id="abc123",
                                  metadata={"filename": "clip"},
                                  mime="video/mp4")
            path = rt._artifact_export_path(None, art)
            assert str(path).endswith("clip.mp4"), path
        finally:
            tmp.cleanup()

    def test_named_extension_kept(self):
        rt, tmp = self._runtime()
        try:
            art = SimpleNamespace(type="file", id="abc123",
                                  metadata={"filename": "song.mp3"},
                                  mime="")
            path = rt._artifact_export_path(None, art)
            assert str(path).endswith("song.mp3"), path
        finally:
            tmp.cleanup()
