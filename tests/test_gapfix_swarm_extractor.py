"""The download swarm engages for extractor URLs too.

Pins (all against fakes — no network, no yt-dlp on this box):

* the direct stage tries the swarm FIRST for an extractor URL when
  ``swarm`` is opted in: the page URL resolves to its single file
  URL through the existing yt-dlp extract-only seam, and that file
  URL downloads through the SAME ``_attempt_swarm`` the kit stage
  uses (same chunking, same stage log, same journaling);
* without the opt-in nothing changes: the swarm is never called
  and the yt-dlp direct download runs exactly as before;
* a resolver that finds nothing fails HONESTLY (the reason is named
  in the swarm stage log) and the chain falls through to the direct
  stage — never silence, never a swallowed attempt;
* a swarm refusal for the resolved URL falls through the same way;
* the resolver itself: one progressive URL is picked; a playlist,
  separate video+audio streams needing a merge, and segmented
  (HLS/DASH) playlists are refused with named reasons; a missing
  yt-dlp is an honest dependency error, never a fabricated URL.
"""
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.media import download_swarm as ds
from nomorals.media.download_kit import (
    ExtractorDescriptor,
    ExtractorRegistry,
)
from nomorals.media.downloader import DownloadReport, MediaDownloader
from nomorals.tools.media import MediaError, resolve_file_url

PAGE_URL = "https://www.youtube.com/watch?v=swarmtest123"
FILE_URL = "https://cdn.example.org/files/clip.mp4"
DATA = os.urandom(10 << 20)


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _ctx(tmp_path: Path):
    return SimpleNamespace(
        db=None,
        settings=SimpleNamespace(home_path=tmp_path,
                                 workspace_dir=str(tmp_path)),
        extras={}, tools=None)


def _ytdlp_registry() -> ExtractorRegistry:
    return ExtractorRegistry(descriptors=[
        ExtractorDescriptor(
            name="youtube", url_patterns=(r"youtube\.com",),
            extraction_rule="ytdlp")])


class _FakeSwarm:
    """Swarm-shaped stand-in (the test_download_swarm.py shape)."""

    def __init__(self, data: bytes, *, refused_reason: str = ""):
        self.data = data
        self.refused_reason = refused_reason
        self.calls: list[str] = []

    def download(self, url: str, dest, **kw) -> ds.SwarmReport:
        self.calls.append(url)
        if self.refused_reason:
            return ds.SwarmReport(refused_reason=self.refused_reason)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(self.data)
        return ds.SwarmReport(ok=True, path=str(dest),
                              size_bytes=len(self.data),
                              segments_total=4, segments_done=4,
                              sha256=_sha(self.data))


def _fake_direct(data: bytes, calls: list):
    def _download(url, destination, **kw):
        calls.append(url)
        dest = Path(destination) / "clip.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"path": str(dest), "bytes": len(data),
                "title": "Clip", "extractor": "yt-dlp"}
    return _download


def _downloader(tmp_path, *, swarm=None, resolver=None):
    return MediaDownloader(
        _ctx(tmp_path), extractors=_ytdlp_registry(),
        swarm_downloader=swarm, file_url_resolver=resolver)


def _resolver_ok(calls: list):
    def _resolve(url, **kw):
        calls.append(url)
        return {"url": FILE_URL, "title": "Clip", "ext": "mp4",
                "extractor": "youtube"}
    return _resolve


def _resolver_fail(reason: str):
    def _resolve(url, **kw):
        raise MediaError(reason)
    return _resolve


# ── the direct stage: swarm-first for extractor URLs ─────────────────


def test_direct_stage_tries_swarm_for_extractor_url(tmp_path):
    fake = _FakeSwarm(DATA)
    resolve_calls: list[str] = []
    dl = _downloader(tmp_path, swarm=fake,
                     resolver=_resolver_ok(resolve_calls))
    report = DownloadReport(url=PAGE_URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    with mock.patch("nomorals.tools.media.download") as direct:
        assert dl._stage_direct(
            report, PAGE_URL, tmp_path, False,
            "bestvideo*+bestaudio/best", 1800.0, budget,
            swarm=True, swarm_min_bytes=1 << 20) is True
        direct.assert_not_called()
    assert resolve_calls == [PAGE_URL]
    assert fake.calls == [FILE_URL]
    assert report.ok and report.path != ""
    assert report.extractor == "download-swarm"
    assert report.title == "Clip"
    assert Path(report.path).read_bytes() == DATA
    assert any(s.stage == "swarm" and s.ok for s in report.stages)


def test_direct_stage_ignores_swarm_when_not_opted_in(tmp_path):
    fake = _FakeSwarm(DATA)
    resolve_calls: list[str] = []
    dl = _downloader(tmp_path, swarm=fake,
                     resolver=_resolver_ok(resolve_calls))
    direct_calls: list[str] = []
    report = DownloadReport(url=PAGE_URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    with mock.patch("nomorals.tools.media.download",
                    _fake_direct(DATA, direct_calls)):
        assert dl._stage_direct(
            report, PAGE_URL, tmp_path, False,
            "bestvideo*+bestaudio/best", 1800.0, budget) is True
    assert fake.calls == []
    assert resolve_calls == []
    assert direct_calls == [PAGE_URL]
    assert report.ok and report.extractor == "yt-dlp"


def test_swarm_extract_falls_back_when_resolver_finds_nothing(tmp_path):
    fake = _FakeSwarm(DATA)
    dl = _downloader(tmp_path, swarm=fake, resolver=_resolver_fail(
        "the extractor returned no file URL"))
    direct_calls: list[str] = []
    report = DownloadReport(url=PAGE_URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    with mock.patch("nomorals.tools.media.download",
                    _fake_direct(DATA, direct_calls)):
        assert dl._stage_direct(
            report, PAGE_URL, tmp_path, False,
            "bestvideo*+bestaudio/best", 1800.0, budget,
            swarm=True, swarm_min_bytes=1 << 20) is True
    assert fake.calls == []
    assert direct_calls == [PAGE_URL]
    assert report.ok and report.extractor == "yt-dlp"
    swarm_logs = [s for s in report.stages if s.stage == "swarm"]
    assert swarm_logs, "no swarm stage log for the failed resolve"
    assert "no file URL" in (swarm_logs[0].error or "")


def test_swarm_extract_falls_back_when_swarm_refuses(tmp_path):
    fake = _FakeSwarm(DATA, refused_reason=(
        "server does not serve byte ranges"))
    resolve_calls: list[str] = []
    dl = _downloader(tmp_path, swarm=fake,
                     resolver=_resolver_ok(resolve_calls))
    direct_calls: list[str] = []
    report = DownloadReport(url=PAGE_URL)
    budget = {"deadline": time.perf_counter() + 60.0}
    with mock.patch("nomorals.tools.media.download",
                    _fake_direct(DATA, direct_calls)):
        assert dl._stage_direct(
            report, PAGE_URL, tmp_path, False,
            "bestvideo*+bestaudio/best", 1800.0, budget,
            swarm=True, swarm_min_bytes=1 << 20) is True
    assert fake.calls == [FILE_URL]
    assert direct_calls == [PAGE_URL]
    assert report.ok and report.extractor == "yt-dlp"
    assert any(s.stage == "swarm" and not s.ok for s in report.stages)
    assert any(s.stage == "direct" and s.ok for s in report.stages)


def test_swarm_extract_respects_an_expired_budget(tmp_path):
    fake = _FakeSwarm(DATA)
    resolve_calls: list[str] = []
    dl = _downloader(tmp_path, swarm=fake,
                     resolver=_resolver_ok(resolve_calls))
    report = DownloadReport(url=PAGE_URL)
    budget = {"deadline": time.perf_counter() - 1.0}
    with mock.patch("nomorals.tools.media.download") as direct:
        assert dl._stage_direct(
            report, PAGE_URL, tmp_path, False,
            "bestvideo*+bestaudio/best", 1800.0, budget,
            swarm=True, swarm_min_bytes=1 << 20) is False
        direct.assert_not_called()
    assert resolve_calls == []
    assert fake.calls == []


def test_download_chain_reaches_swarm_for_extractor_url(tmp_path):
    fake = _FakeSwarm(DATA)
    resolve_calls: list[str] = []
    dl = _downloader(tmp_path, swarm=fake,
                     resolver=_resolver_ok(resolve_calls))
    direct_calls: list[str] = []
    with mock.patch("nomorals.tools.media.download",
                    _fake_direct(DATA, direct_calls)):
        report = dl.download(PAGE_URL, dest_dir=tmp_path, swarm=True,
                             swarm_min_bytes=1 << 20)
    assert report.ok, report.summary()
    assert report.extractor == "download-swarm"
    assert Path(report.path).read_bytes() == DATA
    assert resolve_calls == [PAGE_URL]
    assert fake.calls == [FILE_URL]
    assert direct_calls == []


def test_chain_with_old_signature_stage_direct_still_works(tmp_path):
    """Regression: the chain must not pass swarm kwargs to a
    _stage_direct with the pre-swarm fixed signature — a stub (or
    an overrider) that takes only the old seven args still works
    when the swarm was never opted in."""
    from nomorals.media.downloader import StageLog

    fake = _FakeSwarm(DATA)
    dl = _downloader(tmp_path, swarm=fake,
                     resolver=_resolver_ok([]))

    def _old_stage(report, url, target_dir, audio_only, format_spec,
                   timeout, budget):
        dest = Path(target_dir) / "clip.mp4"
        dest.write_bytes(DATA)
        dl._fill_success(report, {
            "path": str(dest), "bytes": len(DATA), "title": "Clip",
            "extractor": "yt-dlp"}, len(DATA))
        report.stages.append(StageLog(stage="direct", url=url, ok=True))
        return True

    dl._stage_direct = _old_stage
    report = dl.download(PAGE_URL, dest_dir=tmp_path)
    assert report.ok, report.summary()
    assert report.extractor == "yt-dlp"
    assert fake.calls == []


# ── the resolver: picking one file URL out of yt-dlp info ────────────


class _FakeYDL:
    """YoutubeDL-shaped stand-in: extract_info returns canned info."""

    def __init__(self, info: dict, calls: list):
        self._info = info
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        self._calls.append((url, download))
        return dict(self._info)


def _fake_yt_dlp(info: dict, calls: list):
    return SimpleNamespace(YoutubeDL=lambda opts: _FakeYDL(info, calls))


def test_resolver_picks_single_progressive_url(tmp_path):
    info = {"id": "swarmtest123", "title": "Clip",
            "extractor_key": "youtube", "ext": "mp4",
            "url": FILE_URL, "protocol": "https"}
    calls: list = []
    with mock.patch("nomorals.tools.media.load_optional",
                    return_value=_fake_yt_dlp(info, calls)):
        resolved = resolve_file_url(PAGE_URL)
    assert resolved["url"] == FILE_URL
    assert resolved["title"] == "Clip"
    assert resolved["ext"] == "mp4"
    assert resolved["extractor"] == "youtube"
    assert calls == [(PAGE_URL, False)]


def test_resolver_picks_the_single_requested_format(tmp_path):
    info = {"id": "swarmtest123", "title": "Clip",
            "extractor_key": "youtube", "ext": "mp4",
            "requested_formats": [
                {"format_id": "22", "ext": "mp4", "url": FILE_URL,
                 "protocol": "https"}]}
    with mock.patch("nomorals.tools.media.load_optional",
                    return_value=_fake_yt_dlp(info, [])):
        resolved = resolve_file_url(PAGE_URL)
    assert resolved["url"] == FILE_URL


def test_resolver_refuses_merged_video_plus_audio(tmp_path):
    info = {"id": "swarmtest123", "title": "Clip",
            "extractor_key": "youtube",
            "requested_formats": [
                {"format_id": "137", "ext": "mp4",
                 "url": "https://cdn.example.org/files/video.mp4",
                 "protocol": "https"},
                {"format_id": "140", "ext": "m4a",
                 "url": "https://cdn.example.org/files/audio.m4a",
                 "protocol": "https"}]}
    with mock.patch("nomorals.tools.media.load_optional",
                    return_value=_fake_yt_dlp(info, [])):
        try:
            resolve_file_url(PAGE_URL)
        except MediaError as exc:
            assert "merg" in str(exc).lower()
        else:
            raise AssertionError("merged streams were not refused")


def test_resolver_refuses_playlist(tmp_path):
    info = {"id": "PLtest", "title": "A playlist",
            "extractor_key": "youtube",
            "entries": [{"id": "a"}, {"id": "b"}]}
    with mock.patch("nomorals.tools.media.load_optional",
                    return_value=_fake_yt_dlp(info, [])):
        try:
            resolve_file_url(PAGE_URL)
        except MediaError as exc:
            assert "playlist" in str(exc).lower()
        else:
            raise AssertionError("a playlist was not refused")


def test_resolver_refuses_hls_playlist_url(tmp_path):
    info = {"id": "swarmtest123", "title": "Clip",
            "extractor_key": "youtube", "ext": "mp4",
            "url": "https://cdn.example.org/files/hls.m3u8",
            "protocol": "m3u8_native"}
    with mock.patch("nomorals.tools.media.load_optional",
                    return_value=_fake_yt_dlp(info, [])):
        try:
            resolve_file_url(PAGE_URL)
        except MediaError as exc:
            assert "segmented" in str(exc).lower() or \
                "hls" in str(exc).lower()
        else:
            raise AssertionError("an HLS playlist URL was not refused")


def test_resolver_raises_when_yt_dlp_missing(tmp_path):
    with mock.patch("nomorals.tools.media.load_optional",
                    return_value=None), \
            mock.patch("nomorals.tools.media.which",
                       return_value=""):
        try:
            resolve_file_url(PAGE_URL)
        except MediaError as exc:
            assert "yt-dlp" in str(exc)
        else:
            raise AssertionError("a missing yt-dlp was not reported")
