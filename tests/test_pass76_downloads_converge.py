"""Pass #76 — downloads convergence.

Three convergences, each proven against the pre-pass behaviour:

(a) MediaDownloader reaches the download kit: a direct-media URL whose
    primary (yt-dlp) path fails is fetched by the kit's
    ResumeDownloader in the default configuration, and a successful
    download is recorded in the kit's IntegrityLedger.
(b) One source registry: extractor descriptors are a VIEW of
    media/source_registry.py's SourceRegistry — sources that existed
    only there (NotJustOk, TooXclusive) resolve with their true
    extraction rule, and kit-only sources (TikTok, Instagram, X) now
    live in the registry itself.
(c) cookies_status() tells the truth about the store the downloader
    actually uses at fetch time (user file + browser-profile export),
    including the present-but-stale case.
(d) A kit-path failure degrades to the chain's honest failure report —
    never a fake success.

All fetches are stubbed — no network.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest

from nomorals.media.download_kit import ResumeReport


# ── helpers ───────────────────────────────────────────────────────────


def _ctx(tmp: Path):
    ctx = mock.MagicMock()
    ctx.home = Path(tmp)
    ctx.db = mock.MagicMock()
    ctx.settings = None
    return ctx


def _fake_kit_success(calls: list):
    """A ResumeDownloader.download stand-in: writes real bytes."""

    def _download(self, url, dest, **kw):
        calls.append((url, str(dest)))
        data = b"ID3" + b"x" * 64
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
        import hashlib
        return ResumeReport(ok=True, path=str(dest), size_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest())

    return _download


def _fake_kit_failure(calls: list):
    def _download(self, url, dest, **kw):
        calls.append((url, str(dest)))
        return ResumeReport(error="fetch failed: boom (stubbed)")

    return _download


def _landed(url, dest, **kw):
    """tools.media.download stand-in that really lands a file."""
    p = Path(dest) / "song.mp3"
    p.write_bytes(b"ID3" + b"y" * 32)
    return {"url": url, "path": str(p), "bytes": p.stat().st_size,
            "title": "Song", "extractor": "fake", "seconds": 0.1}


# ── (a) the kit is wired into MediaDownloader ─────────────────────────


def test_direct_media_falls_through_to_kit_and_succeeds(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    dl = MediaDownloader(_ctx(tmp_path), direct_retries=1)
    kit_calls: list = []
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset by peer")), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _fake_kit_success(kit_calls)), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://cdn.example.test/track.mp3",
                             dest_dir=tmp_path)
    assert kit_calls, "the kit path was never reached"
    assert report.ok, report.summary()
    assert report.extractor == "download-kit"
    assert Path(report.path).is_file()
    assert "kit" in report.tried()


def test_success_is_recorded_in_integrity_ledger(tmp_path):
    from nomorals.media.download_kit import IntegrityLedger
    from nomorals.media.downloader import MediaDownloader

    ledger = IntegrityLedger(tmp_path / "ledger.jsonl")
    dl = MediaDownloader(_ctx(tmp_path), ledger=ledger)
    with mock.patch("nomorals.tools.media.download",
                    side_effect=_landed):
        report = dl.download("https://www.youtube.com/watch?v=abc",
                             dest_dir=tmp_path)
    assert report.ok, report.summary()
    ok, reason = ledger.verify(report.path, url=report.url)
    assert ok, reason


def test_non_downloadable_source_refused_before_any_fetch(tmp_path):
    """Spotify/Boomplay descriptors say 'never download' — the chain
    must refuse up front instead of burning every stage."""
    from nomorals.media.downloader import MediaDownloader

    dl = MediaDownloader(_ctx(tmp_path), direct_retries=1)
    with mock.patch("nomorals.tools.media.download") as md, \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://open.spotify.com/track/abc",
                             dest_dir=tmp_path)
    assert md.call_count == 0, "a fetch was attempted for a no-download source"
    assert not report.ok
    assert "download" in (report.likely_cause + report.hint).lower()


def test_quality_label_translated_via_descriptor(tmp_path):
    """A caller passing a quality LABEL gets the descriptor's yt-dlp
    format for the matched source, not the raw label as a format."""
    from nomorals.media.downloader import MediaDownloader

    dl = MediaDownloader(_ctx(tmp_path), direct_retries=1)
    seen: dict = {}

    def capture(url, dest, **kw):
        seen.update(kw)
        return _landed(url, dest, **kw)

    with mock.patch("nomorals.tools.media.download", side_effect=capture):
        report = dl.download("https://www.youtube.com/watch?v=abc",
                             dest_dir=tmp_path, format_spec="720p")
    assert report.ok, report.summary()
    assert seen.get("format_spec") == "best[height<=720]"


def test_disk_preflight_fails_before_any_fetch(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    dl = MediaDownloader(_ctx(tmp_path), direct_retries=1)
    with mock.patch("nomorals.tools.media.download") as md, \
            mock.patch("nomorals.media.download_kit.shutil.disk_usage",
                       return_value=(10**12, 10**12, 0)):
        report = dl.download("https://www.youtube.com/watch?v=abc",
                             dest_dir=tmp_path)
    assert md.call_count == 0
    assert not report.ok
    assert "disk" in report.summary().lower()


# ── (d) kit failure is honest ─────────────────────────────────────────


def test_kit_failure_degrades_to_honest_failure(tmp_path):
    from nomorals.media.downloader import MediaDownloader

    dl = MediaDownloader(_ctx(tmp_path), direct_retries=1)
    kit_calls: list = []
    with mock.patch("nomorals.tools.media.download",
                    side_effect=RuntimeError("connection reset by peer")), \
            mock.patch("nomorals.media.download_kit.ResumeDownloader.download",
                       _fake_kit_failure(kit_calls)), \
            mock.patch.object(MediaDownloader, "_stage_proxy",
                              return_value=False), \
            mock.patch.object(MediaDownloader, "_stage_browser",
                              return_value=False):
        report = dl.download("https://cdn.example.test/track.mp3",
                             dest_dir=tmp_path)
    assert kit_calls, "the kit path was never reached"
    assert not report.ok
    kit_logs = [s for s in report.stages if s.stage == "kit"]
    assert kit_logs and "boom" in kit_logs[0].error


# ── (b) one source registry ───────────────────────────────────────────


def test_registry_only_sources_resolve_with_true_rule():
    """NotJustOk / TooXclusive existed only in source_registry (and as
    scraper classes) — the kit sent them to the yt-dlp generic, which
    has no extractor for them.  As a registry view they now resolve
    as scrape sources."""
    from nomorals.media.download_kit import ExtractorRegistry

    reg = ExtractorRegistry()
    njok = reg.best_for("https://notjustok.com/download/some-song")
    assert njok is not None and njok.name == "notjustok"
    assert njok.extraction_rule == "scrape"
    tx = reg.best_for("https://tooxclusive.com/download/some-song")
    assert tx is not None and tx.name == "tooxclusive"
    assert tx.extraction_rule == "scrape"


def test_kit_only_sources_now_live_in_the_registry():
    from nomorals.media.source_registry import DEFAULT_REGISTRY

    domains = {d.domain for d in DEFAULT_REGISTRY.all()}
    for must in ("tiktok.com", "instagram.com", "x.com"):
        assert must in domains, f"{must} missing from the source registry"


def test_extractor_view_follows_the_registry_instance():
    """The view derives from the registry it is given: a source
    registered only in a custom registry resolves through it."""
    from nomorals.media.download_kit import ExtractorRegistry
    from nomorals.media.source_registry import (
        Licence,
        SourceDescriptor,
        SourceRegistry,
        TrustTier,
    )

    reg = SourceRegistry()
    reg.register(SourceDescriptor(
        nation="NG", language="en", domain="example.fm", name="ExampleFM",
        kinds=("music",), licence=Licence.USER_UPLOAD,
        trust_tier=TrustTier.COMMUNITY,
        extraction_rule="scrape",
        url_patterns=(r"example\.fm",),
        quality_map=(("audio", "bestaudio/best"),)))
    view = ExtractorRegistry(registry=reg)
    desc = view.best_for("https://example.fm/track/1")
    assert desc is not None and desc.name == "examplefm"
    assert desc.extraction_rule == "scrape"
    # The structural fallbacks still close the view.
    generic = view.best_for("https://unrelated.example.org/watch")
    assert generic is not None and generic.name == "ytdlp-generic"


def test_play_service_extractors_share_its_registry():
    from nomorals.media.play_service import PlayService

    svc = PlayService()
    desc = svc.extractors.best_for("https://notjustok.com/x")
    assert desc is not None and desc.extraction_rule == "scrape"


def test_old_kit_descriptor_set_still_resolves():
    """Every descriptor the kit used to hand-maintain still resolves
    with the same name/rule after the convergence."""
    from nomorals.media.download_kit import ExtractorRegistry

    reg = ExtractorRegistry()
    cases = {
        "https://www.youtube.com/watch?v=1": ("youtube", "ytdlp"),
        "https://soundcloud.com/a/b": ("soundcloud", "ytdlp"),
        "https://audiomack.com/a/song/b": ("audiomack", "ytdlp"),
        "https://thenetnaija.net/music/x": ("netnaija", "scrape"),
        "https://netnaija.com/music/x": ("netnaija", "scrape"),
        "https://boomplay.com/songs/1": ("boomplay", "none"),
        "https://open.spotify.com/track/1": ("spotify", "none"),
        "https://www.tiktok.com/@u/video/1": ("tiktok", "ytdlp"),
        "https://www.instagram.com/reel/1": ("instagram", "ytdlp"),
        "https://x.com/u/status/1": ("x-twitter", "ytdlp"),
        "https://cdn.example.com/a.mp3": ("direct-media", "direct"),
        "https://obscure.example.org/watch": ("ytdlp-generic", "ytdlp"),
    }
    for url, (name, rule) in cases.items():
        desc = reg.best_for(url)
        assert desc is not None, url
        assert desc.name == name, (url, desc.name)
        assert desc.extraction_rule == rule, (url, desc.extraction_rule)


# ── (c) cookie truth ──────────────────────────────────────────────────


def _write_session(home: Path, entries: list[dict]) -> Path:
    sessions = home / ".nomorals" / "data" / "browser" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    f = sessions / "s1.json"
    f.write_text(json.dumps(entries), encoding="utf-8")
    return f


def test_status_sees_browser_profile_cookies(tmp_path, monkeypatch):
    from nomorals.media import cookies as ck

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("NM_HOME", raising=False)
    _write_session(tmp_path, [{
        "name": "SID", "domain": ".youtube.com", "path": "/",
        "secure": True, "expires": 9999999999, "value": "x",
    }])
    st = ck.cookies_status()
    # The user-file keys keep their meaning…
    assert st["found"] is None
    # …and the status now reports the store downloads actually use.
    assert st["browser_export"], "browser-profile cookies invisible to status"
    assert st["browser_valid"] == 1
    assert st["browser_domains"] == ["youtube.com"]
    assert st["effective"]["source"] == "browser_profile"
    assert st["effective"]["path"] == st["browser_export"]
    assert st["stale"] is False
    # Agreement: status's effective file IS what a download resolves.
    assert ck.cookies_file_for("https://www.youtube.com/watch?v=1") \
        == st["effective"]["path"]


def test_status_flags_present_but_stale_cookies(tmp_path, monkeypatch):
    from nomorals.media import cookies as ck

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("NM_HOME", raising=False)
    _write_session(tmp_path, [{
        "name": "SID", "domain": ".youtube.com", "path": "/",
        "secure": True, "expires": 1000000000, "value": "x",
    }])
    st = ck.cookies_status()
    assert st["browser_export"]
    assert st["browser_valid"] == 0
    assert st["stale"] is True


def test_status_user_file_wins_and_is_not_stale(tmp_path, monkeypatch):
    from nomorals.media import cookies as ck

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("NM_HOME", raising=False)
    f = tmp_path / "cookies.txt"
    f.write_text("# Netscape HTTP Cookie File\n"
                 ".youtube.com\tTRUE\t/\tTRUE\t9999999999\tCONSENT\tYES\n",
                 encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES", str(f))
    st = ck.cookies_status()
    assert st["found"] == str(f)
    assert st["valid_youtube"] == 1
    assert st["effective"]["source"] == "user_file"
    assert st["effective"]["path"] == str(f)
    assert st["stale"] is False


def test_status_no_cookies_anywhere(tmp_path, monkeypatch):
    from nomorals.media import cookies as ck

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.delenv("NM_HOME", raising=False)
    st = ck.cookies_status()
    assert st["found"] is None
    assert st["browser_export"] is None
    assert st["effective"]["source"] == "none"
    assert st["stale"] is False
