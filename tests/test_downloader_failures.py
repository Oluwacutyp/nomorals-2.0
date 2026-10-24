"""Pass #116 pins: the downloader failure ledger records every converted site.

Ledger basics + bound + unprintable-exc safety + format render, the
direct-import surface, one pin per converted site (29), proof that the
two deliberate probe fall-throughs stay unrecorded, and a scan pin
asserting the subsystem remainder is exactly those two sites.
"""
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media import downloader as dl
from nomorals.media.downloader_failures import (
    Failure,
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def sites():
    return {f.site for f in recent_failures(500)}


class FakeDesc:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeRegistry:
    """No-op registry; individual methods replaced per pin."""

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


def md_with(registry=None, **kw):
    return dl.MediaDownloader(context=None, source_registry=registry, **kw)


# ── ledger basics ──────────────────────────────────────────────────

def test_ledger_basics_and_exports():
    assert record_failure is not None and Failure is not None
    record_failure("pin.site", ValueError("boom"))
    assert failure_count() == 1
    [entry] = recent_failures()
    assert entry.site == "pin.site" and entry.exc_type == "ValueError"
    assert entry.message == "boom" and entry.ts <= time.time()
    assert "pin.site: ValueError: boom" in format_failures()


def test_ledger_bound():
    for i in range(250):
        record_failure(f"pin.{i}", RuntimeError("x"))
    assert failure_count() == 250
    assert len(recent_failures(500)) == 200
    assert recent_failures(1)[0].site == "pin.249"


def test_ledger_unprintable_exc_and_empty():
    assert format_failures() == "no recorded downloader failures"

    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("pin.bad", BadStr())
    assert "unprintable" in format_failures()
    clear_failures()
    assert failure_count() == 0 and recent_failures() == []


def test_direct_import_surface():
    import nomorals.media.downloader_failures as lf

    assert lf.__all__ == ["Failure", "record_failure", "recent_failures",
                          "failure_count", "clear_failures", "format_failures"]
    # The ledger is a sibling module: downloader's own __all__ is untouched.
    assert "record_failure" not in dl.__all__


# ── module-level sites ─────────────────────────────────────────────

def test_detect_profile(monkeypatch):
    monkeypatch.delenv("NM_PROFILE", raising=False)
    monkeypatch.delenv("PREFIX", raising=False)
    with monkeypatch.context() as m:
        m.setattr(os, "cpu_count", lambda: (_ for _ in ()).throw(RuntimeError("pin")))
        assert dl.detect_profile() == "laptop"
    assert sites() == {"detect_profile"}
    assert dl.detect_profile() in ("laptop", "workstation")
    assert sites() == {"detect_profile"}


def test_mask_proxy():
    assert dl._mask_proxy("http://user:pass@host:notaport/") == "(proxy)"
    assert sites() == {"_mask_proxy"}
    assert dl._mask_proxy("http://user:pass@example.com:8080").startswith("http://")
    assert sites() == {"_mask_proxy"}


def test_is_short_link():
    class BadStr:
        def __str__(self):
            raise RuntimeError("pin")

    assert dl.is_short_link(BadStr()) is False
    assert sites() == {"is_short_link"}
    assert dl.is_short_link("https://example.com/x") is False
    assert sites() == {"is_short_link"}


def test_expand_short_url_outer_and_inner_probe():
    with mock.patch("nomorals.core.http.HttpClient", side_effect=RuntimeError("pin")):
        assert dl.expand_short_url("https://short.example/abc") == "https://short.example/abc"
    assert sites() == {"expand_short_url"}
    clear_failures()
    # Deliberate exclusion proof: both verbs failing is the probe
    # fall-through, not a failure — nothing is recorded.
    client = SimpleNamespace(
        head=mock.Mock(side_effect=RuntimeError("pin")),
        get=mock.Mock(return_value=SimpleNamespace(url="https://short.example/abc")),
    )
    with mock.patch("nomorals.core.http.HttpClient", return_value=client):
        assert dl.expand_short_url("https://short.example/abc") == "https://short.example/abc"
    assert sites() == set()


def test_embedded_media_urls_broken_block_and_healthy():
    broken = '<script type="application/ld+json">{broken</script>'
    assert dl._embedded_media_urls(broken) == []
    assert sites() == {"_embedded_media_urls"}
    good = ('<script type="application/ld+json">'
            '{"contentUrl": "https://cdn.example.com/v.mp4"}</script>')
    assert dl._embedded_media_urls(good) == ["https://cdn.example.com/v.mp4"]
    assert sites() == {"_embedded_media_urls"}


def test_embedded_media_urls_fallback_exclusion():
    # Deliberate exclusion proof: a failed JSON parse is the signal to
    # try the hand-unescape, and the URL is still captured — no record.
    markup = '{"playAddr":"https://example.com/a\\xb.mp4"}'
    assert dl._embedded_media_urls(markup) == ["https://example.com/a\\xb.mp4"]
    assert sites() == set()


def test_sha256_of(tmp_path):
    assert dl._sha256_of(tmp_path / "nope.bin") == ""
    assert sites() == {"_sha256_of"}
    f = tmp_path / "ok.bin"
    f.write_bytes(b"abc")
    import hashlib

    assert dl._sha256_of(f) == hashlib.sha256(b"abc").hexdigest()
    assert sites() == {"_sha256_of"}


# ── downloader-method sites ────────────────────────────────────────

def test_source_registry_db_failure():
    class RaisingDbCtx:
        @property
        def db(self):
            raise RuntimeError("pin")

    md = dl.MediaDownloader(context=RaisingDbCtx())
    assert md.source_registry() is not None
    assert sites() == {"source_registry"}


def test_record_download_outcome():
    reg = FakeRegistry()
    reg.for_url = mock.Mock(side_effect=RuntimeError("pin"))
    md = md_with(registry=reg)
    md._record_download_outcome(dl.DownloadReport(url="https://x"), ok=True)
    assert sites() == {"_record_download_outcome"}


def test_intake_candidate():
    reg = FakeRegistry()
    reg.for_url = mock.Mock(side_effect=RuntimeError("pin"))
    md = md_with(registry=reg)
    md._intake_candidate("https://new.example/", via="pin")
    assert sites() == {"_intake_candidate"}


def test_learned_for():
    reg = FakeRegistry()
    reg.learned_for_url = mock.Mock(side_effect=RuntimeError("pin"))
    md = md_with(registry=reg)
    assert md._learned_for("https://x") is None
    assert sites() == {"_learned_for"}


def test_note_rule_failure():
    reg = FakeRegistry()
    reg.note_rule_failure = mock.Mock(side_effect=RuntimeError("pin"))
    md = md_with(registry=reg)
    md._note_rule_failure(FakeDesc())
    assert sites() == {"_note_rule_failure"}


def test_learn_from_success():
    reg = FakeRegistry()
    reg.for_url = mock.Mock(return_value=FakeDesc())
    reg.learn_rule = mock.Mock(side_effect=RuntimeError("pin"))
    md = md_with(registry=reg)
    report = dl.DownloadReport(url="https://x", extractor="direct",
                               stages=[dl.StageLog(stage="direct", url="https://x", ok=True)])
    md._learn_from_success(report, None)
    assert sites() == {"_learn_from_success"}


def test_descriptor_for():
    extractors = SimpleNamespace(best_for=mock.Mock(side_effect=RuntimeError("pin")))
    md = dl.MediaDownloader(context=None, extractors=extractors)
    assert md._descriptor_for("https://x") is None
    assert sites() == {"_descriptor_for"}


def test_cf_cookie_store():
    with mock.patch("nomorals.accounts.cf_cookie_store.default_cf_cookie_store",
                    side_effect=RuntimeError("pin")):
        md = dl.MediaDownloader(context=None)
        assert md._cf_cookie_store() is None
    assert sites() == {"_cf_cookie_store"}


def test_succeed(tmp_path):
    ledger = SimpleNamespace(append=mock.Mock(side_effect=RuntimeError("pin")))
    md = md_with(registry=FakeRegistry(), ledger=ledger)
    f = tmp_path / "landed.bin"
    f.write_bytes(b"data")
    report = dl.DownloadReport(url="https://x", ok=True, path=str(f), size_bytes=4,
                               stages=[dl.StageLog(stage="direct", url="https://x", ok=True)])
    md._succeed(report, time.perf_counter())
    assert sites() == {"_succeed"}


def test_session_dir():
    ctx = SimpleNamespace(settings=SimpleNamespace(
        resolve=mock.Mock(side_effect=RuntimeError("pin"))))
    md = dl.MediaDownloader(context=ctx)
    assert md._session_dir() == ""
    assert sites() == {"_session_dir"}


def test_cookies_file():
    with mock.patch("nomorals.media.cookies.cookies_file_for",
                    side_effect=RuntimeError("pin")):
        md = dl.MediaDownloader(context=None)
        assert md._cookies_file("https://x") == ""
    assert sites() == {"_cookies_file"}


def test_stage_proxy_bookkeeping(tmp_path, monkeypatch):
    pool = SimpleNamespace(
        healthy_proxies=lambda: [{"id": "p1", "protocol": "http",
                                  "proxy_url": "http://127.0.0.1:9"}],
        record_success=mock.Mock(side_effect=RuntimeError("pin")),
        record_failure=mock.Mock(),
    )
    landed = tmp_path / "landed.bin"
    landed.write_bytes(b"data")
    md = dl.MediaDownloader(context=None)
    monkeypatch.setattr(md, "_proxy_pool", lambda: pool)
    with mock.patch("nomorals.tools.media.download",
                    return_value={"path": str(landed), "extractor": "pin"}), \
            mock.patch("nomorals.media.cookies.cookies_file_for", return_value=""):
        ok = md._stage_proxy(dl.DownloadReport(url="https://x"), "https://x",
                             tmp_path, False, "best", 60.0,
                             {"deadline": time.perf_counter() + 60})
    assert ok is True
    assert sites() == {"_stage_proxy.bookkeeping"}


def test_media_candidates_all_three_and_healthy():
    class RaisingSession:
        url = "https://example.com/page"

        def extract(self, kind="meta"):
            raise RuntimeError("pin")

        def html(self, max_chars=0):
            raise RuntimeError("pin")

        def links(self, max_links=0):
            raise RuntimeError("pin")

    md = dl.MediaDownloader(context=None)
    assert md._media_candidates(RaisingSession()) == []
    assert sites() == {"_media_candidates.meta", "_media_candidates.raw_html",
                       "_media_candidates.links"}
    healthy = SimpleNamespace(
        url="https://example.com/page",
        extract=lambda kind="meta": {"og": {"og:video": "https://cdn.example.com/v.mp4"}},
        html=lambda max_chars=0: {"html": ""},
        links=lambda max_links=0: {"links": []},
    )
    assert md._media_candidates(healthy) == ["https://cdn.example.com/v.mp4"]
    assert sites() == {"_media_candidates.meta", "_media_candidates.raw_html",
                       "_media_candidates.links"}


def test_fetch_candidate_progress(monkeypatch, tmp_path):
    with mock.patch("nomorals.core.jobs.report_progress",
                    side_effect=RuntimeError("pin")):
        md = dl.MediaDownloader(context=None)
        session = SimpleNamespace(
            fetch_bytes=lambda url, dest: {"bytes": 0, "content_type": "", "status": 404})
        ok = md._fetch_candidate(dl.DownloadReport(url="https://x"), session,
                                 "https://x", tmp_path, False, "https://x")
    assert ok is False
    assert "_fetch_candidate.progress" in sites()


def _unlink_case(monkeypatch, tmp_path, fetch):
    monkeypatch.setattr(Path, "unlink",
                        lambda self, missing_ok=False: (_ for _ in ()).throw(OSError("pin")))
    md = dl.MediaDownloader(context=None)
    session = SimpleNamespace(fetch_bytes=fetch)
    ok = md._fetch_candidate(dl.DownloadReport(url="https://x"), session,
                             "https://x", tmp_path, False, "https://x")
    assert ok is False
    return sites()


def test_fetch_candidate_unlink_branches(tmp_path, monkeypatch):
    got = _unlink_case(monkeypatch, tmp_path,
                       lambda url, dest: {"bytes": 0, "content_type": "", "status": 404})
    assert "_fetch_candidate.unlink" in got
    clear_failures()
    got = _unlink_case(monkeypatch, tmp_path,
                       lambda url, dest: {"bytes": 10, "content_type": "text/html",
                                          "status": 200})
    assert "_fetch_candidate.unlink#2" in got
    clear_failures()

    def fetch_html(url, dest):
        dest.write_bytes(b"<html><body>not media</body></html>")
        return {"bytes": 38, "content_type": "video/mp4", "status": 200}

    got = _unlink_case(monkeypatch, tmp_path, fetch_html)
    assert "_fetch_candidate.unlink#3" in got


def test_dest_dir(tmp_path, monkeypatch):
    with mock.patch("nomorals.tools.filesystem.safe_path",
                    side_effect=RuntimeError("pin")):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        md = dl.MediaDownloader(context=None)
        out = md._dest_dir(None)
    assert out == tmp_path / "workspace" / "media" and out.is_dir()
    assert sites() == {"_dest_dir"}


def test_cb():
    cb = dl.MediaDownloader._job_progress("pin")
    with mock.patch("nomorals.core.jobs.report_progress",
                    side_effect=RuntimeError("pin")):
        cb("hello")
    assert sites() == {"_cb"}
    cb("hello")
    assert sites() == {"_cb"}


def test_download_preflight_and_mapped(tmp_path, monkeypatch):
    md = dl.MediaDownloader(context=None)
    monkeypatch.setattr(md, "_attempt_stage", lambda *a, **k: False)
    with mock.patch("nomorals.media.download_kit.disk_preflight",
                    side_effect=RuntimeError("pin")):
        md.download("https://example.com/video.mp4", dest_dir=str(tmp_path))
    assert "download.preflight" in sites()
    clear_failures()
    desc = FakeDesc(name="pin", downloadable=True,
                    format_for=mock.Mock(side_effect=RuntimeError("pin")))
    extractors = SimpleNamespace(best_for=lambda url: desc)
    md2 = dl.MediaDownloader(context=None, extractors=extractors)
    monkeypatch.setattr(md2, "_attempt_stage", lambda *a, **k: False)
    md2.download("https://example.com/video.mp4", dest_dir=str(tmp_path))
    assert "download.mapped" in sites()


# ── scan pin: subsystem remainder is exactly the 2 deliberate probes ──

def test_subsystem_scan_remainder():
    import sys
    sys.path.insert(0, "/home/hatch/workspace/nm116-tree")
    from nomorals.tools import error_scan as es

    repo = "/home/hatch/workspace/nm116-tree"
    flagged = es.scan([repo + "/nomorals/media/downloader.py",
                       repo + "/nomorals/media/downloader_failures.py"])
    assert [f.file for f in flagged.findings] == []
    es._noqa_suppresses = lambda line, rule: False
    try:
        true = es.scan([repo + "/nomorals/media/downloader.py",
                        repo + "/nomorals/media/downloader_failures.py"])
    finally:
        import importlib
        importlib.reload(es)
    by_file = {}
    for f in true.findings:
        by_file.setdefault(f.file.split("/")[-1], []).append(f)
    assert by_file.get("downloader_failures.py", []) == []
    remainder = by_file.get("downloader.py", [])
    assert len(remainder) == 2
    assert {f.rule for f in remainder} == {"E104"}
    contexts = " | ".join(f.context for f in remainder)
    assert "try the other verb" in contexts
    assert "hand-unescape the common case" in contexts
