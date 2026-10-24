"""Pass #117 pins: the download_kit failure ledger records every one of
the cluster's 19 silent handlers, with fallbacks held and healthy paths
recording nothing. No deliberate exclusions in this cluster.
"""
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media import download_kit as dk
from nomorals.media.download_kit_failures import (
    Failure, clear_failures, failure_count, format_failures, recent_failures,
    record_failure,
)
from nomorals.media.source_registry import Licence


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(1000)]


# ── ledger basics ─────────────────────────────────────────────────────

def test_ledger_basics_and_format():
    record_failure("matches", ValueError("bad pattern"))
    assert failure_count() == 1
    entries = recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "matches"
    assert entries[0].exc_type == "ValueError"
    assert entries[0].message == "bad pattern"
    assert "matches: ValueError: bad pattern" in format_failures()
    assert isinstance(entries[0], Failure)


def test_ledger_bound_and_count():
    for i in range(250):
        record_failure(f"site-{i}", RuntimeError("boom"))
    assert failure_count() == 250
    assert len(recent_failures(1000)) == 200  # ring keeps the newest 200
    assert recent_failures(1)[0].site == "site-249"


def test_ledger_unprintable_exc():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("matches", BadStr())
    assert "unprintable" in recent_failures(1)[0].message


def test_format_empty():
    assert format_failures() == "no recorded download_kit failures"


def test_direct_import_surface():
    for fn in (record_failure, recent_failures, failure_count,
               clear_failures, format_failures):
        assert callable(fn)


def test_download_kit_all_unchanged():
    assert set(dk.__all__) == {
        "ExtractorDescriptor", "ExtractorRegistry", "descriptors_for_registry",
        "DEFAULT_EXTRACTORS", "CloudflareState", "CloudflareStateMachine",
        "ResumeReport", "ResumeDownloader", "Preflight", "disk_preflight",
        "IntegrityLedger", "get_or_request_cookie", "store_owner_cookie",
    }


# ── real converted sites ──────────────────────────────────────────────

def test_matches_bad_pattern_recorded():
    desc = dk.ExtractorDescriptor(url_patterns=("[",))
    assert desc.matches("http://example.com/x") is False
    assert _sites() == ["matches"]
    good = dk.ExtractorDescriptor(url_patterns=(r"example\.com",))
    assert good.matches("http://example.com/x") is True
    assert _sites() == ["matches"]  # healthy match records nothing


def test_format_for_bad_label_recorded():
    desc = dk.ExtractorDescriptor(quality_map=((None, "fmt"),))
    assert desc.format_for("audio") == ""
    assert _sites() == ["format_for"]
    good = dk.ExtractorDescriptor(quality_map=(("audio", "bestaudio/best"),))
    assert good.format_for("audio") == "bestaudio/best"
    assert _sites() == ["format_for"]


def _src(**over):
    base = dict(name="Example", domain="example.com",
                url_patterns=(r"example\.com",), auth_needed=False,
                extraction_rule="ytdlp", quality_map=(("audio", "bestaudio"),),
                licence=Licence.UNKNOWN, notes="", downloadable=True)
    base.update(over)
    return SimpleNamespace(**base)


def _raising(msg):
    def _f(*args, **kwargs):
        raise RuntimeError(msg)
    return _f


def test_effective_rule_falls_back_to_seed_and_records():
    reg = SimpleNamespace(all=lambda: [_src()],
                          effective_extraction_rule=_raising("evidence down"))
    out = dk.descriptors_for_registry(reg)
    assert out[0].extraction_rule == "ytdlp"  # seed facts held
    assert _sites() == ["descriptors_for_registry:effective_rule"]


def test_effective_qmap_falls_back_to_seed_and_records():
    reg = SimpleNamespace(all=lambda: [_src(quality_map=())],
                          effective_quality_map=_raising("evidence down"))
    out = dk.descriptors_for_registry(reg)
    assert out[0].quality_map == ()
    assert _sites() == ["descriptors_for_registry:effective_qmap"]


def test_registry_outer_failure_keeps_structural_fallbacks_only():
    reg = SimpleNamespace(all=_raising("registry down"))
    out = dk.descriptors_for_registry(reg)
    assert [d.name for d in out] == ["direct-media", "ytdlp-generic"]
    assert _sites() == ["descriptors_for_registry"]


def test_register_failure_recorded():
    registry = dk.ExtractorRegistry(descriptors=[])
    registry.register(None)  # desc.name raises AttributeError
    assert registry._descs == []
    assert _sites() == ["ExtractorRegistry.register"]


def test_for_url_failure_recorded():
    class RaisingDesc:
        def matches(self, url):
            raise RuntimeError("matcher down")

    registry = dk.ExtractorRegistry(descriptors=[RaisingDesc()])
    assert registry.for_url("http://example.com/x") == []
    assert _sites() == ["ExtractorRegistry.for_url"]


def test_is_cloudflare_challenge_bad_input_recorded():
    assert dk.is_cloudflare_challenge(123) is False
    assert _sites() == ["is_cloudflare_challenge"]
    assert dk.is_cloudflare_challenge("just a moment, verifying") is True
    assert _sites() == ["is_cloudflare_challenge"]


def test_feed_bad_signal_recorded():
    machine = dk.CloudflareStateMachine()
    assert machine.feed(123) is dk.CloudflareState.IDLE
    assert _sites() == ["CloudflareStateMachine.feed"]
    assert machine.feed("challenge") is dk.CloudflareState.CHALLENGE_DETECTED
    assert _sites() == ["CloudflareStateMachine.feed"]


def test_ask_owner_move_failure_recorded(monkeypatch):
    machine = dk.CloudflareStateMachine()
    monkeypatch.setattr(machine, "_move", _raising("move down"))
    assert machine.ask_owner() == ""
    assert _sites() == ["CloudflareStateMachine.ask_owner"]
    fresh = dk.CloudflareStateMachine()
    assert fresh.ask_owner() != ""
    assert _sites() == ["CloudflareStateMachine.ask_owner"]


def test_store_owner_cookie_failure_recorded():
    class RaisingStore:
        def set(self, key, value):
            raise RuntimeError("vault down")

    assert dk.store_owner_cookie(RaisingStore(), "example.com", "abc") is False
    assert _sites() == ["store_owner_cookie"]
    saved = {}

    class DictStore:
        def set(self, key, value):
            saved[key] = value

    assert dk.store_owner_cookie(DictStore(), "example.com", "abc") is True
    assert _sites() == ["store_owner_cookie"]


def test_vault_read_failure_recorded_inner_only():
    class RaisingStore:
        def get(self, key):
            raise RuntimeError("vault down")

    status, _msg = dk.get_or_request_cookie(RaisingStore(), "example.com")
    assert status == "ask"
    assert _sites() == ["get_or_request_cookie:vault_read"]


def test_outer_cookie_failure_recorded_outer_only():
    class StaleStore:
        def get(self, key):
            return {"value": "v", "expires_at": object()}  # float() raises

    status, msg = dk.get_or_request_cookie(StaleStore(), "example.com")
    assert status == "ask"
    assert msg == "paste a fresh cf_clearance cookie value."
    assert _sites() == ["get_or_request_cookie"]


def test_meta_read_failure_forces_restart_and_records(tmp_path):
    dest = tmp_path / "vid.mp4"
    dest.write_bytes(b"partial")
    meta = dest.with_suffix(".mp4.resume.json")
    meta.write_text("{not json", encoding="utf-8")

    def head(url):
        return {"etag": "e1", "last_modified": "lm", "content_length": ""}

    dl = dk.ResumeDownloader(head=head, get_range=lambda u, s, e: b"CHUNK")
    report = dl.download("http://example.com/vid.mp4", dest)
    assert report.ok is True
    assert report.restarted is True  # corrupt meta -> never splice blind
    assert _sites() == ["ResumeDownloader.download:meta_read"]


def test_partial_save_failure_recorded(monkeypatch, tmp_path):
    def head(url):
        return {"etag": "e1", "last_modified": "lm", "content_length": ""}

    def failing_range(url, start, end):
        raise RuntimeError("net down")

    def raise_os(self, data):
        raise OSError("disk down")

    monkeypatch.setattr(Path, "write_bytes", raise_os)
    dl = dk.ResumeDownloader(head=head, get_range=failing_range)
    report = dl.download("http://example.com/vid.mp4", tmp_path / "vid.mp4")
    assert report.error.startswith("fetch failed")
    assert _sites() == ["ResumeDownloader.download:partial_save"]


def test_meta_unlink_failure_recorded(monkeypatch, tmp_path):
    def head(url):
        return {"etag": "e1", "last_modified": "lm", "content_length": "4"}

    def raise_os(self, missing_ok=False):
        raise OSError("unlink down")

    monkeypatch.setattr(Path, "unlink", raise_os)
    dl = dk.ResumeDownloader(head=head, get_range=lambda u, s, e: b"DATA")
    report = dl.download("http://example.com/vid.mp4", tmp_path / "vid.mp4")
    assert report.ok is True  # unlink failure never fails the download
    assert _sites() == ["ResumeDownloader.download:meta_unlink"]


def test_ledger_init_fallback_recorded(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    ledger = dk.IntegrityLedger(path=blocker / "ledger.jsonl")
    assert ledger.path == Path(tempfile.gettempdir()) / "nomorals-integrity-ledger.jsonl"
    assert _sites() == ["IntegrityLedger.__init__"]
    healthy = dk.IntegrityLedger(path=tmp_path / "sub" / "l.jsonl")
    assert healthy.path == tmp_path / "sub" / "l.jsonl"
    assert _sites() == ["IntegrityLedger.__init__"]


def test_append_failure_recorded(tmp_path):
    ledger = dk.IntegrityLedger(path=tmp_path / "l.jsonl")
    assert ledger.append(url="http://x", sha256="a" * 64, size_bytes=3) is True
    assert _sites() == []
    ledger.path = tmp_path  # a directory: open(..., "a") raises
    assert ledger.append(url="http://x", sha256="a" * 64, size_bytes=3) is False
    assert _sites() == ["IntegrityLedger.append"]


def test_bad_ledger_line_skipped_and_recorded(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    row = {"url": "http://x", "sha256": digest, "size_bytes": 5,
           "dest": str(dest), "source": "", "ts": 0}
    ledger_file = tmp_path / "led.jsonl"
    ledger_file.write_text("{oops\n" + json.dumps(row) + "\n", encoding="utf-8")
    ledger = dk.IntegrityLedger(path=ledger_file)
    ok, reason = ledger.verify(dest)
    assert ok is True
    assert reason == "hash + size match ledger"
    assert _sites() == ["IntegrityLedger.verify:bad_line"]
    clean_file = tmp_path / "clean.jsonl"
    clean_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    clean = dk.IntegrityLedger(path=clean_file)
    assert clean.verify(dest)[0] is True
    assert _sites() == ["IntegrityLedger.verify:bad_line"]


def test_subsystem_scan_clean(monkeypatch):
    from nomorals.tools import error_scan as es
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    root = Path(__file__).resolve().parent.parent
    report = es.scan([str(root / "nomorals" / "media" / "download_kit.py"),
                      str(root / "nomorals" / "media" / "download_kit_failures.py")])
    assert report.findings == []  # all 19 converted, no deliberate exclusions
