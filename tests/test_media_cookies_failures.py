"""Pins for nomorals/media/cookies.py — tail-14 (pins-only, 0 converts / 10 exclusions).

Every site maps a failure to a documented miss value (or, for 218, to a
rebuild); the module contract is "Never raises" and "if no file exists,
everything behaves exactly as before". NOT-recording is pinned via the
media failures ledger snapshot (tail-8 method): cookies.py records
nothing into it, so recent_failures() must be unchanged after each
trigger. The pins file itself contains no except handlers (AST-checked).
"""
import json
from pathlib import Path

from nomorals.media import cookies as C
from nomorals.media import failures


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _snapshot():
    return failures.recent_failures()


def _assert_unrecorded(before):
    assert failures.recent_failures() == before


class _BoomStr:
    def __bool__(self):
        return True

    def __str__(self):
        raise RuntimeError("str boom")


def test_124_discovery_failure_returns_none(monkeypatch):
    before = _snapshot()
    monkeypatch.setattr(C, "_candidate_paths", _boom)
    assert C.find_cookies_file() is None
    _assert_unrecorded(before)


def test_168_corrupt_session_returns_empty(tmp_path):
    before = _snapshot()
    bad = tmp_path / "bad.json"
    bad.write_text("{invalid json", encoding="utf-8")
    assert C._session_cookie_entries(bad) == []
    _assert_unrecorded(before)


def test_218_freshness_oserror_rebuilds_export(tmp_path, monkeypatch):
    before = _snapshot()
    sdir = tmp_path / "sessions"
    sdir.mkdir()
    (sdir / "s1.json").write_text(
        json.dumps([{"name": "SID", "domain": ".youtube.com", "path": "/",
                     "secure": True, "expires": 0, "value": "fresh123"}]),
        encoding="utf-8")
    stale = sdir / C.BROWSER_COOKIES_NAME
    stale.write_text(
        "# Netscape HTTP Cookie File\n.old\tFALSE\t/\tFALSE\t0\tOLD\tstale999\n",
        encoding="utf-8")
    orig_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == C.BROWSER_COOKIES_NAME:
            raise OSError("stat boom")
        return orig_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert C.export_browser_cookies(str(sdir)) == str(stale)
    text = stale.read_text(encoding="utf-8")
    assert "fresh123" in text and "stale999" not in text
    _assert_unrecorded(before)


def test_242_export_failure_returns_none(monkeypatch):
    before = _snapshot()
    monkeypatch.setattr(C, "browser_session_dirs", _boom)
    assert C.export_browser_cookies() is None
    _assert_unrecorded(before)


def test_264_unreadable_file_has_no_cookie(tmp_path):
    before = _snapshot()
    assert C._file_has_cookie_for(str(tmp_path / "nope.txt"), "youtube.com") is False
    _assert_unrecorded(before)


def test_289_resolve_failure_returns_none(monkeypatch):
    before = _snapshot()
    monkeypatch.setattr(C, "find_cookies_file", _boom)
    assert C.cookies_file_for("https://youtube.com/watch?v=x") is None
    _assert_unrecorded(before)


def test_322_unstringifiable_message_not_bot_error():
    before = _snapshot()
    assert C.is_bot_detection_error(_BoomStr()) is False
    _assert_unrecorded(before)


def test_359_unreadable_file_counts_zero(tmp_path):
    before = _snapshot()
    assert C._count_valid_youtube_cookies(str(tmp_path / "nope.txt")) == 0
    _assert_unrecorded(before)


def test_392_unreadable_file_stats_empty(tmp_path):
    before = _snapshot()
    assert C._netscape_stats(str(tmp_path / "nope.txt")) == {
        "total": 0, "valid": 0, "domains": []}
    _assert_unrecorded(before)


def test_439_status_failure_returns_fallback(monkeypatch):
    before = _snapshot()
    monkeypatch.setattr(C, "find_cookies_file", _boom)
    st = C.cookies_status()
    assert st["found"] is None
    assert st["effective"] == {"source": "none", "path": None}
    assert st["stale"] is False
    _assert_unrecorded(before)
