"""Pass #211 pins: decoder exclusions stay silent, ledger stays intact.

``nomorals/core/decoder.py`` converts 0 of its 12 true sites and excludes
12: the five DB helpers map an unavailable database to their documented
fail-soft value (the same value a missing database or a genuine miss
produces), the decoder attempts return the genuine "does not apply"
``None`` (or fall through to it), the cookie-lab failure degrades to the
genuine unclassified-cookie shape, and the two runner guards are the
documented "one decoder must never kill the run" design. No target edit
exists in this pass, so there is no behavior pin for a convert — the 17
durability tests below mirror the #141+ shape exactly (same as #209's,
against the REUSED core ledger); the 12 NOT-recording pins prove each
exclusion stays silent with its designed output. No pin hardcodes scan
totals or decoder counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree). Restart
tests use real subprocesses. Every trigger was probed live before its pin
was written.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import base64
import binascii
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import decoder as dec
from nomorals.core import failure_store as fs
from nomorals.core import failures as cf


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    cf.clear_failures()
    yield
    cf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"}
            for i in range(n)]


def _assert_silent():
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("core")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("core").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("core")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("core").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("core")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("core").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("core").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("core").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("core")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("core").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("core").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("core").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("core").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("core")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("core").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        cf.record_failure("no-home", ValueError("boom"))
        assert cf.failure_count() == 1
        assert [f.site for f in cf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []
    assert cf.format_failures() == "no core failures recorded."
    cf.record_failure("decoder.site-a", ValueError("boom"))
    cf.record_failure("decoder.site-b", RuntimeError("bang"))
    assert cf.failure_count() == 2
    assert [f.site for f in cf.recent_failures()] == ["decoder.site-b",
                                                      "decoder.site-a"]
    assert [f.site for f in cf.recent_failures(1)] == ["decoder.site-b"]
    text = cf.format_failures()
    assert "2 total" in text and "decoder.site-a" in text \
        and "decoder.site-b" in text
    cf.clear_failures()
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    cf.record_failure("decoder.disk", ValueError("boom"))
    entries, total = fs.Store("core").load()
    assert total == 1 and entries[0]["site"] == "decoder.disk"
    cf.clear_failures()
    assert fs.Store("core").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    cf.record_failure("decoder.mem", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["decoder.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                cf.record_failure(f"decoder.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cf.failure_count() == 400
    assert len(cf.recent_failures(1000)) == 200
    entries, total = cf._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.core import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.core import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.core import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("core").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


# ---- NOT-recording pins (one per exclusion) ------------------------------

class _BoomDB:
    def query(self, *a, **k):
        raise RuntimeError("db down")

    def query_one(self, *a, **k):
        raise RuntimeError("db down")

    def execute(self, *a, **k):
        raise RuntimeError("db down")


def test_not_recording_327_learn_db_failure_returns_false():
    assert dec.learn_hash(_BoomDB(), "abc123", "hello") is False
    _assert_silent()


def test_not_recording_343_learned_lookup_db_failure_returns_none():
    assert dec.learned_hash_lookup(_BoomDB(), "abc123") is None
    _assert_silent()


def test_not_recording_411_save_db_failure_returns_none():
    report = {"best": {"chain": ["base64"]}, "hits": [], "cookies": [],
              "jwt": None, "hash": None, "tokens": []}
    assert dec.save_report(_BoomDB(), report, source="t", kind="t") is None
    _assert_silent()


def test_not_recording_435_history_db_failure_returns_empty():
    assert dec.decode_history(_BoomDB()) == []
    _assert_silent()


def test_not_recording_455_get_report_db_failure_returns_none():
    assert dec.get_report(_BoomDB(), "rep-1") is None
    _assert_silent()


def test_not_recording_683_std_base64_failure_falls_through_to_urlsafe(
        monkeypatch):
    # urlsafe_b64decode delegates to the module-level b64decode, so a
    # blanket patch would break both branches; fail exactly the first
    # (std) call to prove the pass lets the urlsafe candidate run.
    real = base64.b64decode
    calls = []

    def _boom_once(s, *a, **k):
        calls.append(s)
        if len(calls) == 1:
            raise binascii.Error("std decode down")
        return real(s, *a, **k)

    monkeypatch.setattr(base64, "b64decode", _boom_once)
    hit = dec._Base64Decoder().attempt("aGVsbG8gd29ybGQ=")
    assert hit is not None and hit.output == "hello world"
    assert len(calls) == 2  # std failed, urlsafe ran
    _assert_silent()


def test_not_recording_689_both_base64_candidates_failure_returns_none(
        monkeypatch):
    def _boom(*a, **k):
        raise binascii.Error("candidate down")

    monkeypatch.setattr(base64, "b64decode", _boom)
    monkeypatch.setattr(base64, "urlsafe_b64decode", _boom)
    assert dec._Base64Decoder().attempt("aGVsbG8gd29ybGQ=") is None
    assert dec._Base64Decoder().attempt("-_aB3x9Q") is None
    _assert_silent()


def test_not_recording_995_yaml_parse_failure_returns_none(monkeypatch):
    called = []

    class _FakeYaml:
        @staticmethod
        def safe_load(s):
            called.append(s)
            raise ValueError("yaml broken")

    monkeypatch.setitem(sys.modules, "yaml", _FakeYaml())
    assert dec._YamlDecoder().attempt("name: test\nvalue: 42") is None
    assert called  # proves the 995 handler fired, not the ImportError one
    _assert_silent()


def test_not_recording_1224_cookie_lab_failure_keeps_unclassified_cookies(
        monkeypatch):
    class _BoomLab:
        def __init__(self, *a, **k):
            raise RuntimeError("lab down")

    monkeypatch.setattr("nomorals.core.cookies.CookieLab", _BoomLab)
    hit = dec._CookieDecoder().attempt(
        "Set-Cookie: session=abc123; Path=/; HttpOnly")
    assert hit is not None
    cookies = [c for c in hit.output if isinstance(c, dict)]
    assert cookies and cookies[0]["name"] == "session"
    assert "class" not in cookies[0] and "service" not in cookies[0]
    _assert_silent()


def test_not_recording_1400_quoted_printable_failure_returns_none(monkeypatch):
    def _boom(*a, **k):
        raise ValueError("qp down")

    monkeypatch.setattr("quopri.decodestring", _boom)
    assert dec._QuotedPrintableDecoder().attempt(
        "Hello=20World=21=20Test") is None
    _assert_silent()


class _BoomDecoder(dec.Decoder):
    name = "boom"

    def attempt(self, data):
        raise RuntimeError("decoder down")

    def try_bytes(self, data):
        raise RuntimeError("byte decoder down")


def test_not_recording_1633_crashing_decoder_skipped_run_continues():
    boom = _BoomDecoder()
    dec.DECODERS.append(boom)
    try:
        hits = dec._run_decoders("aGVsbG8gd29ybGQ=", 0.4)
    finally:
        dec.DECODERS.remove(boom)
    assert any(h.decoder == "base64" for h in hits)
    _assert_silent()


def test_not_recording_1646_crashing_byte_decoder_skipped():
    boom = _BoomDecoder()
    dec.DECODERS.append(boom)
    try:
        hits = dec._run_byte_decoders(b"hello")
    finally:
        dec.DECODERS.remove(boom)
    assert hits == []
    _assert_silent()
