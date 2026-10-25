"""Pass #192 pins: the NEW media failure ledger + film_sources conversion.

``nomorals/media/failures.py`` (namespace ``media``) is created by this
pass — no generic media ledger existed before it (the existing media
ledgers are per-subsystem: download_kit, downloader, render,
source_registry, studio_automation, vocal_lite). The 17 durability tests
below mirror the #141+ shape exactly (same as #191's, against the same
Store backend); the 1 site pin proves the conversion records without
changing flow (fallback value unchanged AND a ledger entry under its
site name); the 14 NOT-recording pins prove each exclusion stays silent
with the designed output. No pin hardcodes scan totals or
film_sources counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import json
import os
import subprocess
import sys
import threading
import types
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.media import failures as mf
from nomorals.media import film_sources as fsrc
from nomorals.media.source_registry import DEFAULT_REGISTRY


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mf.clear_failures()
    yield
    mf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in mf.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("media")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("media").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("media")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("media").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("media")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("media").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("media").save(_entries(2, "c"), 2) is True
    assert fs.Store("media.download_kit").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("media").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("media.download_kit").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("media")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("media").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("media").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("media").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("media").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("media")
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
    entries, _ = fs.Store("media").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mf.record_failure("no-home", ValueError("boom"))
        assert mf.failure_count() == 1
        assert [f.site for f in mf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []
    assert mf.format_failures() == "no recorded media failures"
    mf.record_failure("media.site-a", ValueError("boom"))
    mf.record_failure("media.site-b", RuntimeError("bang"))
    assert mf.failure_count() == 2
    assert [f.site for f in mf.recent_failures()] == ["media.site-b", "media.site-a"]
    assert [f.site for f in mf.recent_failures(1)] == ["media.site-b"]
    text = mf.format_failures()
    assert "media.site-a" in text and "media.site-b" in text
    mf.clear_failures()
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mf.record_failure("media.disk", ValueError("boom"))
    entries, total = fs.Store("media").load()
    assert total == 1 and entries[0]["site"] == "media.disk"
    mf.clear_failures()
    assert fs.Store("media").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mf.record_failure("media.mem", ValueError("boom"))
    assert mf.failure_count() == 1
    assert [f.site for f in mf.recent_failures()] == ["media.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mf.record_failure(f"media.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mf.failure_count() == 400
    assert len(mf.recent_failures(1000)) == 200
    entries, total = mf._store.load()
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
        "from nomorals.media import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.media import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.media import failures as f\n"
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
    store_path = fs.Store("media").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site pin (convert) --------------------------------------------------

def test_pin_film_sources_registry_bind(monkeypatch):
    import nomorals.storage.kv as kv_mod

    class BoomKV:
        def __init__(self, *a, **k):
            raise RuntimeError("kv down")

    monkeypatch.setattr(kv_mod, "KVStore", BoomKV)
    reg = fsrc.film_registry_for_context(types.SimpleNamespace(db=object()))
    assert reg is DEFAULT_REGISTRY
    assert _sites() == ["film_sources.registry_bind"]


# ---- NOT-recording pins (exclusions) --------------------------------------

class _BoomSearch(fsrc.FilmSource):
    name = "boom"
    def search(self, query, limit=8):
        raise RuntimeError("search boom")


class _OkSearch(fsrc.FilmSource):
    name = "ok"
    def search(self, query, limit=8):
        return [fsrc.FilmCandidate(source="ok", title="t",
                                   url="https://example.com/1")]


class _BoomSearchMapped(fsrc.FilmSource):
    name = "archiveorg"
    def search(self, query, limit=8):
        raise RuntimeError("search boom")


class _BoomDownload(fsrc.FilmSource):
    name = "archiveorg"
    def search(self, query, limit=8):
        return []
    def download_url(self, candidate):
        raise RuntimeError("dl boom")


class _FakeReg:
    """Registry double capturing evidence calls (never raises)."""

    def __init__(self):
        self.calls = []

    def for_domain(self, d):
        return f"desc:{d}"
    def stats_for(self, d):
        return {"ok": 0, "fail": 0, "no_result": 0}
    def is_active(self, d):
        return True
    def score(self, d):
        return 0.0
    def all(self):
        return []
    def origin_for(self, d):
        return "seed"
    def record_outcome(self, desc, ok, latency_s=0.0, error_class=""):
        self.calls.append(("outcome", desc, ok, error_class))
    def record_blocked(self, desc, latency_s=0.0):
        self.calls.append(("blocked", desc))
    def record_rate_limited(self, desc, latency_s=0.0):
        self.calls.append(("rate", desc))
    def record_no_result(self, desc):
        self.calls.append(("no_result", desc))
    def note_candidate(self, url, via="", kinds=()):
        pass


class _BadCtx:
    @property
    def db(self):
        raise RuntimeError("db boom")


class _BadStatusSrc:
    @property
    def _last_status(self):
        raise RuntimeError("status boom")


class _RaisingReg:
    def for_domain(self, d):
        raise RuntimeError("reg boom")
    def record_outcome(self, *a, **k):
        raise RuntimeError("record boom")
    def note_candidate(self, *a, **k):
        raise RuntimeError("note boom")
    def stats_for(self, d):
        raise RuntimeError("stats boom")
    def all(self):
        raise RuntimeError("all boom")


def test_pin_http_body_read_not_recorded():
    src = fsrc.ArchiveOrgFilmSource()
    err = urllib.error.HTTPError("http://x/", 403, "Forbidden", {}, None)
    err.read = lambda: (_ for _ in ()).throw(OSError("read boom"))
    with patch("urllib.request.urlopen", side_effect=err):
        assert src._get("http://x/") == ""
    assert src._last_status == 403
    assert _sites() == []


def test_pin_archive_search_parse_not_recorded():
    src = fsrc.ArchiveOrgFilmSource()
    src._get = lambda url, timeout=30: "{not json"
    assert src.search("x") == []
    assert _sites() == []


def test_pin_archive_metadata_parse_not_recorded():
    src = fsrc.ArchiveOrgFilmSource()
    src._get = lambda url, timeout=30: "{broken"
    cand = fsrc.FilmCandidate(source="archiveorg",
                              url="https://archive.org/details/abc",
                              extra={"identifier": "abc"})
    assert src.download_url(cand) == "https://archive.org/details/abc"
    assert _sites() == []


def test_pin_registry_context_outer_not_recorded():
    assert fsrc.film_registry_for_context(_BadCtx()) is None
    assert _sites() == []


def test_pin_desc_lookup_not_recorded():
    assert fsrc._desc_for_film_source(_RaisingReg(),
                                      fsrc.ArchiveOrgFilmSource()) is None
    assert _sites() == []


def test_pin_fetch_evidence_not_recorded():
    assert fsrc._film_fetch_evidence(_BadStatusSrc()) == ("none", "")
    assert _sites() == []


def test_pin_apply_evidence_not_recorded():
    fsrc._apply_film_evidence(_RaisingReg(), object(), "served", "", 0.1)
    assert _sites() == []


def test_pin_record_search_not_recorded():
    fsrc._record_film_search(
        _RaisingReg(), object(), object(),
        [fsrc.FilmCandidate(url="https://example.com/p")], 0.1, False)
    assert _sites() == []


def test_pin_order_film_not_recorded():
    items = [("a", "da"), ("b", "db")]
    assert fsrc._order_film(_RaisingReg(), items) == items
    assert _sites() == []


def test_pin_grown_sources_not_recorded():
    assert fsrc._grown_film_sources(_RaisingReg()) == []
    assert _sites() == []


def test_pin_chain_continues_not_recorded(monkeypatch):
    monkeypatch.setattr(fsrc, "FILM_CHAIN", [_BoomSearch(), _OkSearch()])
    got = fsrc.search_films("q")
    assert len(got) == 1 and got[0].source == "ok"
    assert _sites() == []


def test_pin_chain_raised_flag_not_recorded(monkeypatch):
    monkeypatch.setattr(fsrc, "FILM_CHAIN", [_BoomSearchMapped()])
    reg = _FakeReg()
    assert fsrc.search_films("q", registry=reg) == []
    assert any(c[0] == "outcome" and c[2] is False
               and c[3] == "search_failed" for c in reg.calls)
    assert _sites() == []


def test_pin_resolve_fallback_not_recorded(monkeypatch):
    monkeypatch.setattr(fsrc, "FILM_CHAIN", [_BoomDownload()])
    cand = fsrc.FilmCandidate(source="archiveorg",
                              url="https://example.com/post")
    assert fsrc.resolve_film_url(cand) == "https://example.com/post"
    assert _sites() == []


def test_pin_resolve_failed_evidence_not_recorded(monkeypatch):
    monkeypatch.setattr(fsrc, "FILM_CHAIN", [_BoomDownload()])
    reg = _FakeReg()
    cand = fsrc.FilmCandidate(source="archiveorg",
                              url="https://example.com/post")
    assert fsrc.resolve_film_url(cand, registry=reg) == "https://example.com/post"
    assert any(c[0] == "outcome" and c[2] is False
               and c[3] == "resolve_failed" for c in reg.calls)
    assert _sites() == []
