"""Pass #193 pins: the media failure ledger (reused from #192) + the
resolver conversion.

``nomorals/media/failures.py`` (namespace ``media``) already exists
from #192 — this pass REUSES it, it does not create a second media
ledger. The 17 durability tests below mirror the #141+ shape exactly
(same as #192's, against the same Store backend); the 1 site pin
proves the conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name); the 14
NOT-recording pins prove each exclusion stays silent with the
designed output. No pin hardcodes scan totals or resolver counts.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Restart tests use real
subprocesses. Every trigger was probed live before its pin was
written (probe.py).

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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.media import failures as mf
from nomorals.media import resolver as R
from nomorals.media.resolver import ResolutionError, ResolvedAudio, SourceResolver
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


def _boom_expanduser(path):
    raise OSError("no home")


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", _boom_expanduser)
    before = fs.write_failures()
    assert fs.Store("media").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    errors = []

    def worker(k):
        try:
            for i in range(30):
                fs.Store("media").save(_entries(3, f"w{k}-{i}-"), 3)
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("media").load()
    assert len(entries) == 3
    assert all(e["site"].startswith("w") for e in entries)


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", _boom_expanduser)
    mf.record_failure("resolver.probe", ValueError("boom"))
    assert _sites() == ["resolver.probe"]


def test_ledger_api_record_recent_count_clear_format():
    mf.record_failure("resolver.a", ValueError("boom0"))
    mf.record_failure("resolver.b", RuntimeError("boom1"))
    assert mf.failure_count() == 2
    assert _sites() == ["resolver.b", "resolver.a"]
    rendered = mf.format_failures()
    assert "resolver.b: RuntimeError: boom1" in rendered
    assert "resolver.a: ValueError: boom0" in rendered
    mf.clear_failures()
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []
    assert mf.format_failures() == "no recorded media failures"


def test_ledger_record_and_clear_hit_disk():
    mf.record_failure("resolver.disk", ValueError("boom"))
    entries, total = fs.Store("media").load()
    assert total == 1
    assert [e["site"] for e in entries] == ["resolver.disk"]
    mf.clear_failures()
    assert fs.Store("media").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mf.record_failure("resolver.mem", ValueError("boom"))
    assert _sites() == ["resolver.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mf.record_failure(f"media.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
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
    assert data["sites"][-1] == "site50"


def test_restart_corrupt_store_recovers(tmp_path, monkeypatch):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    monkeypatch.setenv("NM_HOME", str(home))
    fs.Store("media").path().write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    assert _run_subprocess(home, _record_code(1)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site pin (the conversion) -------------------------------------------

def test_pin_resolver_registry_bind(monkeypatch):
    import nomorals.storage.kv as kv_mod

    class BoomKV:
        def __init__(self, *a, **k):
            raise RuntimeError("kv down")

    monkeypatch.setattr(kv_mod, "KVStore", BoomKV)
    r = SourceResolver(context=types.SimpleNamespace(db=object()))
    reg = r._registry()
    assert reg is DEFAULT_REGISTRY
    assert _sites() == ["resolver.registry_bind"]


# ---- NOT-recording pins (exclusions) --------------------------------------

class _FakeReg:
    """Never-raises registry stand-in; individual methods patchable."""

    def __init__(self, **overrides):
        self._over = overrides

    def for_domain(self, domain):
        if "for_domain" in self._over:
            raise self._over["for_domain"]
        return types.SimpleNamespace(domain=domain, key=domain, name=domain,
                                     kinds=("music",))

    def stats_for(self, desc):
        if "stats_for" in self._over:
            raise self._over["stats_for"]
        return {"ok": 0, "fail": 0, "no_result": 0}

    def is_active(self, desc):
        if "is_active" in self._over:
            raise self._over["is_active"]
        return True

    def score(self, desc):
        if "score" in self._over:
            raise self._over["score"]
        return 0.5

    def all(self):
        if "all" in self._over:
            raise self._over["all"]
        return []

    def origin_for(self, desc):
        if "origin_for" in self._over:
            raise self._over["origin_for"]
        return "registered"

    def record_outcome(self, desc, ok, latency_s=0.0, error_class=""):
        if "record_outcome" in self._over:
            raise self._over["record_outcome"]

    def record_no_result(self, desc):
        if "record_no_result" in self._over:
            raise self._over["record_no_result"]

    def note_candidate(self, url, via=""):
        if "note_candidate" in self._over:
            raise self._over["note_candidate"]

    def for_url(self, url):
        if "for_url" in self._over:
            raise self._over["for_url"]
        return None


_DESC = types.SimpleNamespace(domain="example.com", key="example.com",
                              name="example", kinds=("music",))


def test_pin_ytdlp_spec_lookup_not_recorded(monkeypatch):
    def _boom_spec(name):
        raise ValueError("broken spec")

    monkeypatch.setattr("importlib.util.find_spec", _boom_spec)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/yt-dlp")
    assert R.ytdlp_available() is True
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert R.ytdlp_available() is False
    assert _sites() == []


def test_pin_ytdlp_which_not_recorded(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    def _boom_which(name):
        raise OSError("bad PATH")

    monkeypatch.setattr("shutil.which", _boom_which)
    assert R.ytdlp_available() is False
    assert _sites() == []


def test_pin_registry_outer_not_recorded():
    class RaisingDbContext:
        @property
        def db(self):
            raise RuntimeError("db unreadable")

    r = SourceResolver(context=RaisingDbContext())
    assert r._registry() is None
    assert _sites() == []


def test_pin_desc_lookup_not_recorded():
    r = SourceResolver()
    assert r._desc_for_strategy(_FakeReg(for_domain=RuntimeError("boom")),
                                "youtube-search") is None
    assert _sites() == []


def test_pin_record_served_not_recorded():
    r = SourceResolver()
    r._record_served(_FakeReg(record_outcome=RuntimeError("boom")), _DESC, 0.1,
                     ResolvedAudio(ok=True, path_or_url="https://x.example/a.mp3"))
    assert _sites() == []


def test_pin_record_no_result_not_recorded():
    r = SourceResolver()
    r._record_no_result(_FakeReg(record_no_result=RuntimeError("boom")), _DESC)
    assert _sites() == []


def test_pin_record_failure_not_recorded():
    r = SourceResolver()
    r._record_failure(_FakeReg(record_outcome=RuntimeError("boom")), _DESC,
                      "search_failed", 0.1)
    assert _sites() == []


def test_pin_strategy_error_classifier_not_recorded(monkeypatch):
    def _boom_available():
        raise RuntimeError("boom")

    monkeypatch.setattr("nomorals.media.resolver.ytdlp_available", _boom_available)
    r = SourceResolver()
    exc = ResolutionError("youtube-search", "search failed")
    r._record_strategy_error(_FakeReg(), "youtube-search", _DESC, exc, 0.1)
    assert _sites() == []


def test_pin_order_music_not_recorded():
    r = SourceResolver()
    items = [("a", lambda: None, _DESC), ("b", lambda: None, _DESC)]
    assert r._order_music(_FakeReg(stats_for=RuntimeError("boom")), items) == items
    assert _sites() == []


def test_pin_grown_music_not_recorded():
    r = SourceResolver()
    assert r._grown_music(_FakeReg(all=RuntimeError("boom"))) == []
    assert _sites() == []


def test_pin_extra_note_candidate_not_recorded(monkeypatch):
    c1 = ResolvedAudio(ok=True, path_or_url="https://a.example/1.mp3",
                       title="Song One", kind="netnaija")
    c2 = ResolvedAudio(ok=True, path_or_url="https://b.example/2.mp3",
                       title="Song Two", kind="netnaija")
    reg = _FakeReg(note_candidate=RuntimeError("boom"))
    r = SourceResolver(source_registry=reg)
    for attr in ("_netnaija_search_many", "_notjustok_search_many",
                 "_tooxclusive_search_many", "_soundcloud_search_many",
                 "_bandcamp_search_many", "_youtube_search_many",
                 "_youtube_music_search_many", "_archiveorg_search_many",
                 "_musicbrainz_search_many", "_openverse_search_many",
                 "_wikimedia_search_many", "_itunes_search_many",
                 "_deezer_search_many"):
        monkeypatch.setattr(r, attr, lambda q, limit: [])
    monkeypatch.setattr(r, "_netnaija_search_many", lambda q, limit: [c1, c2])
    out, _attempts = r.search_candidates_detailed("song one", limit=8)
    assert [o.title for o in out] == ["Song One", "Song Two"]
    assert _sites() == []


def test_pin_url_outcome_not_recorded():
    r = SourceResolver(source_registry=_FakeReg(for_url=RuntimeError("boom")))
    r._record_url_outcome("https://unknown.example/x.mp3", ResolvedAudio(ok=False))
    assert _sites() == []


def test_pin_local_file_fallthrough_not_recorded(monkeypatch):
    def _boom_safe(*a, **k):
        raise RuntimeError("no workspace")

    monkeypatch.setattr("nomorals.tools.filesystem.safe_path", _boom_safe)
    r = SourceResolver()
    with pytest.raises(ResolutionError, match="no audio file"):
        r._s_local_file("nope/nonexistent.mp3")
    assert _sites() == []


def test_pin_audiomack_title_not_recorded(monkeypatch):
    class FakeAudiomack:
        def handles_url(self, url):
            return True

    def _boom_probe(url):
        raise RuntimeError("no yt-dlp")

    monkeypatch.setattr("nomorals.media.resolver.probe", _boom_probe)
    monkeypatch.setattr("nomorals.media.sources.AudiomackSource", FakeAudiomack)
    r = SourceResolver()
    out = r._s_audiomack_url("https://audiomack.com/artist/song")
    assert out.ok is True
    assert out.title == "https://audiomack.com/artist/song"
    assert _sites() == []
