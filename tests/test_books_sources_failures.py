"""Pass #170 pins: the books failure ledger (REUSED from the muselibsites
followup) + sources conversions.

``nomorals/books/failures.py`` (namespace ``books``) is reused, not new:
sources.py is a books-subsystem module. The 17 durability tests below
mirror the #141+ shape exactly; the 14 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger entry
under its site name); the 5 exclusion pins prove the excluded probes
record nothing. No pin hardcodes scan totals or sources counts. Every
test redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written. No bare
swallow blocks anywhere in this file (pins-rule from #169): every
handler binds ``as exc`` and uses the bound name.
"""
import json
import os
import subprocess
import sys
import threading
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.books import failures as bf
from nomorals.books import sources as S
import nomorals.media.source_registry as sr


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    bf.clear_failures()
    yield
    bf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in bf.recent_failures()]


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("books")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("books").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("books")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("books").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("books")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("books").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("books").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("books").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("books")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("books").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("books").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("books").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("books").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("books")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 - test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("books").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        bf.record_failure("no-home", ValueError("boom"))
        assert bf.failure_count() == 1
        assert [f.site for f in bf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert bf.failure_count() == 0
    assert bf.recent_failures() == []
    assert bf.format_failures() == "no books failures recorded."
    bf.record_failure("books.site-a", ValueError("boom"))
    bf.record_failure("books.site-b", RuntimeError("bang"))
    assert bf.failure_count() == 2
    assert [f.site for f in bf.recent_failures()] == ["books.site-b", "books.site-a"]
    assert [f.site for f in bf.recent_failures(1)] == ["books.site-b"]
    text = bf.format_failures()
    assert "2 total" in text and "books.site-a" in text and "books.site-b" in text
    bf.clear_failures()
    assert bf.failure_count() == 0
    assert bf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    bf.record_failure("books.disk", ValueError("boom"))
    entries, total = fs.Store("books").load()
    assert total == 1 and entries[0]["site"] == "books.disk"
    bf.clear_failures()
    assert fs.Store("books").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    bf.record_failure("books.mem", ValueError("boom"))
    assert bf.failure_count() == 1
    assert [f.site for f in bf.recent_failures()] == ["books.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                bf.record_failure(f"books.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 - test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert bf.failure_count() == 400
    assert len(bf.recent_failures(1000)) == 200
    entries, total = bf._store.load()
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
        "from nomorals.books import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.books import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.books import failures as f\n"
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
    store_path = fs.Store("books").path()
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


# ---- Site pins ------------------------------------------------------------

class _BoomReg:
    """Registry stub whose every method raises."""

    def bind_store(self, *a, **k):
        raise RuntimeError("kv down")

    def register(self, *a, **k):
        raise RuntimeError("registry down")

    def for_domain(self, *a, **k):
        raise RuntimeError("registry down")

    def all(self, *a, **k):
        raise RuntimeError("registry down")

    def stats_for(self, *a, **k):
        raise RuntimeError("registry down")

    def note_candidate(self, *a, **k):
        raise RuntimeError("registry down")

    def record_outcome(self, *a, **k):
        raise RuntimeError("registry down")

    def record_no_result(self, *a, **k):
        raise RuntimeError("registry down")

    def record_blocked(self, *a, **k):
        raise RuntimeError("registry down")


class _DescReg:
    """Registry stub that resolves a desc but fails on recording."""

    def __init__(self):
        self.desc = SimpleNamespace(kinds=("book",))

    def for_domain(self, *a, **k):
        return self.desc

    def record_outcome(self, *a, **k):
        raise RuntimeError("record down")

    def record_no_result(self, *a, **k):
        raise RuntimeError("record down")


def _boom(*a, **k):
    raise RuntimeError("boom")


class _BadCtx:
    @property
    def db(self):
        raise RuntimeError("db exploded")


class _BadHit:
    @property
    def url(self):
        raise RuntimeError("hit exploded")


def test_site_registry_context_records_and_returns_none():
    assert S.book_registry_for_context(_BadCtx()) is None
    assert "sources.registry_context" in _sites()


def test_site_ensure_book_source_records_and_returns_none():
    assert S._ensure_book_source(
        _BoomReg(), S.GenericAdapter(name="g", domains=("example.com",))) is None
    assert "sources.ensure_book_source" in _sites()


def test_site_registry_desc_records_and_returns_none():
    assert S.registry_desc_for_adapter(
        _BoomReg(), S.GenericAdapter(name="g", domains=("example.com",))) is None
    assert "sources.registry_desc" in _sites()


def test_site_result_urls_records_and_returns_empty():
    assert S._book_result_urls("search", [_BadHit()]) == []
    assert "sources.result_urls" in _sites()


def test_site_note_result_urls_records_and_returns_none():
    hit = SimpleNamespace(url="http://x.example/novel/1")
    assert S._note_book_result_urls(_BoomReg(), "search", [hit]) is None
    assert "sources.note_result_urls" in _sites()


def test_site_record_success_records_and_returns_none():
    assert S._record_book_success(
        _DescReg(), S.GenericAdapter(name="g"), "search", [], 0.1,
        "http://a.example/novel/1") is None
    assert "sources.record_success" in _sites()


def test_site_record_failure_records_and_returns_none():
    assert S._record_book_failure(
        _DescReg(), S.GenericAdapter(name="g", domains=("example.com",)),
        "search", RuntimeError("boom"), 0.1,
        "http://a.example/novel/1") is None
    assert "sources.record_failure" in _sites()


def test_site_grown_adapters_records_and_returns_empty():
    assert S.grown_book_adapters(_BoomReg()) == []
    assert "sources.grown_adapters" in _sites()


def test_site_ordered_adapters_records_and_returns_pool():
    pool = [S.GenericAdapter(name="g", domains=("example.com",))]
    assert S.ordered_book_adapters(pool, _DescReg()) == pool
    assert "sources.ordered_adapters" in _sites()


class _FakeResp:
    def __init__(self, body, encoding):
        self._body = body
        self._encoding = encoding
        self.status = 200
        self.headers = {"Content-Encoding": encoding}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *a):
        return self._body

    def geturl(self):
        return "http://x.example/chapter/1"


class _FakeOpener:
    def __init__(self, resp):
        self._resp = resp

    def open(self, *a, **k):
        return self._resp


def test_site_gzip_decompress_records_and_returns_garbage_html():
    body = b"x-not-gzip" * 200  # 1800 bytes: survives the short-body block check
    resp = _FakeResp(body, "gzip")
    with patch.object(urllib.request, "build_opener",
                      return_value=_FakeOpener(resp)):
        final_url, html = S.Fetcher()._http_get_once("http://x.example/chapter/1")
    assert final_url == "http://x.example/chapter/1"
    assert len(html) >= 800  # the undecodable body came back as "content"
    assert "sources.gzip_decompress" in _sites()


def test_site_note_discoveries_records_and_returns_none():
    fetcher = S.Fetcher()
    fetcher._book_registry = _BoomReg()
    assert fetcher._note_book_discoveries(
        "http://a.example/novel/1", "http://b.example/novel/1",
        S.parse_html("<html><body></body></html>")) is None
    assert "sources.note_discoveries" in _sites()


class _ExplodingFetcher(S.Fetcher):
    def __setattr__(self, name, value):
        if name == "_book_registry":
            raise RuntimeError("binding exploded")
        super().__setattr__(name, value)


def test_site_bind_registry_records_and_returns_self():
    adapter = S.GenericAdapter(name="g", domains=("example.com",))
    adapter.fetcher = _ExplodingFetcher.__new__(_ExplodingFetcher)
    assert adapter.bind_book_registry(_BoomReg()) is adapter
    assert "sources.bind_registry" in _sites()


class _OneChapterAdapter(S.SourceAdapter):
    name = "one"
    domains = ("one.example",)

    def search(self, query, *, limit=10):
        return []

    def novel(self, url):
        return S.StoryMeta(title="T", url=url, source=self.name)

    def chapter_list(self, novel_url, *, limit=0):
        return [(1, "Ch1", "http://one.example/ch1")]

    def fetch_chapter(self, url):
        return S.Chapter(number=1, title="Ch1", url=url,
                         paragraphs=["hello world"])


def test_site_on_chapter_cb_records_and_download_continues():
    result = _OneChapterAdapter().download_story(
        "http://one.example/novel", on_chapter=_boom)
    assert result["total"] == 1
    assert len(result["chapters"]) == 1
    assert "sources.on_chapter_cb" in _sites()


def test_site_descriptor_lookup_records_and_falls_through_to_generic():
    bad = types.ModuleType("nomorals.books.descriptors")
    bad.descriptor_adapters = _boom
    had_real = "nomorals.books.descriptors" in sys.modules
    real = sys.modules.get("nomorals.books.descriptors")
    sys.modules["nomorals.books.descriptors"] = bad
    try:
        adapter = S.adapter_for_url("http://unknown.example/novel/1")
    finally:
        if had_real:
            sys.modules["nomorals.books.descriptors"] = real
        else:
            del sys.modules["nomorals.books.descriptors"]
    assert isinstance(adapter, S.GenericAdapter)
    assert "sources.descriptor_lookup" in _sites()


# ---- Exclusion pins (NOT-recording) --------------------------------------

def test_excl_normalize_url_malformed_ipv6_records_nothing():
    assert S.normalize_url("http://[::1") == "http://[::1"
    assert bf.failure_count() == 0


def test_excl_book_host_malformed_ipv6_records_nothing():
    assert S._book_host("http://[::1") == ""
    assert S._book_host("") == ""
    assert bf.failure_count() == 0


def test_excl_bind_store_failure_records_nothing(monkeypatch):
    reg = _BoomReg()
    monkeypatch.setattr(sr, "DEFAULT_REGISTRY", reg)
    assert S.book_registry_for_context(SimpleNamespace(db=object())) is reg
    assert bf.failure_count() == 0


def test_excl_decode_replace_never_raises_and_records_nothing():
    assert b"\xff\xfe\x80".decode("utf-8", errors="replace") == "���"
    assert b"\x00\xff".decode("utf-8", errors="replace") == "\x00�"
    assert bf.failure_count() == 0


class _RRStubFetcher:
    """Chapter URLs return a chapter DOM; the fiction-page URL raises."""

    def get(self, url):
        if "/chapter/" in url:
            html = ("<html><body><div class='chapter-inner chapter-content'>"
                    "<p>Once upon a time in a land far away.</p>"
                    "</div></body></html>")
            return url, S.parse_html(html), html
        raise S.SourceError("fiction page down")


def test_excl_royalroad_prev_next_failure_records_nothing():
    chapter = S.RoyalRoadAdapter(fetcher=_RRStubFetcher()).fetch_chapter(
        "http://royalroad.com/fiction/1/t/chapter/2/1-x")
    assert chapter.prev_url == "" and chapter.next_url == ""
    assert chapter.paragraphs == ["Once upon a time in a land far away."]
    assert bf.failure_count() == 0
