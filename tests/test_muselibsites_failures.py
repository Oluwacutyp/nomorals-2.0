"""Muselib-sites pins: the NEW books + museum failure ledgers + conversions.

The Museum + Library build (9bea288) added 9 silent swallows to the
census (books/library.py 6, museum.py 3); this pass converts all 9.
Neither ledger existed before this pass; both are created here,
durability-wired through the shared failure_store helper (the #141
shape: Store.save() resolves its path INSIDE the try). Every test
redirects stores with the app's own ``NM_HOME`` override (never the real
home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.books.failures`` / ``nomorals.museum_failures``;
restart tests use real subprocesses (a fresh interpreter is the honest
simulated restart — no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow: each
converted fallback returns exactly its old value AND leaves a ledger
entry under its site name. Zero exclusions ⇒ zero exclusion pins.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.books import failures as bf
import nomorals.museum_failures as mf
from nomorals.books import library as lm
from nomorals.books.library import Library, _strip_html_fragment, _bytes_to_text, LibraryError
from nomorals.museum import _dir_size, library_exhibitor, characters_exhibitor

LEDGERS = [
    pytest.param(bf, "books", "no books failures recorded.", id="books"),
    pytest.param(mf, "museum", "no museum failures recorded.", id="museum"),
]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    bf.clear_failures()
    mf.clear_failures()
    yield
    bf.clear_failures()
    mf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _ctx(tmp):
    return SimpleNamespace(db=None, settings=SimpleNamespace(workspace_dir=tmp))


def _ingest_book(lib, tmp):
    src = Path(tmp) / "book.txt"
    body = "\n\n".join(f"Paragraph {i} about markets and gold and yields."
                        for i in range(60))
    src.write_text("# Ch One\n\n" + body + "\n", encoding="utf-8")
    return lib.ingest(src, title="T")


# ---- Store-level -------------------------------------------------------

@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_round_trip(ns):
    store = fs.Store(ns)
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store(ns).load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_cap_enforced_on_disk(ns):
    store = fs.Store(ns)
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store(ns).load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_clear_removes_file(ns):
    store = fs.Store(ns)
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store(ns).load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("books").save(_entries(2, "b"), 2) is True
    assert fs.Store("museum").save(_entries(1, "m"), 1) is True
    assert fs.Store("legal").save(_entries(1, "l"), 1) is True
    entries, _ = fs.Store("books").load()
    assert [e["site"] for e in entries] == ["b0", "b1"]
    entries, _ = fs.Store("museum").load()
    assert [e["site"] for e in entries] == ["m0"]
    entries, _ = fs.Store("legal").load()
    assert [e["site"] for e in entries] == ["l0"]


@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_corrupt_file_falls_back_and_recovers(ns):
    store = fs.Store(ns)
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store(ns).load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store(ns).load()
    assert total == 1 and len(entries) == 1


@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_save_failure_never_raises_and_counts(ns, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store(ns).save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_save_home_resolution_failure_never_raises_and_counts(ns, monkeypatch):
    # The #141 fix-up shape: save() must resolve its path INSIDE the
    # try, so a home-resolution failure degrades to a counted write
    # failure, never raises. NM_HOME must be unset: an absolute
    # NM_HOME never calls expanduser, which is how scratch-home
    # batches mask this.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store(ns).save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


@pytest.mark.parametrize("ns", ["books", "museum"])
def test_store_concurrent_saves_never_raise_and_never_tear(ns):
    store = fs.Store(ns)
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store(ns).load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

@pytest.mark.parametrize("mod,ns,empty", LEDGERS)
def test_ledger_record_home_resolution_failure_degrades_to_memory(mod, ns, empty, monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mod.record_failure("no-home", ValueError("boom"))
        assert mod.failure_count() == 1
        assert [f.site for f in mod.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


@pytest.mark.parametrize("mod,ns,empty", LEDGERS)
def test_ledger_api_record_recent_count_clear_format(mod, ns, empty):
    assert mod.failure_count() == 0
    assert mod.recent_failures() == []
    assert mod.format_failures() == empty
    mod.record_failure(f"{ns}.site-a", ValueError("boom"))
    mod.record_failure(f"{ns}.site-b", RuntimeError("bang"))
    assert mod.failure_count() == 2
    assert [f.site for f in mod.recent_failures()] == [f"{ns}.site-b", f"{ns}.site-a"]
    assert [f.site for f in mod.recent_failures(1)] == [f"{ns}.site-b"]
    text = mod.format_failures()
    assert "2 total" in text and f"{ns}.site-a" in text and f"{ns}.site-b" in text
    mod.clear_failures()
    assert mod.failure_count() == 0
    assert mod.recent_failures() == []


@pytest.mark.parametrize("mod,ns,empty", LEDGERS)
def test_ledger_record_and_clear_hit_disk(mod, ns, empty):
    mod.record_failure(f"{ns}.disk", ValueError("boom"))
    entries, total = fs.Store(ns).load()
    assert total == 1 and entries[0]["site"] == f"{ns}.disk"
    mod.clear_failures()
    assert fs.Store(ns).load() == ([], 0)


@pytest.mark.parametrize("mod,ns,empty", LEDGERS)
def test_ledger_write_failure_degrades_to_memory(mod, ns, empty, tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mod.record_failure(f"{ns}.mem", ValueError("boom"))
    assert mod.failure_count() == 1
    assert [f.site for f in mod.recent_failures()] == [f"{ns}.mem"]


@pytest.mark.parametrize("mod,ns,empty", LEDGERS)
def test_ledger_concurrent_records_no_loss_beyond_cap(mod, ns, empty):
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mod.record_failure(f"{ns}.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mod.failure_count() == 400
    assert len(mod.recent_failures(1000)) == 200
    entries, total = mod._store.load()
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


def _record_code(import_stmt: str, n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        import_stmt + "\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


def _read_code(import_stmt: str) -> str:
    return (
        "import json\n"
        + import_stmt + "\n"
        "print(json.dumps({'count': f.failure_count(), "
        "'sites': [x.site for x in f.recent_failures(1000)]}))\n")


def _clear_code(import_stmt: str) -> str:
    return (import_stmt + "\n"
            "f.clear_failures()\nprint('ok')\n")


@pytest.mark.parametrize("import_stmt,ns", [
    pytest.param("from nomorals.books import failures as f", "books", id="books"),
    pytest.param("import nomorals.museum_failures as f", "museum", id="museum"),
])
def test_restart_record_survives(import_stmt, ns, tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(import_stmt, 3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _read_code(import_stmt))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


@pytest.mark.parametrize("import_stmt,ns", [
    pytest.param("from nomorals.books import failures as f", "books", id="books"),
    pytest.param("import nomorals.museum_failures as f", "museum", id="museum"),
])
def test_restart_clear_is_gone(import_stmt, ns, tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(import_stmt, 2)).returncode == 0
    assert _run_subprocess(home, _clear_code(import_stmt)).returncode == 0
    proc = _run_subprocess(home, _read_code(import_stmt))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


@pytest.mark.parametrize("import_stmt,ns", [
    pytest.param("from nomorals.books import failures as f", "books", id="books"),
    pytest.param("import nomorals.museum_failures as f", "museum", id="museum"),
])
def test_restart_total_preserved_and_cap_holds(import_stmt, ns, tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(import_stmt, 250)).returncode == 0
    proc = _run_subprocess(home, _read_code(import_stmt))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


@pytest.mark.parametrize("import_stmt,ns", [
    pytest.param("from nomorals.books import failures as f", "books", id="books"),
    pytest.param("import nomorals.museum_failures as f", "museum", id="museum"),
])
def test_restart_corrupt_store_recovers(import_stmt, ns, tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(import_stmt, 2)).returncode == 0
    store_path = fs.Store(ns).path()
    # path() resolves against the test process env, not the
    # subprocess home — resolve it under the subprocess home instead.
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _read_code(import_stmt))
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(import_stmt, 1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _read_code(import_stmt))
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Converted-site behavior: fallback unchanged + recorded -------------

class _RaisingTag:
    def sub(self, *args, **kwargs):
        raise RuntimeError("tag boom")


def test_converted_strip_html_records_and_returns_raw(monkeypatch):
    monkeypatch.setattr(lm, "_TAG_RE", _RaisingTag())
    assert _strip_html_fragment("<b>hi</b>") == "<b>hi</b>"
    assert [f.site for f in bf.recent_failures()] == ["library._strip_html_fragment"]


class _BadDecode(bytes):
    def decode(self, *args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad")


def test_converted_bytes_to_text_records_and_falls_through():
    # The old path fell through to the document parsers on decode
    # failure; garbage input makes the parsers raise LibraryError —
    # the pin proves the fall-through ran (not a new early return)
    # AND the failure was recorded.
    with pytest.raises(LibraryError):
        _bytes_to_text(_BadDecode(b"hello world"), "text/plain", "http://x")
    assert [f.site for f in bf.recent_failures()] == ["library._bytes_to_text"]


def _fts5_off_library(tmp_path, monkeypatch):
    """Library on a db that decided 'no FTS5' at first open — the
    documented non-FTS5 configuration, so digest_fts never exists and
    both FTS index writes take their OperationalError path."""
    monkeypatch.setattr(lm.Library, "_fts5_available", staticmethod(lambda: False))
    lib = Library(_ctx(tmp_path))
    _ingest_book(lib, tmp_path)
    return lib


def test_converted_create_digest_fts_records_and_returns_digest(tmp_path, monkeypatch):
    lib = _fts5_off_library(tmp_path, monkeypatch)
    source_id = lib.list_sources()[0]["source_id"]
    d = lib.create_digest(source_id, "gold rises when real yields fall")
    assert d["claim"] == "gold rises when real yields fall"
    assert [f.site for f in bf.recent_failures()] == ["library.Library.create_digest.fts_insert"]


def test_converted_revise_digest_fts_records_and_returns_revised(tmp_path, monkeypatch):
    lib = _fts5_off_library(tmp_path, monkeypatch)
    source_id = lib.list_sources()[0]["source_id"]
    d = lib.create_digest(source_id, "gold rises when real yields fall")
    bf.clear_failures()
    d2 = lib.revise_digest(d["record_id"], claim="gold rises when real yields fall fast")
    assert d2["claim"] == "gold rises when real yields fall fast"
    assert [f.site for f in bf.recent_failures()] == ["library.Library.revise_digest.fts_replace"]


def test_converted_vector_status_vectored_records_and_reports_unvectorized(tmp_path, monkeypatch):
    lib = Library(_ctx(tmp_path))
    _ingest_book(lib, tmp_path)
    source_id = lib.list_sources()[0]["source_id"]
    lib.create_digest(source_id, "gold rises when real yields fall")
    import nomorals.storage.db as dbmod
    monkeypatch.setattr(dbmod.Database, "query", _boom)
    st = lib.vector_status()
    # The digest IS vectored (create embedded it); the failed read
    # reports it unvectorized — wrong data from a silent guard read.
    assert st["unvectorized_digests"] == 1
    assert [f.site for f in bf.recent_failures()] == ["library.Library.vector_status.vectored"]


def test_converted_backfill_vectors_existing_records_and_returns_stats(tmp_path, monkeypatch):
    lib = Library(_ctx(tmp_path))
    import nomorals.storage.db as dbmod
    monkeypatch.setattr(dbmod.Database, "query", _boom)
    out = lib.backfill_vectors()
    assert isinstance(out, dict) and "digests" in out and "passages" in out
    assert [f.site for f in bf.recent_failures()] == ["library.Library.backfill_vectors.existing"]


class _FailingWalk:
    def rglob(self, *args, **kwargs):
        raise OSError("walk gone")


def test_converted_dir_size_records_and_returns_partial():
    assert _dir_size(_FailingWalk()) == 0
    assert [f.site for f in mf.recent_failures()] == ["museum._dir_size"]


def test_converted_library_exhibitor_book_meta_records_and_emits_fallback_card(tmp_path):
    # The book is listed, then its book.json vanishes mid-census: the
    # exhibitor still emits the card, with an empty digest.
    fake_lib = SimpleNamespace(
        root=lambda: tmp_path,
        db_path=lambda: tmp_path / "library.db",
        list_books=lambda: [{"slug": "t", "title": "T", "chapters": 2, "tags": []}],
        list_digests=lambda status="": [],
        list_sources=lambda: [],
    )
    cards = library_exhibitor(fake_lib)
    assert len(cards) == 1
    assert cards[0]["id"] == "library-book:t"
    assert cards[0]["hash"] == ""
    assert cards[0]["path"] == str(tmp_path / "t")
    assert [f.site for f in mf.recent_failures()] == ["museum.library_exhibitor.book_meta"]


def test_converted_characters_exhibitor_learn_doc_records_and_skips_character(tmp_path):
    class _Chars:
        def all(self):
            return [SimpleNamespace(id="c1")]

    class _Learn:
        def doc(self, cid):
            raise RuntimeError("doc gone")

    cards = characters_exhibitor(_Chars(), _Learn(), learn_dir=tmp_path)
    assert cards == []
    assert [f.site for f in mf.recent_failures()] == ["museum.characters_exhibitor.learn_doc"]
