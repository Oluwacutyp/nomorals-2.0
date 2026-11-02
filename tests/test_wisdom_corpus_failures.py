"""Tail-110 pins for nomorals/wisdom/corpus.py — 4 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was really taken (trigger == fallback AND
control != fallback). None of the four guarded paths writes anywhere:
they are read/degradation paths whose only outputs are their return
values, which the pins read back directly.

Sites:
* 423 E103 — _packaged_slugs(): the packaged seed manifest read is
  guarded by a TYPED (OSError, ValueError) handler; a missing/corrupt
  packaged manifest degrades the slug set to empty, so legacy entries
  resolve to the "manual" seed label instead of "packaged-seed" in
  _seed_of()/seeds(). Provenance display only; no gate consumes it.
* 854 E104 — _expand_query(): a library search crash returns the
  original query unchanged — the docstring's printed contract
  ("Returns the original query when nothing useful comes back").
  The ask path proceeds unexpanded, never with invented terms.
* 1069 E104 — _index_metadata(): a books-table read crash returns {}
  so a semantic-index build proceeds with empty per-key metadata
  (vectors still built; tradition/work enrichment absent) instead of
  failing the whole build over an enrichment lookup.
* 1161 E104 — semantic_index_status(): a vectors.db whose schema is
  missing/broken returns the same status dict as a genuinely absent
  index (built False, vectors 0). Reporting surface only
  (``nm wisdom status``); the build-skip decision never reads this
  dict — build_semantic_index decides via open_index()/count(),
  which raise honestly, and a connect() failure here propagates
  (it sits outside the guard).
"""

from __future__ import annotations

import json
import sqlite3
import types
from pathlib import Path
from unittest import mock

from nomorals.wisdom.corpus import CanonCorpus, ManifestEntry


def _bare(tmp_path: Path) -> CanonCorpus:
    """A CanonCorpus with only the attributes these methods touch."""
    c = CanonCorpus.__new__(CanonCorpus)
    c.context = types.SimpleNamespace(
        settings=types.SimpleNamespace(workspace_dir=str(tmp_path)))
    c._manifest = {}
    return c


# ── 423: _packaged_slugs ────────────────────────────────────────────────

def test_423_control_real_manifest_slugs(tmp_path):
    slugs = _bare(tmp_path)._packaged_slugs()
    assert "bible-kjv" in slugs and len(slugs) >= 40


def test_423_oserror_degrades_to_empty_set(tmp_path):
    c = _bare(tmp_path)
    with mock.patch.object(Path, "read_text", side_effect=OSError("gone")):
        assert c._packaged_slugs() == set()


def test_423_corrupt_json_degrades_to_empty_set(tmp_path):
    c = _bare(tmp_path)
    with mock.patch.object(Path, "read_text", return_value="{not json"):
        assert c._packaged_slugs() == set()


# ── 854: _expand_query ──────────────────────────────────────────────────

class _LibRaise:
    def search(self, q, top=5):
        raise RuntimeError("index down")


class _Hit:
    passage = "karma rebirth samsara karma liberation"


class _LibOk:
    def search(self, q, top=5):
        return [_Hit()]


def test_854_search_crash_returns_original_query(tmp_path):
    c = _bare(tmp_path)
    c.library = _LibRaise()
    assert c._expand_query("karma rebirth") == "karma rebirth"


def test_854_control_expansion_appends_terms(tmp_path):
    c = _bare(tmp_path)
    c.library = _LibOk()
    expanded = c._expand_query("karma rebirth")
    assert expanded.startswith("karma rebirth ") and len(expanded) > 13


# ── 1069: _index_metadata ───────────────────────────────────────────────

def _db(tmp_path: Path, name: str, ddl: str, rows=()) -> str:
    path = tmp_path / name
    con = sqlite3.connect(str(path))
    con.execute(ddl)
    for row in rows:
        con.execute("INSERT INTO books VALUES (?, ?)", row)
    con.commit()
    con.close()
    return str(path)


def test_1069_missing_books_table_returns_empty(tmp_path):
    path = _db(tmp_path, "lib1.db", "CREATE TABLE other (x TEXT)")
    c = _bare(tmp_path)
    c.library = types.SimpleNamespace(db_path=lambda: path)
    assert c._index_metadata() == {}


def test_1069_control_metadata_resolved(tmp_path):
    path = _db(tmp_path, "lib2.db", "CREATE TABLE books (slug TEXT, title TEXT)",
               rows=[("bk1", "[myslug] Some Title")])
    c = _bare(tmp_path)
    c.library = types.SimpleNamespace(db_path=lambda: path)
    c._manifest = {"myslug": ManifestEntry(
        slug="myslug", title="Some Title", tradition="buddhism",
        canon_status="canon", source_url="http://x")}
    assert c._index_metadata() == {
        "bk1#": {"tradition": "buddhism", "work": "Some Title"}}


# ── 1161: semantic_index_status ─────────────────────────────────────────

def test_1161_no_db_reports_not_built(tmp_path):
    st = _bare(tmp_path).semantic_index_status()
    assert st["built"] is False and st["vectors"] == 0


def test_1161_schemaless_db_reports_not_built_and_writes_nothing(tmp_path):
    root = tmp_path / "wisdom"
    root.mkdir(parents=True)
    db = root / "vectors.db"
    db.write_bytes(b"")
    st = _bare(tmp_path).semantic_index_status()
    assert st["built"] is False and st["vectors"] == 0
    assert db.read_bytes() == b""  # status read leaves the file untouched


def test_1161_control_built_index_reports_truthfully(tmp_path):
    root = tmp_path / "wisdom"
    root.mkdir(parents=True)
    con = sqlite3.connect(str(root / "vectors.db"))
    con.execute("CREATE TABLE meta (key TEXT, value TEXT)")
    con.execute("CREATE TABLE vectors (id TEXT)")
    con.execute("INSERT INTO meta VALUES ('backend', 'hash')")
    con.execute("INSERT INTO meta VALUES ('engine', 'ngram')")
    con.executemany("INSERT INTO vectors VALUES (?)", [("a",), ("b",), ("c",)])
    con.commit()
    con.close()
    st = _bare(tmp_path).semantic_index_status()
    assert st["built"] is True and st["vectors"] == 3
    assert st["backend"] == "hash" and st["engine"] == "ngram"
