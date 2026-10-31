"""Tail-78 failure pins — nomorals/memory/backup.py.

The module's posture is in its docstrings: everything is
additive and never raises, and the file inventory treats
missing files as simply absent from the map ("a store without
a two-tier DB is fine").  The six guarded sites are the
inventory helpers' tolerance guards, the manifest's
record-count sentinel, and the importer's dedupe-source guard.
Each pin proves the failure value is the documented design
value — and, where the contract is silent-by-design, that
nothing is recorded on the failure path (snapshot method).

Probed live first (nm-tail78-verify/probes.py).
"""
import logging
from types import SimpleNamespace

import pytest

from nomorals.memory import backup as B


class _Raising:
    def __init__(self, exc):
        self._exc = exc

    def __get__(self, obj, objtype=None):
        raise self._exc


def _boom_fn(*a, **k):
    raise RuntimeError("resolver boom")


# ── 49 (E104) _settings_home: raising home_path → None ──────────

def test_49_settings_home_failure_returns_none_like_no_home():
    class BadSettings:
        home_path = _Raising(RuntimeError("settings boom"))

    mgr = SimpleNamespace(context=SimpleNamespace(settings=BadSettings()))
    assert B._settings_home(mgr) is None
    # byte-identical to the genuine no-home value
    assert B._settings_home(SimpleNamespace()) is None
    # control: a readable home comes through as a Path
    ok = SimpleNamespace(context=SimpleNamespace(
        settings=SimpleNamespace(home_path="/tmp/hh")))
    assert str(B._settings_home(ok)) == "/tmp/hh"


# ── 64 (E103) _resolve: resolver crash → key simply absent ──────

def test_64_resolver_failure_leaves_key_absent_and_silent(
        monkeypatch, caplog):
    from nomorals.memory import repetition as rep_mod
    monkeypatch.setattr(rep_mod, "repetition_db_path", _boom_fn)
    with caplog.at_level(logging.DEBUG, logger="nomorals.memory.backup"):
        files = B._memory_files(SimpleNamespace(db=None))
    assert "repetition.db" not in files
    # the sibling resolvers still ran — one crash removes one key
    assert "knowledge_state.db" in files or "two_tier.db" in files
    # silent by design: the inventory helpers record nothing
    assert caplog.records == []


# ── 76 (E103) main memories DB block: db.path crash → absent ────

def test_76_db_path_failure_leaves_memories_db_absent(caplog):
    class BadDB:
        path = _Raising(RuntimeError("db path boom"))

    with caplog.at_level(logging.DEBUG, logger="nomorals.memory.backup"):
        files = B._memory_files(SimpleNamespace(db=BadDB()))
    assert "memories.db" not in files
    assert caplog.records == []


# ── 90 (E103) entities block: entity_home crash → absent ────────

def test_90_entity_home_failure_leaves_entities_absent(
        monkeypatch, caplog):
    from nomorals.memory import types as types_mod
    monkeypatch.setattr(types_mod, "entity_home", _boom_fn)
    with caplog.at_level(logging.DEBUG, logger="nomorals.memory.backup"):
        files = B._memory_files(SimpleNamespace(db=None))
    assert "entities/" not in files
    assert caplog.records == []


# ── 152 (E104) record_count: repo.count crash → sentinel -1 ─────

def test_152_count_failure_records_minus_one_sentinel(tmp_path):
    class BadRepo:
        def count(self):
            raise RuntimeError("count boom")

    mgr = SimpleNamespace(db=None, repo=BadRepo(), semantic=None,
                          embedder=None)
    report = B.backup_to(mgr, tmp_path / "dest", label="pin")
    assert report["ok"] is True
    # -1 is distinguishable from every real count (>= 0): the
    # manifest says "unknown", never a fabricated number
    assert report["manifest"]["record_count"] == -1


# ── 304 (E104) dedupe source: db.query crash → empty existing ───

def test_304_dedupe_source_failure_imports_without_dedupe():
    class SickDB:
        def query(self, sql):
            raise RuntimeError("sick db")

    class RecManager:
        def __init__(self):
            self.db = SickDB()
            self.remembered = []

        def remember(self, content, **kw):
            self.remembered.append(content)
            return "id-1"

        def mark_private(self, rid):
            pass

    mgr = RecManager()
    report = B.import_records(mgr, [{"content": "already stored fact"}])
    # current behavior (tail-78 exclusion, tension in SURVEY.md):
    # the dedupe set degrades to empty, the record imports, and
    # the counts honestly report what was actually done
    assert report["ok"] is True
    assert report["imported"] == 1
    assert report["skipped_duplicate"] == 0
    assert mgr.remembered == ["already stored fact"]
