"""Mongo program P6 (FINAL): cutover machinery, backend-aware backups,
streaming parity — and the dress rehearsal that proves the flip.

The rehearsal is the phase's acceptance: a realistic migrated install
(KV + ported domains + queue + pinned domains + residue sidecars) is
carried into a real local mongod replica set, gated, flipped, smoked,
backed up, restored to a scratch store, and rolled back — every step
asserted, and the SQLite source proven untouched by per-collection
checksums taken before and after.

Unit legs pin the pieces separately: streaming/in-memory comparator
equivalence (same data, same planted divergences, identical verdicts),
gate routing above the size threshold, the document-dump codec,
backend-aware BackupManager behavior, the snapshot document leg,
config-flip mechanics (including the environment-override refusal),
preflight refusals, and residue disposition records.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.core.config import Settings, load_settings
from nomorals.os.snapshots import SnapshotManager, SnapshotRefused
from nomorals.storage import cutover
from nomorals.storage.backup import (
    BackupManager,
    load_document_dump,
    write_document_dump,
)
from nomorals.storage.carry import RowDocumentView, carry, run_parity_gate
from nomorals.storage.conversations import ConversationLog
from nomorals.storage.db import Database, open_database
from nomorals.storage.docstore import (
    BackendHealth,
    MongoDocumentStore,
    SQLiteDocumentStore,
    StoreResolution,
    attach_document_store,
)
from nomorals.storage.kv import KVStore
from nomorals.storage.parity import (
    check_document_parity,
    check_document_parity_streaming,
    collection_checksum,
)
from nomorals.storage.queue import WorkQueue

pymongo = pytest.importorskip("pymongo")


# ── fixtures ─────────────────────────────────────────────────────────────

def _find_mongod() -> str | None:
    env = os.environ.get("NM_MONGOD_BIN")
    if env and Path(env).exists():
        return env
    found = shutil.which("mongod")
    if found:
        return found
    candidate = Path.home() / "workspace" / "tools" / "mongodb" / "bin" / "mongod"
    return str(candidate) if candidate.exists() else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def mongo_uri(tmp_path_factory) -> str:
    # Mirrors tests/test_carry.py's fixture: a single-node replica set
    # with the census-prescribed capped WiredTiger cache (0.25 GB).
    binary = _find_mongod()
    if binary is None:
        pytest.skip("no mongod binary available for the P6 rehearsal")
    port = _free_port()
    data_dir = tmp_path_factory.mktemp("mongod-p6")
    log = open(tmp_path_factory.mktemp("mongod-p6-log") / "mongod.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [binary, "--dbpath", str(data_dir), "--bind_ip", "127.0.0.1",
         "--port", str(port), "--replSet", "rs0",
         "--wiredTigerCacheSizeGB", "0.25"],
        stdout=log, stderr=log)
    base = f"mongodb://127.0.0.1:{port}"
    try:
        direct = pymongo.MongoClient(base + "/?directConnection=true",
                                     serverSelectionTimeoutMS=500)
        deadline = time.time() + 30
        while True:
            try:
                direct.admin.command("ping")
                break
            except Exception:
                if time.time() > deadline:
                    raise RuntimeError("mongod did not come up") from None
                time.sleep(0.25)
        with contextlib.suppress(pymongo.errors.OperationFailure):
            direct.admin.command("replSetInitiate")
        client = pymongo.MongoClient(base + "/?replicaSet=rs0",
                                     serverSelectionTimeoutMS=500)
        deadline = time.time() + 30
        while True:
            try:
                if client.admin.command("hello").get("isWritablePrimary"):
                    break
            except Exception:
                pass
            if time.time() > deadline:
                raise RuntimeError("mongod replica set did not elect a primary")
            time.sleep(0.25)
        yield base + "/?replicaSet=rs0"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _doc_store() -> SQLiteDocumentStore:
    return SQLiteDocumentStore(Database(":memory:"))


def _fill(store: SQLiteDocumentStore, collection: str, n: int,
          start: int = 0) -> None:
    for i in range(start, start + n):
        store.put(collection, f"doc-{i:04d}",
                  {"n": i, "label": f"value-{i}"})


# ── streaming comparator equivalence ─────────────────────────────────────

class TestStreamingParityEquivalence:
    def test_identical_collections_identical_verdicts(self):
        a, b = _doc_store(), _doc_store()
        _fill(a, "c", 37)
        _fill(b, "c", 37)
        plain = check_document_parity(a, b, "c")
        streamed = check_document_parity_streaming(a, b, "c", page=5)
        assert plain.to_dict() == streamed.to_dict()
        assert streamed.ok

    def test_planted_divergences_identical_verdicts(self):
        a, b = _doc_store(), _doc_store()
        _fill(a, "c", 20)
        _fill(b, "c", 20)
        a.put("c", "only-a", {"n": -1})
        b.put("c", "only-b", {"n": -2})
        b.put("c", "doc-0007", {"n": 7, "label": "TAMPERED"})
        a.put("c", "doc-0009", {"n": 9, "label": "value-9"},
              expires_at=9999999999.0)
        b.put("c", "doc-0009", {"n": 9, "label": "value-9"},
              expires_at=9999999999.0)
        plain = check_document_parity(a, b, "c")
        streamed = check_document_parity_streaming(a, b, "c", page=4)
        assert plain.to_dict() == streamed.to_dict()
        assert not streamed.ok
        assert streamed.only_in_a == ["only-a"]
        assert streamed.only_in_b == ["only-b"]
        assert streamed.differing == ["doc-0007"]
        assert streamed.count_a == streamed.count_b == 21

    def test_divergence_cap_and_page_boundaries(self):
        a, b = _doc_store(), _doc_store()
        _fill(a, "c", 60)          # 60 docs only in a: lists cap at 50
        _fill(b, "c", 0)
        plain = check_document_parity(a, b, "c")
        streamed = check_document_parity_streaming(a, b, "c", page=7)
        assert plain.to_dict() == streamed.to_dict()
        assert len(streamed.only_in_a) == 50
        assert not streamed.ok and not streamed.truncated

    def test_empty_and_single_doc(self):
        a, b = _doc_store(), _doc_store()
        assert check_document_parity_streaming(a, b, "nope").ok
        _fill(a, "c", 1)
        report = check_document_parity_streaming(a, b, "c", page=1)
        assert report.only_in_a == ["doc-0000"] and not report.ok
        assert check_document_parity(a, b, "c").to_dict() == report.to_dict()

    def test_checksums_match_collection_checksum(self):
        a = _doc_store()
        _fill(a, "c", 25)
        count, checksum = collection_checksum(a, "c")
        report = check_document_parity_streaming(a, a, "c", page=3)
        assert report.count_a == count == 25
        assert report.checksum_a == checksum


# ── gate routing ─────────────────────────────────────────────────────────

class TestGateRouting:
    def _seeded(self, tmp_path: Path):
        db, _r, _n = open_database(tmp_path / "src.db")
        kv = KVStore(db)
        for i in range(30):
            kv.set(f"route-{i}", {"i": i})
        target = SQLiteDocumentStore(Database(tmp_path / "dst.db"))
        result = carry(db, target)
        assert result.ok
        return db, target

    def test_small_collections_use_in_memory(self, tmp_path):
        db, target = self._seeded(tmp_path)
        gate = run_parity_gate(db, target, record=False)
        assert gate["passed"]
        by_name = {c["collection"]: c for c in gate["collections"]}
        assert by_name["kv_store"]["comparator"] == "in_memory"
        assert all(c["comparator"] == "in_memory"
                   for c in gate["collections"])

    def test_forced_streaming_same_verdicts(self, tmp_path):
        db, target = self._seeded(tmp_path)
        plain = run_parity_gate(db, target, record=False)
        streamed = run_parity_gate(db, target, record=False,
                                   streaming_threshold=0)
        assert streamed["passed"] == plain["passed"] is True
        by_name = {c["collection"]: c for c in streamed["collections"]}
        assert by_name["kv_store"]["comparator"] == "streaming"
        for entry in streamed["collections"]:
            twin = next(c for c in plain["collections"]
                        if c["collection"] == entry["collection"])
            assert entry["ok"] == twin["ok"]
            assert entry["source_checksum"] == twin["source_checksum"]
            assert entry["target_checksum"] == twin["target_checksum"]

    def test_streaming_gate_catches_divergence(self, tmp_path):
        db, target = self._seeded(tmp_path)
        victim = target.find("kv_store", limit=1)[0]["_id"]
        assert target.delete("kv_store", victim)
        gate = run_parity_gate(db, target, record=False,
                               streaming_threshold=0)
        assert not gate["passed"]
        entry = next(c for c in gate["collections"]
                     if c["collection"] == "kv_store")
        assert entry["comparator"] == "streaming"
        assert victim in entry["only_in_source"]


# ── document dump codec + backend-aware backups ──────────────────────────

class TestDocumentDump:
    def test_round_trip_preserves_checksums_and_expiry(self, tmp_path):
        store = _doc_store()
        _fill(store, "alpha", 12)
        store.put("beta", "nested", {"a": {"b": [1, 2, 3]}, "flag": True})
        store.put("beta", "ttl", {"x": 1}, expires_at=9999999999.0)
        store.put("beta", "blob", {"content": {"$blob_b64": "aGVsbG8="}})
        dest = tmp_path / "dump.jsonl"
        summary = write_document_dump(store, dest)
        assert set(summary) == {"alpha", "beta"}
        assert summary["beta"]["count"] == 3

        scratch = _doc_store()
        counts = load_document_dump(dest, scratch)
        assert counts == {"alpha": 12, "beta": 3}
        for name in ("alpha", "beta"):
            assert collection_checksum(scratch, name)[1] == \
                summary[name]["checksum"]
        kept = scratch.get("beta", "ttl", include_expired=True)
        assert kept["expires_at"] == 9999999999.0
        assert scratch.get("beta", "nested")["a"] == {"b": [1, 2, 3]}

    def test_load_rejects_garbage(self, tmp_path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"nope": true}\n', encoding="utf-8")
        with pytest.raises(Exception):
            load_document_dump(bad, _doc_store())


def _mongo_shaped_handle(tmp_path: Path):
    """A database handle whose boot resolution says Mongo is active,
    with a SQLite-rung document store standing in as the seam store
    (unit speed; the rehearsal exercises the real Mongo store)."""
    db, _r, _n = open_database(tmp_path / "install.db")
    store = SQLiteDocumentStore(Database(tmp_path / "seam.db"))
    db.document_store = store
    db.store_resolution = StoreResolution(
        configured="mongo", active="mongo", store=store,
        mongo_health=BackendHealth("mongo", "live"),
        sqlite_health=BackendHealth("sqlite", "live"))
    return db, store


class TestBackendAwareBackups:
    def test_sqlite_install_unchanged(self, tmp_path):
        db, _r, _n = open_database(tmp_path / "plain.db")
        KVStore(db).set("k", {"v": 1})
        mgr = BackupManager(db, tmp_path / "backups")
        info = mgr.create(label="plain")
        assert info.backend == ""
        assert info.doc_collections == {}
        assert Path(info.path).name.endswith(".db.gz")
        assert mgr.verify(info) == []
        listed = mgr.list()
        assert listed and listed[-1].backend == ""

    def test_mongo_install_document_backup_cycle(self, tmp_path):
        db, store = _mongo_shaped_handle(tmp_path)
        _fill(store, "kv_store", 9)
        store.put("conversations", "c1", {"title": "hello"})
        mgr = BackupManager(db, tmp_path / "backups")
        info = mgr.create(label="mongo")
        assert info.backend == "mongo"
        assert Path(info.path).name.endswith(".docs.jsonl.gz")
        assert info.doc_collections["kv_store"]["count"] == 9
        assert info.doc_collections["conversations"]["count"] == 1
        assert mgr.verify(info) == []
        # Manifest round-trip: a fresh manager sees the backend leg.
        again = BackupManager(db, tmp_path / "backups").list()[-1]
        assert again.backend == "mongo"
        assert again.doc_collections == info.doc_collections
        # Restore through the seam into a scratch store: parity-equal.
        scratch = _doc_store()
        report = mgr.restore_documents(info, scratch)
        assert report["documents"] == 10
        for name in ("kv_store", "conversations"):
            assert collection_checksum(scratch, name)[1] == \
                info.doc_collections[name]["checksum"]
        # restore() with no store available refuses, naming the need.
        plain_db, _r, _n = open_database(tmp_path / "plain2.db")
        bare = BackupManager(plain_db, tmp_path / "backups")
        with pytest.raises(Exception, match="document store"):
            bare.restore(info.name)

    def test_verify_catches_manifest_lies(self, tmp_path):
        db, store = _mongo_shaped_handle(tmp_path)
        _fill(store, "kv_store", 4)
        mgr = BackupManager(db, tmp_path / "backups")
        info = mgr.create()
        assert mgr.verify(info) == []
        from dataclasses import replace
        lying = replace(info, doc_collections={
            "kv_store": {"count": 99,
                         "checksum": info.doc_collections["kv_store"]["checksum"]}})
        problems = mgr.verify(lying)
        assert any("kv_store" in p for p in problems)

    def test_active_mongo_without_store_fails_loudly(self, tmp_path):
        db, _r, _n = open_database(tmp_path / "broken.db")
        db.store_resolution = StoreResolution(
            configured="mongo", active="mongo", store=None,
            mongo_health=BackendHealth("mongo", "live"),
            sqlite_health=BackendHealth("sqlite", "live"))
        db.document_store = None
        mgr = BackupManager(db, tmp_path / "backups")
        with pytest.raises(Exception, match="no document store"):
            mgr.create()


# ── snapshot document leg ────────────────────────────────────────────────

class TestSnapshotDocumentsLeg:
    def _manager(self, tmp_path: Path, **kwargs) -> SnapshotManager:
        db_path = tmp_path / "state.db"
        db, _r, _n = open_database(db_path)
        db.close()
        return SnapshotManager(home=tmp_path / "home", db_path=db_path,
                               **kwargs)

    def test_sqlite_snapshot_has_no_leg(self, tmp_path):
        mgr = self._manager(tmp_path)
        snap = mgr.create(label="plain")
        assert snap.manifest["documents"] == {"status": "not-applicable"}
        assert mgr.verify(snap.id) == []

    def test_leg_round_trip_and_restore_refusal(self, tmp_path):
        store = _doc_store()
        _fill(store, "kv_store", 5)
        mgr = self._manager(tmp_path, document_store=store)
        snap = mgr.create(label="with-docs")
        leg = snap.manifest["documents"]
        assert leg["status"] == "included"
        assert leg["collections"]["kv_store"]["count"] == 5
        assert mgr.verify(snap.id) == []
        # A manager without the store refuses BEFORE swapping files.
        bare = self._manager(tmp_path)
        with pytest.raises(SnapshotRefused, match="document"):
            bare.restore(snap.id, force=True)
        # With a store attached, restore converges it to the snapshot.
        other = _doc_store()
        other.put("kv_store", "extra", {"n": -1})
        mgr2 = self._manager(tmp_path, document_store=other)
        mgr2.restore(snap.id, force=True)
        assert other.count("kv_store") == 6  # 5 restored + 1 upsert-kept
        assert other.get("kv_store", "doc-0003")["n"] == 3

    def test_unavailable_leg_fails_verification(self, tmp_path):
        mgr = self._manager(
            tmp_path, document_note="mongo configured but unreachable")
        snap = mgr.create(label="degraded")
        assert snap.manifest["documents"]["status"] == "unavailable"
        problems = mgr.verify(snap.id)
        assert any("documents leg unavailable" in p for p in problems)


# ── config flip mechanics ────────────────────────────────────────────────

class TestConfigFlip:
    def _settings(self, tmp_path: Path, monkeypatch) -> Settings:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("NM_HOME", str(home))
        cfg = home / "config.toml"
        cfg.write_text(
            '# rehearsal install\nprofile = "workstation"\n\n'
            '[storage]\nbackend = "sqlite"\n', encoding="utf-8")
        settings = Settings(home=str(home))
        settings.config_file = str(cfg)
        return settings

    def test_flip_and_reload(self, tmp_path, monkeypatch):
        settings = self._settings(tmp_path, monkeypatch)
        result = cutover.flip_config(
            settings, backend="mongo",
            mongo_uri="mongodb://127.0.0.1:29999/?replicaSet=rs0",
            mongo_db="p6flip")
        assert result["effective"]
        text = Path(result["path"]).read_text(encoding="utf-8")
        assert "# rehearsal install" in text  # untouched content kept
        fresh = load_settings(config_file=result["path"], env=os.environ)
        assert fresh.storage.backend == "mongo"
        assert fresh.storage.mongo_db == "p6flip"
        back = cutover.flip_config(settings, backend="sqlite")
        assert back["effective"]

    def test_flip_creates_section_when_absent(self, tmp_path, monkeypatch):
        settings = self._settings(tmp_path, monkeypatch)
        Path(settings.config_file).write_text(
            'profile = "workstation"\n', encoding="utf-8")
        result = cutover.flip_config(settings, backend="mongo")
        assert result["effective"]
        assert "[storage]" in Path(result["path"]).read_text(
            encoding="utf-8")

    def test_env_override_defeats_flip_and_is_named(self, tmp_path,
                                                   monkeypatch):
        settings = self._settings(tmp_path, monkeypatch)
        monkeypatch.setenv("NM_DB_BACKEND", "sqlite")
        result = cutover.flip_config(settings, backend="mongo")
        assert not result["effective"]
        assert any("NM_DB_BACKEND" in c for c in result["culprits"])

    def test_set_storage_keys_preserves_neighbours(self):
        text = ('[chat]\nfoo = 1\n\n[storage]\nbackend = "sqlite"\n'
                'path = "data/x.db"\n\n[other]\nbar = 2\n')
        out = cutover._set_storage_keys(
            text, {"backend": "mongo", "mongo_db": "nm2"})
        assert 'backend = "mongo"' in out
        assert 'path = "data/x.db"' in out
        assert 'mongo_db = "nm2"' in out
        assert "[other]\nbar = 2" in out
        assert out.index("[storage]") < out.index('mongo_db = "nm2"') \
            < out.index("[other]")


# ── dispositions + residue probe + preflight refusals ────────────────────

def _residue_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    conn = sqlite3.connect(home / "identities.db")
    conn.execute("CREATE TABLE identities (id TEXT PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO identities VALUES (?, ?)",
                     [("a", "Ada"), ("b", "Bola")])
    conn.commit()
    conn.close()
    conn = sqlite3.connect(home / "library.db")
    conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO books VALUES ('b1', 'Things Fall Apart')")
    conn.commit()
    conn.close()
    conn = sqlite3.connect(home / "mystery.db")
    conn.execute("CREATE TABLE stuff (id TEXT)")
    conn.commit()
    conn.close()
    return home


class TestDispositionsAndProbe:
    def test_probe_attributes_and_reports(self, tmp_path):
        home = _residue_home(tmp_path)
        db_path = home / "data" / "nomorals.db"
        db_path.parent.mkdir(parents=True)
        report = cutover.residue_report(home, db_path)
        ident = report["stores"]["agent-identities"]
        assert ident["has_data"]
        assert ident["files"][0]["rows"] == 2
        books = report["stores"]["books-library"]
        assert books["has_data"] and books["files"][0]["rows"] == 1
        assert [u["path"] for u in report["unattributed"]] == \
            [str(home / "mystery.db")]

    def test_record_and_read_back(self, tmp_path):
        home = _residue_home(tmp_path)
        db, _r, _n = open_database(home / "data.db")
        recorded = cutover.record_disposition(
            db, "agent-identities", "ports-later-accepted", "queued")
        assert recorded["disposition"] == "ports-later-accepted"
        assert cutover.dispositions(db)["agent-identities"]["note"] == "queued"
        with pytest.raises(ValueError):
            cutover.record_disposition(db, "no-such-store", "exported")
        with pytest.raises(ValueError):
            cutover.record_disposition(db, "backup-machinery", "exported")
        with pytest.raises(ValueError):
            cutover.record_disposition(db, "agent-identities", "whatever")
        status = cutover.cutover_status(db, SimpleNamespace(
            home_path=home, db_path=home / "data.db"))
        assert status["missing_dispositions"] == ["books-library"]
        db.close()


class _StubTarget:
    name = "stub"

    def __init__(self, facts: dict, ok: bool = True):
        self._facts = facts
        self._ok = ok

    def health(self) -> BackendHealth:
        return BackendHealth(
            "mongo", "live" if self._ok else "unreachable",
            "stub" if self._ok else "stub is down")

    def server_facts(self) -> dict:
        return dict(self._facts)


_GOOD_FACTS = {"reachable": True, "set_name": "rs0",
               "is_writable_primary": True, "version": "7.0.14",
               "storage_engine": "wiredTiger",
               "wiredtiger_cache_bytes": 256 * 1024 * 1024}


class TestPreflightRefusals:
    def _setup(self, tmp_path: Path):
        home = _residue_home(tmp_path)
        db_path = home / "nomorals.db"
        db, _r, _n = open_database(db_path)
        settings = SimpleNamespace(home_path=home, db_path=db_path)
        return db, settings

    def test_all_green_except_dispositions(self, tmp_path):
        db, settings = self._setup(tmp_path)
        pre = cutover.preflight(db, settings, _StubTarget(_GOOD_FACTS))
        assert not pre["ok"]
        assert pre["checks"]["target_reachable"]
        assert pre["checks"]["replica_set"]
        assert pre["checks"]["writable_primary"]
        assert pre["checks"]["cache_capped"]
        assert not pre["checks"]["residue_dispositions"]
        assert set(pre["missing_dispositions"]) == \
            {"agent-identities", "books-library"}
        cutover.record_disposition(db, "agent-identities", "exported")
        cutover.record_disposition(db, "books-library", "owner-data-risk")
        pre = cutover.preflight(db, settings, _StubTarget(_GOOD_FACTS))
        assert pre["ok"], pre["checks"]
        db.close()

    def test_uncapped_cache_refused(self, tmp_path):
        db, settings = self._setup(tmp_path)
        facts = dict(_GOOD_FACTS,
                     wiredtiger_cache_bytes=4 * 1024 * 1024 * 1024)
        pre = cutover.preflight(db, settings, _StubTarget(facts))
        assert not pre["checks"]["cache_capped"] and not pre["ok"]
        db.close()

    def test_no_replica_set_and_down_target_refused(self, tmp_path):
        db, settings = self._setup(tmp_path)
        facts = dict(_GOOD_FACTS, set_name=None, is_writable_primary=False)
        pre = cutover.preflight(db, settings, _StubTarget(facts))
        assert not pre["checks"]["replica_set"]
        assert not pre["checks"]["writable_primary"]
        pre = cutover.preflight(db, settings,
                                _StubTarget(_GOOD_FACTS, ok=False))
        assert not pre["checks"]["target_reachable"]
        db.close()


# ── the dress rehearsal ──────────────────────────────────────────────────

def _seed_pinned_row(db: Database, table: str, marker: str) -> None:
    """Insert one row into a pinned-domain table, filling NOT NULL
    columns from the live schema (the rehearsal proves the pinned set
    keeps reading from SQLite — the row's exact shape is not the
    subject)."""
    cols = db.query(f'PRAGMA table_info("{table}")')
    values: dict[str, object] = {}
    for col in cols:
        if col["dflt_value"] is not None:
            continue
        name, typ = col["name"], (col["type"] or "").upper()
        if col["pk"] and "INT" in typ:
            continue
        if "INT" in typ:
            values[name] = 1
        elif any(t in typ for t in ("REAL", "FLOA", "DOUB")):
            values[name] = 1.0
        elif "BLOB" in typ:
            values[name] = b"x"
        else:
            values[name] = marker if col["pk"] else f"{marker}-{name}"
    names = ", ".join(f'"{k}"' for k in values)
    marks = ", ".join("?" for _ in values)
    db.execute(f'INSERT INTO "{table}" ({names}) VALUES ({marks})',
               tuple(values.values()))


def _source_checksums(db: Database) -> dict[str, str]:
    view = RowDocumentView(db)
    out = {}
    for collection in ("kv_store", "conversations", "messages"):
        out[collection] = collection_checksum(view, collection)[1]
    return out


class TestCutoverRehearsal:
    """The full flip, on a realistic install, against a real mongod."""

    def test_full_cutover_cycle(self, tmp_path, monkeypatch, mongo_uri):
        home = tmp_path / "home"
        (home / "data").mkdir(parents=True)
        monkeypatch.setenv("NM_HOME", str(home))
        db_path = home / "data" / "nomorals.db"
        db, _r, _n = open_database(db_path)

        # Seed across KV + a ported domain + the queue + pinned domains.
        KVStore(db).set("p6:seed", {"answer": 42})
        log = ConversationLog(db)
        now = time.time()
        log.ensure_conversation("conv-p6", title="rehearsal", agent="p6",
                                channel="test", now=now)
        log.append_message(row_id="msg-p6", conversation_id="conv-p6",
                           role="user", content="hello mongo",
                           created_at=now)
        queue = WorkQueue(db)
        seeded_job = queue.enqueue("p6.pending", {"seed": True})
        _seed_pinned_row(db, "memories", "p6-memory")
        _seed_pinned_row(db, "game_wallet", "p6-wallet")
        # Residue sidecar files with data.
        conn = sqlite3.connect(home / "identities.db")
        conn.execute("CREATE TABLE identities (id TEXT PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO identities VALUES ('a', 'Ada')")
        conn.commit()
        conn.close()
        conn = sqlite3.connect(home / "library.db")
        conn.execute("CREATE TABLE books (id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO books VALUES ('b1', 'Rehearsal')")
        conn.commit()
        conn.close()

        before = _source_checksums(db)

        mongo_db_name = f"p6-{uuid.uuid4().hex[:12]}"
        settings = Settings(home=str(home))
        settings.config_file = str(home / "config.toml")
        Path(settings.config_file).write_text(
            '# rehearsal install\nprofile = "workstation"\n\n'
            '[storage]\nbackend = "sqlite"\n', encoding="utf-8")
        settings.storage.mongo_uri = mongo_uri
        settings.storage.mongo_db = mongo_db_name
        target = MongoDocumentStore(mongo_uri, mongo_db_name)

        # 1. Preflight refuses: residue dispositions unrecorded.
        pre = cutover.preflight(db, settings, target)
        assert not pre["ok"]
        assert set(pre["missing_dispositions"]) == \
            {"agent-identities", "books-library"}
        assert pre["checks"]["target_reachable"]
        assert pre["checks"]["replica_set"]
        assert pre["checks"]["writable_primary"]
        assert pre["checks"]["cache_capped"]

        # 2. Record the dispositions; preflight goes green.
        cutover.record_disposition(db, "agent-identities",
                                   "ports-later-accepted")
        cutover.record_disposition(db, "books-library", "owner-data-risk")
        assert cutover.preflight(db, settings, target)["ok"]

        # 3. Dry run: walks everything, flips nothing, records no gate.
        dry = cutover.run_cutover(db, settings, target=target,
                                  dry_run=True)
        assert dry["ok"] and not dry["flipped"]
        # The dry gate is informational: against an uncarried target
        # it truthfully reports divergence without refusing the walk.
        assert dry["dry_gate_passed"] is False
        step_status = {s["step"]: s["status"] for s in dry["steps"]}
        assert step_status["flip"] == "skipped"
        assert step_status["smoke"] == "skipped"
        assert "backend = \"sqlite\"" in Path(
            settings.config_file).read_text(encoding="utf-8")
        from nomorals.storage.carry import parity_gate_status
        assert parity_gate_status(db)["at"] is None
        assert target.count("kv_store") == 0  # dry carry moved nothing

        # 4. The real cutover.
        report = cutover.run_cutover(db, settings, target=target)
        assert report["ok"], json.dumps(report, indent=2, default=str)
        assert report["flipped"]
        assert parity_gate_status(db)["passed"]
        flipped = load_settings(config_file=settings.config_file,
                                env=os.environ)
        assert flipped.storage.backend == "mongo"
        assert flipped.storage.mongo_db == mongo_db_name

        # 5. Smoke + live evidence on the flipped install.
        smoke = cutover.smoke_test(settings)
        assert smoke["ok"], json.dumps(smoke, indent=2, default=str)
        # (The recorded gate PASS above is the full parity proof for
        # every mapped collection; these are spot checks.)
        assert target.count("kv_store") >= 1
        assert target.get("conversations", "conv-p6") is not None
        # Pinned-domain TABLES are carried like every mapped table
        # (the gate's parity covers them) — the pin governs the LIVE
        # path, which keeps serving them from SQLite (proven below).
        assert target.count("memories") == 1
        assert target.count("game_wallet") == 1
        # The seeded queue job is coordination state (bulk carry
        # excludes it by name); post-flip queue writes ride the seam.
        flipped_db, _r2, _n2 = open_database(db_path)
        try:
            resolution = attach_document_store(flipped_db, flipped)
            assert resolution.active == "mongo"
            wq = WorkQueue(flipped_db)
            new_job = wq.enqueue("p6.postflip", {"live": True})
            assert wq.get(new_job) is not None
            assert flipped_db.scalar(
                "SELECT COUNT(*) FROM work_queue WHERE id = ?",
                (new_job,), default=0) == 0  # served by Mongo, not SQLite
            # Pinned set intact in the status surface, and the pinned
            # domains still live on SQLite: a fresh pinned write lands
            # in the file and is read back from it, while the carried
            # Mongo copy does not move — the live path never consults
            # the seam for these domains.
            from nomorals.storage.docstore import storage_backend_status
            status = storage_backend_status(flipped, flipped_db)
            pinned = status["pinned_domains"]
            pinned_names = pinned.keys() if isinstance(pinned, dict) \
                else pinned
            assert "games-arena" in pinned_names
            assert "memory-recall" in pinned_names
            _seed_pinned_row(flipped_db, "game_wallet", "p6-wallet-2")
            assert flipped_db.scalar(
                "SELECT COUNT(*) FROM game_wallet") == 2
            assert target.count("game_wallet") == 1
            assert flipped_db.scalar(
                "SELECT COUNT(*) FROM memories") >= 1

            # 6. Backup on the flipped install, verify, restore to a
            # scratch Mongo database, prove parity with the live store.
            mgr = BackupManager(flipped_db, tmp_path / "backups")
            info = mgr.create(label="rehearsal")
            assert info.backend == "mongo"
            assert mgr.verify(info) == []
            scratch = MongoDocumentStore(
                mongo_uri, f"p6s-{uuid.uuid4().hex[:12]}")
            restored = mgr.restore_documents(info, scratch)
            assert restored["documents"] > 0
            live_store = flipped_db.document_store
            for collection in ("kv_store", "conversations"):
                assert collection_checksum(scratch, collection) == \
                    collection_checksum(live_store, collection)
        finally:
            flipped_db.close()

        # 7. Rollback: SQLite boots clean, all seeded data intact, and
        # the source's carried collections are byte-identical to
        # before the whole exercise (carry never wrote the source).
        back = cutover.rollback_cutover(db, settings)
        assert back["ok"]
        db2, _r3, _n3 = open_database(db_path)
        try:
            resolution = attach_document_store(
                db2, load_settings(config_file=settings.config_file,
                                   env=os.environ))
            assert resolution.active == "sqlite"
            assert KVStore(db2).get("p6:seed") == {"answer": 42}
            conv = ConversationLog(db2).get_conversation("conv-p6")
            assert conv is not None
            assert WorkQueue(db2).get(seeded_job) is not None
            assert db2.scalar("SELECT COUNT(*) FROM memories") >= 1
        finally:
            db2.close()
        assert _source_checksums(db) == before
        status = cutover.cutover_status(db, settings)
        actions = [r["action"] for r in status["recent_runs"]]
        assert actions[0] == "rollback" and "cutover" in actions
        db.close()


# ── CLI surface ──────────────────────────────────────────────────────────

class TestCutoverCli:
    def test_dispositions_via_cli(self, tmp_path, capsys):
        home = _residue_home(tmp_path)
        db, _r, _n = open_database(home / "data.db")
        settings = Settings(home=str(home))
        context = SimpleNamespace(settings=settings, db=db)
        from nomorals.cmdline.commands.db import _cmd_db

        args = SimpleNamespace(
            task=["cutover", "dispositions",
                  "--record", "agent-identities",
                  "--as", "ports-later-accepted"], json=False)
        assert _cmd_db(args, context) == 0
        out = capsys.readouterr().out
        assert "agent-identities" in out
        args = SimpleNamespace(
            task=["cutover", "dispositions",
                  "--record-missing", "--as", "exported"], json=False)
        assert _cmd_db(args, context) == 0
        capsys.readouterr()
        assert cutover.dispositions(db)["books-library"]["disposition"] \
            == "exported"
        args = SimpleNamespace(task=["cutover", "status"],
                               json=True)
        assert _cmd_db(args, context) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["missing_dispositions"] == []
        db.close()
