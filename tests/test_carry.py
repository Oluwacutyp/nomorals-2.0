"""Mongo program P2: the carry tool + the parity gate.

One mapping (storage/carry.py) owns every row↔document rule; these tests
pin the mapping's shape rules on synthetic tables, pin the inventory of
the REAL migrated schema (every table mapped or excluded by name with a
reason), and drive carry + gate end-to-end against both document-store
rungs — SQLiteDocumentStore for speed and a real mongod (single-node
replica set, fixture-started like test_docstore's) for the true target.
Nothing mocks the wire protocol.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.storage.carry import (
    EXCLUDED_TABLES,
    MAPPING_VERSION,
    STATE_COLLECTION,
    RowDocumentView,
    carry,
    inventory,
    parity_gate_status,
    row_to_document,
    run_parity_gate,
)
from nomorals.storage.db import Database, open_database
from nomorals.storage.docstore import MongoDocumentStore, SQLiteDocumentStore
from nomorals.storage.kv import KVStore
from nomorals.storage.parity import ensure_shadow_parity_job

pymongo = pytest.importorskip("pymongo")


# ── fixtures ─────────────────────────────────────────────────────────────────

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
    # Mirrors tests/test_docstore.py's fixture: the proven way to bring a
    # single-node replica set up on this box.
    binary = _find_mongod()
    if binary is None:
        pytest.skip("no mongod binary available for the carry run")
    port = _free_port()
    data_dir = tmp_path_factory.mktemp("mongod-carry")
    log = open(tmp_path_factory.mktemp("mongod-carry-log") / "mongod.log", "wb")  # noqa: SIM115
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
        log.close()


@pytest.fixture()
def mongo_store(mongo_uri):
    store = MongoDocumentStore(mongo_uri, f"carrytest-{uuid.uuid4().hex[:12]}")
    yield store
    store.close()


def _synthetic_db(tmp_path) -> Database:
    """A small hand-built source exercising every mapping rule."""
    db = Database(tmp_path / "src.db")
    db.execute("CREATE TABLE by_int (id INTEGER PRIMARY KEY, label TEXT, "
               "weight REAL, payload BLOB, note TEXT)")
    db.execute("CREATE TABLE by_text (id TEXT PRIMARY KEY, expires_at REAL, "
               "body TEXT)")
    db.execute("CREATE TABLE composite (a TEXT NOT NULL, b INTEGER NOT NULL, "
               "v TEXT, PRIMARY KEY (a, b))")
    db.execute("CREATE TABLE no_pk (x TEXT, y INTEGER)")
    db.execute("CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
               "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
               "expires_at REAL)")
    db.insert("by_int", {"id": 7, "label": "seven", "weight": 1.5,
                         "payload": b"\x00\x01\xff", "note": None})
    db.insert("by_text", {"id": "t1", "expires_at": 1893456000.0, "body": "hi"})
    db.insert("composite", {"a": "x", "b": 2, "v": "cv"})
    db.insert("no_pk", {"x": "loose", "y": 3})
    db.insert("kv_store", {"key": "ns:k1", "value": '{"a": 1}', "kind": "json",
                           "updated_at": 100.0, "expires_at": None})
    db.insert("kv_store", {"key": "ns:k2", "value": "plain", "kind": "text",
                           "updated_at": 101.0,
                           "expires_at": time.time() + 3600})
    return db


# ── the mapping ──────────────────────────────────────────────────────────────

def test_mapping_rules(tmp_path):
    db = _synthetic_db(tmp_path)
    inv = {m.table: m for m in inventory(db)}
    view = RowDocumentView(db)

    by_int = view.get("by_int", "7", include_expired=True)
    assert by_int is not None
    assert by_int["label"] == "seven"
    assert by_int["weight"] == 1.5
    assert by_int["payload"] == {"$blob_b64": "AAH/"}
    assert by_int["note"] is None
    assert by_int["expires_at"] is None  # store-managed, not the row's data

    by_text = view.get("by_text", "t1", include_expired=True)
    assert by_text["sql_expires_at"] == 1893456000.0  # renamed, value kept
    assert by_text["expires_at"] is None

    composite = view.get("composite", "x\x1f2", include_expired=True)
    assert composite["v"] == "cv"

    loose = view.find("no_pk", include_expired=True)
    assert len(loose) == 1 and loose[0]["_id"].startswith("rowid:")
    assert loose[0]["x"] == "loose"

    kv = view.get("kv_store", "ns:k1", include_expired=True)
    assert kv == {"_id": "ns:k1", "value": '{"a": 1}', "kind": "json",
                  "updated_at": 100.0, "expires_at": None}
    kv2 = view.get("kv_store", "ns:k2", include_expired=True)
    assert kv2["expires_at"] is not None  # the row's TTL became managed expiry
    assert set(inv) >= {"by_int", "by_text", "composite", "no_pk", "kv_store"}


def test_row_to_document_rejects_nothing_silently(tmp_path):
    db = _synthetic_db(tmp_path)
    for mapping in inventory(db):
        if not mapping.mapped:
            assert mapping.excluded_reason  # a reason or it doesn't ship
            continue
        rows = db.query(f'SELECT rowid AS __carry_rowid, * FROM "{mapping.table}"')
        for row in rows:
            doc_id, fields, _ = row_to_document(mapping, dict(row))
            assert doc_id
            assert "_id" not in fields and "expires_at" not in fields


def test_inventory_of_real_schema_is_total(tmp_path):
    """Every table the real migrations build is mapped or named-excluded."""
    db, recovered, _ = open_database(tmp_path / "real.db")
    assert not recovered
    inv = inventory(db)
    names = {m.table for m in inv}
    for known in EXCLUDED_TABLES:
        if known in names:
            entry = next(m for m in inv if m.table == known)
            assert not entry.mapped and entry.excluded_reason
    mapped = [m for m in inv if m.mapped]
    excluded = [m for m in inv if not m.mapped]
    assert len(mapped) >= 100, f"only {len(mapped)} mapped of {len(inv)}"
    for entry in excluded:
        assert entry.excluded_reason, f"{entry.table} excluded without a reason"
    # The named P4/bookkeeping exclusions are present in the real schema.
    for expected in ("work_queue", "schedule_jobs", "schema_migrations"):
        assert expected in {m.table for m in excluded}, expected
    # kv_store is mapped (through its transform), never excluded.
    assert "kv_store" in {m.table for m in mapped}
    db.close()


# ── carry + gate against the SQLite rung (fast path) ─────────────────────────

def test_carry_and_gate_sqlite_target(tmp_path):
    src = _synthetic_db(tmp_path)
    target = SQLiteDocumentStore(Database(tmp_path / "dst.db"))

    dry = carry(src, target, dry_run=True)
    assert dry.ok and all(c.status == "dry_run" for c in dry.collections)
    assert target.collections() == []  # dry run moved nothing at all

    report = carry(src, target)
    assert report.ok, report.to_dict()
    assert all(c.status == "carried" for c in report.collections)
    assert target.get(STATE_COLLECTION, "by_int")["status"] == "done"

    again = carry(src, target)
    assert all(c.status == "verified_skip" for c in again.collections)

    status = parity_gate_status(src)
    assert status["passed"] is False and status["at"] is None  # no record yet
    gate = run_parity_gate(src, target)
    assert gate["passed"], gate
    status = parity_gate_status(src)
    assert status["passed"] is True
    assert status["mapping_version"] == MAPPING_VERSION
    assert status["divergences"] == []

    # Source loses a row: the re-carry prunes the target document.
    src.execute("DELETE FROM by_text WHERE id = 't1'")
    report = carry(src, target)
    entry = next(c for c in report.collections if c.collection == "by_text")
    assert entry.deleted == 1 and report.ok
    assert run_parity_gate(src, target)["passed"]


def test_gate_names_a_planted_divergence(tmp_path):
    src = _synthetic_db(tmp_path)
    target = SQLiteDocumentStore(Database(tmp_path / "dst.db"))
    assert carry(src, target).ok
    assert run_parity_gate(src, target)["passed"]

    target.put("by_int", "7", {"label": "tampered", "weight": 1.5,
                               "payload": {"$blob_b64": "AAH/"}, "note": None})
    gate = run_parity_gate(src, target)
    assert not gate["passed"]
    status = parity_gate_status(src)
    assert status["passed"] is False
    div = next(d for d in status["divergences"] if d["collection"] == "by_int")
    assert "7" in div["divergent_ids"]


def test_carry_resumes_after_a_killed_run(tmp_path):
    src = _synthetic_db(tmp_path)
    real_target = SQLiteDocumentStore(Database(tmp_path / "dst.db"))

    class DiesAfter:
        """A target that dies mid-collection, like a killed process."""
        name = "dies"

        def __init__(self, inner, budget):
            self._inner, self._budget = inner, budget

        def get(self, *a, **kw):
            return self._inner.get(*a, **kw)

        def find(self, *a, **kw):
            return self._inner.find(*a, **kw)

        def delete(self, *a, **kw):
            return self._inner.delete(*a, **kw)

        def put(self, collection, doc_id, doc, *, expires_at=None):
            if collection != STATE_COLLECTION:
                self._budget -= 1
                if self._budget < 0:
                    raise RuntimeError("killed mid-carry")
            return self._inner.put(collection, doc_id, doc,
                                   expires_at=expires_at)

    with pytest.raises(RuntimeError):
        carry(src, DiesAfter(real_target, budget=2))
    # The resume converges: completed state (if any) skips, the rest lands.
    report = carry(src, real_target)
    assert report.ok, report.to_dict()
    assert run_parity_gate(src, real_target)["passed"]


# ── carry + gate against a real mongod ───────────────────────────────────────

def test_carry_and_gate_mongo_target(tmp_path, mongo_store):
    src = _synthetic_db(tmp_path)
    report = carry(src, mongo_store)
    assert report.ok, report.to_dict()
    assert run_parity_gate(src, mongo_store)["passed"]

    # The documents are really in the server, in the seam's shapes.
    raw = pymongo.MongoClient(mongo_store._uri)[mongo_store._database_name]
    assert raw["by_int"].find_one({"_id": "7"})["label"] == "seven"
    kv_doc = raw["kv_store"].find_one({"_id": "ns:k1"})
    assert kv_doc["value"] == '{"a": 1}' and kv_doc["kind"] == "json"

    # A direct edit on the server is caught, by id.
    raw["by_int"].update_one({"_id": "7"}, {"$set": {"label": "planted"}})
    gate = run_parity_gate(src, mongo_store)
    assert not gate["passed"]
    div = next(d for d in parity_gate_status(src)["divergences"]
               if d["collection"] == "by_int")
    assert "7" in div["divergent_ids"]


def test_real_schema_carry_to_mongo(tmp_path, mongo_store):
    """The full 125-table shape: built by the real migrations, carried whole."""
    db, recovered, _ = open_database(tmp_path / "real.db")
    assert not recovered
    kv = KVStore(db)
    kv.set("carry:probe", {"hello": "world"})
    kv.set("carry:counter", 41)
    db.insert("conversations", {"id": "conv-1", "title": "probe",
                                "agent": "devon", "channel": "cli",
                                "created_at": 1.0, "updated_at": 2.0})
    report = carry(db, mongo_store)
    assert report.ok, [c for c in report.to_dict()["collections"]
                       if c["status"] == "diverged"]
    gate = run_parity_gate(db, mongo_store)
    assert gate["passed"], [c for c in gate["collections"] if not c["ok"]]
    db.close()


# ── the shadow-parity beat ───────────────────────────────────────────────────

class _FakeScheduler:
    def __init__(self):
        self.jobs: list[dict] = []
        self._n = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload):
        self._n += 1
        job = {"id": f"job-{self._n}", "name": name, "spec": spec,
               "kind": kind, "payload": payload}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


def _ctx(interval: float | None):
    storage = SimpleNamespace(
        shadow_parity_interval_seconds=interval)
    return SimpleNamespace(settings=SimpleNamespace(storage=storage))


def test_ensure_shadow_parity_job_registers_idempotently():
    # One scheduler instance is the durable store in production.
    shared = _FakeScheduler()
    result = ensure_shadow_parity_job(_ctx(3600.0),
                                      scheduler_factory=lambda c: shared)
    assert result["scheduled"] and result["spec"] == "every 3600s"
    assert shared.jobs[0]["payload"] == {"tool": "db_shadow_parity", "args": {}}

    again = ensure_shadow_parity_job(_ctx(3600.0),
                                     scheduler_factory=lambda c: shared)
    assert again["already_scheduled"] and len(shared.jobs) == 1

    changed = ensure_shadow_parity_job(_ctx(60.0),
                                       scheduler_factory=lambda c: shared)
    assert changed["scheduled"] and len(shared.jobs) == 1
    assert shared.jobs[0]["spec"] == "every 60s"


def test_ensure_shadow_parity_job_reports_honestly_without_factory():
    result = ensure_shadow_parity_job(_ctx(3600.0))
    assert result["scheduled"] is False
    assert "no scheduler factory" in result["reason"]
    disabled = ensure_shadow_parity_job(
        _ctx(0.0), scheduler_factory=lambda c: _FakeScheduler())
    assert disabled["scheduled"] is False and disabled["reason"] == "disabled"
