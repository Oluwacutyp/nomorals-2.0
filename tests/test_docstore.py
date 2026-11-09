"""Mongo program P0+P1: the DocumentStore seam, its two backends, the KV
port over both, backend resolution, shadow parity, and the doctor surface.

The conformance functions run ONE contract against every backend — native
SQL KV, KV over SQLiteDocumentStore, KV over a real mongod — so "identical
semantics" is a test result, not a claim. Mongo tests use a real
single-node-replica-set mongod started by the fixture (skipped when no
mongod binary is available); nothing here mocks the wire protocol.
"""

from __future__ import annotations

import asyncio
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

from nomorals.core.config import Settings, StorageSettings, load_settings
from nomorals.core.errors import StorageError
from nomorals.storage.db import Database
from nomorals.storage.docstore import (
    MongoDocumentStore,
    ShadowDocumentStore,
    SQLiteDocumentStore,
    attach_document_store,
    open_document_store,
    resolve_document_store,
    storage_backend_status,
)
from nomorals.storage.kv import KVStore
from nomorals.storage.parity import check_document_parity, run_kv_parity_housekeeping

pymongo = pytest.importorskip("pymongo")


# ── fixtures ─────────────────────────────────────────────────────────────────

class Clock:
    """A manually advanced clock for TTL semantics."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


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
    binary = _find_mongod()
    if binary is None:
        pytest.skip("no mongod binary available for the mongo conformance run")
    port = _free_port()
    dbpath = tmp_path_factory.mktemp("mongodata")
    # noqa: SIM115 - the handle must outlive the yield; closed in finally.
    log = open(tmp_path_factory.mktemp("mongolog") / "mongod.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [binary, "--dbpath", str(dbpath), "--bind_ip", "127.0.0.1",
         "--port", str(port), "--replSet", "rs0",
         "--wiredTigerCacheSizeGB", "0.25"],
        stdout=log, stderr=log,
    )
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
            direct.admin.command("replSetInitiate")  # may already be initiated
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


def _mongo_store(mongo_uri: str, clock: Clock | None = None) -> MongoDocumentStore:
    return MongoDocumentStore(
        mongo_uri, f"test_{uuid.uuid4().hex[:12]}",
        now_fn=clock or time.time, server_selection_timeout_ms=3000)


def _sqlite_docstore(clock: Clock | None = None) -> SQLiteDocumentStore:
    db = Database(":memory:")
    return SQLiteDocumentStore(db, now_fn=clock or time.time)


def _kv_db() -> Database:
    db = Database(":memory:")
    db.execute(
        "CREATE TABLE kv_store (key TEXT PRIMARY KEY, value TEXT NOT NULL, "
        "kind TEXT NOT NULL DEFAULT 'json', updated_at REAL NOT NULL, "
        "expires_at REAL)"
    )
    return db


# ── DocumentStore conformance (one contract, both backends) ─────────────────

def check_docstore_contract(store) -> None:
    coll = "things"
    # put/get roundtrip + managed fields
    store.put(coll, "a", {"n": 1, "s": "x"})
    doc = store.get(coll, "a")
    assert doc is not None and doc["_id"] == "a" and doc["n"] == 1
    assert doc["expires_at"] is None
    assert store.get(coll, "missing") is None
    # overwrite
    store.put(coll, "a", {"n": 2})
    assert store.get(coll, "a")["n"] == 2
    # update_fields
    assert store.update_fields(coll, "a", {"extra": True}) is True
    assert store.get(coll, "a")["extra"] is True
    assert store.update_fields(coll, "ghost", {"x": 1}) is False
    assert store.update_fields(coll, "ghost", {"x": 1}, upsert=True) is True
    with pytest.raises(StorageError):
        store.update_fields(coll, "a", {"_id": "other"})
    # inc
    assert store.inc(coll, "counter", "n", 5) == 5.0
    assert store.inc(coll, "counter", "n", 2.5) == 7.5
    store.put(coll, "texty", {"n": "abc"})
    with pytest.raises(StorageError):
        store.inc(coll, "texty", "n", 1)
    # cas: absent / match / mismatch
    assert store.cas(coll, "k1", None, {"v": 1}) is True
    assert store.cas(coll, "k1", None, {"v": 2}) is False
    assert store.cas(coll, "k1", {"v": 1}, {"v": 3}) is True
    assert store.cas(coll, "k1", {"v": 1}, {"v": 4}) is False
    assert store.get(coll, "k1")["v"] == 3
    # replace returns the live predecessor
    previous = store.replace(coll, "k1", {"v": 10})
    assert previous is not None and previous["v"] == 3
    assert store.replace(coll, "fresh", {"v": 0}) is None
    # take is read+delete
    taken = store.take(coll, "k1")
    assert taken is not None and taken["v"] == 10
    assert store.get(coll, "k1") is None
    assert store.take(coll, "k1") is None
    # delete
    store.put(coll, "doomed", {"v": 1})
    assert store.delete(coll, "doomed") is True
    assert store.delete(coll, "doomed") is False
    # find: equality, ranges (the prefix pattern), $in, paging
    for i in range(5):
        store.put("items", f"ns:{i}", {"i": i})
    store.put("items", "other:0", {"i": 99})
    ranged = store.find("items", {"_id": {"$gte": "ns:", "$lt": "ns:\uffff"}})
    assert [d["_id"] for d in ranged] == [f"ns:{i}" for i in range(5)]
    assert store.count("items", {"i": {"$in": [1, 3, 99]}}) == 3
    page1 = store.find("items", limit=2)
    page2 = store.find("items", after_id=page1[-1]["_id"], limit=2)
    assert page1[0]["_id"] != page2[0]["_id"]
    assert store.count("items") == 6
    # delete_many with a filter
    assert store.delete_many("items", {"i": {"$gte": 3, "$lt": 90}}) == 2
    assert store.count("items") == 4
    # collections + health
    assert coll in store.collections()
    assert store.health().ok


def check_docstore_expiry(store, clock: Clock) -> None:
    coll = "exp"
    store.put(coll, "short", {"v": 1}, expires_at=clock() + 10)
    assert store.get(coll, "short") is not None
    clock.advance(11)
    assert store.get(coll, "short") is None                      # lazy expiry
    assert store.get(coll, "short", include_expired=True) is not None
    # an expired doc counts as absent for CAS and replace
    assert store.cas(coll, "short", None, {"v": 2}) is True
    assert store.get(coll, "short")["v"] == 2
    # set_expiry + sweep
    assert store.set_expiry(coll, "short", clock() + 5) is True
    assert store.set_expiry(coll, "ghost", clock() + 5) is False
    store.put(coll, "ns:a", {"v": 1}, expires_at=clock() - 1)
    store.put(coll, "ns:b", {"v": 1}, expires_at=clock() - 1)
    store.put(coll, "zz:c", {"v": 1}, expires_at=clock() - 1)
    assert store.sweep_expired(coll, prefix="ns:") == 2
    clock.advance(6)  # now "short" (expiry +5) is expired too
    assert store.sweep_expired(coll) == 2  # "short" and "zz:c"
    assert store.count(coll, include_expired=True) == 0


def test_docstore_conformance_sqlite():
    clock = Clock()
    store = _sqlite_docstore(clock)
    check_docstore_contract(store)
    check_docstore_expiry(store, clock)


def test_docstore_conformance_mongo(mongo_uri):
    clock = Clock()
    store = _mongo_store(mongo_uri, clock)
    check_docstore_contract(store)
    check_docstore_expiry(store, clock)


# ── KV conformance (one contract, three implementations) ────────────────────

def check_kv_contract(kv: KVStore, clock: Clock) -> None:
    kv.set("s", "hello")
    kv.set("n", 42)
    kv.set("lst", [1, 2, 3])
    kv.set("d", {"a": 1})
    assert kv.get("s") == "hello"
    assert kv.get("n") == 42
    assert kv.get("lst") == [1, 2, 3]
    assert kv.get("missing", default="dflt") == "dflt"
    assert kv.get_int("n") == 42
    assert kv.get_str("s") == "hello"
    kv.set("b", True)
    assert kv.get_bool("b") is True
    assert kv.get_many(["s", "n", "nope"]) == {"s": "hello", "n": 42}
    # legacy plain-string tolerance on counters
    kv.set_raw("legacy", "123")
    assert kv.get_int("legacy") == 123
    assert kv.incr("legacy") == 124.0
    # counters
    assert kv.incr("count") == 1.0
    assert kv.incr("count", 4) == 5.0
    assert kv.decr("count") == 4.0
    # setnx / CAS on decoded values
    assert kv.setnx("nx", "one") is True
    assert kv.setnx("nx", "two") is False
    assert kv.compare_and_set("d", {"a": 1}, {"a": 2}) is True
    assert kv.compare_and_set("d", {"a": 1}, {"a": 3}) is False
    assert kv.get("d") == {"a": 2}
    # getset / getdel
    assert kv.getset("s", "world") == "hello"
    assert kv.get("s") == "world"
    assert kv.getdel("s") == "world"
    assert kv.exists("s") is False
    # hashes
    assert kv.hset("user", "name", "devon") == 1
    assert kv.hset("user", "name", "devon2") == 0
    kv.hincrby("user", "logins", 3)
    assert kv.hget("user", "logins") == 3
    assert kv.hgetall("user") == {"name": "devon2", "logins": 3}
    assert kv.hdel("user", "name") == 1
    # append
    kv.set("log", "a")
    assert kv.append("log", "bc") == 3
    assert kv.get("log") == "abc"
    # ttl: lazy expiry, ttl(), persist, sweep
    kv.set("temp", "x", ttl=30)
    assert 0 < kv.ttl("temp") <= 30
    clock.advance(31)
    assert kv.get("temp") is None
    assert kv.ttl("temp") == 0.0
    assert kv.delete_expired() >= 1
    kv.set("keep", "y", ttl=10)
    assert kv.persist("keep") is True
    clock.advance(100)
    assert kv.get("keep") == "y"
    assert kv.expire("missing", 10) is False
    # rename keeps the value
    kv.set("old", {"v": 9})
    assert kv.rename("old", "new") is True
    assert kv.get("new") == {"v": 9}
    assert kv.rename("old", "new2") is False
    # namespaces + prefix ops
    child = kv.namespaced("arena")
    child.set("plays", 7)
    child.set("players", 2)
    assert child.get("plays") == 7
    assert sorted(child.keys()) == ["players", "plays"]
    assert child.count() == 2
    assert "arena" in kv.namespaces()
    cursor, page = kv.scan_cursor("arena:", count=1)
    assert len(page) == 1 and cursor != ""
    cursor2, page2 = kv.scan_cursor("arena:", cursor=cursor, count=10)
    assert cursor2 == "" and len(page2) == 1
    assert child.delete_prefix("play") == 2
    assert child.count() == 0
    # set_many + scan
    assert kv.set_many({"m1": 1, "m2": "two"}) == 2
    scanned = dict(kv.scan("m"))
    assert scanned == {"m1": 1, "m2": "two"}
    stats = kv.stats()
    assert stats["keys"] >= 5


def test_kv_contract_native_sql():
    clock = Clock()
    check_kv_contract(KVStore(_kv_db(), now_fn=clock), clock)


def test_kv_contract_sqlite_docstore():
    clock = Clock()
    store = _sqlite_docstore(clock)
    check_kv_contract(KVStore(store=store, now_fn=clock), clock)


def test_kv_contract_mongo(mongo_uri):
    clock = Clock()
    store = _mongo_store(mongo_uri, clock)
    check_kv_contract(KVStore(store=store, now_fn=clock), clock)


# ── backend resolution ───────────────────────────────────────────────────────

def _settings(**storage_kwargs) -> Settings:
    settings = Settings()
    for key, value in storage_kwargs.items():
        setattr(settings.storage, key, value)
    return settings


def test_resolution_sqlite_default():
    db = _kv_db()
    resolution = resolve_document_store(_settings(), db)
    assert resolution.configured == "sqlite"
    assert resolution.active == "sqlite"
    assert resolution.degraded is False
    assert resolution.mongo_health.state == "off"
    assert isinstance(resolution.store, SQLiteDocumentStore)
    assert isinstance(open_document_store(_settings(), db), SQLiteDocumentStore)


def test_resolution_mongo_unreachable_degrades_loudly():
    settings = _settings(backend="mongo", mongo_uri="mongodb://127.0.0.1:1")
    resolution = resolve_document_store(settings, _kv_db())
    assert resolution.configured == "mongo"
    assert resolution.active == "sqlite"          # fell back, did not brick
    assert resolution.degraded is True
    assert resolution.mongo_health.state == "unreachable"
    assert "unreachable" in resolution.reason
    status = storage_backend_status(settings, _kv_db())
    assert status["mongo"]["state"] == "unreachable"   # broken != off
    assert status["active_backend"] == "sqlite"


def test_resolution_mongo_driver_missing(monkeypatch):
    import nomorals.storage.docstore as docstore
    monkeypatch.setattr(docstore, "load_optional", lambda name: None)
    settings = _settings(backend="mongo", mongo_uri="mongodb://127.0.0.1:27017")
    resolution = resolve_document_store(settings, _kv_db())
    assert resolution.active == "sqlite"
    assert resolution.degraded is True
    assert resolution.mongo_health.state == "driver_missing"


def test_resolution_profile_rule():
    caps = SimpleNamespace(low_memory=True, storage_backend="sqlite")
    settings = _settings(backend="mongo")  # no explicit URI
    resolution = resolve_document_store(settings, _kv_db(), platform=caps)
    assert resolution.configured == "sqlite"
    assert "profile_default" in resolution.reason
    # an explicit URI is honored even on a mobile-class profile
    settings2 = _settings(backend="mongo", mongo_uri="mongodb://127.0.0.1:1")
    resolution2 = resolve_document_store(settings2, _kv_db(), platform=caps)
    assert resolution2.configured == "mongo"


def test_platform_caps_autodetect():
    """The profile rule consults the real detector when no caps are passed
    (regression: a wrong import name once made this silently return None,
    which would have disabled the Termux veto in production)."""
    from nomorals.storage.docstore import _platform_caps
    caps = _platform_caps()
    assert caps is not None
    assert caps.storage_backend == "sqlite"  # this dev box: desktop default
    status = storage_backend_status(_settings(), _kv_db())
    assert status["profile_storage_default"] == "sqlite"


def test_resolution_mongo_live(mongo_uri):
    settings = _settings(backend="mongo", mongo_uri=mongo_uri,
                         mongo_db=f"test_{uuid.uuid4().hex[:10]}")
    resolution = resolve_document_store(settings, _kv_db())
    assert resolution.active == "mongo"
    assert resolution.degraded is False
    assert isinstance(resolution.store, MongoDocumentStore)
    assert resolution.mongo_health.ok


def test_attach_and_consumer_routing(mongo_uri):
    """The boot path: KVStore(db) consumers reach Mongo with no changes."""
    db = _kv_db()
    settings = _settings(backend="mongo", mongo_uri=mongo_uri,
                         mongo_db=f"test_{uuid.uuid4().hex[:10]}")
    resolution = attach_document_store(db, settings)
    assert resolution.active == "mongo"
    assert db.document_store is not None
    kv = KVStore(db)  # exactly how the 54 consumer files construct it
    assert kv.backend_name == "mongo"
    kv.set("boot:proof", {"via": "attach"})
    assert kv.get("boot:proof") == {"via": "attach"}
    # read the document back from the server directly — it really is Mongo
    raw = pymongo.MongoClient(mongo_uri)[settings.storage.mongo_db]
    doc = raw["kv_store"].find_one({"_id": "boot:proof"})
    assert doc is not None and doc["kind"] == "json"
    # and the SQLite kv_store table stayed empty: no silent dual-write
    assert db.scalar("SELECT COUNT(*) FROM kv_store") == 0


def test_attach_sqlite_default_leaves_native_path():
    db = _kv_db()
    resolution = attach_document_store(db, _settings())
    assert resolution.active == "sqlite"
    assert db.document_store is None
    kv = KVStore(db)
    assert kv.backend_name == "sql"
    kv.set("plain", 1)
    assert db.scalar("SELECT COUNT(*) FROM kv_store") == 1


def test_kv_from_settings_live_and_degraded(mongo_uri):
    db = _kv_db()
    live = _settings(backend="mongo", mongo_uri=mongo_uri,
                     mongo_db=f"test_{uuid.uuid4().hex[:10]}")
    kv = KVStore.from_settings(db, live)
    assert kv.backend_name == "mongo"
    assert kv.store_resolution.active == "mongo"
    kv.set("x", [1])

    dead = _settings(backend="mongo", mongo_uri="mongodb://127.0.0.1:1")
    kv2 = KVStore.from_settings(db, dead)
    assert kv2.backend_name == "sql"               # degraded to native rung
    assert kv2.store_resolution.degraded is True
    kv2.set("x", [2])                              # still fully functional
    assert kv2.get("x") == [2]


def test_env_rows_select_mongo(monkeypatch, tmp_path):
    monkeypatch.setenv("NM_HOME", str(tmp_path))
    monkeypatch.setenv("NM_DB_BACKEND", "mongo")
    monkeypatch.setenv("NM_MONGO_URI", "mongodb://127.0.0.1:27017/?replicaSet=rs0")
    monkeypatch.setenv("NM_MONGO_DB", "devon_test")
    monkeypatch.setenv("NM_MONGO_SHADOW", "true")
    settings = load_settings()
    assert settings.storage.backend == "mongo"
    assert settings.storage.mongo_uri.endswith("replicaSet=rs0")
    assert settings.storage.mongo_db == "devon_test"
    assert settings.storage.mongo_shadow is True


def test_config_validation_rejects_unknown_backend():
    from nomorals.core.config import ConfigError
    with pytest.raises(ConfigError):
        load_settings(overrides={"storage.backend": "couchdb"})


def test_status_masks_uri_password(mongo_uri):
    settings = _settings(
        backend="mongo",
        mongo_uri="mongodb://devon:secretpw@127.0.0.1:1/nomorals")
    status = storage_backend_status(settings, _kv_db())
    assert "secretpw" not in str(status)
    assert "***" in status["mongo_uri"]


# ── shadow mode + parity ─────────────────────────────────────────────────────

def test_shadow_dual_write_and_parity(mongo_uri):
    clock = Clock()
    primary = _mongo_store(mongo_uri, clock)
    secondary = _sqlite_docstore(clock)
    shadow = ShadowDocumentStore(primary, secondary)
    kv = KVStore(store=shadow, now_fn=clock)
    kv.set("one", {"a": 1})
    kv.set("two", "text", ttl=500)
    kv.incr("three")
    # the write really landed on BOTH rungs
    assert secondary.get("kv_store", "one")["value"] == '{"a": 1}'
    assert primary.get("kv_store", "one")["value"] == '{"a": 1}'
    assert shadow.divergences == 0
    report = check_document_parity(primary, secondary, "kv_store")
    assert report.ok, report.to_dict()
    assert report.count_a == report.count_b == 3
    assert report.checksum_a == report.checksum_b
    # plant a divergence: parity must catch it, by id
    secondary.put("kv_store", "two", {"value": "tampered", "kind": "text",
                                      "updated_at": 1.0})
    report2 = check_document_parity(primary, secondary, "kv_store")
    assert not report2.ok
    assert report2.differing == ["two"]


def test_shadow_housekeeping(mongo_uri):
    clock = Clock()
    primary = _mongo_store(mongo_uri, clock)
    secondary = _sqlite_docstore(clock)
    shadow = ShadowDocumentStore(primary, secondary)
    db = _kv_db()
    db.document_store = shadow
    db.store_resolution = SimpleNamespace(active="mongo")
    kv = KVStore(store=shadow, now_fn=clock)
    kv.set("alive", 1)
    kv.set("gone", 2, ttl=5)
    clock.advance(10)
    report = run_kv_parity_housekeeping(db, _settings(backend="mongo"))
    assert report is not None and report.ok
    assert primary.count("kv_store", include_expired=True) == 1  # swept
    # no shadow pair -> housekeeping is a no-op
    plain = _kv_db()
    assert run_kv_parity_housekeeping(plain, _settings()) is None


def test_shadow_config_wiring(mongo_uri):
    db = _kv_db()
    settings = _settings(backend="mongo", mongo_uri=mongo_uri,
                         mongo_db=f"test_{uuid.uuid4().hex[:10]}",
                         mongo_shadow=True)
    resolution = attach_document_store(db, settings)
    assert isinstance(db.document_store, ShadowDocumentStore)
    kv = KVStore(db)
    kv.set("shadowed", "yes")
    shadow = db.document_store
    assert shadow.secondary.get("kv_store", "shadowed") is not None
    assert resolution.active == "mongo"


# ── doctor surface ───────────────────────────────────────────────────────────

def _make_home_db(tmp_path: Path) -> None:
    home = tmp_path / ".nomorals"
    home.mkdir(parents=True)
    import sqlite3
    conn = sqlite3.connect(str(home / "nomorals.db"))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()


def test_doctor_reports_sqlite_by_choice(monkeypatch, tmp_path):
    from nomorals.core.verify import LiveVerifier
    monkeypatch.setenv("HOME", str(tmp_path))
    _make_home_db(tmp_path)
    verifier = LiveVerifier(config=SimpleNamespace(storage=StorageSettings()))
    check = asyncio.run(verifier.verify_database())
    assert check.passed
    assert check.details["backend"] == "sqlite"
    assert check.details["mongo_state"] == "off"


def test_doctor_reports_mongo_broken(monkeypatch, tmp_path):
    from nomorals.core.verify import LiveVerifier
    monkeypatch.setenv("HOME", str(tmp_path))
    _make_home_db(tmp_path)
    storage = StorageSettings(backend="mongo", mongo_uri="mongodb://127.0.0.1:1")
    verifier = LiveVerifier(config=SimpleNamespace(storage=storage))
    check = asyncio.run(verifier.verify_database())
    assert not check.passed
    assert check.details["mongo_state"] == "unreachable"
    assert "unreachable" in check.message.lower()
