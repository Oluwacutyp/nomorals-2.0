"""Mongo program P4: coordination ports, SQLite pins, price truth.

P4 settles the domains P3 declared as residue:

* **Work queue claim** — the whole :class:`WorkQueue` API runs on
  the document seam when a store is attached, with the Batch B
  lease-token fencing carried by conditional single-document
  writes (``RowTable.cas_row``). These tests prove the claim is
  exactly-once under thread contention on BOTH seam rungs
  (SQLite document store and a real mongod), that stale tokens are
  refused, and that queueing-lock dedup holds under concurrent
  enqueue.
* **Scheduler claim** — :class:`ScheduleClaimStore` ports the
  claim/complete book for ``schedule_jobs``/``schedule_runs`` with
  the same fencing; the live Scheduler stays native (its claim is
  single-process by design and the job record's other readers are
  native — the fork reason is recorded in ``carry.PORT_RESIDUE``).
* **Pins** — games/arena, the memory recall core and FTS are
  SQLite-PINNED by design (``carry.PINNED_DOMAINS``), surfaced by
  ``storage_backend_status``. The memory test proves recall keeps
  working — FTS lane included — with a Mongo store attached, and
  that no memory/FTS data ever lands in the document store.
* **Games impossibility demonstration** — the seam offers
  single-document atomicity only (``multi_doc_transactions`` is
  advertised False). The games suite pins *atomic abort
  invisibility* (a mid-settle crash leaves NOBODY paid); the demo
  below shows the seam's exactly-once protocol instead leaves a
  landed leg VISIBLE with the plan still unapplied — a different,
  weaker contract — which is why games is pinned, not ported.
* **price_snapshots** — the name collision is resolved: the live
  engine shape owns ``naija_price_snapshots`` (with legacy
  adoption), the migrations shape keeps the contested name, and
  the engine now constructs and works on a migrated database.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nomorals.storage import carry
from nomorals.storage.db import Database
from nomorals.storage.docstore import (
    MongoDocumentStore,
    SQLiteDocumentStore,
    storage_backend_status,
)
from nomorals.storage.queue import WorkQueue
from nomorals.storage.schedule_claim import ScheduleClaimStore

# ---------------------------------------------------------------------------
# mongod fixture — a real mongod 7.0 on an ephemeral port (P3 pattern)
# ---------------------------------------------------------------------------

def _find_mongod() -> str | None:
    env = os.environ.get("NM_MONGOD_BIN")
    if env and os.path.exists(env):
        return env
    found = shutil.which("mongod")
    if found:
        return found
    candidate = os.path.expanduser("~/workspace/tools/mongodb/bin/mongod")
    return candidate if os.path.exists(candidate) else None


_MONGOD = _find_mongod()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def mongod(tmp_path_factory):
    """Start a throwaway single-node replica-set mongod (or skip).

    Mirrors the test_mongo_p3_conformance fixture: the replica set
    is initiated over a direct connection, then clients connect
    with the replicaSet URI.
    """
    import contextlib

    pymongo = pytest.importorskip("pymongo")
    if _MONGOD is None:
        pytest.skip("mongod binary not installed")
    data = tmp_path_factory.mktemp("mongo4-data")
    port = _free_port()
    log = open(data / "mongod.log", "wb")  # noqa: SIM115
    proc = subprocess.Popen(
        [_MONGOD, "--dbpath", str(data), "--bind_ip", "127.0.0.1",
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
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    pytest.skip("mongod did not come up")
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
            except Exception:  # noqa: BLE001
                pass
            if time.time() > deadline:
                pytest.skip("mongod replica set did not elect a primary")
            time.sleep(0.25)
        yield base + "/?replicaSet=rs0"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(data, ignore_errors=True)


# ---------------------------------------------------------------------------
# seam-queue fixture: the same WorkQueue API over both seam rungs
# ---------------------------------------------------------------------------


def _sqlite_seam_queue(tmp_path) -> WorkQueue:
    db = Database(str(tmp_path / "seam.db"))
    db.migrate()
    db.document_store = SQLiteDocumentStore(db)
    return WorkQueue(db)


def _mongo_seam_queue(mongod_uri: str, name: str) -> WorkQueue:
    db = Database(":memory:")
    db.migrate()
    db.document_store = MongoDocumentStore(mongod_uri, name)
    return WorkQueue(db)


def _mongo_dbname(node_name: str) -> str:
    import hashlib

    return "mongo4_" + hashlib.md5(node_name.encode()).hexdigest()[:16]


@pytest.fixture(params=["sqlite-doc", "mongo"])
def seam_queue(request, tmp_path, mongod):
    if request.param == "sqlite-doc":
        queue = _sqlite_seam_queue(tmp_path)
    else:
        queue = _mongo_seam_queue(mongod, _mongo_dbname(request.node.name))
    yield queue
    store = queue.db.document_store
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()


def _run_claimers(queue: WorkQueue, topic: str, n_threads: int,
                  batch: int = 1) -> list[str]:
    winners: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()

    def claim(i: int) -> None:
        try:
            got = queue.lease(topic, f"claimer-{i}", batch=batch)
            with lock:
                winners.extend(job.id for job in got)
        except Exception as exc:  # noqa: BLE001 - surfaced by assert
            with lock:
                errors.append(repr(exc))

    threads = [threading.Thread(target=claim, args=(i,))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], errors
    return winners


class TestQueueClaimOnTheSeam:
    """Item 1: the claim is exactly-once on both seam rungs."""

    def test_competing_claimers_exactly_one_winner_per_job(
            self, seam_queue: WorkQueue) -> None:
        ids = [seam_queue.enqueue("race", {"i": i}) for i in range(4)]
        winners = _run_claimers(seam_queue, "race", 16)
        assert sorted(winners) == sorted(ids)

    def test_single_job_many_claimers_one_winner(
            self, seam_queue: WorkQueue) -> None:
        job_id = seam_queue.enqueue("solo", {})
        winners = _run_claimers(seam_queue, "solo", 12)
        assert winners == [job_id]

    def test_stale_token_completion_refused(
            self, seam_queue: WorkQueue) -> None:
        job_id = seam_queue.enqueue("fenced", {"x": 1})
        # A fabricated token never completes a leased job.
        job = seam_queue.lease_one("fenced", "w1", lease_seconds=0.05)
        assert job is not None
        assert seam_queue.complete(job_id, "nope", lease_token="bogus") \
            is False
        assert seam_queue.fail(job_id, "nope", lease_token="bogus") \
            == "stale"
        # The lease expires; a second owner reclaims. The first
        # owner's token is now stale and fenced out (Batch B).
        time.sleep(0.08)
        assert seam_queue.reclaim_expired() >= 1
        job2 = seam_queue.lease_one("fenced", "w2")
        assert job2 is not None and job2.lease_owner == "w2"
        assert job2.lease_token != job.lease_token
        assert seam_queue.complete(job_id, "late", lease_token=job.lease_token) \
            is False
        assert seam_queue.complete(job_id, {"ok": True},
                                   lease_token=job2.lease_token) is True
        assert seam_queue.result_of(job_id) == {"ok": True}

    def test_queueing_lock_dedup_under_concurrent_enqueue(
            self, seam_queue: WorkQueue) -> None:
        # Warm the seam mapping on this thread first (schema reads
        # belong to the owning thread; the race under test is the
        # lock claim itself, which is all store operations).
        seam_queue.enqueue("warmup", {})
        results: list[str] = []
        errors: list[str] = []
        lock = threading.Lock()

        def produce(i: int) -> None:
            try:
                jid = seam_queue.enqueue("dedup", {"i": i},
                                         queueing_lock="one-live")
                with lock:
                    results.append(jid)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(repr(exc))

        threads = [threading.Thread(target=produce, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], errors
        assert len(set(results)) == 1
        assert seam_queue.pending("dedup") == 1
        # Terminal transition releases the lock for the next live job.
        job = seam_queue.lease_one("dedup", "w")
        assert job is not None
        assert seam_queue.complete(job.id, None, lease_token=job.lease_token)
        fresh = seam_queue.enqueue("dedup", {"i": 99},
                                   queueing_lock="one-live")
        assert fresh != job.id

    def test_lifecycle_parity(self, seam_queue: WorkQueue) -> None:
        q = seam_queue
        q.backoff_base = 0.0  # retry backoff would park the job (native too)
        job_id = q.enqueue("life", [1, 2], max_attempts=2)
        assert q.get(job_id) is not None
        first = q.lease_one("life", "w")
        assert first is not None and first.attempts == 1
        assert q.fail(job_id, "try again", lease_token=first.lease_token) \
            == "retrying"
        second = q.lease_one("life", "w")
        assert second is not None and second.attempts == 2
        assert q.fail(job_id, "final", lease_token=second.lease_token) \
            == "dead"
        assert [j.id for j in q.dead_jobs("life")] == [job_id]
        assert q.retry_dead(job_id) is True
        third = q.lease_one("life", "w")
        assert third is not None
        assert q.cancel(job_id) is True
        assert q.get(job_id).status == "cancelled"
        assert q.pending("life") == 0

    def test_recurring_tick_fires_once_per_period(
            self, seam_queue: WorkQueue) -> None:
        q = seam_queue
        q.schedule_recurring("beat", "heartbeat", "* * * * *")
        listed = q.list_recurring()
        assert [r["name"] for r in listed] == ["beat"]
        nxt = float(listed[0]["next_fire_at"])
        fired = q.tick_recurring(now=nxt + 1)
        assert fired == ["beat"]
        assert q.pending("heartbeat") == 1
        # Re-ticking inside the same period must not double-fire:
        # the fired job still holds the queueing lock.
        assert q.tick_recurring(now=nxt + 1) == []
        job = q.lease_one("heartbeat", "w")
        assert job is not None
        assert q.complete(job.id, None, lease_token=job.lease_token)
        assert q.tick_recurring(now=nxt + 61) == ["beat"]
        assert q.unschedule_recurring("beat") is True
        assert q.list_recurring() == []


class TestNativePathUntouched:
    """No store attached → the native SQLite queue, byte-for-byte."""

    def test_no_store_means_native(self, tmp_path) -> None:
        db = Database(str(tmp_path / "native.db"))
        db.migrate()
        queue = WorkQueue(db)
        assert queue._queue_rows() is None
        job_id = queue.enqueue("t", {"a": 1})
        job = queue.lease_one("t", "w")
        assert job is not None and job.id == job_id
        assert queue.complete(job_id, 7, lease_token=job.lease_token)
        assert queue.result_of(job_id) == 7


# ---------------------------------------------------------------------------
# Schedule claim book
# ---------------------------------------------------------------------------


def _job_row(job_id: str, next_run: float) -> dict:
    now = time.time()
    return {"id": job_id, "name": job_id, "kind": "interval",
            "spec": "5m", "payload_kind": "tool", "payload": "{}",
            "enabled": 1, "next_run": next_run, "last_run": 0.0,
            "last_result": "", "created_at": now, "updated_at": now}


@pytest.fixture(params=["sqlite-doc", "mongo"])
def claim_book(request, tmp_path, mongod):
    db = Database(str(tmp_path / "claims.db")) \
        if request.param == "sqlite-doc" else Database(":memory:")
    db.migrate()
    if request.param == "sqlite-doc":
        store: object = SQLiteDocumentStore(db)
    else:
        store = MongoDocumentStore(
            mongod, _mongo_dbname("claims_" + request.node.name))
    book = ScheduleClaimStore(db, store)
    yield book
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()


class TestScheduleClaim:
    def test_exactly_one_claimer(self, claim_book: ScheduleClaimStore):
        now = time.time()
        claim_book.put_job(_job_row("j1", now - 1))
        assert [j["id"] for j in claim_book.due_jobs(now)] == ["j1"]
        tokens: list[str | None] = []
        errors: list[str] = []
        lock = threading.Lock()

        def claim(i: int) -> None:
            try:
                token = claim_book.claim("j1", f"w{i}", now=now)
                with lock:
                    tokens.append(token)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(repr(exc))

        threads = [threading.Thread(target=claim, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], errors
        winners = [t for t in tokens if t]
        assert len(winners) == 1
        # Fencing: only the winner's token completes, and the
        # completion records the run.
        assert claim_book.complete("j1", "bogus", {"next_run": now + 60}) \
            is False
        run = {"id": "run-1", "job_id": "j1", "started_at": now,
               "finished_at": now + 0.5, "seconds": 0.5, "ok": 1,
               "result": "ok", "retry_attempt": 0,
               "trigger_source": "tick"}
        assert claim_book.complete(
            "j1", winners[0],
            {"next_run": now + 300, "last_run": now,
             "last_result": "ok", "updated_at": now}, run=run) is True
        assert [r["id"] for r in claim_book.runs_for("j1")] == ["run-1"]
        assert claim_book.due_jobs(now) == []

    def test_expired_claim_reclaimable_and_fenced(
            self, claim_book: ScheduleClaimStore):
        now = time.time()
        claim_book.put_job(_job_row("j2", now - 1))
        first = claim_book.claim("j2", "w1", lease_seconds=0.05, now=now)
        assert first
        assert claim_book.claim("j2", "w2", now=now) is None
        second = claim_book.claim("j2", "w2", now=now + 1)
        assert second and second != first
        assert claim_book.complete("j2", first, {"next_run": now + 9}) \
            is False
        assert claim_book.release("j2", second) is True
        assert claim_book.claim("j2", "w3", now=now + 1)


# ---------------------------------------------------------------------------
# Pins: registry, status surfacing, memory/FTS coherence
# ---------------------------------------------------------------------------


class TestPinnedRegistry:
    def test_domains_declared_and_disjoint(self) -> None:
        pinned = set(carry.PINNED_TABLES)
        assert {"games-arena", "memory-recall", "fts"} <= set(
            carry.PINNED_DOMAINS)
        # The games settlement graph and the recall core are pinned.
        for table in ("game_wallet", "game_ledger",
                      "game_settlements", "game_elo", "game_players"):
            assert table in pinned, table
            assert carry.PINNED_TABLES[table].startswith("[games-arena]")
        for table in ("memories", "memory_links", "embeddings"):
            assert table in pinned, table
            assert carry.PINNED_TABLES[table].startswith("[memory-recall]")
        # A table is pinned, ported, residual or carried — never two.
        assert not (pinned & set(carry.PORTED_TABLES))
        assert not (pinned & set(carry.PORT_RESIDUE))
        for spec in carry.PINNED_DOMAINS.values():
            assert spec["tables"] or spec["reason"]
            assert spec["reason"]

    def test_coordination_residue_recorded(self) -> None:
        # The scheduler's live path stays native in P4 with the fork
        # reason on record; its claim book is the seam component.
        for table in ("schedule_jobs", "schedule_runs"):
            assert table in carry.PORT_RESIDUE
            assert "claim" in carry.PORT_RESIDUE[table].lower()
        assert "work_queue" in carry.PORTED_TABLES

    def test_status_surfaces_the_pinned_set(self, tmp_path) -> None:
        from nomorals.core.config import Settings

        db = Database(str(tmp_path / "status.db"))
        db.migrate()
        status = storage_backend_status(Settings(), db)
        assert status["active_backend"] == "sqlite"
        domains = status["pinned_domains"]
        assert {"games-arena", "memory-recall", "fts"} <= set(domains)
        assert "memories" in domains["memory-recall"]["tables"]
        from nomorals.memory.manager import MemoryManager
        assert MemoryManager.STORAGE_PIN == "sqlite"

    def test_memory_recall_coherent_with_mongo_attached(
            self, mongod: str, tmp_path) -> None:
        """Recall (FTS lane included) reads SQLite with Mongo active;
        no memory or FTS data ever lands in the document store."""
        from nomorals.memory.manager import MemoryManager

        db = Database(str(tmp_path / "mem.db"))
        db.migrate()
        store = MongoDocumentStore(mongod, "mongo4_memory")
        db.document_store = store
        try:
            manager = MemoryManager(SimpleNamespace(db=db),
                                    vector_backend="legacy")
            record_id = manager.remember(
                "The xylophone protocol rehearses every Friday.",
                kind="episode")
            result = manager.recall("xylophone", limit=5)
            assert any(r.id == record_id for r in result.records)
            # The FTS lane itself is the same SQLite index.
            from nomorals.storage.fts import FTSIndex
            hits = FTSIndex(db, "memories_fts",
                            columns=["content"]).search("xylophone")
            assert hits
            # Nothing memory-shaped crossed into the document store.
            assert store.count("memories") == 0
            assert store.count("memories_fts") == 0
        finally:
            store.close()


# ---------------------------------------------------------------------------
# Games: the demonstrated seam gap behind the pin (option b)
# ---------------------------------------------------------------------------


def _seam_stores(tmp_path, mongod_uri: str | None):
    """One store per rung for the settlement-gap demonstration."""
    db = Database(str(tmp_path / "games.db"))
    db.migrate()
    yield "sqlite-doc", SQLiteDocumentStore(db)
    if mongod_uri is not None:
        store = MongoDocumentStore(mongod_uri, "mongo4_games")
        yield "mongo", store
        store.close()


class TestGamesSeamGap:
    def test_seam_advertises_no_multi_doc_transactions(
            self, tmp_path, mongod) -> None:
        # The portable contract is set by the weakest rung: Mongo —
        # the target backend — offers single-document atomicity
        # only. (The SQLite document rung inherits real SQL
        # transactions, but a protocol that must run identically on
        # both rungs cannot rely on them.)
        for name, store in _seam_stores(tmp_path, mongod):
            caps = store.capabilities
            assert caps["atomic_single_doc"] is True
            if name == "mongo":
                assert caps["multi_doc_transactions"] is False

    def test_partial_settlement_is_visible_on_the_seam(
            self, tmp_path, mongod) -> None:
        """The (a)-protocol's failure mode, demonstrated on both rungs.

        Settlement on the seam can only be: plan document first,
        then one conditional write per leg. A crash after leg 1
        leaves leg 1's credit VISIBLE while the plan still reads
        ``planned``. The games suite pins the opposite on SQL (a
        mid-settle crash leaves NOBODY paid) — that atomic abort
        invisibility needs a multi-document transaction the seam
        does not have. Hence the pin, not a port.
        """
        for _name, store in _seam_stores(tmp_path, mongod):
            plans, wallets = "settlement_plans", "wallets"
            store.put(plans, "plan-1", {
                "_id": "plan-1", "status": "planned",
                "legs": [{"wallet": "w-a", "amount": 50},
                         {"wallet": "w-b", "amount": 50}]})
            store.put(wallets, "w-a", {"_id": "w-a", "balance": 0})
            store.put(wallets, "w-b", {"_id": "w-b", "balance": 0})

            def apply_legs_then_crash(store=store, wallets=wallets) -> None:
                # Leg 1 lands as its own atomic document write…
                store.inc(wallets, "w-a", "balance", 50)
                raise RuntimeError("simulated crash mid-settle")

            with pytest.raises(RuntimeError):
                apply_legs_then_crash()
            # …and stays visible: exactly-once recovery can avoid
            # paying twice, but it cannot un-show the partial state.
            assert store.get(wallets, "w-a")["balance"] == 50
            assert store.get(wallets, "w-b")["balance"] == 0
            assert store.get(plans, "plan-1")["status"] == "planned"


# ---------------------------------------------------------------------------
# price_snapshots: the collision resolved to the live shape
# ---------------------------------------------------------------------------


class TestPriceSnapshotsTruth:
    def test_engine_constructs_on_a_migrated_database(
            self, tmp_path) -> None:
        from nomorals.integrations.naija_shopping import NaijaShoppingEngine, PriceSnapshot

        db = Database(str(tmp_path / "shop.db"))
        db.migrate()
        # Before P4 this construction raised (its index DDL hit the
        # migrations-shaped table, which has no product_url).
        engine = NaijaShoppingEngine(db, MagicMock())
        assert db.table_exists("naija_price_snapshots")
        # The migrations shape keeps the contested name, untouched.
        cols = {r["name"] for r in db.table_info("price_snapshots")}
        assert "listing_id" in cols and "product_url" not in cols
        engine._store_snapshot(PriceSnapshot(
            product_url="https://jumia.example/p/1", site="jumia",
            price_ngn=4200.0, title="Rice 50kg",
            timestamp=time.time()))
        rows = db.query(
            "SELECT price_ngn FROM naija_price_snapshots "
            "WHERE product_url = ?", ("https://jumia.example/p/1",))
        assert [r["price_ngn"] for r in rows] == [4200.0]

    def test_legacy_engine_table_is_adopted_not_dropped(
            self, tmp_path) -> None:
        from nomorals.integrations.naija_shopping import NaijaShoppingEngine

        db = Database(str(tmp_path / "legacy.db"))
        db.execute(
            """CREATE TABLE price_snapshots (
                 product_url TEXT NOT NULL, site TEXT NOT NULL,
                 price_ngn REAL NOT NULL, title TEXT NOT NULL,
                 timestamp REAL NOT NULL,
                 in_stock INTEGER NOT NULL DEFAULT 1,
                 original_price REAL NOT NULL DEFAULT 0,
                 discount_badge REAL NOT NULL DEFAULT 0,
                 PRIMARY KEY (product_url, timestamp))""")
        db.execute(
            "INSERT INTO price_snapshots VALUES "
            "('https://x.example/1', 'jumia', 100.0, 't', 5.0, 1, 0, 0)")
        NaijaShoppingEngine(db, MagicMock())
        rows = db.query("SELECT price_ngn FROM naija_price_snapshots")
        assert [r["price_ngn"] for r in rows] == [100.0]
        assert not db.table_exists("price_snapshots")
