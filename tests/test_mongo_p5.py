"""Mongo program P5: sidecar federation, analytics backend-awareness.

P5 settles the stores that live BESIDE the central database:

* **The registry** — storage.sidecars.SIDECAR_STORES classifies every
  sidecar with a verdict (ported / federated / pinned / residue /
  ephemeral / reader). The drift guard scans the tree mechanically:
  a new raw ``sqlite3.connect`` file or ``VectorStore`` site outside
  ``storage/`` that is not classified fails this suite the day it is
  written.
* **The ports** — challenge-memory, preferences and social-drafts run
  over the DocumentStore seam whenever a store is resolved, behind
  their unchanged APIs, on BOTH seam rungs (SQLite document store and
  a real mongod), with the SQLite default byte-identical. Their
  existing file data imports lazily, once, through the same mapping.
* **Analytics** — the sync source is backend-aware: for seam-served
  tables the rows come from the seam (identical output on every
  rung); the SQLite passthrough counts them honestly and refuses
  loudly to read their fossil.
* **Coherence** — storage_backend_status answers the whole picture
  in one call: backend + pinned domains + the sidecar federation.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.storage import sidecars
from nomorals.storage.db import Database, open_database
from nomorals.storage.docstore import (
    MongoDocumentStore,
    SQLiteDocumentStore,
    set_ambient_document_store,
    storage_backend_status,
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "nomorals"


@pytest.fixture(autouse=True)
def _no_ambient():
    set_ambient_document_store(None)
    yield
    set_ambient_document_store(None)


# ---------------------------------------------------------------------------
# mongod fixture — a real mongod 7.0 on an ephemeral port (P3/P4 pattern)
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
    """Start a throwaway single-node replica-set mongod (or skip)."""
    import contextlib

    pymongo = pytest.importorskip("pymongo")
    if _MONGOD is None:
        pytest.skip("mongod binary not installed")
    data = tmp_path_factory.mktemp("mongo5-data")
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
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        log.close()


def _mongo_store(mongod) -> MongoDocumentStore:
    # Short database names: a uuid-suffixed name stays far below the
    # 63-char namespace limit the P4 fixture hit.
    return MongoDocumentStore(mongod, f"mongo5{uuid.uuid4().hex[:10]}")


@pytest.fixture(params=["native", "sqlite_doc", "mongo"])
def rung(request, mongod):
    """One of the three execution rungs a port/analytics path can serve."""
    if request.param == "native":
        yield SimpleNamespace(kind="native", store=None)
    elif request.param == "sqlite_doc":
        scratch = Database(":memory:")
        yield SimpleNamespace(kind="sqlite_doc",
                              store=SQLiteDocumentStore(scratch))
        scratch.close()
    else:
        store = _mongo_store(mongod)
        yield SimpleNamespace(kind="mongo", store=store)
        store.close()


# ---------------------------------------------------------------------------
# the registry: completeness, sanity, drift guard
# ---------------------------------------------------------------------------

class TestSidecarRegistry:
    def test_no_unclassified_sidecar_modules(self):
        uncovered = sidecars.uncovered_modules(PACKAGE_ROOT)
        assert uncovered == {"connect": [], "vector_sites": []}, (
            "a module opens durable state outside the central store "
            f"without a sidecar classification: {uncovered}")

    def test_every_spec_is_complete_and_real(self):
        for name, spec in sidecars.SIDECAR_STORES.items():
            assert spec.verdict in sidecars.VERDICTS, name
            assert spec.reason.strip(), f"{name}: a verdict needs its reason"
            for module in spec.modules:
                assert (PACKAGE_ROOT / module).exists(), \
                    f"{name}: owner module {module} does not exist"
            for module, marker in spec.anchors:
                src = (PACKAGE_ROOT / module).read_text(encoding="utf-8")
                assert marker in src, \
                    f"{name}: anchor marker {marker!r} gone from {module} — reclassify"

    def test_no_module_double_classified(self):
        seen: dict[str, str] = {}
        for name, spec in sidecars.SIDECAR_STORES.items():
            for module in spec.modules:
                assert module not in seen, \
                    f"{module} claimed by both {seen[module]} and {name}"
                seen[module] = name

    def test_ported_set_is_exactly_the_p5_ports(self):
        ported = {n for n, s in sidecars.SIDECAR_STORES.items()
                  if s.verdict == "ported"}
        assert ported == {"challenge-memory", "preferences", "social-drafts"}

    def test_pinned_sidecars_follow_real_central_pins(self):
        from nomorals.storage.carry import PINNED_DOMAINS

        for name, spec in sidecars.SIDECAR_STORES.items():
            if spec.verdict == "pinned" and spec.follows:
                assert spec.follows in PINNED_DOMAINS or \
                    spec.follows == "mongo-p6-cutover", \
                    f"{name} follows unknown pin {spec.follows!r}"


class TestStatusCoherence:
    def test_one_call_answers_the_whole_federation(self):
        settings = SimpleNamespace(storage=SimpleNamespace(
            backend="sqlite", mongo_uri="", mongo_db="nomorals",
            mongo_shadow=False))
        db = Database(":memory:")
        try:
            status = storage_backend_status(settings, db)
        finally:
            db.close()
        assert status["active_backend"] == "sqlite"
        assert set(status["pinned_domains"]) == {
            "games-arena", "memory-recall", "fts"}
        federation = status["sidecars"]
        assert set(federation["stores"]) == set(sidecars.SIDECAR_STORES)
        assert sum(federation["counts"].values()) == len(sidecars.SIDECAR_STORES)
        for verdict in ("ported", "federated", "pinned", "residue"):
            assert federation["counts"][verdict] >= 1


# ---------------------------------------------------------------------------
# the ports, per rung
# ---------------------------------------------------------------------------

class TestChallengeMemoryPort:
    def _scenario(self, memory) -> dict:
        first = memory.record(
            site="https://sub.example.com/login", kind="Captcha",
            strategies=[("wait", True, 120), ("click", False, 40)],
            cleared=True, via="wait", ms=900)
        second = memory.record(
            site="example.com", kind="captcha",
            strategies=[("click", True, 30)], cleared=True, via="click",
            takeover_needed=True, ms=300)
        return {
            "ids": (first, second),
            "rank": memory.rank_strategies(
                "example.com", "captcha", ["click", "wait", "solve"]),
            "stats": memory.stats(),
            "stats_site": memory.stats(site="example.com", kind="captcha"),
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.browser.challenge_memory import ChallengeMemory

        memory = ChallengeMemory(tmp_path / "cm.db", store=rung.store)
        try:
            observed = self._scenario(memory)
        finally:
            memory.close()
        assert observed["ids"] == (1, 2)
        # wait: 1 win / 1 try at the site (smoothed 2/3) outranks
        # click's site record (0/1 → 1/3), even with its global win.
        assert observed["rank"][0] == "wait"
        assert observed["stats"] == {
            "encounters": 2, "cleared": 2, "cleared_rate": 1.0,
            "takeovers": 1, "takeover_rate": 0.5}
        assert observed["stats_site"]["encounters"] == 2

    def test_seam_shape_and_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.browser.challenge_memory import ChallengeMemory

        path = tmp_path / "cm.db"
        native = ChallengeMemory(path)
        native.record(site="example.com", kind="age",
                      strategies=[("confirm", True, 10)], cleared=True)
        native.close()

        memory = ChallengeMemory(path, store=rung.store)
        try:
            assert memory.stats()["encounters"] == 1  # imported, visible
            assert rung.store.count("challenge-memory__encounters") == 1
            doc = rung.store.find("challenge-memory__encounters")[0]
            assert doc["site"] == "example.com" and doc["cleared"] == 1
            # the AUTOINCREMENT counter was seeded by the import:
            assert memory.record(site="example.com", kind="age") == 2
        finally:
            memory.close()
        # second construction: adopted, not re-imported or duplicated
        again = ChallengeMemory(path, store=rung.store)
        try:
            assert again.stats()["encounters"] == 2
            assert rung.store.count("challenge-memory__encounters") == 2
        finally:
            again.close()


class TestPreferenceStorePort:
    def _scenario(self, store) -> dict:
        store.update_weight("music", "energy", "High", 0.5)
        store.update_weight("music", "energy", "high", 0.25)
        store.update_weight("music", "tempo", "fast", -0.5)
        store.log_signal("music", {"energy": "high"}, True, 0.5, source="t1")
        store.log_signal("music", {"tempo": "fast"}, False, -0.5, source="t2")
        store.log_signal("food", {"spice": "hot"}, True, 1.0, source="t3")
        return {
            "weight": store.get_weight("music", "energy", "HIGH"),
            "weights": store.get_weights("music"),
            "signals": [(s["domain"], s["approve"], s["delta"])
                        for s in store.signals()],
            "music_signals": len(store.signals("music")),
        }

    @staticmethod
    def _port_for(path, rung):
        # preferences is L0: the composition layer (this test) builds
        # the port and injects it — the module never imports storage.
        from nomorals.preferences.store import _PREFS_DDL
        from nomorals.storage.sidecars import sidecar_port

        return sidecar_port("preferences", _PREFS_DDL,
                            ("preferences", "preference_signals"),
                            str(path), store=rung.store)

    def test_conformance_file_variant(self, rung, tmp_path):
        from nomorals.preferences.store import PreferenceStore

        store = PreferenceStore(str(tmp_path / "prefs.db"),
                                port=self._port_for(tmp_path / "prefs.db", rung))
        observed = self._scenario(store)
        store.close()
        assert observed["weight"] == 0.75
        assert observed["weights"] == {
            "energy": {"high": 0.75}, "tempo": {"fast": -0.5}}
        assert observed["signals"] == [
            ("music", True, 0.5), ("music", False, -0.5), ("food", True, 1.0)]
        assert observed["music_signals"] == 2

    def test_json_and_memory_variants_never_port(self, rung, tmp_path):
        from nomorals.preferences.store import PreferenceStore

        port = self._port_for(tmp_path / "prefs.db", rung)
        json_store = PreferenceStore(str(tmp_path / "prefs.json"), port=port)
        assert json_store._port is None
        json_store.update_weight("food", "spice", "hot", 1.0)
        assert json_store.get_weight("food", "spice", "hot") == 1.0
        json_store.close()
        memory_store = PreferenceStore(port=port)
        assert memory_store._port is None
        memory_store.close()

    def test_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.preferences.store import PreferenceStore

        path = tmp_path / "prefs.db"
        native = PreferenceStore(str(path))
        native.update_weight("books", "pace", "slow", 2.0)
        native.log_signal("books", {"pace": "slow"}, True, 2.0)
        native.close()

        ported = PreferenceStore(str(path), port=self._port_for(path, rung))
        try:
            assert ported.get_weight("books", "pace", "slow") == 2.0
            assert len(ported.signals("books")) == 1
            doc = rung.store.find("preferences__preferences")[0]
            assert doc["_id"] == "books\x1fpace\x1fslow"
            # signal ids continue from the imported maximum:
            ported.log_signal("books", {"pace": "slow"}, True, 0.5)
            assert len(ported.signals("books")) == 2
        finally:
            ported.close()


class TestDraftQueuePort:
    def _scenario(self, queue) -> dict:
        import time as _time

        first = queue.create_draft("first post", platforms=["telegram"],
                                   metadata={"n": 1})
        _time.sleep(0.01)  # distinct created_at: pending order is pinned
        second = queue.create_draft("second post")
        queue.propose(first.id)
        queue.approve(first.id)
        queue.schedule(first.id, _time.time() + 3600)
        pending = {d.id: d for d in queue.pending()}
        queue.mark_posted(first.id)
        posted = queue.get(first.id)
        return {
            "pending_contents": sorted(d.content for d in pending.values()),
            "posted_status": posted.status,
            "posted_at_set": posted.posted_at is not None,
            "scheduled_set": pending[first.id].scheduled_time is not None,
            "platforms": pending[first.id].platforms,
            "metadata": pending[first.id].metadata,
            "pending_after": [d.content for d in queue.pending()],
            "second_status": queue.get(second.id).status,
        }

    def test_conformance(self, rung, tmp_path):
        from nomorals.social.drafts import DraftQueue

        queue = DraftQueue(tmp_path / "drafts.db", store=rung.store)
        try:
            observed = self._scenario(queue)
        finally:
            queue.close()
        assert observed["pending_contents"] == ["first post", "second post"]
        assert observed["posted_status"] == "posted"
        assert observed["posted_at_set"] is True
        assert observed["scheduled_set"] is True
        assert observed["pending_after"] == ["second post"]
        assert observed["second_status"] == "draft"

    def test_transition_graph_enforced_on_seam(self, rung, tmp_path):
        from nomorals.social.drafts import DraftQueue

        queue = DraftQueue(tmp_path / "drafts.db", store=rung.store)
        try:
            draft = queue.create_draft("x")
            with pytest.raises(ValueError):
                queue.approve(draft.id)  # draft → approved is not a step
            with pytest.raises(KeyError):
                queue.propose("nope")
        finally:
            queue.close()

    def test_lazy_import(self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung has no seam to inspect")
        from nomorals.social.drafts import DraftQueue

        path = tmp_path / "drafts.db"
        native = DraftQueue(path)
        kept = native.create_draft("from the file days", platforms=["x"])
        native.propose(kept.id)
        native.close()

        queue = DraftQueue(path, store=rung.store)
        try:
            got = queue.get(kept.id)
            assert got is not None and got.status == "pending_review"
            assert [d.id for d in queue.pending()] == [kept.id]
            doc = rung.store.get("social-drafts__social_drafts", kept.id)
            assert doc is not None and doc["content"] == "from the file days"
        finally:
            queue.close()


class TestAmbientPickup:
    def test_boot_ambient_routes_sidecars_without_signature_changes(
            self, rung, tmp_path):
        if rung.store is None:
            pytest.skip("native rung: ambient stays empty by design")
        from nomorals.social.drafts import DraftQueue

        set_ambient_document_store(rung.store)
        queue = DraftQueue(tmp_path / "ambient.db")  # no store kwarg
        try:
            assert queue._port is not None
            draft = queue.create_draft("ambient-routed")
            assert rung.store.get(
                "social-drafts__social_drafts", draft.id) is not None
        finally:
            queue.close()
            set_ambient_document_store(None)
        native = DraftQueue(tmp_path / "native.db")
        try:
            assert native._port is None
        finally:
            native.close()


# ---------------------------------------------------------------------------
# analytics backend-awareness
# ---------------------------------------------------------------------------

def _migrated_db(tmp_path) -> Database:
    db, _recovered, _backup = open_database(tmp_path / "nomorals.db")
    return db


def _watcher_rows(db: Database) -> list[dict]:
    """Two watcher rows, materialized through a native insert + read-back.

    Defaults and storage types are whatever the migrated schema really
    produces — the seam rungs must reproduce these dicts exactly.
    The fossil table is emptied again before returning, so a seam
    rung's analytics can only be reading the seam.
    """
    for i, name in enumerate(("gold", "repo")):
        db.insert("watchers", {
            "id": f"w{i + 1}", "name": name, "kind": "price",
            "interval_s": 3600.0, "severity": "info", "state": "active",
        })
    rows = [dict(r) for r in db.query("SELECT * FROM watchers ORDER BY id")]
    db.execute("DELETE FROM watchers")
    return rows


class TestAnalyticsBackendAwareness:
    def test_source_rows_equal_on_every_rung(self, rung, tmp_path):
        from nomorals.storage.analytics import analytics_source_rows
        from nomorals.storage.rowstore import attached_rows

        db = _migrated_db(tmp_path)
        try:
            expected = _watcher_rows(db)
            if rung.store is None:
                for row in expected:
                    db.insert("watchers", row)
                assert analytics_source_rows(db, "watchers") is None
                live = [dict(r) for r in db.query(
                    "SELECT * FROM watchers ORDER BY id")]
            else:
                db.document_store = rung.store
                table = attached_rows(db, "watchers")
                assert table is not None
                for row in expected:
                    table.put(row)
                live = analytics_source_rows(db, "watchers")
            assert live == expected
        finally:
            db.close()

    def test_sqlite_analytics_counts_and_guard(self, rung, tmp_path):
        from nomorals.core.errors import StorageError
        from nomorals.storage.analytics import SQLiteAnalytics
        from nomorals.storage.rowstore import attached_rows

        db = _migrated_db(tmp_path)
        try:
            expected = _watcher_rows(db)
            engine = SQLiteAnalytics(db)
            if rung.store is None:
                for row in expected:
                    db.insert("watchers", row)
                assert engine.load_table(db, "watchers") == 2
                assert len(engine.query("SELECT * FROM watchers")) == 2
            else:
                db.document_store = rung.store
                table = attached_rows(db, "watchers")
                for row in expected:
                    table.put(row)
                # the count is the seam's truth, not the fossil's 0:
                assert engine.load_table(db, "watchers") == 2
                assert engine.load_table(
                    db, "watchers", where="name = ?",
                    params=("gold",)) == 1
                # passthrough SQL over a seam-served table refuses
                # loudly; a pinned/SQLite-served table still reads:
                with pytest.raises(StorageError):
                    engine.query("SELECT * FROM watchers")
                assert engine.query("SELECT * FROM memories") == []
        finally:
            db.close()

    def test_duckdb_sync_reads_the_seam(self, rung, tmp_path):
        pytest.importorskip("duckdb")
        from nomorals.storage.analytics import DuckDBAnalytics
        from nomorals.storage.rowstore import attached_rows

        db = _migrated_db(tmp_path)
        try:
            expected = _watcher_rows(db)
            engine = DuckDBAnalytics()
            if rung.store is None:
                for row in expected:
                    db.insert("watchers", row)
            else:
                db.document_store = rung.store
                table = attached_rows(db, "watchers")
                for row in expected:
                    table.put(row)
            assert engine.load_table(db, "watchers") == 2
            names = [r["name"] for r in engine.query(
                "SELECT name FROM watchers ORDER BY name")]
            assert names == ["gold", "repo"]
            assert engine.load_table(
                db, "watchers", where="name = ?",
                params=("gold",)) == 1
        finally:
            db.close()

    def test_kv_store_rows_come_from_the_seam(self, rung, tmp_path):
        from nomorals.storage.analytics import analytics_source_rows
        from nomorals.storage.kv import KVStore

        db = _migrated_db(tmp_path)
        try:
            if rung.store is None:
                pytest.skip("kv seam shape only exists with a store attached")
            db.document_store = rung.store
            kv = KVStore(db)
            kv.set("analytics:probe", {"v": 7})
            rows = analytics_source_rows(db, "kv_store")
            assert rows is not None and len(rows) == 1
            assert rows[0]["key"].endswith("analytics:probe")
            assert rows[0]["value"] == '{"v": 7}' or rows[0]["value"] == {"v": 7}
        finally:
            db.close()
