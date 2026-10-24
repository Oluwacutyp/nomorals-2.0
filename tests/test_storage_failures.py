"""Pass #105 pins: storage failures stop vanishing.

Every pin asserts the same contract: the fallback still holds AND the
failure is recorded. Ledger pins mirror #100–#104; real-site pins hit
converted handlers across the storage subsystem.
"""

from __future__ import annotations

import sqlite3
import sys
from types import SimpleNamespace

import pytest

from nomorals.storage import failures as fl
from nomorals.storage import (Failure, clear_failures, failure_count,
                              format_failures, recent_failures,
                              record_failure)


@pytest.fixture(autouse=True)
def _clean_ledger():
    fl.clear_failures()
    yield
    fl.clear_failures()


class _Boom(Exception):
    pass


class _Unprintable(Exception):
    def __str__(self):
        raise RuntimeError("nope")


def test_ledger_basics():
    record_failure("storage.test", _Boom("kaboom"))
    assert failure_count() == 1
    items = recent_failures()
    assert len(items) == 1
    assert items[0].site == "storage.test"
    assert items[0].exc_type == "_Boom"
    assert items[0].message == "kaboom"
    assert isinstance(items[0], Failure)


def test_ledger_bounded():
    for _ in range(250):
        record_failure("storage.test", _Boom("x"))
    assert failure_count() == 250
    assert len(recent_failures(1000)) == 200


def test_ledger_unprintable_exc():
    record_failure("storage.test", _Unprintable())
    assert recent_failures()[0].message.startswith("<unprintable")


def test_format_render_and_empty():
    assert format_failures() == "no storage failures recorded."
    record_failure("storage.test", _Boom("b"))
    out = format_failures()
    assert "storage failures (1 total" in out
    assert "storage.test: _Boom: b" in out


def test_package_exports():
    import nomorals.storage as st

    for name in ("Failure", "record_failure", "recent_failures",
                 "failure_count", "clear_failures", "format_failures"):
        assert hasattr(st, name), name


def test_queue_lock_column_records_and_falls_back():
    from nomorals.storage.queue import WorkQueue

    class _RaisingDB:
        def table_info(self, table):
            raise _Boom("no table yet")

    q = WorkQueue(_RaisingDB())
    assert q._has_lock_column() is False
    sites = [f.site for f in recent_failures()]
    assert "queue.WorkQueue._has_lock_column" in sites


def test_repository_decode_keeps_raw_and_records():
    from nomorals.storage.repository import Repository

    repo = Repository.__new__(Repository)
    repo.json_columns = ["meta"]
    out = repo._decode({"meta": "{not json", "name": "x"})
    assert out["meta"] == "{not json"
    assert out["name"] == "x"
    assert "repository.Repository._decode" in [f.site for f in recent_failures()]


def test_fts_suggest_returns_empty_and_records():
    from nomorals.storage.fts import FTSIndex

    class _RaisingDB:
        def query(self, *args):
            raise _Boom("no vocab")

    idx = FTSIndex.__new__(FTSIndex)
    idx.db = _RaisingDB()
    idx.table = "t"
    idx.available = True
    assert idx.suggest("te") == []
    assert "fts.FTSIndex.suggest" in [f.site for f in recent_failures()]


def test_telemetry_snapshot_records_and_defaults_hold():
    from nomorals.storage import router_telemetry as rt

    rows = [
        {"key": rt.KEY_LAST_PLAN_ERROR, "value": "{bad json", "updated_at": 0},
        {"key": rt.KEY_LAST_REEVALUATION, "value": "{bad json", "updated_at": 0},
        {"key": "gauge:x", "value": "not-a-number", "updated_at": 0},
    ]

    class _DB:
        def query(self, *args):
            return rows

    out = rt.snapshot(_DB())
    assert out["last_plan_error"] == {"error": "{bad json", "route": "", "at": 0}
    assert out["last_reevaluation"] == {"action": "{bad json", "reason": "",
                                        "goal": "", "at": 0}
    sites = [f.site for f in recent_failures()]
    assert "router_telemetry.snapshot" in sites
    assert "router_telemetry.snapshot#2" in sites
    assert "router_telemetry.snapshot#3" in sites


def test_telemetry_record_timing_records_and_sketch_updates():
    from nomorals.storage import router_telemetry as rt

    class _DB:
        def query_one(self, *args):
            return {"value": "{bad json"}

        def execute(self, *args):
            pass

    rt.record_timing(_DB(), "k", 2.0)
    assert "router_telemetry.record_timing" in [f.site for f in recent_failures()]


def test_vectors_native_fallback_records():
    from nomorals.storage.vectors import VectorStore

    real = sys.modules.get("nomorals.native")
    sys.modules["nomorals.native"] = None  # extension missing, as on some hosts
    try:
        vs = VectorStore(object(), use_native=True)
    finally:
        if real is None:
            sys.modules.pop("nomorals.native", None)
        else:
            sys.modules["nomorals.native"] = real
    assert vs._native is False
    assert "vectors.VectorStore.__init__" in [f.site for f in recent_failures()]


def test_artifacts_verify_returns_false_and_records():
    from nomorals.storage.artifacts import ArtifactStore

    store = ArtifactStore.__new__(ArtifactStore)
    store.get = lambda aid: SimpleNamespace(content_hash="abc")

    class _RaisingBlobs:
        def get_bytes(self, h):
            raise _Boom("gone")

    store.blobs = _RaisingBlobs()
    assert store.verify("anything") is False
    assert "artifacts.ArtifactStore.verify" in [f.site for f in recent_failures()]


class _StubPath:
    """A path-like whose stat()/resolve() raise OSError on demand."""

    def __init__(self, name, *, stat_raises=False, resolve_raises=False):
        self._name = name
        self._stat_raises = stat_raises
        self._resolve_raises = resolve_raises

    @property
    def name(self):
        return self._name

    @property
    def suffix(self):
        return ""

    @property
    def stem(self):
        return self._name

    def is_file(self):
        return True

    def is_dir(self):
        return True

    def stat(self):
        if self._stat_raises:
            raise OSError("stat broke")
        return SimpleNamespace(st_size=7)

    def resolve(self):
        if self._resolve_raises:
            raise OSError("resolve broke")
        return self

    def __str__(self):
        return self._name

    @property
    def parents(self):
        return ()

    def iterdir(self):
        return iter([self])

    def rglob(self, pattern):
        return iter([self])


class _StubDir:
    """A directory-like whose ``/`` yields a configured child."""

    def __init__(self, name, *, children=()):
        self._name = name
        self._children = list(children)

    def __str__(self):
        return self._name

    def __truediv__(self, other):
        return _StubDir(other, children=self._children)

    def is_dir(self):
        return True

    def iterdir(self):
        return iter(self._children)

    def rglob(self, pattern):
        return iter(self._children)

    def resolve(self):
        return self

    @property
    def parents(self):
        return ()


def test_blob_describe_records_and_counts_on():
    from nomorals.storage.blob import BlobStore

    store = BlobStore.__new__(BlobStore)
    store.TABLE = "blobs"
    store.stats = {}
    store.db = SimpleNamespace(
        query_one=lambda *a: {"n": 1, "size": 9, "stored": 9})
    store.root = _StubDir("root", children=[
        _StubPath("dead.bin", stat_raises=True)])
    out = store.describe()
    assert out["trashed_blobs"] == 1
    assert "blob.BlobStore.describe" in [f.site for f in recent_failures()]


def test_blob_orphans_records_and_continues():
    from nomorals.storage.blob import BlobStore

    store = BlobStore.__new__(BlobStore)
    store.TABLE = "blobs"
    store.db = SimpleNamespace(query=lambda *a: [])
    store.root = _StubDir("root", children=[
        _StubPath("a" * 64, resolve_raises=True)])
    out = store.orphans()
    assert out == [("a" * 64)]
    assert "blob.BlobStore.orphans" in [f.site for f in recent_failures()]


def test_db_wal_health_records_and_returns_health():
    from nomorals.storage.db import Database

    db = Database.__new__(Database)
    db.wal = True
    db.journal_mode = lambda: "wal"
    db.wal_size_bytes = lambda: 0

    def _query_one(*args):
        raise sqlite3.Error("wal gone")

    db.query_one = _query_one
    health = db.wal_health()
    assert health["busy"] == 0
    assert "db.Database.wal_health" in [f.site for f in recent_failures()]


def test_db_release_thread_connection_records_without_raising():
    from nomorals.storage import db as dbmod

    class _RaisingDB:
        def release_thread(self):
            raise _Boom("teardown broke")

    dbmod.release_thread_connection(_RaisingDB())  # must not raise
    assert "db.release_thread_connection" in [f.site for f in recent_failures()]


def test_migration_helper_records_and_skips():
    from nomorals.storage import migrations as mg
    from nomorals.storage import failures as fl2

    class _RaisingDB:
        def query(self, *args):
            raise _Boom("no schema yet")

        def execute(self, *args):
            raise AssertionError("no ALTER should run on a skipped table")

    mg._apply_skills_lessons_columns(_RaisingDB())  # must not raise
    sites = [f.site for f in fl2.recent_failures(10)]
    assert sites.count("migrations._apply_skills_lessons_columns") == 2


def test_scan_storage_subsystem_clean():
    from nomorals.tools.error_scan import scan

    files = [
        "nomorals/storage/failures.py",
        "nomorals/storage/artifacts.py",
        "nomorals/storage/backup.py",
        "nomorals/storage/blob.py",
        "nomorals/storage/db.py",
        "nomorals/storage/fts.py",
        "nomorals/storage/migrations.py",
        "nomorals/storage/queue.py",
        "nomorals/storage/repository.py",
        "nomorals/storage/router_telemetry.py",
        "nomorals/storage/vectors.py",
    ]
    rep = scan(files)
    assert [f"{f.file}:{f.line} [{f.rule}]" for f in rep.findings] == []
