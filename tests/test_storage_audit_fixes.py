"""Regression tests for the 2026-10-03 storage audit fixes.

Each test pins down a real defect found in ``nomorals/storage/``:

* vectors: ``topk_native`` was a documented stub (removed); ``search()``
  silently truncated mismatched-dimension queries via ``zip`` in the
  pure-python path while numpy raised — now it fails fast everywhere.
* blob: ``put_stream`` ignored the compression policy (never compressed),
  unlike ``put_bytes``/``put_file``.
* backup: ``create()`` checkpointed *after* the snapshot although the code
  comment (and the self-contained-snapshot invariant) said "checkpoint
  first"; ``push_to_git``'s fresh-remote fallback crashed with
  ``FileNotFoundError`` because ``git init`` ran in a not-yet-created cwd.
* db: ``executemany`` dropped the ``OperationalError -> retryable`` mapping
  that ``execute`` had.
"""

from __future__ import annotations

import io
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path

from nomorals.core.errors import StorageError, ValidationError
from nomorals.storage.backup import BackupManager
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database
from nomorals.storage.vectors import VectorStore


class StubRemovalTest(unittest.TestCase):
    def test_topk_native_stub_is_gone(self) -> None:
        import nomorals.storage.vectors as vectors

        self.assertFalse(
            hasattr(vectors, "topk_native"),
            "the documented stub must not exist; use VectorStore.search",
        )


class VectorDimValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()
        # Pure-python path: deterministic, no numpy involved.
        self.store = VectorStore(self.db, use_numpy=False)

    def tearDown(self) -> None:
        self.db.close()

    def test_search_rejects_query_with_wrong_dimension(self) -> None:
        self.store.put([1.0, 0.0, 0.0], owner_type="m", owner_id="a")
        with self.assertRaises(ValidationError):
            self.store.search([1.0, 0.0])  # 2 dims vs stored 3

    def test_search_rejects_mixed_dimensions_in_table(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        # Bypass put() to simulate a corrupted/foreign-mixed table.
        self.db.execute(
            "INSERT INTO embeddings (id, owner_type, owner_id, model, dim, norm, vector, created_at)"
            " VALUES ('x', 'm', 'b', 'default', 3, 1.0, ?, 0.0)",
            (sqlite3.Binary(b"\x00" * 12),),
        )
        self.store.invalidate()
        with self.assertRaises(StorageError):
            self.store.search([1.0, 0.0])

    def test_search_still_works_when_dims_match(self) -> None:
        self.store.put([1.0, 0.0], owner_type="m", owner_id="a")
        self.store.put([0.0, 1.0], owner_type="m", owner_id="b")
        hits = self.store.search([1.0, 0.0])
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].owner_id, "a")
        self.assertGreater(hits[0].score, hits[1].score)


class BlobPutStreamCompressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()
        self.tmp = tempfile.TemporaryDirectory()
        self.store = BlobStore(self.db, Path(self.tmp.name) / "blobs")

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_stream_compresses_like_put_bytes(self) -> None:
        data = (b'{"key": "value", "padding": "' + b"x" * 6000 + b'"}')
        info = self.store.put_stream(io.BytesIO(data), mime="application/json")
        self.assertTrue(info.compressed, "stream path must honor the compression policy")
        self.assertLess(info.stored, info.size)
        self.assertEqual(self.store.get_bytes(info.sha256), data)
        # Same bytes through put_bytes dedupe to the same blob.
        again = self.store.put_bytes(data, mime="application/json")
        self.assertEqual(again.sha256, info.sha256)

    def test_stream_leaves_incompressible_bytes_alone(self) -> None:
        import os

        data = os.urandom(6000)
        info = self.store.put_stream(io.BytesIO(data), mime="application/octet-stream")
        self.assertFalse(info.compressed)
        self.assertEqual(self.store.get_bytes(info.sha256), data)


class BackupCheckpointOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()
        self.tmp = tempfile.TemporaryDirectory()
        self.manager = BackupManager(self.db, Path(self.tmp.name) / "backups")

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_snapshot_is_self_contained(self) -> None:
        self.db.execute("INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', 0)",
                        ("probe", "1"))
        info = self.manager.create(label="order")
        self.assertEqual(self.manager.verify(info), [])
        probe = Path(self.tmp.name) / "probe.db"
        self.manager.restore(info, target=probe)
        conn = sqlite3.connect(str(probe))
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            value = conn.execute("SELECT value FROM kv_store WHERE key='probe'").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(mode.lower(), "delete",
                         "checkpoint must precede the snapshot: no -wal sidecar needed")
        self.assertEqual(value, "1")


class PushToGitFreshRepoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.manager = BackupManager(self.db, self.dir / "backups")
        # A bare remote with no branches: forces the broken-then-fixed
        # "clone fails -> init locally" fallback in push_to_git().
        self.remote = self.dir / "remote.git"
        subprocess.run(["git", "init", "--bare", str(self.remote)],
                       check=True, capture_output=True)

    def tearDown(self) -> None:
        self.db.close()
        self.tmp.cleanup()

    def test_push_to_brand_new_remote_returns_dict_not_exception(self) -> None:
        manager = BackupManager(self.db, self.dir / "backups2", git_repo=str(self.remote))
        info = manager.create(label="gitprobe")
        result = manager.push_to_git(backup=info)
        self.assertTrue(result["pushed"], f"push failed: {result}")
        self.assertEqual(result["backup"], info.name)
        # The backup really landed on the branch in the remote.
        ls = subprocess.run(
            ["git", "--git-dir", str(self.remote), "ls-tree", "backups", "--name-only"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn(info.name, ls.stdout)


class ExecutemanyRetryableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = Database(":memory:")
        self.db.migrate()

    def tearDown(self) -> None:
        self.db.close()

    def test_operational_error_marks_retryable(self) -> None:
        real_conn = self.db._connection()

        class _LockedProxy:
            """Delegates everything except executemany, which reports 'locked'."""

            def __init__(self, inner: sqlite3.Connection) -> None:
                self._inner = inner

            def executemany(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                raise sqlite3.OperationalError("database is locked")

            def __getattr__(self, name: str):
                return getattr(self._inner, name)

        original = Database._connection
        self.db._connection = lambda: _LockedProxy(real_conn)  # type: ignore[method-assign]
        try:
            with self.assertRaises(StorageError) as ctx:
                self.db.executemany("INSERT INTO kv_store (key, value, kind, updated_at)"
                                    " VALUES (?, ?, 'json', 0)",
                                    [("k", "v")])
        finally:
            del self.db._connection  # restore the class-level method
        self.assertTrue(ctx.exception.retryable)
        self.assertIs(Database._connection, original)

    def test_integrity_error_still_maps_to_constraint_violation(self) -> None:
        from nomorals.core.errors import ConstraintViolation

        self.db.execute("INSERT INTO kv_store (key, value, kind, updated_at)"
                        " VALUES ('dup', '1', 'json', 0)")
        with self.assertRaises(ConstraintViolation):
            self.db.executemany("INSERT INTO kv_store (key, value, kind, updated_at)"
                                " VALUES (?, ?, 'json', 0)",
                                [("dup", "2")])


if __name__ == "__main__":
    unittest.main()
