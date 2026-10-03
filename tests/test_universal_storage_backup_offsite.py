"""Tests for backup offsite shipping (BackupManager.push_to_s3) and the
``open_blob_store("auto")`` backend selector.

The S3 round trip here runs against the spec-faithful fake S3 endpoint from
``test_universal_storage_s3`` (independent SigV4 verifier, no network), so
the backup bytes really travel the signed HTTP path end to end.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from nomorals.core.errors import ConfigError, StorageError
from nomorals.storage.backup import BackupManager
from nomorals.storage.blob import BlobInfo, BlobStore
from nomorals.storage.db import Database
from nomorals.storage.s3blob import S3BlobStore, S3Config, open_blob_store
from tests.test_universal_storage_s3 import (
    BUCKET,
    TEST_KEY,
    TEST_REGION,
    TEST_SECRET,
    FakeS3Server,
)

_S3_ENV_KEYS = (
    "S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY",
    "S3_REGION", "S3_PREFIX", "S3_PATH_STYLE", "S3_TIMEOUT",
)


class RecordingStore:
    """Minimal blob-API double: records uploads, dedups by content hash."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.put_calls = 0

    def put_file(self, source: Any, *, mime: str = "", **_: Any) -> BlobInfo:
        data = Path(str(source)).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.put_calls += 1
        if digest not in self.files:
            self.files[digest] = data
        import time

        return BlobInfo(digest, len(data), mime, False, len(data), 1, time.time())


class CorruptStore(RecordingStore):
    """put_file that lies about the content hash — must fail fast."""

    def put_file(self, source: Any, *, mime: str = "", **_: Any) -> BlobInfo:
        info = super().put_file(source, mime=mime)
        return BlobInfo(
            "f" * 64, info.size, info.mime, info.compressed,
            info.stored, info.refcount, info.created_at,
        )


class BackupOffsiteFixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = Database(root / "live.db")
        self.db.migrate()
        self.db.insert(
            "conversations",
            {"id": "c1", "title": "offsite", "created_at": 1.0, "updated_at": 1.0},
        )
        blobs = root / "blobs"
        blobs.mkdir()
        (blobs / "a.bin").write_bytes(b"offsite-blob-a" * 100)
        self.manager = BackupManager(
            self.db, root / "backups", keep=3,
            include_blobs=True, blob_dir=blobs,
        )

    def cleanup(self) -> None:
        self.db.close()
        self.tmp.cleanup()


class TestPushToS3(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = BackupOffsiteFixture()

    def tearDown(self) -> None:
        self.fx.cleanup()

    def test_push_uploads_db_and_sidecar_with_integrity(self) -> None:
        info = self.fx.manager.create(label="offsite")
        store = RecordingStore()
        result = self.fx.manager.push_to_s3(store)
        self.assertTrue(result["pushed"])
        self.assertEqual(result["backup"], info.name)
        self.assertEqual(result["sha256"], info.sha256)
        shipped = {obj["file"]: obj for obj in result["objects"]}
        self.assertIn(info.name, shipped)
        self.assertIn(info.blob_name, shipped)
        # The bytes that arrived are byte-identical to the backup on disk.
        self.assertEqual(
            store.files[shipped[info.name]["sha256"]],
            Path(info.path).read_bytes(),
        )
        self.assertEqual(
            store.files[shipped[info.blob_name]["sha256"]],
            (self.fx.manager.directory / info.blob_name).read_bytes(),
        )

    def test_push_is_idempotent_content_addressed(self) -> None:
        self.fx.manager.create(label="offsite")
        store = RecordingStore()
        first = self.fx.manager.push_to_s3(store)
        second = self.fx.manager.push_to_s3(store)
        self.assertTrue(first["pushed"] and second["pushed"])
        # Same content hashes both times: no duplicate bytes stored.
        self.assertEqual(len(store.files), len(first["objects"]))
        self.assertEqual(
            [o["sha256"] for o in first["objects"]],
            [o["sha256"] for o in second["objects"]],
        )

    def test_push_without_sidecar_ships_db_only(self) -> None:
        plain = BackupManager(
            self.fx.db, self.fx.manager.directory, keep=3, include_blobs=False
        )
        info = plain.create(label="noblobs")
        self.assertEqual(info.blob_name, "")
        store = RecordingStore()
        result = plain.push_to_s3(store)
        self.assertTrue(result["pushed"])
        self.assertEqual(len(result["objects"]), 1)

    def test_push_no_backup_reports_cleanly(self) -> None:
        store = RecordingStore()
        result = self.fx.manager.push_to_s3(store)
        self.assertEqual(result, {"pushed": False, "reason": "no backup to push"})
        self.assertEqual(store.put_calls, 0)

    def test_push_fails_fast_on_hash_mismatch(self) -> None:
        self.fx.manager.create(label="offsite")
        with self.assertRaises(StorageError) as ctx:
            self.fx.manager.push_to_s3(CorruptStore())
        self.assertIn("offsite upload corrupted", str(ctx.exception))

    def test_push_end_to_end_over_fake_s3(self) -> None:
        """Backup bytes travel the real signed S3BlobStore HTTP path."""
        info = self.fx.manager.create(label="offsite")
        server = FakeS3Server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            config = S3Config(
                endpoint=f"http://127.0.0.1:{port}", bucket=BUCKET,
                access_key=TEST_KEY, secret_key=TEST_SECRET,
                region=TEST_REGION, timeout=10.0,
            )
            store = S3BlobStore(config)
            result = self.fx.manager.push_to_s3(store)
            self.assertTrue(result["pushed"])
            for obj in result["objects"]:
                local = (
                    Path(info.path)
                    if obj["file"] == info.name
                    else self.fx.manager.directory / obj["file"]
                )
                self.assertEqual(store.get_bytes(obj["sha256"]), local.read_bytes())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_scheduler_uses_offsite_store(self) -> None:
        store = RecordingStore()
        manager = BackupManager(
            self.fx.db, self.fx.manager.directory, keep=3, offsite_store=store
        )
        stop = threading.Event()
        worker = threading.Thread(
            target=manager.run_scheduler, args=(3600,), kwargs={"should_stop": stop.is_set}
        )
        worker.start()
        stop.wait(1.0)  # let one scheduler iteration run
        stop.set()
        worker.join(timeout=10)
        self.assertGreater(store.put_calls, 0, "scheduler never shipped the backup")
        self.assertEqual(len(manager.list()), 1)


class TestOpenBlobStoreAuto(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = {k: os.environ[k] for k in _S3_ENV_KEYS if k in os.environ}
        for key in _S3_ENV_KEYS:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in _S3_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.update(self.saved)
        self.tmp.cleanup()

    def test_auto_falls_back_to_local_without_env(self) -> None:
        db = Database(":memory:")
        db.migrate()
        try:
            store = open_blob_store(
                "auto", db=db, root=Path(self.tmp.name) / "blobs"
            )
            self.assertIsInstance(store, BlobStore)
            info = store.put_bytes(b"auto-local")
            self.assertEqual(store.get_bytes(info.sha256), b"auto-local")
        finally:
            db.close()

    def test_auto_selects_s3_when_env_configured(self) -> None:
        os.environ.update(
            {
                "S3_ENDPOINT": "https://minio.example.com:9000",
                "S3_BUCKET": "blobs",
                "S3_ACCESS_KEY": "ak",
                "S3_SECRET_KEY": "sk",
            }
        )
        store = open_blob_store("auto", db=object(), root="/tmp/whatever")
        self.assertIsInstance(store, S3BlobStore)
        self.assertEqual(store.config.bucket, "blobs")

    def test_auto_local_still_needs_db_and_root(self) -> None:
        with self.assertRaises(ConfigError):
            open_blob_store("auto")


if __name__ == "__main__":
    unittest.main()
