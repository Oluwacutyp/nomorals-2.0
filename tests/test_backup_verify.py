"""Backup rotation + verification: production-grade restore checks.

Tier: unit (fully offline — tmp dirs, a real local SQLite file, no network,
no subprocesses). These tests prove ``BackupManager.verify()`` actually
*restores* the latest backup into a temp dir and checks integrity — checksums,
SQLite openability + ``integrity_check``, and per-blob presence — so it
catches corruption instead of just confirming a file exists.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.cmdline.commands.backup import _cmd_backup
from nomorals.storage.backup import BackupManager
from nomorals.storage.db import Database


def _fake_stamps():
    """Distinct backup stamps so same-second creates don't collide."""
    counter = itertools.count(1)
    real_strftime = time.strftime

    def fake(fmt, t=None):
        if fmt == "%Y%m%d-%H%M%S":
            return f"20260102-0000{next(counter):02d}"
        return real_strftime(fmt, t)

    return mock.patch("time.strftime", fake)


class _Fixture:
    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = Database(root / "live.db")
        self.db.migrate()
        self.db.insert(
            "conversations",
            {"id": "c1", "title": "t", "created_at": 1.0, "updated_at": 1.0},
        )
        self.blobs = root / "blobs"
        self.blobs.mkdir()
        (self.blobs / "a.bin").write_bytes(b"hello-blob-a" * 100)
        (self.blobs / "sub").mkdir()
        (self.blobs / "sub" / "b.bin").write_bytes(b"hello-blob-b" * 200)
        self.manager = BackupManager(
            self.db,
            root / "backups",
            keep=3,
            include_blobs=True,
            blob_dir=self.blobs,
        )

    def cleanup(self) -> None:
        self.db.close() if hasattr(self.db, "close") else None
        self.tmp.cleanup()


class BackupVerifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = _Fixture()

    def tearDown(self) -> None:
        self.fx.cleanup()

    def test_verify_clean_backup(self) -> None:
        info = self.fx.manager.create(label="good")
        self.assertEqual(info.blob_count, 2)
        self.assertEqual(self.fx.manager.verify(), [])

    def test_verify_no_backups(self) -> None:
        self.assertEqual(self.fx.manager.verify(), ["no backups present"])

    def test_rotation_keeps_n_newest(self) -> None:
        with _fake_stamps():
            infos = [self.fx.manager.create() for _ in range(5)]
        removed = self.fx.manager.rotate()
        remaining = [b.name for b in self.fx.manager.list()]
        self.assertEqual(len(remaining), 3)
        self.assertEqual(remaining, [i.name for i in infos[-3:]])
        self.assertEqual(sorted(removed), sorted(i.name for i in infos[:2]))
        # Blob sidecars of pruned snapshots must go with them.
        leftover = list(self.fx.manager.directory.glob("*-blobs.tar.gz"))
        self.assertEqual(len(leftover), 3)
        self.assertEqual(
            sorted(p.name for p in leftover),
            sorted(i.blob_name for i in infos[-3:]),
        )

    def test_verify_fails_on_bitflipped_blob_archive(self) -> None:
        info = self.fx.manager.create(label="bitflip")
        archive = self.fx.manager.directory / info.blob_name
        data = bytearray(archive.read_bytes())
        data[len(data) // 2] ^= 0xFF
        archive.write_bytes(bytes(data))
        problems = self.fx.manager.verify()
        self.assertTrue(
            any("blob" in p and "checksum" in p for p in problems),
            f"expected a blob checksum problem, got: {problems}",
        )

    def test_verify_fails_on_truncated_database(self) -> None:
        plain = BackupManager(
            self.fx.db,
            self.fx.manager.directory,
            keep=3,
            compress=False,
            include_blobs=False,
        )
        info = plain.create(label="trunc")
        path = Path(info.path)
        # Truncate the DB, then repair the manifest checksum so the corruption
        # has to be caught by the restore + integrity_check, not the checksum.
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 3])
        manifest = json.loads(plain.manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            if entry["name"] == info.name:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        plain.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        problems = plain.verify()
        self.assertTrue(
            any("integrity_check" in p or "cannot open" in p for p in problems),
            f"expected an integrity problem, got: {problems}",
        )

    def test_verify_fails_on_missing_blob_file(self) -> None:
        info = self.fx.manager.create(label="missing-blob")
        archive = self.fx.manager.directory / info.blob_name
        # Rebuild the archive without one blob and repair the archive checksum,
        # so the per-file presence check is what catches it.
        tmpdir = Path(tempfile.mkdtemp())
        try:
            self.fx.manager._extract_blob_archive(archive, tmpdir)  # noqa: SLF001
            gone = tmpdir / "sub" / "b.bin"
            gone.unlink()
            with tarfile.open(archive, "w:gz") as tar:
                for p in sorted(tmpdir.rglob("*")):
                    if p.is_file():
                        tar.add(str(p), arcname=p.relative_to(tmpdir).as_posix())
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)
        manifest = json.loads(self.fx.manager.manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            if entry["name"] == info.name:
                entry["blob_sha256"] = hashlib.sha256(
                    archive.read_bytes()
                ).hexdigest()
        self.fx.manager.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        problems = self.fx.manager.verify()
        self.assertTrue(
            any("blob missing from archive" in p and "sub/b.bin" in p for p in problems),
            f"expected a missing-blob problem, got: {problems}",
        )

    def test_verify_fails_on_corrupt_gzip(self) -> None:
        info = self.fx.manager.create(label="badgzip")
        path = Path(info.path)
        # Corrupt the gzip header; repair the manifest checksum so the failure
        # surfaces at extraction, proving verify() reports rather than raises.
        data = bytearray(path.read_bytes())
        data[0] ^= 0xFF
        path.write_bytes(bytes(data))
        manifest = json.loads(self.fx.manager.manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            if entry["name"] == info.name:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fx.manager.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        problems = self.fx.manager.verify()
        self.assertTrue(
            any("cannot extract" in p for p in problems),
            f"expected an extraction problem, got: {problems}",
        )

    def test_verify_never_raises_on_garbage(self) -> None:
        with _fake_stamps():
            info = self.fx.manager.create(label="garbage")
        path = Path(info.path)
        path.write_bytes(b"this is not a database at all" * 40)
        manifest = json.loads(self.fx.manager.manifest_path.read_text(encoding="utf-8"))
        for entry in manifest:
            if entry["name"] == info.name:
                entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fx.manager.manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        problems = self.fx.manager.verify()  # must return problems, not raise
        self.assertTrue(problems)


class BackupCliVerbTests(unittest.TestCase):
    """``nm backup list|verify|prune`` verbs coexist with the legacy flags."""

    def setUp(self) -> None:
        self.fx = _Fixture()

    def tearDown(self) -> None:
        self.fx.cleanup()

    def _context(self) -> SimpleNamespace:
        backup_settings = SimpleNamespace(
            keep=3, compress=True, git_repo="", include_blobs=True
        )
        settings = SimpleNamespace(
            backup_dir=self.fx.manager.directory,
            backup=backup_settings,
            blob_dir=self.fx.blobs,
        )
        return SimpleNamespace(db=self.fx.db, settings=settings)

    def _args(self, **kwargs) -> argparse.Namespace:
        base = dict(
            verb=None, create=False, verify=False, list=False,
            restore="", push=False, json=False,
        )
        base.update(kwargs)
        return argparse.Namespace(**base)

    def test_parser_accepts_verbs(self) -> None:
        from nomorals.cmdline.parser import _parser  # noqa: SLF001 - same package

        parser = _parser()
        for verb in ("list", "verify", "prune"):
            self.assertEqual(parser.parse_args(["backup", verb]).verb, verb)
        # Legacy flag form still parses.
        args = parser.parse_args(["backup", "--verify"])
        self.assertIsNone(args.verb)
        self.assertTrue(args.verify)

    def test_verb_verify_clean(self) -> None:
        self.fx.manager.create(label="cli")
        rc = _cmd_backup(self._args(verb="verify"), self._context())
        self.assertEqual(rc, 0)

    def test_verb_verify_dirty_returns_1(self) -> None:
        info = self.fx.manager.create(label="cli-dirty")
        Path(info.path).write_bytes(b"corrupted")
        rc = _cmd_backup(self._args(verb="verify"), self._context())
        self.assertEqual(rc, 1)

    def test_verb_prune(self) -> None:
        with _fake_stamps():
            for _ in range(4):
                self.fx.manager.create()
        rc = _cmd_backup(self._args(verb="prune"), self._context())
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.fx.manager.list()), 3)

    def test_verb_list(self) -> None:
        self.fx.manager.create(label="cli-list")
        rc = _cmd_backup(self._args(verb="list"), self._context())
        self.assertEqual(rc, 0)

    def test_legacy_flags_still_work(self) -> None:
        self.fx.manager.create(label="legacy")
        self.assertEqual(_cmd_backup(self._args(verify=True), self._context()), 0)
        self.assertEqual(_cmd_backup(self._args(), self._context()), 0)  # default list


if __name__ == "__main__":
    unittest.main()
