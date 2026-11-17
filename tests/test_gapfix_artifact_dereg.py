"""Artifact deregister for member-blocked downloads.

Failing-first: the AV archive-member scan (420984f8) blocks a download
whose archive holds an infected member — but the artifact row the
download registered in ``_succeed`` (BEFORE the unpack) survives the
block, so the store keeps citing a container the owner never receives.
Every behavior test here fails on the pristine base: the row is still
there after the block, and ``report.artifact_id`` does not exist.

The build withdraws exactly the row THIS download minted
(``report.artifact_id``, set by ``_register_download_artifact`` when
it mints a row) in the member-block path of
``_scan_extracted_members``, best-effort, never changing the block
outcome. ``drop_blob=True``: every ``put_file`` bumps the blob
refcount by one, so the release drops exactly this row's reference
and bytes shared with another artifact survive.

One test-shape fact the build depends on (same as the member-scan
suite): the builtin engine matches the EICAR string over raw bytes,
so a zip member stored uncompressed trips the *container* scan first
— only deflated members reach the member scan.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# The EICAR standard test string, assembled from parts so this file
# never contains the contiguous signature.
_EICAR_A = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_B = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def eicar_bytes() -> bytes:
    return (_EICAR_A + _EICAR_B).encode("ascii")


def _ctx(home: Path):
    """Fake Devon context: settings resolve home + quarantine + media."""
    home = Path(home)
    settings = SimpleNamespace(
        home=str(home), home_path=home,
        workspace_dir=home / "workspace")

    def _resolve(relative: str) -> Path:
        p = Path(os.path.expanduser(relative))
        return p if p.is_absolute() else (home / p).resolve()

    settings.resolve = _resolve
    return SimpleNamespace(db=None, settings=settings)


def _zip_bytes(members: list[tuple[str, bytes]], *,
               compress_type: int = zipfile.ZIP_DEFLATED) -> bytes:
    """A zip holding *members* — deflated by default, so the container
    scan sees compressed bytes, never the raw member content."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compress_type) as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def _store(tmp_path):
    from nomorals.storage.artifacts import ArtifactStore
    from nomorals.storage.blob import BlobStore
    from nomorals.storage.db import Database

    db = Database(":memory:")
    db.migrate()
    store = ArtifactStore(db, BlobStore(db, tmp_path / "blobs"))
    return store, db


def _download_rows(store):
    return [a for a in store.search(creator="media_downloader",
                                    limit=100)
            if (a.metadata or {}).get("origin") == "download"]


class _AvEnvCase(unittest.TestCase):
    """Base: the process-wide antivirus cache is restored after each
    test so a faked engine never leaks into a later test file."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("NM_AV", None)
        os.environ.pop("NM_AV_QUARANTINE", None)
        from nomorals.core.antivirus import detect_antivirus

        detect_antivirus(refresh=True)
        self._tmpdirs: list[str] = []

    def tearDown(self):
        os.environ.pop("NM_AV", None)
        os.environ.pop("NM_AV_QUARANTINE", None)
        from nomorals.core.antivirus import detect_antivirus

        detect_antivirus(refresh=True)
        self._env.stop()
        for td in self._tmpdirs:
            shutil.rmtree(td, ignore_errors=True)

    def _mkdtemp(self) -> Path:
        td = tempfile.mkdtemp()
        self._tmpdirs.append(td)
        return Path(td)


class _SucceedCase(_AvEnvCase):
    """Drive the real ``_succeed`` funnel with a REAL artifact store
    and the bookkeeping rungs stubbed, the way the AV wiring suite
    does — but registration itself is real, because that is what
    this lane withdraws."""

    def _downloader(self, home: Path, store):
        from nomorals.media.downloader import MediaDownloader

        dl = MediaDownloader(_ctx(home), direct_retries=1,
                             max_proxies=1, artifact_store=store)
        self._calls: list[str] = []
        calls = self._calls
        dl._record_download_outcome = lambda *a, **k: calls.append(
            "outcome")
        dl._learn_from_success = lambda *a, **k: calls.append("learn")
        dl._intake_exploration = lambda *a, **k: calls.append(
            "explore")
        return dl

    def _report(self, path: Path):
        from nomorals.media.downloader import DownloadReport

        return DownloadReport(url="https://example.invalid/pack.zip",
                              ok=True, path=str(path),
                              size_bytes=path.stat().st_size,
                              extractor="test")

    def _archive(self, home: Path, name: str, data: bytes) -> Path:
        """Land an archive inside the workspace sandbox, where the
        Archivist is allowed to extract it."""
        ws = home / "workspace"
        ws.mkdir(parents=True, exist_ok=True)
        target = ws / name
        target.write_bytes(data)
        return target


class DeregisterTests(_SucceedCase):
    def test_member_blocked_download_leaves_no_artifact_row(self):
        home = self._mkdtemp()
        store, db = _store(self._mkdtemp())
        try:
            data = _zip_bytes([("hello.txt", b"hello, world\n"),
                               ("evil.com", eicar_bytes())])
            dl = self._downloader(home, store)
            target = self._archive(home, "pack.zip", data)
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertEqual(out.archive["av"]["verdict"], "infected")
            # the row this download minted is withdrawn
            self.assertEqual(_download_rows(store), [])
            self.assertEqual(out.artifact_id, "")
            # drop_blob=True purged the container bytes: nothing else
            # referenced them, so the blob is gone too
            digest = hashlib.sha256(data).hexdigest()
            self.assertIsNone(store.blobs.info(digest))
        finally:
            db.close()

    def test_clean_download_keeps_artifact_row(self):
        home = self._mkdtemp()
        store, db = _store(self._mkdtemp())
        try:
            data = _zip_bytes([("hello.txt", b"hello, world\n")])
            dl = self._downloader(home, store)
            target = self._archive(home, "pack.zip", data)
            out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            rows = _download_rows(store)
            self.assertEqual(len(rows), 1)
            self.assertEqual(out.artifact_id, rows[0].id)
            self.assertEqual(store.read(rows[0].id), data)
        finally:
            db.close()

    def test_container_blocked_download_registers_nothing(self):
        """A member stored uncompressed trips the container scan
        first — blocked before the unpack, so registration never
        runs. Pinned: today's behavior, unchanged by this lane."""
        home = self._mkdtemp()
        store, db = _store(self._mkdtemp())
        try:
            data = _zip_bytes([("evil.com", eicar_bytes())],
                              compress_type=zipfile.ZIP_STORED)
            dl = self._downloader(home, store)
            target = self._archive(home, "pack.zip", data)
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertEqual(out.av["verdict"], "infected")
            self.assertIsNone(out.archive)
            self.assertEqual(_download_rows(store), [])
            self.assertEqual(out.artifact_id, "")
        finally:
            db.close()

    def test_dedup_preexisting_row_survives_member_block(self):
        """Identical bytes were already registered as a download
        artifact by an earlier download: registration dedups (first
        registration wins, no row minted, no id stashed) and the
        member block withdraws nothing — the pre-existing row is not
        this download's record."""
        home = self._mkdtemp()
        store, db = _store(self._mkdtemp())
        try:
            data = _zip_bytes([("hello.txt", b"hello, world\n"),
                               ("evil.com", eicar_bytes())])
            seed = home / "seed.zip"
            seed.write_bytes(data)
            old = store.put_file(
                seed, type="file", creator="media_downloader",
                metadata={"origin": "download", "filename": "pack.zip"})
            dl = self._downloader(home, store)
            target = self._archive(home, "pack.zip", data)
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertEqual(out.archive["av"]["verdict"], "infected")
            rows = _download_rows(store)
            self.assertEqual([r.id for r in rows], [old.id])
            self.assertEqual(store.read(old.id), data)
        finally:
            db.close()

    def test_shared_blob_survives_deregister(self):
        """Another artifact cites the same bytes (chat-inbound
        origin): the member block withdraws the download row, and
        drop_blob=True releases only that row's reference — the
        shared bytes still read back through the other row."""
        home = self._mkdtemp()
        store, db = _store(self._mkdtemp())
        try:
            data = _zip_bytes([("hello.txt", b"hello, world\n"),
                               ("evil.com", eicar_bytes())])
            seed = home / "seed.zip"
            seed.write_bytes(data)
            chat = store.put_file(
                seed, type="file", creator="chat-inbound",
                metadata={"origin": "chat", "filename": "pack.zip"})
            dl = self._downloader(home, store)
            target = self._archive(home, "pack.zip", data)
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertEqual(_download_rows(store), [])
            self.assertEqual(out.artifact_id, "")
            self.assertIsNotNone(store.get(chat.id))
            self.assertEqual(store.read(chat.id), data)
            digest = hashlib.sha256(data).hexdigest()
            self.assertIsNotNone(store.blobs.info(digest))
        finally:
            db.close()

    def test_deregister_failure_never_breaks_block(self):
        home = self._mkdtemp()
        store, db = _store(self._mkdtemp())
        try:
            data = _zip_bytes([("hello.txt", b"hello, world\n"),
                               ("evil.com", eicar_bytes())])
            dl = self._downloader(home, store)
            target = self._archive(home, "pack.zip", data)
            with mock.patch.object(
                    store, "delete",
                    side_effect=RuntimeError("store down")):
                out = dl._succeed(self._report(target), 0.0)
            # the block outcome is exactly what it was without the
            # deregister attempt
            self.assertFalse(out.ok)
            self.assertEqual(out.archive["av"]["verdict"], "infected")
            self.assertIn("EICAR-Test-File", out.likely_cause)
            self.assertIn("evil.com", out.likely_cause)
            self.assertEqual(out.path, "")
            # the row survives (the delete failed) — but the
            # download is still blocked, never delivered
            self.assertEqual(len(_download_rows(store)), 1)
        finally:
            db.close()

    def test_no_store_member_block_still_blocks(self):
        """No artifact store at all: the member block works exactly
        as before — the deregister is a no-op, never a crash."""
        home = self._mkdtemp()
        from nomorals.media.downloader import MediaDownloader

        dl = MediaDownloader(_ctx(home), direct_retries=1,
                             max_proxies=1)
        dl._record_download_outcome = lambda *a, **k: None
        dl._learn_from_success = lambda *a, **k: None
        dl._intake_exploration = lambda *a, **k: None
        target = self._archive(home, "pack.zip", _zip_bytes([
            ("evil.com", eicar_bytes())]))
        out = dl._succeed(self._report(target), 0.0)
        self.assertFalse(out.ok)
        self.assertEqual(out.archive["av"]["verdict"], "infected")
        self.assertIn("EICAR-Test-File", out.likely_cause)
        self.assertEqual(out.path, "")


class CallbackSeamTests(_AvEnvCase):
    def test_member_block_invokes_callback_with_report(self):
        """The hook point: ``_scan_extracted_members`` calls the
        callback in its member-block path, with the blocked report."""
        from nomorals.media.downloader import (
            DownloadReport,
            _scan_extracted_members,
        )

        home = self._mkdtemp()
        dest = home / "pack.extracted"
        (dest / "sub").mkdir(parents=True)
        (dest / "evil.com").write_bytes(eicar_bytes())
        (dest / "sub" / "hello.txt").write_bytes(b"hello\n")
        report = DownloadReport(
            url="https://example.invalid/pack.zip", ok=True,
            path=str(home / "pack.zip"), extractor="test")
        result: dict = {"dest": str(dest), "extracted": 2,
                        "blocked": False}
        seen: list = []
        _scan_extracted_members(report, result, _ctx(home),
                                on_member_blocked=seen.append)
        self.assertFalse(report.ok)
        self.assertEqual(result["av"]["verdict"], "infected")
        self.assertEqual(seen, [report])

    def test_no_callback_clean_scan_unchanged(self):
        """Without a callback the member scan behaves exactly as
        before — the hook is optional, never load-bearing."""
        from nomorals.media.downloader import (
            DownloadReport,
            _scan_extracted_members,
        )

        home = self._mkdtemp()
        dest = home / "pack.extracted"
        dest.mkdir(parents=True)
        (dest / "hello.txt").write_bytes(b"hello\n")
        report = DownloadReport(
            url="https://example.invalid/pack.zip", ok=True,
            path=str(home / "pack.zip"), extractor="test")
        result: dict = {"dest": str(dest), "extracted": 1,
                        "blocked": False}
        _scan_extracted_members(report, result, _ctx(home))
        self.assertTrue(report.ok)
        self.assertEqual(result["av"]["verdict"], "clean")


if __name__ == "__main__":
    unittest.main()
