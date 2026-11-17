"""AV archive-member intake scan: members scanned at intake, not 24h later.

Failing-first: on the pristine base ``_scan_intake`` scans only the
landed container — ``_unpack_archive`` extracts members without ever
scanning them, so an EICAR file inside a deflated zip lands clean and
only the daily sweep would catch it, up to 24 hours later. Every test
here fails there (``report.archive`` carries no ``"av"`` key, or the
download stays ``ok`` with the infected member on disk).

The build scans every extracted member right after the unpack, with
the same verdicts as the container scan: clean flows on, unknown
fails open with the reason on ``report.archive["av"]``, infected
quarantines that member and blocks the download the way an infected
container does today.

One test-shape fact the build depends on: the builtin engine matches
the EICAR string over raw bytes, so a zip member stored uncompressed
trips the *container* scan first — only deflated members reach the
member scan. The tests write deflated zips except where the container
scan itself is the point.
"""

from __future__ import annotations

import io
import os
import tarfile
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


def _tar_bytes(members: list[tuple[str, bytes]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


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

    def tearDown(self):
        os.environ.pop("NM_AV", None)
        os.environ.pop("NM_AV_QUARANTINE", None)
        from nomorals.core.antivirus import detect_antivirus

        detect_antivirus(refresh=True)
        self._env.stop()


class _SucceedCase(_AvEnvCase):
    """Drive the real ``_succeed`` funnel with the bookkeeping rungs
    stubbed, the way the AV wiring suite does."""

    def _downloader(self, home: Path):
        from nomorals.media.downloader import MediaDownloader

        dl = MediaDownloader(_ctx(home), direct_retries=1,
                             max_proxies=1)
        self._calls: list[str] = []
        calls = self._calls
        dl._record_download_outcome = lambda *a, **k: calls.append(
            "outcome")
        dl._learn_from_success = lambda *a, **k: calls.append("learn")
        dl._intake_exploration = lambda *a, **k: calls.append(
            "explore")
        dl._register_download_artifact = lambda *a, **k: calls.append(
            "artifact")
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


class MemberScanTests(_SucceedCase):
    def test_clean_zip_members_pass_intake(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = self._archive(home, "pack.zip", _zip_bytes([
                ("hello.txt", b"hello, world\n"),
                ("notes/readme.txt", b"readme\n")]))
            out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            self.assertEqual(out.av["verdict"], "clean")
            self.assertEqual(out.archive["extracted"], 2)
            self.assertEqual(out.archive["av"]["verdict"], "clean")
            self.assertEqual(out.archive["av"]["engine"], "builtin")
            dest = Path(out.archive["dest"])
            self.assertTrue((dest / "hello.txt").exists())
            self.assertTrue((dest / "notes" / "readme.txt").exists())
            self.assertIn("member scan: clean", out.summary())

    def test_clean_tar_members_pass_intake(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = self._archive(home, "pack.tar", _tar_bytes([
                ("a.txt", b"a\n"), ("b.txt", b"b\n")]))
            out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            self.assertEqual(out.archive["extracted"], 2)
            self.assertEqual(out.archive["av"]["verdict"], "clean")

    def test_eicar_member_blocked_and_quarantined(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = self._archive(home, "pack.zip", _zip_bytes([
                ("hello.txt", b"hello, world\n"),
                ("evil.com", eicar_bytes())]))
            out = dl._succeed(self._report(target), 0.0)
            # the container itself scanned clean (deflated bytes) —
            # the member scan is what catches it
            self.assertEqual(out.av["verdict"], "clean")
            self.assertFalse(out.ok)
            self.assertEqual(out.failure.value, "site_error")
            self.assertIn("EICAR-Test-File", out.likely_cause)
            self.assertIn("evil.com", out.likely_cause)
            self.assertIn("quarantined", out.likely_cause)
            self.assertEqual(out.path, "")
            self.assertEqual(out.archive["av"]["verdict"], "infected")
            dest = Path(out.archive["dest"])
            self.assertFalse((dest / "evil.com").exists())
            # the clean sibling is untouched
            self.assertTrue((dest / "hello.txt").exists())
            # the member was moved to the quarantine dir
            member_av = [r for r in out.archive["av"]["results"]
                         if r.get("verdict") == "infected"]
            self.assertEqual(len(member_av), 1)
            qdir = (home / "quarantine").resolve()
            self.assertTrue(member_av[0]["quarantined_to"].startswith(
                str(qdir)))
            self.assertTrue(
                Path(member_av[0]["quarantined_to"]).exists())
            # the stage log names the block
            self.assertTrue(any(
                s.stage == "antivirus" and not s.ok
                for s in out.stages))
            self.assertIn("EICAR-Test-File", out.summary())

    def test_nested_zip_member_scanned(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            inner = _zip_bytes([("evil.com", eicar_bytes())])
            target = self._archive(home, "outer.zip", _zip_bytes([
                ("inner.zip", inner)]))
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertIn("EICAR-Test-File", out.likely_cause)
            self.assertEqual(out.archive["av"]["verdict"], "infected")
            # the nested member was extracted AND quarantined
            dest = Path(out.archive["dest"])
            self.assertFalse(
                (dest / "inner.zip.extracted" / "evil.com").exists())
            member_av = [r for r in out.archive["av"]["results"]
                         if r.get("verdict") == "infected"]
            self.assertEqual(len(member_av), 1)
            self.assertIn("evil.com", member_av[0]["path"])

    def test_unknown_member_verdict_fails_open(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = self._archive(home, "pack.zip", _zip_bytes([
                ("hello.txt", b"hello, world\n")]))
            with mock.patch.dict(os.environ, {"NM_AV": "none"}):
                from nomorals.core.antivirus import detect_antivirus

                detect_antivirus(refresh=True)
                out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            self.assertEqual(out.path, str(target))
            self.assertEqual(out.archive["av"]["verdict"], "unknown")
            self.assertIn("no antivirus engine",
                          out.archive["av"]["detail"])
            dest = Path(out.archive["dest"])
            self.assertTrue((dest / "hello.txt").exists())
            self.assertIn("member scan: unknown", out.summary())

    def test_member_scan_failure_fails_open(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = self._archive(home, "pack.zip", _zip_bytes([
                ("hello.txt", b"hello, world\n")]))
            with mock.patch(
                    "nomorals.core.antivirus.scan_path",
                    side_effect=RuntimeError("boom")):
                out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            self.assertEqual(out.archive["av"]["verdict"], "unknown")
            self.assertIn("member scan failed",
                          out.archive["av"]["detail"])
            self.assertIn("boom", out.archive["av"]["detail"])

    def test_quarantine_failure_still_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            # the quarantine dir is a plain file: the move must fail
            (home / "quarantine").write_text("not a dir",
                                             encoding="utf-8")
            dl = self._downloader(home)
            target = self._archive(home, "pack.zip", _zip_bytes([
                ("evil.com", eicar_bytes())]))
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertIn("EICAR-Test-File", out.likely_cause)
            self.assertIn("quarantine failed", out.likely_cause)
            # the file stays where it was, but the report no longer
            # points at it — nothing hands it to the owner
            dest = Path(out.archive["dest"])
            self.assertTrue((dest / "evil.com").exists())
            self.assertEqual(out.path, "")

    def test_container_scan_unchanged_stored_eicar(self):
        """A member stored uncompressed trips the container scan
        first — blocked before the unpack, archive never set."""
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = self._archive(home, "pack.zip", _zip_bytes([
                ("evil.com", eicar_bytes())],
                compress_type=zipfile.ZIP_STORED))
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertEqual(out.av["verdict"], "infected")
            self.assertIsNone(out.archive)

    def test_non_archive_never_scans_members(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = home / "workspace" / "hello.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("hello, world\n", encoding="utf-8")
            out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            self.assertIsNone(out.archive)


class ScanHelperTests(_AvEnvCase):
    def test_missing_dest_is_unknown_not_a_crash(self):
        from nomorals.media.downloader import (
            DownloadReport,
            _scan_extracted_members,
        )

        report = DownloadReport(
            url="https://example.invalid/pack.zip", ok=True,
            path="/definitely/not/here.zip", extractor="test")
        result: dict = {"dest": "/definitely/not/here.extracted",
                        "extracted": 0, "blocked": False}
        _scan_extracted_members(report, result, _ctx(Path(
            tempfile.mkdtemp())))
        self.assertTrue(report.ok)
        self.assertEqual(result["av"]["verdict"], "unknown")
        self.assertIn("no extracted directory",
                      result["av"]["detail"])
