"""RE Pass 2 — the self-provisioning machinery.

The outdo over REA is mechanical, so it is tested mechanically:
pinned descriptors with publisher checksums, a download whose
sha256 does not match the pin installs NOTHING (and the attempt
is recorded by name), an install that lands is probe-executed
before it is called ready, zip installs keep their launch
scripts' exec bits, and every state the doctor matrix reports
(ready / provisionable / unavailable-with-reason) is reachable
and honest.

No test downloads a real engine: downloaders are injected and
the artifacts are tiny fakes shaped like the real ones (a JDK
tarball whose bin/java answers -version, a Ghidra zip whose
analyzeHeadless answers with its usage). The REAL engines were
provisioned and run for real in the build's verify area; these
tests pin the machinery around them.
"""

from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from nomorals.agents.reverse import provision

JAVA_SCRIPT = (
    "#!/bin/sh\n"
    "echo 'openjdk version \"25.0.5\" 2026-10-20 LTS' 1>&2\n"
    "echo 'OpenJDK Runtime Environment Temurin-25.0.5+8' 1>&2\n"
    "exit 0\n")

GHIDRA_SCRIPT = (
    "#!/bin/sh\n"
    "echo 'USAGE: analyzeHeadless <project_location> <project_name>'\n"
    "exit 1\n")


def _jdk_tarball() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = JAVA_SCRIPT.encode()
        info = tarfile.TarInfo("jdk-25.0.5+8/bin/java")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _ghidra_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo(
            "ghidra_12.2_PUBLIC/support/analyzeHeadless")
        info.create_system = 3
        info.external_attr = 0o100755 << 16
        zf.writestr(info, GHIDRA_SCRIPT)
        props = zipfile.ZipInfo(
            "ghidra_12.2_PUBLIC/Ghidra/application.properties")
        props.create_system = 3
        props.external_attr = 0o100644 << 16
        zf.writestr(props, "application.version=12.2\n")
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _Root:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-provision-")
        return Path(self.dir)

    def __exit__(self, *exc):
        return False


class DescriptorTest(unittest.TestCase):
    def test_pins_are_publisher_shaped(self):
        jdk = provision.DESCRIPTORS["jdk"]
        self.assertEqual(jdk["version"], "25.0.5+8")
        self.assertEqual(jdk["required_major"], 25)
        for key in ("linux-x64", "linux-aarch64", "windows-x64"):
            art = jdk["artifacts"][key]
            self.assertTrue(art["url"].startswith(
                "https://github.com/adoptium/"), key)
            self.assertEqual(len(art["sha256"]), 64, key)
        gh = provision.DESCRIPTORS["ghidra"]
        self.assertEqual(gh["version"], "12.2")
        art = gh["artifacts"]["any"]
        # the digest published in the official Ghidra 12.2
        # release notes
        self.assertTrue(art["sha256"].startswith("2c2cd776db3c1f6a"))
        self.assertIn("NationalSecurityAgency/ghidra", art["url"])


class JdkProvisionTest(unittest.TestCase):
    def setUp(self):
        self.key = provision.platform_key()
        if self.key not in provision.DESCRIPTORS["jdk"]["artifacts"]:
            self.skipTest(f"no pinned JDK artifact for {self.key}")
        self.tarball = _jdk_tarball()
        self._saved = dict(
            provision.DESCRIPTORS["jdk"]["artifacts"][self.key])
        provision.DESCRIPTORS["jdk"]["artifacts"][self.key] = {
            "url": "https://example.invalid/fake-jdk.tar.gz",
            "sha256": _sha(self.tarball), "kind": "tar.gz"}

    def tearDown(self):
        provision.DESCRIPTORS["jdk"]["artifacts"][self.key] = \
            self._saved

    def _downloader(self, payload: bytes, calls: list):
        def _dl(url: str, dest: Path) -> None:
            calls.append(url)
            dest.write_bytes(payload)
        return _dl

    def test_provision_lands_probes_and_records(self):
        with _Root() as root:
            calls: list = []
            row = provision.provision_engine(
                "jdk", root=root,
                downloader=self._downloader(self.tarball, calls))
            self.assertTrue(row["available"], row)
            self.assertEqual(row["source"], "provisioned")
            self.assertIn("25", row["version"])
            state = provision.jdk_state(root)
            self.assertEqual(state["state"], "ready")
            # second drive: already ready — no second download
            row2 = provision.provision_engine(
                "jdk", root=root,
                downloader=self._downloader(self.tarball, calls))
            self.assertTrue(row2["available"])
            self.assertEqual(len(calls), 1)
            self.assertTrue(any("already provisioned" in a
                                for a in row2["attempts"]))

    def test_tampered_archive_installs_nothing(self):
        with _Root() as root:
            bad = self.tarball + b"tampered"
            with self.assertRaises(provision.ProvisionError) as ctx:
                provision.provision_engine(
                    "jdk", root=root,
                    downloader=self._downloader(bad, []))
            self.assertTrue(any("sha256 mismatch" in a
                                for a in ctx.exception.attempts))
            dest = root / "re" / "jdk"
            self.assertFalse((dest / ".pinned-sha256").exists())
            recorded = provision.recorded_attempts(root, "jdk")
            self.assertTrue(any("sha256 mismatch" in a
                                for a in recorded))
            self.assertEqual(provision.jdk_state(root)["state"],
                             "provisionable")

    def test_old_system_java_is_present_but_unusable(self):
        def runner(argv, **kw):
            return 0, 'openjdk version "17.0.9" 2023-10-17\n'

        with _Root() as root, mock.patch.object(provision.shutil, "which",
                               return_value="/usr/lib/jvm17/java"):
            state = provision.jdk_state(root, runner=runner)
        self.assertFalse(state["available"])
        self.assertEqual(state["state"], "provisionable")
        self.assertTrue(any("major 17" in a
                            for a in state["attempts"]))


class GhidraProvisionTest(unittest.TestCase):
    def setUp(self):
        self.key = provision.platform_key()
        if self.key not in provision.DESCRIPTORS["jdk"]["artifacts"]:
            self.skipTest(f"no pinned JDK artifact for {self.key}")
        self._saved_gh = dict(
            provision.DESCRIPTORS["ghidra"]["artifacts"]["any"])
        self._saved_jdk = dict(
            provision.DESCRIPTORS["jdk"]["artifacts"][self.key])
        self.zip_bytes = _ghidra_zip()
        self.tarball = _jdk_tarball()
        provision.DESCRIPTORS["ghidra"]["artifacts"]["any"] = {
            "url": "https://example.invalid/fake-ghidra.zip",
            "sha256": _sha(self.zip_bytes), "kind": "zip"}
        provision.DESCRIPTORS["jdk"]["artifacts"][self.key] = {
            "url": "https://example.invalid/fake-jdk.tar.gz",
            "sha256": _sha(self.tarball), "kind": "tar.gz"}

    def tearDown(self):
        provision.DESCRIPTORS["ghidra"]["artifacts"]["any"] = \
            self._saved_gh
        provision.DESCRIPTORS["jdk"]["artifacts"][self.key] = \
            self._saved_jdk

    def _dl(self, url: str, dest: Path) -> None:
        dest.write_bytes(self.zip_bytes
                         if url.endswith(".zip") else self.tarball)

    def test_provision_all_lands_both_and_keeps_exec_bits(self):
        with _Root() as root:
            report = provision.provision_all(root=root,
                                             downloader=self._dl)
            self.assertTrue(report["ok"], report)
            gh = provision.ghidra_state(root)
            self.assertEqual(gh["state"], "ready")
            self.assertEqual(gh["version"], "12.2")
            script = Path(gh["path"]) / "support" / "analyzeHeadless"
            self.assertTrue(os.access(script, os.X_OK),
                            "zip install lost the launch script's "
                            "exec bit")

    def test_ghidra_without_jdk_is_unavailable_with_reason(self):
        with _Root() as root:
            # install only the ghidra half, by hand, with its pin
            dest = root / "re" / "ghidra"
            dest.mkdir(parents=True)
            with zipfile.ZipFile(io.BytesIO(self.zip_bytes)) as zf:
                zf.extractall(dest)
            (dest / ".pinned-sha256").write_text(
                provision.DESCRIPTORS["ghidra"]["artifacts"]["any"]
                ["sha256"] + "\n")

            def runner(argv, **kw):
                return 0, 'openjdk version "17.0.9"\n'

            with mock.patch.object(provision.shutil, "which",
                                   return_value="/usr/lib/jvm17/java"):
                state = provision.ghidra_state(root, runner=runner)
            self.assertEqual(state["state"], "unavailable")
            self.assertIn("JDK", state["detail"])

    def test_marker_mismatch_is_not_trusted(self):
        with _Root() as root:
            dest = root / "re" / "ghidra" / "ghidra_12.2_PUBLIC"
            (dest / "support").mkdir(parents=True)
            (dest / "support" / "analyzeHeadless").write_text("x")
            (root / "re" / "ghidra" / ".pinned-sha256").write_text(
                "0" * 64 + "\n")
            state = provision.ghidra_state(root)
            self.assertFalse(state["available"])
            self.assertEqual(state["state"], "provisionable")
            self.assertIn("marker", state["detail"])


if __name__ == "__main__":
    unittest.main()
