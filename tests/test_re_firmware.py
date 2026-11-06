"""RE Pass 3 — the firmware rung as Evidence.

Everything here is pinned on REAL artifacts: the firmware fixtures
were built byte-honestly during the build (a hand-assembled
cpio-newc rootfs with planted files incl. a gcc-built ELF and a
credential-shaped config, gzip-wrapped in a uImage header with
real CRCs; a real mksquashfs image), and the unblob reports are
verbatim ``--report`` JSON from real unblob 26.6.4 runs over those
fixtures (only the verify-area path prefix was scrubbed). No
engine output was invented. Provisioning is tested with an
injected runner (the machinery around pip/venv), per Pass 2
convention; the real provisioning drive is exercised by the
engine-level dogfood recorded in the build log.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents.reverse import ReEngine, firmware, provision
from nomorals.agents.reverse.evidence import EvidenceStore
from nomorals.storage.db import Database

FIXTURES = Path(__file__).parent / "fixtures"


def _fw() -> bytes:
    return (FIXTURES / "re_firmware_image.bin").read_bytes()


def _cpio() -> bytes:
    return (FIXTURES / "re_firmware_rootfs.cpio").read_bytes()


def _squashfs() -> bytes:
    return (FIXTURES / "re_firmware_rootfs.squashfs").read_bytes()


def _report(name: str) -> list:
    return json.loads((FIXTURES / name).read_text())


class _TmpHome:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-fw-")
        return self.dir

    def __exit__(self, *exc):
        return False


class CensusTest(unittest.TestCase):
    def test_uimage_header_parsed_with_real_crcs(self):
        facts = firmware.census(_fw())
        self.assertTrue(facts["is_firmware"])
        self.assertEqual(facts["primary"], "uimage")
        ui = facts["parsed"]["uimage"]
        self.assertEqual(ui["name"], "DevonFixtureOS")
        self.assertEqual(ui["os"], "Linux")
        self.assertEqual(ui["arch"], "ARM")
        self.assertTrue(ui["header_crc_ok"])
        self.assertTrue(ui["data_crc_ok"])
        kinds = {(s["kind"], s["offset"])
                 for s in facts["signatures"]}
        self.assertIn(("uimage", 0), kinds)
        # the gzip wrapper peeks through to the cpio inside
        gzip_sig = [s for s in facts["signatures"]
                    if s["kind"] == "gzip"][0]
        self.assertEqual(gzip_sig["offset"], 64)
        self.assertEqual(gzip_sig["inner"], "cpio")

    def test_squashfs_superblock_parsed(self):
        facts = firmware.census(_squashfs())
        self.assertTrue(facts["is_firmware"])
        self.assertEqual(facts["primary"], "squashfs")
        sq = facts["parsed"]["squashfs"]
        self.assertEqual(sq["version"], "4.0")
        self.assertEqual(sq["inodes"], 10)
        self.assertEqual(sq["compression"], "gzip")
        self.assertEqual(sq["bytes_used"], 2773)

    def test_bare_cpio_census_lists_planted_entries(self):
        facts = firmware.census(_cpio())
        self.assertTrue(facts["is_firmware"])
        self.assertEqual(facts["parsed"]["cpio"]["entries"], 9)
        names = set(facts["parsed"]["cpio"]["names"])
        for want in ("etc/passwd", "etc/config/system",
                     "www/index.html", "usr/bin/hello"):
            self.assertIn(want, names)
        self.assertTrue(facts["parsed"]["cpio"]["complete"])

    def test_non_firmware_bytes_are_not_firmware(self):
        self.assertFalse(firmware.census(b"\x7fELF" + bytes(200))
                         ["is_firmware"])
        self.assertFalse(firmware.census(bytes(512))["is_firmware"])
        self.assertFalse(firmware.is_firmware_image(b"hello world"))


class NativeExtractionTest(unittest.TestCase):
    def test_extract_and_inventory_with_credential_flags(self):
        with _TmpHome() as home:
            dest = Path(home) / "out"
            ext = firmware.extract_native(_fw(), dest)
            self.assertTrue(ext["extracted"])
            self.assertEqual(ext["shape"], "uimage+gzip+cpio")
            inv = firmware.build_inventory(dest)
            self.assertEqual(inv["files_total"], 4)
            self.assertIn("etc/passwd", inv["notable_paths"])
            self.assertIn("etc/config/system", inv["notable_paths"])
            self.assertEqual(len(inv["carved_binaries"]), 1)
            self.assertEqual(inv["carved_binaries"][0]["path"],
                             "usr/bin/hello")
            self.assertEqual(inv["carved_binaries"][0]["kind"],
                             "elf")
            creds = inv["credential_flags"]
            self.assertEqual(creds["flag_count"], 2)
            patterns = {f["pattern"] for f in creds["flags"]}
            self.assertEqual(
                patterns, {"password-assignment",
                           "secret-assignment"})
            for flag in creds["flags"]:
                self.assertEqual(flag["path"], "etc/config/system")
                self.assertIn("…", flag["redacted"])
            # the planted values are never copied into the record —
            # only a two-character shape survives
            blob = json.dumps(creds)
            self.assertNotIn("hunter2-fixture", blob)

    def test_inventory_cap_refusal_deletes_overflow(self):
        with _TmpHome() as home:
            dest = Path(home) / "out"
            firmware.extract_native(_cpio(), dest)
            inv = firmware.build_inventory(dest, max_files=2)
            self.assertIn("refusal", inv)
            self.assertIn("cap", inv["refusal"]["reason"])
            self.assertGreater(inv["refusal"]["files_removed"], 0)
            remaining = [p for p in dest.rglob("*") if p.is_file()]
            self.assertLessEqual(len(remaining), 2)

    def test_native_extract_cap_is_a_journaled_refusal(self):
        with _TmpHome() as home:
            ext = firmware.extract_native(
                _cpio(), Path(home) / "out", max_bytes=10)
            self.assertTrue(ext["extracted"])
            self.assertIn("refused", ext)
            self.assertIn("cap", ext["refused"])

    def test_path_escape_is_never_materialized(self):
        # a cpio carrying a traversal name extracts nothing outside
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            data = b"escaped"
            info = tarfile.TarInfo("../evil.txt")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
            ok = b"fine"
            info2 = tarfile.TarInfo("ok.txt")
            info2.size = len(ok)
            tf.addfile(info2, io.BytesIO(ok))
        with _TmpHome() as home:
            dest = Path(home) / "out"
            ext = firmware.extract_native(buf.getvalue(), dest)
            self.assertTrue(ext["extracted"])
            self.assertFalse((Path(home) / "evil.txt").exists())
            self.assertTrue((dest / "ok.txt").is_file())


class UnblobReportTest(unittest.TestCase):
    def test_firmware_report_gzip_chunk_and_missing_7z(self):
        facts = firmware.parse_unblob_report(
            _report("re_unblob_fw_report.json"))
        self.assertEqual(len(facts["chunks"]), 1)
        chunk = facts["chunks"][0]
        self.assertEqual(chunk["handler"], "gzip")
        self.assertEqual((chunk["start"], chunk["end"]), (64, 2603))
        missing = {m for err in facts["extraction_errors"]
                   for m in err["missing"]}
        self.assertEqual(missing, {"7z"})
        # the uImage header itself is an unknown region to unblob
        self.assertEqual(
            [(u["start"], u["end"]) for u in facts["unknown_chunks"]],
            [(0, 64)])
        subject = [f for f in facts["files"]
                   if f["path"].endswith("firmware.bin")]
        self.assertTrue(subject and subject[0]["sha256"])

    def test_cpio_report_full_extraction_no_errors(self):
        facts = firmware.parse_unblob_report(
            _report("re_unblob_cpio_report.json"))
        handlers = {c["handler"] for c in facts["chunks"]}
        self.assertIn("cpio_portable_ascii", handlers)
        self.assertEqual(facts["extraction_errors"], [])

    def test_squashfs_report_names_sasquatch(self):
        facts = firmware.parse_unblob_report(
            _report("re_unblob_squashfs_report.json"))
        chunk = facts["chunks"][0]
        self.assertEqual(chunk["handler"], "squashfs_v4_le")
        self.assertEqual((chunk["start"], chunk["end"]), (0, 4096))
        missing = {m for err in facts["extraction_errors"]
                   for m in err["missing"]}
        self.assertEqual(missing, {"sasquatch"})

    def test_fold_names_missing_extractors_as_unknowns(self):
        store = EvidenceStore(Database(":memory:"))
        facts = firmware.parse_unblob_report(
            _report("re_unblob_fw_report.json"))
        folded = firmware.fold_evidence(
            store, subject_sha256="ab" * 32, subject_name="fw.bin",
            subject_size=2603, subject_path="/x/fw.bin",
            census_facts=firmware.census(_fw()),
            unblob_facts={**facts, "rc": 1, "version": "26.6.4"})
        kinds = [store.get(eid).predicate_type
                 for eid in folded["evidence_ids"]]
        self.assertIn("firmware.chunks", kinds)
        questions = [u["question"] for u in folded["unknowns"]]
        self.assertTrue(any("7z" in q for q in questions))
        self.assertTrue(any("0–64" in q for q in questions))
        chunk_rec = [store.get(eid) for eid in folded["evidence_ids"]
                     if store.get(eid).predicate_type
                     == "firmware.chunks"][0]
        self.assertEqual(chunk_rec.provider_id, "devon.unblob")
        self.assertTrue(chunk_rec.limitations)


class BinwalkProbeTest(unittest.TestCase):
    def test_binwalk_is_probe_only_with_reason(self):
        state = firmware.probe_binwalk()
        self.assertIn("probe-only", state["detail"])
        self.assertIn("source-only", state["detail"])
        if not state["available"]:
            self.assertIn("no system binwalk", state["detail"])


class _FakePipRunner:
    """Simulates venv creation + a hash-verified pip install +
    the version probe, creating the layout the probe expects."""

    def __init__(self, pip_rc: int = 0):
        self.calls: list[list[str]] = []
        self.pip_rc = pip_rc

    def __call__(self, argv, timeout=None, env=None):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        if argv[1:3] == ["-m", "venv"]:
            venv = Path(argv[3])
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python").write_text("#!fake\n")
            return 0, "venv created"
        if "pip" in argv and "install" in argv:
            if self.pip_rc != 0:
                return self.pip_rc, "ERROR: hash mismatch"
            bindir = Path(argv[0]).parent
            (bindir / "unblob").write_text("#!fake\n")
            return 0, "Successfully installed unblob-26.6.4"
        if argv[0].endswith("unblob"):
            return 0, "26.6.4\n"
        return 0, ""


class UnblobProvisionTest(unittest.TestCase):
    def test_provision_drive_installs_and_probes(self):
        with _TmpHome() as home:
            root = Path(home)
            runner = _FakePipRunner()
            state = provision.provision_engine(
                "unblob", root=root, runner=runner)
            self.assertTrue(state["available"])
            self.assertEqual(state["source"], "provisioned")
            self.assertEqual(state["version"], "26.6.4")
            joined = [" ".join(c) for c in runner.calls]
            self.assertTrue(any("--require-hashes" in c
                                for c in joined))
            self.assertTrue(any("created the engine venv" in a
                                for a in state["attempts"]))

    def test_pip_failure_is_a_named_attempt_and_trusts_nothing(self):
        with _TmpHome() as home:
            root = Path(home)
            runner = _FakePipRunner(pip_rc=1)
            with self.assertRaises(provision.ProvisionError) as ctx:
                provision.provision_engine(
                    "unblob", root=root, runner=runner)
            attempts = " ".join(ctx.exception.attempts)
            self.assertIn("--require-hashes failed", attempts)
            self.assertIn("nothing is trusted", attempts)
            state = provision.unblob_state(root, runner=runner)
            self.assertFalse(state["available"])

    def test_marker_mismatch_degrades_to_provisionable(self):
        with _TmpHome() as home:
            root = Path(home)
            runner = _FakePipRunner()
            provision.provision_engine("unblob", root=root,
                                       runner=runner)
            base = provision.re_root(root) / "unblob"
            (base / ".pinned-sha256").write_text("0" * 64 + "\n")
            state = provision.unblob_state(root, runner=runner)
            self.assertFalse(state["available"])
            self.assertEqual(state["state"], "provisionable")
            self.assertIn("does not match", state["attempts"][0])

    def test_pins_file_carries_published_digests(self):
        pins = provision.unblob_pins_path()
        text = pins.read_text()
        self.assertIn("--hash=sha256:", text)
        self.assertGreater(text.count("--hash=sha256:"), 400)
        self.assertIn("unblob==26.6.4", text)


class FirmwareRoutingTest(unittest.TestCase):
    def _engine(self, root: Path) -> ReEngine:
        return ReEngine(db=Database(":memory:"), tools_root=root)

    def test_engine_routes_firmware_and_extracts_natively(self):
        with _TmpHome() as home:
            target = Path(home) / "router.bin"
            target.write_bytes(_fw())
            engine = self._engine(Path(home) / "tools")
            with mock.patch.object(
                    provision, "unblob_state",
                    return_value={"engine": "unblob",
                                  "state": "provisionable",
                                  "available": False, "source": "",
                                  "version": "26.6.4", "path": "",
                                  "attempts": [],
                                  "detail": "not provisioned"}):
                res = engine.inspect_path(str(target))
            self.assertEqual(res["format"], "firmware")
            self.assertEqual(res["extraction"], "native")
            kinds = [r["predicate_type"] for r in res["evidence"]]
            self.assertIn("firmware.census", kinds)
            self.assertIn("firmware.inventory", kinds)
            self.assertIn("firmware.credential_flags", kinds)
            self.assertIn("native extraction", res["rung"])

    def test_engine_degrades_honestly_without_an_extractor(self):
        with _TmpHome() as home:
            target = Path(home) / "rootfs.squashfs"
            target.write_bytes(_squashfs())
            engine = self._engine(Path(home) / "tools")
            with mock.patch.object(
                    provision, "unblob_state",
                    return_value={"engine": "unblob",
                                  "state": "provisionable",
                                  "available": False, "source": "",
                                  "version": "26.6.4", "path": "",
                                  "attempts": [],
                                  "detail": "not provisioned"}):
                res = engine.inspect_path(str(target))
            self.assertEqual(res["format"], "firmware")
            self.assertEqual(res["extraction"], "none")
            questions = [u["question"] for u in res["unknowns"]]
            self.assertTrue(any("unblob" in q for q in questions))
            self.assertIn("unblob", res["card_text"])

    def test_content_routing_ignores_the_file_name(self):
        with _TmpHome() as home:
            target = Path(home) / "definitely-not-firmware.txt"
            target.write_bytes(_cpio())
            engine = self._engine(Path(home) / "tools")
            res = engine.inspect_path(str(target))
            self.assertEqual(res["format"], "firmware")


if __name__ == "__main__":
    unittest.main()
