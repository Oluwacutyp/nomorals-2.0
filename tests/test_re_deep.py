"""RE Pass 2 — the deep rungs as Evidence.

Everything here is pinned on REAL engine output recorded during
the build (tests/fixtures/): Ghidra 12.2 headless — provisioned
for real from the pinned artifacts — analyzed a gcc-built fixture
binary, and its verbatim export JSON is what the fold consumes;
the readelf fixture is the verbatim stdout of binutils over the
same binary. No engine JSON was invented for these tests. Where a
test needs the engine itself it either injects a runner (the
machinery around the engine) or skips when the tool is absent.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents.reverse import ReEngine, ghidra, lightbin
from nomorals.agents.reverse.evidence import EvidenceStore
from nomorals.storage.db import Database
from tests.test_re_inspectors import build_elf64

FIXTURES = Path(__file__).parent / "fixtures"


def _export() -> dict:
    return json.loads(
        (FIXTURES / "re_ghidra_fixture.json").read_text())


def _readelf_views() -> dict[str, str]:
    text = (FIXTURES / "re_readelf_fixture.txt").read_text()
    views: dict[str, str] = {}
    current = ""
    for line in text.splitlines(keepends=True):
        if line.startswith("=== "):
            current = line.strip().strip("= ").strip()
            views[current] = ""
        elif current:
            views[current] += line
    return views


class _TmpHome:
    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="re-deep-")
        return self.dir

    def __exit__(self, *exc):
        return False


class GhidraFoldTest(unittest.TestCase):
    def test_fold_recorded_real_export(self):
        store = EvidenceStore(Database(":memory:"))
        folded = ghidra.fold_evidence(
            store, subject_sha256="ab" * 32, subject_name="fixture",
            subject_size=17000, subject_path="/x/fixture",
            export=_export(), link_to="")
        self.assertGreaterEqual(len(folded["evidence_ids"]), 5)
        records = [store.get(eid) for eid in folded["evidence_ids"]]
        kinds = [r.predicate_type for r in records]
        self.assertEqual(kinds[0], "native.program")
        for want in ("native.functions", "native.decompile",
                     "native.imports", "native.strings"):
            self.assertIn(want, kinds)
        for rec in records:
            self.assertEqual(rec.provider_id,
                             "devon.ghidra-headless")
            self.assertEqual(rec.authority, "shipped-artifact")
            # the no-source-recovery limit rides on every record
            self.assertTrue(any("not the original source" in lim
                                for lim in rec.limitations), rec)
        fns = records[kinds.index("native.functions")]
        names = {f["name"] for f in fns.result["functions"]}
        self.assertIn("main", names)
        main = next(f for f in fns.result["functions"]
                    if f["name"] == "main")
        self.assertIn("strtol", main["calls"])
        decomp = records[kinds.index("native.decompile")]
        snippets = {s["function"]: s["code"]
                    for s in decomp.result["snippets"]}
        self.assertIn("main", snippets)
        # the fixture's real gate constant, in the engine's own
        # reconstructed pseudocode (4242 == 0x1092)
        self.assertIn("0x1092", snippets["main"])
        self.assertEqual(decomp.confidence, "derived")
        imports = records[kinds.index("native.imports")]
        imp_names = {i["name"] for i in imports.result["imports"]}
        self.assertIn("__libc_start_main", imp_names)
        strings = records[kinds.index("native.strings")]
        values = {s["value"] for s in strings.result["strings"]}
        self.assertIn("DEVON_FIXTURE_BANNER string-for-ghidra",
                      values)


class GhidraAnalyzeTest(unittest.TestCase):
    def _provisioned_root(self, root: Path) -> None:
        from nomorals.agents.reverse import provision
        jdk_bin = root / "re" / "jdk" / "jdk-25" / "bin"
        jdk_bin.mkdir(parents=True)
        (jdk_bin / "java").write_text(
            "#!/bin/sh\necho 'openjdk version \"25.0.5\"' 1>&2\n")
        os.chmod(jdk_bin / "java", 0o755)
        (root / "re" / "jdk" / ".pinned-sha256").write_text(
            provision.DESCRIPTORS["jdk"]["artifacts"]
            [provision.platform_key()]["sha256"] + "\n")
        gh = root / "re" / "ghidra" / "ghidra_12.2_PUBLIC"
        (gh / "support").mkdir(parents=True)
        (gh / "support" / "analyzeHeadless").write_text("#!/bin/sh\n")
        (gh / "Ghidra").mkdir(parents=True)
        (gh / "Ghidra" / "application.properties").write_text(
            "application.version=12.2\n")
        (root / "re" / "ghidra" / ".pinned-sha256").write_text(
            provision.DESCRIPTORS["ghidra"]["artifacts"]["any"]
            ["sha256"] + "\n")

    def test_analyze_with_injected_runner(self):
        export = _export()
        with _TmpHome() as home:
            root = Path(home)
            self._provisioned_root(root)
            target = root / "target.bin"
            target.write_bytes(build_elf64())

            def runner(argv, **kw):
                if any("DevonGhidraExport.java" in a for a in argv):
                    out = Path(argv[argv.index(
                        "DevonGhidraExport.java") + 1])
                    out.write_text(json.dumps(export))
                    return 0, "INFO  analysis succeeded"
                if str(argv[0]).endswith("java"):
                    return 0, 'openjdk version "25.0.5" 2026\n'
                return 0, ""

            data = ghidra.analyze(target, root=root, runner=runner)
        self.assertEqual(data["program"]["name"], "fixture")
        self.assertEqual(data["_engine"]["version"], "12.2")

    def test_analyze_refuses_when_unprovisioned(self):
        with _TmpHome() as home:
            target = Path(home) / "target.bin"
            target.write_bytes(build_elf64())
            with self.assertRaises(ghidra.GhidraError) as ctx:
                ghidra.analyze(target, root=Path(home))
            self.assertIn("unavailable", str(ctx.exception))


class LightRungTest(unittest.TestCase):
    def test_parse_recorded_real_readelf(self):
        views = _readelf_views()
        facts = lightbin.parse_readelf(
            views["header"], views["dynamic"], views["dynsyms"],
            views["sections"])
        self.assertEqual(facts["elf_class"], "ELF64")
        self.assertEqual(facts["machine"],
                         "Advanced Micro Devices X86-64")
        self.assertEqual(facts["needed_libraries"], ["libc.so.6"])
        self.assertTrue(facts["bind_now"])
        self.assertGreater(facts["defined_function_count"], 0)
        self.assertIn("main", facts["defined_functions"])
        self.assertIn("strtol", facts["imported_symbols"])
        self.assertIn(".text", facts["sections"])

    def test_probe_is_honest_on_this_box(self):
        state = lightbin.probe()
        # the test box carries binutils; where it does not, the
        # probe must say so with the reason instead of guessing
        self.assertIn("detail", state)
        if state["available"]:
            self.assertIn("readelf", state["version"])

    def test_triage_live_on_synthetic_elf(self):
        if not lightbin.probe()["available"]:
            self.skipTest("binutils not on this box")
        with _TmpHome() as home:
            target = Path(home) / "synthetic"
            target.write_bytes(build_elf64())
            facts = lightbin.triage(target)
        self.assertEqual(facts["elf_class"], "ELF64")


class RoutingTest(unittest.TestCase):
    def _engine(self, root: Path) -> ReEngine:
        return ReEngine(db=Database(":memory:"), tools_root=root)

    def _write_elf(self, home: str) -> str:
        path = os.path.join(home, "sample")
        with open(path, "wb") as fh:
            fh.write(build_elf64())
        return path

    def test_deep_off_keeps_pass1_shape(self):
        with _TmpHome() as home:
            engine = self._engine(Path(home))
            res = engine.inspect_path(self._write_elf(home),
                                      deep="off")
        self.assertEqual(res["rung"], "built-in static spine")
        self.assertFalse(any(
            "deep behavior" in u["question"]
            for u in res["unknowns"]))

    def test_no_rung_available_names_the_gap_as_unknown(self):
        with _TmpHome() as home:
            engine = self._engine(Path(home))
            with mock.patch.object(
                    lightbin, "probe",
                    return_value={"available": False, "path": "",
                                  "version": "", "detail": "no"}):
                res = engine.inspect_path(self._write_elf(home))
        self.assertEqual(res["rung"], "built-in static spine")
        deep_unknowns = [u for u in res["unknowns"]
                         if "deep behavior" in u["question"]]
        self.assertEqual(len(deep_unknowns), 1)
        self.assertEqual(deep_unknowns[0]["required_authority"],
                         "shipped-artifact")
        self.assertEqual(deep_unknowns[0]["required_confidence"],
                         "derived")

    def test_light_rung_answers_when_present(self):
        if not lightbin.probe()["available"]:
            self.skipTest("binutils not on this box")
        with _TmpHome() as home:
            engine = self._engine(Path(home))
            res = engine.inspect_path(self._write_elf(home))
        self.assertEqual(res["rung"], "binutils-triage")
        kinds = [e["predicate_type"] for e in res["evidence"]]
        self.assertIn("elf.triage", kinds)

    def test_ghidra_rung_is_deepest_when_ready(self):
        export = _export()
        with _TmpHome() as home:
            engine = self._engine(Path(home))
            fake_state = {"available": True, "state": "ready",
                          "version": "12.2", "path": "/fake",
                          "detail": "fake", "source": "provisioned",
                          "attempts": []}
            from nomorals.agents.reverse import provision
            with mock.patch.object(provision, "ghidra_state",
                                   return_value=fake_state), \
                    mock.patch.object(ghidra, "analyze",
                                      return_value=export):
                res = engine.inspect_path(self._write_elf(home))
        self.assertEqual(res["rung"], "ghidra-headless")
        kinds = [e["predicate_type"] for e in res["evidence"]]
        self.assertIn("native.functions", kinds)
        self.assertIn("native.decompile", kinds)
        self.assertEqual(res["deep"]["functions"], 18)


class MatrixTest(unittest.TestCase):
    def test_matrix_states_on_empty_root(self):
        from nomorals.agents.reverse import capability_matrix
        with _TmpHome() as home:
            rows = {r["engine"]: r
                    for r in capability_matrix(Path(home))}
        self.assertEqual(rows["ghidra-headless"]["state"],
                         "provisionable")
        self.assertFalse(rows["ghidra-headless"]["available"])
        self.assertEqual(rows["temurin-jdk"]["state"],
                         "provisionable")
        self.assertTrue(rows["android-graph"]["available"])
        self.assertEqual(rows["rizin-objdump"]["available"],
                         lightbin.probe()["available"])


if __name__ == "__main__":
    unittest.main()
