"""RE Pass 4 — reconstruction verify: evidence-tied claims,
cross-rung agreement, contradiction demotion, structural refusal.

The folds consume the Pass 2/3 recorded REAL engine output
(tests/fixtures/: the Ghidra 12.2 export and the binutils stdout
of the same gcc-built binary, the recorded firmware rootfs, the
recorded capture). The one planted disagreement is built the
honest way: two records over the same subject whose dynamic-
section readings differ — exactly the shape a real contradiction
would take when it lands in the store.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nomorals.agents.reverse import firmware, lightbin, traffic
from nomorals.agents.reverse.evidence import EvidenceStore
from nomorals.agents.reverse.verify import (
    Claim,
    ReconstructionError,
    build_reconstruction,
    build_report,
)
from nomorals.storage.db import Database

FIXTURES = Path(__file__).parent / "fixtures"
SHA = "ab" * 32


def _store() -> EvidenceStore:
    return EvidenceStore(Database(":memory:"))


def _export() -> dict:
    return json.loads(
        (FIXTURES / "re_ghidra_fixture.json").read_text())


def _readelf_facts() -> dict:
    text = (FIXTURES / "re_readelf_fixture.txt").read_text()
    views: dict[str, str] = {}
    current = ""
    for line in text.splitlines(keepends=True):
        if line.startswith("=== "):
            current = line.strip().strip("= ").strip()
            views[current] = ""
        elif current:
            views[current] += line
    return lightbin.parse_readelf(views["header"], views["dynamic"],
                                  views["dynsyms"], views["sections"])


def _fold_deep(store: EvidenceStore) -> None:
    from nomorals.agents.reverse import ghidra

    ghidra.fold_evidence(
        store, subject_sha256=SHA, subject_name="fixture",
        subject_size=17000, subject_path="/x/fixture",
        export=_export(), link_to="")
    lightbin.fold_evidence(
        store, subject_sha256=SHA, subject_name="fixture",
        subject_size=17000, subject_path="/x/fixture",
        facts=_readelf_facts(), link_to="")


def _assert_all_tied(test: unittest.TestCase,
                     report: dict, store: EvidenceStore) -> None:
    chain = {r.evidence_id for r in store.list_for_subject(SHA)}
    test.assertTrue(report["claims"])
    for claim in report["claims"]:
        test.assertTrue(claim["evidence_ids"], claim)
        for eid in claim["evidence_ids"]:
            test.assertIn(eid, chain, claim)


class DeepReconstructionTest(unittest.TestCase):
    def test_claims_from_recorded_real_folds(self):
        store = _store()
        _fold_deep(store)
        report = build_reconstruction(store, SHA)
        _assert_all_tied(self, report, store)
        texts = [c["text"] for c in report["claims"]]
        # identity + behavior from the deep rung's own records
        self.assertTrue(any("Executable and Linking Format" in t
                            for t in texts), texts)
        self.assertTrue(any("functions" in t for t in texts), texts)
        # the import census is corroborated: two engines read the
        # same symbol table and their censuses overlap
        imports = [c for c in report["claims"]
                   if c["text"].startswith("imports ")]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0]["status"], "corroborated")
        self.assertIn("strtol", imports[0]["text"])
        checks = {c["name"]: c["verdict"] for c in report["checks"]}
        self.assertEqual(checks.get("import-census overlap"),
                         "agree")
        # main's pseudocode literally references recorded strings
        refs = [c for c in report["claims"]
                if "references" in c["text"]
                and "DEVON_FIXTURE_BANNER" in c["text"]]
        self.assertTrue(refs, texts)
        self.assertEqual(refs[0]["status"], "corroborated")
        # the report itself is an evidence record over the chain
        rec = store.get(report["report_evidence_id"])
        self.assertIsNotNone(rec)
        self.assertEqual(rec.predicate_type, "reconstruction.report")
        self.assertEqual(rec.authority, "analyst-inference")
        self.assertEqual(rec.confidence, "derived")

    def test_rebuild_over_same_chain_is_the_same_record(self):
        store = _store()
        _fold_deep(store)
        first = build_reconstruction(store, SHA)
        second = build_reconstruction(store, SHA)
        self.assertEqual(first["report_evidence_id"],
                         second["report_evidence_id"])


class StructuralRefusalTest(unittest.TestCase):
    def test_claim_without_evidence_is_refused(self):
        claim = Claim(kind="behavior", text="it phones home",
                      evidence_ids=())
        with self.assertRaises(ReconstructionError):
            build_report([claim], subject={"sha256": SHA},
                         chain_ids={"ev:x"}, checks=[], conflicts=[])

    def test_claim_citing_outside_the_chain_is_refused(self):
        claim = Claim(kind="behavior", text="it phones home",
                      evidence_ids=("ev:elsewhere",))
        with self.assertRaises(ReconstructionError):
            build_report([claim], subject={"sha256": SHA},
                         chain_ids={"ev:mine"}, checks=[],
                         conflicts=[])

    def test_no_chain_no_reconstruction(self):
        store = _store()
        with self.assertRaises(ReconstructionError):
            build_reconstruction(store, "cd" * 32)


class ContradictionDemotionTest(unittest.TestCase):
    def test_disagreeing_parsers_demote_the_claim(self):
        store = _store()
        spine = store.record(
            subject_sha256=SHA, subject_name="fixture",
            provider_id="devon.builtin-static",
            predicate_type="elf.structure", operation="inspect",
            parameters={"format": "elf"},
            result={"machine": "x86-64",
                    "needed_libraries": ["liba.so.1"]})
        lightbin.fold_evidence(
            store, subject_sha256=SHA, subject_name="fixture",
            subject_size=10, subject_path="/x/fixture",
            facts={"needed_libraries": ["libb.so.1"],
                   "imported_symbols": ["printf"],
                   "imported_symbol_count": 1,
                   "defined_functions": [],
                   "defined_function_count": 0,
                   "dynamic_symbols": 1},
            link_to="")
        report = build_reconstruction(store, SHA)
        texts = [c["text"] for c in report["claims"]]
        self.assertFalse(
            any("dynamically links against" in t for t in texts),
            texts)
        self.assertEqual(len(report["conflicts"]), 1)
        conflict = report["conflicts"][0]
        self.assertEqual(conflict["check"],
                         "needed-libraries agreement")
        self.assertIn("liba.so.1", conflict["detail"])
        self.assertIn("libb.so.1", conflict["detail"])
        # the demotion is an open unknown carrying the dissent
        self.assertTrue(conflict["unknown_id"])
        unk = store.get_unknown(conflict["unknown_id"])
        self.assertIsNotNone(unk)
        self.assertEqual(unk.status, "open")
        self.assertIn("reconstruction conflict", unk.question)
        self.assertTrue(unk.contradicting)
        self.assertIn(spine.evidence_id, unk.contradicting)


class FirmwareReconstructionTest(unittest.TestCase):
    def test_census_and_extraction_agree(self):
        store = _store()
        data = (FIXTURES / "re_firmware_rootfs.cpio").read_bytes()
        census = firmware.census(data)
        self.assertTrue(census["is_firmware"])
        with tempfile.TemporaryDirectory() as home:
            native = firmware.extract_native(
                data, Path(home) / "out")
            self.assertTrue(native.get("extracted"))
            inventory = firmware.build_inventory(
                Path(home) / "out")
            firmware.fold_evidence(
                store, subject_sha256=SHA, subject_name="rootfs.cpio",
                subject_size=len(data), subject_path="/x/rootfs.cpio",
                census_facts=census, unblob_facts=None,
                inventory=inventory, extraction_source="native",
                link_to="")
            report = build_reconstruction(store, SHA)
        _assert_all_tied(self, report, store)
        inv = [c for c in report["claims"]
               if c["text"].startswith("extraction yielded")]
        self.assertEqual(len(inv), 1)
        self.assertEqual(inv[0]["status"], "corroborated")
        checks = {c["name"]: c["verdict"] for c in report["checks"]}
        self.assertEqual(
            checks.get("firmware census ↔ extraction"), "agree")


class TrafficReconstructionTest(unittest.TestCase):
    def test_capture_claims_from_recorded_pcap(self):
        store = _store()
        data = (FIXTURES / "re_capture.pcap").read_bytes()
        packets, fmt = traffic.read_packets(data, "pcap")
        facts = traffic.dissect(packets, fmt)
        traffic.fold_evidence(
            store, subject_sha256=SHA, subject_name="capture.pcap",
            subject_size=len(data), subject_path="/x/capture.pcap",
            kind="pcap", facts=facts, tshark_facts=None, link_to="")
        report = build_reconstruction(store, SHA)
        _assert_all_tied(self, report, store)
        texts = [c["text"] for c in report["claims"]]
        self.assertTrue(
            any(f"holds {facts['packets']} packets" in t
                for t in texts), texts)
        if facts.get("protocols"):
            self.assertTrue(any(t.startswith("protocol census")
                                for t in texts), texts)


if __name__ == "__main__":
    unittest.main()
