"""RE Pass 1 — the Evidence core: content-addressed records,
provenance chains, and the residual-unknowns lifecycle.

These tests pin the discipline itself: the same observation always
lands at the same address; a record with the same address but a
different result can never silently overwrite; closing an unknown
needs the authority and confidence the unknown demands, stated;
closing without the required resolution is refused.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from nomorals.agents.reverse.evidence import EvidenceStore, UnknownCloseError
from nomorals.storage.db import Database


def _store(tmp: bool = False) -> tuple[EvidenceStore, Database]:
    if tmp:
        home = tempfile.mkdtemp(prefix="re-ev-")
        db = Database(os.path.join(home, "re.db"))
    else:
        db = Database(":memory:")
    return EvidenceStore(db), db


def _record(store: EvidenceStore, predicate: str = "format.identify",
            result=None, **kw) -> str:
    return store.record(
        subject_sha256="ab" * 32, subject_name="sample.bin",
        provider_id="test.provider", predicate_type=predicate,
        operation="identify", parameters={"format": "pe"},
        result=result or {"format": "pe"}, **kw).evidence_id


class RecordRoundTripTest(unittest.TestCase):
    def test_record_get_list(self):
        store, _ = _store()
        eid = _record(store, limitations=["static parse only"])
        got = store.get(eid)
        self.assertIsNotNone(got)
        self.assertEqual(got.subject_name, "sample.bin")
        self.assertEqual(got.confidence, "observed")
        self.assertEqual(got.authority, "shipped-artifact")
        self.assertEqual(list(got.limitations), ["static parse only"])
        self.assertEqual(got.to_dict()["result"], {"format": "pe"})
        rows = store.list_recent()
        self.assertEqual([r.evidence_id for r in rows], [eid])
        self.assertEqual(store.audit()["evidence_records"], 1)

    def test_content_address_is_stable_and_dedupes(self):
        store, _ = _store()
        a = _record(store)
        b = _record(store)
        self.assertEqual(a, b)
        self.assertEqual(store.audit()["evidence_records"], 1)

    def test_same_address_different_result_keeps_original(self):
        store, _ = _store()
        a = _record(store, result={"format": "pe"})
        b = _record(store, result={"format": "elf"})
        self.assertEqual(a, b)  # same identity inputs → same address
        got = store.get(a)
        self.assertEqual(got.result, {"format": "pe"})

    def test_prefix_lookup(self):
        store, _ = _store()
        eid = _record(store)
        short = eid[:16]
        got = store.get(short)
        self.assertIsNotNone(got)
        self.assertEqual(got.evidence_id, eid)
        self.assertIsNone(store.get("ev:does-not-exist"))

    def test_raw_result_capped(self):
        store, _ = _store()
        big = {"blob": "x" * 50000}
        eid = _record(store, result={"format": "pe"}, raw_result=big)
        got = store.get(eid)
        self.assertLessEqual(len(got.raw_result), 20020)

    def test_persistence_across_connections(self):
        home = tempfile.mkdtemp(prefix="re-ev-")
        path = os.path.join(home, "re.db")
        store = EvidenceStore(Database(path))
        eid = _record(store)
        store2 = EvidenceStore(Database(path))
        self.assertIsNotNone(store2.get(eid))


class ProvenanceChainTest(unittest.TestCase):
    def test_chain_walks_links_in_discovery_order(self):
        store, _ = _store()
        root = _record(store, predicate="format.identify")
        mid = store.record(
            subject_sha256="ab" * 32, subject_name="sample.bin",
            provider_id="test.provider", predicate_type="pe.structure",
            operation="inspect", parameters={}, result={},
            links=[root]).evidence_id
        leaf = store.record(
            subject_sha256="ab" * 32, subject_name="sample.bin",
            provider_id="test.provider",
            predicate_type="pe.hardening", operation="inspect",
            parameters={}, result={}, confidence="derived",
            links=[mid]).evidence_id
        chain = store.provenance_chain(leaf)
        self.assertEqual([r.evidence_id for r in chain],
                         [leaf, mid, root])

    def test_chain_tolerates_cycles(self):
        store, _ = _store()
        a = _record(store, predicate="a.one")
        b = store.record(
            subject_sha256="ab" * 32, subject_name="sample.bin",
            provider_id="test.provider", predicate_type="b.two",
            operation="inspect", parameters={}, result={},
            links=[a]).evidence_id
        chain = store.provenance_chain(b)
        self.assertEqual(len(chain), 2)


class UnknownLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.store, _ = _store()
        self.anchor = _record(self.store)

    def test_open_and_get(self):
        unk = self.store.open_unknown(
            "does it phone home?", subject_sha256="ab" * 32,
            raised_by=self.anchor,
            supporting=[self.anchor])
        self.assertEqual(unk.status, "open")
        self.assertTrue(unk.unknown_id.startswith("un:"))
        got = self.store.get_unknown(unk.unknown_id)
        self.assertEqual(got.question, "does it phone home?")

    def test_reopen_same_question_is_idempotent(self):
        a = self.store.open_unknown("q?", raised_by=self.anchor)
        b = self.store.open_unknown("q?", raised_by=self.anchor)
        self.assertEqual(a.unknown_id, b.unknown_id)
        self.assertEqual(self.store.audit()["unknowns_open"], 1)

    def test_close_resolved_happy_path(self):
        unk = self.store.open_unknown("q?", raised_by=self.anchor)
        closed = self.store.close_unknown(
            unk.unknown_id, status="resolved",
            resolution="no network strings or imports",
            resolution_authority="shipped-artifact",
            resolution_confidence="observed")
        self.assertEqual(closed.status, "resolved")
        self.assertEqual(closed.resolution_confidence, "observed")

    def test_close_without_resolution_refused(self):
        unk = self.store.open_unknown("q?", raised_by=self.anchor)
        with self.assertRaises(UnknownCloseError):
            self.store.close_unknown(unk.unknown_id, status="resolved")

    def test_close_weaker_than_required_refused(self):
        unk = self.store.open_unknown(
            "runtime behavior?", raised_by=self.anchor,
            required_authority="controlled-replay",
            required_confidence="derived")
        with self.assertRaises(UnknownCloseError) as cm:
            self.store.close_unknown(
                unk.unknown_id, status="resolved", resolution="guess",
                resolution_authority="analyst-inference",
                resolution_confidence="inferred")
        self.assertIn("requires authority", str(cm.exception))
        # still open after the refused close
        self.assertEqual(
            self.store.get_unknown(unk.unknown_id).status, "open")

    def test_contradicted_needs_evidence(self):
        unk = self.store.open_unknown("q?", raised_by=self.anchor)
        with self.assertRaises(UnknownCloseError):
            self.store.close_unknown(unk.unknown_id,
                                     status="contradicted")
        closed = self.store.close_unknown(
            unk.unknown_id, status="contradicted",
            evidence_ids=[self.anchor])
        self.assertEqual(closed.status, "contradicted")

    def test_double_close_refused(self):
        unk = self.store.open_unknown("q?", raised_by=self.anchor)
        self.store.close_unknown(
            unk.unknown_id, status="resolved", resolution="done")
        with self.assertRaises(UnknownCloseError):
            self.store.close_unknown(
                unk.unknown_id, status="resolved", resolution="again")

    def test_audit_flags_unclosable(self):
        # its own store: the only evidence on record is weak
        # (analyst-inference / inferred), so a question demanding
        # shipped-artifact observation is unclosable TODAY, while
        # the class fixture's strong anchor would satisfy it
        store, _ = _store()
        weak = store.record(
            subject_sha256="cd" * 32, subject_name="w.bin",
            provider_id="analyst.notes", predicate_type="claim.note",
            operation="note", parameters={}, result={},
            confidence="inferred", authority="analyst-inference",
        ).evidence_id
        needy = store.open_unknown(
            "needs artifact proof", raised_by=weak,
            supporting=[weak],
            required_authority="shipped-artifact",
            required_confidence="observed")
        orphan = store.open_unknown("orphan question",
                                    raised_by=weak)
        audit = store.audit()
        self.assertEqual(audit["unknowns_open"], 2)
        self.assertEqual(audit["open_unclosable_on_current_evidence"],
                         [needy.unknown_id])
        self.assertEqual(audit["open_without_supporting_evidence"],
                         [orphan.unknown_id])

    def test_export_import_round_trip(self):
        other = _record(self.store, predicate="pe.structure",
                        result={"sections": 2})
        self.store.open_unknown("q?", raised_by=self.anchor,
                                supporting=[other])
        payload = self.store.export()
        self.assertEqual(len(payload["evidence"]), 2)
        self.assertEqual(len(payload["unknowns"]), 1)
        fresh, _ = _store()
        counts = fresh.import_(payload)
        self.assertEqual(counts, {"evidence": 2, "unknowns": 1})
        # re-import is a no-op (content addressing)
        counts2 = fresh.import_(payload)
        self.assertEqual(counts2, {"evidence": 0, "unknowns": 0})


if __name__ == "__main__":
    unittest.main()
