"""Docs 22+23 Stage A — the cooperative foundation, pinned.

Four pieces, each a projection or schema over organs that already
run: the CraftGraph projection (P1), role contracts over the
shipped delegation/role machinery (P2), the pack schema in the
filebook skill-form (P3), and the universal receipt schema whose
grades derive through core/proof_grade (P4).  Plus the levels
mapping table PG1's adoption rule requires.

The legitimacy-filter items (R1–R5) were severed by the owner's
order when docs 22+23 were adopted: no test here pins one because
none was built — no pack carries a refusals section and no goal
anywhere is inspected.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.agents import role_contracts as rc
from nomorals.agents.manifests import MANIFESTS
from nomorals.agents.manifests import manifest as manifest_for
from nomorals.agents.role_specs import BUILTIN_ROLES as BUILTIN_SPECS
from nomorals.agents.role_specs import RoleSpec, render_role_prompt
from nomorals.core.proof_grade import EvidenceRef, ProofGradeError
from nomorals.craft import QualityReceipt, cap_claim, grade_for_level, is_handoff_level, parse_level
from nomorals.craft.levels import LEVEL_GRADE_MAPPINGS, LevelError
from nomorals.craft.receipts import (
    HandoffReceipt,
    adapt_bill_receipt,
    adapt_case_receipt,
    adapt_content_receipt,
    adapt_delivery_receipt,
    adapt_police_receipt,
    handoff_for,
    render_receipt_chain,
)
from nomorals.craft.seed import SEED_DOMAINS
from nomorals.skills.filebook import FileSkillLibrary
from nomorals.skills.packs import is_pack, load_pack, pack_summary, validate_pack
from nomorals.storage.db import Database


def _tmp_db(testcase: unittest.TestCase) -> Database:
    tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR"))
    testcase.addCleanup(
        lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    return db


class GraphProjectionTests(unittest.TestCase):
    """P1: the graph computes from real shipped record shapes."""

    def setUp(self) -> None:
        from nomorals.agents.delegation import DelegationStore
        from nomorals.browser.jobs import BrowserRunRecordStore

        self.db = _tmp_db(self)
        store = DelegationStore(self.db)
        self.contract = store.create_contract(
            goal="verify the thing", worker="worker-a",
            verifier="verifier-b", witness="witness-c",
            specialist_class="backend", consequential=True,
            required_evidence=("domain-evidence",),
            scope=("files",), step_budget=10)
        now = time.time()
        BrowserRunRecordStore(self.db).record({
            "run_id": "run-1", "url": "https://example.com/a",
            "goal": "check the thing", "status": "completed",
            "steps": 2, "stopped_reason": "",
            "final_url": "https://example.com/a",
            "wall_outcome": "no wall", "error": "",
            "cancel_state": "", "cancel_requested_at": 0.0,
            "created_at": now - 60, "started_at": now - 60,
            "finished_at": now - 30, "updated_at": now - 30})
        from nomorals.storage.artifacts import ArtifactStore
        from nomorals.storage.blob import BlobStore
        tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmp,
                                                ignore_errors=True))
        arts = ArtifactStore(self.db, BlobStore(self.db,
                                                Path(tmp) / "blobs"))
        self.first = arts.put_text("first", type="text",
                                   creator="tester")
        self.second = arts.derive(b"second", from_ids=[self.first.id],
                                  type="text", creator="tester")
        self.library = FileSkillLibrary(
            user_dir=Path("/nonexistent-user-root-for-tests"))

    def _graph(self):
        from nomorals.craft.graph import project_graph
        return project_graph(self.db, library=self.library)

    def test_contract_is_a_node_with_role_edges(self):
        graph = self._graph()
        node = graph.node(f"contract:{self.contract.id}")
        self.assertIsNotNone(node)
        self.assertEqual(node.kind, "contract")
        self.assertEqual(node.detail["status"], "open")
        edges = {e.kind: e.dst for e in graph.edges_from(node.key)}
        self.assertEqual(edges["worker"], "actor:worker-a")
        self.assertEqual(edges["verifier"], "actor:verifier-b")
        self.assertEqual(edges["witness"], "actor:witness-c")
        self.assertEqual(edges["governed_by"], "role:backend")

    def test_contract_projects_to_a_ticket_node(self):
        graph = self._graph()
        ticket = graph.node(f"ticket:contract:{self.contract.id}")
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.detail["kind"], "contract")
        self.assertEqual(ticket.label, "verify the thing")
        self.assertIn("acceptance", ticket.detail["absent_fields"])

    def test_browser_run_record_is_a_ticket_node(self):
        graph = self._graph()
        ticket = graph.node("ticket:browser:run-1")
        self.assertIsNotNone(ticket)
        self.assertEqual(ticket.detail["kind"], "browser_run")
        self.assertEqual(ticket.detail["status"], "completed")

    def test_artifact_nodes_carry_provenance_edges(self):
        graph = self._graph()
        node = graph.node(f"artifact:{self.second.id}")
        self.assertIsNotNone(node)
        self.assertEqual(node.detail["type"], "text")
        derived = [e.dst for e in graph.edges_from(node.key)
                   if e.kind == "derived_from"]
        self.assertEqual(derived, [f"artifact:{self.first.id}"])

    def test_pack_is_a_node_bound_to_its_domain(self):
        graph = self._graph()
        pack = graph.node("pack:software-craft-pack")
        self.assertIsNotNone(pack)
        self.assertEqual(pack.detail["domain"], "build")
        edges = [(e.kind, e.dst) for e in graph.edges_from(pack.key)]
        self.assertIn(("about_domain", "domain:build"), edges)

    def test_nodes_resolve_to_their_source_records(self):
        from nomorals.agents.delegation import DelegationStore
        from nomorals.agents.run_ticket import ticket_for

        graph = self._graph()
        contract = DelegationStore(self.db).get_contract(
            self.contract.id)
        self.assertIsNotNone(contract)
        self.assertEqual(
            graph.node(f"contract:{contract.id}").detail["status"],
            contract.status)
        ticket = ticket_for(f"contract:{self.contract.id}", self.db)
        self.assertTrue(ticket["found"])
        self.assertEqual(
            graph.node(f"ticket:contract:{self.contract.id}").label,
            ticket["label"])
        pack = self.library.load("software-craft-pack")
        self.assertIsNotNone(pack)
        self.assertIsNotNone(graph.node("pack:software-craft-pack"))

    def test_static_sources_present_without_a_db(self):
        from nomorals.craft.graph import project_graph
        graph = project_graph()
        self.assertGreaterEqual(len(graph.by_kind("domain")), 16)
        self.assertIsNotNone(graph.node("role:backend"))
        self.assertIsNotNone(graph.node("domain:build"))
        seed_domains = [n for n in graph.by_kind("domain")
                        if n.detail.get("seed")]
        self.assertEqual(len(seed_domains), len(SEED_DOMAINS))
        self.assertGreater(len(graph.by_kind("occupation")), 0)

    def test_empty_stores_project_an_empty_record_layer(self):
        from nomorals.craft.graph import project_graph
        db = _tmp_db(self)
        graph = project_graph(db)
        self.assertEqual(graph.by_kind("contract"), [])
        self.assertEqual(graph.by_kind("ticket"), [])
        self.assertEqual(graph.by_kind("artifact"), [])


class RoleContractTests(unittest.TestCase):
    """P2: contracts over the shipped role machinery."""

    def test_every_manifest_class_has_a_valid_contract(self):
        for m in MANIFESTS:
            contract = rc.contract_for_class(m.class_id)
            self.assertIsNotNone(contract, m.class_id)
            self.assertEqual(contract.validate(), [], m.class_id)
            contract.require_valid()

    def test_manifest_contract_mirrors_the_manifest(self):
        m = manifest_for("backend")
        contract = rc.contract_for_class("backend")
        self.assertEqual(contract.authority, m.authority)
        self.assertEqual(contract.output_schema, m.output_schema)
        self.assertEqual(contract.quality_rubric, m.quality_rubric)
        self.assertEqual(contract.escalation, m.escalation)
        self.assertEqual(contract.domain, "software")
        self.assertIn("workspace_write", rc.AUTHORITIES)
        for tool in ("fs_read", "edit_file", "run_tests"):
            self.assertIn(tool, contract.tools)
        self.assertGreater(contract.budget.wall_seconds, 0)
        self.assertGreater(contract.budget.tokens, 0)

    def test_verifier_independence_is_refused(self):
        contract = rc.RoleContract(
            role_id="solo", domain="software",
            competencies=("python",), authority="operate",
            inputs=("brief",), tools=("file_read",),
            output_schema="report", quality_rubric="evidence",
            verifier="solo", escalation="owner")
        problems = contract.validate()
        self.assertTrue(any("itself" in p for p in problems))
        with self.assertRaises(rc.RoleContractError):
            contract.require_valid()

    def test_read_only_authority_refuses_mutating_tools(self):
        contract = rc.RoleContract(
            role_id="reader", domain="software",
            competencies=("python",), authority="read_only",
            inputs=("brief",), tools=("file_read", "run_shell"),
            output_schema="report", quality_rubric="evidence",
            verifier="critic", escalation="owner")
        problems = contract.validate()
        self.assertTrue(any("run_shell" in p for p in problems))

    def test_unknown_authority_is_refused(self):
        contract = rc.RoleContract(
            role_id="x", domain="software", competencies=("a",),
            authority="unbounded", inputs=("brief",),
            tools=("file_read",), output_schema="report",
            quality_rubric="evidence", verifier="critic",
            escalation="owner")
        self.assertTrue(any("authority" in p
                            for p in contract.validate()))

    def test_builtin_spec_contract_names_its_competency_gap(self):
        contract = rc.contract_for_role_spec(BUILTIN_SPECS["coder"])
        problems = contract.validate()
        self.assertEqual(len(problems), 1)
        self.assertIn("competencies", problems[0])
        self.assertEqual(contract.authority, "workspace_write")

    def test_contracts_index_covers_every_class(self):
        rows = rc.contracts_index()
        self.assertEqual(len(rows), len(MANIFESTS))
        for role_id, row in rows.items():
            self.assertEqual(row["role_id"], role_id)
            self.assertIn("authority", row)
            self.assertIn("verifier", row)


class PersonaRenderingTests(unittest.TestCase):
    """P2 seam: specialist prompts are renderings of contracts."""

    def test_builtin_prompts_are_their_contract_renderings(self):
        for spec in BUILTIN_SPECS.values():
            self.assertEqual(spec.system_prompt,
                             render_role_prompt(spec), spec.name)

    def test_changing_a_term_changes_the_render(self):
        import dataclasses
        spec = BUILTIN_SPECS["coder"]
        changed = dataclasses.replace(
            spec, output_contract=("a different contract",))
        self.assertNotEqual(render_role_prompt(changed),
                            spec.system_prompt)
        self.assertIn("a different contract",
                      render_role_prompt(changed))

    def test_legacy_explicit_prompt_is_kept_verbatim(self):
        spec = RoleSpec(name="raw", system_prompt="raw prompt text")
        self.assertEqual(spec.system_prompt, "raw prompt text")


PACK_BODY = """# Fixture Pack

Intro line.

## Terminology

Words mean one thing here.

## Workflow

Do the steps in order.

## Tools

The shipped tools.

## Rubric

What good looks like.

## Anti-patterns

What to never do.

## Exemplars

Past good work.

## Evaluators

Suites that check it.

## Failure Modes

Known ways it breaks.
"""


def _pack_doc(body: str = PACK_BODY, fixtures=("tests/x.py",)):
    from nomorals.skills.filebook import SkillDoc
    return SkillDoc(
        id="fixture-pack", title="Fixture Pack", summary="d",
        version=1, origin="seed", layer="craft", body=body,
        kind="pack", domain="build", fixtures=fixtures)


class PackSchemaTests(unittest.TestCase):
    """P3: the pack is a skill-form artifact, schema-validated."""

    def test_seed_pack_loads_and_validates(self):
        library = FileSkillLibrary(
            user_dir=Path("/nonexistent-user-root-for-tests"))
        doc = library.load("software-craft-pack")
        self.assertIsNotNone(doc)
        self.assertTrue(is_pack(doc))
        self.assertEqual(validate_pack(doc), [])
        view = load_pack(library, "software-craft-pack")
        self.assertIsNotNone(view)
        summary = pack_summary(view)
        self.assertEqual(summary["problems"], [])
        self.assertIn("rubric", summary["sections"])
        self.assertTrue(summary["fixtures"])

    def test_seed_pack_fixtures_exist_on_disk(self):
        library = FileSkillLibrary(
            user_dir=Path("/nonexistent-user-root-for-tests"))
        view = load_pack(library, "software-craft-pack")
        self.assertIsNotNone(view)
        root = Path(__file__).resolve().parent.parent
        for fixture in view.fixtures:
            self.assertTrue((root / fixture).exists(), fixture)

    def test_router_carries_the_pack(self):
        library = FileSkillLibrary(
            user_dir=Path("/nonexistent-user-root-for-tests"))
        routed = library.route(
            "software craft delegation contract tests")
        self.assertIn("software-craft-pack",
                      [doc.id for doc, _score in routed])

    def test_valid_fixture_pack_validates(self):
        self.assertEqual(validate_pack(_pack_doc()), [])

    def test_malformed_pack_fails_with_named_errors(self):
        body = PACK_BODY.replace("## Rubric\n\nWhat good looks "
                                 "like.\n", "")
        problems = validate_pack(_pack_doc(body=body, fixtures=()))
        self.assertTrue(any("rubric" in p for p in problems))
        self.assertTrue(any("fixtures" in p for p in problems))

    def test_a_plain_skill_is_not_a_pack(self):
        from nomorals.skills.filebook import SkillDoc
        doc = SkillDoc(id="plain", title="Plain", summary="d",
                       version=1, origin="seed", layer="", body="x")
        self.assertFalse(is_pack(doc))
        self.assertTrue(validate_pack(doc))

    def test_promote_keeps_the_pack_fields(self):
        tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR"))
        self.addCleanup(
            lambda: __import__("shutil").rmtree(tmp,
                                                ignore_errors=True))
        user_dir = Path(tmp) / "skills"
        (user_dir / "drafts").mkdir(parents=True)
        from nomorals.skills.filebook import render_skill_text
        draft = _pack_doc()
        draft = draft.__class__(
            **{**draft.__dict__, "origin": "draft"})
        (user_dir / "drafts" / "fixture-pack.md").write_text(
            render_skill_text(draft), encoding="utf-8")
        library = FileSkillLibrary(user_dir=user_dir)
        library.promote("fixture-pack")
        promoted = library.load("fixture-pack")
        self.assertIsNotNone(promoted)
        self.assertEqual(promoted.kind, "pack")
        self.assertEqual(promoted.domain, "build")
        self.assertEqual(promoted.fixtures, ("tests/x.py",))
        self.assertEqual(validate_pack(promoted), [])


class ReceiptSchemaTests(unittest.TestCase):
    """P4: receipts claim only recorded outcomes; grades derive."""

    def setUp(self) -> None:
        from nomorals.agents.content_cell import ReceiptStore
        from nomorals.agents.outcome_cases import OutcomeCaseStore

        self.db = _tmp_db(self)
        store = ReceiptStore(self.db)
        self.draft_row = store.record(
            contract_id="dc-1", kind="post", label="Unposted draft",
            platform="x", draft_id="d-2", content="wip")
        row = store.record(
            contract_id="dc-1", kind="post",
            label="Spring launch post", platform="x", draft_id="d-1",
            content="hello world")
        self.posted_row = store.attach_post(
            row["id"], post_ref="https://x.example/post/1")
        cases = OutcomeCaseStore(self.db)
        self.case, _ = cases.open_case(
            domain="content", kind="delivery",
            subject="Spring launch post", subject_key="post-1",
            reason="overdue")
        self.case_receipt = cases.add_receipt(
            self.case.id, kind="investigation", source="steward",
            actor="steward", note="found the post live",
            evidence={"post_ref": "https://x.example/post/1"})

    def test_content_receipt_grades_by_its_record(self):
        posted = adapt_content_receipt(self.posted_row)
        self.assertEqual(posted.outcome, "delivered")
        self.assertEqual(posted.grade.name, "G4")
        draft = adapt_content_receipt(self.draft_row)
        self.assertEqual(draft.outcome, "drafted")
        self.assertEqual(draft.grade.name, "G1")

    def test_case_receipt_grades_its_evidence_not_its_verdict(self):
        receipt = adapt_case_receipt(self.case_receipt, self.case)
        self.assertEqual(receipt.grade.name, "G4")
        self.assertEqual(receipt.subject, "Spring launch post")

    def test_police_receipt_grades_by_recorded_effect(self):
        from nomorals.immune.police import Receipt
        performed = adapt_police_receipt(Receipt(
            action="disarm", target="runaway job",
            details={"effect": "performed"}))
        self.assertEqual(performed.grade.name, "G5")
        partial = adapt_police_receipt(Receipt(
            action="slow", target="job",
            details={"effect": "partial"}))
        self.assertEqual(partial.grade.name, "G4")
        report = adapt_police_receipt(Receipt(
            action="report_only", target="job"))
        self.assertEqual(report.grade.name, "G1")

    def test_delivery_receipt_grades_by_the_send(self):
        from nomorals.media.mix_surface import DeliveryReceipt
        sent = adapt_delivery_receipt(DeliveryReceipt(
            chat="c", mix_id="m1", fingerprint="fp1", kind="audio",
            sent_at=time.time()))
        self.assertEqual((sent.outcome, sent.grade.name),
                         ("sent", "G4"))
        unsent = adapt_delivery_receipt(DeliveryReceipt(
            chat="c", mix_id="m2", fingerprint="fp2", kind="audio"))
        self.assertEqual((unsent.outcome, unsent.grade.name),
                         ("unsent", "G1"))

    def test_bill_receipt_claims_only_the_recording(self):
        from nomorals.community.miniapps2 import Receipt
        receipt = adapt_bill_receipt(Receipt(
            id="b1", group_key="g", payer_id="p", payer_name="Peace",
            total_cents=1000, description="dinner",
            shares={"a": 1.0}, created_ts=time.time()))
        self.assertEqual((receipt.outcome, receipt.grade.name),
                         ("recorded", "G1"))
        self.assertEqual(receipt.detail["total_cents"], 1000)

    def test_chain_renders_one_shape_without_source_ids(self):
        content = adapt_content_receipt(self.posted_row)
        case = adapt_case_receipt(self.case_receipt, self.case)
        case.handoff = handoff_for(case, to_actor="owner",
                                   note="post verified live")
        text = render_receipt_chain([case, content])
        self.assertIn("content cell", text)
        self.assertIn("outcome case", text)
        self.assertIn("grade G4", text)
        self.assertIn("handed to owner", text)
        self.assertNotIn(self.posted_row["id"], text)
        self.assertNotIn(self.case_receipt.id, text)

    def test_stored_grade_above_evidence_is_refused(self):
        receipt = adapt_content_receipt(self.posted_row)
        payload = receipt.to_dict()
        payload["grade"] = "G6"
        with self.assertRaises(ProofGradeError):
            QualityReceipt.from_dict(payload)

    def test_asserting_above_evidence_is_refused(self):
        receipt = adapt_content_receipt(self.posted_row)
        with self.assertRaises(ProofGradeError):
            QualityReceipt.asserting(
                "G6", source="content_cell", source_id="x",
                evidence=receipt.evidence[:1])
        ok = QualityReceipt.asserting(
            "G1", source="content_cell", source_id="x",
            evidence=receipt.evidence[:1])
        self.assertEqual(ok.grade.name, "G1")

    def test_roundtrip_keeps_the_derived_grade(self):
        receipt = adapt_case_receipt(self.case_receipt, self.case)
        clone = QualityReceipt.from_dict(receipt.to_dict())
        self.assertEqual(clone.grade, receipt.grade)
        self.assertEqual(clone.source_id, receipt.source_id)

    def test_handoff_requires_recorded_work_and_a_receiver(self):
        receipt = adapt_content_receipt(self.posted_row)
        with self.assertRaises(ProofGradeError):
            handoff_for(receipt, to_actor="")
        with self.assertRaises(ProofGradeError):
            HandoffReceipt(from_actor="a", to_actor="b", subject="s",
                           basis="")
        handoff = handoff_for(receipt, to_actor="owner")
        self.assertEqual(handoff.basis,
                         f"content_cell:{receipt.source_id}")


class LevelMappingTests(unittest.TestCase):
    """L-levels map onto the one proof ladder or say why not."""

    def test_the_mapping_table(self):
        self.assertEqual(
            [row.source_value for row in LEVEL_GRADE_MAPPINGS],
            ["L0", "L1", "L2", "L3", "L4", "L5"])
        for row in LEVEL_GRADE_MAPPINGS:
            self.assertTrue(row.reason.strip())
        self.assertEqual(grade_for_level("L0").name, "G1")
        self.assertEqual(grade_for_level("L1").name, "G2")
        self.assertEqual(grade_for_level("L2").name, "G4")
        self.assertEqual(grade_for_level("L3").name, "G5")

    def test_handoff_classes_carry_no_grade(self):
        self.assertIsNone(grade_for_level("L4"))
        self.assertIsNone(grade_for_level("L5"))
        self.assertTrue(is_handoff_level("L4"))
        self.assertTrue(is_handoff_level("L5"))
        self.assertFalse(is_handoff_level("L2"))

    def test_cap_claim_takes_the_lower_of_level_and_evidence(self):
        strong = (EvidenceRef("artifact_pass", "x"),)
        weak = (EvidenceRef("logical_check", "x"),)
        self.assertEqual(cap_claim("L0", strong).name, "G1")
        self.assertEqual(cap_claim("L3", weak).name, "G1")
        self.assertEqual(cap_claim("L3", strong).name, "G5")

    def test_handoff_level_claims_grade_on_their_evidence(self):
        strong = (EvidenceRef("artifact_pass", "x"),)
        self.assertEqual(cap_claim("L4", strong).name, "G5")

    def test_unknown_level_is_refused_not_guessed(self):
        self.assertEqual(parse_level("l2"), "L2")
        with self.assertRaises(LevelError):
            parse_level("L9")
        with self.assertRaises(LevelError):
            grade_for_level("senior")


if __name__ == "__main__":
    unittest.main()
