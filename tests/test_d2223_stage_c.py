"""Docs 22+23 Stage C — the machinery, pinned.

P8 (composer): a goal composes a cell from a named preset with no
hand-wiring; its consequential steps open REAL contracts through
the shipped G9 seam (agents/plan_contracts, unmodified) before any
handler runs; tiny read-only work skips the ceremony under the
same rule; a step the seam refuses stays refused in the seam's
own words.

P9 (temporary competence): an unknown field goes research →
draft pack → benchmark → claim; a failing benchmark leaves the
work on record and the claim unmade; a grant is bounded (scope,
expiry, benchmark evidence), recorded, lapses on time, and never
outlives its evidence — planted standing/level tokens in the
grant store are never read.

P10 (levels from records + coverage/intake): levels derive from
records alone — a role with strong records levels above one with
none, a planted fiat level is never read, a level changes only
when its underlying record changes; coverage names an uncovered
domain absent-with-reason; the intake renders decomposition,
team, gate, risks and cost from records.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents import plan_contracts
from nomorals.agents.content_cell import ReceiptStore
from nomorals.agents.delegation import DelegationStore, IndependenceError
from nomorals.craft.composer import (
    ComposerError,
    compose_plan,
    get_preset,
    open_plan,
    preset_names,
    settle_plan,
)
from nomorals.craft.coverage import (
    coverage_report,
    render_coverage,
    render_intake,
)
from nomorals.craft.levels import LevelFacts, derive_level
from nomorals.craft.pilots import run_cbt_pilot, run_software_pilot
from nomorals.craft.temporary import (
    CompetenceGrantStore,
    TemporaryCompetenceError,
    acquire_competence,
    capability_status,
)
from nomorals.skills.filebook import FileSkillLibrary, SkillDoc
from nomorals.skills.packs import load_pack
from nomorals.storage.artifacts import ArtifactStore
from nomorals.storage.blob import BlobStore
from nomorals.storage.db import Database


def _tmp(testcase: unittest.TestCase) -> Path:
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")))
    testcase.addCleanup(
        lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
    return tmp


def _tmp_db(testcase: unittest.TestCase) -> Database:
    tmp = _tmp(testcase)
    db = Database(str(tmp / "test.db"))
    db.migrate()
    return db


def _seed_library() -> FileSkillLibrary:
    return FileSkillLibrary(
        user_dir=Path("/nonexistent-user-root-for-tests"))


def _empty_library(tmp: Path) -> FileSkillLibrary:
    (tmp / "seed").mkdir(parents=True, exist_ok=True)
    return FileSkillLibrary(seed_dir=tmp / "seed",
                            user_dir=tmp / "user")


_SECTIONS = {
    "terminology": "Hive, apiary, frame, smoker.",
    "workflow": "Inspect, record, treat, review.",
    "tools": "Smoker, hive tool, frames.",
    "rubric": "A claim is proven only by its benchmark.",
    "anti_patterns": "Never claim a field from prose alone.",
    "exemplars": "The benchmark fixture in this pack.",
    "evaluators": "The fixtures declared on this pack.",
    "failure_modes": "Benchmark theatre: fixtures that cannot "
                     "fail prove nothing.",
}


def _research(fixture_content: str, *, sections: dict | None = None):
    def research(field: str) -> dict:
        return {
            "summary": f"research findings for {field}",
            "sections": dict(sections or _SECTIONS),
            "fixtures": [{"filename": "test_benchmark.py",
                          "content": fixture_content}],
        }
    return research


_PASSING_FIXTURE = "def test_field_basics():\n    assert 1 + 1 == 2\n"
_FAILING_FIXTURE = "def test_field_basics():\n    assert 1 + 1 == 3\n"


# ── P8: the composer ─────────────────────────────────────────────


class ComposerTests(unittest.TestCase):

    def test_presets_are_a_named_catalogue(self):
        self.assertEqual(
            preset_names(),
            ("solo", "pipeline", "studio", "research", "incident"))
        for name in preset_names():
            self.assertTrue(get_preset(name).steps)
        with self.assertRaises(ComposerError) as caught:
            get_preset("swarm")
        self.assertIn("swarm", str(caught.exception))
        self.assertIn("pipeline", str(caught.exception))

    def test_pipeline_composes_over_the_software_pack(self):
        pack = load_pack(_seed_library(), "software-craft-pack")
        plan = compose_plan("ship the weekly digest",
                            preset="pipeline", pack=pack,
                            writes=("post:digest",))
        self.assertEqual(plan.domain, "build")
        self.assertEqual(plan.pack_id, "software-craft-pack")
        self.assertFalse(plan.pack_draft)
        self.assertEqual([s["name"] for s in plan.steps],
                         ["research", "build", "deliver"])
        verdicts = {name: (fires, reasons) for name, fires, reasons
                    in plan.consequentiality()}
        self.assertFalse(verdicts["research"][0])
        self.assertTrue(verdicts["build"][0])
        self.assertTrue(any("writes" in r
                            for r in verdicts["build"][1]))
        self.assertFalse(verdicts["deliver"][0])
        # the steps are the seam's duck-typed shape
        build = plan.step("build")
        self.assertEqual(build["role"], "execution")
        self.assertEqual(build["writes"], ["post:digest"])
        self.assertEqual(build["payload"]["pack"],
                         "software-craft-pack")

    def test_open_plan_contract_precedes_any_settle(self):
        db = _tmp_db(self)
        store = DelegationStore(db)
        pack = load_pack(_seed_library(), "software-craft-pack")
        plan = compose_plan("ship the weekly digest",
                            preset="pipeline", pack=pack,
                            writes=("post:digest",))
        opening = open_plan(store, plan, mission_id="run-c1")
        self.assertFalse(opening.refused)
        build = opening.opening_for("build")
        self.assertTrue(build.contract_id)
        # the contract is OPEN in the ledger before any handler ran
        contract = store.get_contract(build.contract_id)
        self.assertEqual(contract.status, "open")
        self.assertTrue(contract.consequential)
        self.assertEqual(contract.verifier, plan_contracts.VERIFIER)
        self.assertNotEqual(contract.worker, contract.verifier)
        # skipped steps earn no contract and say why
        for name in ("research", "deliver"):
            skipped = opening.opening_for(name)
            self.assertEqual(skipped.contract_id, "")
            self.assertIn("below the consequential rule",
                          skipped.skipped)
        settlement = settle_plan(store, plan, opening,
                                 {"build": {"posted": "yes"}},
                                 mission_id="run-c1")
        self.assertTrue(settlement.completed)
        done = store.get_contract(build.contract_id)
        self.assertEqual(done.status, "completed")
        stamps = {s.name: s.outcome for s in settlement.steps}
        self.assertIsNone(stamps["research"])
        self.assertTrue(stamps["build"]["completed"])

    def test_settle_defect_refuses_in_the_seams_own_words(self):
        db = _tmp_db(self)
        store = DelegationStore(db)
        plan = compose_plan("write the file", preset="solo",
                            writes=("out.txt",))
        opening = open_plan(store, plan, mission_id="run-c2")
        contract_id = opening.opening_for("work").contract_id
        settlement = settle_plan(store, plan, opening, {"work": {}},
                                 mission_id="run-c2")
        outcome = settlement.steps[0].outcome
        self.assertFalse(outcome["completed"])
        # the refusal surfaces in the seam's own words: the
        # completion gate's blockers, carried verbatim
        self.assertTrue(
            any("contradicted" in r for r in outcome["reasons"]),
            outcome)
        # and the concrete defect itself is on the ledger, in the
        # seam's defect wording (the challenge resolution's note)
        ledger = " ".join(
            claim.claim for claim in store.claims(contract_id))
        self.assertIn("recorded result is empty", ledger)
        self.assertNotEqual(
            store.get_contract(contract_id).status, "completed")

    def test_seam_refusal_is_verbatim_and_never_routed_around(self):
        db = _tmp_db(self)
        store = DelegationStore(db)
        plan = compose_plan("ship the weekly digest",
                            preset="pipeline",
                            writes=("post:digest",))
        sentinel = ("verifier 'plan:verifier' is the worker — "
                    "verification must be independent SENTINEL")

        def refuse(store_, *, mission_id, step):
            raise IndependenceError(sentinel)

        with mock.patch.object(plan_contracts, "open_step_contract",
                               side_effect=refuse):
            opening = open_plan(store, plan, mission_id="run-c3")
        self.assertTrue(opening.refused)
        build = opening.opening_for("build")
        self.assertEqual(build.refusal, sentinel)
        self.assertEqual(build.contract_id, "")
        # no contract was substituted, and no later step opened
        self.assertEqual(store.recent_contracts(limit=10), [])
        deliver = opening.opening_for("deliver")
        self.assertIn("not opened", deliver.skipped)

    def test_tiny_readonly_work_skips_the_ceremony(self):
        db = _tmp_db(self)
        store = DelegationStore(db)
        plan = compose_plan("look the figure up", preset="solo")
        opening = open_plan(store, plan, mission_id="run-c4")
        self.assertEqual(opening.contract_ids(), ())
        self.assertEqual(store.recent_contracts(limit=10), [])

    def test_invalid_pack_refuses_in_the_schema_words(self):
        tmp = _tmp(self)
        bad = tmp / "seed"
        bad.mkdir()
        (bad / "thin-pack.md").write_text(
            "---\nid: thin-pack\ntitle: Thin pack\nkind: pack\n"
            "domain: build\n---\n\n# Thin\n\n## Terminology\n\n"
            "Only one section.\n", encoding="utf-8")
        library = FileSkillLibrary(
            seed_dir=bad,
            user_dir=Path("/nonexistent-user-root-for-tests"))
        pack = load_pack(library, "thin-pack")
        with self.assertRaises(ComposerError) as caught:
            compose_plan("anything", pack=pack)
        self.assertIn("does not validate", str(caught.exception))
        self.assertIn("fixtures", str(caught.exception))

    def test_draft_pack_composes_labelled_unproven(self):
        tmp = _tmp(self)
        library = _empty_library(tmp)
        doc = SkillDoc(
            id="field-notes-pack", title="Field notes pack",
            summary="a drafted pack", topics=("field notes",),
            layer="craft", kind="pack", domain="learning",
            fixtures=("tests/test_d2223_stage_c.py",),
            body="\n\n".join(
                f"## {heading}\n\n{text}" for heading, text in [
                    ("Terminology", "term"),
                    ("Workflow", "flow"),
                    ("Tools", "tools"),
                    ("Rubric", "rubric"),
                    ("Anti-patterns", "anti"),
                    ("Exemplars", "exemplar"),
                    ("Evaluators", "evaluators"),
                    ("Failure modes", "failure")]))
        library.write_draft(doc)
        pack = load_pack(library, "field-notes-pack")
        self.assertTrue(pack.doc.is_draft)
        plan = compose_plan("study the field", preset="research",
                            pack=pack)
        self.assertTrue(plan.pack_draft)
        self.assertEqual(
            plan.steps[0]["payload"]["pack_status"],
            "draft (unproven)")

    def test_custom_steps_compose_and_contract(self):
        db = _tmp_db(self)
        store = DelegationStore(db)
        plan = compose_plan(
            "ad hoc work",
            steps=[{"name": "probe", "role": "research"},
                   {"name": "patch", "role": "execution",
                    "writes": ["a.py"]}])
        self.assertEqual(plan.preset, "custom")
        opening = open_plan(store, plan, mission_id="run-c5")
        self.assertEqual(opening.opening_for("probe").contract_id,
                         "")
        self.assertTrue(opening.opening_for("patch").contract_id)


# ── P9: temporary competence ─────────────────────────────────────


class TemporaryCompetenceTests(unittest.TestCase):

    def _setup(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = _empty_library(tmp)
        grants = CompetenceGrantStore(tmp / "grants.json")
        return tmp, db, library, grants

    def test_unknown_field_runs_research_to_grant(self):
        tmp, db, library, grants = self._setup()
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research(_PASSING_FIXTURE),
            blob_dir=tmp / "blobs", grants=grants,
            ttl_seconds=600, now=1000.0)
        self.assertEqual(run.stage, "granted", run.detail)
        self.assertTrue(run.claim_made)
        self.assertEqual(run.pack_id, "temp-beekeeping-pack")
        # the grant is recorded, scoped and expiring
        grant = run.grant
        self.assertIsNotNone(grant)
        self.assertEqual(grant.scope, "field:beekeeping")
        self.assertEqual(grant.granted_at, 1000.0)
        self.assertEqual(grant.expires_at, 1600.0)
        self.assertEqual(grants.latest_for_pack(run.pack_id).id,
                         grant.id)
        # the research contract completed under independence
        store = DelegationStore(db)
        contract = store.get_contract(run.contract_id)
        self.assertEqual(contract.status, "completed")
        self.assertNotEqual(contract.worker, contract.verifier)
        # the pack stands as a draft — labelled unproven
        pack = load_pack(library, run.pack_id)
        self.assertTrue(pack.doc.is_draft)
        # capability answers read the grant: claimed, temporary,
        # level L0, grade capped by the level at G1
        status = capability_status(library, "beekeeping",
                                   grants=grants, now=1100.0)
        self.assertTrue(status["claimed"])
        self.assertTrue(status["temporary"])
        self.assertEqual(status["level"], "L0")
        self.assertEqual(status["grade"], "G1")

    def test_benchmark_record_is_the_grants_evidence(self):
        tmp, db, library, grants = self._setup()
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research(_PASSING_FIXTURE),
            blob_dir=tmp / "blobs", grants=grants, now=1000.0)
        artifacts = ArtifactStore(
            db, BlobStore(db, tmp / "blobs"))
        recorded = json.loads(
            artifacts.read_text(run.grant.benchmark_ref))
        self.assertGreaterEqual(recorded["tests_passed"], 1)
        self.assertEqual(recorded["tests_failed"], 0)
        self.assertEqual(run.grant.benchmark["tests_failed"], 0)

    def test_failing_benchmark_leaves_work_and_unmakes_claim(self):
        tmp, db, library, grants = self._setup()
        run = acquire_competence(
            db, library, field="astrology",
            research_fn=_research(_FAILING_FIXTURE),
            blob_dir=tmp / "blobs", grants=grants, now=1000.0)
        self.assertEqual(run.stage, "benchmark_failed")
        self.assertFalse(run.claim_made)
        self.assertIsNone(run.grant)
        self.assertEqual(grants.grants(), [])
        self.assertEqual(run.benchmark["files_failed"], 1)
        # the work is all on record: contract (failed), draft pack
        store = DelegationStore(db)
        self.assertEqual(
            store.get_contract(run.contract_id).status, "failed")
        pack = load_pack(library, "temp-astrology-pack")
        self.assertIsNotNone(pack)
        self.assertTrue(pack.doc.is_draft)
        # and the claim reads unmade, with the reason named
        status = capability_status(library, "astrology",
                                   grants=grants, now=1000.0)
        self.assertFalse(status["claimed"])
        self.assertIn("unproven draft", status["reason"])

    def test_invalid_pack_names_problems_and_unmakes_claim(self):
        tmp, db, library, grants = self._setup()
        sections = dict(_SECTIONS)
        del sections["rubric"]
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research(_PASSING_FIXTURE,
                                  sections=sections),
            blob_dir=tmp / "blobs", grants=grants, now=1000.0)
        self.assertEqual(run.stage, "invalid_pack")
        self.assertFalse(run.claim_made)
        self.assertTrue(any("rubric" in p
                            for p in run.validation_problems),
                        run.validation_problems)
        self.assertIsNotNone(
            load_pack(library, "temp-beekeeping-pack"))

    def test_grant_lapses_on_time(self):
        tmp, db, library, grants = self._setup()
        acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research(_PASSING_FIXTURE),
            blob_dir=tmp / "blobs", grants=grants,
            ttl_seconds=600, now=1000.0)
        inside = capability_status(library, "beekeeping",
                                   grants=grants, now=1599.0)
        self.assertTrue(inside["claimed"])
        lapsed = capability_status(library, "beekeeping",
                                   grants=grants, now=1600.0)
        self.assertFalse(lapsed["claimed"])
        self.assertIsNone(lapsed["level"])
        self.assertIn("lapsed", lapsed["reason"])
        # the draft and the grant record remain — only the claim
        # lapsed
        self.assertIsNotNone(grants.latest_for_pack(
            "temp-beekeeping-pack"))
        self.assertIsNotNone(
            load_pack(library, "temp-beekeeping-pack"))

    def test_planted_fiat_tokens_in_the_store_are_never_read(self):
        tmp, db, library, grants = self._setup()
        run = acquire_competence(
            db, library, field="beekeeping",
            research_fn=_research(_PASSING_FIXTURE),
            blob_dir=tmp / "blobs", grants=grants,
            ttl_seconds=600, now=1000.0)
        path = tmp / "grants.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["grants"][0]["status"] = "active"
        payload["grants"][0]["level"] = "L3"
        payload["grants"][0]["grade"] = "G6"
        path.write_text(json.dumps(payload), encoding="utf-8")
        # inside the window the claim derives L0/G1 — not the
        # planted L3/G6; past expiry the planted "active" does not
        # keep the claim alive
        inside = capability_status(library, "beekeeping",
                                   grants=grants, now=1100.0)
        self.assertEqual(inside["level"], "L0")
        self.assertEqual(inside["grade"], "G1")
        lapsed = capability_status(library, "beekeeping",
                                   grants=grants, now=1700.0)
        self.assertFalse(lapsed["claimed"])
        self.assertEqual(run.grant.pack_id, "temp-beekeeping-pack")

    def test_covered_field_spawns_no_research(self):
        called = []

        def research(field: str) -> dict:
            called.append(field)
            return _research(_PASSING_FIXTURE)(field)

        tmp = _tmp(self)
        db = _tmp_db(self)
        grants = CompetenceGrantStore(tmp / "grants.json")
        run = acquire_competence(
            db, _seed_library(), field="build", research_fn=research,
            blob_dir=tmp / "blobs", grants=grants, now=1000.0)
        self.assertEqual(run.stage, "already_covered")
        self.assertEqual(called, [])
        self.assertTrue(run.claim_made)
        self.assertEqual(run.status["pack_status"], "live")

    def test_misshaped_research_product_is_named_not_minted(self):
        tmp, db, library, grants = self._setup()
        with self.assertRaises(TemporaryCompetenceError):
            acquire_competence(
                db, library, field="beekeeping",
                research_fn=lambda field: {"nope": 1},
                blob_dir=tmp / "blobs", grants=grants, now=1000.0)
        store = DelegationStore(db)
        recent = store.recent_contracts(limit=1)
        self.assertEqual(recent[0].status, "failed")
        self.assertIsNone(load_pack(library, "temp-beekeeping-pack"))


# ── P10: levels derived from records ─────────────────────────────


class DerivedLevelTests(unittest.TestCase):

    def test_the_ladder_from_facts(self):
        self.assertIsNone(derive_level(LevelFacts()).level)
        self.assertEqual(
            derive_level(LevelFacts(live_knowledge=True)).level,
            "L0")
        self.assertEqual(
            derive_level(LevelFacts(live_knowledge=True,
                                    produced=True)).level, "L1")
        self.assertEqual(
            derive_level(LevelFacts(live_knowledge=True,
                                    produced=True,
                                    tool_executed=True)).level, "L2")
        full = derive_level(LevelFacts(
            live_knowledge=True, produced=True, tool_executed=True,
            operated=True))
        self.assertEqual(full.level, "L3")
        self.assertEqual(len(full.basis), 4)

    def test_a_draft_alone_earns_no_level(self):
        derived = derive_level(LevelFacts(draft_knowledge=True))
        self.assertIsNone(derived.level)
        self.assertIn("unproven draft", derived.notes[0])

    def test_handoff_levels_are_never_derived(self):
        derived = derive_level(LevelFacts(
            live_knowledge=True, produced=True, tool_executed=True,
            operated=True))
        self.assertNotIn(derived.level, ("L4", "L5"))

    def test_no_implications_are_invented(self):
        derived = derive_level(LevelFacts(operated=True))
        self.assertEqual(derived.level, "L3")
        # the operated record is its own earning fact — the
        # derivation does not fabricate the rungs beneath it
        self.assertEqual(len(derived.basis), 1)
        self.assertIn("operated", derived.basis[0])


# ── P10: coverage + intake ───────────────────────────────────────

_SOFTWARE_TASK = {
    "title": "fix the add function",
    "goal": "make add() add, proven by its acceptance test",
    "files": {"calc.py": "def add(a, b):\n    return a - b\n"},
    "change": {"calc.py": "def add(a, b):\n    return a + b\n"},
    "tests": {"test_calc.py": "from calc import add\n\n\n"
                              "def test_add():\n"
                              "    assert add(2, 3) == 5\n"},
}

_CBT_EXERCISE = {
    "kind": "behavioral_activation",
    "fields": {"activity": "A 20-minute walk after lunch",
               "scheduled_for": "Tomorrow at 13:00",
               "expected_pleasure": 6, "expected_mastery": 5},
}


class CoverageTests(unittest.TestCase):

    def _pilots(self, db, tmp):
        library = _seed_library()
        run_software_pilot(
            db, workspace=tmp / "ws", blob_dir=tmp / "blobs",
            task=dict(_SOFTWARE_TASK), library=library, fixtures=())
        run_cbt_pilot(
            db, exercise=dict(_CBT_EXERCISE),
            interventions_path=tmp / "interventions.db",
            library=library)
        return library

    def test_levels_derive_from_pilot_records(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = self._pilots(db, tmp)
        report = coverage_report(db, library=library)
        software = report.row("software")
        self.assertEqual(software.level, "L2", software)
        self.assertTrue(any("tool-executed" in b
                            for b in software.basis))
        health = report.row("health")
        self.assertEqual(health.level, "L2", health)
        build = report.row("build")
        self.assertEqual(build.level, "L0", build)
        self.assertEqual([p["id"] for p in build.packs],
                         ["software-craft-pack"])

    def test_uncovered_domain_is_absent_with_reason(self):
        db = _tmp_db(self)
        report = coverage_report(db, library=_seed_library())
        row = report.row("hospitality")
        self.assertIsNone(row.level)
        self.assertIn("pack", row.absent)
        self.assertIn("contracts", row.absent)
        self.assertTrue(any("greenfield" in n for n in row.notes),
                        row.notes)
        text = render_coverage(report)
        self.assertIn("Hospitality", text)
        self.assertIn("absent — pack", text)

    def test_level_changes_only_when_records_change(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = _seed_library()
        before = coverage_report(db, library=library)
        self.assertEqual(before.row("health").level, "L0")
        again = coverage_report(db, library=library)
        self.assertEqual(again.row("health").level, "L0")
        run_cbt_pilot(
            db, exercise=dict(_CBT_EXERCISE),
            interventions_path=tmp / "interventions.db",
            library=library)
        after = coverage_report(db, library=library)
        self.assertEqual(after.row("health").level, "L2")

    def test_planted_fiat_level_is_never_read(self):
        tmp = _tmp(self)
        library = _empty_library(tmp)
        body = "\n\n".join(
            f"## {heading}\n\n{text}" for heading, text in [
                ("Terminology", "term"), ("Workflow", "flow"),
                ("Tools", "tools"), ("Rubric", "rubric"),
                ("Anti-patterns", "anti"), ("Exemplars", "exemplar"),
                ("Evaluators", "evaluators"),
                ("Failure modes", "failure")])
        (tmp / "user").mkdir(parents=True, exist_ok=True)
        (tmp / "user" / "astro-pack.md").write_text(
            "---\nid: astro-pack\ntitle: Astro pack\nkind: pack\n"
            "domain: astro-cartography\nlevel: L5\n"
            "fixtures: tests/test_d2223_stage_c.py\n---\n\n"
            + body + "\n", encoding="utf-8")
        library.invalidate()
        report = coverage_report(None, library=library)
        row = report.row("astro-cartography")
        self.assertIsNotNone(row)
        # the records earn L0 (a live pack) — the planted L5 in
        # the frontmatter is not a field the derivation reads
        self.assertEqual(row.level, "L0")

    def test_operated_level_from_a_delivery_receipt(self):
        db = _tmp_db(self)
        store = DelegationStore(db)
        contract = store.create_contract(
            goal="deliver the digest", worker="cell-worker",
            verifier="cell-verifier", arbiter="owner",
            scope=("health",), consequential=True)
        ReceiptStore(db).record(
            contract_id=contract.id, kind="digest",
            label="weekly digest", platform="telegram",
            post_ref="tg:post/1", content="the digest")
        report = coverage_report(db, library=_seed_library())
        health = report.row("health")
        self.assertEqual(health.level, "L3", health)
        self.assertEqual(health.deliveries, 1)

    def test_render_carries_no_internal_ids(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = self._pilots(db, tmp)
        report = coverage_report(db, library=library)
        text = render_coverage(report)
        store = DelegationStore(db)
        for contract in store.recent_contracts(limit=10):
            self.assertNotIn(contract.id, text)
        self.assertIn("craft coverage", text)

    def test_coverage_without_a_store_names_the_unread(self):
        report = coverage_report(None, library=_seed_library())
        build = report.row("build")
        self.assertEqual(build.level, "L0")
        self.assertIn("not read", build.absent["contracts"])

    def test_intake_renders_the_hire_from_records(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = self._pilots(db, tmp)
        pack = load_pack(library, "software-craft-pack")
        plan = compose_plan("ship the weekly digest",
                            preset="pipeline", pack=pack,
                            writes=("post:digest",))
        report = coverage_report(db, library=library)
        text = render_intake(plan, report.row("build"), pack=pack)
        self.assertIn("intake — ship the weekly digest", text)
        self.assertIn("decomposition:", text)
        self.assertIn("research", text)
        self.assertIn("build", text)
        self.assertIn("gate (the shipped consequential rule):",
                      text)
        self.assertIn("a delegation contract opens before it "
                      "runs", text)
        self.assertIn("no contract — the step meets none", text)
        self.assertIn("team (role contracts):", text)
        self.assertIn("authority", text)
        self.assertIn("risks:", text)
        self.assertIn("cost (role-contract budgets):", text)
        self.assertIn("domain standing:", text)
        # no internal ids anywhere in the render
        store = DelegationStore(db)
        for contract in store.recent_contracts(limit=10):
            self.assertNotIn(contract.id, text)


if __name__ == "__main__":
    unittest.main()
