"""Docs 22+23 Stage B — the pilots, pinned.

Three worked pilots exercise the Stage A machinery end-to-end in
three domains: the software pilot (P5) runs the seed software
pack's engineering loop on a real workspace task; the media pilot
(P6) runs a real media job through the shipped media substrate
with rights + edit graph in the deliverable's provenance; the CBT
pilot (P7) structures exercises into intervention records on the
health shell.  Everywhere: grades derive from the run's own
records — no pilot authors one, and a failed run's receipt grades
honestly lower than a completed run's.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from nomorals.core.proof_grade import ProofGradeError
from nomorals.craft import QualityReceipt
from nomorals.craft.pilots import (
    PilotError,
    run_cbt_pilot,
    run_media_pilot,
    run_pack_fixtures,
    run_software_pilot,
)
from nomorals.craft.receipts import render_receipt_chain
from nomorals.health.interventions import (
    InterventionStore,
    diagnostic_scan,
    render_intervention,
)
from nomorals.skills.filebook import FileSkillLibrary
from nomorals.skills.packs import load_pack, validate_pack
from nomorals.storage.db import Database

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:  # noqa: BLE001 - the media pilot's rung needs it
    _HAS_PIL = False


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


def _library() -> FileSkillLibrary:
    return FileSkillLibrary(
        user_dir=Path("/nonexistent-user-root-for-tests"))


# ── the software pilot (P5) ──────────────────────────────────────

_TASK_FILES = {"calc.py": "def add(a, b):\n    return a - b\n"}
_TASK_FIX = {"calc.py": "def add(a, b):\n    return a + b\n"}
_TASK_TESTS = {
    "test_calc.py": "from calc import add\n\n\n"
                    "def test_add():\n    assert add(2, 3) == 5\n"}


def _software_task(change: dict, files: dict | None = None) -> dict:
    return {"title": "fix the add function",
            "goal": "make add() add, proven by its acceptance test",
            "files": dict(files or _TASK_FILES),
            "change": dict(change),
            "tests": dict(_TASK_TESTS)}


class SoftwarePilotTests(unittest.TestCase):

    def _run(self, task: dict, **kwargs):
        tmp = _tmp(self)
        db = _tmp_db(self)
        return run_software_pilot(
            db, workspace=tmp / "ws", blob_dir=tmp / "blobs",
            task=task, library=_library(), **kwargs), db

    def test_end_to_end_run_completes_and_receipts(self):
        run, db = self._run(
            _software_task(_TASK_FIX),
            fixtures=("tests/test_d18p2_run_ticket.py",))
        self.assertTrue(run.completed, run.detail)
        self.assertEqual(run.outcome, "completed")
        self.assertTrue(all(row["passed"] for row in run.rubric),
                        run.rubric)
        self.assertEqual(
            run.detail["deliverable"]["files_changed"], ["calc.py"])
        self.assertEqual(
            run.detail["deliverable"]["tests_passed"], 1)
        # the skeptic's counterfactual is a recorded fact
        self.assertNotEqual(run.detail["pristine_rc"], 0)
        self.assertEqual(run.detail["patched_rc"], 0)
        # duties the task did not commit to are named absent
        self.assertIn("api transcript", run.absent)
        # receipt: derived from the run's records, practical proof
        receipt = run.receipt
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.outcome, "completed")
        self.assertEqual(receipt.grade.name, "G5")
        kinds = [ref.kind for ref in receipt.evidence]
        self.assertIn("empirical_source", kinds)
        self.assertIn("artifact_pass", kinds)
        # fixture economics: measured, not guessed
        economics = run.fixtures
        self.assertEqual(economics["declared"], 1)
        self.assertEqual(economics["files_passed"], 1)
        self.assertGreaterEqual(economics["tests_passed"], 1)
        self.assertGreater(economics["wall_seconds"], 0)

    def test_failed_run_grades_honestly_lower(self):
        run, _db = self._run(
            _software_task({"calc.py": "def add(a, b):\n"
                                       "    return a * b\n"}),
            fixtures=())
        self.assertFalse(run.completed)
        self.assertEqual(run.outcome, "failed")
        self.assertEqual(run.receipt.outcome, "failed")
        self.assertEqual(run.receipt.grade.name, "G1")
        kinds = [ref.kind for ref in run.receipt.evidence]
        self.assertNotIn("artifact_pass", kinds)
        self.assertNotIn("empirical_source", kinds)
        rubric = {row["check"]: row["passed"] for row in run.rubric}
        self.assertFalse(
            rubric["acceptance checks pass and the run is "
                   "recorded, not remembered"])

    def test_counterfactual_contradicts_a_vacuous_fix(self):
        already = {"calc.py": "def add(a, b):\n    return a + b\n"}
        run, db = self._run(
            _software_task(dict(already), files=already),
            fixtures=())
        self.assertEqual(run.outcome, "failed")
        from nomorals.agents.delegation import DelegationStore
        claims = DelegationStore(db).claims(run.contract_id)
        evidence = [c for c in claims if c.kind == "evidence"]
        self.assertTrue(evidence)
        self.assertEqual(evidence[0].verdict, "contradicted")
        self.assertEqual(run.receipt.grade.name, "G1")

    def test_pack_fixtures_are_real_files(self):
        root = Path(__file__).resolve().parent.parent
        for pack_id in ("software-craft-pack",
                        "visual-media-craft-pack",
                        "cbt-wellbeing-pack"):
            view = load_pack(_library(), pack_id)
            self.assertIsNotNone(view, pack_id)
            self.assertEqual(validate_pack(view.doc), [], pack_id)
            for fixture in view.fixtures:
                self.assertTrue((root / fixture).exists(),
                                f"{pack_id}: {fixture}")

    def test_fixture_economics_shape(self):
        economics = run_pack_fixtures(
            ("tests/test_d18p2_run_ticket.py",))
        self.assertEqual(economics["declared"], 1)
        self.assertEqual(economics["run"], 1)
        self.assertEqual(economics["files_failed"], 0)
        row = economics["per_fixture"][0]
        self.assertEqual(row["rc"], 0)
        self.assertGreater(row["tests_passed"], 0)


# ── the media pilot (P6) ─────────────────────────────────────────

def _png_bytes() -> bytes:
    img = Image.new("RGB", (400, 300), (30, 90, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 200, 150], fill=(220, 180, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_MEDIA_OPS = [{"op": "resize", "width": 320, "height": 240},
              {"op": "border", "width": 10, "color": "black"},
              {"op": "flip", "direction": "horizontal"}]
_MEDIA_RIGHTS = {
    "source": "owner-provided image (procedural test image)",
    "rights_holder": "the owner",
    "licence": "owner's own work — all rights held by the owner",
    "third_party": [],
}


@unittest.skipUnless(_HAS_PIL, "the Pillow edit rung is not "
                               "available on this box")
class MediaPilotTests(unittest.TestCase):

    def test_media_job_end_to_end_with_provenance(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        run = run_media_pilot(
            db, workdir=tmp / "work", blob_dir=tmp / "blobs",
            source_bytes=_png_bytes(), ops=_MEDIA_OPS,
            rights=_MEDIA_RIGHTS, library=_library())
        self.assertTrue(run.completed, run.detail)
        self.assertTrue(all(row["passed"] for row in run.rubric),
                        run.rubric)
        self.assertIn("generation lanes", run.absent)
        # the deliverable carries rights + edit graph, derived
        # from the source — additive provenance metadata
        from nomorals.storage.artifacts import ArtifactStore
        from nomorals.storage.blob import BlobStore
        artifacts = ArtifactStore(db, BlobStore(db, tmp / "blobs"))
        output = artifacts.get(run.detail["output_artifact"])
        self.assertIsNotNone(output)
        self.assertEqual(output.provenance.derived_from,
                         [run.detail["source_artifact"]])
        self.assertEqual(output.metadata["rights"]["licence"],
                         _MEDIA_RIGHTS["licence"])
        self.assertEqual(
            [step["op"] for step in output.metadata["edit_graph"]],
            ["resize", "border", "flip"])
        # the probe the verifier read back: 320x240 + 10px border
        self.assertEqual(
            (run.detail["probe"]["width"],
             run.detail["probe"]["height"]), (340, 260))
        # the job's durable record stands done
        from nomorals.media_edit.jobs import MediaJobRecordStore
        job_id = run.detail["job_ref"].split(":", 1)[1]
        record = MediaJobRecordStore(db).get(job_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "done")
        self.assertEqual(run.receipt.grade.name, "G5")
        self.assertEqual(run.receipt.outcome, "completed")

    def test_failed_job_fails_the_run_openly(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        run = run_media_pilot(
            db, workdir=tmp / "work", blob_dir=tmp / "blobs",
            source_bytes=b"this is not an image", ops=_MEDIA_OPS,
            rights=_MEDIA_RIGHTS, library=_library())
        self.assertEqual(run.outcome, "failed")
        self.assertEqual(run.detail["job_status"], "failed")
        self.assertEqual(run.receipt.grade.name, "G1")
        self.assertEqual(run.receipt.outcome, "failed")


# ── the CBT pilot (P7) ───────────────────────────────────────────

_EXERCISES = {
    "thought_record": {
        "situation": "My manager moved my deadline forward",
        "automatic_thought": "I will never finish and everyone "
                             "will think I am careless",
        "emotion": "anxious",
        "evidence_for": "The deadline is real and the task list "
                        "is long",
        "evidence_against": "I finished a similar task last "
                            "month in three days",
        "balanced_thought": "It is tight, and I have done tight "
                            "before; I will start with the "
                            "outline today",
        "intensity_before": 8, "intensity_after": 4,
    },
    "behavioral_activation": {
        "activity": "A 20-minute walk after lunch",
        "scheduled_for": "Tomorrow at 13:00",
        "expected_pleasure": 6, "expected_mastery": 5,
    },
    "cognitive_restructuring": {
        "trigger": "A friend did not reply to my message",
        "automatic_thought": "They are angry with me",
        "reframe": "They may simply be busy; one unanswered "
                   "message is weak evidence",
        "distortions": ["mind reading"],
    },
    "sleep_routine": {
        "bedtime": "23:00", "wake_time": "06:30",
        "wind_down": ["warm shower", "paper book",
                      "phone charges in the kitchen"],
    },
    "activity_routine": {
        "activity": "Swimming", "days": ["Tuesday", "Thursday"],
        "duration_minutes": 45,
    },
    "relapse_prevention": {
        "triggers": ["late nights before deadlines"],
        "warning_signs": ["skipping lunch",
                          "staying in bed past nine"],
        "coping_steps": ["call my brother",
                         "walk the long route home"],
        "support_contacts": ["brother"],
    },
    "session_prep": {
        "topics": ["sleep this month",
                   "the deadline change at work"],
        "questions": ["how do I keep the routine on travel days"],
    },
}


class CbtPilotTests(unittest.TestCase):

    def _setup(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        return db, tmp / "interventions.db"

    def test_every_exercise_kind_flows_end_to_end(self):
        db, path = self._setup()
        store = InterventionStore(path)
        for kind, fields in _EXERCISES.items():
            run = run_cbt_pilot(
                db, exercise={"kind": kind, "fields": fields},
                interventions_path=path, library=_library())
            self.assertTrue(run.completed, (kind, run.detail))
            self.assertTrue(
                all(row["passed"] for row in run.rubric),
                (kind, run.rubric))
            self.assertEqual(run.receipt.grade.name, "G5", kind)
            record = store.get(run.detail["record_id"])
            self.assertIsNotNone(record, kind)
            self.assertEqual(record["status"], "recorded")
            self.assertEqual(store.validate_record(record["id"]),
                             [], kind)
            # zero diagnostic output across the fixture set
            rendered = render_intervention(kind, fields)
            self.assertEqual(diagnostic_scan(rendered), [], kind)

    def test_diagnostic_language_is_blocked_by_the_skeptic(self):
        db, path = self._setup()
        fields = dict(_EXERCISES["thought_record"])
        fields["balanced_thought"] = (
            "Writing it out proves you have a disorder and the "
            "next step is to prescribe rest")
        run = run_cbt_pilot(
            db, exercise={"kind": "thought_record",
                          "fields": fields},
            interventions_path=path, library=_library())
        self.assertEqual(run.outcome, "failed")
        record = InterventionStore(path).get(
            run.detail["record_id"])
        self.assertEqual(record["status"], "blocked")
        self.assertEqual(run.receipt.grade.name, "G1")
        rubric = {row["check"]: row["passed"] for row in run.rubric}
        self.assertFalse(
            rubric["no diagnostic or prescriptive language in "
                   "the record"])

    def test_incomplete_exercise_is_refused_by_name(self):
        db, path = self._setup()
        fields = dict(_EXERCISES["thought_record"])
        del fields["balanced_thought"]
        run = run_cbt_pilot(
            db, exercise={"kind": "thought_record",
                          "fields": fields},
            interventions_path=path, library=_library())
        self.assertEqual(run.outcome, "failed")
        self.assertIn("balanced_thought", run.detail["refusal"])
        self.assertEqual(InterventionStore(path).list(), [])
        self.assertEqual(run.receipt.grade.name, "G1")


# ── cross-cutting: grades, chains, the graph ─────────────────────

class CrossCuttingTests(unittest.TestCase):

    def _all_runs(self):
        tmp = _tmp(self)
        db = _tmp_db(self)
        library = _library()
        software = run_software_pilot(
            db, workspace=tmp / "ws", blob_dir=tmp / "blobs",
            task=_software_task(_TASK_FIX), library=library,
            fixtures=())
        media = run_media_pilot(
            db, workdir=tmp / "work", blob_dir=tmp / "blobs",
            source_bytes=_png_bytes(), ops=_MEDIA_OPS,
            rights=_MEDIA_RIGHTS, library=library)
        cbt = run_cbt_pilot(
            db, exercise={"kind": "session_prep",
                          "fields": _EXERCISES["session_prep"]},
            interventions_path=tmp / "interventions.db",
            library=library)
        return db, library, tmp, software, media, cbt

    @unittest.skipUnless(_HAS_PIL, "the Pillow edit rung is not "
                                   "available on this box")
    def test_no_pilot_authors_a_grade(self):
        _db, _lib, _tmp_dir, software, media, cbt = self._all_runs()
        for run in (software, media, cbt):
            receipt = run.receipt
            payload = receipt.to_dict()
            payload["grade"] = "G6"
            with self.assertRaises(ProofGradeError):
                QualityReceipt.from_dict(payload)
            with self.assertRaises(ProofGradeError):
                QualityReceipt.asserting(
                    "G6", source=receipt.source,
                    source_id=receipt.source_id,
                    evidence=receipt.evidence)

    @unittest.skipUnless(_HAS_PIL, "the Pillow edit rung is not "
                                   "available on this box")
    def test_receipt_chain_renders_all_three_in_one_shape(self):
        _db, _lib, _tmp_dir, software, media, cbt = self._all_runs()
        text = render_receipt_chain(
            [software.receipt, media.receipt, cbt.receipt],
            title="pilot receipts")
        self.assertIn("delegation contract", text)
        self.assertIn("media job", text)
        self.assertIn("CBT intervention", text)
        self.assertIn("grade G5", text)
        for run in (software, media, cbt):
            self.assertNotIn(run.contract_id, text)
        self.assertNotIn(cbt.detail["record_id"], text)
        self.assertEqual(diagnostic_scan(text), [])

    @unittest.skipUnless(_HAS_PIL, "the Pillow edit rung is not "
                                   "available on this box")
    def test_graph_projects_the_pilot_runs(self):
        db, library, _tmp_dir, software, media, cbt = self._all_runs()
        from nomorals.craft.graph import project_graph
        graph = project_graph(db, library=library)
        for pack_id in ("software-craft-pack",
                        "visual-media-craft-pack",
                        "cbt-wellbeing-pack"):
            node = graph.node(f"pack:{pack_id}")
            self.assertIsNotNone(node, pack_id)
            self.assertEqual(node.kind, "pack")
        for run in (software, media, cbt):
            self.assertIsNotNone(
                graph.node(f"contract:{run.contract_id}"))
            self.assertIsNotNone(
                graph.node(f"ticket:contract:{run.contract_id}"))
        job_ticket = graph.node(
            f"ticket:{media.detail['job_ref']}")
        self.assertIsNotNone(job_ticket)
        self.assertEqual(job_ticket.detail["kind"], "media_job")
        output = graph.node(
            f"artifact:{media.detail['output_artifact']}")
        self.assertIsNotNone(output)
        derived = [e.dst for e in graph.edges_from(output.key)
                   if e.kind == "derived_from"]
        self.assertEqual(
            derived,
            [f"artifact:{media.detail['source_artifact']}"])

    def test_router_carries_the_pilot_packs(self):
        library = _library()
        media = library.route("edit graph rights manifest",
                              limit=5)
        self.assertIn("visual-media-craft-pack",
                      [doc.id for doc, _score in media])
        cbt = library.route("thought record intervention record",
                            limit=5)
        self.assertIn("cbt-wellbeing-pack",
                      [doc.id for doc, _score in cbt])

    def test_a_malformed_pack_refuses_the_pilot(self):
        db = _tmp_db(self)
        tmp = _tmp(self)
        with self.assertRaises(PilotError):
            run_cbt_pilot(
                db, exercise={"kind": "session_prep",
                              "fields": _EXERCISES["session_prep"]},
                interventions_path=tmp / "iv.db",
                library=FileSkillLibrary(
                    seed_dir=Path("/nonexistent-seed-for-tests"),
                    user_dir=Path("/nonexistent-user-for-tests")))


if __name__ == "__main__":
    unittest.main()
