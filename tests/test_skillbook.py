"""SKILLS system: flat-file skillbook, routing, index/layer drift pins,
usage + promotion, post-task distillation, and the context seam."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.skills import layers as layers_mod
from nomorals.skills.distill_files import distill_task, should_distill
from nomorals.skills.filebook import (
    FileSkillLibrary,
    SkillDoc,
    parse_skill_text,
    render_index,
    seed_library_dir,
)

REPO = Path(__file__).resolve().parent.parent


def _tmp_library(test: unittest.TestCase) -> FileSkillLibrary:
    tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR") or None)
    test.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
    return FileSkillLibrary(seed_dir=Path(tmp) / "no-seed",
                            user_dir=Path(tmp) / "user")


class ParseTests(unittest.TestCase):
    def test_parse_full_frontmatter(self):
        doc = parse_skill_text(
            "---\nid: demo\ntitle: Demo Skill\nsummary: one line\n"
            "topics: a, b\ntriggers: do it, make it\nlayer: agents\n"
            "version: 3\n---\n\n# Body\n\nRules.\n")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.id, "demo")
        self.assertEqual(doc.topics, ("a", "b"))
        self.assertEqual(doc.triggers, ("do it", "make it"))
        self.assertEqual(doc.version, 3)
        self.assertIn("Rules.", doc.body)

    def test_parse_requires_frontmatter_and_id(self):
        self.assertIsNone(parse_skill_text("# Just a heading\n"))
        self.assertIsNone(parse_skill_text(
            "---\ntitle: No id\n---\nbody\n"))


class SeedLibraryTests(unittest.TestCase):
    def setUp(self):
        self.lib = FileSkillLibrary(
            user_dir=Path("/nonexistent-user-root-for-tests"))

    def test_seed_library_is_a_real_starter_set(self):
        seed = [d for d in self.lib.docs(include_drafts=False)
                if d.origin == "seed"]
        self.assertGreaterEqual(len(seed), 12)
        ids = [d.id for d in seed]
        self.assertEqual(len(ids), len(set(ids)), "duplicate skill ids")
        for doc in seed:
            self.assertTrue(doc.summary, doc.id)
            self.assertTrue(doc.topics, doc.id)
            self.assertGreater(len(doc.body), 200, doc.id)

    def test_checked_in_index_matches_render(self):
        seed = [d for d in self.lib.docs(include_drafts=False)
                if d.origin == "seed"]
        checked_in = (seed_library_dir() / "INDEX.md").read_text(
            encoding="utf-8")
        self.assertEqual(checked_in, render_index(seed))

    def test_routing_picks_the_right_skill(self):
        cases = {
            "run a long mission in the background": "mission-execution",
            "delegate the work and verify the claims":
                "accountable-delegation",
            "send a gif and a couple profile picture":
                "media-delivery-shapes",
            "pull the latest market read for gold": "trading-read",
            "why is this test failing": "failure-triage",
        }
        for query, expected in cases.items():
            routed = self.lib.route(query, limit=1)
            self.assertTrue(routed, query)
            self.assertEqual(routed[0][0].id, expected, query)


class LayerInstructionTests(unittest.TestCase):
    def test_layer_groups_match_the_layering_test_map(self):
        tree = ast.parse(
            (REPO / "tests" / "test_layering.py").read_text(
                encoding="utf-8"))
        layers_map = shared = top_level = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and getattr(
                    node.target, "id", "") == "LAYERS":
                layers_map = ast.literal_eval(node.value)
            if isinstance(node, ast.Assign) and getattr(
                    node.targets[0], "id", "") == "SHARED":
                shared = ast.literal_eval(node.value)
            if isinstance(node, ast.Assign) and getattr(
                    node.targets[0], "id", "") == "TOP_LEVEL_LAYER":
                top_level = ast.literal_eval(node.value)
        self.assertIsNotNone(layers_map)
        expected: dict[int, set[str]] = {}
        for package, layer in layers_map.items():
            expected.setdefault(layer, set()).add(package)
        for package in shared or ():
            expected.setdefault(0, set()).add(package)
        for module, layer in (top_level or {}).items():
            expected.setdefault(layer, set()).add(module)
        actual: dict[int, set[str]] = {}
        for number, group in layers_mod.LAYER_GROUPS.items():
            names = set(group.get("packages", ())) | set(
                group.get("modules", ()))
            actual[number] = names
        self.assertEqual(actual, expected)

    def test_layer_files_match_canonical_render(self):
        for number, group in layers_mod.LAYER_GROUPS.items():
            path = layers_mod.layers_dir() / (
                f"layer-{number}-{group['name']}.md")
            self.assertTrue(path.is_file(), str(path))
            self.assertEqual(path.read_text(encoding="utf-8"),
                             layers_mod.render_layer_file(number),
                             f"layer {number} file drifted — regenerate "
                             "from layers.render_layer_file")


class UsagePromotionTests(unittest.TestCase):
    def setUp(self):
        self.lib = _tmp_library(self)

    def _draft(self, skill_id="learned-thing", revises=""):
        return SkillDoc(id=skill_id, title="Learned Thing",
                        summary="a distilled playbook",
                        topics=("testing",), layer="distilled",
                        body="# Learned\n\nDo the thing.",
                        origin="draft", revises=revises)

    def test_draft_promotes_after_two_successful_missions(self):
        self.lib.write_draft(self._draft())
        self.lib.record_load(["learned-thing"], mission_id="m1")
        out = self.lib.settle_mission("m1", success=True)
        self.assertEqual(out["promoted"], [])
        self.assertIsNotNone(self.lib.get_draft("learned-thing"))
        self.lib.record_load(["learned-thing"], mission_id="m1")
        self.lib.settle_mission("m1", success=True)  # same mission again
        self.assertIsNotNone(self.lib.get_draft("learned-thing"),
                             "one mission twice must not promote")
        self.lib.record_load(["learned-thing"], mission_id="m2")
        out = self.lib.settle_mission("m2", success=True)
        self.assertEqual(out["promoted"], ["learned-thing"])
        self.assertIsNone(self.lib.get_draft("learned-thing"))
        live = self.lib.get("learned-thing")
        self.assertIsNotNone(live)
        self.assertEqual(live.origin, "user")

    def test_failed_mission_does_not_count_toward_promotion(self):
        self.lib.write_draft(self._draft())
        for mission_id in ("m1", "m2"):
            self.lib.record_load(["learned-thing"], mission_id=mission_id)
            self.lib.settle_mission(mission_id, success=False)
        self.assertIsNotNone(self.lib.get_draft("learned-thing"))

    def test_explicit_promote_and_revision_bump(self):
        live = SkillDoc(id="base-skill", title="Base", summary="s",
                        topics=("x",), body="v1 body", origin="user",
                        version=1)
        self.lib.user_dir.mkdir(parents=True, exist_ok=True)
        from nomorals.skills.filebook import render_skill_text
        (self.lib.user_dir / "base-skill.md").write_text(
            render_skill_text(live), encoding="utf-8")
        self.lib.invalidate()
        draft = SkillDoc(id="base-skill", title="Base", summary="s2",
                         topics=("x",), body="v2 body", origin="draft",
                         revises="base-skill", version=1)
        self.lib.write_draft(draft)
        result = self.lib.promote("base-skill")
        self.assertTrue(result["ok"])
        promoted = self.lib.get("base-skill")
        self.assertEqual(promoted.version, 2)
        self.assertIn("v2 body", promoted.body)


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.lib = _tmp_library(self)

    def test_should_distill_truth_table(self):
        self.assertFalse(should_distill({"steps": [{}], "lessons": []}))
        self.assertTrue(should_distill({"steps": [{}, {}, {}]}))
        self.assertTrue(should_distill({"lessons": ["x"]}))
        self.assertTrue(should_distill({"outcome": "failed"}))

    def test_distill_writes_a_routable_draft(self):
        result = distill_task(self.lib, {
            "goal": "Repair the WhatsApp bridge reconnect loop",
            "outcome": "done", "mission_id": "m9",
            "steps": [{"name": "trace", "role": "research",
                       "tools": ["grep"]},
                      {"name": "patch", "role": "coding",
                       "tools": ["edit"]},
                      {"name": "prove", "role": "critic", "tools": []}],
            "lessons": ["The reconnect loop hid in the supervisor, "
                        "not the socket layer."],
        })
        self.assertTrue(result["ok"])
        draft = self.lib.get_draft(result["draft_id"])
        self.assertIsNotNone(draft)
        self.assertIn("supervisor", draft.body)
        routed = self.lib.route("whatsapp bridge reconnect", limit=3)
        self.assertIn(result["draft_id"], [d.id for d, _ in routed])


class ContextSeamTests(unittest.TestCase):
    def test_engine_loads_routed_skills_into_mission_context(self):
        from nomorals.context.engine import ContextEngine

        tmp = tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")
                               or None)
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, True))
        mission = SimpleNamespace(
            goal="run a long background mission and delegate the work",
            id="m-seam-1", status="running", acceptance_criteria=[],
            required_artifacts=[], state={})
        with mock.patch.dict(os.environ, {"NM_HOME": tmp}):
            engine = ContextEngine()
            built = engine.build(system="sys", mission=mission)
        names = [s.name for s in built.sections]
        self.assertIn("skills", names)
        self.assertIn("mission-execution",
                      built.meta.get("skills_loaded", []))
        usage = Path(tmp) / "skills" / "usage.jsonl"
        self.assertTrue(usage.is_file())
        self.assertIn("m-seam-1", usage.read_text(encoding="utf-8"))

    def test_engine_without_task_text_has_no_skills_section(self):
        from nomorals.context.engine import ContextEngine

        built = ContextEngine().build(system="sys")
        self.assertNotIn("skills", [s.name for s in built.sections])

    def test_engine_loads_long_skills_whole_no_local_ceiling(self):
        """Power mode: the skills section carries NO local char ceiling.

        The section builder used to pre-truncate routed skill docs at
        a hardcoded 4000-char pool (and never reach the second routed
        doc once the pool was spent) before the engine's own budget
        layer saw them.  Docs now arrive whole; overflow is governed
        by the caller's ContextBudget (fit/compress), the assembly's
        declared budget seam.  The engine here is given a roomy
        explicit budget, so any cut in the output is the old local
        ceiling's doing — and there must be none.
        """
        from nomorals.context.budget import ContextBudget
        from nomorals.context.engine import ContextEngine

        class Doc:
            def __init__(self, doc_id, body):
                self.id = doc_id
                self.body = body

            def header(self):
                return self.id

        class Library:
            def route(self, query, limit=2):
                return [(d, 1.0) for d in self._docs[:limit]]

            def record_load(self, loaded, mission_id=""):
                self.recorded = list(loaded)

        tail_a = "TAIL-A-" + "a" * 100
        tail_b = "TAIL-B-" + "b" * 100
        lib = Library()
        lib._docs = [Doc("skill-a", "a" * 5000 + tail_a),
                     Doc("skill-b", "b" * 3000 + tail_b)]
        engine = ContextEngine(budget=ContextBudget(total=100000))
        engine._skillbook = lib
        mission = SimpleNamespace(
            goal="delegate the long background work", id="m-seam-2",
            status="running", acceptance_criteria=[],
            required_artifacts=[], state={})
        built = engine.build(system="sys", mission=mission)
        # doc A alone exceeds the old 4000-char ceiling; doc B was
        # unreachable behind it.  Both tails arrive whole.
        self.assertIn(tail_a, built.text)
        self.assertIn(tail_b, built.text)
        self.assertNotIn("[skill truncated to context budget]",
                         built.text)
        self.assertEqual(built.meta.get("skills_loaded"),
                         ["skill-a", "skill-b"])


if __name__ == "__main__":
    unittest.main()
