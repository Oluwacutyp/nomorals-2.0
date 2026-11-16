"""Skills bundle convention: SKILL.md + scripts/examples/resources
beside it, description-as-trigger (Antigravity)."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

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
    test.addCleanup(lambda: shutil.rmtree(tmp, True))
    return FileSkillLibrary(seed_dir=Path(tmp) / "seed",
                            user_dir=Path(tmp) / "user")


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _devon_skill(skill_id: str, summary: str, *, title: str = "",
                 topics: str = "", triggers: str = "",
                 body: str = "Do the work.") -> str:
    lines = ["---", f"id: {skill_id}",
             f"title: {title or skill_id}",
             f"summary: {summary}"]
    if topics:
        lines.append(f"topics: {topics}")
    if triggers:
        lines.append(f"triggers: {triggers}")
    lines += ["layer: test", "version: 1", "---", "", body, ""]
    return "\n".join(lines)


class ParseAliasTests(unittest.TestCase):
    def test_open_standard_frontmatter_parses(self):
        doc = parse_skill_text(
            "---\nname: hairline-create\n"
            "description: Use when someone asks for a new Hairline "
            "figure.\n---\n\n# Body\n\nRules.\n")
        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.id, "hairline-create")
        self.assertEqual(
            doc.summary,
            "Use when someone asks for a new Hairline figure.")
        self.assertEqual(doc.title, "hairline-create")

    def test_devon_fields_win_over_standard_aliases(self):
        doc = parse_skill_text(
            "---\nid: devon-id\nname: other-name\n"
            "title: Devon Title\nsummary: devon summary\n"
            "description: standard description\n---\n\nBody.\n")
        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertEqual(doc.id, "devon-id")
        self.assertEqual(doc.summary, "devon summary")
        self.assertEqual(doc.title, "Devon Title")

    def test_standard_skill_still_needs_a_name(self):
        self.assertIsNone(parse_skill_text(
            "---\ndescription: no name here\n---\n\nBody.\n"))


class BundleDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.lib = _tmp_library(self)
        seed = self.lib.seed_dir
        _write(seed / "flat-twin.md", _devon_skill(
            "flat-twin", "the flat twin skill",
            title="Flat Twin", topics="twin", triggers="flat twin"))
        _write(seed / "bundled" / "SKILL.md", _devon_skill(
            "bundled-skill", "a bundled skill",
            title="Bundled Skill", topics="bundled",
            triggers="bundled skill"))
        _write(seed / "bundled" / "scripts" / "run.py", "# run\n")
        _write(seed / "bundled" / "examples" / "demo.md", "# demo\n")
        _write(seed / "bundled" / "resources" / "schema.json", "{}\n")
        _write(seed / "twin-pack" / "SKILL.md",
               "---\nname: flat-twin\n"
               "description: the bundled twin of the flat skill\n"
               "---\n\nBundled body.\n")
        _write(seed / "twin-pack" / "scripts" / "helper.py", "# help\n")
        _write(seed / "nested-pack" / "skills" / "deep" / "SKILL.md",
               "---\nname: deep-skill\n"
               "description: two levels down\n---\n\nDeep body.\n")
        _write(seed / "no-skill" / "README.md", "# no skill here\n")
        self.lib.invalidate()

    def test_bundle_dir_is_one_skill_with_discoverable_files(self):
        doc = self.lib.get("bundled-skill")
        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertTrue(doc.bundle_dir.endswith("bundled"))
        self.assertEqual(doc.bundle_files(), (
            "examples/demo.md",
            "resources/schema.json",
            "scripts/run.py",
        ))

    def test_flat_twin_wins_the_doc_and_the_bundle_attaches(self):
        ids = [d.id for d in self.lib.docs(include_drafts=False)]
        self.assertEqual(ids.count("flat-twin"), 1)
        doc = self.lib.get("flat-twin")
        assert doc is not None
        # The flat file is the routed text: its Devon metadata stands.
        self.assertEqual(doc.title, "Flat Twin")
        self.assertEqual(doc.summary, "the flat twin skill")
        self.assertEqual(doc.topics, ("twin",))
        # ...and the bundle directory attaches as its companion store.
        self.assertTrue(doc.bundle_dir.endswith("twin-pack"))
        self.assertEqual(doc.bundle_files(), ("scripts/helper.py",))

    def test_nested_skill_packs_and_skill_less_dirs_are_not_discovered(self):
        self.assertIsNone(self.lib.get("deep-skill"))
        ids = [d.id for d in self.lib.docs(include_drafts=False)]
        self.assertEqual(sorted(ids), ["bundled-skill", "flat-twin"])

    def test_flat_skills_have_no_bundle(self):
        doc = self.lib.get("bundled-skill")
        assert doc is not None
        flat = self.lib.get("flat-twin")
        assert flat is not None
        self.assertTrue(flat.bundle_dir)
        plain = _write(self.lib.seed_dir / "plain.md", _devon_skill(
            "plain-skill", "a plain flat skill"))
        self.assertTrue(plain.is_file())
        self.lib.invalidate()
        self.assertEqual(self.lib.get("plain-skill").bundle_dir, "")
        self.assertEqual(self.lib.get("plain-skill").bundle_files(), ())


class SeedBundleTests(unittest.TestCase):
    def test_hairline_bundle_attaches_without_changing_the_index(self):
        lib = FileSkillLibrary(
            user_dir=Path("/nonexistent-user-root-for-tests"))
        doc = lib.get("hairline-create")
        self.assertIsNotNone(doc)
        assert doc is not None
        self.assertTrue(doc.bundle_dir.endswith("library/hairline"))
        files = doc.bundle_files()
        self.assertIn("examples/terrain.js", files)
        self.assertIn("kernel.js", files)
        self.assertNotIn("SKILL.md", files)
        seed = [d for d in lib.docs(include_drafts=False)
                if d.origin == "seed"]
        checked_in = (seed_library_dir() / "INDEX.md").read_text(
            encoding="utf-8")
        self.assertEqual(checked_in, render_index(seed))


class DescriptionTriggerTests(unittest.TestCase):
    def setUp(self):
        self.lib = _tmp_library(self)
        seed = self.lib.seed_dir
        _write(seed / "desc-triggered.md", _devon_skill(
            "desc-triggered",
            "Use when someone asks for a new hairline figure",
            title="Hairline", body="Do the work."))
        _write(seed / "topic-listed.md", _devon_skill(
            "topic-listed", "drawing figures",
            title="Figure basics", topics="figure",
            body="Draw a figure."))
        self.lib.invalidate()

    def test_description_words_route_at_trigger_weight(self):
        routed = self.lib.route("new hairline figure", limit=2)
        self.assertTrue(routed)
        self.assertEqual(routed[0][0].id, "desc-triggered")

    def test_description_does_not_fire_on_unrelated_words(self):
        routed = self.lib.route("unrelated query about databases",
                                limit=2)
        self.assertEqual(
            [d.id for d, _ in routed], [])


class DraftBundleTests(unittest.TestCase):
    def setUp(self):
        self.lib = _tmp_library(self)

    def _draft(self, skill_id="bundle-draft"):
        return SkillDoc(id=skill_id, title="Bundle Draft",
                        summary="a bundled draft",
                        topics=("testing",), layer="distilled",
                        body="# Bundle\n\nDo the thing.",
                        origin="draft")

    def test_draft_can_land_as_a_bundle(self):
        path = self.lib.write_draft(self._draft(), bundle=True)
        self.assertEqual(path.name, "SKILL.md")
        self.assertEqual(path.parent.name, "bundle-draft")
        for sub in ("scripts", "examples", "resources"):
            self.assertTrue((path.parent / sub).is_dir(), sub)
        draft = self.lib.get_draft("bundle-draft")
        self.assertIsNotNone(draft)
        assert draft is not None
        self.assertTrue(draft.bundle_dir.endswith("bundle-draft"))

    def test_bundle_draft_promotes_with_its_companions(self):
        path = self.lib.write_draft(self._draft(), bundle=True)
        _write(path.parent / "scripts" / "helper.py", "# help\n")
        self.lib.invalidate()
        result = self.lib.promote("bundle-draft")
        self.assertTrue(result["ok"])
        live = self.lib.get("bundle-draft")
        self.assertIsNotNone(live)
        assert live is not None
        self.assertEqual(live.origin, "user")
        self.assertTrue(live.bundle_dir.endswith("bundle-draft"))
        self.assertIn("scripts/helper.py", live.bundle_files())
        self.assertIsNone(self.lib.get_draft("bundle-draft"))
        self.assertFalse((self.lib.user_dir / "drafts"
                          / "bundle-draft").exists())

    def test_flat_draft_still_lands_as_a_flat_file(self):
        path = self.lib.write_draft(self._draft("flat-draft"))
        self.assertEqual(path.name, "flat-draft.md")
        self.assertEqual(self.lib.get_draft("flat-draft").bundle_dir, "")


if __name__ == "__main__":
    unittest.main()
