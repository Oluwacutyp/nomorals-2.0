"""Motion-studio skill ports (2026-11-14): the six ADOPT-AS-SKILL items
from the @0xMovez motion-studio survey, ported into Devon's skill
library — motion-design (LottieFiles), anatomy (wheresryan22),
hairline-create + hairline-adopt (lucasmarkes), anatomy-reveal
(annnguyentt), motion-web (Motion ai-kit, free part), remotion-video
(Remotion skills, pattern adoption — the upstream repo has no LICENSE
file, so nothing is copied verbatim).

Pins: every new skill file parses with valid frontmatter, is
registered in library/INDEX.md under the media layer, and its
vendored engine/template files are present next to it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from nomorals.skills.filebook import parse_skill_text, seed_library_dir

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "nomorals" / "skills" / "library"

# id -> (skill file, vendored files that must exist beside it)
MOTION_SKILLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "motion-design": ("motion-design.md", ()),
    "anatomy": (
        "anatomy.md",
        (
            "anatomy/kit/iso-kit.mjs",
            "anatomy/kit/iso-kit.ts",
            "anatomy/kit/lathe.mjs",
            "anatomy/kit/tube.mjs",
            "anatomy/kit/audit.mjs",
            "anatomy/references/craft.md",
            "anatomy/scripts/capture.mjs",
            "anatomy/LICENSE",
        ),
    ),
    "hairline-create": (
        "hairline-create.md",
        (
            "hairline/kernel.js",
            "hairline/build.mjs",
            "hairline/validate.mjs",
            "hairline/look.mjs",
            "hairline/bench.html",
            "hairline/rules.md",
            "hairline/concepts.md",
            "hairline/LICENSE",
        ),
    ),
    "hairline-adopt": ("hairline-adopt.md", ()),
    "anatomy-reveal": (
        "anatomy-reveal.md",
        (
            "anatomy-reveal/template/object/geometry.mjs",
            "anatomy-reveal/template/motion/timeline.mjs",
            "anatomy-reveal/template/motion/render.mjs",
            "anatomy-reveal/scripts/new-project.mjs",
            "anatomy-reveal/references/shots.md",
            "anatomy-reveal/LICENSE",
        ),
    ),
    "motion-web": ("motion-web.md", ()),
    "remotion-video": ("remotion-video.md", ()),
}


def _index_ids() -> set[str]:
    ids = set()
    for line in (LIB / "INDEX.md").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- `"):
            ids.add(line[3:].split("`", 1)[0])
    return ids


class MotionSkillsFrontmatterTests(unittest.TestCase):
    def test_every_motion_skill_parses_with_valid_frontmatter(self):
        for skill_id, (filename, _vendored) in MOTION_SKILLS.items():
            with self.subTest(skill=skill_id):
                path = LIB / filename
                self.assertTrue(path.is_file(), str(path))
                doc = parse_skill_text(
                    path.read_text(encoding="utf-8"), path=str(path))
                self.assertIsNotNone(doc, filename)
                assert doc is not None
                self.assertEqual(doc.id, skill_id)
                self.assertEqual(doc.id, filename[: -len(".md")])
                self.assertTrue(doc.summary, skill_id)
                self.assertTrue(doc.topics, skill_id)
                self.assertEqual(doc.layer, "media", skill_id)
                self.assertEqual(doc.version, 1, skill_id)
                self.assertGreater(len(doc.body), 200, skill_id)

    def test_every_motion_skill_is_registered_in_index(self):
        registered = _index_ids()
        for skill_id in MOTION_SKILLS:
            with self.subTest(skill=skill_id):
                self.assertIn(skill_id, registered)

    def test_index_media_section_lists_the_motion_skills(self):
        text = (LIB / "INDEX.md").read_text(encoding="utf-8")
        media = text.split("## media", 1)[1].split("\n## ", 1)[0]
        for skill_id in MOTION_SKILLS:
            with self.subTest(skill=skill_id):
                self.assertIn(f"`{skill_id}`", media)

    def test_vendored_engines_are_present(self):
        for skill_id, (_filename, vendored) in MOTION_SKILLS.items():
            for rel in vendored:
                with self.subTest(skill=skill_id, file=rel):
                    self.assertTrue((LIB / rel).is_file(), rel)

    def test_seed_library_dir_is_the_checked_in_library(self):
        self.assertEqual(seed_library_dir(), LIB)


class MotionSkillsContentTests(unittest.TestCase):
    def _body(self, skill_id: str) -> str:
        filename = MOTION_SKILLS[skill_id][0]
        doc = parse_skill_text(
            (LIB / filename).read_text(encoding="utf-8"))
        assert doc is not None
        return doc.body

    def test_motion_design_carries_the_principles_and_tables(self):
        body = self._body("motion-design")
        for needle in (
            "Squash and stretch",
            "Anticipation",
            "cubic-bezier(0.4,0,0.2,1)",
            "1/3 Rule",
            "keyframes.py",
        ):
            self.assertIn(needle, body)

    def test_anatomy_carries_the_one_idea_and_sizes(self):
        body = self._body("anatomy")
        for needle in (
            "The object is the explanation",
            "Figure",
            "Hero",
            "Epic",
            "audit passed",
        ):
            self.assertIn(needle, body)

    def test_motion_web_records_the_motionscore_gate(self):
        body = self._body("motion-web")
        self.assertIn("MotionScore", body)
        self.assertIn("never promises a MotionScore grade", body)
        self.assertIn("motion.dev/plus", body)
        self.assertIn("linear()", body)

    def test_remotion_video_names_the_deferred_engine(self):
        body = self._body("remotion-video")
        self.assertIn("does not have Remotion", body)
        self.assertIn("deferred", body)
        self.assertIn("no LICENSE file", body)
        self.assertIn("motion_studio", body)

    def test_anatomy_reveal_names_its_host_requirements(self):
        body = self._body("anatomy-reveal")
        self.assertIn("magick", body)
        self.assertIn("ffmpeg", body)
        self.assertIn("new-project.mjs", body)

    def test_hairline_adopt_delegates_to_create(self):
        body = self._body("hairline-adopt")
        self.assertIn("hairline-create", body)


if __name__ == "__main__":
    unittest.main()
