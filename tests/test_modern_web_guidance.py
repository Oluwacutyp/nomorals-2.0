"""Modern Web Guidance 1.0 skill port (2026-11-14): Google's
modern-web-guidance skill, ported into Devon's skill library as a
router skill — the SKILL.md workflow (search → retrieve → verify →
disclose, plus the modernisation flow) adapted into Devon's format
and voice, with the 209 guides vendored as-is under the skill's
directory (Apache-2.0, LICENSE kept alongside).

Pins: the router skill parses with valid frontmatter, is registered
in library/INDEX.md under the media layer, all 209 guides are
present in their 17 categories, the LICENSE is present, and the
skill-version pin is recorded.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from nomorals.skills.filebook import parse_skill_text, seed_library_dir

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "nomorals" / "skills" / "library"
GUIDES = LIB / "modern-web-guidance" / "guides"

# category -> guide count (verified against the published artifact)
GUIDE_COUNTS: dict[str, int] = {
    "accessibility": 4,
    "built-in-ai": 5,
    "css": 18,
    "forms": 19,
    "html": 8,
    "js": 5,
    "performance": 26,
    "privacy": 1,
    "pwa": 2,
    "security": 17,
    "ui-atoms": 13,
    "ui-behaviors": 40,
    "ui-components": 22,
    "user-experience": 3,
    "visual-design": 22,
    "wasm": 1,
    "webmcp": 3,
}
TOTAL_GUIDES = 209
SKILL_VERSION_PIN = "2026_11_09-5bcc021b"


def _index_ids() -> set[str]:
    ids = set()
    for line in (LIB / "INDEX.md").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- `"):
            ids.add(line[3:].split("`", 1)[0])
    return ids


class ModernWebGuidanceFrontmatterTests(unittest.TestCase):
    def test_router_skill_parses_with_valid_frontmatter(self):
        path = LIB / "modern-web-guidance.md"
        self.assertTrue(path.is_file(), str(path))
        doc = parse_skill_text(
            path.read_text(encoding="utf-8"), path=str(path))
        self.assertIsNotNone(doc, "modern-web-guidance.md")
        assert doc is not None
        self.assertEqual(doc.id, "modern-web-guidance")
        self.assertEqual(doc.id, "modern-web-guidance.md"[: -len(".md")])
        self.assertTrue(doc.summary)
        self.assertTrue(doc.topics)
        self.assertEqual(doc.layer, "media")
        self.assertEqual(doc.version, 1)
        self.assertGreater(len(doc.body), 200)

    def test_router_skill_is_registered_in_index(self):
        self.assertIn("modern-web-guidance", _index_ids())

    def test_index_media_section_lists_the_skill(self):
        text = (LIB / "INDEX.md").read_text(encoding="utf-8")
        media = text.split("## media", 1)[1].split("\n## ", 1)[0]
        self.assertIn("`modern-web-guidance`", media)

    def test_seed_library_dir_is_the_checked_in_library(self):
        self.assertEqual(seed_library_dir(), LIB)


class ModernWebGuidanceGuidesTests(unittest.TestCase):
    def test_all_209_guides_are_present(self):
        found = sorted(GUIDES.rglob("*.md"))
        self.assertEqual(len(found), TOTAL_GUIDES,
                         f"expected {TOTAL_GUIDES} guides, found {len(found)}")

    def test_every_category_has_its_guides(self):
        for category, count in GUIDE_COUNTS.items():
            with self.subTest(category=category):
                files = sorted((GUIDES / category).glob("*.md"))
                self.assertEqual(len(files), count, category)
                for path in files:
                    self.assertTrue(path.is_file(), str(path))

    def test_guides_are_plain_markdown_with_a_title(self):
        for path in sorted(GUIDES.rglob("*.md")):
            with self.subTest(guide=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertTrue(text.startswith("# "),
                                f"{path} does not start with a title")
                self.assertGreater(len(text), 100, str(path))

    def test_apache_license_is_kept_with_the_guides(self):
        license_path = LIB / "modern-web-guidance" / "LICENSE"
        self.assertTrue(license_path.is_file(), str(license_path))
        self.assertIn("Apache License", license_path.read_text(
            encoding="utf-8"))
        self.assertIn("Version 2.0, January 2004", license_path.read_text(
            encoding="utf-8"))

    def test_chrome_extensions_skill_is_not_ported(self):
        self.assertFalse((LIB / "chrome-extensions.md").exists())
        self.assertFalse((LIB / "modern-web-guidance" / "chrome-extensions").exists())


class ModernWebGuidanceContentTests(unittest.TestCase):
    def _body(self) -> str:
        doc = parse_skill_text(
            (LIB / "modern-web-guidance.md").read_text(encoding="utf-8"))
        assert doc is not None
        return doc.body

    def test_router_carries_the_workflow(self):
        body = self._body()
        for needle in (
            "Step 1. Search",
            "Step 2. Retrieve",
            "Step 3. Verify",
            "Step 4. Disclose",
            "Modern Web Guidance",
        ):
            self.assertIn(needle, body)

    def test_router_scopes_to_devons_web_surfaces(self):
        body = self._body()
        for needle in (
            "handoff link",
            "pairing page",
            "computer organ",
            "motion-server.mjs",
        ):
            self.assertIn(needle, body)

    def test_router_records_the_skill_version_pin(self):
        body = self._body()
        self.assertIn(SKILL_VERSION_PIN, body)

    def test_router_names_devon_not_your_project(self):
        body = self._body()
        self.assertIn("Devon", body)
        self.assertNotIn("your project", body)

    def test_router_names_the_honest_gaps(self):
        body = self._body()
        for needle in (
            "Content Security Policy",
            "HTML sanitisation",
            "Web vitals",
            "Passkeys",
            "legacy-scan",
        ):
            self.assertIn(needle, body)

    def test_router_names_the_guide_inventory(self):
        body = self._body()
        self.assertIn("209 guides", body)
        self.assertIn("accessibility", body)
        self.assertIn("security", body)
        self.assertIn("performance", body)


if __name__ == "__main__":
    unittest.main()
