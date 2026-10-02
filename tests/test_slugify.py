"""Parity tests for the unified ``nomorals.core.text.slugify``.

Six ``_slug``/``_slugify`` copies used to live across agents, media, tools,
and builders. Each call site is now a thin delegate of one canonical
``slugify(text, limit=None, fallback=..., ...)``. This file embeds the six
replaced implementations verbatim as fixtures and asserts the new delegate
produces byte-identical output on a battery of inputs.

Deliberate robustness gains (documented, not regressions):
- roles/fanout crashed on ``None`` (``text.lower()``); the shared helper
  returns the fallback instead.
"""

from __future__ import annotations

import re
import time
import unittest

from nomorals.core.text import slugify


# ── verbatim fixtures of the replaced implementations ──────────────────────


def _old_roles_slug(text, limit=32):
    """nomorals/agents/roles/__init__.py."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit] or "task"


def _old_reflection_slug(text, limit=40):
    """nomorals/agents/reflection.py."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:limit].strip("-") or "goal"


def _old_fanout_slug(text):
    """nomorals/agents/fanout.py (underscore separator, hard limit 40)."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "angle"


def _old_music_slugify(text, limit=48):
    """nomorals/media/music.py."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:limit] or "untitled"


_OLD_FS_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


def _old_filesend_slug(name, fallback):
    """nomorals/tools/filesend.py (case kept, '._' allowed, '-.' stripped,
    timestamped fallback)."""
    name = (name or "").strip()
    name = _OLD_FS_SLUG.sub("-", name).strip("-.")
    return name[:80] or f"{fallback}-{int(time.time())}"


def _old_appbuilder_slug(name):
    """nomorals/builders/app_builder.py (no limit)."""
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or "app"


BATTERY = [
    "Hello World!",
    "",
    "!!!",
    "  spaced out  ",
    "under_score test",
    "caf\u00e9 au lait",
    "a" * 100,
    "x" * 31 + "-rest-of-it",
    "a" * 39 + "-b",          # truncation leaves a trailing separator
    "a" * 40 + "-b",          # truncation lands exactly on a separator
    "hello--world",
    "MiXeD CaSe 123",
    "file.name_v2.PDF",
    "a/b\\c:d",
    "trailing-",
    "-leading",
    "...",
    "___",
    "with\ttab\nand newline",
]


class SlugParityTest(unittest.TestCase):
    def _check_parity(self, old_fn, new_fn, battery, *, skip_none=False):
        for text in battery:
            with self.subTest(repr(text)):
                if text is None and skip_none:
                    with self.assertRaises(Exception):
                        old_fn(text)
                    self.assertTrue(new_fn(text))  # shared helper returns the fallback
                    continue
                self.assertEqual(new_fn(text), old_fn(text),
                                 f"diverged on {text!r}")

    def test_roles_slug_parity(self):
        from nomorals.agents.roles import _slug

        self._check_parity(_old_roles_slug, _slug, BATTERY + [None], skip_none=True)
        # non-default limit honored exactly
        for text in BATTERY:
            with self.subTest(f"limit-5/{text!r}"):
                self.assertEqual(_slug(text, limit=5),
                                 _old_roles_slug(text, limit=5))

    def test_reflection_slug_parity(self):
        from nomorals.agents.reflection import _slug

        self._check_parity(_old_reflection_slug, _slug, BATTERY + [None])
        for text in BATTERY:
            with self.subTest(f"limit-5/{text!r}"):
                self.assertEqual(_slug(text, limit=5),
                                 _old_reflection_slug(text, limit=5))

    def test_fanout_slug_parity(self):
        from nomorals.agents.fanout import _slug

        self._check_parity(_old_fanout_slug, _slug, BATTERY + [None], skip_none=True)
        # underscore separator preserved
        self.assertEqual(_slug("Hello World!"), "hello_world")

    def test_music_slugify_parity(self):
        from nomorals.media.music import _slugify

        self._check_parity(_old_music_slugify, _slugify, BATTERY + [None])
        for text in BATTERY:
            with self.subTest(f"limit-7/{text!r}"):
                self.assertEqual(_slugify(text, limit=7),
                                 _old_music_slugify(text, limit=7))

    def test_filesend_slug_parity(self):
        from nomorals.tools.filesend import _slug

        for name in BATTERY + [None]:
            for fallback in ("file", "report"):
                with self.subTest(f"{name!r}/{fallback}"):
                    new = _slug(name, fallback)
                    if not (name or "").strip("-. \t\n") or not re.sub(
                            r"[^A-Za-z0-9._-]+", "-", (name or "")).strip("-."):
                        # empty after slugging: both sides produce a timestamped
                        # fallback — compare the shape, not the clock
                        self.assertRegex(new, rf"{fallback}-\d+")
                        self.assertRegex(_old_filesend_slug(name, fallback),
                                         rf"{fallback}-\d+")
                    else:
                        self.assertEqual(new, _old_filesend_slug(name, fallback),
                                         f"diverged on {name!r}")
        # case is preserved and dots/underscores survive
        self.assertEqual(_slug("My File.PDF", "file"), "My-File.PDF")
        self.assertEqual(_slug("_foo_", "file"), "_foo_")
        self.assertEqual(_slug("file.", "file"), "file")

    def test_appbuilder_slug_parity(self):
        from nomorals.builders.app_builder import _slug

        self._check_parity(_old_appbuilder_slug, _slug, BATTERY + [None])
        # no limit: long input is not truncated
        self.assertEqual(_slug("a" * 100), "a" * 100)

    def test_slugify_defaults(self):
        self.assertEqual(slugify("Hello World!"), "hello-world")
        self.assertEqual(slugify(""), "")
        self.assertEqual(slugify("", fallback="x"), "x")
        self.assertEqual(slugify(None, fallback="x"), "x")


if __name__ == "__main__":
    unittest.main()
