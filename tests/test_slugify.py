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

import pytest

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


def _check_parity(old_fn, new_fn, battery, *, skip_none=False):
    for text in battery:
        if text is None and skip_none:
            with pytest.raises(Exception):
                old_fn(text)
            assert new_fn(text)  # shared helper returns the fallback
            continue
        assert new_fn(text) == old_fn(text), f"diverged on {text!r}"


def test_roles_slug_parity():
    from nomorals.agents.roles import _slug

    _check_parity(_old_roles_slug, _slug, BATTERY + [None], skip_none=True)
    # non-default limit honored exactly
    for text in BATTERY:
        assert _slug(text, limit=5) == _old_roles_slug(text, limit=5)


def test_reflection_slug_parity():
    from nomorals.agents.reflection import _slug

    _check_parity(_old_reflection_slug, _slug, BATTERY + [None])
    for text in BATTERY:
        assert _slug(text, limit=5) == _old_reflection_slug(text, limit=5)


def test_fanout_slug_parity():
    from nomorals.agents.fanout import _slug

    _check_parity(_old_fanout_slug, _slug, BATTERY + [None], skip_none=True)
    # underscore separator preserved
    assert _slug("Hello World!") == "hello_world"


def test_music_slugify_parity():
    from nomorals.media.music import _slugify

    _check_parity(_old_music_slugify, _slugify, BATTERY + [None])
    for text in BATTERY:
        assert _slugify(text, limit=7) == _old_music_slugify(text, limit=7)


def test_filesend_slug_parity():
    from nomorals.tools.filesend import _slug

    for name in BATTERY + [None]:
        for fallback in ("file", "report"):
            new = _slug(name, fallback)
            if not (name or "").strip("-. \t\n") or not re.sub(
                    r"[^A-Za-z0-9._-]+", "-", (name or "")).strip("-."):
                # empty after slugging: both sides produce a timestamped
                # fallback — compare the shape, not the clock
                assert re.fullmatch(rf"{fallback}-\d+", new), new
                assert re.fullmatch(rf"{fallback}-\d+",
                                    _old_filesend_slug(name, fallback))
            else:
                assert new == _old_filesend_slug(name, fallback), \
                    f"diverged on {name!r}"
    # case is preserved and dots/underscores survive
    assert _slug("My File.PDF", "file") == "My-File.PDF"
    assert _slug("_foo_", "file") == "_foo_"
    assert _slug("file.", "file") == "file"


def test_appbuilder_slug_parity():
    from nomorals.builders.app_builder import _slug

    _check_parity(_old_appbuilder_slug, _slug, BATTERY + [None])
    # no limit: long input is not truncated
    assert _slug("a" * 100) == "a" * 100


def test_slugify_defaults():
    assert slugify("Hello World!") == "hello-world"
    assert slugify("") == ""
    assert slugify("", fallback="x") == "x"
    assert slugify(None, fallback="x") == "x"
