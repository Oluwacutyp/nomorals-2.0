"""Tail-107 pins: nomorals/books/fiction.py census sites
(1390, 1703, 1771).

All three sites are held as EXCLUSIONS.

1390 — ``list_stories`` per-story guard: one story whose
``state.json`` is corrupt (or whose state fails to rebuild) is
skipped so the healthy stories still list. This is the standard
per-record parse guard: nothing is fabricated (the story is
OMITTED, never described), nothing is rewritten, and the direct
path stays honest — ``load(slug)`` on the same corrupt story
raises, pinned as the control. The consumer is the books tool
surface's story listing (``books/tools.py``). Observability
sketch (shelved): the skip is silent while the module logs other
degradations at debug; a ``_log.debug`` on the skip would make a
vanishing story diagnosable.

1703 — ``_model_chapter`` import guard: importing the LLM stack
(``llm.base`` / ``llm.brain`` / ``books.write``) raising returns
``""`` — byte-identical to the function's own legitimate
no-model return a line later (``model_available`` False -> ``""``)
and to its model-failure return at the end. ``""`` is the module's
"no model prose" signal; the caller (``write_next``) falls back
to the heuristic composer, so a broken LLM import degrades the
chapter to the same composer path as an unavailable model.
Pinned: import failure -> ``""``, and the healthy-import /
no-model control -> the identical ``""``.

1771 — ``_repair``: the same import guard, the same ``""``
contract. A repair that cannot run leaves the original chapter
text in place (the caller keeps ``text`` when the repair returns
falsy) — degradation to the unrepaired chapter, never a fabricated
"repaired" text. Pinned both ways like 1703.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.books import write as writemod
from nomorals.books.fiction import FictionWriter, StoryState


@pytest.fixture()
def writer(tmp_path):
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    return FictionWriter(ctx)


# ── 1390: corrupt story skipped in the listing; load stays honest ─

def test_1390_corrupt_story_skipped_healthy_listed(writer):
    good = StoryState(slug="good-one", title="Good", premise="p",
                      genre="mystery", mode="novel", total_planned=5)
    writer._save_state(good)
    broken_dir = writer.story_dir("broken-one")
    (broken_dir / "state.json").write_text("{corrupt json",
                                           encoding="utf-8")
    stories = writer.list_stories()
    assert [s["slug"] for s in stories] == ["good-one"]
    assert stories[0]["title"] == "Good"


def test_1390_control_direct_load_of_corrupt_story_raises(writer):
    broken_dir = writer.story_dir("broken-one")
    (broken_dir / "state.json").write_text("{corrupt json",
                                           encoding="utf-8")
    with pytest.raises(Exception) as ei:
        writer.load("broken-one")
    assert isinstance(ei.value, Exception)


def test_1390_control_healthy_story_loads(writer):
    good = StoryState(slug="good-one", title="Good", premise="p",
                      genre="mystery", mode="novel", total_planned=5)
    writer._save_state(good)
    assert writer.load("good-one").title == "Good"


# ── 1703/1771: LLM import failure == the no-model "" ──────────────

def test_1703_import_failure_returns_empty(writer):
    with mock.patch.dict(sys.modules, {"nomorals.llm.brain": None}):
        assert writer._model_chapter(object(), object(), object()) == ""


def test_1703_control_no_model_returns_same_empty(writer):
    with mock.patch.object(writemod, "model_available",
                           return_value=False):
        assert writer._model_chapter(object(), object(), object()) == ""


def test_1771_import_failure_returns_empty(writer):
    with mock.patch.dict(sys.modules, {"nomorals.llm.brain": None}):
        assert writer._repair(object(), object(), object(),
                              "chapter text", ["problem"]) == ""


def test_1771_control_no_model_returns_same_empty(writer):
    with mock.patch.object(writemod, "model_available",
                           return_value=False):
        assert writer._repair(object(), object(), object(),
                              "chapter text", ["problem"]) == ""
