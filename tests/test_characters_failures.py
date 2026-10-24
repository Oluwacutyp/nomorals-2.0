"""Pass #101: character failures stop vanishing.

The character engine is best-effort by design — prompt building, context
assembly, dialogue, matching, session processing, and store I/O all fall
back to safe defaults instead of raising. These pins prove the fallback
half now leaves a trace: every formerly silent handler records into the
characters failure ledger while returning exactly what it returned
before.
"""

import json
import tempfile
from pathlib import Path

import pytest

from nomorals.characters import Character, CharacterStore
from nomorals.characters import dialogue as dialogue_mod
from nomorals.characters import processing as processing_mod
from nomorals.characters.failures import (
    clear_failures, failure_count, format_failures, recent_failures,
    record_failure,
)


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(100)]


def _make_store():
    return CharacterStore(data_dir=Path(tempfile.mkdtemp()))


# ── store ─────────────────────────────────────────────────────────────────


def test_store_get_corrupt_file_records_and_returns_none():
    store = _make_store()
    c = Character(name="Zara", persona={"witty": 0.9})
    store.save(c)
    (store.dir / f"{c.id}.json").write_text("{not json", encoding="utf-8")
    assert store.get(c.id) is None
    assert "store.get:from_dict" in _sites()


def test_store_all_skips_corrupt_file_and_records():
    store = _make_store()
    good = Character(name="Good", persona={"warm": 0.8})
    store.save(good)
    bad = Character(name="Bad", persona={"cold": 0.8})
    store.save(bad)
    (store.dir / f"{bad.id}.json").write_text("{not json", encoding="utf-8")
    names = [c.name for c in store.all()]
    assert names == ["Good"]
    assert "store.all:append" in _sites()


def test_import_all_save_failure_records_and_continues():
    store = _make_store()
    kept = Character(name="Kept", persona={"warm": 0.8})
    store.save(kept)
    orig_save = store.save

    def boom_save(c):
        if c.name == "Boom":
            raise RuntimeError("disk full")
        return orig_save(c)

    store.save = boom_save  # type: ignore[method-assign]
    stats = store.import_all([
        {"name": "Boom", "persona": {"bold": 0.5}},
        {"name": "Safe", "persona": {"calm": 0.5}},
    ])
    assert stats["created"] == 2
    assert store.get_by_name("Safe") is not None
    assert "store.import_all:save" in _sites()


# ── session processing ────────────────────────────────────────────────────


def test_process_session_step_failure_records_and_processing_continues():
    from nomorals.characters.processing import SessionEvent
    from nomorals.characters.relationships import RelationshipGraph

    def boom(*args, **kwargs):
        raise RuntimeError("arc engine down")

    old = processing_mod.grow_from_interaction
    processing_mod.grow_from_interaction = boom
    try:
        c = Character(name="Zara", persona={"witty": 0.9})
        graph = RelationshipGraph()
        out = processing_mod.process_session(
            [SessionEvent("zara", "devon", "chat")],
            {"zara": c}, graph)
    finally:
        processing_mod.grow_from_interaction = old
    assert out["processed"] == 1
    assert "processing.process_session:grow_from_interaction" in _sites()


# ── dialogue / speak fallback ─────────────────────────────────────────────


def test_speak_suggest_failure_records_and_falls_back():
    c = Character(name="Zara", persona={"witty": 0.9})

    def boom_suggest(prompt):
        raise RuntimeError("model down")

    line = c.speak("hello there", boom_suggest)
    assert line  # model-free fallback line, never empty
    assert "character.speak:strip" in _sites()


def test_brain_say_failure_records_and_returns_devon_prefix():
    def boom_suggest(prompt):
        raise RuntimeError("model down")

    out = dialogue_mod._brain_say("You are Devon.", "context here", boom_suggest)
    assert out.startswith("[Devon] ")
    assert "dialogue._brain_say:strip" in _sites()


# ── ledger mechanics ──────────────────────────────────────────────────────


def test_ledger_bounded_but_count_total():
    for i in range(250):
        record_failure(f"test.site{i % 3}", ValueError(f"boom {i}"))
    assert len(recent_failures(1000)) == 200
    assert failure_count() == 250


def test_format_empty_and_populated():
    assert format_failures() == "no character failures recorded."
    record_failure("store.get:from_dict", ValueError("bad json"))
    out = format_failures()
    assert "character failures (1 total" in out
    assert "store.get:from_dict: ValueError: bad json" in out


def test_tool_failures_action_renders_ledger():
    captured = {}

    class FakeRegistry:
        def register(self, name, **kwargs):
            def deco(fn):
                captured[name] = fn
                return fn
            return deco

    from nomorals.tools.characters import register
    register(FakeRegistry())
    fn = captured["characters"]
    assert fn(action="failures") == "no character failures recorded."
    record_failure("store.get:from_dict", ValueError("bad json"))
    out = fn(action="failures")
    assert "character failures (1 total" in out
    assert "store.get:from_dict" in out


def test_error_scan_characters_clean():
    from nomorals.tools.error_scan import scan
    report = scan(["nomorals/characters"])
    assert report.findings == []
