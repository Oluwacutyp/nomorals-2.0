"""Spine-native memory tools + proactive recall."""

import pytest

from nomorals.tools.memory import register as register_memory_tools
from nomorals.tools.registry import ToolRegistry
from nomorals.memory.proactive import surface, distill_candidates


class _FakeRecord:
    def __init__(self, id, text, kind="fact", score=0.9, created_at="2026-10-09"):
        self.id = id
        self.text = text
        self.kind = kind
        self.score = score
        self.created_at = created_at


class _FakeRecall:
    def __init__(self, records):
        self.records = records


class _FakeMemory:
    def __init__(self):
        self.remembered = []
        self.forgotten = []

    def remember(self, text, kind="fact", importance=0.7, scope=None,
                 source=""):
        self.remembered.append({"text": text, "kind": kind})
        return f"rec-{len(self.remembered)}"

    def recall(self, query, limit=5, kind=None, scope=None, origin=None):
        return _FakeRecall([
            _FakeRecord("r1", "owner likes afrobeats", "preference", 0.9),
            _FakeRecord("r2", "unrelated trivia", "fact", 0.2),
        ])

    def forget(self, record_id):
        self.forgotten.append(record_id)
        return 1

    def find_one(self, query):
        return _FakeRecord("r1", "old text")


class _FakeCtx:
    def __init__(self):
        self.memory = _FakeMemory()


def _registry():
    r = ToolRegistry(context=_FakeCtx())
    register_memory_tools(r)
    return r


def test_all_seven_tools_registered():
    r = _registry()
    for name in ("memory_remember", "memory_recall", "memory_forget",
                 "memory_update", "memory_consolidate", "memory_timeline",
                 "memory_anticipate"):
        assert name in r._tools, name


def test_remember_roundtrip():
    r = _registry()
    out = r.call("memory_remember", text="owner likes afrobeats",
                 kind="preference", actor="owner")
    assert out.ok and out.value["ok"]


def test_forget_refuses_without_confirmation():
    r = _registry()
    out = r.call("memory_forget", record_id="r1", actor="owner")
    assert out.ok  # tool ran
    assert out.value["ok"] is False  # but refused
    assert "confirmed" in out.value["error"]
    assert r.context.memory.forgotten == []


def test_forget_works_when_confirmed():
    r = _registry()
    out = r.call("memory_forget", record_id="r1", confirmed=True,
                 actor="owner")
    assert out.ok and out.value["ok"] is True
    assert r.context.memory.forgotten == ["r1"]


def test_proactive_surface_threshold():
    mem = _FakeMemory()
    lines = surface("what music do I like?", mem, threshold=0.55)
    assert len(lines) == 1
    assert "afrobeats" in lines[0]
    # high threshold surfaces nothing
    assert surface("what music?", mem, threshold=0.99) == []


def test_proactive_surface_never_raises():
    class _Bad:
        def recall(self, *a, **k):
            raise RuntimeError("db down")
    assert surface("hello", _Bad()) == []
    assert surface("", _FakeMemory()) == []


def test_distill_skips_empty():
    assert distill_candidates("k", [], _FakeMemory()) == []
    assert distill_candidates("k", ["hi"], _FakeMemory()) == []
