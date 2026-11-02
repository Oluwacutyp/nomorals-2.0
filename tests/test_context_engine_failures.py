"""Tail-101 pins: nomorals/context/engine.py census sites.

All four sites are held as exclusions — two fallback ladders and
two best-effort section builders, all documented in-place:

363 — compact_history(): the configured summarizer raises ->
fall back to the built-in extractive_summary. The compaction
still happens; only the summary's author changes. If the
extractive fallback itself failed, that error propagates (it
sits outside the try) — the ladder ends loud, not silent.
565 — _memory_section(): MemoryManager.build_context raises ->
the memory section is empty. Memory is explicitly best-effort
("never breaks context") and the section is load_bearing=False;
an empty section is absence, and the build continues with every
other section intact.
607 — _tools_section(): tools.prompt_listing() raises -> fall
through to the schemas rung. Ladder rung one of two.
612 — _tools_section(): tools.schemas() raises -> schemas=[]
and the ladder falls through to the plain list/dict rendering
and finally to its documented terminal rung, str(tools) —
never a fabricated tool list, never a broken build.

NOT-recording is pinned with the snapshot method: none of these
sites call a failures-ledger record function, so both ledgers
must be unchanged across each trigger.
"""
from __future__ import annotations

from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.context.engine import ContextEngine


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


_HISTORY = [("user", f"message {i} about the plan and the budget")
            for i in range(10)]


# ── 363: summarizer crash -> extractive fallback ───────────────

def test_363_summarizer_crash_falls_back_to_extractive():
    before = _snap()
    engine = ContextEngine(summarizer=_boom)
    entries, info = engine.compact_history(_HISTORY, keep_last_n=2)
    assert info["compacted"] is True
    assert len(entries) == 3  # boundary marker + 2 kept verbatim
    assert entries[0][0] == "system"
    assert "[compact boundary]" in entries[0][1]
    assert entries[-1] == _HISTORY[-1]
    assert _snap() == before


def test_363_summarizer_control_uses_configured_summary():
    engine = ContextEngine(
        summarizer=lambda text, room: "CUSTOM SUMMARY")
    entries, info = engine.compact_history(_HISTORY, keep_last_n=2)
    assert info["compacted"] is True
    assert "CUSTOM SUMMARY" in entries[0][1]


# ── 565: memory section is best-effort ─────────────────────────

def test_565_memory_crash_yields_empty_section():
    before = _snap()
    engine = ContextEngine()
    section = engine._memory_section(
        SimpleNamespace(build_context=_boom), "query")
    assert section.name == "memory"
    assert section.content == ""
    assert section.load_bearing is False
    assert _snap() == before


def test_565_memory_control_carries_text():
    engine = ContextEngine()
    section = engine._memory_section(
        SimpleNamespace(build_context=lambda q: "the facts"), "q")
    assert section.content == "the facts"


# ── 607/612: tools section fallback ladder ─────────────────────

class _ToolsListingBoom:
    def prompt_listing(self):
        raise RuntimeError("listing boom")

    def schemas(self):
        return [{"name": "search", "description": "web search"}]


class _ToolsAllBoom:
    def prompt_listing(self):
        raise RuntimeError("listing boom")

    def schemas(self):
        raise RuntimeError("schemas boom")


def test_607_listing_crash_falls_through_to_schemas():
    before = _snap()
    engine = ContextEngine()
    section = engine._tools_section(_ToolsListingBoom())
    assert "- search: web search" in section.content
    assert _snap() == before


def test_612_schemas_crash_reaches_str_terminal_rung():
    before = _snap()
    engine = ContextEngine()
    tools = _ToolsAllBoom()
    section = engine._tools_section(tools)
    # both callable rungs failed; the ladder's last resort is the
    # object's own string form — no tool list is invented.
    assert section.content == "Available tools:\n" + str(tools)
    assert _snap() == before


def test_607_listing_control_short_circuits():
    engine = ContextEngine()
    tools = SimpleNamespace(prompt_listing=lambda: "LISTING",
                            schemas=_boom)
    section = engine._tools_section(tools)
    assert section.content == "Available tools:\nLISTING"
