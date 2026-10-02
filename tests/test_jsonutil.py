"""Parity tests for the unified ``nomorals.core.jsonutil.extract_json``.

Six ``_extract_json`` copies used to live across the agents layer
(reasoning, orchestrator, subagents, brief, devon, evolution). They are now
thin aliases/delegates of one canonical implementation — reasoning's, the
most robust of the six.

This file embeds the five *replaced* implementations verbatim as fixtures
(reasoning's old copy was byte-identical to the canonical one) and runs a
shared battery through old and new. Where an old copy agreed with the
canonical behavior we assert ``new == old``; where they diverged we assert
the canonical (more correct) behavior and document which copy differed and
why the canonical behavior won.
"""

from __future__ import annotations

import json
import re

import pytest

from nomorals.core.jsonutil import extract_json


# ── verbatim fixtures of the replaced implementations ──────────────────────


def _old_orchestrator(text):
    """nomorals/agents/orchestrator.py (module-level)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:  # noqa: E103 - falls through to brace-extraction fallback
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _old_subagents(text):
    """nomorals/agents/subagents.py (module-level)."""
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()
    try:
        parsed = json.loads(candidate)
    except ValueError:
        parsed = None  # fall through to the balanced-brace scan below
    if parsed is not None:
        return parsed
    start = candidate.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start:i + 1])
                except ValueError:
                    return None
    return None


def _old_brief(text):
    """nomorals/agents/brief.py (BriefAgent method). Identical twin in devon.py."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _old_evolution(text):
    """nomorals/agents/evolution.py (EvolutionAgent method)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


OLD_IMPLS = {
    "orchestrator": _old_orchestrator,
    "subagents": _old_subagents,
    "brief": _old_brief,
    "devon": _old_brief,  # devon's copy was identical to brief's
    "evolution": _old_evolution,
}

# DIVERGENCES documents every battery input where an old implementation
# disagreed with the canonical behavior, with the rationale for the winner.
#
# - orchestrator: grabbed first-"{" to last-"}", so two JSON blobs (or one
#   blob plus prose containing braces) collapsed into unparseable mush and it
#   gave up. Canonical wins: first *valid balanced* blob.
# - orchestrator/subagents: the whole-text ``json.loads`` fallback returned
#   bare scalars (42, "str", true) and orchestrator crashed on None.
#   Canonical wins: objects/arrays only, None-safe — the contract the other
#   four copies already honored.
# - brief/devon: brace scan was not string-aware, so a "}" (or "{") inside a
#   quoted value ended the scan early and the candidate failed to parse.
#   Canonical wins: string-aware scan recovers the real object.
# - subagents/brief/devon/evolution: objects only — a *bare* top-level JSON
#   array ("[1, 2, 3]") returned None. Canonical wins: arrays are first-class,
#   like reasoning's original. (When an object sits *inside* an array in
#   prose, the canonical brace-first scan returns the first balanced object
#   that parses — reasoning's long-standing behavior, which five modules
#   already depend on; unification preserves it rather than redesigning it.)
# - orchestrator/subagents/brief/devon/evolution: gave up after the first
#   balanced candidate failed to parse. Canonical wins: keeps scanning so a
#   malformed blob does not hide a valid one later in the reply.
_CRASH = object()  # sentinel: the old impl raised instead of returning

BATTERY = [
    # (input, canonical expected, {old_name: old output when it differed})
    ("", None, {}),
    ("   ", None, {}),
    ("{}", {}, {}),
    ('{"a": 1}', {"a": 1}, {}),
    ("```json\n{\"a\": 1}\n```", {"a": 1}, {}),
    ("```\n{\"a\": 1}\n```", {"a": 1}, {}),
    ('```{"a": 1}```', {"a": 1}, {}),
    ('Here is the plan: {"steps": [1, 2]} done', {"steps": [1, 2]}, {}),
    ('prefix {"a": 1} middle {"b": 2} suffix', {"a": 1}, {"orchestrator": None}),
    ('{"a": 1}\n{"b": 2}', {"a": 1}, {"orchestrator": None}),
    ('{"msg": "a}b"}', {"msg": "a}b"}, {"brief": None, "devon": None}),
    ('{"a": "{", "b": 1}', {"a": "{", "b": 1}, {"brief": None, "devon": None}),
    ("[1, 2, 3]", [1, 2, 3],
     {"brief": None, "devon": None, "evolution": None}),
    ('[{"a": 1}]', {"a": 1},
     {"orchestrator": [{"a": 1}], "subagents": [{"a": 1}]}),
    ('items: [{"x": 1}, {"x": 2}]', {"x": 1}, {"orchestrator": None}),
    ('{"a": 1} trailing garbage', {"a": 1}, {}),
    ('{"unclosed": true', None, {}),
    ("no json here", None, {}),
    ('{"a": }', None, {}),
    ('{"a": } then {"b": 2}', {"b": 2},
     {"orchestrator": None, "subagents": None, "brief": None,
      "devon": None, "evolution": None}),
    ("42", None, {"orchestrator": 42, "subagents": 42}),
    ('"just a string"', None,
     {"orchestrator": "just a string", "subagents": "just a string"}),
    ("true", None, {"orchestrator": True, "subagents": True}),
    ('{"nested": {"deep": [1, {"x": 2}]}}',
     {"nested": {"deep": [1, {"x": 2}]}}, {}),
    ('  {"a": 1}  ', {"a": 1}, {}),
    ('{"q": "say \\"hi\\""}', {"q": 'say "hi"'}, {}),
    ('Some intro ```json\n{"a": 1}\n``` outro', {"a": 1}, {}),
    ("```json\n{\"a\": 1", None, {}),
    (None, None, {"orchestrator": _CRASH}),
]


@pytest.mark.parametrize("text,expected,div", BATTERY,
                         ids=[f"case-{i}" for i in range(len(BATTERY))])
def test_canonical_ground_truth(text, expected, div):
    """The canonical implementation returns the documented result."""
    assert extract_json(text) == expected


@pytest.mark.parametrize("name", sorted(OLD_IMPLS))
@pytest.mark.parametrize("text,expected,div", BATTERY,
                         ids=[f"case-{i}" for i in range(len(BATTERY))])
def test_parity_with_old_implementation(name, text, expected, div):
    """New behavior matches each old copy wherever the old copy was right,
    and the documented divergences are exactly the enumerated ones."""
    old = OLD_IMPLS[name]
    if div.get(name) is _CRASH:
        with pytest.raises(Exception):
            old(text)
        assert extract_json(text) == expected  # canonical is None-safe
        return
    got = old(text)
    if name in div:
        assert got == div[name], f"{name} fixture drifted on {text!r}"
    else:
        assert got == expected, f"{name} unexpectedly diverged on {text!r}"
    assert extract_json(text) == expected


def test_reasoning_alias_is_canonical():
    """reasoning._extract_json is the canonical function itself (its old
    module-level copy was the unification base), so its five external
    importers (task_type, reflection, structuring, toolmaker, benchmark)
    are unaffected."""
    from nomorals.agents.reasoning import _extract_json
    assert _extract_json is extract_json


def test_delegate_methods_call_canonical():
    """The three former method copies now delegate to the shared function."""
    from nomorals.agents.brief import BriefAgent
    from nomorals.agents.devon import DevonAgent
    from nomorals.agents.evolution import EvolutionAgent

    for cls in (BriefAgent, DevonAgent, EvolutionAgent):
        assert cls._extract_json('{"ok": true}') == {"ok": True}
        assert cls._extract_json("nothing here") is None
        assert cls._extract_json('x {"k": "v}"} y') == {"k": "v}"}
