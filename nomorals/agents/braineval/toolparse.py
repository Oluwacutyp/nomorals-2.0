"""Devon tool-format scoring for the Tools suite.

Parsing is NOT mirrored here: responses are parsed by Devon's own
:func:`nomorals.agents.partner.tool_loop.parse_tool_calls`, so the suite
measures fluency against the exact parser the live loop runs — fenced
```tool blocks preferred, bare {"tool": ...} JSON via balanced-brace
scan as fallback; a call needs a string "tool" and an object "args".

Scoring an expected call (the recorded pack's rule, 2026-10-22): 0.5
for the exact tool name + 0.5 spread over the expected argument fields.
Args matchers per field:
  {"equals": v}        exact equality (strings compared stripped)
  {"contains": "sub"}  case-insensitive substring of the stringified value
  {"not_equals": v}    must differ (retry-with-different-args cases)
  {"nonempty": true}   present and not "" / None / empty
Args beyond the expected ones cost 0.1 each ONLY when they are not real
parameters of the tool (Devon's schemas set additionalProperties:
false, so invented args are a real — if lesser — fault; legitimate
optional params with defaults are free). Repeating a failed call with
identical args after an error observation scores 0 — Devon would stall
on it too. Answer expectations require NO parseable tool call plus
content requirements.
"""
from __future__ import annotations

from typing import Any

from ..partner.tool_loop import parse_tool_calls as _devon_parse_tool_calls

EXPECT_TYPES = ("call", "answer", "no_call")


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Devon's own parser, adapted to plain dicts: [{name, args, raw}]."""
    return [{"name": c.name, "args": c.args, "raw": c.raw}
            for c in _devon_parse_tool_calls(text)]


# ------------------------------------------------------------------ scoring

def _arg_match(actual_args: dict[str, Any], field: str, matcher: Any) -> bool:
    """All keys present in a matcher must hold (conjunction)."""
    if not isinstance(matcher, dict):
        matcher = {"equals": matcher}
    if field not in actual_args:
        return False
    actual = actual_args[field]
    if "equals" in matcher:
        want = matcher["equals"]
        if isinstance(want, str) and isinstance(actual, str):
            equal = actual.strip() == want.strip()
        else:
            equal = actual == want
        if not equal:
            return False
    if "contains" in matcher and \
            str(matcher["contains"]).lower() not in str(actual).lower():
        return False
    if "not_equals" in matcher:
        want = matcher["not_equals"]
        if isinstance(want, str) and isinstance(actual, str):
            same = actual.strip() == want.strip()
        else:
            same = actual == want
        if same:
            return False
    return not (matcher.get("nonempty")
                and (actual is None or str(actual).strip() == ""))


def _score_one_call(parsed: list[dict[str, Any]], expected: dict[str, Any],
                    used: set[int], tool_params: dict[str, set[str]] | None = None
                    ) -> tuple[float, bool, str, int]:
    """Score one expected call against unused parsed calls."""
    want_tool = expected["tool"]
    want_args: dict[str, Any] = expected.get("args", {})
    best: tuple[float, int, list[str]] | None = None
    for idx, call in enumerate(parsed):
        if idx in used:
            continue
        if call["name"] != want_tool:
            continue
        notes: list[str] = []
        if want_args:
            hits = 0
            for field, matcher in want_args.items():
                if _arg_match(call["args"], field, matcher):
                    hits += 1
                else:
                    notes.append(f"arg {field!r} mismatch "
                                 f"(got {call['args'].get(field)!r})")
            arg_score = hits / len(want_args)
        else:
            arg_score = 1.0
        # Args beyond the expected ones are only a fault when they are
        # NOT real parameters of the tool (Devon tools legitimately take
        # optional params with defaults; invented params are the fault).
        real_params = (tool_params or {}).get(want_tool, set())
        extra = [k for k in call["args"]
                 if k not in want_args and k not in real_params]
        score = 0.5 + 0.5 * arg_score - 0.1 * len(extra)
        if extra:
            notes.append(f"extra args {extra}")
        score = max(0.0, min(1.0, score))
        if best is None or score > best[0]:
            best = (score, idx, notes)
    if best is None:
        names = [c["name"] for c in parsed] or ["<none>"]
        return 0.0, False, f"no call to {want_tool!r}; calls seen: {names}", -1
    score, idx, notes = best
    passed = score >= 0.99
    detail = f"call {want_tool} score {score:.2f}"
    if notes:
        detail += "; " + "; ".join(notes)
    return score, passed, detail, idx


def _content_score(resp: str, expect: dict[str, Any]) -> tuple[float, list[str]]:
    notes: list[str] = []
    checks: list[bool] = []
    low = (resp or "").lower()
    for sub in expect.get("must_contain", []):
        ok = sub.lower() in low
        checks.append(ok)
        if not ok:
            notes.append(f"missing required content {sub!r}")
    any_items = expect.get("must_contain_any", [])
    if any_items:
        ok = any(s.lower() in low for s in any_items)
        checks.append(ok)
        if not ok:
            notes.append(f"none of {any_items} present")
    if not checks:
        return 1.0, notes
    return sum(checks) / len(checks), notes


def score_tool_case(resp: str, expect: dict[str, Any],
                    tool_params: dict[str, set[str]] | None = None
                    ) -> tuple[float, bool, str]:
    """Score a tool-suite response against its expect spec."""
    parsed = parse_tool_calls(resp)
    etype = expect.get("type")

    # Global guard: a forbidden identical repeat (error-recovery cases).
    forbid = expect.get("forbid_call_with")
    if forbid:
        for call in parsed:
            if call["name"] == forbid.get("tool") and all(
                    _arg_match(call["args"], f, {"equals": v})
                    for f, v in forbid.get("args", {}).items()):
                return 0.0, False, (f"repeated the forbidden identical call "
                                    f"{forbid.get('tool')} with same args")

    if etype == "call":
        expected_calls = expect.get("calls") or [
            {"tool": expect["tool"], "args": expect.get("args", {})}]
        used: set[int] = set()
        scores: list[float] = []
        details: list[str] = []
        all_pass = True
        for exp in expected_calls:
            s, p, d, idx = _score_one_call(parsed, exp, used, tool_params)
            if idx >= 0:
                used.add(idx)
            scores.append(s)
            details.append(d)
            all_pass = all_pass and p
        score = sum(scores) / len(scores) if scores else 0.0
        return score, all_pass and score >= 0.99, "; ".join(details)

    if etype in ("answer", "no_call"):
        if parsed:
            names = [c["name"] for c in parsed]
            return 0.0, False, (f"expected a plain answer (no tool call), "
                                f"but parsed calls: {names}")
        score, notes = _content_score(resp, expect)
        detail = f"answer content score {score:.2f}"
        if notes:
            detail += "; " + "; ".join(notes)
        return score, score >= 0.99, detail

    return 0.0, False, f"unknown expect type {etype!r}"
