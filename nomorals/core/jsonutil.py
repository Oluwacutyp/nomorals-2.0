"""JSON recovery primitives: pull structured data out of messy model replies.

Shared by the agent layer (reasoning, orchestrator, subagents, brief,
devon, evolution) so every agent parses LLM JSON with the same discipline.
Pure stdlib — safe for L1 (core) with zero intra-project imports.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

__all__ = ["extract_json", "parse_lenient"]


def extract_json(text: str) -> Optional[Any]:
    """Pull the first balanced JSON object/array out of a model reply.

    Handles raw JSON, ```json fences, and JSON embedded in prose.
    Returns None when nothing parseable is there.

    Canonical unification of the six ``_extract_json`` copies that used to
    live across the agents layer; this was the most robust of them
    (formerly ``nomorals.agents.reasoning._extract_json``).
    """
    if not text:
        return None
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start = 0
        while True:
            start = text.find(opener, start)
            if start == -1:
                break
            end = _balanced_end(text, start, opener, closer)
            if end is None:
                break
            candidates.append(text[start:end + 1])
            # keep scanning past this candidate so a malformed one does
            # not hide a valid JSON blob later in the same reply
            start = end + 1
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    """Index of the closer matching text[start], or None if unbalanced.

    String-aware: braces inside quoted values do not count.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return i
    return None


#: String-aware scanner states for the lenient pass.
def _strip_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments outside string literals."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _remove_trailing_commas(text: str) -> str:
    """Delete commas that directly precede ``}`` or ``]`` (outside strings)."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1  # drop the comma
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _single_to_double_quotes(text: str) -> str:
    """Convert single-quoted strings to double-quoted (outside strings)."""
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    escape = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue
        if c == "'":
            # scan a single-quoted string; give up (leave as-is) on newline
            j = i + 1
            buf: list[str] = []
            ok = True
            while j < n:
                cj = text[j]
                if cj == "\\" and j + 1 < n:
                    buf.append(text[j:j + 2])
                    j += 2
                    continue
                if cj == "'":
                    break
                if cj == "\n":
                    ok = False
                    break
                buf.append('"' if cj == '"' else cj)
                j += 1
            else:
                ok = False
            if ok:
                out.append('"' + "".join(buf) + '"')
                i = j + 1
                continue
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def parse_lenient(text: str) -> Optional[Any]:
    """Parse JSON that is *almost* valid — the model-reply reality.

    Repair ladder (first success wins):
    1. strict :func:`extract_json`
    2. strip ``//`` and ``/* */`` comments
    3. remove trailing commas
    4. convert single-quoted strings
    5. quote unquoted object keys (``{key: 1}``)
    6. Python literals (``True``/``None``) via :mod:`ast`

    Returns ``None`` when nothing works. Never raises.
    """
    if not text:
        return None
    hit = extract_json(text)
    if hit is not None:
        return hit
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    blob = fence.group(1).strip() if fence else text.strip()
    candidates = [blob]
    try:
        step = _strip_comments(blob)
        candidates.append(step)
        step = _remove_trailing_commas(step)
        candidates.append(step)
        step = _single_to_double_quotes(step)
        candidates.append(step)
        # unquoted keys: {key: 1, other: "x"} — outside strings only
        step = re.sub(r'(?<=[{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:',
                      r'"\1":', step)
        candidates.append(step)
    except Exception:  # noqa: BLE001 - repairs are best-effort
        pass
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    # last resort: Python literal syntax
    try:
        import ast as _ast

        return _ast.literal_eval(blob)
    except Exception:  # noqa: BLE001
        return None
