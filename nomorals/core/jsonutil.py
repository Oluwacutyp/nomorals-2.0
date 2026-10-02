"""JSON recovery primitives: pull structured data out of messy model replies.

Shared by the agent layer (reasoning, orchestrator, subagents, brief,
devon, evolution) so every agent parses LLM JSON with the same discipline.
Pure stdlib — safe for L1 (core) with zero intra-project imports.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

__all__ = ["extract_json"]


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
