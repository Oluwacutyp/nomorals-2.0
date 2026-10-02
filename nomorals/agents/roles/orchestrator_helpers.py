"""Shared helpers for role agents."""

from __future__ import annotations

import json
import re
from typing import Any

from ...core.text import truncate as _truncate


def truncate(text: str, limit: int = 4000) -> str:
    """Backward-compatible alias of :func:`nomorals.core.text.truncate`."""
    return _truncate(text, limit)


def parse_json_loose(text: str) -> Any:
    """Extract JSON from a model reply that may wrap it in prose or fences."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:  # noqa: E103 - probe failed; bracket-matching fallback follows
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None
