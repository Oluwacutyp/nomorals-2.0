"""Innate immunity — fast, pattern-based, no learning.

The innate layer is the skin and the reflexes: compiled-regex scanning
against injected pattern data. It never learns, never calls a model,
never touches the network. Two entry points share one engine:

* :func:`scan_content` — scan text against caller-supplied patterns.
* :func:`evaluate_live` — the red-team catalog's patterns evaluated
  LIVE on inbound content, not just in CI. The catalog is injected as
  plain data (a list of ``{id, class_id, pattern, severity}`` dicts);
  this module never imports the catalog.

Barriers are DATA, not implementation: the list of structural
defenses and what each one blocks, for callers to consult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

__all__ = [
    "BARRIERS",
    "InnateHit",
    "scan_content",
    "evaluate_live",
]


# ── barriers as data ───────────────────────────────────────────────────
# Names + what they block. No implementation lives here: each barrier is
# enforced by its owning organ; the immune system only names them so a
# threat report can say which barrier should have held.

BARRIERS: list[dict[str, str]] = [
    {
        "id": "execbox",
        "name": "Execbox sandbox",
        "blocks": "Untrusted code execution: tool / shell / code runs outside "
        "the sandbox, or sandbox escape attempts.",
    },
    {
        "id": "capability-grants",
        "name": "Capability grants",
        "blocks": "Use of a capability without a live grant: ungranted tool "
        "calls, replayed or forged grant tokens.",
    },
    {
        "id": "layer-discipline",
        "name": "Layer discipline",
        "blocks": "Upward imports / layer violations: a lower layer reaching "
        "into a higher one, bypassing the architecture.",
    },
    {
        "id": "exif-stripping",
        "name": "EXIF stripping",
        "blocks": "Location / device metadata leaking out in shared images: "
        "EXIF, GPS tags, and other embedded metadata.",
    },
    {
        "id": "dnsleak",
        "name": "DNS-leak protection",
        "blocks": "DNS queries escaping the tunnel / proxy and exposing "
        "which hosts the system resolves.",
    },
    {
        "id": "netleak",
        "name": "Net-leak protection",
        "blocks": "Non-proxied network egress: direct connections that "
        "bypass the proxy pool and expose the real origin.",
    },
]


@dataclass
class InnateHit:
    pattern_id: str
    class_id: str
    span: tuple[int, int]
    severity: str = "medium"
    matched_text: str = ""


@lru_cache(maxsize=512)
def _compile(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


def _pattern_fields(pattern_obj: Any) -> tuple[str, str, str, str]:
    """Normalize a pattern given as a dict or an object with attributes.

    Returns (pattern_id, class_id, pattern, severity).
    """
    if isinstance(pattern_obj, dict):
        return (
            str(pattern_obj.get("id") or pattern_obj.get("pattern_id") or "unknown"),
            str(pattern_obj.get("class_id") or pattern_obj.get("class") or "unknown"),
            str(pattern_obj.get("pattern") or ""),
            str(pattern_obj.get("severity") or "medium"),
        )
    return (
        str(
            getattr(pattern_obj, "id", None)
            or getattr(pattern_obj, "pattern_id", None)
            or "unknown"
        ),
        str(
            getattr(pattern_obj, "class_id", None)
            or getattr(pattern_obj, "pathogen_class", None)
            or "unknown"
        ),
        str(getattr(pattern_obj, "pattern", "") or ""),
        str(getattr(pattern_obj, "severity", None) or "medium"),
    )


def scan_content(text: str, patterns: Iterable[Any]) -> list[InnateHit]:
    """Scan ``text`` against ``patterns`` with compiled regexes.

    Every pattern that matches yields one :class:`InnateHit` per
    match span. Invalid regexes are skipped (a broken pattern must not
    take down the whole scan); callers that need strictness should
    validate patterns up front. No learning happens here.
    """
    hits: list[InnateHit] = []
    haystack = text or ""
    for pattern_obj in patterns or []:
        pattern_id, class_id, pattern, severity = _pattern_fields(pattern_obj)
        if not pattern:
            continue
        try:
            compiled = _compile(pattern)
        except re.error:
            continue
        for match in compiled.finditer(haystack):
            hits.append(
                InnateHit(
                    pattern_id=pattern_id,
                    class_id=class_id,
                    span=(match.start(), match.end()),
                    severity=severity,
                    matched_text=match.group(0),
                )
            )
    return hits


def evaluate_live(content: str, catalog_patterns: Iterable[Any]) -> list[InnateHit]:
    """Evaluate red-team catalog patterns LIVE on inbound content.

    ``catalog_patterns`` is injected data — a list of
    ``{id, class_id, pattern, severity}`` dicts, e.g. materialized from
    ``agents/redteam_catalog.py`` by the caller. The catalog is never
    imported here, so live evaluation stays a pure function of the
    content and the patterns the caller chose to inject.
    """
    return scan_content(content, catalog_patterns)
