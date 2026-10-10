"""Prompt-cache-aware assembly planning.

Providers cache on exact byte prefixes: everything up to the first changed
byte is reprocessed.  The discipline is *stable prefix first, volatile tail
last* — system and tool definitions up front, per-turn history and fresh
results at the back — with cache breakpoints at the end of each large stable
block (most providers allow ~4).

This module plans that layout: :func:`cache_plan` splits an assembled
section list into a stable prefix and a volatile tail and suggests where
the breakpoints go; :func:`audit_prefix_stability` scans the stable prefix
for the usual cache-killers (timestamps, request/session ids, UUIDs,
trailing-whitespace drift) that silently invalidate caches even when the
prompt "looks" stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .sections import Section, token_count

__all__ = [
    "CachePlan",
    "StabilityWarning",
    "cache_plan",
    "audit_prefix_stability",
    "stable_fingerprint",
]

#: Patterns that almost always mean "this changes every call" and therefore
#: must not live in the stable prefix.
_VOLATILE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("timestamp", re.compile(
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?|\b\d{10}(\.\d+)?\b"
        r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        re.IGNORECASE)),
    ("uuid", re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE)),
    ("request/session id", re.compile(
        r"\b(?:request|session|trace|run|call)[ _-]?(?:id|key)\b\s*[:=]\s*[\w-]+",
        re.IGNORECASE)),
    ("working directory", re.compile(
        r"\b(?:cwd|working dir(?:ectory)?|pwd)\b\s*[:=]\s*\S+", re.IGNORECASE)),
]


@dataclass
class StabilityWarning:
    section: str
    pattern: str
    sample: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "pattern": self.pattern,
            "sample": self.sample,
        }

    def __str__(self) -> str:
        return (
            f"section '{self.section}' looks volatile "
            f"({self.pattern}): {self.sample!r}"
        )


@dataclass
class CachePlan:
    """Where the cache breakpoints should go for one assembled prompt."""

    #: Section names forming the byte-stable prefix, in order.
    stable_prefix: list[str] = field(default_factory=list)
    #: Section names forming the volatile tail, in order.
    volatile_tail: list[str] = field(default_factory=list)
    #: Section names after which a cache breakpoint belongs (max ~4).
    breakpoints: list[str] = field(default_factory=list)
    #: Estimated cacheable tokens (sum of the stable prefix).
    cacheable_tokens: int = 0
    #: Total tokens across all sections.
    total_tokens: int = 0
    #: Warnings from the prefix-stability audit.
    warnings: list[StabilityWarning] = field(default_factory=list)

    @property
    def cacheable_ratio(self) -> float:
        if not self.total_tokens:
            return 0.0
        return self.cacheable_tokens / self.total_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_prefix": self.stable_prefix,
            "volatile_tail": self.volatile_tail,
            "breakpoints": self.breakpoints,
            "cacheable_tokens": self.cacheable_tokens,
            "total_tokens": self.total_tokens,
            "cacheable_ratio": round(self.cacheable_ratio, 3),
            "warnings": [w.to_dict() for w in self.warnings],
        }

    def render(self, *, style: str = "plain") -> str:
        fancy = style == "fancy"
        lines = ["Cache plan"]
        lines.append(
            f"  cacheable: ~{self.cacheable_tokens} of ~{self.total_tokens} "
            f"tokens ({self.cacheable_ratio:.0%})"
        )
        mark = "◆" if fancy else "*"
        for name in self.stable_prefix:
            bp = f"  [{mark} breakpoint]" if name in self.breakpoints else ""
            lines.append(f"  [stable]   {name}{bp}")
        for name in self.volatile_tail:
            lines.append(f"  [volatile] {name}")
        if self.warnings:
            lines.append("  warnings:")
            for warning in self.warnings:
                lines.append(f"    ! {warning}")
        elif fancy:
            lines.append("  ✓ prefix looks stable")
        else:
            lines.append("  prefix looks stable")
        return "\n".join(lines)


def stable_fingerprint(sections: list[Section]) -> str:
    """Fingerprint of the stable prefix: drift here means cache misses.

    Compare fingerprints across turns; any change means every downstream
    byte is reprocessed by the provider.
    """
    stable = [s for s in sections if not s.volatile and not s.dropped]
    joined = "\x00".join(f"{s.name}:{s.fingerprint()}" for s in stable)
    import hashlib

    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def audit_prefix_stability(sections: list[Section]) -> list[StabilityWarning]:
    """Scan stable sections for per-call volatile content.

    Returns one warning per (section, pattern) hit.  Volatile sections are
    skipped — they are *supposed* to change.
    """
    warnings: list[StabilityWarning] = []
    for section in sections:
        if section.volatile or section.dropped or not section.content:
            continue
        for label, pattern in _VOLATILE_PATTERNS:
            match = pattern.search(section.content)
            if match:
                sample = match.group(0)
                if len(sample) > 60:
                    sample = sample[:57] + "…"
                warnings.append(StabilityWarning(
                    section=section.name, pattern=label, sample=sample))
    return warnings


def cache_plan(
    sections: list[Section],
    *,
    max_breakpoints: int = 4,
) -> CachePlan:
    """Plan cache breakpoints for an assembled section list.

    The stable prefix is every non-volatile, non-dropped section in order;
    the volatile tail is the rest.  Breakpoints land at the end of the
    largest stable blocks (never more than ``max_breakpoints``).
    """
    survivors = [s for s in sections if not s.dropped]
    stable = [s for s in survivors if not s.volatile]
    volatile = [s for s in survivors if s.volatile]

    cacheable = sum(s.tokens for s in stable)
    total = sum(s.tokens for s in survivors)

    # Breakpoints after the largest stable blocks, in section order.
    largest = sorted(stable, key=lambda s: s.tokens, reverse=True)
    chosen = {s.name for s in largest[: max(0, max_breakpoints)]}
    breakpoints = [s.name for s in stable if s.name in chosen]

    return CachePlan(
        stable_prefix=[s.name for s in stable],
        volatile_tail=[s.name for s in volatile],
        breakpoints=breakpoints,
        cacheable_tokens=cacheable,
        total_tokens=total,
        warnings=audit_prefix_stability(sections),
    )
