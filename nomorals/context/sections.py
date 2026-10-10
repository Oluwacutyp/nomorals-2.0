"""Sections: the unit of assembly for token-budgeted context.

A :class:`Section` is one named slab of a prompt (system instructions, mission
state, tool manifests, history, ...).  Sections carry the metadata the budget
and compressor need to make load-aware decisions: a priority, a load-bearing
flag, and ``keep`` — verbatim content that must survive compression.

Sections also carry a ``volatile`` flag: stable sections (system, tools) form
the cache-friendly prefix of an assembled prompt; volatile sections (history)
belong at the tail.  Providers cache on exact byte prefixes, so keeping the
stable prefix byte-identical across calls is what turns cache misses into
hits.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable

from ..core.text import approx_token_count

__all__ = [
    "Section",
    "SECTION_PRIORITIES",
    "priority_for",
    "set_token_counter",
    "token_count",
    "try_tiktoken_counter",
]


#: Canonical assembly order is this dict's insertion order.  Higher number =
#: more important; the budget drops low-priority sections first.
SECTION_PRIORITIES: dict[str, float] = {
    "system": 100.0,
    "mission": 90.0,
    "artifacts": 70.0,
    "tools": 60.0,
    "project": 40.0,
    "user_profile": 30.0,
    "memory": 25.0,
    "history": 20.0,
}


def priority_for(name: str, default: float = 50.0) -> float:
    """Priority for a section name, falling back to ``default``."""
    return SECTION_PRIORITIES.get(name, default)


#: Module-level token counter hook.  The default is the cheap estimator;
#: install an exact counter (tiktoken) with :func:`set_token_counter`.
_token_counter: Callable[[str], int] | None = None


def set_token_counter(fn: Callable[[str], int] | None) -> None:
    """Install a module-wide token counter (e.g. tiktoken).  ``None`` resets."""
    global _token_counter
    _token_counter = fn


def token_count(text: str) -> int:
    """Token count via the installed counter, falling back to the estimator."""
    if _token_counter is not None:
        try:
            return int(_token_counter(text))
        except Exception:  # noqa: BLE001 - counter must never break assembly
            pass
    return approx_token_count(text)


def try_tiktoken_counter(model: str = "gpt-4o") -> Callable[[str], int] | None:
    """Return a tiktoken-based counter, or ``None`` when tiktoken is absent."""
    try:
        import tiktoken  # type: ignore[import]

        enc = tiktoken.encoding_for_model(model)
    except Exception:  # noqa: BLE001 - tiktoken is optional
        return None

    def _count(text: str) -> int:
        return len(enc.encode(text))

    return _count


_DENSE_TOKEN_RE = re.compile(r"[A-Za-z]*\d[\w\-/.:]*|[A-Z]{2,}|`[^`]+`")


@dataclass
class Section:
    """One named slab of assembled context.

    ``load_bearing`` sections (acceptance criteria, active mission state,
    safety-of-state) are never dropped by the budget and their ``keep``
    content is never silently removed by the compressor: if it cannot fit,
    the section is marked ``truncated`` explicitly rather than vanishing.

    ``volatile`` sections change every call (history); stable sections
    (system, tools) form the cache-friendly prefix.  The engine's
    ``ordering="cache"`` mode puts stable sections first and the volatile
    tail last so provider prompt-caches keep hitting.
    """

    name: str
    content: str = ""
    priority: float = 50.0
    load_bearing: bool = False
    keep: tuple[str, ...] = ()
    truncated: bool = False
    dropped: bool = False
    volatile: bool = False

    #: free-form provenance for this section (artifact ids, tool names, ...).
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return token_count(self.content)

    def keep_text(self) -> str:
        return "\n".join(k for k in self.keep if k)

    def fingerprint(self) -> str:
        """Stable sha256 of the content: byte drift breaks prompt caches."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]

    def density(self) -> float:
        """Information density: identifier/number/code tokens per 100 tokens.

        A cheap proxy for the "facts per token" utility score used by
        context-optimization frameworks: sections dense in identifiers,
        numbers, and code carry more signal per token.
        """
        tokens = self.tokens
        if tokens <= 0 or not self.content:
            return 0.0
        dense = len(_DENSE_TOKEN_RE.findall(self.content))
        return round(100.0 * dense / max(1, tokens), 2)

    def render(self) -> str:
        """The section as it appears in the final prompt (markdown style)."""
        title = self.name.replace("_", " ").title()
        return f"## {title}\n{self.content}" if self.content else f"## {title}\n(empty)"

    def render_tagged(self) -> str:
        """The section wrapped in XML tags (Anthropic-style structure)."""
        body = self.content if self.content else "(empty)"
        return f'<section name="{self.name}">\n{body}\n</section>'

    def summary_line(self) -> str:
        """One-line status summary for dashboards and reports."""
        flags = []
        if self.load_bearing:
            flags.append("pinned")
        if self.truncated:
            flags.append("truncated")
        if self.dropped:
            flags.append("dropped")
        if self.volatile:
            flags.append("volatile")
        flag_text = f" [{', '.join(flags)}]" if flags else ""
        return f"{self.name}: ~{self.tokens} tokens{flag_text}"
