"""God-tier result presentation for federated search.

``SearchResponse.to_dict()`` is the machine surface; this module is the
*human* surface. Mined from the search-UX canon (count first, match
highlighting, source badges, metadata per hit, grouped sections, and a
designed no-results state) and from colorized terminal search CLIs like
``searxngr``.

Styles:

- ``rich`` — ANSI color: source badges, bold highlighted query matches,
  dim metadata rail; no-TTY safe (degrades to ``plain`` automatically).
- ``compact`` — one line per hit: ``[badge] title — url``.
- ``plain`` — no color codes at all (pipes, logs, bots).
- ``markdown`` — chat/markdown rendering: headers, bold matches, links.

``group_by_source=True`` sections hits under per-source headers instead
of one merged list (useful when comparing what each subsystem knew).
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "render_search",
    "highlight_matches",
    "source_badge",
    "STYLE_NAMES",
]

STYLE_NAMES = ("rich", "compact", "plain", "markdown")

# ── ANSI ───────────────────────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_MATCH = "\033[1;93m"  # bright-yellow bold for query matches
_TITLE = "\033[1;97m"  # bright-white bold titles
_URL = "\033[36m"  # cyan urls
_SCORE = "\033[32m"  # green scores

#: per-source badge colors — stable identity per source
_BADGE_COLORS = {
    "memory": "\033[95m",  # magenta
    "wisdom": "\033[35m",  # purple
    "books": "\033[33m",  # yellow
    "docs": "\033[94m",  # blue
    "code": "\033[92m",  # green
    "timeline": "\033[96m",  # cyan
    "web_searxng": "\033[91m",
    "web_ddgs": "\033[91m",
    "web_tavily": "\033[91m",
    "web_serper": "\033[91m",
    "web_exa": "\033[91m",
    "web_brave": "\033[91m",
}
_DEFAULT_BADGE = "\033[90m"  # grey for OSINT + anything new


def _supports_color() -> bool:
    return sys.stdout.isatty()


def source_badge(source: str, *, color: bool = True) -> str:
    """``[source]`` badge, colored when ``color`` and the terminal allows."""
    if color and _supports_color():
        c = _BADGE_COLORS.get(source, _DEFAULT_BADGE)
        return f"{c}[{source}]{_RESET}"
    return f"[{source}]"


def highlight_matches(
    text: str, query: str, *, color: bool = True, markdown: bool = False
) -> str:
    """Bold the query's terms inside ``text`` (ANSI or markdown).

    Terms are matched case-insensitively, longest-first so multi-word
    phrases win over their parts; already-highlighted regions are not
    double-wrapped.
    """
    text = text or ""
    if not text:
        return ""
    terms = sorted(
        {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 2},
        key=len, reverse=True,
    )
    if not terms:
        return text
    if markdown:
        start, end = "**", "**"
    elif color and _supports_color():
        start, end = _MATCH, _RESET
    else:
        return text
    pattern = re.compile(
        "(" + "|".join(re.escape(t) for t in terms) + ")", re.IGNORECASE)
    return pattern.sub(lambda m: f"{start}{m.group(1)}{end}", text)


def _fmt_date(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    except (OverflowError, OSError, ValueError):
        return ""


def _hit_url(hit: Any) -> str:
    prov = getattr(hit, "provenance", None) or {}
    for key in ("url", "result_source", "profileUrl"):
        val = prov.get(key)
        if val:
            return str(val)
    return ""


def _hit_meta(hit: Any, *, color: bool) -> str:
    """The metadata rail: score · date · type · url."""
    dim = _DIM if color and _supports_color() else ""
    rst = _RESET if color and _supports_color() else ""
    parts = [f"{_SCORE if color and _supports_color() else ''}"
             f"★ {hit.score:.2f}{rst}"]
    date = _fmt_date(getattr(hit, "timestamp", None))
    if date:
        parts.append(date)
    parts.append(str(getattr(hit, "type", "")))
    url = _hit_url(hit)
    if url:
        ucolor = _URL if color and _supports_color() else ""
        parts.append(f"{ucolor}{url[:80]}{rst}")
    rank = (getattr(hit, "provenance", None) or {}).get("source_rank")
    if rank is not None:
        parts.append(f"#{rank}")
    return f"{dim}{' · '.join(p for p in parts if p)}{rst}"


def _render_hit_rich(hit: Any, index: int, query: str) -> str:
    n = f"{index}."
    badge = source_badge(str(getattr(hit, "source", "?")))
    title = highlight_matches(str(getattr(hit, "title", "")), query)
    snippet = highlight_matches(str(getattr(hit, "snippet", "")), query)
    snippet = snippet[:280] + ("…" if len(snippet) > 280 else "")
    lines = [
        f"{_DIM}{n}{_RESET} {badge} {_TITLE}{title}{_RESET}",
        f"    {snippet}",
        f"    {_hit_meta(hit, color=True)}",
    ]
    return "\n".join(lines)


def _render_hit_plain(hit: Any, index: int, query: str) -> str:
    badge = source_badge(str(getattr(hit, "source", "?")), color=False)
    title = str(getattr(hit, "title", ""))
    snippet = str(getattr(hit, "snippet", ""))[:280]
    lines = [f"{index}. {badge} {title}"]
    if snippet:
        lines.append(f"   {snippet}")
    lines.append(f"   {_hit_meta(hit, color=False)}")
    return "\n".join(lines)


def _render_hit_markdown(hit: Any, index: int, query: str) -> str:
    title = highlight_matches(
        str(getattr(hit, "title", "")), query, markdown=True)
    snippet = highlight_matches(
        str(getattr(hit, "snippet", "")), query, markdown=True)[:400]
    url = _hit_url(hit)
    date = _fmt_date(getattr(hit, "timestamp", None))
    badge = f"`{getattr(hit, 'source', '?')}`"
    head = f"**{index}. {badge}** "
    head += f"[{title}]({url})" if url else title
    meta = f"★ {hit.score:.2f} · {getattr(hit, 'type', '')}"
    if date:
        meta += f" · {date}"
    lines = [head, f"> {snippet}", f"_{meta}_"]
    return "\n".join(lines)


def _no_results_block(response: Any, query: str, *, color: bool) -> list[str]:
    """The most-designed state in search: explain + suggest, never just empty."""
    lines = [
        f"No matches for {query!r}.",
        "",
    ]
    skipped = getattr(response, "sources_skipped", None) or {}
    if skipped:
        lines.append("Sources skipped (configure them to widen coverage):")
        for name, note in skipped.items():
            lines.append(f"  • {name}: {note}")
        lines.append("")
    searched = getattr(response, "sources_searched", None) or []
    if searched:
        lines.append(f"Searched: {', '.join(searched)} — the query matched nothing there.")
        lines.append("")
    lines.append("Try:")
    lines.append("  • fewer / simpler terms (each extra word narrows the match)")
    lines.append("  • a different phrasing of the same idea")
    lines.append("  • dropping date filters if any were applied")
    deduped = getattr(response, "deduped", 0)
    if deduped:
        lines.append(
            f"  • note: {deduped} duplicate hit(s) were folded away — "
            "the results may exist under another source")
    return lines


def _timings_footer(response: Any, *, color: bool) -> str:
    timings = getattr(response, "timings", None) or {}
    elapsed = getattr(response, "elapsed", 0.0) or 0.0
    if not timings and not elapsed:
        return ""
    dim = _DIM if color and _supports_color() else ""
    rst = _RESET if color and _supports_color() else ""
    bits = [f"{n} {t * 1000:.0f}ms" for n, t in timings.items()]
    text = f"{elapsed * 1000:.0f}ms total"
    if bits:
        text += " (" + ", ".join(bits) + ")"
    return f"{dim}{text}{rst}"


def render_search(
    response: Any,
    style: str = "rich",
    *,
    group_by_source: bool = False,
    show_timings: bool = True,
) -> str:
    """Render a :class:`SearchResponse` for a human.

    ``style`` is one of ``rich`` (ANSI, auto-degrades to plain when not a
    TTY), ``compact`` (one line per hit), ``plain`` (no color codes),
    ``markdown``. ``group_by_source`` sections hits under per-source
    headers. Raises ``ValueError`` on an unknown style.
    """
    if style not in STYLE_NAMES:
        raise ValueError(
            f"unknown render style: {style!r}; valid: {', '.join(STYLE_NAMES)}")
    query = str(getattr(response, "query", ""))
    hits = list(getattr(response, "hits", None) or [])
    total = getattr(response, "total", len(hits))
    color_ok = style == "rich" and _supports_color()
    use_color = style == "rich" and color_ok

    if style == "compact":
        lines = [f"{total} result(s) for {query!r}"]
        for i, hit in enumerate(hits, 1):
            badge = source_badge(str(getattr(hit, "source", "?")), color=False)
            url = _hit_url(hit)
            tail = f" — {url[:100]}" if url else ""
            lines.append(f"{i}. {badge} {getattr(hit, 'title', '')}{tail}")
        if not hits:
            lines.extend(_no_results_block(response, query, color=False))
        if show_timings:
            foot = _timings_footer(response, color=False)
            if foot:
                lines += ["", foot]
        return "\n".join(lines)

    if style == "markdown":
        lines = [f"## {total} result(s) for {query!r}", ""]
        if group_by_source:
            order: list[str] = []
            buckets: dict[str, list[Any]] = {}
            for hit in hits:
                s = str(getattr(hit, "source", "?"))
                buckets.setdefault(s, []).append(hit)
                if s not in order:
                    order.append(s)
            idx = 0
            for s in order:
                lines.append(f"### `{s}`")
                for hit in buckets[s]:
                    idx += 1
                    lines.append(_render_hit_markdown(hit, idx, query))
                    lines.append("")
        else:
            for i, hit in enumerate(hits, 1):
                lines.append(_render_hit_markdown(hit, i, query))
                lines.append("")
        if not hits:
            lines.extend(_no_results_block(response, query, color=False))
        if show_timings:
            foot = _timings_footer(response, color=False)
            if foot:
                lines += ["", f"_{foot}_"]
        return "\n".join(lines).rstrip() + "\n"

    # rich / plain
    header = f"{total} result(s) for {query!r}"
    if use_color:
        header = f"{_BOLD}{header}{_RESET}"
    lines = [header, ""]
    render_hit = _render_hit_rich if use_color else _render_hit_plain
    if group_by_source:
        order = []
        buckets: dict[str, list[Any]] = {}
        for hit in hits:
            s = str(getattr(hit, "source", "?"))
            buckets.setdefault(s, []).append(hit)
            if s not in order:
                order.append(s)
        idx = 0
        for s in order:
            badge = source_badge(s, color=use_color)
            label = f"── {badge} ──"
            lines.append(label)
            for hit in buckets[s]:
                idx += 1
                lines.append(render_hit(hit, idx, query))
            lines.append("")
    else:
        for i, hit in enumerate(hits, 1):
            lines.append(render_hit(hit, i, query))
    if not hits:
        lines.extend(_no_results_block(response, query, color=use_color))
    if show_timings:
        foot = _timings_footer(response, color=use_color)
        if foot:
            lines += ["", foot]
    return "\n".join(lines).rstrip() + "\n"


#: terminal width helper for callers that wrap snippets themselves
def term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size().columns or default
    except OSError:
        return default
