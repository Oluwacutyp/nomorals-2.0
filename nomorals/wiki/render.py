"""Wiki rendering: chat text and plain HTML.

Two renders, one source. Chat text resolves ``[[links]]`` to plain
titles and names backlinks; HTML is plain and escaped — no scripts,
no styles, the same CSP posture as the public-face pages — with
wikilinks as ``/wiki/<slug>`` anchors. A broken link renders as an
anchor with class ``broken`` and a ``missing page`` title, so the
reader sees the break instead of a silent dead end.
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

from .links import backlinks_for, extract_wikilinks, resolve_link

__all__ = ["render_chat", "render_html"]

_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
_CODE_RE = re.compile(r"`([^`\n]+)`")


def _plain_md(text: str) -> str:
    """Markdown → readable plain text for chat."""
    out = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        stripped = _BOLD_RE.sub(r"\1", stripped)
        stripped = _ITALIC_RE.sub(r"\1", stripped)
        stripped = _CODE_RE.sub(r"\1", stripped)
        stripped = _MD_LINK_RE.sub(r"\1", stripped)
        stripped = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
                          lambda m: m.group(2) or m.group(1), stripped)
        out.append(stripped)
    return "\n".join(out).strip()


def render_chat(store: Any, page: Any) -> str:
    """A page as chat text: title, summary, body, links, backlinks."""
    from .nav import neighbors

    lines = [page.title]
    if page.summary:
        lines.append(page.summary)
    if page.cover:
        lines.append(f"Cover: {page.cover}")
    lines.append("")
    lines.append(_plain_md(page.body))
    targets = extract_wikilinks(page.body)
    if targets:
        resolved = []
        for target in targets:
            slug = resolve_link(store, target)
            resolved.append(
                target if slug is None else (store.get(slug) or page).title)
        lines.append("")
        lines.append("Links: " + ", ".join(resolved))
    backs = backlinks_for(store, page.slug)
    if backs:
        lines.append("Backlinks: " + ", ".join(
            (store.get(slug) or page).title for slug in backs))
    prev, nxt = neighbors(store, page.slug)
    nav = []
    if prev is not None:
        nav.append(f"Previous: {prev.title} — /wiki read {prev.slug}")
    if nxt is not None:
        nav.append(f"Next: {nxt.title} — /wiki read {nxt.slug}")
    if nav:
        lines.append("")
        lines.extend(nav)
    return "\n".join(lines)


def _inline_html(text: str, store: Any) -> str:
    """Inline markdown → HTML on already-escaped text.

    The caller escapes first, so labels and hrefs are used as-is —
    escaping them again would double-escape ``&`` into ``&amp;amp;``.
    """
    def _wiki(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        label = match.group(2) or target
        slug = resolve_link(store, target)
        if slug is None:
            return (f'<a class="broken" href="/wiki/{slugify_fallback(target)}" '
                    f'title="missing page">{label}</a>')
        return f'<a href="/wiki/{slug}">{label}</a>'

    out = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)
    out = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = _CODE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _MD_LINK_RE.sub(
        lambda m: (f'<a href="{m.group(2)}">'
                   f"{m.group(1)}</a>"), out)
    out = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", _wiki, out)
    return out


def slugify_fallback(target: str) -> str:
    """A slug-shaped guess for a broken link's href (never a claim)."""
    from .pages import slugify

    try:
        return slugify(target)
    except ValueError:
        return "missing"


def _sections(body: str) -> list[tuple[str, list[str]]]:
    """Split a body into (heading, lines) sections; ``""`` heads the intro."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    in_code = False
    for line in (body or "").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            sections[-1][1].append(line)
            continue
        if not in_code and line.startswith("## "):
            sections.append((line[3:].strip(), []))
            continue
        sections[-1][1].append(line)
    return sections


def render_html(store: Any, page: Any) -> str:
    """A page as a standalone plain-HTML document (no scripts)."""
    from .nav import neighbors

    parts = ["<!DOCTYPE html>", "<html lang=\"en\">", "<head>",
             "<meta charset=\"utf-8\">",
             f"<title>{_html.escape(page.title)} — Devon wiki</title>",
             "</head>", "<body>",
             f"<h1>{_html.escape(page.title)}</h1>"]
    if page.summary:
        parts.append(f"<p>{_html.escape(page.summary)}</p>")
    if page.cover:
        parts.append(
            f"<img src=\"{_html.escape(page.cover)}\" "
            f"alt=\"{_html.escape(page.title)}\">")
    for heading, lines in _sections(page.body):
        if heading:
            parts.append(f"<h2>{_html.escape(heading)}</h2>")
        buf: list[str] = []
        in_code = False
        code_lines: list[str] = []
        in_list = False

        def _flush_para() -> None:
            nonlocal buf
            if buf:
                parts.append("<p>" + "<br>".join(
                    _inline_html(_html.escape(line), store)
                    for line in buf) + "</p>")
                buf = []

        def _close_list() -> None:
            nonlocal in_list
            if in_list:
                parts.append("</ul>")
                in_list = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    parts.append("<pre><code>"
                                 + _html.escape("\n".join(code_lines))
                                 + "</code></pre>")
                    code_lines = []
                    in_code = False
                else:
                    _flush_para()
                    _close_list()
                    in_code = True
                continue
            if in_code:
                code_lines.append(line)
                continue
            if stripped.startswith("#"):
                _flush_para()
                _close_list()
                level = len(stripped) - len(stripped.lstrip("#"))
                tag = f"h{min(3 + level, 6)}"
                parts.append(f"<{tag}>{_inline_html(_html.escape(stripped.lstrip('#').strip()), store)}</{tag}>")
                continue
            if stripped.startswith(("- ", "* ")):
                _flush_para()
                if not in_list:
                    parts.append("<ul>")
                    in_list = True
                parts.append("<li>" + _inline_html(
                    _html.escape(stripped[2:]), store) + "</li>")
                continue
            if not stripped:
                _flush_para()
                _close_list()
                continue
            buf.append(stripped)
        if in_code:
            parts.append("<pre><code>"
                         + _html.escape("\n".join(code_lines))
                         + "</code></pre>")
        _flush_para()
        _close_list()
    targets = extract_wikilinks(page.body)
    if targets:
        parts.append("<hr>")
        links = []
        for target in targets:
            slug = resolve_link(store, target)
            if slug is None:
                links.append(_html.escape(target) + " (missing page)")
            else:
                links.append(
                    f'<a href="/wiki/{_html.escape(slug)}">'
                    f"{_html.escape((store.get(slug) or page).title)}</a>")
        parts.append("<p>Links: " + ", ".join(links) + "</p>")
    prev, nxt = neighbors(store, page.slug)
    nav = []
    if prev is not None:
        nav.append(
            f'<a href="/wiki/{_html.escape(prev.slug)}">'
            f"← {_html.escape(prev.title)}</a>")
    if nxt is not None:
        nav.append(
            f'<a href="/wiki/{_html.escape(nxt.slug)}">'
            f"{_html.escape(nxt.title)} →</a>")
    if nav:
        parts.append("<hr>")
        parts.append("<p>" + " | ".join(nav) + "</p>")
    parts.extend(["</body>", "</html>"])
    return "\n".join(parts)
