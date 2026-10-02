"""Summarize-then-truncate compression for context sections.

Compression never silently drops load-bearing content: a section's ``keep``
strings are carried verbatim, and whenever content is removed the section is
marked ``truncated`` and the marker names what happened.  A section that
"vanishes" is a bug; a section that says it was cut is information.
"""

from __future__ import annotations

from typing import Callable

from ..core.text import approx_token_count, truncate_to_tokens
from .sections import Section

__all__ = [
    "compress_section",
    "summarize_then_truncate",
    "extractive_summary",
]

#: A compressor callback: (text, max_tokens) -> condensed text.
Summarizer = Callable[[str, int], str]


def extractive_summary(text: str, max_tokens: int) -> str:
    """Deterministic extractive summary: keep head and tail paragraphs.

    Paragraphs are split on blank lines; head and tail alternate until the
    budget is spent and the omitted middle is named explicitly.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) <= 2:
        return text
    head: list[str] = []
    tail: list[str] = []
    used = 0
    omitted = 0
    i, j = 0, len(paras) - 1
    take_head = True
    while i <= j:
        para = paras[i] if take_head else paras[j]
        cost = approx_token_count(para)
        if used + cost > max_tokens:
            omitted += 1
            if take_head:
                i += 1
            else:
                j -= 1
            continue
        used += cost
        if take_head:
            head.append(para)
            i += 1
        else:
            tail.append(para)
            j -= 1
        take_head = not take_head
    omitted_tokens = approx_token_count(text) - used
    body = "\n\n".join(head + tail[::-1])
    if omitted:
        body += (
            f"\n\n[… {omitted} paragraph(s) omitted from the middle "
            f"(~{omitted_tokens} tokens) …]"
        )
    return body


def _marker(name: str, original_tokens: int, kept_tokens: int) -> str:
    removed = max(0, original_tokens - kept_tokens)
    return (
        f"\n[… section '{name}' truncated: ~{removed} of ~{original_tokens} "
        f"tokens omitted …]"
    )


def summarize_then_truncate(
    text: str,
    max_tokens: int,
    *,
    section_name: str = "section",
    summarizer: Summarizer | None = None,
    keep: tuple[str, ...] = (),
    keep_tail: bool = False,
) -> tuple[str, bool]:
    """Condense ``text`` to ``max_tokens``.

    Order of operations: first the summarizer condenses (only when the text
    is well over budget, so small overruns truncate directly), then the
    remainder is truncated.  ``keep`` strings are pinned verbatim at the
    head: their first occurrence is lifted out of the body (never duplicated)
    and always survives.  ``keep_tail=True`` truncates from the front instead
    (history: newest entries live at the end).  Returns (text, truncated);
    the truncation marker appears only when content was actually cut.
    """
    original_tokens = approx_token_count(text)
    if original_tokens <= max_tokens and not keep:
        return text, False

    # Pin keep strings verbatim: lift the first occurrence out of the body so
    # it is never duplicated and never truncated away.
    body = text
    pinned: list[str] = []
    for keep_str in keep:
        if keep_str and keep_str in body:
            pinned.append(keep_str)
            body = body.replace(keep_str, "", 1)
    body = "\n".join(line for line in body.splitlines() if line.strip()).strip()
    keep_text = "\n".join(pinned)
    keep_tokens = approx_token_count(keep_text)
    room = max(0, max_tokens - keep_tokens)

    rest = body
    if summarizer is not None and approx_token_count(rest) > 2 * max(1, room):
        rest = summarizer(rest, room)

    if keep_tail:
        # Drop from the front: split lines, keep the tail that fits.
        lines = rest.splitlines()
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = approx_token_count(line) + 1
            if used + cost > room and kept:
                break
            used += cost
            kept.append(line)
        rest = "\n".join(reversed(kept))
    else:
        rest = truncate_to_tokens(rest, room)

    truncated = original_tokens > max_tokens
    parts = [p for p in (keep_text, rest) if p]
    out = "\n".join(parts)
    if truncated:
        out += _marker(section_name, original_tokens, max_tokens)
    return out, truncated


def compress_section(
    section: Section,
    max_tokens: int,
    *,
    summarizer: Summarizer | None = None,
    keep_tail: bool = False,
) -> Section:
    """Compress one section in place to ``max_tokens``.

    ``section.keep`` is preserved verbatim even for load-bearing sections —
    it is pinned, never silently removed.  When anything is cut,
    ``section.truncated`` is set and the content carries an explicit marker.
    """
    if section.tokens <= max_tokens and not section.keep:
        section.truncated = False
        return section
    summarizer = summarizer if summarizer is not None else extractive_summary
    text, truncated = summarize_then_truncate(
        section.content,
        max_tokens,
        section_name=section.name,
        summarizer=summarizer,
        keep=section.keep,
        keep_tail=keep_tail,
    )
    section.content = text
    section.truncated = truncated
    return section
