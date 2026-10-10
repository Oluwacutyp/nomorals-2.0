"""Summarize-then-truncate compression for context sections.

Compression never silently drops load-bearing content: a section's ``keep``
strings are carried verbatim, and whenever content is removed the section is
marked ``truncated`` and the marker names what happened.  A section that
"vanishes" is a bug; a section that says it was cut is information.

Two summarization strategies ship with the module:

* :func:`extractive_summary` — deterministic head/tail paragraph selection.
* :func:`salient_extract` — salience-scored extractive summary: sentences are
  scored by term rarity (TF-IDF style) plus information-density signals
  (identifiers, numbers, code), the highest-utility sentences survive in
  original order.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable

from ..core.text import truncate_to_tokens
from .sections import Section, token_count

__all__ = [
    "compress_section",
    "summarize_then_truncate",
    "extractive_summary",
    "salient_extract",
]

#: A compressor callback: (text, max_tokens) -> condensed text.
Summarizer = Callable[[str, int], str]

_WORD_RE = re.compile(r"[A-Za-z][\w\-/.:]{1,}")
_DENSE_RE = re.compile(r"[A-Za-z]*\d[\w\-/.:]*|[A-Z]{2,}|`[^`]+`")
_STOPWORDS = frozenset(
    "a an the and or but if then else for of to in on at as is are was were "
    "be been being it its this that these those with from by we you he she "
    "they them his her our your their will would can could should shall may "
    "might do does did done have has had not no yes so such than too very "
    "just also more most some any each other into over after before between "
    "during under again once here there when where which who whom whose what "
    "how why because while until".split()
)


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
        cost = token_count(para)
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
    omitted_tokens = token_count(text) - used
    body = "\n\n".join(head + tail[::-1])
    if omitted:
        body += (
            f"\n\n[… {omitted} paragraph(s) omitted from the middle "
            f"(~{omitted_tokens} tokens) …]"
        )
    return body


def _sentences(text: str) -> list[str]:
    """Split into sentences, keeping line-oriented entries (logs, lists) whole."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[-*•\d.)\]]", line) or len(line) < 220:
            out.append(line)
            continue
        for part in re.split(r"(?<=[.!?])\s+", line):
            part = part.strip()
            if part:
                out.append(part)
    return out


def salient_extract(text: str, max_tokens: int) -> str:
    """Salience-scored extractive summary.

    Every sentence is scored by ``tf * idf`` over the document plus a density
    bonus for identifier/number/code tokens (the "facts per token" utility
    score).  The highest-utility sentences fill the budget and are emitted in
    original document order, so the summary reads like the source.
    Deterministic — no model calls, no randomness.
    """
    sentences = _sentences(text)
    if len(sentences) <= 2:
        return text

    tokenized = []
    for sent in sentences:
        words = [
            w.lower() for w in _WORD_RE.findall(sent)
            if w.lower() not in _STOPWORDS
        ]
        tokenized.append(words)

    doc_count = len(tokenized)
    df: Counter[str] = Counter()
    for words in tokenized:
        for word in set(words):
            df[word] += 1
    idf = {w: math.log((1 + doc_count) / (1 + c)) + 1.0 for w, c in df.items()}

    scored: list[tuple[float, int, str, int]] = []
    for idx, (sent, words) in enumerate(zip(sentences, tokenized)):
        if not words:
            continue
        tf = Counter(words)
        utility = sum(tf[w] * idf.get(w, 1.0) for w in tf)
        utility /= math.sqrt(len(words))  # length-normalized
        density_bonus = 1.0 + 0.5 * len(_DENSE_RE.findall(sent))
        cost = token_count(sent)
        scored.append((utility * density_bonus, idx, sent, cost))

    if not scored:
        return text

    # Greedily take highest-utility sentences that fit, then restore order.
    scored.sort(key=lambda s: s[0], reverse=True)
    picked: list[tuple[float, int, str, int]] = []
    used = 0
    for score, idx, sent, cost in scored:
        if used + cost > max_tokens and picked:
            continue
        picked.append((score, idx, sent, cost))
        used += cost
    picked.sort(key=lambda s: s[1])

    omitted = len(sentences) - len(picked)
    body = "\n".join(sent for _, _, sent, _ in picked)
    if omitted:
        omitted_tokens = token_count(text) - used
        body += (
            f"\n[… {omitted} lower-salience sentence(s) omitted "
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
    original_tokens = token_count(text)
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
    keep_tokens = token_count(keep_text)
    room = max(0, max_tokens - keep_tokens)

    rest = body
    if summarizer is not None and token_count(rest) > 2 * max(1, room):
        rest = summarizer(rest, room)

    if keep_tail:
        # Drop from the front: split lines, keep the tail that fits.
        lines = rest.splitlines()
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = token_count(line) + 1
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
