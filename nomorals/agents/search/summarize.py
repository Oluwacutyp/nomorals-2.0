"""Summarization: model-synthesized when a real model is answering,
extractive (top sentences, cited) when it is not.

The extractive path exists on purpose: a bot whose model is down should
still return *something true and cited*, not silence.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ...core.text import normalize_text
from .curate import domain

__all__ = ["extractive_summarize", "model_summarize"]

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WORD = re.compile(r"[a-z0-9]{2,}")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE.split(normalize_text(text)) if s.strip()]
    return [p for p in parts if 25 <= len(p) <= 400]


def extractive_summarize(query: str, pages: Sequence[dict[str, Any]], max_sentences: int = 6) -> str:
    """Frequency-ranked sentences from the fetched pages, cited by number.

    Every sentence cites its source as ``[n]`` and the footer carries the
    numbered ``Sources:`` block with full URLs — a claim is verifiable
    only if the reader can click through to it. The domain rides along
    in each citation (``[1 · a.com]``) so the source's identity is visible
    at a glance.
    """
    q_terms = set(_WORD.findall(query.lower()))
    numbered = list(pages)  # [n] below indexes into this list
    scored: list[tuple[float, int, str, int]] = []
    for n, page in enumerate(numbered):
        for idx, sent in enumerate(_sentences(page.get("text", ""))):
            terms = set(_WORD.findall(sent.lower()))
            if not terms:
                continue
            overlap = len(terms & q_terms) / max(len(q_terms), 1)
            density = min(len(sent) / 180.0, 1.0)
            scored.append((round(0.8 * overlap + 0.2 * density, 6), idx, sent, n))
    scored.sort(key=lambda t: (-t[0], t[1]))
    picked = scored[:max_sentences]
    # restore document order within the pick so it reads as prose, not a scatter
    picked.sort(key=lambda t: (t[3], t[1]))
    lines = ["extractive summary (no live model — top sentences, cited):"]
    for _s, _idx, sent, n in picked:
        dom = numbered[n].get("domain") or domain(numbered[n].get("url", ""))
        lines.append(f"  • {sent}  [{n + 1} · {dom}]")
    if not picked:
        lines.append("  • (no readable text on the fetched pages)")
    lines.append("")
    lines.append("Sources:")
    for n, page in enumerate(numbered):
        title = (page.get("title") or "").strip() or "(untitled)"
        lines.append(f"  [{n + 1}] {title} — {page.get('url', '')}")
    return "\n".join(lines)


def model_summarize(router: Any, query: str, pages: Sequence[dict[str, Any]],
                    max_chars: int = 24000) -> str:
    """Ask the active model for a sourced synthesis. Raises on failure so
    the caller can fall back to the extractive pass.

    ``router`` may be a raw provider router or a
    :class:`~nomorals.llm.brain.Brain` — a Brain is used as-is.
    """
    from ...llm.base import Message, SamplingParams
    from ...llm.brain import Brain

    brain = router if isinstance(router, Brain) else Brain(router=router)

    chunks: list[str] = []
    budget = max_chars
    for page in pages:
        header = f"--- source: {page.get('url', '')} (title: {page.get('title', '')}) ---"
        body = normalize_text(page.get("text", ""))[: min(6000, budget)]
        chunks.append(header + "\n" + body)
        budget -= len(header) + len(body)
        if budget <= 0:
            break
    prompt = (
        "You are a research assistant. Answer the query using ONLY the pages below. "
        "Format: a 2-4 sentence answer, then 'key facts:' with 3-7 bullets, then "
        "'sources:' listing the URLs you actually used. If the pages don't answer it, "
        "say so plainly — do not invent.\n\n"
        f"query: {query}\n\n" + "\n\n".join(chunks)
    )
    response = brain.chat(
        [Message.system("You synthesize research from provided pages; you never browse."),
         Message.user(prompt)],
        SamplingParams(temperature=0.2),
        task_kind="summarize",
    )
    if not response.ok or not (response.text or "").strip():
        raise RuntimeError(f"model summarization failed: {response.error}")
    return response.text.strip()
