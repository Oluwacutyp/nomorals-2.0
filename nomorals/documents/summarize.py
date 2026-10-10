"""Extractive summarization and keyword hooks for parsed documents (L4).

:func:`summarize` picks the highest-scoring sentences of a
:class:`~nomorals.documents.model.Document` and returns them in document
order; :func:`summarize_text` does the same for raw text.  Four scoring
methods (the sumy playbook, stdlib-only, deterministic):

* ``"tf"`` — term-frequency density over content words (the original).
* ``"textrank"`` — PageRank over the sentence-similarity graph
  (TextRank; best coherence for articles and reports).
* ``"luhn"`` — significant-word clusters (fully deterministic,
  keyword-heavy summaries).
* ``"lead"`` — position baseline (news-style: first sentences win).

``diversity=True`` applies an MMR-style redundancy penalty so
near-duplicate sentences are not both picked; ``position_bias=True``
gives lead sentences a mild boost; ``query=`` focuses the summary on
query terms (search-result summarization).

:func:`summarize_abstractive` is the optional LLM path (via
``nomorals.llm``) — extractive stays the default, the model is opt-in.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .errors import DocumentError
from .model import Document, full_text

__all__ = [
    "bullet_digest",
    "keyphrases",
    "keywords",
    "summarize",
    "summarize_abstractive",
    "summarize_query",
    "summarize_sections",
    "summarize_text",
    "tldr",
]

#: Common English function words excluded from term-frequency scoring.
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can cannot could did do does
doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just like me
more most my myself no nor not now of off on once only or other ought our
ours ourselves out over own same she should so some such than that the their
theirs them themselves then there these they this those through to too under
until up very was we were what when where which while who whom why will with
you your yours yourself yourselves would within without per via etc
""".split())

_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'“‘(\[]?[A-Z0-9])")

_METHODS = ("tf", "textrank", "luhn", "lead")


def _sentences(text: str) -> list[str]:
    """Split text into sentences; keeps order, drops fragments < 4 words.

    Paragraph-aware: blank lines are hard boundaries, so a title or
    heading never glues itself onto the first sentence of the body.
    """
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        collapsed = " ".join(para.split())
        if not collapsed:
            continue
        for chunk in _SENTENCE_SPLIT_RE.split(collapsed):
            clean = " ".join(chunk.split())
            if len(_WORD_RE.findall(clean.lower())) >= 4:
                out.append(clean)
    return out


def _sentences_or_fallback(text: str) -> list[str]:
    """Sentences, or the whole collapsed text as one unit when nothing
    passes the fragment filter.

    A document with real (if tiny) text gets a best-effort summary
    instead of an error — fail-fast is reserved for empty input, which
    is checked before this is ever called.
    """
    sents = _sentences(text)
    if not sents:
        collapsed = " ".join(text.split())
        if collapsed:
            sents = [collapsed]
    return sents


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower())
            if w not in _STOPWORDS and len(w) >= 3]


# ── scoring methods ─────────────────────────────────────────────────────────


def _score_tf(sentences: list[str]) -> list[float]:
    """Term-frequency density (the original scorer)."""
    freq = Counter()
    for sentence in sentences:
        freq.update(_content_words(sentence))
    if not freq:
        return [0.0] * len(sentences)
    scores: list[float] = []
    for sentence in sentences:
        words = _content_words(sentence)
        if not words:
            scores.append(0.0)
            continue
        raw = sum(freq[w] for w in words)
        # Length normalization: reward density, not bulk.
        scores.append(raw / math.sqrt(len(words)))
    return scores


def _score_textrank(sentences: list[str]) -> list[float]:
    """TextRank: PageRank over the sentence-similarity graph.

    Edge weight is the classic TextRank similarity
    ``|common| / (log|Si| + log|Sj|)`` on content-word sets; power
    iteration from a uniform start is fully deterministic.
    """
    n = len(sentences)
    if n == 1:
        return [1.0]
    word_sets = [set(_content_words(s)) for s in sentences]
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = word_sets[i], word_sets[j]
            if a and b:
                common = len(a & b)
                if common:
                    weight = common / (math.log(len(a)) + math.log(len(b)))
                    sim[i][j] = sim[j][i] = weight
    damping, scores = 0.85, [1.0 / n] * n
    for _ in range(100):
        updated = [(1.0 - damping) / n] * n
        for i in range(n):
            for j in range(n):
                if i != j and sim[j][i] > 0:
                    out_weight = sum(sim[j])
                    if out_weight > 0:
                        updated[i] += damping * sim[j][i] / out_weight * scores[j]
        if max(abs(updated[i] - scores[i]) for i in range(n)) < 1e-9:
            break
        scores = updated
    return scores


def _score_luhn(sentences: list[str]) -> list[float]:
    """Luhn: significant-word clusters.

    Significant words are the most frequent content words; each sentence
    scores the best cluster of significant words separated by fewer than
    4 insignificant words: ``count² / span``.
    """
    freq = Counter()
    for sentence in sentences:
        freq.update(_content_words(sentence))
    if not freq:
        return [0.0] * len(sentences)
    cutoff = max(4, len(freq) // 5)
    significant = {w for w, _ in freq.most_common(cutoff)}

    def sentence_score(words: list[str]) -> float:
        best = 0.0
        start: int | None = None
        sig_count = 0
        gap = 0
        for i, word in enumerate(words):
            is_sig = word in significant
            if is_sig:
                if start is None:
                    start, sig_count, gap = i, 0, 0
                sig_count += 1
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= 4:
                    span = i - start
                    if span > 0:
                        best = max(best, sig_count * sig_count / span)
                    start = None
        if start is not None:
            span = len(words) - start
            if span > 0:
                best = max(best, sig_count * sig_count / span)
        return best

    return [sentence_score(_content_words(s)) for s in sentences]


def _score_lead(sentences: list[str]) -> list[float]:
    """Position baseline: earlier sentences score higher (news style)."""
    return [1.0 / (i + 1) for i in range(len(sentences))]


_SCORERS = {
    "tf": _score_tf,
    "textrank": _score_textrank,
    "luhn": _score_luhn,
    "lead": _score_lead,
}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _select(sentences: list[str], scores: list[float], count: int, *,
            diversity: bool, position_bias: bool,
            query: str | None) -> list[str]:
    """Pick ``count`` sentences: top-N by score, MMR diversity penalty,
    document-order output."""
    n = len(sentences)
    if n <= count:
        return sentences
    adjusted = list(scores)
    if position_bias:
        adjusted = [s / (1.0 + 0.15 * i) for i, s in enumerate(adjusted)]
    if query:
        query_words = set(_content_words(query))
        if query_words:
            peak = max(adjusted) if adjusted else 1.0
            for i, sentence in enumerate(sentences):
                overlap = len(set(_content_words(sentence)) & query_words)
                adjusted[i] += overlap * peak * 0.5
    word_sets = [set(_content_words(s)) for s in sentences]
    chosen: list[int] = []
    remaining = set(range(n))
    while remaining and len(chosen) < count:
        best, best_value = -1, float("-inf")
        for i in sorted(remaining):
            value = adjusted[i]
            if diversity and chosen:
                redundancy = max(_jaccard(word_sets[i], word_sets[j])
                                 for j in chosen)
                value -= 0.7 * redundancy * max(adjusted)
            if value > best_value:
                best, best_value = i, value
        chosen.append(best)
        remaining.discard(best)
    return [sentences[i] for i in sorted(chosen)]


def _resolve_method(method: str) -> str:
    if method not in _METHODS:
        raise DocumentError(
            f"unknown summarize method {method!r}: choose from "
            f"{list(_METHODS)}")
    return method


def summarize_text(text: str, *, sentences: int = 5, method: str = "tf",
                   diversity: bool = True, position_bias: bool = False,
                   query: str | None = None) -> list[str]:
    """Return the ``sentences`` most representative sentences of ``text``.

    ``method`` is one of ``tf`` (default), ``textrank``, ``luhn``,
    ``lead``.  Sentences come back in document order.  Raises
    :class:`DocumentError` on empty text or a non-positive sentence count.
    """
    if sentences <= 0:
        raise DocumentError(f"sentences must be positive, got {sentences}")
    if not text or not text.strip():
        raise DocumentError("cannot summarize empty text")
    _resolve_method(method)
    sents = _sentences_or_fallback(text)
    if not sents:
        raise DocumentError("no summarizable sentences found in text")
    scores = _SCORERS[method](sents)
    return _select(sents, scores, sentences, diversity=diversity,
                   position_bias=position_bias, query=query)


def summarize(doc: Document, *, sentences: int = 5, method: str = "tf",
              diversity: bool = True, position_bias: bool = False,
              query: str | None = None) -> list[str]:
    """Extractive summary of a parsed :class:`Document`.

    Raises :class:`DocumentError` when the document has no summarizable
    text.
    """
    if not isinstance(doc, Document):
        raise DocumentError("summarize needs a Document object")
    return summarize_text(full_text(doc), sentences=sentences, method=method,
                          diversity=diversity, position_bias=position_bias,
                          query=query)


def summarize_sections(doc: Document, *, sentences: int = 2,
                       method: str = "tf") -> list[dict[str, Any]]:
    """Per-section summaries: [{heading, level, page, summary}].

    The long-document digest — each headed section summarized on its own
    instead of one global pick.  Sections with no summarizable text are
    skipped.
    """
    if not isinstance(doc, Document):
        raise DocumentError("summarize_sections needs a Document object")
    if sentences <= 0:
        raise DocumentError(f"sentences must be positive, got {sentences}")
    _resolve_method(method)
    out: list[dict[str, Any]] = []
    for section in doc.sections:
        body = (section.heading + "\n" + section.text).strip()
        try:
            summary = summarize_text(body, sentences=sentences, method=method,
                                     diversity=False, position_bias=False)
        except DocumentError:
            continue
        if summary:
            out.append({"heading": section.heading, "level": section.level,
                        "page": section.page, "summary": summary})
    if not out:
        raise DocumentError("document has no summarizable sections")
    return out


def summarize_query(text: str | Document, query: str, *,
                    sentences: int = 3, method: str = "tf") -> list[str]:
    """Query-focused summary: sentences most relevant to ``query``."""
    if not query or not query.strip():
        raise DocumentError("summarize_query needs a non-empty query")
    body = full_text(text) if isinstance(text, Document) else text
    return summarize_text(body, sentences=sentences, method=method,
                          query=query)


def tldr(text: str | Document, *, words: int = 60,
         method: str = "textrank") -> str:
    """Ultra-short summary: top-ranked sentences within a word budget."""
    if words <= 0:
        raise DocumentError(f"words must be positive, got {words}")
    body = full_text(text) if isinstance(text, Document) else text
    if not body or not body.strip():
        raise DocumentError("cannot summarize empty text")
    _resolve_method(method)
    sents = _sentences_or_fallback(body)
    if not sents:
        raise DocumentError("no summarizable sentences found in text")
    scores = _SCORERS[method](sents)
    ranked = sorted(range(len(sents)), key=lambda i: (-scores[i], i))
    picked: list[str] = []
    used = 0
    for i in ranked:
        count = len(sents[i].split())
        if used + count > words and picked:
            break
        picked.append(sents[i])
        used += count
    picked.sort(key=lambda s: sents.index(s))
    return " ".join(picked)


def bullet_digest(doc: Document, *, sentences: int = 5,
                  method: str = "textrank") -> str:
    """A markdown bullet digest of a document (chat/paste friendly)."""
    if not isinstance(doc, Document):
        raise DocumentError("bullet_digest needs a Document object")
    summary = summarize(doc, sentences=sentences, method=method)
    title = f"# {doc.title}\n\n" if doc.title else ""
    return title + "\n".join(f"- {sentence}" for sentence in summary) + "\n"


def summarize_abstractive(doc: Document | str, *, max_words: int = 150,
                          hint: str = "") -> str:
    """Abstractive summary via the LLM brain (opt-in; extractive is default).

    Raises :class:`DocumentError` when no LLM brain is available — the
    caller should fall back to :func:`summarize`.
    """
    if max_words <= 0:
        raise DocumentError(f"max_words must be positive, got {max_words}")
    body = full_text(doc) if isinstance(doc, Document) else doc
    if not body or not body.strip():
        raise DocumentError("cannot summarize empty text")
    try:
        from ..llm.brain import get_brain
        brain = get_brain()
    except Exception as exc:
        raise DocumentError(
            f"abstractive summarization needs the LLM brain: {exc}") from exc
    prompt = (
        f"Summarize the following document in at most {max_words} words. "
        "Write a fluent paragraph, no bullet points, no preamble."
        + (f" Focus on: {hint}." if hint else "")
        + f"\n\nDocument:\n{body[:12000]}"
    )
    try:
        response = brain.complete(prompt)
    except Exception as exc:
        raise DocumentError(
            f"abstractive summarization failed: {exc}") from exc
    text = getattr(response, "text", "") or str(response)
    text = text.strip()
    if not text:
        raise DocumentError("the LLM returned an empty summary")
    return text


# ── keywords & keyphrases ───────────────────────────────────────────────────


def keywords(text: str, *, top_n: int = 10) -> list[str]:
    """Top content words of ``text`` by frequency (document keyword hook).

    Raises :class:`DocumentError` on empty text or non-positive top_n.
    """
    if top_n <= 0:
        raise DocumentError(f"top_n must be positive, got {top_n}")
    if not text or not text.strip():
        raise DocumentError("cannot extract keywords from empty text")
    freq = Counter(_content_words(text))
    if not freq:
        raise DocumentError("no keyword candidates found in text")
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:top_n]]


_PHRASE_SPLIT_RE = re.compile(r"[^a-z0-9\s-]+")


def keyphrases(text: str, *, top_n: int = 10,
               max_words: int = 3) -> list[str]:
    """Multi-word keyphrases of ``text`` (RAKE-style, stdlib).

    Candidate phrases are content-word runs between stopwords/punctuation;
    each word scores ``degree/frequency`` (RAKE) and phrases sum their
    words.  Returns phrases of up to ``max_words`` words, best first.
    Raises :class:`DocumentError` on empty text or non-positive top_n.
    """
    if top_n <= 0:
        raise DocumentError(f"top_n must be positive, got {top_n}")
    if max_words < 1:
        raise DocumentError(f"max_words must be >= 1, got {max_words}")
    if not text or not text.strip():
        raise DocumentError("cannot extract keyphrases from empty text")
    # Candidate phrases: content-word runs between stopwords/punctuation,
    # extracted per sentence so phrases never span sentence boundaries.
    phrases: list[list[str]] = []
    for sentence in _sentences(text):
        cleaned = _PHRASE_SPLIT_RE.sub(" ", sentence.lower())
        current: list[str] = []
        for token in cleaned.split():
            token = token.strip("-")
            if not token or token in _STOPWORDS or len(token) < 3:
                if current:
                    phrases.append(current)
                    current = []
                continue
            current.append(token)
        if current:
            phrases.append(current)
    phrases = [p for p in phrases if 1 <= len(p) <= max_words]
    if not phrases:
        raise DocumentError("no keyphrase candidates found in text")
    freq: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in phrases:
        for word in phrase:
            freq[word] += 1
            degree[word] += len(phrase)
    word_score = {w: degree[w] / freq[w] for w in freq}
    scored = sorted(
        {tuple(p) for p in phrases},
        key=lambda p: (-sum(word_score[w] for w in p), " ".join(p)),
    )
    return [" ".join(p) for p in scored[:top_n]]
