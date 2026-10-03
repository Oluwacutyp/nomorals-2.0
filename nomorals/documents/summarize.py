"""Extractive summarization and keyword hooks for parsed documents (L4).

:func:`summarize` picks the highest-scoring sentences of a
:class:`~nomorals.documents.model.Document` and returns them in document
order; :func:`summarize_text` does the same for raw text.  Scoring is
term-frequency over content words (stopwords excluded), length-normalized
so long sentences don't win by bulk.  Stdlib only, no models — fast and
deterministic.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .errors import DocumentError
from .model import Document, full_text

__all__ = ["keywords", "summarize", "summarize_text"]

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


def _sentences(text: str) -> list[str]:
    """Split text into sentences; keeps order, drops fragments < 4 words."""
    chunks = _SENTENCE_SPLIT_RE.split(text.strip())
    out: list[str] = []
    for chunk in chunks:
        clean = " ".join(chunk.split())
        if len(_WORD_RE.findall(clean.lower())) >= 4:
            out.append(clean)
    return out


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower())
            if w not in _STOPWORDS and len(w) >= 3]


def _score_sentences(sentences: list[str]) -> list[float]:
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


def summarize_text(text: str, *, sentences: int = 5) -> list[str]:
    """Return the ``sentences`` most representative sentences of ``text``.

    Sentences come back in document order.  Raises :class:`DocumentError`
    on empty text or a non-positive sentence count.
    """
    if sentences <= 0:
        raise DocumentError(f"sentences must be positive, got {sentences}")
    if not text or not text.strip():
        raise DocumentError("cannot summarize empty text")
    sents = _sentences(text)
    if not sents:
        raise DocumentError("no summarizable sentences found in text")
    if len(sents) <= sentences:
        return sents
    scores = _score_sentences(sents)
    # Top-N by score, ties broken by document order (deterministic).
    ranked = sorted(range(len(sents)), key=lambda i: (-scores[i], i))
    chosen = sorted(ranked[:sentences])
    return [sents[i] for i in chosen]


def summarize(doc: Document, *, sentences: int = 5) -> list[str]:
    """Extractive summary of a parsed :class:`Document`.

    Raises :class:`DocumentError` when the document has no summarizable
    text.
    """
    if not isinstance(doc, Document):
        raise DocumentError("summarize needs a Document object")
    return summarize_text(full_text(doc), sentences=sentences)


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
