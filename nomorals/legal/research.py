"""RAG + confidence + citations — the anti-hallucination stack.

"Does not hallucinate" is the marketing message: this module retrieves
from the *verified legal corpus only* and never answers from model
weights. Every result carries a calibrated confidence score and traceable
citations (document + section). When the corpus does not cover the query,
it says so honestly instead of fabricating.

One stack, three surfaces: this module serves #77's consumer legal aid
(:func:`nomorals.legal.aid.answer_legal_question` takes an optional
``researcher``) and #46's source-grounded tutoring, and it powers the
``/research`` chat command directly.

Query confidentiality (Harvey pattern): user queries never enter training
pipelines. This is architectural, not a promise:

- queries live only in memory for the duration of one ``research()`` call;
- nothing is written to disk (the index is in-memory unless the caller
  explicitly passes a file-backed one they own);
- no network calls — retrieval is local SQLite FTS5 BM25; there is no
  embeddings API, no telemetry endpoint, no training hook anywhere in the
  data path;
- results are derived statistics over verified documents, never training
  data.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .contracts import DISCLAIMER, information_only_check

__all__ = [
    "CONFIDENTIALITY",
    "Citation",
    "Evidence",
    "ResearchResult",
    "LegalResearch",
    "control_research",
    "format_research",
]

#: The confidentiality guarantee, as documentation the tests assert exists.
CONFIDENTIALITY = (
    "Queries to LegalResearch are never stored, transmitted, or used for "
    "training. Each query exists only in memory for the duration of the "
    "research() call. Retrieval is local SQLite FTS5 BM25 — no network "
    "calls, no embeddings API, no telemetry, no training hooks. Results are "
    "derived statistics over verified corpus documents. Your questions do "
    "not become Devon's training data."
)

_CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")

#: Minimum top BM25 score before a corpus hit counts as real coverage.
#: Below this the module says "I don't know" rather than fabricate.
_MIN_TOP_SCORE = 1.0
#: Hits considered per query.
_TOP_K = 5
#: Verified seed summaries: real provisions, but explicitly "not the
#: statute text" — quality is high, not perfect.
_SOURCE_QUALITY = 0.85

_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

#: Common words that don't discriminate — excluded when deciding whether a
#: corpus hit genuinely covers the query (the anti-hallucination gate).
_STOPWORDS = frozenset(
    "what which when where who whom whose that this these those then than "
    "with from into your yours their theirs there here have has had were was "
    "been being does did done make made take took your my our their his her "
    "its about over under between through during before after again once "
    "very just only also such like more most some such than then them they "
    "them".split()
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _distinctive_terms(query: str) -> list[str]:
    """Query terms that discriminate — len>=4, not stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+", (query or "").lower())
            if len(t) >= 4 and t not in _STOPWORDS]


def _term_in(text: str, term: str) -> bool:
    """Loose match: exact, depluralized, or light-stem match."""
    low = text.lower()
    if term in low:
        return True
    if len(term) > 4:
        if term.rstrip("s") in low or term[:-2] in low:
            return True
    return False


def _covers(section_text: str, terms: Sequence[str]) -> bool:
    """Does the section genuinely contain the query's distinctive terms?

    The anti-hallucination gate: a BM25 hit sharing only filler words is a
    spurious match. Require at least 2 covered terms (or all of them when
    the query has fewer than 2).
    """
    if not terms:
        return False
    covered = sum(1 for t in terms if _term_in(section_text, t))
    return covered >= min(2, len(terms))


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections on ``##`` headings."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text.strip())]
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1).strip(), text[start:end].strip()))
    return sections


@dataclass
class Citation:
    """One traceable citation: document + section."""
    doc_id: str
    title: str
    section: str
    snippet: str

    def label(self) -> str:
        return f"{self.title} — {self.section}" if self.section else self.title


@dataclass
class Evidence:
    """One ranked piece of corpus evidence."""
    doc_id: str
    title: str
    section: str
    snippet: str
    score: float


@dataclass
class ResearchResult:
    """One researched question. ``answered`` is False when the corpus
    doesn't cover the query — the module says so instead of guessing."""
    query: str
    answered: bool
    text: str = ""
    confidence: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    confidential: bool = True

    def render(self) -> str:
        parts = [self.text]
        if self.answered and self.confidence > 0:
            parts.append(f"📊 Confidence: {self.confidence:.2f} "
                         f"({_confidence_band(self.confidence)}) — "
                         "based on corpus match strength, coverage, and "
                         "verified-source quality.")
        if self.citations:
            parts.append("Sources:\n" + "\n".join(
                f"• {c.label()}" for c in self.citations))
        parts.append(DISCLAIMER)
        return "\n\n".join(parts)


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.55:
        return "moderate"
    return "low — treat as a starting point, not certainty"


def _calibrate(top_score: float, n_docs: int) -> float:
    """Calibrated confidence in [0, 1].

    - ``strength``: monotone, bounded transform of the top BM25 score.
    - ``coverage``: how many distinct verified documents support the answer.
    - ``quality``: corpus docs are verified seed summaries (0.85, not 1.0
      — they are explicitly "not the statute text").
    """
    strength = top_score / (top_score + 4.0)
    coverage = min(1.0, n_docs / 3.0)
    confidence = (0.45 * strength + 0.35 * coverage + 0.20 * _SOURCE_QUALITY)
    return round(max(0.0, min(1.0, confidence)), 2)


class LegalResearch:
    """Grounded legal research over the verified corpus.

    Never answers from model weights: the only words in a result come from
    corpus snippets, or the honest "I don't know" fallback. Never raises.
    """

    def __init__(
        self,
        *,
        corpus_dir: str = _CORPUS_DIR,
        index: Any = None,
        top_k: int = _TOP_K,
    ) -> None:
        self._top_k = max(1, int(top_k or _TOP_K))
        # doc_id -> (title, section heading, section text) — text powers the
        # term-coverage anti-hallucination gate in research().
        self._sections: dict[str, tuple[str, str, str]] = {}
        self._closed = False
        if index is not None:
            self._index = index
            return
        self._index = self._build_index(corpus_dir)

    def _build_index(self, corpus_dir: str) -> Optional[Any]:
        try:
            from ..documents.index import DocumentIndex
            from ..documents.model import Document, Section
        except Exception:  # noqa: BLE001 — never raise from construction
            return None
        try:
            idx = DocumentIndex()
            if not os.path.isdir(corpus_dir):
                return None
            for fname in sorted(os.listdir(corpus_dir)):
                if not fname.endswith(".md"):
                    continue
                path = os.path.join(corpus_dir, fname)
                try:
                    with open(path, encoding="utf-8") as fh:
                        text = fh.read()
                except OSError:
                    continue
                title = (text.splitlines()[0].lstrip("# ").strip()
                         if text else fname)
                stem = fname.replace(".md", "")
                for heading, body in _split_sections(text):
                    if not body:
                        continue
                    doc_id = f"legal-{stem}#{_slug(heading)}"
                    doc = Document(
                        id=doc_id,
                        title=title,
                        source=path,
                        sections=[Section(level=2, heading=heading, text=body)],
                    )
                    try:
                        idx.add(doc)
                    except Exception:  # noqa: BLE001 — one bad doc can't kill it
                        continue
                    self._sections[doc_id] = (title, heading, body)
            return idx if len(idx) else None
        except Exception:  # noqa: BLE001 — legal research must never raise
            return None

    def close(self) -> None:
        if self._index is not None and not self._closed:
            try:
                self._index.close()
            except Exception:  # noqa: BLE001
                pass
            self._closed = True

    def _search(self, query: str) -> list[dict]:
        if self._index is None or self._closed:
            return []
        try:
            return self._index.search(query or "", limit=self._top_k)
        except Exception:  # noqa: BLE001
            return []

    def research(self, query: str) -> ResearchResult:
        """Research ``query`` against the verified corpus. Never raises.

        Returns an honest "I don't know" when the corpus has no coverage —
        it never fabricates an answer from outside the corpus.
        """
        q = (query or "").strip()
        if not q:
            return self._unknown(q, "Ask a legal question first.")

        hits = self._search(q)
        scored = [(h, float(h.get("score") or 0.0)) for h in hits]
        top_score = scored[0][1] if scored else 0.0
        if not scored or top_score < _MIN_TOP_SCORE:
            return self._unknown(q)

        # Term-coverage gate: the top hit must genuinely contain the query's
        # distinctive terms, not just share filler words. Spurious BM25
        # matches are answered honestly ("I don't know"), never fabricated.
        top_doc_id = str(scored[0][0].get("doc_id") or "")
        top_text = self._sections.get(top_doc_id, ("", "", ""))[2]
        if not top_text:
            top_text = str(scored[0][0].get("snippet") or "")
        if not _covers(top_text, _distinctive_terms(q)):
            return self._unknown(q)

        evidence: list[Evidence] = []
        citations: list[Citation] = []
        seen_docs: set[str] = set()
        for h, score in scored:
            doc_id = str(h.get("doc_id") or "")
            title, section, _text = self._sections.get(
                doc_id, (str(h.get("title") or doc_id), "", ""))
            snippet = str(h.get("snippet") or "")
            evidence.append(Evidence(doc_id=doc_id, title=title,
                                    section=section, snippet=snippet,
                                    score=score))
            seen_docs.add(doc_id.split("#")[0])
            citations.append(Citation(doc_id=doc_id, title=title,
                                     section=section, snippet=snippet))

        confidence = _calibrate(top_score, len(seen_docs))
        return ResearchResult(
            query=q,
            answered=True,
            text=self._compose(q, evidence),
            confidence=confidence,
            citations=citations,
            evidence=evidence,
        )

    def _unknown(self, query: str, note: str = "") -> ResearchResult:
        """The honest fallback: no coverage, no fabrication."""
        text = ("I couldn't find anything about this in the legal corpus, so "
                "I won't guess. The corpus covers Lagos tenancy law, the "
                "Labour Act, consumer protection (FCCPC), and constitutional "
                "rights — nothing else yet. " + note).strip()
        return ResearchResult(query=query, answered=False, text=text,
                              confidence=0.0)

    def _compose(self, query: str, evidence: list[Evidence]) -> str:
        """Evidence-only summary: every claim comes from a corpus snippet."""
        lines = ["Here's what the verified legal corpus says about this:"]
        for i, ev in enumerate(evidence, 1):
            heading = f"{ev.title} — {ev.section}" if ev.section else ev.title
            lines.append(f"{i}. *{heading}*\n   {ev.snippet}")
        return "\n\n".join(lines)


def format_research(result: ResearchResult) -> str:
    """Render a :class:`ResearchResult` for chat. Never raises."""
    try:
        return result.render()
    except Exception:  # noqa: BLE001
        return DISCLAIMER


# ── shared stack: wire into #77 aid ────────────────────────────────────

def research_citations(result: ResearchResult) -> list[str]:
    """Citation labels from a research result, deduplicated, order-kept."""
    seen: set[str] = set()
    out: list[str] = []
    for c in result.citations:
        label = c.label()
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


# ── chat ───────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("/research <legal question> — grounded legal research: RAG over the "
            "verified corpus with confidence score and traceable citations. "
            "Legal information, never legal advice; Devon is not a lawyer.\n"
            "Your question is never stored or used for training.\n" + DISCLAIMER)


def control_research(tail: str, context: Any = None, chat: Any = None,
                     sender_id: str = "", sender: str = "") -> str:
    """/research — grounded legal research. Owner-only; never raises."""
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() == "help":
            return _usage()
        researcher: Optional[LegalResearch] = None
        if context is not None:
            researcher = getattr(context, "legal_researcher", None)
        r = LegalResearch() if researcher is None else researcher
        try:
            result = r.research(rest)
        finally:
            if researcher is None:
                r.close()
        # Information-only enforcement: evidence comes from the corpus, but
        # scan anyway so advice-shaped text can never ship.
        hits = information_only_check(result.text)
        if hits:
            result = ResearchResult(query=result.query, answered=result.answered,
                                    text=("Corpus evidence found, but the "
                                          "rendered text tripped the "
                                          "information-only filter; showing "
                                          "citations only."),
                                    confidence=result.confidence,
                                    citations=result.citations,
                                    evidence=result.evidence)
        return format_research(result)
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Research hit an error ({e}). {DISCLAIMER}"
