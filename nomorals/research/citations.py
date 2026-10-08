"""Claim-level evidence and citation infrastructure (build-map #20).

Extends the existing "[S<n>] in, deterministic numbers out, invented
citations stripped" discipline (``pipeline._map_citations``,
``grounded._number_citations``) with a formal path:

* ``CitationManager`` — registered sources with sha256 content hashes
  and access timestamps, quote-verified citations, deterministic
  Works Cited, and an audit trail for drift auditing (pages change;
  the hash proves what we saw).
* ``extract_claims`` — honest OFFLINE extractive claim extraction
  (heuristic, documented). An LLM re-ranker hook is optional.
* ``verify_claims`` / ``sentence_supported`` — the code side of the
  loop-closer: the model can't smuggle an unsupported claim past a
  valid-looking citation.

Never invents quotes or citations: ``cite()`` raises on an unknown
quote; the pipeline strips sentences whose citations don't verify.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger("nomorals.research.citations")


# ── claims ────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """One factual assertion extracted from a source sentence."""

    text: str        # the claim sentence
    quote: str       # exact source sentence it came from
    confidence: str  # "high" | "medium" | "low"
    source_id: str   # e.g. "S1"
    verified: bool = False


# Small local stopword set — deliberately self-contained so this module
# never imports the pipeline (which imports this module inside synthesize).
_STOPWORDS = frozenset(
    "a an the and or but if then else when at by for with about into through "
    "during before after above below to from up down in out on off over under "
    "again further once here there all any both each few more most other some "
    "such no nor not only own same so than too very can will just should now "
    "me my i you your we us is are was were be been being have has had do "
    "does did of as it its this that these those am s t tell said says say "
    "also per".split()
)

#: predications that signal a factual assertion ("X is …", "X has …")
_PREDICATION = re.compile(r"\b(is|are|was|were|has|have)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")


def _sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empties and headers."""
    out: list[str] = []
    for s in _SENT_SPLIT.split(re.sub(r"\s+", " ", text or "").strip()):
        s = s.strip()
        if not s or s.startswith("#") or len(s.split()) < 4:
            continue
        out.append(s)
    return out


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower())
            if w not in _STOPWORDS and len(w) > 2]


def _named_entities(sentence: str) -> list[str]:
    """Heuristic named entities: capitalised words beyond sentence start."""
    words = re.findall(r"[A-Z][a-zA-Z]{1,}", sentence)
    first = sentence.split()[0] if sentence.split() else ""
    return [w for w in words if w != first.strip(".,;:!?\"'()")]


def _claim_confidence(sentence: str) -> str:
    """Signal-strength confidence for an extractive claim.

    Has a number AND (a named entity or an is/are/was/were/has/have predication)
    → high. Has a number or a named entity → medium. Vague → low.
    """
    has_number = bool(_NUMBER.search(sentence))
    has_entity = bool(_named_entities(sentence))
    has_pred = bool(_PREDICATION.search(sentence))
    if has_number and (has_entity or has_pred):
        return "high"
    if has_number or has_entity:
        return "medium"
    return "low"


def _is_factual(sentence: str) -> bool:
    """Heuristic: does the sentence make a factual assertion?"""
    s = sentence.strip()
    if s.endswith("?"):
        return False
    if _NUMBER.search(s):
        return True
    if _named_entities(s):
        return True
    return bool(_PREDICATION.search(s))


def extract_claims(source_text: str, *, source_id: str,
                   rerank_fn: Any = None) -> list[Claim]:
    """Extract factual claims from source text, OFFLINE.

    Splits into sentences, keeps sentences with factual assertions
    (numbers, named entities, is/are/was/were/has/have predications —
    heuristic,
    documented above), and pairs each claim with its exact source
    sentence as the quote. ``rerank_fn`` is an optional hook
    ``(claims) -> claims`` for an LLM re-ranker; the extractive
    baseline works without it.
    """
    claims = [
        Claim(text=s, quote=s, confidence=_claim_confidence(s),
              source_id=source_id)
        for s in _sentences(source_text)
        if _is_factual(s)
    ]
    if rerank_fn is not None:
        try:
            reranked = rerank_fn(claims)
            if reranked:
                return list(reranked)
        except Exception as exc:  # noqa: BLE001 - rerank is advisory only
            _log.debug("extract_claims rerank failed (%s), keeping baseline", exc)
    return claims


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def verify_claims(claims: list[Claim], source_text: str) -> list[Claim]:
    """Keep only claims whose quote is in the source text.

    Exact or near-exact match (whitespace-normalised, case-insensitive,
    trailing punctuation ignored). Marks survivors verified. Never
    raises — a bad claim is dropped, not fatal.
    """
    hay = _norm(source_text)
    out: list[Claim] = []
    for claim in claims or []:
        try:
            needle = _norm(claim.quote).rstrip(".,;:!?")
            if needle and (needle in hay or _norm(claim.quote) in hay):
                claim.verified = True
                out.append(claim)
            else:
                _log.debug("verify_claims: dropped unsupported claim %r",
                           claim.text[:60])
        except Exception:  # noqa: BLE001 - never fail verification
            continue
    return out


def sentence_supported(sentence: str, source_text: str) -> bool:
    """Does ``source_text`` support the factual content of ``sentence``?

    Heuristic, documented and strict on purpose (this is the
    loop-closer for citations):

    * every number in the sentence must appear in the source
      (numbers are the highest-signal check — a smuggled figure fails);
    * otherwise, content-word overlap: >= 50% of the sentence's
      content words when numbers are present, >= 60% without;
      sentences under 4 content words need all of them.
    * sentences with no content words and no numbers pass (can't judge
      transitional prose — the citation gate doesn't punish it).
    """
    hay = _norm(source_text)
    words = _content_words(sentence)
    numbers = {n.replace(",", "") for n in _NUMBER.findall(sentence)}
    hay_numbers = {n.replace(",", "") for n in _NUMBER.findall(hay)}
    if numbers and not numbers <= hay_numbers:
        return False  # invented or mismatched figure
    if not words:
        return True
    hay_words = set(_WORD.findall(hay))
    hits = sum(1 for w in words if w in hay_words)
    if len(words) < 4:
        return hits == len(words)
    threshold = 0.5 if numbers else 0.6
    return hits / len(words) >= threshold


# ── citation manager ──────────────────────────────────────────────────────

@dataclass
class _SourceRecord:
    id: str
    url: str
    title: str
    text: str
    sha256: str
    accessed_ts: float


#: model output markers: [S1], [s2], [S 3] …
_MARKER_RE = re.compile(r"\[\s*S\s*(\d+)\s*\]", re.IGNORECASE)


class CitationManager:
    """Registered sources, verified citations, deterministic Works Cited.

    Typical flow::

        mgr = CitationManager()
        sid = mgr.register_source(url, title, text)   # "S1"
        marker = mgr.cite(sid, exact_quote)           # "[S1]"
        text, works = mgr.resolve(model_output)       # "[1]" + Works Cited
        trail = mgr.audit_trail()                     # drift-audit evidence
    """

    def __init__(self, *, clock: Any = None) -> None:
        self._sources: dict[str, _SourceRecord] = {}
        self._url_to_id: dict[str, str] = {}
        self._citations: list[tuple[str, str]] = []  # (source_id, quote)
        self._counter = 0
        self._clock = clock or time.time

    # -- registration ----------------------------------------------------
    def register_source(self, url: str, title: str, text: str,
                        *, accessed_ts: float | None = None) -> str:
        """Register a source; returns its id (``S1``, ``S2``, …).

        The same URL registered twice returns the same id (dedupe —
        first registration wins). Stores the sha256 content hash and
        access timestamp for drift auditing.
        """
        url = (url or "").strip()
        if url in self._url_to_id:
            return self._url_to_id[url]
        self._counter += 1
        sid = f"S{self._counter}"
        body = text or ""
        rec = _SourceRecord(
            id=sid,
            url=url,
            title=(title or url or sid).strip(),
            text=body,
            sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            accessed_ts=accessed_ts if accessed_ts is not None
            else float(self._clock()),
        )
        self._sources[sid] = rec
        self._url_to_id[url] = sid
        return sid

    def get(self, source_id: str) -> _SourceRecord | None:
        return self._sources.get(source_id)

    # -- citing -----------------------------------------------------------
    def cite(self, source_id: str, quote: str) -> str:
        """Record a citation of ``source_id`` with an exact quote.

        Returns the inline marker ``[S<n>]``. The quote MUST appear in
        the source text (case-insensitive substring) — raises
        ``ValueError`` on an invented quote or an unknown source.
        Fail fast: no invented quotes, ever.
        """
        rec = self._sources.get(source_id)
        if rec is None:
            raise ValueError(f"cite: unknown source {source_id!r}")
        q = (quote or "").strip()
        if not q or q.lower() not in rec.text.lower():
            raise ValueError(
                f"cite: quote not found in {source_id} — invented quotes "
                f"are not allowed")
        self._citations.append((source_id, q))
        return f"[{source_id}]"

    # -- resolution --------------------------------------------------------
    def resolve(self, text: str) -> tuple[str, list[str]]:
        """Remap ``[S<n>]`` markers to deterministic ``[1]``, ``[2]``…

        Numbers follow order of first appearance (mirroring
        ``pipeline._map_citations``). Markers pointing at unknown
        sources are dropped. Returns the resolved text and the
        Works Cited lines::

            [1] Title — url (accessed 2026-10-08, sha256:abc123…)
        """
        order: list[str] = []

        def repl(m: re.Match[str]) -> str:
            sid = f"S{int(m.group(1))}"
            if sid not in self._sources:
                return ""  # citation to nothing — strip it
            if sid not in order:
                order.append(sid)
            return f"[{order.index(sid) + 1}]"

        resolved = _MARKER_RE.sub(repl, text or "")

        works = []
        for i, sid in enumerate(order, 1):
            rec = self._sources[sid]
            day = datetime.fromtimestamp(
                rec.accessed_ts, tz=timezone.utc).strftime("%Y-%m-%d")
            works.append(
                f"[{i}] {rec.title} — {rec.url} "
                f"(accessed {day}, sha256:{rec.sha256[:12]}…)")
        return resolved, works

    # -- audit --------------------------------------------------------------
    def audit_trail(self) -> list[dict]:
        """Per-source evidence: id, url, title, content hash, access
        time, and the quotes actually cited from it.

        The hash is the drift audit: re-fetch the URL later and compare
        hashes to prove what the research run actually saw.
        """
        quotes_by_source: dict[str, list[str]] = {}
        for sid, quote in self._citations:
            quotes_by_source.setdefault(sid, []).append(quote)
        trail = []
        for sid, rec in self._sources.items():
            trail.append({
                "source_id": sid,
                "url": rec.url,
                "title": rec.title,
                "sha256": rec.sha256,
                "accessed_ts": rec.accessed_ts,
                "quotes": quotes_by_source.get(sid, []),
            })
        return trail
