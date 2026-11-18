"""Requirements analysis (dynavio REA core, 52-websites #51).

Dynavio Cooperative Oy's REA — Requirement Extractor and Analyser —
mined from the published method (Mokammel, Coatanéa, Nenchev et al.,
"Automatic requirements extraction, analysis, and graph representation
using an approach derived from computational linguistics", *Systems
Engineering* 2018): feed it a natural-language specification and it
(1) detects and extracts requirement statements, (2) categorizes them,
(3) scores their quality against the classic requirement smells,
(4) analyzes relations between them, and (5) represents the result as
a graph. Devon-native on the :class:`Document` model — analysis
results attach to a parsed document, exportable like every other
document product.

Two rungs, per the implementation-ladder doctrine. The deterministic
code rung is first and fully offline: modal-verb/pattern extraction,
a smell lexicon, TF-IDF cosine + latent-semantic (truncated-SVD)
similarity + clustering, and keyword dependency detection. The brain
rung adds judgment on top — reclassification, contradiction calls,
and missed-requirement recovery — but every brain output must cite a
real requirement id or a verbatim span of the source document, or it
is dropped. Nothing here is ever invented: every requirement names
its exact span (character offsets into the document's full text),
every flag names its evidence word, every link names its method.

The honest limit, stated in the mining report and kept here: the
code rung measures similarity; contradiction is judgment. The code
rung NEVER emits a ``contradicts`` link — only the brain rung does,
and each such link is labelled ``judgment=True`` with its reason,
open to owner override.

Identification note: the owner's list names "dynavio" with no URL.
The only product-bearing Dynavio on record is Dynavio Cooperative
Oy's REA (dynavio.com itself is a parked domain); this organ mines
that system shape. If the owner meant a different "dynavio", the
shape below is still the only evidence-backed reading.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from .model import Document, Section, full_text, new_document
from .parsers import parse_path as _parse_path

_log = get_logger(__name__)

__all__ = [
    "Span",
    "QualityFlag",
    "RequirementLink",
    "Requirement",
    "RequirementGraph",
    "RequirementAnalysis",
    "RequirementStore",
    "open_requirements_store",
    "extract_requirements",
    "score_links",
    "analyze_document",
    "analyze_corpus",
    "verify_spans",
    "format_requirements_reply",
    "render_report_markdown",
    "write_report",
]

#: Requirement kinds (the report sketch's four values).
REQUIREMENT_KINDS = frozenset({"functional", "quality", "constraint",
                               "unknown"})

#: Quality-flag kinds (the classic requirement smells).
FLAG_KINDS = frozenset({"ambiguous", "weak_verb", "unverifiable",
                        "incomplete", "compound"})

#: Link kinds. ``contradicts`` is brain-rung only (judgment=True).
LINK_KINDS = frozenset({"depends", "similar", "contradicts"})

#: Link methods: how the link was measured or judged.
LINK_METHODS = frozenset({"tfidf_cosine", "lsa_cosine", "keyword",
                          "brain"})

#: Similarity threshold for a ``similar`` link (cosine, 0..1).
SIMILARITY_THRESHOLD = 0.5

#: LSA latent dimensions (truncated SVD rank cap).
LSA_RANK = 10

#: Store file under <home>/data/.
STORE_FILENAME = "requirements.json"
REPORT_DIRNAME = "requirements_reports"

# ── lexicon ──────────────────────────────────────────────────────────────

#: Modal verbs that mark a requirement statement (REA's modal patterns).
_MODAL_RE = re.compile(
    r"\b(shall|must|should|is required to|are required to|needs? to"
    r"|has to|have to)\b", re.IGNORECASE)

#: "will" only counts with a system subject (a future commitment,
#: not a weather forecast).
_WILL_RE = re.compile(
    r"\b(the system|the service|the api|the application|the platform"
    r"|devon|the user)\b[^.]{0,40}\bwill\b", re.IGNORECASE)

#: Explicit requirement ids: REQ-12, FR-3, NFR-7, BR-2 …
_ID_RE = re.compile(r"\b([A-Z]{2,}-\d+)\b")

#: A leading id on a unit ("REQ-1: …", "[FR-3] …").
_LEADING_ID_RE = re.compile(r"^\s*\[?([A-Z]{2,}-\d+)\]?\s*[:.\-–—]\s*")

#: Imperative sentence openers (a bare command is a requirement).
_IMPERATIVE_STARTERS = frozenset({
    "support", "provide", "allow", "encrypt", "log", "store", "send",
    "display", "validate", "enforce", "ensure", "require", "return",
    "generate", "create", "delete", "retain", "rotate", "audit",
    "authenticate", "record", "notify", "reject", "accept", "handle",
    "process", "check", "verify", "export", "import", "backup",
    "restore", "deploy", "monitor", "alert", "block", "permit",
})

#: Vague terms — the ambiguity smell (REA's classic list, extended
#: with the report sketch's own examples: "appropriate", "as required").
AMBIGUOUS_TERMS = (
    "appropriate", "appropriately", "as required", "as needed",
    "sufficient", "adequate",
    "suitable", "user-friendly", "fast", "robust", "flexible",
    "seamless", "easy", "easily", "quickly", "promptly", "several",
    "some", "various", "etc.", "and/or", "may", "might", "could",
    "usually", "generally", "normally", "ideally", "preferably",
    "in a timely manner", "high quality", "good", "timely",
)

#: Weak verbs — a requirement that says what without saying how much
#: of what (REA's weak-verb smell).
WEAK_VERBS = (
    "support", "handle", "manage", "process", "provide", "allow",
    "enable", "facilitate", "deal with",
)

#: Quality claims that are unverifiable without a number — a
#: performance-flavoured claim with no digit anywhere in the sentence.
UNVERIFIABLE_CUES = (
    "performance", "latency", "response time", "throughput", "quickly",
    "fast", "slow", "efficient", "load time",
)

#: Incompleteness markers.
_INCOMPLETE_RES = (
    re.compile(r"\b(TBD|TODO)\b", re.IGNORECASE),
    re.compile(r"\bto be (determined|defined|decided)\b", re.IGNORECASE),
    re.compile(r"\?\s*$"),
    re.compile(r"(…|\.\.\.)\s*$"),
    # a dangling reference: "as defined in." / "see section." with no target
    re.compile(r"\b(as defined in|see section|see appendix)\s*[.!?]?$",
               re.IGNORECASE),
    # a modal + weak verb with nothing after it: "shall support."
    re.compile(r"\b(shall|must|should)\s+(support|provide|handle|manage)"
               r"\s*[.!?]?$", re.IGNORECASE),
)

#: Constraint cues (checked before quality cues).
_CONSTRAINT_RES = (
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bcompliance\b|\bGDPR\b|\bregulation\b|\bregulatory\b",
               re.IGNORECASE),
    re.compile(r"\blicen[cs]e\b", re.IGNORECASE),
    re.compile(r"\b(must|shall) run on\b", re.IGNORECASE),
    re.compile(r"\boperating system\b", re.IGNORECASE),
    re.compile(r"\bstandard\s+(ISO|IEEE|WCAG)\b"),
)

#: Quality cues (checked after constraint cues).
_QUALITY_RES = (
    re.compile(r"\bperformance\b|\blatency\b|\bresponse time\b"
               r"|\bthroughput\b", re.IGNORECASE),
    re.compile(r"\bavailability\b|\buptime\b|\breliab", re.IGNORECASE),
    re.compile(r"\bsecurity\b|\bsecur\w*\b|\bencrypt\w*\b",
               re.IGNORECASE),
    re.compile(r"\bauthent\w*\b|\bauthori\w*\b", re.IGNORECASE),
    re.compile(r"\busab\w*\b|\baccessib\w*\b|\bWCAG\b", re.IGNORECASE),
    re.compile(r"\bscalab\w*\b|\bmaintainab\w*\b|\btestab\w*\b",
               re.IGNORECASE),
    re.compile(r"\bquickly\b|\bfast\b|\bslow\b", re.IGNORECASE),
    re.compile(r"\bwithin\s+\d", re.IGNORECASE),
    re.compile(r"\b\d+(\.\d+)?\s*(ms|milliseconds?|seconds?|minutes?)\b",
               re.IGNORECASE),
    re.compile(r"\b99\.\d+\s*%", re.IGNORECASE),
)

#: Dependency verbs for the keyword ``depends`` link. Temporal words
#: ("after", "before") are deliberately NOT here — "after a user
#: registers" is a sequence, not a dependency between requirements.
_DEPENDENCY_VERBS = ("requires", "require", "depends on",
                     "prerequisite", "relies on", "builds on")

#: Content-word stopwords for TF-IDF (small stdlib set).
_STOPWORDS = frozenset([
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "let",
    "me",
    "more",
    "most",
    "my",
    "nor",
    "of",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "out",
    "over",
    "own",
    "same",
    "shan",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "would",
    "within",
    "without",
    "per",
    "via",
    "etc",
])

#: Modal scaffolding is not content: "shall" in every requirement
#: would make everything "similar" to everything.
_STOPWORDS = _STOPWORDS | {"shall", "must", "may", "might"}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Light suffix stemmer (Porter-lite): conflates email/emails,
    user/users, registers/registered so near-duplicate requirements
    actually measure as similar. Deterministic, no tables."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        stem = word[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("ed"):
        stem = word[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _content_words(text: str) -> list[str]:
    """Stemmed content words: lowercase alphanumerics, len >= 3,
    stopwords removed."""
    return [_stem(w) for w in _WORD_RE.findall(text.lower())
            if len(w) >= 3 and w not in _STOPWORDS]


# ── shapes ───────────────────────────────────────────────────────────────

@dataclass
class Span:
    """Where a requirement lives: character offsets into the
    document's full text, plus the human-readable provenance. The
    ``text`` at ``[start:end)`` is verbatim — never paraphrased."""

    section: str = ""
    page: int = 0
    start: int = 0
    end: int = 0
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Span:
        return cls(
            section=str(data.get("section", "") or ""),
            page=int(data.get("page", 0) or 0),
            start=int(data.get("start", 0) or 0),
            end=int(data.get("end", 0) or 0),
            text=str(data.get("text", "") or ""),
        )


@dataclass
class QualityFlag:
    """One requirement smell. ``evidence`` is the exact word or phrase
    in the requirement text that triggered the flag."""

    kind: str = ""
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityFlag:
        kind = str(data.get("kind", "") or "")
        return cls(kind=kind if kind in FLAG_KINDS else "ambiguous",
                   evidence=str(data.get("evidence", "") or ""))


@dataclass
class RequirementLink:
    """One relation between two requirements. ``score`` is a measured
    similarity (0..1) for similar/depends links; for a brain
    ``contradicts`` link it is the brain's stated confidence, and the
    link is labelled ``judgment=True`` — a judgment, not a measurement."""

    to: str = ""
    kind: str = "similar"
    score: float = 0.0
    method: str = "tfidf_cosine"
    judgment: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequirementLink:
        kind = str(data.get("kind", "similar") or "similar")
        method = str(data.get("method", "tfidf_cosine") or "tfidf_cosine")
        return cls(
            to=str(data.get("to", "") or ""),
            kind=kind if kind in LINK_KINDS else "similar",
            score=float(data.get("score", 0.0) or 0.0),
            method=method if method in LINK_METHODS else "tfidf_cosine",
            judgment=bool(data.get("judgment", False)),
            reason=str(data.get("reason", "") or ""),
        )


@dataclass
class Requirement:
    """One extracted requirement statement. ``id`` is the document's
    own id when the statement carries one (``REQ-12``), else a
    generated ``REQ-<n>`` in document order. ``kind`` is functional /
    quality / constraint / unknown. ``links`` are filled by
    :func:`score_links` (code rung) and the brain rung."""

    id: str = ""
    doc_ref: str = ""
    span: Span = field(default_factory=Span)
    text: str = ""
    kind: str = "unknown"
    quality_flags: list[QualityFlag] = field(default_factory=list)
    links: list[RequirementLink] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["span"] = self.span.to_dict()
        data["quality_flags"] = [f.to_dict() for f in self.quality_flags]
        data["links"] = [ln.to_dict() for ln in self.links]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirement:
        kind = str(data.get("kind", "unknown") or "unknown")
        span = data.get("span") or {}
        return cls(
            id=str(data.get("id", "") or ""),
            doc_ref=str(data.get("doc_ref", "") or ""),
            span=Span.from_dict(span) if isinstance(span, dict) else Span(),
            text=str(data.get("text", "") or ""),
            kind=kind if kind in REQUIREMENT_KINDS else "unknown",
            quality_flags=[QualityFlag.from_dict(f) for f in
                           data.get("quality_flags", [])
                           if isinstance(f, dict)],
            links=[RequirementLink.from_dict(ln) for ln in
                   data.get("links", []) if isinstance(ln, dict)],
        )


@dataclass
class RequirementAnalysis:
    """One analysis run: the requirements of a document (or a pooled
    corpus), their graph, and how the run went. ``ok`` is False when
    the intake itself failed (missing file, no input) — the error is
    named, never swallowed."""

    analysis_id: str = ""
    doc_ref: str = ""
    doc_title: str = ""
    doc_revision: int = 0
    created_at: float = 0.0
    requirements: list[Requirement] = field(default_factory=list)
    rung: str = "code"
    brain_rung: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""

    def __post_init__(self) -> None:
        if not self.analysis_id:
            self.analysis_id = ulid_now()
        if not self.created_at:
            self.created_at = time.time()

    @property
    def graph(self) -> RequirementGraph:
        """The requirement graph over this run's requirements."""
        return RequirementGraph(self.requirements)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["requirements"] = [r.to_dict() for r in self.requirements]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequirementAnalysis:
        return cls(
            analysis_id=str(data.get("analysis_id", "") or ulid_now()),
            doc_ref=str(data.get("doc_ref", "") or ""),
            doc_title=str(data.get("doc_title", "") or ""),
            doc_revision=int(data.get("doc_revision", 0) or 0),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            requirements=[Requirement.from_dict(r) for r in
                          data.get("requirements", [])
                          if isinstance(r, dict)],
            rung=str(data.get("rung", "code") or "code"),
            brain_rung=str(data.get("brain_rung", "") or ""),
            summary=dict(data.get("summary", {}) or {}),
            ok=bool(data.get("ok", True)),
            error=str(data.get("error", "") or ""),
        )


class RequirementGraph:
    """The set + link index: requirements as nodes, dependencies /
    similarities / contradictions as edges."""

    def __init__(self, requirements: list[Requirement]) -> None:
        self.requirements = list(requirements)
        self.by_id: dict[str, Requirement] = {
            r.id: r for r in self.requirements}

    def neighbors(self, req_id: str) -> list[str]:
        """Ids linked from ``req_id`` (any kind)."""
        req = self.by_id.get(req_id or "")
        if req is None:
            return []
        return [ln.to for ln in req.links]

    def links_from(self, req_id: str,
                   kind: str = "") -> list[RequirementLink]:
        """Links out of ``req_id``, optionally filtered by kind."""
        req = self.by_id.get(req_id or "")
        if req is None:
            return []
        return [ln for ln in req.links if not kind or ln.kind == kind]

    def contradictions(self) -> list[dict[str, Any]]:
        """All ``contradicts`` links, each as {a, b, reason, method,
        judgment, score}. Brain judgments only — the code rung never
        produces these."""
        out = []
        for req in self.requirements:
            for ln in req.links:
                if ln.kind == "contradicts":
                    out.append({"a": req.id, "b": ln.to,
                                "reason": ln.reason,
                                "method": ln.method,
                                "judgment": ln.judgment,
                                "score": ln.score})
        return out

    def clusters(self, min_score: float = SIMILARITY_THRESHOLD
                 ) -> list[list[str]]:
        """Connected components over ``similar`` links at >= min_score
        (single-link clustering, deterministic order). Singletons are
        not clusters — a requirement that stands alone is not a group."""
        parent: dict[str, str] = {r.id: r.id for r in self.requirements}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        for req in self.requirements:
            for ln in req.links:
                if ln.kind == "similar" and ln.score >= min_score \
                        and ln.to in parent:
                    union(req.id, ln.to)
        groups: dict[str, list[str]] = {}
        for req in self.requirements:
            groups.setdefault(find(req.id), []).append(req.id)
        return sorted([sorted(g) for g in groups.values() if len(g) > 1])

    def orphans(self) -> list[str]:
        """Requirement ids with no links at all."""
        return [r.id for r in self.requirements if not r.links]

    def summary(self) -> dict[str, Any]:
        """Graph summary: counts by kind, links by kind, clusters,
        orphans, contradictions."""
        kinds: dict[str, int] = {}
        flags: dict[str, int] = {}
        for req in self.requirements:
            kinds[req.kind] = kinds.get(req.kind, 0) + 1
            for fl in req.quality_flags:
                flags[fl.kind] = flags.get(fl.kind, 0) + 1
        similar = depends = contradicts = 0
        for req in self.requirements:
            for ln in req.links:
                if ln.kind == "similar":
                    similar += 1
                elif ln.kind == "depends":
                    depends += 1
                elif ln.kind == "contradicts":
                    contradicts += 1
        return {
            "requirements": len(self.requirements),
            "by_kind": kinds,
            "flags": flags,
            "similar_links": similar // 2,
            "depends_links": depends,
            "contradictions": contradicts,
            "clusters": len(self.clusters()),
            "orphans": len(self.orphans()),
        }

# ── segmentation ─────────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'“‘(\[]?[A-Z0-9])")
_BULLET_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)]|[a-z][.)])\s+")


def _split_units(text: str) -> list[tuple[int, int, str]]:
    """Split prose into candidate units with (start, end, text) offsets
    into ``text``. Blank lines are hard boundaries; bullet/list lines
    are their own units; the rest splits on sentence boundaries.
    Units under 3 words are dropped — a heading fragment is not a
    requirement."""
    units: list[tuple[int, int, str]] = []
    for para in re.split(r"\n\s*\n", (text or "").strip()):
        collapsed = " ".join(para.split())
        if not collapsed:
            continue
        # Locate the paragraph in the original text for offsets.
        base = (text or "").find(para)
        if base < 0:
            base = 0
        for line in para.split("\n"):
            line = line.strip()
            if not line:
                continue
            if _BULLET_RE.match(line):
                # A bullet is one unit, even if it holds two sentences:
                # the bullet is the requirement, not its sentences.
                start = (text or "").find(line, base)
                if start < 0:
                    continue
                units.append((start, start + len(line), line))
                continue
            for chunk in _SENTENCE_SPLIT_RE.split(line):
                clean = " ".join(chunk.split())
                if len(_WORD_RE.findall(clean.lower())) < 3:
                    continue
                start = (text or "").find(clean, base)
                if start < 0:
                    continue
                units.append((start, start + len(clean), clean))
    units.sort()
    return units


# ── detection / classification / smells ──────────────────────────────────

def _leading_id(unit: str) -> str:
    match = _LEADING_ID_RE.match(unit)
    return match.group(1) if match else ""


def _is_requirement(unit: str) -> bool:
    """Modal-verb / pattern detection (REA's detection step)."""
    if _leading_id(unit):
        return True
    if _MODAL_RE.search(unit):
        return True
    if _WILL_RE.search(unit):
        return True
    first = (unit.split() or [""])[0].strip("\"'“‘([].,;:").lower()
    return first in _IMPERATIVE_STARTERS


def _classify(text: str) -> str:
    """functional / quality / constraint by cue priority (constraint
    first, then quality, else functional)."""
    for rx in _CONSTRAINT_RES:
        if rx.search(text):
            return "constraint"
    for rx in _QUALITY_RES:
        if rx.search(text):
            return "quality"
    return "functional"


def _quality_flags(text: str, kind: str) -> list[QualityFlag]:
    """The classic requirement smells, each with its verbatim evidence."""
    flags: list[QualityFlag] = []
    lowered = text.lower()
    for term in AMBIGUOUS_TERMS:
        # Phrases match as written; single words match on word
        # boundaries only ("may" must not fire inside "maybe", and
        # "appropriate" must not fire inside "appropriately" — that
        # word has its own entry above).
        if " " in term or term in ("etc.", "and/or"):
            idx = lowered.find(term.lower())
            if idx < 0:
                continue
            evidence = text[idx:idx + len(term)]
        else:
            match = re.search(r"\b" + re.escape(term) + r"\b", text,
                              re.IGNORECASE)
            if not match:
                continue
            evidence = match.group(0)
        flags.append(QualityFlag(kind="ambiguous", evidence=evidence))
    for verb in WEAK_VERBS:
        match = re.search(r"\b" + re.escape(verb) + r"\b", text,
                          re.IGNORECASE)
        if match:
            flags.append(QualityFlag(kind="weak_verb",
                                     evidence=match.group(0)))
    if kind == "quality" and not re.search(r"\d", text):
        for cue in UNVERIFIABLE_CUES:
            match = re.search(re.escape(cue), text, re.IGNORECASE)
            if match:
                flags.append(QualityFlag(kind="unverifiable",
                                         evidence=match.group(0)))
                break
    for rx in _INCOMPLETE_RES:
        match = rx.search(text)
        if match:
            flags.append(QualityFlag(kind="incomplete",
                                     evidence=match.group(0).strip()))
            break
    modals = _MODAL_RE.findall(text)
    if len(modals) >= 2:
        flags.append(QualityFlag(kind="compound", evidence=modals[0]))
    return flags


# ── extraction (code rung) ───────────────────────────────────────────────

def extract_requirements(doc: Document,
                         doc_ref: str | None = None) -> list[Requirement]:
    """Extract requirements from a parsed :class:`Document` (code rung).

    Every requirement names its exact span: character offsets into
    ``full_text(doc)`` whose text is the requirement verbatim. Explicit
    ids (``REQ-12``) survive; the rest get sequential ``REQ-<n>`` ids
    in document order, the counter running past explicit ids so ids
    never collide.
    """
    ref = doc_ref or doc.id or ""
    text = full_text(doc)
    requirements: list[Requirement] = []
    counter = 1

    def next_id(explicit: str) -> str:
        nonlocal counter
        if explicit:
            match = re.match(r"REQ-(\d+)$", explicit)
            if match:
                counter = max(counter, int(match.group(1)) + 1)
            return explicit
        rid = f"REQ-{counter}"
        counter += 1
        return rid

    cursor = 0
    for section in doc.sections:
        body = section.text or ""
        if not body.strip():
            continue
        base = text.find(body, cursor)
        if base < 0:
            # The section text is not in the full text verbatim (a
            # hand-built Document whose parts were reordered) — fall
            # back to locating each unit in the full text directly.
            base = 0
        else:
            cursor = base + len(body)
        for start, end, unit in _split_units(body):
            if not _is_requirement(unit):
                continue
            explicit = _leading_id(unit)
            has_modal = bool(_MODAL_RE.search(unit) or
                             _WILL_RE.search(unit))
            first = (unit.split() or [""])[0].strip(
                "\"'“‘([].,;:").lower()
            has_imperative = first in _IMPERATIVE_STARTERS
            if explicit and not has_modal and not has_imperative:
                kind = "unknown"
            else:
                kind = _classify(unit)
            abs_start = base + start if base else start
            abs_end = base + end if base else end
            # Guard the span: it must be verbatim in the full text.
            if text[abs_start:abs_end] != unit:
                found = text.find(unit, cursor - len(body) if cursor
                                  else 0)
                if found < 0:
                    found = text.find(unit)
                if found < 0:
                    continue
                abs_start, abs_end = found, found + len(unit)
            span = Span(section=section.heading or "",
                        page=int(section.page or 0),
                        start=abs_start, end=abs_end, text=unit)
            requirements.append(Requirement(
                id=next_id(explicit),
                doc_ref=ref,
                span=span,
                text=unit,
                kind=kind,
                quality_flags=_quality_flags(unit, kind),
            ))
    return requirements


def verify_spans(analysis: RequirementAnalysis,
                 doc: Document) -> list[str]:
    """Check every requirement's span against the document: the span
    text must be the verbatim substring at [start:end). Returns the
    problems (empty = all spans verbatim)."""
    text = full_text(doc)
    problems = []
    for req in analysis.requirements:
        got = text[req.span.start:req.span.end]
        if got != req.text or req.span.text != req.text:
            problems.append(
                f"{req.id}: span {req.span.start}:{req.span.end} "
                f"reads {got!r}, expected {req.text!r}")
    return problems

# ── similarity: TF-IDF cosine + latent semantics ─────────────────────────

def _tfidf_vectors(requirements: list[Requirement]
                   ) -> tuple[list[dict[str, float]], list[str]]:
    """Sparse TF-IDF vectors (stemmed content words) + vocabulary, in
    requirement order. Deterministic: term order is first-seen order."""
    docs = [_content_words(r.text) for r in requirements]
    vocab: list[str] = []
    seen: set[str] = set()
    for words in docs:
        for word in words:
            if word not in seen:
                seen.add(word)
                vocab.append(word)
    n_docs = len(docs)
    df = {term: sum(1 for words in docs if term in words)
          for term in vocab}
    vectors: list[dict[str, float]] = []
    for words in docs:
        tf: dict[str, int] = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
            vec[term] = count * idf
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm > 0:
            vec = {term: v / norm for term, v in vec.items()}
        vectors.append(vec)
    return vectors, vocab


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(term, 0.0) for term, v in a.items())


def _lsa_vectors(vectors: list[dict[str, float]], vocab: list[str],
                 rank: int = LSA_RANK) -> list[list[float]] | None:
    """Latent-semantic vectors: truncated SVD of the term×requirement
    TF-IDF matrix via power iteration with deflation on the Gram
    matrix A^T A. Pure stdlib, deterministic (fixed start vectors,
    fixed iteration count). Returns None when the corpus is too small
    for a latent space (fewer than 3 requirements or 5 terms) — the
    caller falls back to TF-IDF cosine and says so."""
    n_reqs = len(vectors)
    if n_reqs < 3 or len(vocab) < 5:
        return None
    # Gram matrix G[i][j] = cosine between TF-IDF vectors i and j.
    gram = [[_cosine_sparse(vectors[i], vectors[j])
             for j in range(n_reqs)] for i in range(n_reqs)]
    k = min(rank, n_reqs - 1)
    # Working copy for deflation.
    work = [row[:] for row in gram]
    components: list[tuple[float, list[float]]] = []
    for _ in range(k):
        # Fixed start vector (deterministic): normalized ones.
        vec = [1.0 / math.sqrt(n_reqs)] * n_reqs
        for _ in range(30):
            nxt = [sum(work[i][j] * vec[j] for j in range(n_reqs))
                   for i in range(n_reqs)]
            norm = math.sqrt(sum(v * v for v in nxt))
            if norm <= 1e-12:
                break
            nxt = [v / norm for v in nxt]
            shift = math.sqrt(sum((a - b) ** 2
                                  for a, b in zip(nxt, vec, strict=True)))
            vec = nxt
            if shift < 1e-9:
                break
        lam = sum(vec[i] * sum(work[i][j] * vec[j]
                               for j in range(n_reqs))
                  for i in range(n_reqs))
        if lam <= 1e-9:
            break
        components.append((lam, list(vec)))
        # Deflate: G -= lam * v v^T.
        for i in range(n_reqs):
            for j in range(n_reqs):
                work[i][j] -= lam * vec[i] * vec[j]
    if not components:
        return None
    # Latent coords for requirement j: sqrt(lam_i) * v_i[j].
    latent: list[list[float]] = []
    for j in range(n_reqs):
        latent.append([math.sqrt(lam) * vec[j]
                       for lam, vec in components])
    return latent


def _cosine_dense(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return num / (na * nb)


def score_links(requirements: list[Requirement], *,
                use_lsa: bool = True) -> dict[str, Any]:
    """Score relations between requirements (code rung, in place).

    ``similar`` links come from TF-IDF cosine and, when the corpus is
    big enough, latent-semantic cosine — the stronger of the two wins
    and its method is recorded. ``depends`` links come from explicit
    id citations ("as defined in REQ-10") and dependency verbs with a
    shared content word. The code rung NEVER emits ``contradicts``.
    Existing links are replaced (re-scoring is idempotent). Returns
    a note dict: which methods ran and why LSA skipped, if it did.
    """
    for req in requirements:
        req.links = [ln for ln in req.links if ln.kind == "contradicts"]
    if len(requirements) < 2:
        return {"methods": ["keyword"], "lsa": "skipped: < 2 requirements"}
    vectors, vocab = _tfidf_vectors(requirements)
    latent = _lsa_vectors(vectors, vocab) if use_lsa else None
    note: dict[str, Any] = {"methods": ["tfidf_cosine", "keyword"]}
    if latent is not None:
        note["methods"].insert(1, "lsa_cosine")
        note["lsa"] = f"rank {len(latent[0])}"
    else:
        note["lsa"] = ("skipped: < 3 requirements or < 5 terms"
                       if use_lsa else "disabled")
    by_id = {r.id: r for r in requirements}
    for i, req in enumerate(requirements):
        # ── depends: explicit id citation ──
        for cited in _ID_RE.findall(req.text):
            if cited != req.id and cited in by_id:
                req.links.append(RequirementLink(
                    to=cited, kind="depends", score=0.95,
                    method="keyword",
                    reason=f"{req.id} cites {cited}"))
        # ── depends: dependency verb + shared content word ──
        # Word-boundary matching only: "require" must not fire inside
        # "required" ("as required" is a smell, not a dependency).
        lowered = req.text.lower()
        if any(re.search(r"\b" + re.escape(verb) + r"\b", lowered)
               for verb in _DEPENDENCY_VERBS):
            mine = set(_content_words(req.text))
            for other in requirements:
                if other.id == req.id:
                    continue
                shared = mine & set(_content_words(other.text))
                shared.discard(_stem("system"))
                if shared:
                    req.links.append(RequirementLink(
                        to=other.id, kind="depends", score=0.7,
                        method="keyword",
                        reason="dependency verb + shared terms: "
                               + ", ".join(sorted(shared)[:3])))
        # ── similar: TF-IDF cosine, LSA cosine, stronger wins ──
        for j in range(i + 1, len(requirements)):
            other = requirements[j]
            tfidf = _cosine_sparse(vectors[i], vectors[j])
            best, method = tfidf, "tfidf_cosine"
            if latent is not None:
                lsa = _cosine_dense(latent[i], latent[j])
                if lsa > best:
                    best, method = lsa, "lsa_cosine"
            if best >= SIMILARITY_THRESHOLD:
                req.links.append(RequirementLink(
                    to=other.id, kind="similar", score=round(best, 3),
                    method=method))
                other.links.append(RequirementLink(
                    to=req.id, kind="similar", score=round(best, 3),
                    method=method))
    return note

# ── brain rung ─────────────────────────────────────────────────────────────

_BRAIN_PROMPT = """\
You are judging requirements extracted from a specification document.
You MUST answer with JSON only, no other text:
{"reclassify": [{"id": "<requirement id>", "kind": "<functional|quality|constraint|unknown>", "reason": "<one line>"}],
 "contradictions": [{"a": "<id>", "b": "<id>", "reason": "<why they cannot both hold>", "confidence": <0..1>}],
 "missed": [{"text": "<a requirement sentence copied VERBATIM from the document text below>", "kind": "<functional|quality|constraint|unknown>", "reason": "<one line>"}]}

Rules:
- Reclassify ONLY when the code rung's kind is clearly wrong; say why in one line.
- A contradiction needs two requirements that cannot BOTH hold. Similarity is not contradiction. Name both ids and quote the clash in the reason.
- A missed requirement MUST be copied verbatim from the document text below. If you cannot copy it exactly, do not list it.
- Empty lists are fine. Never invent a requirement, an id, or a span.

Document text:
---
{doc_text}
---

Extracted requirements (id, kind, text):
{req_lines}
"""


def _brain_pass(analysis: RequirementAnalysis, doc_text: str,
                llm_fn: Any) -> str:
    """The brain rung: reclassification, contradiction judgments, and
    missed-requirement recovery — all citation-validated. Returns a
    status string for the analysis record. Never raises."""
    if llm_fn is None:
        return "skipped: no brain available"
    req_lines = "\n".join(
        f"{r.id} [{r.kind}] {r.text}" for r in analysis.requirements)
    # NB: .replace, not .format — the prompt's JSON example carries
    # literal braces that str.format would try to interpolate.
    prompt = (_BRAIN_PROMPT.replace("{doc_text}", doc_text)
              .replace("{req_lines}", req_lines))
    try:
        raw = llm_fn(prompt)
    except Exception as exc:  # noqa: BLE001 — a dead brain is a note
        return f"failed: brain call raised {type(exc).__name__}: {exc}"
    try:
        data = json.loads(raw if isinstance(raw, str) else str(raw))
        if not isinstance(data, dict):
            raise ValueError("brain reply is not a JSON object")
    except (ValueError, TypeError) as exc:
        return f"failed: brain reply is not usable JSON: {exc}"
    by_id = {r.id: r for r in analysis.requirements}
    applied = {"reclassified": 0, "contradictions": 0, "recovered": 0}
    dropped = {"reclassified": 0, "contradictions": 0, "recovered": 0}
    for item in data.get("reclassify", []) or []:
        if not isinstance(item, dict):
            continue
        req = by_id.get(str(item.get("id", "") or ""))
        kind = str(item.get("kind", "") or "")
        if req is None or kind not in REQUIREMENT_KINDS:
            dropped["reclassified"] += 1
            continue
        if req.kind != kind:
            req.kind = kind
            applied["reclassified"] += 1
    for item in data.get("contradictions", []) or []:
        if not isinstance(item, dict):
            continue
        a = by_id.get(str(item.get("a", "") or ""))
        b = by_id.get(str(item.get("b", "") or ""))
        reason = str(item.get("reason", "") or "").strip()
        if a is None or b is None or a.id == b.id or not reason:
            dropped["contradictions"] += 1
            continue
        try:
            conf = float(item.get("confidence", 0.8) or 0.8)
        except (TypeError, ValueError):
            conf = 0.8
        conf = min(1.0, max(0.0, conf))
        if any(ln.to == b.id and ln.kind == "contradicts"
               for ln in a.links):
            continue
        a.links.append(RequirementLink(
            to=b.id, kind="contradicts", score=round(conf, 3),
            method="brain", judgment=True, reason=reason))
        applied["contradictions"] += 1
    covered = [(r.span.start, r.span.end) for r in analysis.requirements]
    for item in data.get("missed", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "").strip()
        kind = str(item.get("kind", "") or "")
        if not text or kind not in REQUIREMENT_KINDS:
            dropped["recovered"] += 1
            continue
        start = doc_text.find(text)
        if start < 0:
            # The brain quoted something that is not in the document —
            # dropped, never invented.
            dropped["recovered"] += 1
            continue
        if any(s <= start < e for s, e in covered):
            dropped["recovered"] += 1
            continue
        rid = f"REQ-B{len(analysis.requirements) + 1}"
        span = Span(section="", page=0, start=start,
                    end=start + len(text), text=text)
        analysis.requirements.append(Requirement(
            id=rid, doc_ref=analysis.doc_ref, span=span, text=text,
            kind=kind,
            quality_flags=_quality_flags(text, kind)))
        covered.append((start, start + len(text)))
        applied["recovered"] += 1
    analysis.summary["brain_applied"] = applied
    analysis.summary["brain_dropped"] = dropped
    return (f"brain: reclassified {applied['reclassified']}, "
            f"contradictions {applied['contradictions']}, "
            f"recovered {applied['recovered']}")


# ── analysis entry points ──────────────────────────────────────────────────

def _summarize_run(analysis: RequirementAnalysis,
                   documents: int, link_note: dict[str, Any]) -> None:
    graph = RequirementGraph(analysis.requirements)
    summary = graph.summary()
    summary["documents"] = documents
    summary["rung"] = analysis.rung
    summary["link_methods"] = link_note.get("methods", [])
    if link_note.get("lsa"):
        summary["lsa"] = link_note["lsa"]
    analysis.summary = summary


def _finish(analysis: RequirementAnalysis, doc_text: str,
            llm_fn: Any, documents: int,
            link_note: dict[str, Any]) -> RequirementAnalysis:
    _summarize_run(analysis, documents, link_note)
    if llm_fn is not None:
        analysis.brain_rung = _brain_pass(analysis, doc_text, llm_fn)
        _summarize_run(analysis, documents, link_note)
    else:
        analysis.brain_rung = "skipped: no brain available"
    return analysis


def analyze_document(*, text: str | None = None,
                     path: str | Path | None = None,
                     document: Document | None = None,
                     doc_ref: str | None = None,
                     use_lsa: bool = True,
                     llm_fn: Any = None,
                     store: RequirementStore | None = None
                     ) -> RequirementAnalysis:
    """Analyze one document (code rung, then the brain rung when an
    ``llm_fn`` is given). Intake is exactly one of ``text`` (pasted
    spec), ``path`` (a file the document engine parses), or
    ``document`` (an already-parsed Document). Never raises: intake
    failures return an analysis with ``ok=False`` and a named error."""
    analysis = RequirementAnalysis(doc_ref=doc_ref or "")
    try:
        given = [bool(text), bool(path), document is not None]
        if sum(given) != 1:
            raise ValueError("analyze needs exactly one of "
                             "text= / path= / document=")
        if document is not None:
            doc = document
            ref = doc_ref or doc.id or ""
            doc_text = full_text(doc)
        elif path is not None:
            p = Path(str(path))
            if not p.exists():
                raise FileNotFoundError(f"no such file: {path}")
            doc = _parse_path(p)
            ref = doc_ref or str(path)
            doc_text = full_text(doc)
        else:
            assert text is not None
            doc = new_document(format="text", title="pasted spec")
            doc.sections = [Section(level=1, heading="", text=text)]
            ref = doc_ref or "pasted spec"
            doc_text = text
    except (ValueError, FileNotFoundError, OSError) as exc:
        analysis.ok = False
        analysis.error = str(exc)
        return analysis
    except Exception as exc:  # noqa: BLE001 — analysis never raises
        analysis.ok = False
        analysis.error = f"intake failed: {type(exc).__name__}: {exc}"
        return analysis
    analysis.doc_ref = ref
    analysis.doc_title = doc.title or ""
    analysis.doc_revision = int(doc.revision or 0)
    analysis.requirements = extract_requirements(doc, doc_ref=ref)
    link_note = score_links(analysis.requirements, use_lsa=use_lsa)
    _finish(analysis, doc_text, llm_fn, 1, link_note)
    if store is not None:
        with contextlib.suppress(Exception):  # noqa: BLE001 — the
            # analysis stands even if the store write fails
            store.save(analysis)
    return analysis


def analyze_corpus(documents: list[Document], *,
                   llm_fn: Any = None,
                   use_lsa: bool = True,
                   store: RequirementStore | None = None
                   ) -> RequirementAnalysis:
    """Analyze several documents as one pooled corpus — the lens on
    her own documents (specs, directives, goal briefs). Requirements
    keep their ``doc_ref`` attribution; links may cross documents, so
    two directives that contradict each other surface in one graph.
    A brain judgment naming an id that does not exist in the pool is
    dropped (the brain rung validates every id)."""
    analysis = RequirementAnalysis(
        doc_ref="corpus:" + ",".join(d.id for d in documents))
    pooled: list[Requirement] = []
    texts: list[str] = []
    for doc in documents:
        pooled.extend(extract_requirements(doc, doc_ref=doc.id or ""))
        texts.append(full_text(doc))
    analysis.requirements = pooled
    analysis.doc_title = f"corpus of {len(documents)} documents"
    link_note = score_links(analysis.requirements, use_lsa=use_lsa)
    _finish(analysis, "\n\n".join(texts), llm_fn, len(documents),
            link_note)
    if store is not None:
        with contextlib.suppress(Exception):  # noqa: BLE001 — the
            # analysis stands even if the store write fails
            store.save(analysis)
    return analysis


# ── store ────────────────────────────────────────────────────────────────

def _default_home(home: str | Path | None) -> Path:
    if home:
        return Path(home)
    return Path("~/.nomorals").expanduser()


class RequirementStore:
    """Durable requirement analyses: <home>/data/requirements.json.

    Atomic writes (temp file + replace). A stored analysis round-trips
    through :meth:`RequirementAnalysis.to_dict`, so a saved run can be
    re-read, re-rendered, and re-reported without re-analyzing.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / STORE_FILENAME)
        self._data: dict[str, Any] = {"analyses": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(
                    data.get("analyses"), dict):
                self._data["analyses"] = data["analyses"]
        except (OSError, ValueError):
            self._data = {"analyses": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2),
                       encoding="utf-8")
        tmp.replace(path)

    def save(self, analysis: RequirementAnalysis) -> None:
        self._data["analyses"][analysis.analysis_id] = analysis.to_dict()
        self._save()

    def get(self, analysis_id: str) -> RequirementAnalysis | None:
        row = self._data["analyses"].get(analysis_id or "")
        if not isinstance(row, dict):
            return None
        try:
            return RequirementAnalysis.from_dict(row)
        except Exception:  # noqa: BLE001 — a corrupt row reads as absent
            return None

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for row in self._data["analyses"].values():
            if isinstance(row, dict):
                reqs = row.get("requirements", [])
                rows.append({
                    "analysis_id": str(row.get("analysis_id") or ""),
                    "doc_ref": str(row.get("doc_ref") or ""),
                    "doc_title": str(row.get("doc_title") or ""),
                    "requirements": len(reqs) if isinstance(reqs, list)
                                    else 0,
                    "created_at": row.get("created_at", 0.0),
                })
        return rows


def open_requirements_store(home: str | Path | None = None
                            ) -> RequirementStore:
    """Open (creating if needed) the durable analysis store."""
    return RequirementStore(_default_home(home) / "data" / STORE_FILENAME)


# ── reply + report ───────────────────────────────────────────────────────

def format_requirements_reply(analysis: RequirementAnalysis) -> str:
    """The plain-language reply: counts, the flagged requirements
    with their evidence, the contradictions, and the graph shape."""
    if not analysis.ok:
        return f"requirements analysis failed: {analysis.error}"
    summary = analysis.summary
    lines = [f"{summary.get('requirements', 0)} requirements "
             f"from {analysis.doc_title or analysis.doc_ref or 'the document'}"
             f" — {summary.get('by_kind', {})}."]
    flagged = [r for r in analysis.requirements if r.quality_flags]
    if flagged:
        lines.append("Flagged:")
        for req in flagged:
            flags = ", ".join(f"{fl.kind} ({fl.evidence})"
                              for fl in req.quality_flags)
            lines.append(f"  {req.id} [{req.kind}] {flags} — {req.text}")
    graph = RequirementGraph(analysis.requirements)
    contradictions = graph.contradictions()
    if contradictions:
        lines.append("Contradictions (brain judgments, open to override):")
        for contra in contradictions:
            lines.append(f"  {contra['a']} vs {contra['b']}: "
                         f"{contra['reason']}")
    lines.append(
        f"Graph: {summary.get('similar_links', 0)} similar links, "
        f"{summary.get('depends_links', 0)} depends links, "
        f"{summary.get('clusters', 0)} clusters, "
        f"{summary.get('orphans', 0)} unlinked requirements.")
    if analysis.brain_rung:
        lines.append(f"Brain rung: {analysis.brain_rung}.")
    return "\n".join(lines)


def render_report_markdown(analysis: RequirementAnalysis) -> str:
    """The report artifact: flagged requirements with evidence, the
    contradiction judgments, and the graph summary — markdown, so it
    exports like every other document product."""
    if not analysis.ok:
        return f"# Requirements report\n\nAnalysis failed: {analysis.error}\n"
    summary = analysis.summary
    lines = [f"# Requirements report — "
             f"{analysis.doc_title or analysis.doc_ref}",
             "",
             f"Analysis `{analysis.analysis_id}` · rung `{analysis.rung}`"
             f" · {summary.get('requirements', 0)} requirements · "
             f"{summary.get('documents', 1)} document(s).",
             "",
             "## Counts",
             "",
             f"- by kind: {summary.get('by_kind', {})}",
             f"- flags: {summary.get('flags', {})}",
             f"- similar links: {summary.get('similar_links', 0)}",
             f"- depends links: {summary.get('depends_links', 0)}",
             f"- contradictions: {summary.get('contradictions', 0)}",
             f"- clusters: {summary.get('clusters', 0)}",
             f"- unlinked: {summary.get('orphans', 0)}",
             "",
             "## Flagged requirements",
             ""]
    flagged = [r for r in analysis.requirements if r.quality_flags]
    if not flagged:
        lines.append("None — no requirement smells found.")
    for req in flagged:
        lines.append(f"### {req.id} [{req.kind}]")
        lines.append("")
        lines.append(f"> {req.text}")
        lines.append("")
        for fl in req.quality_flags:
            lines.append(f"- **{fl.kind}**: `{fl.evidence}`")
        lines.append("")
    graph = RequirementGraph(analysis.requirements)
    contradictions = graph.contradictions()
    lines.extend(["## Contradictions", ""])
    if not contradictions:
        lines.append("None found.")
    for contra in contradictions:
        lines.append(f"- **{contra['a']}** vs **{contra['b']}** "
                     f"(brain judgment, confidence {contra['score']}): "
                     f"{contra['reason']}")
    lines.extend(["", "## Graph summary", ""])
    for cluster in graph.clusters():
        lines.append(f"- cluster: {', '.join(cluster)}")
    if not graph.clusters():
        lines.append("No clusters — no requirement is similar to another.")
    lines.append("")
    lines.append(f"Unlinked requirements: "
                 f"{', '.join(graph.orphans()) or 'none'}.")
    lines.append("")
    lines.append("## All requirements")
    lines.append("")
    for req in analysis.requirements:
        lines.append(f"- **{req.id}** [{req.kind}] ({req.doc_ref}): "
                     f"{req.text}")
    return "\n".join(lines)


def write_report(analysis: RequirementAnalysis,
                 home: str | Path | None = None) -> Path:
    """Write the report markdown under
    <home>/data/requirements_reports/<analysis_id>.md and return its
    path."""
    root = _default_home(home) / "data" / REPORT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{analysis.analysis_id}.md"
    path.write_text(render_report_markdown(analysis), encoding="utf-8")
    return path
