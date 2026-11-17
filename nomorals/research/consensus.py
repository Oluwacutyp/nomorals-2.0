"""Research consensus — the Consensus meter over the evidence organ.

Ask a research question, get what the literature actually agrees
on. The evidence organ (extraction.py, the Elicit core) retrieves
papers from open scholarly sources and extracts a cited table per
paper; this layer classifies each paper's finding as yes / no /
possibly / mixed, weighs the papers by quality — citations,
recency, study design — never count alone, and reports the
distribution as a meter with the dissent named by name.

The rules, from the report and from the organ's own honesty
requirements:

- A stance tally is only as good as the retrieved set, so the
  meter always shows its n and its sources — a meter over 6
  papers never wears the authority of one over 60.
- Every verdict quotes its paper verbatim. A verdict whose quote
  does not string-match its source paper cannot exist (the
  evidence table drops unverified cells, and classification reads
  the paper's own text).
- A question that is not yes/no gets prose, not a meter — a meter
  over an open question would be a fabrication.
- A thin set says so: fewer than 3 classified papers is "too thin
  to call", never a consensus.
- What would change it is computed, not asserted: the weighted gap
  the dissenting side would have to close.

Corpus (honest boundary, named in every answer): the paper set is
assembled per question from open scholarly sources — OpenAlex,
Semantic Scholar and arXiv, all keyless — plus nothing else. There
is no rented 200M-paper index here; a question with no papers is
answered "no papers found", never invented.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .extraction import (
    EvidenceTable,
    Paper,
    _content_words,
    _default_home,
    _score_sentence,
    _sentences,
    _yes_no_value,
    answer_question,
    default_schema,
)

_log = get_logger("nomorals.research.consensus")

CONSENSUS_FILENAME = "consensus.json"

#: stances a paper's finding can take on a yes/no proposition.
STANCES: tuple[str, ...] = ("yes", "no", "possibly", "mixed")

#: classified papers below this are too thin to call a consensus.
MIN_CLASSIFIED_FOR_CONSENSUS = 3

#: weighted share of the leading stance for each strength band.
STRONG_SHARE = 0.80
MODERATE_SHARE = 0.60

#: weight bounds — a paper never counts for nothing, never for everything.
MIN_WEIGHT = 0.5
MAX_WEIGHT = 2.5

#: first words that make a question yes/no (a proposition to tally).
_BINARY_OPENERS = frozenset([
    "does", "do", "is", "are", "can", "could", "should", "will",
    "would", "has", "have", "did", "was", "were", "may", "might"])

#: study designs worth extra weight, in detection order (strongest first).
_STRONG_DESIGNS = ("randomized controlled trial", "meta-analysis",
                   "systematic review")
_MEDIUM_DESIGNS = ("cohort study", "case-control study")

_DESIGN_MARKERS: tuple[tuple[str, str], ...] = (
    ("meta-analysis", "meta-analysis"),
    ("systematic review", "systematic review"),
    ("randomized controlled trial", "randomized controlled trial"),
    ("randomised controlled trial", "randomized controlled trial"),
    ("randomized trial", "randomized controlled trial"),
    ("randomised trial", "randomized controlled trial"),
    ("cohort", "cohort study"),
    ("case-control", "case-control study"),
    ("cross-sectional", "cross-sectional study"),
    ("case report", "case report"),
    ("review", "review"),
)

DEPTH_LIMITS: dict[str, int] = {"quick": 10, "deep": 25}


# ── shapes ────────────────────────────────────────────────────────────────

@dataclass
class StanceVerdict:
    """One paper's verdict on a yes/no proposition.

    The finding quote is a verbatim sentence from the paper; a mixed
    verdict carries the opposing sentence as ``counter_quote`` too.
    """

    paper_id: str
    stance: str  # yes | no | possibly | mixed
    finding_quote: str
    counter_quote: str = ""
    study_design: str = "not stated"
    citations_count: int | None = None
    year: int | None = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "stance": self.stance,
                "finding_quote": self.finding_quote,
                "counter_quote": self.counter_quote,
                "study_design": self.study_design,
                "citations_count": self.citations_count,
                "year": self.year, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StanceVerdict:
        return cls(paper_id=data.get("paper_id", ""),
                   stance=data.get("stance", ""),
                   finding_quote=data.get("finding_quote", ""),
                   counter_quote=data.get("counter_quote", ""),
                   study_design=data.get("study_design", "not stated"),
                   citations_count=data.get("citations_count"),
                   year=data.get("year"),
                   weight=float(data.get("weight", 1.0)))


# ── question shape ────────────────────────────────────────────────────────

def is_binary_question(question: str) -> bool:
    """A yes/no question is one that opens with an auxiliary verb.

    "What does the evidence say about X" is not binary — it has no
    proposition to tally, so it gets prose instead of a meter.
    """
    words = (question or "").strip().lower().split(None, 1)
    if not words:
        return False
    return words[0].rstrip("?").strip("“\"'(") in _BINARY_OPENERS


# ── stance classification ─────────────────────────────────────────────────

def detect_study_design(paper: Paper) -> str:
    """Name the study design from the abstract (title as fallback)."""
    text = ((paper.abstract or "").strip() or paper.title).lower()
    for marker, design in _DESIGN_MARKERS:
        if marker in text:
            return design
    return "not stated"


def paper_weight(paper: Paper, design: str, *, now: int | None = None) -> float:
    """Quality weight for one paper — never count alone.

    Base 1.0; +0.5 for 50+ citations (+0.25 for 10+); +0.5 when the
    paper is from the last 5 years; +0.5 for a strong design
    (randomized trial / meta-analysis / systematic review), +0.25
    for a medium one (cohort / case-control); −0.25 when there is
    no abstract to read. Bounded to [0.5, 2.5].
    """
    year = now if now is not None else datetime.now().year
    weight = 1.0
    cites = paper.citations or 0
    if cites >= 50:
        weight += 0.5
    elif cites >= 10:
        weight += 0.25
    if paper.year and year - paper.year <= 5:
        weight += 0.5
    if design in _STRONG_DESIGNS:
        weight += 0.5
    elif design in _MEDIUM_DESIGNS:
        weight += 0.25
    if not (paper.abstract or "").strip():
        weight -= 0.25
    return max(MIN_WEIGHT, min(MAX_WEIGHT, weight))


def classify_stance(paper: Paper, question: str,
                    *, now: int | None = None) -> StanceVerdict | None:
    """Classify one paper's finding on a yes/no proposition.

    Re-scans the paper's own sentences (title + abstract) against
    the question: the best decidable "yes" sentence and the best
    decidable "no" sentence. Both present and the weaker at least
    half as strong as the stronger means the paper itself is mixed
    — both quotes are kept. One direction only means that stance.
    A sentence that shares question words but decides nothing is
    "possibly". No overlapping sentence at all means no verdict:
    the paper stays out of the meter and is named as unclassified.
    """
    schema = default_schema(question)
    stance_col = next((c for c in schema.columns if c.name == "stance"),
                      None)
    cue = stance_col.cue_words() if stance_col else []
    question_words = _content_words(question)
    best_yes: tuple[int, str] | None = None
    best_no: tuple[int, str] | None = None
    best_any: tuple[int, str] | None = None
    for sent in _sentences(paper.text):
        raw = _score_sentence(sent, cue, question_words)
        if raw <= 0:
            continue
        if best_any is None or raw > best_any[0]:
            best_any = (raw, sent)
        verdict = _yes_no_value(sent)
        if verdict == "yes" and (best_yes is None or raw > best_yes[0]):
            best_yes = (raw, sent)
        elif verdict == "no" and (best_no is None or raw > best_no[0]):
            best_no = (raw, sent)
    design = detect_study_design(paper)
    weight = paper_weight(paper, design, now=now)
    base = {"paper_id": paper.id, "study_design": design,
            "citations_count": paper.citations, "year": paper.year,
            "weight": weight}
    if best_yes is not None and best_no is not None:
        weaker = min(best_yes[0], best_no[0])
        stronger = max(best_yes[0], best_no[0])
        if weaker >= 0.5 * stronger:
            main = best_yes if best_yes[0] >= best_no[0] else best_no
            counter = best_no if main is best_yes else best_yes
            return StanceVerdict(stance="mixed", finding_quote=main[1],
                                 counter_quote=counter[1], **base)
    if best_yes is not None:
        return StanceVerdict(stance="yes", finding_quote=best_yes[1],
                             **base)
    if best_no is not None:
        return StanceVerdict(stance="no", finding_quote=best_no[1],
                             **base)
    if best_any is not None:
        return StanceVerdict(stance="possibly",
                             finding_quote=best_any[1], **base)
    return None


# ── the meter ─────────────────────────────────────────────────────────────

def build_meter(verdicts: list[StanceVerdict],
                *, unclassified: int = 0) -> dict[str, Any]:
    """Tally the verdicts into a consensus meter.

    Counts and quality-weighted shares per stance, the leading
    stance, and the strength: "strong" (leading weighted share
    ≥ 0.80), "moderate" (≥ 0.60), "split" (below that), or "too
    thin" (fewer than 3 classified papers — no consensus is
    claimed). Dissent is the non-leading stances that have papers.
    """
    counts = {stance: 0 for stance in STANCES}
    weights = {stance: 0.0 for stance in STANCES}
    for verdict in verdicts:
        if verdict.stance not in counts:
            continue
        counts[verdict.stance] += 1
        weights[verdict.stance] += verdict.weight
    total_weight = sum(weights.values())
    shares = {stance: (weights[stance] / total_weight if total_weight
                       else 0.0) for stance in STANCES}
    n = len(verdicts)
    leading = "" if n == 0 else max(
        STANCES, key=lambda s: (weights[s], counts[s]))
    if n < MIN_CLASSIFIED_FOR_CONSENSUS:
        strength = "too thin"
    elif shares[leading] >= STRONG_SHARE:
        strength = "strong"
    elif shares[leading] >= MODERATE_SHARE:
        strength = "moderate"
    else:
        strength = "split"
    dissent = [s for s in STANCES if s != leading and counts[s] > 0]
    meter = {"n": n, "classified": n, "unclassified": unclassified,
             "counts": counts, "weights": weights, "shares": shares,
             "leading": leading, "strength": strength,
             "dissent": dissent}
    meter["what_would_change_it"] = what_would_change_it(meter)
    return meter


def what_would_change_it(meter: dict[str, Any]) -> str:
    """Name what would move the meter — computed from its own weights.

    A thin meter needs more papers. A split meter names the weighted
    gap a single paper would have to close to flip the lead. A
    standing consensus names the weighted points the dissenting
    side would have to gain to overturn it — a large trial or
    meta-analysis finding the opposite.
    """
    strength = meter.get("strength", "too thin")
    n = int(meter.get("n", 0))
    if strength == "too thin":
        return (f"more papers — only {n} classified paper(s) in the "
                f"set; a consensus needs at least "
                f"{MIN_CLASSIFIED_FOR_CONSENSUS} classified papers")
    weights = meter.get("weights") or {}
    leading = meter.get("leading", "")
    lead_w = float(weights.get(leading, 0.0))
    others = {s: float(w) for s, w in weights.items() if s != leading}
    runner = max(others, key=lambda s: others[s]) if others else ""
    runner_w = others.get(runner, 0.0)
    if strength == "split":
        gap = lead_w - runner_w
        if gap <= 0.05:
            return ("the lead is tied — the next classified paper "
                    "decides it")
        return (f"the lead is only {gap:.1f} weighted points — a "
                f"single paper of weight over {gap:.1f} finding "
                f"“{runner}” would flip it")
    dissent_w = sum(others.values())
    if dissent_w <= 0.0:
        return ("nothing dissents yet — it would take a large trial "
                "or meta-analysis finding the opposite to move it")
    needed = lead_w - dissent_w + 0.1
    return (f"the dissenting side holds {dissent_w:.1f} weighted "
            f"points; {needed:.1f} more points on that side — a "
            f"large trial or meta-analysis finding the opposite — "
            f"would overturn it")


# ── durable review store ──────────────────────────────────────────────────

class ConsensusStore:
    """Durable consensus reviews: <home>/data/consensus.json.

    Atomic writes (temp file + replace). A review holds the
    question, the meter, the per-paper verdicts (each quoting its
    paper), the synthesis from the evidence organ and the evidence
    review id it was built from — so a review can be re-read and
    re-rendered across restarts.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / CONSENSUS_FILENAME)
        self._data: dict[str, Any] = {"reviews": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and isinstance(
                    data.get("reviews"), dict):
                self._data = data
        except (OSError, ValueError):
            self._data = {"reviews": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2),
                       encoding="utf-8")
        tmp.replace(path)

    def save(self, review: dict[str, Any]) -> None:
        self._data["reviews"][review["id"]] = review
        self._save()

    def get(self, review_id: str) -> dict[str, Any] | None:
        return self._data["reviews"].get(review_id)

    def list(self) -> list[dict[str, Any]]:
        """Review metadata, newest first."""
        out = [{
            "id": r.get("id", ""), "question": r.get("question", ""),
            "created_at": r.get("created_at", 0.0),
            "papers_classified": (r.get("meter") or {}).get("n", 0),
            "strength": (r.get("meter") or {}).get("strength", ""),
        } for r in self._data["reviews"].values()]
        out.sort(key=lambda r: r["created_at"], reverse=True)
        return out

    def latest(self) -> dict[str, Any] | None:
        listed = self.list()
        if not listed:
            return None
        return self.get(listed[0]["id"])


def open_consensus_store(home: str | Path | None = None) -> ConsensusStore:
    """Open (creating if needed) the durable consensus store."""
    return ConsensusStore(_default_home(home) / "data" / CONSENSUS_FILENAME)


# ── the question flow ─────────────────────────────────────────────────────

def answer_consensus(
        home: str | Path | None, question: str, *,
        depth: str = "quick",
        store: ConsensusStore | None = None,
        evidence_store: Any = None,
        fetch_json: Any = None,
        fetch_text: Any = None,
        now: int | None = None) -> dict[str, Any]:
    """Ask what the literature agrees on — retrieve, classify, tally.

    The evidence organ retrieves and extracts (one path, no second
    retrieval); this layer classifies each paper's stance, weighs
    it, and builds the meter. A non-binary question gets prose
    from the evidence synthesis instead of a meter. Never raises
    on a dead network or an empty corpus: an unanswered question
    returns ok False with the reason named, and the miss is still
    recorded so it is visible.
    """
    question = (question or "").strip()
    if not question:
        return {"ok": False, "question": question, "review_id": "",
                "meter": None,
                "error": "give me a research question",
                "reply": "give me a research question — "
                         "/consensus <does X …?>"}
    if depth not in DEPTH_LIMITS:
        raise ValueError(
            f"unknown consensus depth “{depth}” — quick|deep")
    store = store or open_consensus_store(home)
    binary = is_binary_question(question)
    review_id = ("cs" + hashlib.sha1(
        f"{question}|{time.time()}".encode()).hexdigest()[:10])
    answer = answer_question(
        home, question, limit=DEPTH_LIMITS[depth], store=evidence_store,
        fetch_json=fetch_json, fetch_text=fetch_text)
    if not answer.get("ok"):
        review = {"id": review_id, "question": question,
                  "binary": binary, "depth": depth,
                  "evidence_review_id": answer.get("review_id", ""),
                  "created_at": time.time(), "papers_screened": 0,
                  "papers_included": 0, "meter": None, "verdicts": [],
                  "papers": [], "synthesis": answer.get("synthesis"),
                  "sources": answer.get("sources") or [],
                  "error": answer.get("error", "no papers found")}
        store.save(review)
        return {"ok": False, "question": question,
                "review_id": review_id,
                "evidence_review_id": answer.get("review_id", ""),
                "depth": depth, "binary": binary, "meter": None,
                "error": answer.get("error", "no papers found"),
                "reply": answer.get("reply") or answer.get("error", "")}
    table = EvidenceTable.from_dict(answer.get("table") or {})
    verdicts: list[StanceVerdict] = []
    unclassified = 0
    for paper in table.paper_set.papers:
        verdict = classify_stance(paper, question, now=now)
        if verdict is None:
            unclassified += 1
            continue
        verdicts.append(verdict)
    meter = build_meter(verdicts, unclassified=unclassified) if binary \
        else None
    review = {"id": review_id, "question": question,
              "binary": binary, "depth": depth,
              "evidence_review_id": answer.get("review_id", ""),
              "created_at": time.time(),
              "papers_screened": answer.get("papers_screened", 0),
              "papers_included": answer.get("papers_included", 0),
              "meter": meter,
              "verdicts": [v.to_dict() for v in verdicts],
              "papers": [p.to_dict() for p in table.paper_set.papers],
              "synthesis": answer.get("synthesis") or {},
              "sources": answer.get("sources") or []}
    store.save(review)
    out: dict[str, Any] = {
        "ok": True, "question": question, "review_id": review_id,
        "evidence_review_id": answer.get("review_id", ""),
        "depth": depth, "binary": binary,
        "papers_screened": answer.get("papers_screened", 0),
        "papers_included": answer.get("papers_included", 0),
        "meter": meter,
        "verdicts": [v.to_dict() for v in verdicts],
        "papers": [p.to_dict() for p in table.paper_set.papers],
        "synthesis": answer.get("synthesis") or {},
        "sources": answer.get("sources") or [],
        "csv_path": answer.get("csv_path", "")}
    out["reply"] = format_consensus_reply(out)
    return out


# ── rendering ─────────────────────────────────────────────────────────────

def _bar(share: float) -> str:
    filled = int(round(share * 10))
    return "█" * filled + "░" * (10 - filled)


def _n_papers(n: int) -> str:
    return "1 paper" if n == 1 else f"{n} papers"


def format_consensus_reply(answer: dict[str, Any]) -> str:
    """Render a consensus answer (or a stored review) as plain chat text."""
    if not answer.get("ok", True):
        return answer.get("reply") or answer.get("error") or "no answer"
    question = answer.get("question", "")
    meter = answer.get("meter")
    papers = [Paper.from_dict(p) for p in answer.get("papers") or []]
    by_id = {p.id: p for p in papers}
    verdicts = [StanceVerdict.from_dict(v)
                for v in answer.get("verdicts") or []]
    synthesis = answer.get("synthesis") or {}
    sources = answer.get("sources") or []
    if meter is None:
        lines = [
            f"No meter for “{question}” — it is not a yes/no "
            f"question, so there is no stance to tally. Here is "
            f"what the papers say instead.",
            "Synthesis — " + str(synthesis.get("summary", "")) + ".",
        ]
        for item in synthesis.get("disagreements", []):
            lines.append(
                f"  Disagrees on {item.get('column', '')}: "
                + " vs ".join(str(v) for v in item.get("values", []))
                + ".")
        for name in synthesis.get("unanswered", []):
            lines.append(f"  Still unanswered: {name}.")
        lines.append("Ask a yes/no question (does / is / can …) for a "
                     "meter of what the field agrees on.")
        lines.append("Corpus: open scholarly sources only (OpenAlex, "
                     "Semantic Scholar, arXiv). Every verdict quotes "
                     "its paper — no paper, no claim.")
        return "\n".join(lines)
    strength = str(meter.get("strength", "too thin")).upper()
    counts = meter.get("counts") or {}
    weights = meter.get("weights") or {}
    shares = meter.get("shares") or {}
    n = int(meter.get("n", 0))
    unclassified = int(meter.get("unclassified", 0))
    lines = [
        f"Consensus on “{question}” — {strength}",
        f"Meter over {_n_papers(n)} ({n} classified"
        + (f", {unclassified} unclassified" if unclassified else "")
        + ") from OpenAlex, Semantic Scholar, arXiv.",
    ]
    if sources:
        lines.append("Sources: " + "; ".join(sources) + ".")
    for stance in STANCES:
        count = int(counts.get(stance, 0))
        if not count:
            continue
        share = float(shares.get(stance, 0.0))
        lines.append(
            f"{stance.upper()} {_bar(share)} {share * 100:.0f}% — "
            f"{_n_papers(count)} (weight {weights.get(stance, 0.0):.1f})")
    for idx, verdict in enumerate(verdicts, 1):
        paper = by_id.get(verdict.paper_id)
        if paper is None:
            continue
        lines.append(
            f"[{idx}] {paper.title} ({paper.year or 'n.d.'}, "
            f"{paper.source}"
            + (f", {paper.citations} citations"
               if paper.citations else "")
            + f", {verdict.study_design}) — "
            f"{verdict.stance.upper()}: “{verdict.finding_quote}”")
        if verdict.counter_quote:
            lines.append(
                f"    but also: “{verdict.counter_quote}”")
    dissent = meter.get("dissent") or []
    for stance in dissent:
        for verdict in verdicts:
            if verdict.stance != stance:
                continue
            paper = by_id.get(verdict.paper_id)
            if paper is None:
                continue
            lines.append(
                f"Dissent ({stance}): {paper.title} — "
                f"“{verdict.finding_quote}”")
    lines.append("What would change it: "
                 + str(meter.get("what_would_change_it", "")) + ".")
    lines.append("Synthesis — " + str(synthesis.get("summary", "")) + ".")
    lines.append("Corpus: open scholarly sources only (OpenAlex, "
                 "Semantic Scholar, arXiv). A meter over "
                 f"{_n_papers(n)} never wears the authority of one over "
                 "60. Every verdict quotes its paper — no paper, no "
                 "claim.")
    return "\n".join(lines)
