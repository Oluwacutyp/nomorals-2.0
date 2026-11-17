"""Evidence tables + question answering over papers (Elicit core).

Ask a research question, get a paper set from open scholarly sources,
screen it against inclusion criteria, extract a cited cell matrix
(claim / method / result / stance per paper), and synthesise what
agrees, what disagrees and what is still unanswered.

The one structural rule, from the report and from grounded.py: a cell
without a quote is not allowed to exist. Every cell carries a verbatim
quote from the paper it came from, and the table drops any cell whose
quote does not string-match its source paper.

Corpus (honest boundary, named in every answer): the paper set is
assembled per question from open scholarly sources — OpenAlex,
Semantic Scholar and arXiv, all keyless — plus nothing else. There is
no rented 138M-paper index here; a question with no papers is
answered "no papers found", never invented.

Retrieval fetches FULL abstracts (OpenAlex reconstructs its inverted
index; Semantic Scholar and arXiv return the whole summary), because
extraction over a 300-character stub would be guessing. Every fetcher
is injectable (``fetch_json`` / ``fetch_text``) so tests run offline
and the organ never raises on a dead network: a failed source is
named in the answer, never faked.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger("nomorals.research.extraction")

UA = "nomorals-research/1.0 (evidence; contact: owner)"
TIMEOUT = 15
DEFAULT_LIMIT = 10
MAX_PER_SOURCE = 25
REVIEWS_FILENAME = "evidence_reviews.json"
TABLE_DIRNAME = "evidence"

#: the open scholarly sources the corpus is assembled from, in order.
SOURCES: tuple[str, ...] = ("openalex", "semantic_scholar", "arxiv")

#: display names for the sources, used in notes and replies.
_SOURCE_NAMES: dict[str, str] = {
    "openalex": "OpenAlex",
    "semantic_scholar": "Semantic Scholar",
    "arxiv": "arXiv",
}

_STOPWORDS = frozenset(
    "a an the and or but if then else when at by for with about into through "  # noqa: SIM905
    "during before after above below to from up down in out on off over under "
    "again further once here there all any both each few more most other some "
    "such no nor not only own same so than too very can will just should now "
    "me my i you your we us is are was were be been being have has had do "
    "does did of as it its this that these those am s t tell said says say "
    "also per what does how why which who".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

#: negation signals for a yes/no (stance) read of a sentence.
_NEGATION = (
    "no significant", "no difference", "no effect", "no evidence",
    "did not", "does not", "do not", "not support", "failed to",
    "no benefit", "no improvement", "not associated",
)
#: positive-outcome signals for a yes/no (stance) read.
_POSITIVE = (
    "reduced", "improved", "increased", "decreased", "significant",
    "effective", "benefit", "support", "associated with",
)


# ── helpers ───────────────────────────────────────────────────────────────

def _stem(word: str) -> str:
    """Light stem so 'infection' matches 'infections', 'reduced' 'reducing'."""
    if len(word) > 4 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 3 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    if len(word) > 4 and word.endswith("e"):
        return word[:-1]
    return word


def _content_words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall((text or "").lower())
            if w not in _STOPWORDS and len(w) > 2]


def _sentences(text: str) -> list[str]:
    """Split text into sentences, dropping fragments and headers."""
    out: list[str] = []
    for s in _SENT_SPLIT.split(re.sub(r"\s+", " ", text or "").strip()):
        s = s.strip()
        if not s or s.startswith("#") or len(s.split()) < 4:
            continue
        out.append(s)
    return out


def _norm_text(value: str) -> str:
    return " ".join(sorted(set(_content_words(value))))


def _norm_doi(doi: str) -> str:
    doi = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


def paper_id_for(title: str, doi: str = "", url: str = "") -> str:
    """Stable paper id: the DOI when there is one, else a title hash."""
    norm = _norm_doi(doi)
    if norm:
        return norm
    seed = ((title or "").strip().lower() + "|" + (url or "").strip())
    return "paper-" + hashlib.sha1(seed.encode()).hexdigest()[:12]


def _default_home(home: str | Path | None) -> Path:
    return Path(home) if home else Path.home() / ".nomorals"


# ── shapes ────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """One paper in the corpus: title + full abstract + provenance."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str = ""
    url: str = ""
    source: str = ""
    abstract: str = ""
    citations: int | None = None
    id: str = ""

    def __post_init__(self) -> None:
        self.doi = _norm_doi(self.doi)
        if not self.id:
            self.id = paper_id_for(self.title, self.doi, self.url)

    @property
    def text(self) -> str:
        """The source text extraction may quote from: title + abstract."""
        if self.abstract.strip():
            return self.title.strip() + ". " + self.abstract.strip()
        return self.title.strip()

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title,
                "authors": list(self.authors), "year": self.year,
                "doi": self.doi, "url": self.url, "source": self.source,
                "abstract": self.abstract, "citations": self.citations}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        return cls(title=data.get("title", ""),
                   authors=list(data.get("authors") or []),
                   year=data.get("year"), doi=data.get("doi", ""),
                   url=data.get("url", ""), source=data.get("source", ""),
                   abstract=data.get("abstract", ""),
                   citations=data.get("citations"),
                   id=data.get("id", ""))


@dataclass
class PaperSet:
    """The screened paper set for one question."""

    id: str
    query: str
    papers: list[Paper] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "query": self.query,
                "papers": [p.to_dict() for p in self.papers],
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperSet:
        return cls(id=data.get("id", ""), query=data.get("query", ""),
                   papers=[Paper.from_dict(p)
                           for p in data.get("papers") or []],
                   created_at=data.get("created_at", 0.0))


@dataclass
class ExtractionColumn:
    """One column of the extraction table: what to pull from each paper."""

    name: str
    question: str
    value_type: str = "text"  # text | number | yes_no
    keywords: list[str] = field(default_factory=list)

    def cue_words(self) -> list[str]:
        if self.keywords:
            return [_stem(w) for w in self.keywords]
        return _content_words(self.question)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "question": self.question,
                "value_type": self.value_type,
                "keywords": list(self.keywords)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionColumn:
        return cls(name=data.get("name", ""),
                   question=data.get("question", ""),
                   value_type=data.get("value_type", "text"),
                   keywords=list(data.get("keywords") or []))


@dataclass
class ExtractionSchema:
    """The columns of one extraction table."""

    columns: list[ExtractionColumn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"columns": [c.to_dict() for c in self.columns]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionSchema:
        return cls(columns=[ExtractionColumn.from_dict(c)
                            for c in data.get("columns") or []])


def default_schema(question: str) -> ExtractionSchema:
    """The standard question schema: claim / method / result / stance.

    The stance column is what lets the synthesis say, per paper,
    whether it supports the question's proposition — the per-paper
    verdict a consensus meter is built from.
    """
    q = (question or "").strip()
    return ExtractionSchema(columns=[
        ExtractionColumn(
            name="claim",
            question=f"What does this paper claim about: {q}",
            value_type="text",
            keywords=["claim", "show", "shows", "found", "report",
                      "conclude", "demonstrate"]),
        ExtractionColumn(
            name="method",
            question=f"How was the claim studied — design, population, "
                     f"intervention? ({q})",
            value_type="text",
            keywords=["trial", "study", "cohort", "participants",
                      "method", "design", "randomized", "controlled",
                      "survey", "review"]),
        ExtractionColumn(
            name="result",
            question=f"What did the paper find — outcome and effect? ({q})",
            value_type="text",
            keywords=["found", "result", "outcome", "effect", "reduced",
                      "increased", "difference", "compared", "percent"]),
        ExtractionColumn(
            name="stance",
            question=f"Does this paper support the proposition in: {q}",
            value_type="yes_no",
            keywords=["support", "reduce", "increase", "effect",
                      "difference", "significant"]),
    ])


@dataclass
class Cell:
    """One filled extraction cell. A cell without a quote cannot exist."""

    paper_id: str
    column: str
    value: str
    quote: str
    location: str = "abstract"  # abstract | title

    def __post_init__(self) -> None:
        if not (self.quote or "").strip():
            raise ValueError(
                "an extraction cell needs a verbatim quote — "
                "a cell without a quote is not allowed to exist")

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "column": self.column,
                "value": self.value, "quote": self.quote,
                "location": self.location}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cell:
        return cls(paper_id=data.get("paper_id", ""),
                   column=data.get("column", ""),
                   value=data.get("value", ""),
                   quote=data.get("quote", ""),
                   location=data.get("location", "abstract"))


def _score_sentence(sentence: str, cue_words: list[str],
                    question_words: list[str]) -> int:
    """Cue words count double — they name the column, not the topic."""
    words = set(_content_words(sentence))
    cue_hit = len(set(cue_words) & words)
    q_hit = len(set(question_words) & words)
    return 2 * cue_hit + q_hit


def _yes_no_value(sentence: str) -> str:
    lowered = sentence.lower()
    if any(neg in lowered for neg in _NEGATION):
        return "no"
    if any(pos in lowered for pos in _POSITIVE):
        return "yes"
    return "unclear"


def extract_cells(paper: Paper, schema: ExtractionSchema,
                  question: str = "") -> list[Cell]:
    """Fill the schema's columns from one paper's text, deterministically.

    Each cell's quote is a verbatim sentence from the paper. A column
    with no matching sentence leaves no cell — an honest gap the
    synthesis reports as unanswered, never a filled-in guess.
    """
    title_sents = [(s, "title") for s in _sentences(paper.title)]
    abstract_sents = [(s, "abstract") for s in _sentences(paper.abstract)]
    sentences = title_sents + abstract_sents
    if not sentences:
        return []
    # No question given means no topic context: score on the column's
    # own cue words only, so an unrelated column stays unfilled instead
    # of matching any sentence that shares a title word.
    question_words = _content_words(question) if question else []
    # The abstract is the evidence text; the title is metadata that
    # names the topic and so shares question words by construction.
    # Abstract sentences score +2 so the evidence wins over the label.
    cells: list[Cell] = []
    for column in schema.columns:
        cue = column.cue_words()
        ranked: list[tuple[int, str, str]] = []
        for sent, loc in sentences:
            # Eligibility is the RAW score: a sentence must share a
            # cue or question word to be considered at all. The +2
            # abstract bonus only orders eligible sentences — it must
            # never admit a sentence with zero overlap.
            raw = _score_sentence(sent, cue, question_words)
            if raw <= 0:
                continue
            ranked.append((raw + (2 if loc == "abstract" else 0),
                           sent, loc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if column.value_type == "yes_no":
            # A stance read needs a decidable sentence (a negation or
            # a positive outcome). Prefer the best decidable one; only
            # fall back to the best overall when nothing decides.
            decidable = [item for item in ranked
                         if _yes_no_value(item[1]) != "unclear"]
            pool = decidable if decidable else ranked
        else:
            pool = ranked
        best = pool[0] if pool else None
        if best is None:
            continue
        _, quote, loc = best
        if column.value_type == "number":
            match = _NUMBER.search(quote)
            if not match:
                # A number may live in another sentence of the paper.
                found = None
                for sent, loc2 in sentences:
                    if _score_sentence(sent, cue, question_words) > 0:
                        m = _NUMBER.search(sent)
                        if m:
                            found = (m.group(0).replace(",", ""), sent, loc2)
                            break
                if found is None:
                    continue
                value, quote, loc = found
            else:
                value = match.group(0).replace(",", "")
        elif column.value_type == "yes_no":
            value = _yes_no_value(quote)
        else:
            value = quote
        cells.append(Cell(paper_id=paper.id, column=column.name,
                          value=value, quote=quote, location=loc))
    return cells


@dataclass
class EvidenceTable:
    """Paper set × extraction schema → cited cell matrix.

    Construction verifies every cell: a cell whose quote does not
    string-match its source paper is dropped and counted in
    ``dropped_unverified`` — it never renders.
    """

    paper_set: PaperSet
    schema: ExtractionSchema
    cells: list[Cell] = field(default_factory=list)
    dropped_unverified: int = 0

    def __post_init__(self) -> None:
        by_id = {p.id: p for p in self.paper_set.papers}
        kept: list[Cell] = []
        dropped = 0
        for cell in self.cells:
            paper = by_id.get(cell.paper_id)
            if paper is None or cell.quote not in paper.text:
                dropped += 1
                continue
            kept.append(cell)
        self.cells = kept
        self.dropped_unverified = dropped

    @property
    def verified_cells(self) -> list[Cell]:
        return list(self.cells)

    def coverage(self) -> dict[str, tuple[int, int]]:
        """Per column: (papers with a cell, papers in the set)."""
        total = len(self.paper_set.papers)
        out: dict[str, tuple[int, int]] = {}
        for column in self.schema.columns:
            filled = {c.paper_id for c in self.cells
                      if c.column == column.name}
            out[column.name] = (len(filled), total)
        return out

    def render_markdown(self) -> str:
        """The extraction table as a markdown table (quote per cell)."""
        by_paper: dict[str, dict[str, Cell]] = {}
        for cell in self.cells:
            by_paper.setdefault(cell.paper_id, {})[cell.column] = cell
        header = ["paper"] + [c.name for c in self.schema.columns]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join("---" for _ in header) + " |"]
        for paper in self.paper_set.papers:
            row = [f"{paper.title} ({paper.year or 'n.d.'})"]
            cells = by_paper.get(paper.id, {})
            for column in self.schema.columns:
                cell = cells.get(column.name)
                row.append(cell.value.replace("|", "/") if cell else "—")
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"paper_set": self.paper_set.to_dict(),
                "schema": self.schema.to_dict(),
                "cells": [c.to_dict() for c in self.cells],
                "dropped_unverified": self.dropped_unverified}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceTable:
        return cls(
            paper_set=PaperSet.from_dict(data.get("paper_set") or {}),
            schema=ExtractionSchema.from_dict(data.get("schema") or {}),
            cells=[Cell.from_dict(c) for c in data.get("cells") or []])


def export_csv(table: EvidenceTable, path: str | Path) -> str:
    """Write the extraction table (with quotes) as CSV. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    by_id = {p.id: p for p in table.paper_set.papers}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["paper_id", "paper", "year", "source", "doi",
                         "url", "column", "value", "quote"])
        for cell in table.cells:
            paper = by_id.get(cell.paper_id)
            writer.writerow([
                cell.paper_id,
                paper.title if paper else "",
                paper.year if paper else "",
                paper.source if paper else "",
                paper.doi if paper else "",
                paper.url if paper else "",
                cell.column, cell.value, cell.quote])
    return str(path)


def export_ris(papers: list[Paper], path: str | Path) -> str:
    """Write the paper set as RIS (Zotero / reference-manager import)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for paper in papers:
        lines.append("TY  - JOUR")
        for author in paper.authors:
            lines.append(f"AU  - {author}")
        lines.append(f"TI  - {paper.title}")
        if paper.year:
            lines.append(f"PY  - {paper.year}")
        if paper.doi:
            lines.append(f"DO  - {paper.doi}")
        if paper.url:
            lines.append(f"UR  - {paper.url}")
        if paper.abstract:
            lines.append(f"AB  - {paper.abstract}")
        lines.append("ER  - ")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ── synthesis ─────────────────────────────────────────────────────────────

def synthesize(table: EvidenceTable, question: str) -> dict[str, Any]:
    """What agrees, what disagrees, what is still unanswered — from cells only.

    yes/no columns tally: one value across papers is agreement, mixed
    values are disagreement with both sides named. Text columns agree
    only when the normalized wording is identical; differing wordings
    are listed as differing findings, never forced into agree/disagree.
    Number columns report their range. A column with no cells at all
    is unanswered.
    """
    agreements: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    differing: list[dict[str, Any]] = []
    ranges: list[dict[str, Any]] = []
    unanswered: list[str] = []
    for column in table.schema.columns:
        filled = [c for c in table.cells if c.column == column.name]
        if not filled:
            unanswered.append(column.name)
            continue
        if column.value_type == "yes_no":
            by_value: dict[str, list[str]] = {}
            for cell in filled:
                by_value.setdefault(cell.value, []).append(cell.paper_id)
            if len(by_value) == 1:
                value = next(iter(by_value))
                agreements.append({"column": column.name, "value": value,
                                   "papers": by_value[value]})
            else:
                disagreements.append(
                    {"column": column.name,
                     "values": sorted(by_value),
                     "papers": by_value})
        elif column.value_type == "number":
            try:
                nums = sorted(float(c.value) for c in filled)
            except ValueError:
                nums = []
            if nums:
                ranges.append({"column": column.name, "min": nums[0],
                               "max": nums[-1],
                               "papers": [c.paper_id for c in filled]})
        else:
            norms = {_norm_text(c.value) for c in filled}
            if len(norms) == 1:
                agreements.append({"column": column.name,
                                   "value": filled[0].value,
                                   "papers": [c.paper_id for c in filled]})
            else:
                differing.append(
                    {"column": column.name,
                     "values": sorted({c.value for c in filled}),
                     "papers": [c.paper_id for c in filled]})
    parts: list[str] = []
    for item in agreements:
        parts.append(
            f"agree on {item['column']}: {item['value']} "
            f"({len(item['papers'])} paper(s))")
    for item in disagreements:
        parts.append(
            f"disagree on {item['column']}: "
            + " vs ".join(str(v) for v in item["values"]))
    for item in differing:
        parts.append(
            f"differ on {item['column']} wording — "
            f"{len(item['values'])} distinct findings")
    for item in ranges:
        parts.append(
            f"{item['column']} ranges {item['min']:g}–{item['max']:g}")
    for name in unanswered:
        parts.append(f"unanswered: {name} — no paper stated it")
    summary = "; ".join(parts) if parts else (
        "no cells extracted — nothing to synthesise")
    return {"agreements": agreements, "disagreements": disagreements,
            "differing": differing, "ranges": ranges,
            "unanswered": unanswered, "summary": summary}


# ── retrieval (open scholarly sources, full abstracts) ─────────────────────

def _default_fetch_json(url: str, timeout: int = TIMEOUT) -> Any | None:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _default_fetch_text(url: str, timeout: int = TIMEOUT) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _reconstruct_openalex_abstract(index: dict[str, Any]) -> str:
    """OpenAlex ships abstracts as {word: [positions]} — invert it."""
    if not isinstance(index, dict) or not index:
        return ""
    placed: list[tuple[int, str]] = []
    for word, positions in index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                placed.append((pos, str(word)))
    placed.sort()
    return " ".join(word for _, word in placed)


def _fetch_openalex(query: str, limit: int,
                    fetch_json: Callable[..., Any]) -> tuple[list[Paper], str]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per_page": max(1, min(limit, MAX_PER_SOURCE)),
         "sort": "cited_by_count:desc"})
    data = fetch_json(url, timeout=TIMEOUT)
    if not isinstance(data, dict):
        return [], ("OpenAlex: unavailable — no response (network down "
                    "or rate-limited). No results invented.")
    papers: list[Paper] = []
    for work in (data.get("results") or [])[:limit]:
        papers.append(Paper(
            title=(work.get("title") or "").strip(),
            year=work.get("publication_year"),
            doi=_norm_doi(work.get("doi") or ""),
            url=work.get("doi") or work.get("id") or "",
            source="OpenAlex",
            abstract=_reconstruct_openalex_abstract(
                work.get("abstract_inverted_index") or {}),
            citations=work.get("cited_by_count")))
    if not papers:
        return [], f"OpenAlex: no papers for '{query}'."
    return papers, f"OpenAlex: {len(papers)} paper(s)"


def _fetch_semantic_scholar(
        query: str, limit: int,
        fetch_json: Callable[..., Any]) -> tuple[list[Paper], str]:
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
           + urllib.parse.urlencode(
               {"query": query, "limit": max(1, min(limit, MAX_PER_SOURCE)),
                "fields": "title,abstract,year,citationCount,"
                          "externalIds,url,authors"}))
    data = fetch_json(url, timeout=TIMEOUT)
    if not isinstance(data, dict):
        return [], ("Semantic Scholar: unavailable — no response "
                    "(rate limit 100 req/5min unauthenticated). "
                    "No results invented.")
    papers: list[Paper] = []
    for item in (data.get("data") or [])[:limit]:
        ext = item.get("externalIds") or {}
        papers.append(Paper(
            title=(item.get("title") or "").strip(),
            authors=[(a or {}).get("name", "")
                     for a in (item.get("authors") or [])],
            year=item.get("year"),
            doi=_norm_doi(ext.get("DOI") or ""),
            url=item.get("url") or "",
            source="Semantic Scholar",
            abstract=(item.get("abstract") or "").strip(),
            citations=item.get("citationCount")))
    if not papers:
        return [], f"Semantic Scholar: no papers for '{query}'."
    return papers, f"Semantic Scholar: {len(papers)} paper(s)"


def _fetch_arxiv(query: str, limit: int,
                 fetch_text: Callable[..., Any]) -> tuple[list[Paper], str]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0,
         "max_results": max(1, min(limit, MAX_PER_SOURCE))})
    text = fetch_text(url, timeout=TIMEOUT)
    if text is None:
        return [], ("arXiv: unavailable — no response from the export "
                    "API. No results invented.")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [], "arXiv: unavailable — unparseable XML."
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        link_el = entry.find("atom:id", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        year = None
        if published_el is not None and published_el.text:
            year = int(published_el.text[:4]) if \
                published_el.text[:4].isdigit() else None
        papers.append(Paper(
            title=((title_el.text or "").strip().replace("\n", " ")
                   if title_el is not None else ""),
            authors=[(a.text or "").strip()
                     for a in entry.findall("atom:author/atom:name", ns)],
            year=year,
            url=link_el.text.strip() if link_el is not None
            and link_el.text else "",
            source="arXiv",
            abstract=((summary_el.text or "").strip()
                      if summary_el is not None else "")))
    if not papers:
        return [], f"arXiv: no papers for '{query}'."
    return papers, f"arXiv: {len(papers)} paper(s)"


def retrieve_papers(
        query: str, *, limit: int = DEFAULT_LIMIT,
        fetch_json: Callable[..., Any] | None = None,
        fetch_text: Callable[..., Any] | None = None,
        sources: tuple[str, ...] = SOURCES) -> tuple[list[Paper], list[str]]:
    """Assemble the paper set for a question from open sources.

    Never raises: a failed source contributes a note naming the
    failure, never invented papers. Papers dedupe by DOI/title
    across sources; the combined set is capped at ``limit``, most
    cited first.
    """
    fetch_json = fetch_json or _default_fetch_json
    fetch_text = fetch_text or _default_fetch_text
    limit = max(1, int(limit))
    papers: list[Paper] = []
    notes: list[str] = []
    seen: set[str] = set()
    for name in sources:
        try:
            if name == "openalex":
                found, note = _fetch_openalex(query, limit, fetch_json)
            elif name == "semantic_scholar":
                found, note = _fetch_semantic_scholar(
                    query, limit, fetch_json)
            elif name == "arxiv":
                found, note = _fetch_arxiv(query, limit, fetch_text)
            else:
                notes.append(f"{name}: unknown source — skipped.")
                continue
        except Exception as exc:  # noqa: BLE001 — a source never kills retrieval
            _log.warning("evidence source %s failed: %s", name, exc)
            notes.append(
                f"{_SOURCE_NAMES.get(name, name)}: unavailable — {exc}. "
                "No results invented.")
            continue
        notes.append(note)
        for paper in found:
            key = paper.id
            if key in seen:
                continue
            seen.add(key)
            papers.append(paper)
    papers.sort(key=lambda p: (p.citations or 0), reverse=True)
    return papers[:limit], notes


# ── durable review store ──────────────────────────────────────────────────

class EvidenceReviewStore:
    """Durable evidence reviews: <home>/data/evidence_reviews.json.

    Atomic writes (temp file + replace). A review holds the question,
    the paper set, the screening decisions, the schema, the cited
    cells and the synthesis — so a review can be re-read, re-rendered
    and re-exported across restarts.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / REVIEWS_FILENAME)
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
            "papers_included": r.get("papers_included", 0),
        } for r in self._data["reviews"].values()]
        out.sort(key=lambda r: r["created_at"], reverse=True)
        return out

    def latest(self) -> dict[str, Any] | None:
        listed = self.list()
        if not listed:
            return None
        return self.get(listed[0]["id"])


def open_review_store(home: str | Path | None = None) -> EvidenceReviewStore:
    """Open (creating if needed) the durable review store."""
    return EvidenceReviewStore(_default_home(home) / "data" / REVIEWS_FILENAME)


# ── the question flow ─────────────────────────────────────────────────────

def answer_question(
        home: str | Path | None, question: str, *,
        schema: ExtractionSchema | None = None,
        criteria: Any = None, limit: int = DEFAULT_LIMIT,
        store: EvidenceReviewStore | None = None,
        fetch_json: Callable[..., Any] | None = None,
        fetch_text: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Retrieve → screen → extract → synthesise, durably recorded.

    Never raises on a dead network or an empty corpus: an unanswered
    question returns ok False with the reason named, and the empty
    review is still recorded so the miss is visible.
    """
    from .screening import InclusionCriteria, screen_papers

    question = (question or "").strip()
    if not question:
        return {"ok": False, "question": question, "review_id": "",
                "papers_screened": 0, "papers_included": 0,
                "error": "give me a research question",
                "reply": "give me a research question — "
                         "/evidence <what does the evidence say about …?>"}
    store = store or open_review_store(home)
    papers, notes = retrieve_papers(
        question, limit=limit, fetch_json=fetch_json,
        fetch_text=fetch_text)
    review_id = ("ev" + hashlib.sha1(
        f"{question}|{time.time()}".encode()).hexdigest()[:10])
    if not papers:
        review = {"id": review_id, "question": question,
                  "query": question, "created_at": time.time(),
                  "papers": [], "papers_screened": 0,
                  "papers_included": 0, "decisions": [],
                  "schema": (schema or default_schema(question)).to_dict(),
                  "table": {"paper_set": {"id": review_id, "query": question,
                                          "papers": []},
                            "schema": (schema or default_schema(question)
                                       ).to_dict(), "cells": [],
                            "dropped_unverified": 0},
                  "synthesis": {"agreements": [], "disagreements": [],
                                "differing": [], "ranges": [],
                                "unanswered": ["everything"],
                                "summary": "no papers found"},
                  "sources": notes, "csv_path": ""}
        store.save(review)
        return {"ok": False, "question": question, "review_id": review_id,
                "papers_screened": 0, "papers_included": 0,
                "error": f"no papers found for “{question}” in the open "
                         f"corpus (OpenAlex, Semantic Scholar, arXiv)",
                "reply": (f"no papers found for “{question}” in the open "
                          f"corpus (OpenAlex, Semantic Scholar, arXiv) — "
                          f"nothing to extract from, so nothing invented. "
                          f"Try a broader question."),
                "table": review["table"], "synthesis": review["synthesis"],
                "sources": notes, "csv_path": ""}
    criteria = criteria if criteria is not None else InclusionCriteria()
    decisions = screen_papers(papers, criteria)
    included_ids = {d.paper_id for d in decisions
                    if d.decision == "include"}
    included = [p for p in papers if p.id in included_ids]
    schema = schema or default_schema(question)
    cells: list[Cell] = []
    for paper in included:
        cells.extend(extract_cells(paper, schema, question=question))
    paper_set = PaperSet(id=review_id, query=question, papers=included)
    table = EvidenceTable(paper_set=paper_set, schema=schema, cells=cells)
    synthesis = synthesize(table, question)
    home_path = _default_home(home)
    csv_path = export_csv(
        table, home_path / TABLE_DIRNAME / f"{review_id}.csv")
    review = {"id": review_id, "question": question, "query": question,
              "created_at": time.time(),
              "papers": [p.to_dict() for p in included],
              "papers_screened": len(papers),
              "papers_included": len(included),
              "decisions": [d.to_dict() for d in decisions],
              "schema": schema.to_dict(),
              "table": table.to_dict(), "synthesis": synthesis,
              "sources": notes, "csv_path": csv_path}
    store.save(review)
    answer = {"ok": True, "question": question, "review_id": review_id,
              "papers_screened": len(papers),
              "papers_included": len(included),
              "table": table.to_dict(), "synthesis": synthesis,
              "sources": notes, "csv_path": csv_path}
    answer["reply"] = format_evidence_reply(answer)
    return answer


def format_evidence_reply(answer: dict[str, Any]) -> str:
    """Render an answer (or a stored review) as plain chat text."""
    if not answer.get("ok", True):
        return answer.get("reply") or answer.get("error") or "no answer"
    table_data = answer.get("table") or {}
    table = table_data if isinstance(table_data, EvidenceTable) \
        else EvidenceTable.from_dict(table_data)
    synthesis = answer.get("synthesis") or {}
    lines = [
        f"Evidence on “{answer.get('question', '')}” — "
        f"{answer.get('papers_included', 0)} of "
        f"{answer.get('papers_screened', 0)} papers included.",
    ]
    sources = answer.get("sources") or []
    if sources:
        lines.append("Sources: " + "; ".join(sources) + ".")
    by_paper: dict[str, dict[str, Cell]] = {}
    for cell in table.cells:
        by_paper.setdefault(cell.paper_id, {})[cell.column] = cell
    for idx, paper in enumerate(table.paper_set.papers, 1):
        lines.append(
            f"[{idx}] {paper.title} ({paper.year or 'n.d.'}, "
            f"{paper.source}"
            + (f", {paper.citations} citations" if paper.citations else "")
            + (f", {paper.url}" if paper.url else "") + ")")
        cells = by_paper.get(paper.id, {})
        for column in table.schema.columns:
            cell = cells.get(column.name)
            if cell is None:
                lines.append(f"    {column.name}: — (not stated)")
            elif column.value_type == "yes_no":
                lines.append(
                    f"    {column.name}: {cell.value} — “{cell.quote}”")
            else:
                lines.append(f"    {column.name}: “{cell.quote}”")
    lines.append("Synthesis — " + str(synthesis.get("summary", "")) + ".")
    for item in synthesis.get("disagreements", []):
        lines.append(
            f"  Disagrees on {item['column']}: "
            + " vs ".join(str(v) for v in item.get("values", [])) + ".")
    for name in synthesis.get("unanswered", []):
        lines.append(f"  Still unanswered: {name}.")
    if answer.get("csv_path"):
        lines.append(f"Extraction table saved to {answer['csv_path']}.")
    lines.append("Corpus: open scholarly sources only (OpenAlex, "
                 "Semantic Scholar, arXiv). Every cell quotes its paper — "
                 "no paper, no claim.")
    return "\n".join(lines)
