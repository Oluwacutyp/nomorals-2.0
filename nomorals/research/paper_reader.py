"""Paper reader — SciSpace (52-websites #38): read a paper, explain it.

The research family already retrieves papers (Elicit), weighs what
they agree on (Consensus), walks the citation neighbourhood
(Connected Papers) and shows the full record (Semantic Scholar) —
but nothing READS a paper for the owner and explains it back in
plain words. This organ is that reader, as an extension of the
research family (report §38 verdict: extend, not a new organ):
hand Devon a paper — DOI, title, pasted text, or a local file — and
get a plain-language explanation: what it asks, what it did, what
it found, why it matters, with every finding carrying the verbatim
quote it came from; the hard parts (equations, tables) unpacked
into named symbols and compared columns; questions answered against
the paper's own sentences; citations in APA / MLA / Chicago /
BibTeX; and a paraphrase surface that refuses to lose a number or
a name.

Sources: the Semantic Scholar record for the paper (its abstract
and open-access PDF link), the open-access PDF itself when one
exists, the arXiv PDF when the record carries an arXiv id — never
a paywalled copy, never a scrape of a publisher page. A paper
whose full text is not openly available is read from its abstract,
and the reply says so.

Never raises on the fetch paths, and never invents: an unknown
paper, a dead source, or a question the paper does not cover is
refused honestly. An LLM rung (``llm_fn``) may reword the
extractive overview and paraphrases into plainer language — but
every rewording passes an invariance check (every number and named
entity in the source must survive); a rewording that drops one is
discarded and the extractive text stands. Without an LLM the
extractive reading is the answer, labelled as such.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..documents.parsers import parse_bytes, parse_path

__all__ = [
    "Explanation",
    "Finding",
    "PaperExplanation",
    "PaperReadingStore",
    "PaperSection",
    "PaperText",
    "acquire_paper",
    "ask_paper",
    "detect_hard_parts",
    "explain_paper",
    "explain_span",
    "format_citation",
    "format_reading_reply",
    "open_reading_store",
    "paraphrase_span",
    "read_paper",
    "structure_text",
]

_log = logging.getLogger(__name__)

UA = "Mozilla/5.0 (compatible; NoMoralsPaperReader/1.0; +personal research)"
TIMEOUT = 30
MAX_PDF_BYTES = 20 * 1024 * 1024

READING_FILENAME = "paper_readings.json"

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

#: canonical section headings, matched case-insensitively, with an
#: optional leading number ("1. ", "2)", "IV. ") and optional colon.
_SECTION_HEADINGS = (
    "abstract", "introduction", "background", "related work",
    "methods", "methodology", "materials and methods",
    "results", "discussion", "conclusion", "conclusions",
    "limitations", "references", "acknowledgments", "acknowledgements",
)
_HEADING_NUM_RE = re.compile(r"^(?:\d+[.)]|[ivxIVX]+[.)])\s+")

#: cue phrases that mark a finding sentence (Results / Discussion).
_FINDING_CUES = (
    "we found", "we show", "we showed", "results show",
    "results indicate", "results suggest", "significantly",
    "increased", "decreased", "improved", "reduced", "fell by",
    "faster", "slower", "demonstrate", "reveals", "suggests",
    "no adverse", "effect was", "outperformed",
)
_QUESTION_CUES = (
    "we ask", "whether", "we investigate", "we examine", "we test",
    "we explore", "how", "what", "does", "can", "is there",
)
_METHOD_CUES = (
    "recruited", "assigned", "measured", "participants", "random",
    "collected", "we used", "we recruited", "procedure",
)
_MATTER_CUES = (
    "matters", "important", "implication", "suggest", "future",
    "practical", "could", "should", "limitation",
)

_STOP = frozenset([
    "that", "this", "with", "from", "have", "were", "they", "will",
    "would", "there", "their", "what", "when", "which", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "which", "while", "did", "does",
    "were", "what", "which", "whom", "whose",
])

#: question-word synonyms for the deterministic ask rung: a question
#: about "participants" should match "recruited ... nurses".
_ASK_SYNONYMS: dict[str, tuple[str, ...]] = {
    "participant": ("participant", "recruited", "nurse", "worker",
                    "subject", "volunteer", "patient", "people"),
    "participants": ("participant", "recruited", "nurse", "worker",
                     "subject", "volunteer", "patient", "people"),
    "studied": ("studied", "recruited", "tested", "measured",
                "examined", "enrolled"),
    "study": ("study", "trial", "experiment", "work"),
    "found": ("found", "show", "showed", "result", "results"),
    "finding": ("finding", "found", "result", "results"),
    "findings": ("finding", "found", "result", "results"),
    "method": ("method", "methods", "methodology", "procedure",
               "protocol"),
    "result": ("result", "results", "found", "showed"),
    "results": ("result", "results", "found", "showed"),
}

#: plain-language glossary for the deterministic paraphrase rung.
#: Word-boundary substitutions only — numbers and names are never
#: touched, which is what makes the invariance check hold.
_PLAIN_WORDS: tuple[tuple[str, str], ...] = (
    ("in order to", "to"),
    ("due to the fact that", "because"),
    ("in spite of the fact that", "although"),
    ("a significant number of", "many"),
    ("a large number of", "many"),
    ("utilized", "used"),
    ("utilize", "use"),
    ("demonstrated", "showed"),
    ("demonstrate", "show"),
    ("facilitate", "help"),
    ("furthermore", "also"),
    ("however", "but"),
    ("therefore", "so"),
    ("methodology", "method"),
    ("investigate", "study"),
    ("investigated", "studied"),
    ("approximately", "about"),
    ("subsequently", "then"),
    ("prior to", "before"),
    ("commence", "start"),
    ("terminate", "end"),
)

_CITATION_PAREN_RE = re.compile(r"\([^)]*et al\.[^)]*\)")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s?%?")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]{2,}\b|\b[A-Z]{2,}\b")


def _default_home(home: str | Path | None) -> Path:
    if home:
        return Path(home)
    return Path("~/.nomorals").expanduser()


def _norm_doi(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    return value


def _looks_like_doi(query: str) -> bool:
    return bool(_DOI_RE.search(query or ""))


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return "-".join(words[:5]) or "paper"


# ── paper text + structure ─────────────────────────────────────────────────


@dataclass
class PaperSection:
    """One headed section of a paper's text."""

    heading: str
    text: str


@dataclass
class PaperText:
    """The text of a paper, with where it came from named honestly."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str = ""
    venue: str = ""
    #: "open-access PDF" | "arXiv PDF" | "abstract only" |
    #: "pasted text" | "local file".
    source: str = ""
    text: str = ""

    def sections(self) -> list[PaperSection]:
        return structure_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "venue": self.venue,
            "source": self.source,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperText:
        return cls(
            paper_id=str(data.get("paper_id") or ""),
            title=str(data.get("title") or ""),
            authors=[str(a) for a in (data.get("authors") or [])],
            year=data.get("year"),
            doi=str(data.get("doi") or ""),
            venue=str(data.get("venue") or ""),
            source=str(data.get("source") or ""),
            text=str(data.get("text") or ""),
        )


def _match_heading(line: str) -> str | None:
    stripped = (line or "").strip()
    if not stripped or len(stripped) > 40:
        return None
    candidate = _HEADING_NUM_RE.sub("", stripped).strip()
    candidate = candidate.rstrip(":").strip().lower()
    if candidate in _SECTION_HEADINGS:
        return candidate
    return None


def structure_text(text: str) -> list[PaperSection]:
    """Split a paper's text into headed sections.

    Lines that are a known heading (case-insensitive, optionally
    numbered) start a new section. Text before the first heading is
    "front matter"; a text with no headings at all is one "full
    text" section. Never raises — empty text yields [].
    """
    sections: list[PaperSection] = []
    heading = "front matter"
    buf: list[str] = []
    for line in (text or "").splitlines():
        found = _match_heading(line)
        if found is not None:
            sections.append(PaperSection(heading, "\n".join(buf).strip()))
            heading, buf = found, []
        else:
            buf.append(line)
    sections.append(PaperSection(heading, "\n".join(buf).strip()))
    sections = [s for s in sections if s.text]
    if len(sections) == 1 and sections[0].heading == "front matter":
        sections = [PaperSection("full text", sections[0].text)]
    return sections


def _sentences(text: str) -> list[str]:
    """Sentence-split, dropping equation / table fragments.

    A fragment carrying ``$$``, a pipe table row, or LaTeX is not a
    sentence — it is a hard part, handled by detect_hard_parts.
    Fragments are returned VERBATIM (only the ends stripped): a
    finding's quote must be a substring of the paper, so internal
    line breaks are kept, never normalized away.
    """
    raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for frag in raw:
        frag = frag.strip()
        if len(frag) < 25:
            continue
        if "$$" in frag or "|" in frag or "\\log" in frag:
            continue
        if frag.startswith(("Table ", "TABLE ")):
            continue
        out.append(frag)
    return out


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOP and len(w) > 2]


def _score_sentence(sentence: str, cues: tuple[str, ...]) -> float:
    lowered = sentence.lower()
    score = 0.0
    for cue in cues:
        if cue in lowered:
            score += 3.0
    if re.search(r"\d", sentence):
        score += 2.0
    if 60 <= len(sentence) <= 320:
        score += 1.0
    return score


def _pick(sections: list[PaperSection], headings: tuple[str, ...],
          cues: tuple[str, ...], count: int) -> list[str]:
    """Top ``count`` sentences from the named sections, by cue score."""
    seen: set[str] = set()
    scored: list[tuple[float, str]] = []
    for section in sections:
        if section.heading.lower() not in headings:
            continue
        for sentence in _sentences(section.text):
            if sentence in seen:
                continue
            seen.add(sentence)
            score = _score_sentence(sentence, cues)
            if score > 0:
                scored.append((score, sentence))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    if scored:
        return [s for _, s in scored[:count]]
    # no cue hit: the section's own opening sentences, honestly plain
    for section in sections:
        if section.heading.lower() in headings:
            return _sentences(section.text)[:count]
    return []


# ── hard parts: equations + tables ─────────────────────────────────────────


@dataclass
class Explanation:
    """A grounded explanation of one hard span (report §38 shape).

    ``span_ref`` names the span ("equation 1 (Results)"),
    ``kind`` is text | equation | table, ``plain_text`` is the
    unpacking in plain words, and ``grounded_quotes`` are the
    verbatim paper lines it came from — never paraphrased away.
    """

    span_ref: str
    kind: str
    plain_text: str
    grounded_quotes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_ref": self.span_ref,
            "kind": self.kind,
            "plain_text": self.plain_text,
            "grounded_quotes": list(self.grounded_quotes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Explanation:
        return cls(
            span_ref=str(data.get("span_ref") or ""),
            kind=str(data.get("kind") or "text"),
            plain_text=str(data.get("plain_text") or ""),
            grounded_quotes=[str(q) for q in
                             (data.get("grounded_quotes") or [])],
        )


_DISPLAY_EQ_RE = re.compile(
    r"\$\$(.+?)\$\$|\\begin\{equation\}(.*?)\\end\{equation\}"
    r"|\\\[(.*?)\\\]",
    re.DOTALL)
_INLINE_EQ_RE = re.compile(r"\$([^$\n]{4,120}?)\$")
_TABLE_CAPTION_RE = re.compile(
    r"^Table\s+(\d+)\s*:?\s*(.*)$", re.IGNORECASE)
_WHERE_SENT_RE = re.compile(r"\bwhere\b", re.IGNORECASE)


def _unpack_equation(eq_text: str, section: PaperSection,
                     eq_start: int, eq_end: int) -> Explanation:
    """Unpack one displayed equation in plain words.

    The lead is the sentence just before the equation; the symbol
    definitions are the "where X is ..." sentences after it, split
    into one plain sentence per symbol. Every line quoted is
    verbatim from the paper.
    """
    before = section.text[:eq_start]
    after = section.text[eq_end:]
    before_sents = _sentences(before)
    lead = before_sents[-1] if before_sents else ""
    after_sents = _sentences(after)
    definitions: list[str] = []
    defining: list[str] = []
    for sent in after_sents[:3]:
        if not _WHERE_SENT_RE.search(sent):
            continue
        defining.append(sent)
        body = sent.strip()
        body = re.sub(r"^where\s+", "", body, flags=re.IGNORECASE)
        body = body.rstrip(".")
        for clause in re.split(r",\s+and\s+|,\s+", body):
            clause = clause.strip()
            if clause and re.search(r"\bis\b|\bdenotes\b|\bstands for\b",
                                    clause):
                definitions.append(clause.rstrip(".") + ".")
        break
    parts: list[str] = []
    if lead:
        parts.append(lead.rstrip(":") + ".")
    if definitions:
        parts.append(" ".join(definitions))
    if not parts:
        parts.append("The paper states this equation without defining "
                     "its symbols nearby.")
    quotes = [eq_text.strip()]
    if lead:
        quotes.append(lead)
    quotes.extend(defining)
    return Explanation(
        span_ref="", kind="equation",
        plain_text=" ".join(parts),
        grounded_quotes=[q for q in quotes if q],
    )


def _unpack_table(caption_line: str, caption: str, number: str,
                  rows: list[str]) -> Explanation:
    """Unpack one markdown table: columns, rows, and the headline range.

    The headline comparison is the numeric column whose values
    spread widest — the column the table most wants you to see —
    with its lowest and highest rows named. All numbers come from
    the table itself. Separator rows (|---|---|) are not data.
    """
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    data_rows: list[list[str]] = []
    for row in rows[1:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != len(header) or not any(cells):
            continue
        if all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue
        data_rows.append(cells)
    caption_text = caption.strip()
    lead = f"Table {number}"
    if caption_text:
        lead += f", '{caption_text}',"
    lead += (f" compares {len(data_rows)} rows across columns "
             f"{', '.join(header)}.")
    best_col = -1
    best_spread = -1.0
    best_vals: list[tuple[float, str]] = []
    for col in range(1, len(header)):
        vals: list[tuple[float, str]] = []
        for row in data_rows:
            try:
                vals.append((float(row[col].replace(",", "")), row[0]))
            except (ValueError, IndexError):
                continue
        if len(vals) >= 2:
            spread = max(v for v, _ in vals) - min(v for v, _ in vals)
            if spread > best_spread:
                best_spread, best_col, best_vals = spread, col, vals
    plain = lead
    if best_col >= 0 and best_vals:
        low = min(best_vals)
        high = max(best_vals)
        col_name = header[best_col]

        def _num(value: float) -> str:
            return str(int(value)) if value == int(value) else str(value)

        plain += (f" {col_name} runs from {_num(low[0])} ({low[1]}) "
                  f"to {_num(high[0])} ({high[1]}).")
    quotes = [caption_line] + [r for r in rows
                               if not all(
                                   re.fullmatch(r":?-+:?", c)
                                   for c in (c.strip() for c in
                                             r.strip().strip("|").split("|")))]
    return Explanation(
        span_ref="", kind="table", plain_text=plain,
        grounded_quotes=[q for q in quotes if q],
    )


def detect_hard_parts(text: str) -> list[Explanation]:
    """Find the hard parts of a paper: displayed equations + tables.

    Each part is an Explanation with a span_ref ("equation 1
    (Results)", "table 1 (Results)"), its kind, a plain-words
    unpacking, and the verbatim quotes it came from.
    """
    parts: list[Explanation] = []
    eq_n = 0
    table_n = 0
    for section in structure_text(text):
        body = section.text
        seen_eq: set[str] = set()
        for match in _DISPLAY_EQ_RE.finditer(body):
            eq_n += 1
            eq_text = next(g for g in match.groups() if g is not None)
            seen_eq.add(eq_text.strip())
            part = _unpack_equation(eq_text.strip(), section,
                                    match.start(), match.end())
            part.span_ref = f"equation {eq_n} ({section.heading})"
            parts.append(part)
        for match in _INLINE_EQ_RE.finditer(body):
            inner = match.group(1).strip()
            if "=" not in inner or inner in seen_eq:
                continue
            seen_eq.add(inner)
            eq_n += 1
            part = _unpack_equation(inner, section,
                                    match.start(), match.end())
            part.span_ref = f"equation {eq_n} ({section.heading})"
            parts.append(part)
        lines = body.splitlines()
        i = 0
        while i < len(lines):
            cap = _TABLE_CAPTION_RE.match(lines[i].strip())
            if not cap:
                i += 1
                continue
            number, caption = cap.group(1), cap.group(2)
            rows: list[str] = []
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            while j < len(lines) and "|" in lines[j]:
                rows.append(lines[j].strip())
                j += 1
            if len(rows) >= 2:
                table_n += 1
                part = _unpack_table(lines[i].strip(), caption, number,
                                     rows)
                part.span_ref = f"table {table_n} ({section.heading})"
                parts.append(part)
                i = j
            else:
                i += 1
    return parts


# ── acquisition ────────────────────────────────────────────────────────────


def _default_fetch_text(url: str) -> str | None:
    """Fetch a URL as text: PDF bytes are parsed, plain text decoded.

    Never raises — a dead URL returns None.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read(MAX_PDF_BYTES)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    if data[:4] == b"%PDF":
        try:
            from ..documents.model import full_text

            doc = parse_bytes(data, filename="paper.pdf",
                              mime="application/pdf")
            return full_text(doc)
        except Exception:  # noqa: BLE001 — an unparsable PDF is no text
            return None
    return data.decode("utf-8", "replace")


def acquire_paper(
        query: str, *,
        fetch_record: Callable[..., Any] | None = None,
        fetch_text: Callable[[str], str | None] | None = None
        ) -> tuple[PaperText | None, str]:
    """Get a paper's text from a DOI, a title, pasted text, or a file.

    Resolution order: a local path that exists is parsed; a long
    pasted block is read as-is; anything else goes to the Semantic
    Scholar record — the open-access PDF when one exists, the arXiv
    PDF when the record carries an arXiv id, the abstract when
    neither exists (and the note says so). Never raises: a dead
    source or an unknown paper returns (None, reason).
    """
    fetch_text = fetch_text or _default_fetch_text
    query = (query or "").strip()
    if not query:
        return None, "no paper given — hand me a DOI, a title, pasted text, or a file"
    path = Path(query)
    if len(query) < 500 and path.exists() and path.is_file():
        try:
            from ..documents.model import full_text

            doc = parse_path(path)
            text = full_text(doc)
        except Exception as exc:  # noqa: BLE001 — unreadable file
            return None, f"could not read '{query}': {exc}"
        if not text.strip():
            return None, f"'{query}' has no readable text"
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()),
                     path.stem)
        title = first if len(first) <= 120 and not first.endswith(".") \
            else path.stem
        return PaperText(
            paper_id=f"file-{_slug(title)}", title=title,
            source="local file", text=text), \
            "read from a local file"
    if len(query) >= 500 and " " in query:
        first = next((ln.strip() for ln in query.splitlines() if ln.strip()),
                     "pasted paper")
        title = first if len(first) <= 120 else first[:117] + "..."
        return PaperText(
            paper_id=f"text-{_slug(title)}", title=title,
            source="pasted text", text=query), \
            "read from pasted text"
    if fetch_record is None:
        from .paper_records import fetch_paper_record as _fetch

        fetch_record = _fetch
    try:
        record, record_note = fetch_record(query)
    except Exception as exc:  # noqa: BLE001 — a dead source never raises
        return None, (f"paper source unavailable: {exc} — no record "
                      "invented")
    if record is None:
        return None, record_note
    pdf_url = getattr(record, "open_access_pdf", None) or ""
    text = None
    if pdf_url:
        text = fetch_text(pdf_url)
    if text is None and getattr(record, "arxiv_id", ""):
        text = fetch_text(
            f"https://arxiv.org/pdf/{record.arxiv_id}")
    if text and text.strip():
        source = "open-access PDF" if pdf_url else "arXiv PDF"
        return PaperText(
            paper_id=record.paper_id or f"doi-{_slug(record.doi)}",
            title=record.title, authors=list(record.authors),
            year=record.year, doi=record.doi, venue=record.venue,
            source=source, text=text), \
            f"read from the {source} ({record_note})"
    abstract = (getattr(record, "abstract", "") or "").strip()
    if abstract:
        return PaperText(
            paper_id=record.paper_id or f"doi-{_slug(record.doi)}",
            title=record.title, authors=list(record.authors),
            year=record.year, doi=record.doi, venue=record.venue,
            source="abstract only", text=abstract), \
            ("read from the abstract only — the full text is not "
             "openly available")
    return None, (f"'{record.title}' has no readable text — no abstract "
                  "and no open-access PDF")


# ── explanation ────────────────────────────────────────────────────────────


@dataclass
class Finding:
    """One key finding, stated and quoted verbatim from the paper."""

    statement: str
    quote: str
    section: str

    def to_dict(self) -> dict[str, Any]:
        return {"statement": self.statement, "quote": self.quote,
                "section": self.section}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(
            statement=str(data.get("statement") or ""),
            quote=str(data.get("quote") or ""),
            section=str(data.get("section") or ""))


@dataclass
class PaperExplanation:
    """The plain-language reading of one paper."""

    paper_id: str
    title: str
    overview: str
    #: "extractive" (the paper's own sentences, selected) or "llm"
    #: (an LLM rewording that passed the number/name invariance check).
    overview_source: str = "extractive"
    question: str = ""
    method: str = ""
    findings: list[Finding] = field(default_factory=list)
    hard_parts: list[Explanation] = field(default_factory=list)
    why_it_matters: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "overview": self.overview,
            "overview_source": self.overview_source,
            "question": self.question,
            "method": self.method,
            "findings": [f.to_dict() for f in self.findings],
            "hard_parts": [h.to_dict() for h in self.hard_parts],
            "why_it_matters": self.why_it_matters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperExplanation:
        return cls(
            paper_id=str(data.get("paper_id") or ""),
            title=str(data.get("title") or ""),
            overview=str(data.get("overview") or ""),
            overview_source=str(data.get("overview_source") or "extractive"),
            question=str(data.get("question") or ""),
            method=str(data.get("method") or ""),
            findings=[Finding.from_dict(f)
                      for f in (data.get("findings") or [])
                      if isinstance(f, dict)],
            hard_parts=[Explanation.from_dict(h)
                        for h in (data.get("hard_parts") or [])
                        if isinstance(h, dict)],
            why_it_matters=str(data.get("why_it_matters") or ""),
        )


def _numbers(text: str) -> list[str]:
    return [n.replace(" ", "") for n in _NUMBER_RE.findall(text or "")]


def _entities(text: str) -> list[str]:
    found = []
    for match in _ENTITY_RE.finditer(text or ""):
        word = match.group(0)
        if match.start() == 0:
            continue
        if word not in found:
            found.append(word)
    return found


def _invariance_holds(source: str, candidate: str) -> bool:
    """Every number and named entity in the source must survive.

    The guard that keeps an LLM rewording honest: a fluent rewrite
    that drops a result's number or a name is not a plainer
    explanation, it is a different claim.
    """
    candidate_lower = (candidate or "").lower()
    for number in _numbers(source):
        if number.lower() not in candidate_lower:
            return False
    return all(entity.lower() in candidate_lower
               for entity in _entities(source))


def _llm_rewrite(text: str, must_keep: str,
                 llm_fn: Callable[[str], str] | None) -> str | None:
    """Ask the LLM to reword ``text`` in plain language, checked.

    Returns the rewording only when every number and named entity
    in ``must_keep`` survives it; anything else returns None and
    the caller keeps the extractive text. Never raises.
    """
    if llm_fn is None:
        return None
    prompt = (
        "Rewrite the following summary of a research paper in plain, "
        "simple language, keeping every number, name, and factual "
        "detail exactly as written. Do not add anything the summary "
        "does not say.\n\nSummary:\n" + text
    )
    try:
        candidate = (llm_fn(prompt) or "").strip()
    except Exception:  # noqa: BLE001 — a dead model is not a failed read
        return None
    if not candidate:
        return None
    if not _invariance_holds(must_keep, candidate):
        _log.info("paper reader: LLM rewording dropped a number or "
                  "name — keeping the extractive text")
        return None
    return candidate


def explain_paper(paper: PaperText, *,
                  llm_fn: Callable[[str], str] | None = None
                  ) -> PaperExplanation:
    """Explain a paper in plain language.

    The extractive reading is always built first: what the paper
    asks, what it did, what it found (each finding quoted verbatim
    from the paper), why it matters, and the hard parts unpacked.
    An LLM rewording of the overview is used only when it passes
    the number/name invariance check; otherwise the extractive
    overview stands, labelled as such.
    """
    sections = paper.sections()
    question_sents = _pick(sections, ("abstract", "introduction"),
                           _QUESTION_CUES, 2)
    method_sents = _pick(sections, ("methods", "methodology",
                                    "materials and methods"),
                         _METHOD_CUES, 2)
    finding_sents: list[tuple[str, str]] = []
    seen: set[str] = set()
    for section in sections:
        if section.heading.lower() not in ("results", "discussion"):
            continue
        for sentence in _sentences(section.text):
            if sentence in seen:
                continue
            seen.add(sentence)
            score = _score_sentence(sentence, _FINDING_CUES)
            if score > 0:
                finding_sents.append((sentence, section.heading))
    finding_sents.sort(
        key=lambda kv: (-_score_sentence(kv[0], _FINDING_CUES),
                        kv[0]))
    findings = [Finding(statement=s, quote=s, section=h)
                for s, h in finding_sents[:4]]
    matter_sents = _pick(sections, ("discussion", "conclusion",
                                    "conclusions"), _MATTER_CUES, 2)
    overview_parts = question_sents + method_sents[:1] + \
        [f.statement for f in findings[:3]]
    overview = " ".join(overview_parts)
    overview_source = "extractive"
    if llm_fn is not None and overview:
        rewritten = _llm_rewrite(overview, overview, llm_fn)
        if rewritten is not None:
            overview, overview_source = rewritten, "llm"
    return PaperExplanation(
        paper_id=paper.paper_id,
        title=paper.title,
        overview=overview,
        overview_source=overview_source,
        question=" ".join(question_sents),
        method=" ".join(method_sents),
        findings=findings,
        hard_parts=detect_hard_parts(paper.text),
        why_it_matters=" ".join(matter_sents),
    )


def explain_span(paper: PaperText, span_ref: str, *,
                 llm_fn: Callable[[str], str] | None = None
                 ) -> dict[str, Any]:
    """Unpack one hard part by ref ("equation 1", "table 1").

    Returns {"ok", "explanation"} — or {"ok": False, "error"} when
    the paper carries no such part. An LLM walkthrough is used only
    when it keeps every symbol and number the deterministic
    unpacking names.
    """
    ref = (span_ref or "").strip().lower()
    for part in detect_hard_parts(paper.text):
        base = part.span_ref.split(" (")[0].lower()
        if ref == base:
            if llm_fn is not None:
                symbols = _entities(part.plain_text) + _numbers(
                    part.plain_text)
                must_keep = part.plain_text
                rewritten = _llm_rewrite(part.plain_text, must_keep, llm_fn)
                if rewritten is not None and all(
                        s.lower() in rewritten.lower() for s in symbols):
                    part = Explanation(
                        span_ref=part.span_ref, kind=part.kind,
                        plain_text=rewritten,
                        grounded_quotes=part.grounded_quotes)
            return {"ok": True, "explanation": part}
    return {"ok": False,
            "error": f"no {ref or 'such span'} in this paper"}


# ── asking against the paper ───────────────────────────────────────────────


def _ask_deterministic(paper: PaperText,
                       question: str) -> dict[str, Any]:
    """Answer from the paper's own sentences, by word overlap.

    "How many" questions prefer sentences carrying numbers; the
    question's nouns match through a small synonym map (participants
    → recruited nurses). A question nothing in the paper touches is
    refused, never answered from outside the paper.
    """
    qwords = _content_words(question)
    expanded: set[str] = set()
    for word in qwords:
        expanded.add(_stem(word))
        for synonym in _ASK_SYNONYMS.get(word, (word,)):
            expanded.add(_stem(synonym))
    wants_number = bool(re.search(r"\bhow many\b|\bhow much\b",
                                  question.lower()))
    scored: list[tuple[float, str, str]] = []
    for section in paper.sections():
        for sentence in _sentences(section.text):
            words = {_stem(w) for w in _content_words(sentence)}
            hits = len(expanded & words)
            score = float(hits)
            if wants_number and re.search(r"\d", sentence):
                score += 2.0
            if score > 0:
                scored.append((score, sentence, section.heading))
    scored.sort(key=lambda kv: (-kv[0], kv[1]))
    if not scored or scored[0][0] < 2.0:
        return {"ok": False,
                "error": ("the paper does not cover that — no sentence "
                          "in it answers the question"),
                "quotes": []}
    top = scored[:3]
    answer = " ".join(s for _, s, _ in top)
    return {"ok": True, "answer": answer,
            "quotes": [s for _, s, _ in top],
            "sections": [h for _, _, h in top]}


def ask_paper(paper: PaperText, question: str, *,
              llm_fn: Callable[[str], str] | None = None
              ) -> dict[str, Any]:
    """Answer a question against the paper's own text.

    With an LLM, the grounded session answers with citations and
    refuses when the paper does not cover the question; without
    one, the paper's most relevant sentences are quoted directly.
    Either way the answer never comes from outside the paper.
    """
    question = (question or "").strip()
    if not question:
        return {"ok": False, "error": "ask needs a question", "quotes": []}
    if llm_fn is not None:
        try:
            from .grounded import GroundedSession

            session = GroundedSession()
            session.add_text(paper.text, title=paper.title)
            grounded = session.ask(question, llm_fn=llm_fn)
            if not grounded.refused and grounded.text.strip():
                quotes = [s.snippet for s in grounded.sources]
                return {"ok": True, "answer": grounded.text,
                        "quotes": quotes,
                        "confidence": grounded.confidence}
        except Exception as exc:  # noqa: BLE001 — fall back to quotes
            _log.debug("paper reader: grounded ask failed (%s) — "
                       "deterministic fallback", exc)
    return _ask_deterministic(paper, question)


# ── citations ────────────────────────────────────────────────────────────


def _apa_authors(authors: list[str]) -> str:
    """'Ada Lovelace, Grace Hopper' -> 'Lovelace, A., & Hopper, G.'"""
    parts = []
    for name in authors:
        bits = name.split()
        if not bits:
            continue
        last = bits[-1]
        first = bits[0][0] + "." if bits[0] else ""
        parts.append(f"{last}, {first}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) <= 3:
        return ", ".join(parts[:-1]) + ", & " + parts[-1]
    return parts[0] + ", et al."


def _mla_authors(authors: list[str]) -> str:
    """First author 'Last, First', the rest 'First Last'."""
    if not authors:
        return ""
    if len(authors) == 1:
        bits = authors[0].split()
        return f"{bits[-1]}, {' '.join(bits[:-1])}" if len(bits) > 1 \
            else authors[0]
    if len(authors) == 2:
        return f"{_mla_authors([authors[0]])}, and {authors[1]}"
    return f"{_mla_authors([authors[0]])}, et al."


def _bibtex_key(record: Any) -> str:
    last = ""
    if getattr(record, "authors", None):
        last = record.authors[0].split()[-1].lower()
    first_words = re.findall(r"[a-z0-9]+",
                             (getattr(record, "title", "") or "").lower())
    first_word = first_words[0] if first_words else "paper"
    year = str(getattr(record, "year", "") or "")
    return f"{last}{year}{first_word}"


def _bibtex_author(name: str) -> str:
    """'Ada Lovelace' -> 'Lovelace, Ada' (BibTeX author form)."""
    bits = name.split()
    if len(bits) > 1:
        return f"{bits[-1]}, {' '.join(bits[:-1])}"
    return name


def format_citation(record: Any, style: str) -> str:
    """Format a paper's citation in APA, MLA, Chicago, or BibTeX.

    Built from the record's own fields — authors, year, title,
    venue, DOI, URL — never from a guessed value. A missing year
    is "n.d." in APA and omitted elsewhere; a missing DOI falls
    back to the record URL. Raises ValueError for an unknown style.
    """
    style = (style or "").strip().lower()
    if style not in ("apa", "mla", "chicago", "bibtex"):
        raise ValueError(
            "unknown citation style — apa | mla | chicago | bibtex")
    title = (getattr(record, "title", "") or "").strip()
    venue = (getattr(record, "venue", "") or "").strip()
    year = getattr(record, "year", None)
    doi = (getattr(record, "doi", "") or "").strip()
    url = (getattr(record, "url", "") or "").strip()
    doi_url = f"https://doi.org/{doi}" if doi else url
    authors = list(getattr(record, "authors", None) or [])
    if style == "bibtex":
        lines = [f"@article{{{_bibtex_key(record)},"]
        lines.append(
            "  author = {" + " and ".join(
                _bibtex_author(a) for a in authors) + "},")
        lines.append(f"  title = {{{title}}},")
        if venue:
            lines.append(f"  journal = {{{venue}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        if doi_url:
            lines.append(f"  url = {{{doi_url}}}")
        lines.append("}")
        return "\n".join(lines)
    if style == "apa":
        who = _apa_authors(authors) or title
        when = f"({year})." if year else "(n.d.)."
        head = f"{who} {when}" if authors else f"{title} {when}"
        body = title if not authors else title
        parts = [head, f"{body}."]
        if venue:
            parts.append(f"{venue}.")
        if doi_url:
            parts.append(doi_url)
        return " ".join(parts)
    if style == "mla":
        who = _mla_authors(authors)
        head = f"{who}. " if who else ""
        parts = [f'{head}"{title}."']
        if venue:
            parts.append(f"{venue},")
        if year:
            parts.append(f"{year},")
        if doi:
            parts.append(f"doi:{doi}.")
        elif doi_url:
            parts.append(f"{doi_url}.")
        return " ".join(parts)
    # chicago
    who = _mla_authors(authors)
    head = f"{who}. " if who else ""
    parts = [f'{head}"{title}."']
    if venue and year:
        parts.append(f"{venue} ({year}).")
    elif venue:
        parts.append(f"{venue}.")
    elif year:
        parts.append(f"({year}).")
    if doi_url:
        parts.append(doi_url + ".")
    return " ".join(parts)


# ── paraphrase ────────────────────────────────────────────────────────────


def _simplify(span: str) -> str:
    """Deterministic simplification: glossary swaps + citation drop.

    Word-boundary substitutions only, so numbers and names survive
    untouched. Parenthetical author citations are dropped — they
    are not part of the claim.
    """
    text = _CITATION_PAREN_RE.sub("", span)
    for fancy, plain in _PLAIN_WORDS:
        pattern = re.compile(r"\b" + re.escape(fancy) + r"\b",
                             re.IGNORECASE)

        def _swap(match: re.Match, _plain: str = plain) -> str:
            word = match.group(0)
            if word[:1].isupper():
                return _plain[:1].upper() + _plain[1:]
            return _plain

        text = pattern.sub(_swap, text)
    return " ".join(text.split())


def paraphrase_span(span: str, *,
                    llm_fn: Callable[[str], str] | None = None
                    ) -> dict[str, Any]:
    """Paraphrase a span more simply, without losing a fact.

    The LLM rung goes first when one is given; the deterministic
    glossary rung is the fallback. Either way the invariance check
    runs: every number and named entity in the source must survive
    in the paraphrase. When no rung preserves them, the original
    span is quoted (method "quoted") rather than serving a paraphrase
    that lost a fact.
    """
    span = (span or "").strip()
    if not span:
        return {"ok": False, "error": "nothing to paraphrase"}
    if llm_fn is not None:
        prompt = (
            "Paraphrase the following sentence more simply, in plain "
            "language, keeping every number, name, and factual detail "
            "exactly as written. Do not add anything it does not say.\n\n"
            f"Sentence:\n{span}"
        )
        try:
            candidate = (llm_fn(prompt) or "").strip()
        except Exception:  # noqa: BLE001 — fall through to the rung below
            candidate = ""
        if candidate and _invariance_holds(span, candidate):
            return {"ok": True, "paraphrase": candidate,
                    "method": "llm", "preserved": True, "source": span}
    simplified = _simplify(span)
    if _invariance_holds(span, simplified):
        if simplified != span:
            return {"ok": True, "paraphrase": simplified,
                    "method": "simplified", "preserved": True,
                    "source": span}
        if llm_fn is None:
            return {"ok": True, "paraphrase": simplified,
                    "method": "unchanged", "preserved": True,
                    "source": span}
    return {"ok": True, "paraphrase": span, "method": "quoted",
            "preserved": True, "source": span}


# ── durable store ────────────────────────────────────────────────────────


class PaperReadingStore:
    """Durable paper readings + asked questions.

    <home>/data/paper_readings.json, atomic writes (temp file +
    replace). A reading is stored once — explain it once, ask
    against it as often as the owner likes.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / READING_FILENAME)
        self._data: dict[str, Any] = {"readings": {}, "questions": []}
        self._load()

    def _load(self) -> None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                if isinstance(data.get("readings"), dict):
                    self._data["readings"] = data["readings"]
                if isinstance(data.get("questions"), list):
                    self._data["questions"] = data["questions"]
        except (OSError, ValueError):
            self._data = {"readings": {}, "questions": []}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def save(self, paper: PaperText,
             explanation: PaperExplanation) -> None:
        self._data["readings"][paper.paper_id] = {
            "paper": paper.to_dict(),
            "explanation": explanation.to_dict(),
            "saved_at": datetime.now(UTC).isoformat(),
        }
        self._save()

    def get(self, reading_id: str) -> dict[str, Any] | None:
        row = self._data["readings"].get(reading_id or "")
        return dict(row) if isinstance(row, dict) else None

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for row in self._data["readings"].values():
            if isinstance(row, dict):
                paper = row.get("paper") or {}
                rows.append({
                    "reading_id": str(paper.get("paper_id") or ""),
                    "title": str(paper.get("title") or ""),
                    "year": paper.get("year"),
                })
        return rows

    def log_question(self, reading_id: str, question: str,
                     answer: str) -> None:
        self._data["questions"].append({
            "reading_id": reading_id,
            "question": question,
            "answer": answer,
            "at": datetime.now(UTC).isoformat(),
        })
        self._save()


def open_reading_store(home: str | Path | None = None
                       ) -> PaperReadingStore:
    """Open (creating if needed) the durable reading store."""
    return PaperReadingStore(_default_home(home) / "data" / READING_FILENAME)


# ── one-call pipeline + reply ────────────────────────────────────────────


def read_paper(query: str, *, home: str | Path | None = None,
               fetch_record: Callable[..., Any] | None = None,
               fetch_text: Callable[[str], str | None] | None = None,
               llm_fn: Callable[[str], str] | None = None
               ) -> dict[str, Any]:
    """Read a paper end to end: acquire → explain → store.

    Returns {"ok", "reading_id", "paper", "explanation", "reply"}
    or {"ok": False, "error"}. Never raises on the fetch paths.
    """
    paper, note = acquire_paper(query, fetch_record=fetch_record,
                                fetch_text=fetch_text)
    if paper is None:
        return {"ok": False, "error": note}
    explanation = explain_paper(paper, llm_fn=llm_fn)
    with contextlib.suppress(Exception):
        open_reading_store(home).save(paper, explanation)
    result = {"ok": True, "reading_id": paper.paper_id,
              "paper": paper.to_dict(),
              "explanation": explanation.to_dict(), "note": note}
    result["reply"] = format_reading_reply(result)
    return result


def format_reading_reply(result: dict[str, Any]) -> str:
    """One chat/CLI rendering for any paper-reader result."""
    if not isinstance(result, dict) or not result.get("ok"):
        error = str((result or {}).get("error") or "paper read failed")
        return f"paper read failed: {error}"
    paper = result.get("paper") or {}
    explanation = result.get("explanation") or {}
    if isinstance(paper, PaperText):
        paper = paper.to_dict()
    if isinstance(explanation, PaperExplanation):
        explanation = explanation.to_dict()
    lines = [f"📄 {paper.get('title') or 'paper'}"]
    meta = " · ".join(p for p in [
        ", ".join((paper.get("authors") or [])[:4]),
        str(paper.get("venue") or ""),
        str(paper.get("year")) if paper.get("year") else "",
    ] if p)
    if meta:
        lines.append(meta)
    overview = (explanation.get("overview") or "").strip()
    if overview:
        source = explanation.get("overview_source") or "extractive"
        lines.append(f"In plain words ({source}): {overview}")
    findings = explanation.get("findings") or []
    if findings:
        lines.append("Key findings:")
        for finding in findings[:4]:
            quote = finding.get("quote") or finding.get("statement") or ""
            section = finding.get("section") or ""
            lines.append(f"  • “{quote}” ({section})")
    hard_parts = explanation.get("hard_parts") or []
    if hard_parts:
        lines.append("Hard parts, unpacked:")
        for part in hard_parts[:6]:
            lines.append(f"  • {part.get('span_ref')} "
                         f"({part.get('kind')}): {part.get('plain_text')}")
    why = (explanation.get("why_it_matters") or "").strip()
    if why:
        lines.append(f"Why it matters: {why}")
    source = str(paper.get("source") or "")
    if source == "abstract only":
        lines.append("read from the abstract only — the full text is "
                     "not openly available.")
    elif source:
        lines.append(f"read from the {source}.")
    reading_id = result.get("reading_id") or paper.get("paper_id") or ""
    if reading_id:
        lines.append(f"ask more: /readpaper ask {reading_id} <question>")
    return "\n".join(lines)
