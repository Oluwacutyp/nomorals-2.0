"""Paper-record intelligence — the Semantic Scholar full record (52-websites #37).

The Elicit organ retrieves papers and the Connected Papers organ walks
the citation neighbourhood, but neither exposes the full record of a
paper itself. This organ is that record, as an extension of the
research family (verdict: extend, not a new organ): authors, venue,
year, citation counts, the influential citations named, a one-line
TLDR, fields of study, and the open-access PDF when one exists — plus
topic feeds that surface new papers.

Sources: the Semantic Scholar Academic Graph API, keyless
(100 requests / 5 minutes unauthenticated). The record's TLDR is
Semantic Scholar's own model TLDR when the API carries one; when it
does not, one is generated locally from the abstract with the
documents summarizer and stored — generated once, reused, never
recomputed per read. Citation intent comes from Semantic Scholar's
own classification when present; when a citing paper carries no
intents, a local cue-word classifier reads the citation contexts.

Never raises on the fetch paths: a dead source, an unknown paper, or
a record with no TLDR and no PDF is reported honestly — no paper,
author, citation or TLDR is ever invented.
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

__all__ = [
    "CitationEdge",
    "PaperRecord",
    "PaperRecordStore",
    "check_feeds",
    "classify_citation_intent",
    "ensure_paper_feed_job",
    "ensure_tldr",
    "fetch_influential_citations",
    "fetch_paper_record",
    "format_record_reply",
    "local_tldr",
    "open_record_store",
    "run_paper_feed_tick_once",
    "search_records",
]

_log = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
UA = "Mozilla/5.0 (compatible; NoMoralsPaperRecords/1.0; +personal research)"
TIMEOUT = 20

#: Fields for a full paper record. ``tldr`` is the model's own
#: one-sentence summary; ``openAccessPdf`` the PDF handoff.
RECORD_FIELDS = (
    "title,abstract,venue,year,authors,citationCount,referenceCount,"
    "influentialCitationCount,externalIds,url,openAccessPdf,"
    "fieldsOfStudy,tldr"
)
CITATION_FIELDS = (
    "citingPaper.title,citingPaper.authors,citingPaper.year,"
    "citingPaper.citationCount,contexts,intents,isInfluential"
)
SEARCH_FIELDS = "paperId,title,year,authors"

RECORD_FILENAME = "paper_records.json"

PAPER_FEED_JOB = "paper feed tick"
PAPER_FEED_TOOL = "__paper_feed_tick__"
#: Daily 08:00 local — 02:00 is the backup drill's, 03:00 the AV
#: sweep's, 04:00 the airport refresh's, 05:00 the breach watch's.
PAPER_FEED_SPEC = "cron 0 8 * * *"

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+")

#: Cue words for the local citation-intent classifier, used only
#: when Semantic Scholar carries no intents for a citing paper.
_INTENT_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("methodology", ("method", "methods", "methodology", "using the",
                     "we employ", "we use", "we applied", "protocol",
                     "procedure")),
    ("result", ("results confirm", "results show", "found that",
                "showed that", "demonstrated that", "consistent with",
                "in agreement with")),
    ("background", ("introduced", "proposed", "first described",
                    "pioneered", "established", "reviewed")),
)


def _default_home(home: str | Path | None) -> Path:
    if home:
        return Path(home)
    return Path("~/.nomorals").expanduser()


def _default_fetch_json(url: str, timeout: int = TIMEOUT) -> Any | None:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _norm_doi(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.I)
    value = re.sub(r"^doi:\s*", "", value, flags=re.I)
    return value


def _looks_like_doi(query: str) -> bool:
    return bool(_DOI_RE.search(query or ""))


# ── records ────────────────────────────────────────────────────────────────


@dataclass
class CitationEdge:
    """One citing paper, with the intelligence Semantic Scholar carries.

    ``intents`` is Semantic Scholar's own classification
    (background / methodology / result) when present; when the API
    carries none, the local cue-word classifier fills it from the
    citation contexts.
    """

    citing_title: str
    citing_authors: list[str] = field(default_factory=list)
    citing_year: int | None = None
    citing_citations: int | None = None
    influential: bool = False
    intents: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)

    @classmethod
    def from_s2(cls, data: dict[str, Any]) -> CitationEdge:
        paper = data.get("citingPaper") or {}
        intents = [str(i) for i in (data.get("intents") or []) if i]
        contexts = [str(c) for c in (data.get("contexts") or []) if c]
        if not intents and contexts:
            intent = classify_citation_intent(contexts)
            if intent != "other":
                intents = [intent]
        return cls(
            citing_title=(paper.get("title") or "").strip(),
            citing_authors=[(a or {}).get("name", "")
                            for a in (paper.get("authors") or [])],
            citing_year=paper.get("year"),
            citing_citations=paper.get("citationCount"),
            influential=bool(data.get("isInfluential")),
            intents=intents,
            contexts=contexts,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "citing_title": self.citing_title,
            "citing_authors": list(self.citing_authors),
            "citing_year": self.citing_year,
            "citing_citations": self.citing_citations,
            "influential": self.influential,
            "intents": list(self.intents),
            "contexts": list(self.contexts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CitationEdge:
        return cls(
            citing_title=str(data.get("citing_title") or ""),
            citing_authors=[str(a) for a in (data.get("citing_authors") or [])],
            citing_year=data.get("citing_year"),
            citing_citations=data.get("citing_citations"),
            influential=bool(data.get("influential")),
            intents=[str(i) for i in (data.get("intents") or [])],
            contexts=[str(c) for c in (data.get("contexts") or [])],
        )


@dataclass
class PaperRecord:
    """The full record of one paper, as Semantic Scholar carries it."""

    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    year: int | None = None
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    abstract: str = ""
    citation_count: int | None = None
    reference_count: int | None = None
    influential_citation_count: int | None = None
    fields_of_study: list[str] = field(default_factory=list)
    tldr: str = ""
    tldr_model: str = ""
    #: "semantic-scholar" (the API's own TLDR), "local" (generated
    #: from the abstract and stored), or "none" (nothing to say).
    tldr_source: str = "none"
    open_access_pdf: str | None = None
    external_ids: dict[str, str] = field(default_factory=dict)
    influential: list[CitationEdge] = field(default_factory=list)
    fetched_at: str = ""

    @classmethod
    def from_s2(cls, data: dict[str, Any]) -> PaperRecord:
        title = (data.get("title") or "").strip()
        if not title:
            raise ValueError("a Semantic Scholar record needs a title")
        tldr = data.get("tldr") or {}
        tldr_text = (tldr.get("text") or "").strip()
        ext = data.get("externalIds") or {}
        oa = data.get("openAccessPdf") or {}
        return cls(
            paper_id=str(data.get("paperId") or ""),
            title=title,
            authors=[(a or {}).get("name", "")
                     for a in (data.get("authors") or [])],
            venue=str(data.get("venue") or ""),
            year=data.get("year"),
            doi=_norm_doi(ext.get("DOI") or ""),
            arxiv_id=str(ext.get("ArXiv") or ""),
            url=str(data.get("url") or ""),
            abstract=(data.get("abstract") or "").strip(),
            citation_count=data.get("citationCount"),
            reference_count=data.get("referenceCount"),
            influential_citation_count=data.get("influentialCitationCount"),
            fields_of_study=[str(f) for f in
                             (data.get("fieldsOfStudy") or []) if f],
            tldr=tldr_text,
            tldr_model=str(tldr.get("model") or ""),
            tldr_source="semantic-scholar" if tldr_text else "none",
            open_access_pdf=(oa.get("url") or None),
            external_ids={str(k): str(v) for k, v in ext.items() if v},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "venue": self.venue,
            "year": self.year,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "abstract": self.abstract,
            "citation_count": self.citation_count,
            "reference_count": self.reference_count,
            "influential_citation_count": self.influential_citation_count,
            "fields_of_study": list(self.fields_of_study),
            "tldr": self.tldr,
            "tldr_model": self.tldr_model,
            "tldr_source": self.tldr_source,
            "open_access_pdf": self.open_access_pdf,
            "external_ids": dict(self.external_ids),
            "influential": [e.to_dict() for e in self.influential],
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PaperRecord:
        record = cls(
            paper_id=str(data.get("paper_id") or ""),
            title=str(data.get("title") or ""),
            authors=[str(a) for a in (data.get("authors") or [])],
            venue=str(data.get("venue") or ""),
            year=data.get("year"),
            doi=str(data.get("doi") or ""),
            arxiv_id=str(data.get("arxiv_id") or ""),
            url=str(data.get("url") or ""),
            abstract=str(data.get("abstract") or ""),
            citation_count=data.get("citation_count"),
            reference_count=data.get("reference_count"),
            influential_citation_count=data.get(
                "influential_citation_count"),
            fields_of_study=[str(f) for f in
                             (data.get("fields_of_study") or [])],
            tldr=str(data.get("tldr") or ""),
            tldr_model=str(data.get("tldr_model") or ""),
            tldr_source=str(data.get("tldr_source") or "none"),
            open_access_pdf=data.get("open_access_pdf"),
            external_ids={str(k): str(v) for k, v in
                          (data.get("external_ids") or {}).items()},
            fetched_at=str(data.get("fetched_at") or ""),
        )
        record.influential = [
            CitationEdge.from_dict(e)
            for e in (data.get("influential") or []) if isinstance(e, dict)]
        return record


def classify_citation_intent(contexts: list[str]) -> str:
    """Local cue-word intent for a citation, when S2 carries none.

    Reads the sentences where the paper is cited: methodology words
    (method, using the, we employ) mean the citing paper builds on
    the method; result words (found that, results confirm) mean it
    leans on the finding; background words (introduced, proposed)
    mean it cites the origin. Nothing matching is "other" — never
    a confident label from no evidence.
    """
    text = " ".join(contexts or []).lower()
    if not text.strip():
        return "other"
    for intent, cues in _INTENT_CUES:
        if any(cue in text for cue in cues):
            return intent
    return "other"


def local_tldr(abstract: str) -> str:
    """A one-line TLDR generated locally from an abstract.

    The documents summarizer's extractive TLDR, one line, never
    raises — an abstract too short to summarize returns its first
    sentence, and an empty abstract returns "".
    """
    abstract = (abstract or "").strip()
    if not abstract:
        return ""
    try:
        from ..documents.summarize import tldr as _summarize_tldr

        text = _summarize_tldr(abstract, words=60)
        return " ".join(text.split())
    except Exception:  # noqa: BLE001 — fall back to the first sentence
        first = re.split(r"(?<=[.!?])\s+", abstract, maxsplit=1)[0]
        return first.strip()


def ensure_tldr(record: PaperRecord) -> PaperRecord:
    """Give a record its TLDR: Semantic Scholar's own first.

    A record that already carries the API's TLDR is untouched. One
    without it gets a locally generated TLDR from the abstract —
    generated once, stored with the record, and reused on every
    later read. A record with neither gets tldr "" and source
    "none", named honestly by the reply.
    """
    if record.tldr:
        return record
    text = local_tldr(record.abstract)
    if text:
        record.tldr = text
        record.tldr_model = "local-extractive"
        record.tldr_source = "local"
    else:
        record.tldr_source = "none"
    return record


# ── fetching ───────────────────────────────────────────────────────────────


def _record_url(paper_id: str) -> str:
    return (f"{S2_BASE}/paper/"
            f"{urllib.parse.quote(paper_id, safe='')}"
            f"?fields={urllib.parse.quote(RECORD_FIELDS, safe=',')}")


def _search_url(query: str, limit: int, fields: str) -> str:
    return (f"{S2_BASE}/paper/search?"
            + urllib.parse.urlencode(
                {"query": query, "limit": max(1, limit), "fields": fields}))


def fetch_paper_record(
        query: str, *,
        fetch_json: Callable[..., Any] | None = None) -> tuple[PaperRecord | None, str]:
    """Fetch the full record for a paper by DOI, S2 id, or title.

    A DOI or Semantic Scholar id goes straight to the paper
    endpoint; a title searches first and fetches the top hit's full
    record. Never raises: a dead source or an unknown paper returns
    (None, reason) — no record is ever invented.
    """
    fetch_json = fetch_json or _default_fetch_json
    query = (query or "").strip()
    if not query:
        return None, "no paper given — hand me a title or a DOI"
    paper_id: str | None = None
    if _looks_like_doi(query):
        paper_id = f"DOI:{_norm_doi(query)}"
    elif re.fullmatch(r"[0-9a-f]{40}", query):
        paper_id = query
    if paper_id is not None:
        data = fetch_json(_record_url(paper_id))
        if data is None:
            return None, ("Semantic Scholar: unavailable — no response "
                          "(rate limit 100 req/5min unauthenticated). "
                          "No record invented.")
        if not isinstance(data, dict) or not (data.get("title") or "").strip():
            return None, (f"Semantic Scholar: no paper for '{query}' "
                          "— the DOI or id is unknown there.")
        try:
            return ensure_tldr(PaperRecord.from_s2(data)), \
                "Semantic Scholar: full record"
        except ValueError:
            return None, (f"Semantic Scholar: no usable record for "
                          f"'{query}'.")
    data = fetch_json(_search_url(query, 1, "paperId"))
    if not isinstance(data, dict):
        return None, ("Semantic Scholar: unavailable — no response "
                      "(rate limit 100 req/5min unauthenticated). "
                      "No record invented.")
    hits = [h for h in (data.get("data") or []) if isinstance(h, dict)]
    if not hits:
        return None, f"Semantic Scholar: no paper for '{query}'."
    full = fetch_json(_record_url(str(hits[0].get("paperId") or "")))
    if full is None:
        return None, ("Semantic Scholar: unavailable — no response "
                      "(rate limit 100 req/5min unauthenticated). "
                      "No record invented.")
    if not isinstance(full, dict) or not (full.get("title") or "").strip():
        return None, (f"Semantic Scholar: no full record for "
                      f"'{query}'.")
    try:
        return ensure_tldr(PaperRecord.from_s2(full)), \
            "Semantic Scholar: full record"
    except ValueError:
        return None, f"Semantic Scholar: no usable record for '{query}'."


def search_records(
        query: str, *, limit: int = 5,
        fetch_json: Callable[..., Any] | None = None) -> tuple[list[dict[str, Any]], str]:
    """Thin search rows (id, title, year, authors) for a topic."""
    fetch_json = fetch_json or _default_fetch_json
    query = (query or "").strip()
    if not query:
        return [], "no topic given"
    data = fetch_json(_search_url(query, limit, SEARCH_FIELDS))
    if not isinstance(data, dict):
        return [], ("Semantic Scholar: unavailable — no response "
                    "(rate limit 100 req/5min unauthenticated). "
                    "No results invented.")
    rows = [{
        "paper_id": str(h.get("paperId") or ""),
        "title": (h.get("title") or "").strip(),
        "year": h.get("year"),
        "authors": [(a or {}).get("name", "")
                    for a in (h.get("authors") or [])],
    } for h in (data.get("data") or []) if isinstance(h, dict)]
    rows = [r for r in rows if r["paper_id"] and r["title"]]
    if not rows:
        return [], f"Semantic Scholar: no papers for '{query}'."
    return rows, f"Semantic Scholar: {len(rows)} paper(s)"


def fetch_influential_citations(
        paper_id: str, *, limit: int = 100,
        fetch_json: Callable[..., Any] | None = None) -> list[CitationEdge]:
    """The influential citations of a paper, named, most-cited first.

    Semantic Scholar flags the citations that shaped the citing
    paper; this keeps only those, with their intents (the API's
    own, or the local classifier's when the API carries none).
    Never raises — a dead source returns [].
    """
    fetch_json = fetch_json or _default_fetch_json
    if not (paper_id or "").strip():
        return []
    url = (f"{S2_BASE}/paper/"
           f"{urllib.parse.quote(paper_id, safe='')}/citations?"
           + urllib.parse.urlencode(
               {"fields": CITATION_FIELDS,
                "limit": max(1, min(limit, 1000))}))
    try:
        data = fetch_json(url)
    except Exception:  # noqa: BLE001 — a source never kills the read
        return []
    if not isinstance(data, dict):
        return []
    edges = [CitationEdge.from_s2(c)
             for c in (data.get("data") or []) if isinstance(c, dict)]
    influential = [e for e in edges if e.influential and e.citing_title]
    influential.sort(key=lambda e: (-(e.citing_citations or 0),
                                    e.citing_title))
    return influential


# ── durable store + feeds ──────────────────────────────────────────────────


class PaperRecordStore:
    """Durable paper records + topic feeds: <home>/data/paper_records.json.

    Atomic writes (temp file + replace). A record is stored with the
    TLDR it carries — Semantic Scholar's own, or the locally
    generated one — so the TLDR is generated once and reused on
    every later read. A feed is a watched topic: the tick checks it
    against Semantic Scholar search and reports only papers the
    store has never seen.
    """

    def __init__(self, path: str | Path = "") -> None:
        self.path = str(path) if path else str(
            _default_home(None) / "data" / RECORD_FILENAME)
        self._data: dict[str, Any] = {"records": {}, "feeds": {}}
        self._load()

    def _load(self) -> None:
        try:
            raw = Path(self.path).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                if isinstance(data.get("records"), dict):
                    self._data["records"] = data["records"]
                if isinstance(data.get("feeds"), dict):
                    self._data["feeds"] = data["feeds"]
        except (OSError, ValueError):
            self._data = {"records": {}, "feeds": {}}

    def _save(self) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def save(self, record: PaperRecord) -> None:
        self._data["records"][record.paper_id] = record.to_dict()
        self._save()

    def get(self, paper_id: str) -> PaperRecord | None:
        row = self._data["records"].get(paper_id or "")
        if not isinstance(row, dict):
            return None
        try:
            return PaperRecord.from_dict(row)
        except Exception:  # noqa: BLE001 — a corrupt row reads as absent
            return None

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for row in self._data["records"].values():
            if isinstance(row, dict):
                rows.append({
                    "paper_id": str(row.get("paper_id") or ""),
                    "title": str(row.get("title") or ""),
                    "year": row.get("year"),
                    "citation_count": row.get("citation_count"),
                })
        return rows

    def watch(self, topic: str) -> bool:
        topic = (topic or "").strip().lower()
        if not topic:
            return False
        if topic not in self._data["feeds"]:
            self._data["feeds"][topic] = {"last_checked": "",
                                          "seen_ids": []}
            self._save()
        return True

    def unwatch(self, topic: str) -> bool:
        topic = (topic or "").strip().lower()
        if topic in self._data["feeds"]:
            del self._data["feeds"][topic]
            self._save()
            return True
        return False

    def feeds(self) -> list[str]:
        return sorted(self._data["feeds"])

    def feed_state(self, topic: str) -> dict[str, Any] | None:
        state = self._data["feeds"].get((topic or "").strip().lower())
        return dict(state) if isinstance(state, dict) else None


def open_record_store(home: str | Path | None = None) -> PaperRecordStore:
    """Open (creating if needed) the durable record store."""
    return PaperRecordStore(_default_home(home) / "data" / RECORD_FILENAME)


def check_feeds(
        store: PaperRecordStore, *,
        fetch_json: Callable[..., Any] | None = None) -> list[tuple[str, list[dict[str, Any]]]]:
    """Check every watched topic; report only papers never seen.

    Returns [(topic, [new rows])] for topics with new papers; topics
    with nothing new are silent. Seen ids persist, so a paper is
    reported exactly once.
    """
    fetch_json = fetch_json or _default_fetch_json
    found: list[tuple[str, list[dict[str, Any]]]] = []
    for topic in store.feeds():
        state = store.feed_state(topic) or {}
        seen = set(state.get("seen_ids") or [])
        rows, _note = search_records(topic, limit=10, fetch_json=fetch_json)
        new = [r for r in rows if r["paper_id"] not in seen]
        seen.update(r["paper_id"] for r in rows)
        state["seen_ids"] = sorted(seen)
        state["last_checked"] = datetime.now(UTC).isoformat()
        store._data["feeds"][topic] = state
        if new:
            found.append((topic, new))
    if found:
        store._save()
    return found


def run_paper_feed_tick_once(
        home: str | Path | None, *, context: Any = None,
        notifier: Any = None,
        fetch_json: Callable[..., Any] | None = None) -> dict[str, Any]:
    """One feed tick: check watched topics, page the owner on new papers.

    Routine and silent — a tick with no watched topics, or with
    nothing new, publishes nothing. Never raises.
    """
    try:
        store = open_record_store(home)
        found = check_feeds(store, fetch_json=fetch_json)
        if not found:
            return {"ok": True, "new": 0, "topics": []}
        if notifier is not None:
            for topic, rows in found:
                lines = [f"new on '{topic}':"]
                for row in rows[:5]:
                    lines.append(f"  {row['title']} ({row.get('year') or '?'})")
                notifier.publish("paper_feed", f"new papers: {topic}",
                                 "\n".join(lines))
        return {"ok": True,
                "new": sum(len(rows) for _, rows in found),
                "topics": [t for t, _ in found]}
    except Exception as exc:  # noqa: BLE001 — one bad tick never kills the job
        return {"ok": False, "error": f"paper feed tick failed: {exc}"}


def ensure_paper_feed_job(scheduler: Any, *,
                          spec: str = PAPER_FEED_SPEC) -> Any | None:
    """Register the daily paper-feed job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job
    that already matches is returned untouched. Best-effort — a
    scheduler that cannot store the job returns None rather than
    raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"), job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != PAPER_FEED_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            PAPER_FEED_JOB, spec, "tool",
            {"tool": PAPER_FEED_TOOL, "args": {}}, routine=True)
    except Exception:
        return None


# ── reply rendering ────────────────────────────────────────────────────────


def _fmt_count(value: int | None) -> str:
    return f"{value:,}" if isinstance(value, int) else "?"


def format_record_reply(result: dict[str, Any]) -> str:
    """One chat/CLI rendering for any paper-record result."""
    if not isinstance(result, dict) or not result.get("ok"):
        error = str((result or {}).get("error") or "paper record failed")
        return f"paper record failed: {error}"
    record = result.get("record")
    if not isinstance(record, PaperRecord):
        return "paper record failed: no record in the answer"
    lines = [f"📄 {record.title}"]
    authors = ", ".join(record.authors[:6])
    if len(record.authors) > 6:
        authors += f" +{len(record.authors) - 6} more"
    meta = " · ".join(p for p in [
        authors,
        record.venue or "",
        str(record.year) if record.year else "",
    ] if p)
    if meta:
        lines.append(meta)
    lines.append(
        f"cited {_fmt_count(record.citation_count)} times · "
        f"{_fmt_count(record.reference_count)} references · "
        f"{_fmt_count(record.influential_citation_count)} influential citations")
    if record.fields_of_study:
        lines.append("fields: " + ", ".join(record.fields_of_study))
    if record.tldr:
        lines.append(f"TLDR ({record.tldr_source}): {record.tldr}")
    else:
        lines.append("TLDR: no TLDR — Semantic Scholar carries none and "
                     "there is no abstract to summarize.")
    if record.open_access_pdf:
        lines.append(f"open-access PDF: {record.open_access_pdf}")
    else:
        lines.append("open-access PDF: no open-access PDF found for this paper.")
    if record.influential:
        lines.append(f"influential citations ({len(record.influential)}):")
        for edge in record.influential[:8]:
            intent = f" [{', '.join(edge.intents)}]" if edge.intents else ""
            lines.append(
                f"  • {edge.citing_title} ({edge.citing_year or '?'})"
                f"{intent}")
    elif record.influential_citation_count:
        lines.append("influential citations: the count is known but the "
                     "citation list could not be read.")
    if record.doi:
        lines.append(f"DOI: {record.doi}")
    return "\n".join(lines)
