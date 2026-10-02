"""HistoryEngine — the historical timeline of esoteric traditions.

Loads the curated seed timeline from the package data directory
(``nomorals/wisdom/data/timeline.json``) and validates it fail-fast at
load: a malformed dataset is a broken organ, not an empty one.

Queries never flatten traditions: every event carries its tradition,
region, title, summary, and sources, and cross-tradition views keep
each passage's tradition and canon status visible.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .errors import HistoryError

# Fields every timeline entry must carry, and how each is validated.
_FIELDS = ("start", "end", "tradition", "region", "title", "summary",
           "sources")
_NONEMPTY_STR_FIELDS = ("tradition", "region", "title", "summary")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _data_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "timeline.json"


class HistoryEngine:
    """Validated, queryable timeline of the esoteric traditions."""

    def __init__(self, context: Any, data_path: Path | None = None) -> None:
        self.context = context
        self._data_path = Path(data_path) if data_path else _data_path()
        self._events: list[dict[str, Any]] = self._load()

    # ── load + fail-fast validation ───────────────────────────────────
    def _load(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HistoryError(
                f"cannot read timeline dataset at {self._data_path}: "
                f"{exc}") from exc
        if not isinstance(raw, list):
            raise HistoryError(
                f"timeline dataset at {self._data_path} must be a list "
                f"of events, got {type(raw).__name__}")
        if not raw:
            raise HistoryError(
                f"timeline dataset at {self._data_path} is empty")
        return [self._validate_entry(item, index)
                for index, item in enumerate(raw)]

    @staticmethod
    def _validate_entry(item: Any, index: int) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise HistoryError(
                f"timeline entry {index} is not an object")
        missing = [f for f in _FIELDS if f not in item]
        if missing:
            raise HistoryError(
                f"timeline entry {index} missing fields: "
                f"{', '.join(missing)}")
        label = f"timeline entry {index} ({item.get('title')!r})"
        start, end = item["start"], item["end"]
        if type(start) is not int or type(end) is not int:
            raise HistoryError(
                f"{label}: start/end must be ints, got "
                f"{type(start).__name__}/{type(end).__name__}")
        if start > end:
            raise HistoryError(
                f"{label}: start {start} is after end {end}")
        for field in _NONEMPTY_STR_FIELDS:
            value = item[field]
            if not isinstance(value, str) or not value.strip():
                raise HistoryError(
                    f"{label}: {field!r} must be a non-empty string")
        sources = item["sources"]
        if (not isinstance(sources, list) or not sources
                or any(not isinstance(s, str) or not s.strip()
                       for s in sources)):
            raise HistoryError(
                f"{label}: sources must be a non-empty list of strings")
        return {field: item[field] for field in _FIELDS}

    # ── queries ───────────────────────────────────────────────────────
    def traditions(self) -> list[str]:
        """All traditions present in the dataset, sorted."""
        return sorted({e["tradition"] for e in self._events})

    def events(self) -> list[dict[str, Any]]:
        """The full validated timeline, sorted by start."""
        return [dict(e) for e in sorted(
            self._events, key=lambda e: (e["start"], e["end"]))]

    def timeline(self, tradition: str = "", start_year: int = -3000,
                 end_year: int = 2100) -> list[dict]:
        """Events overlapping [start_year, end_year], optionally filtered
        by tradition. Sorted by start.

        Unknown tradition strings fail fast (with the known list) rather
        than silently returning empty.
        """
        if not isinstance(tradition, str):
            raise HistoryError(
                f"tradition must be a string, got {type(tradition).__name__}")
        if type(start_year) is not int or type(end_year) is not int:
            raise HistoryError("start_year/end_year must be ints")
        if start_year > end_year:
            raise HistoryError(
                f"start_year {start_year} is after end_year {end_year}")
        known = self.traditions()
        if tradition and tradition not in known:
            raise HistoryError(
                f"unknown tradition {tradition!r}; known traditions: "
                f"{', '.join(known)}")
        matched = [dict(e) for e in self._events
                   if e["start"] <= end_year and e["end"] >= start_year
                   and (not tradition or e["tradition"] == tradition)]
        matched.sort(key=lambda e: (e["start"], e["end"]))
        return matched

    def lineage(self, figure_or_school: str) -> list[dict]:
        """Events whose title or summary mention the query
        (case-insensitive), sorted by start — traces a figure, school,
        or idea across traditions and centuries."""
        query = (figure_or_school or "").strip().lower()
        if not query:
            raise HistoryError("lineage() needs a non-empty query")
        matched = [dict(e) for e in self._events
                   if query in e["title"].lower()
                   or query in e["summary"].lower()]
        matched.sort(key=lambda e: (e["start"], e["end"]))
        return matched

    def compare(self, topic: str, corpus: Any = None) -> dict:
        """Cross-tradition view of a topic.

        Pulls passages from an optional CanonCorpus (or WisdomKeeper) via
        ``ask()`` and pairs them with matching timeline events.
        Passages keep their tradition and canon status visible — nothing
        is flattened into a single view. Without a corpus, passages is
        empty but timeline_context still works.
        """
        topic = (topic or "").strip()
        if not topic:
            raise HistoryError("compare() needs a non-empty topic")
        passages: list[dict[str, Any]] = []
        if corpus is not None:
            tradition_by_work = self._tradition_index(corpus)
            answer = corpus.ask(topic)
            for hit in answer.passages:
                d = hit.to_dict()
                if not d.get("tradition"):
                    d["tradition"] = tradition_by_work.get(
                        d.get("work", ""), "")
                passages.append(d)
        return {
            "topic": topic,
            "passages": passages,
            "timeline_context": self._topic_events(topic),
        }

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _tradition_index(corpus: Any) -> dict[str, str]:
        """Map manifest work titles → tradition for a corpus-like object.

        Accepts a CanonCorpus (has ``list()``) or a WisdomKeeper (has
        ``.corpus.list()``). Anything else yields an empty index rather
        than failing — the passages are still returned with whatever
        provenance they carry.
        """
        entries: list[Any] = []
        if hasattr(corpus, "list"):
            entries = corpus.list() or []
        elif hasattr(corpus, "corpus") and hasattr(corpus.corpus, "list"):
            entries = corpus.corpus.list() or []
        index: dict[str, str] = {}
        for e in entries:
            title = getattr(e, "title", "") or ""
            if title:
                index[title] = getattr(e, "tradition", "") or ""
        return index

    def _topic_events(self, topic: str, limit: int = 8) -> list[dict]:
        """Events whose title/summary/tradition match topic tokens,
        most-relevant first (then by start)."""
        tokens = [t for t in _TOKEN_RE.findall(topic.lower())
                  if len(t) > 2]
        if not tokens:
            return []
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for e in self._events:
            text = f"{e['title']} {e['summary']} {e['tradition']}".lower()
            hits = sum(1 for t in tokens if t in text)
            if hits:
                scored.append((hits, e["start"], e))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [dict(e) for _, _, e in scored[:limit]]
