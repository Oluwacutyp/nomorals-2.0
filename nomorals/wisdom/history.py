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

#: Era buckets for the timeline, mirroring how the great esotericism
#: timelines are organized (Ancient / Medieval / Early Modern /
#: Modern & Contemporary). (start, end) inclusive, BCE years negative.
ERAS: tuple[tuple[str, int, int], ...] = (
    ("Ancient", -3000, 500),
    ("Medieval", 501, 1500),
    ("Early Modern", 1501, 1800),
    ("Modern & Contemporary", 1801, 2100),
)


def era_of(year: int) -> str:
    """The era bucket a year falls in ("" when outside the timeline)."""
    for name, start, end in ERAS:
        if start <= year <= end:
            return name
    return ""


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


    # -- era / parallel / density / gaps ------------------------------
    def events_by_era(self, era: str = "") -> dict[str, list[dict]]:
        """Events grouped by era bucket. ``era`` selects one bucket
        (fail fast on unknown names); empty returns all buckets."""
        known = [name for name, _, _ in ERAS]
        if era and era not in known:
            raise HistoryError(
                f"unknown era {era!r}; known: {', '.join(known)}")
        grouped: dict[str, list[dict]] = {name: [] for name in known}
        for e in self._events:
            name = era_of(e["start"])
            if name:
                grouped[name].append(dict(e))
        for events in grouped.values():
            events.sort(key=lambda e: (e["start"], e["end"]))
        if era:
            return {era: grouped[era]}
        return grouped

    def parallel_at(self, year: int, window: int = 50) -> dict[str, Any]:
        """The "everything at once" view: events overlapping
        [year - window, year + window], grouped by tradition.

        What was happening across all traditions at the same moment
        in history.
        """
        if type(year) is not int or type(window) is not int:
            raise HistoryError("year/window must be ints")
        if window < 0:
            raise HistoryError("window must be >= 0")
        lo, hi = year - window, year + window
        by_tradition: dict[str, list[dict]] = {}
        for e in self._events:
            if e["start"] <= hi and e["end"] >= lo:
                by_tradition.setdefault(e["tradition"], []).append(dict(e))
        for events in by_tradition.values():
            events.sort(key=lambda e: (e["start"], e["end"]))
        return {
            "year": year, "window": window,
            "span": [lo, hi],
            "traditions": sorted(by_tradition),
            "by_tradition": by_tradition,
            "total": sum(len(v) for v in by_tradition.values()),
        }

    def century_density(self) -> dict[str, dict[str, int]]:
        """{century_start: {tradition: count}} - where the timeline is
        thick and where it is thin. Century starts are BCE-negative
        (e.g. -200 = the 200s BCE)."""
        density: dict[str, dict[str, int]] = {}
        for e in self._events:
            first = (e["start"] // 100) * 100
            last = (e["end"] // 100) * 100
            c = first
            while c <= last:
                bucket = density.setdefault(str(c), {})
                bucket[e["tradition"]] = bucket.get(e["tradition"], 0) + 1
                c += 100
        return density

    def gaps(self, min_events_per_tradition: int = 3
             ) -> dict[str, Any]:
        """Where the timeline is thin - feeds the autonomous ingest
        hunter. Returns thin traditions and empty century spans."""
        by_tradition: dict[str, int] = {}
        for e in self._events:
            by_tradition[e["tradition"]] = \
                by_tradition.get(e["tradition"], 0) + 1
        thin = sorted(t for t, n in by_tradition.items()
                      if n < min_events_per_tradition)
        density = self.century_density()
        centuries = sorted(int(c) for c in density)
        empty_spans: list[list[int]] = []
        if centuries:
            run: list[int] = []
            for c in range(centuries[0], centuries[-1] + 100, 100):
                if str(c) not in density:
                    run.append(c)
                else:
                    if len(run) >= 2:
                        empty_spans.append([run[0], run[-1]])
                    run = []
            if len(run) >= 2:
                empty_spans.append([run[0], run[-1]])
        return {
            "thin_traditions": thin,
            "tradition_counts": dict(sorted(by_tradition.items())),
            "empty_century_spans": empty_spans,
            "total_events": len(self._events),
        }

    def render_ascii(self, events: "list[dict] | None" = None,
                     width: int = 72) -> str:
        """Render events as a terminal timeline.

        A century ruler on top, one bar per event spanning its
        [start, end], grouped under era headers with tradition tags -
        presentation, not a JSON dump.
        """
        evs = ([dict(e) for e in events] if events is not None
               else self.events())
        if not evs:
            return "(no events)"
        lo = min(e["start"] for e in evs)
        hi = max(e["end"] for e in evs)
        span = max(1, hi - lo)
        width = max(40, min(width, 120))

        def pos(year: int) -> int:
            return int((year - lo) / span * (width - 1))

        ruler = [" "] * width
        # Label every N centuries so labels never collide: each label
        # needs ~8 columns.
        n_centuries = max(1, (hi - lo) // 100)
        step = max(1, -(-n_centuries // max(1, width // 8)))
        for c in range((lo // 100) * 100, hi + 1, 100 * step):
            x = pos(c)
            label = f"{abs(c)}" + ("BCE" if c < 0 else "")
            for i, ch in enumerate(label):
                if x + i < width:
                    ruler[x + i] = ch
        lines = ["".join(ruler).rstrip(), "\u2500" * width]

        def fmt_year(y: int) -> str:
            return f"{abs(y)} BCE" if y < 0 else f"{y} CE"

        current_era = ""
        for e in sorted(evs, key=lambda e: (e["start"], e["end"])):
            era = era_of(e["start"]) or era_of(e["end"])
            if era and era != current_era:
                current_era = era
                lines.append(f"\n\u25c8 {era.upper()}")
            x0 = pos(e["start"])
            x1 = pos(max(e["end"], e["start"]))
            bar = ["\u00b7"] * width
            for x in range(x0, min(x1 + 1, width)):
                bar[x] = "\u2501"
            if 0 <= x0 < width:
                bar[x0] = "\u25cf"
            title = e["title"]
            room = max(12, width - x1 - 3 - len(e["tradition"]))
            if len(title) > room:
                title = title[:room - 1] + "\u2026"
            bar_line = (f"{''.join(bar[:x1 + 1])} {title} "
                        f"[{e['tradition']}]")
            lines.append(bar_line[:width])
            detail = (f"  {fmt_year(e['start'])} \u2013 "
                      f"{fmt_year(e['end'])} \u00b7 {e['region']}")
            lines.append(detail[:width])
        return "\n".join(lines).rstrip()

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
