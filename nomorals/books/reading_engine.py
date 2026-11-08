"""Story reading engine — she reads what she follows, when it happens.

The defect this organ fixes (owner, 2026-11-07: "the novel class
doesn't seem to go reading supposed novels I followed"): following a
story recorded it and stopped.  The single chat surface that wanted
more hand-rolled a background thread that only DOWNLOADED the story
into its library (ingestion explicitly never reads), the brain-tool
follow path triggered nothing at all, and no consumer anywhere —
no event, no watcher, no job — turned a follow, or a new chapter,
into reading.  Digestion into the story bible happened only when
the owner explicitly asked for a continuation.

The shape now is the system's own event fabric, end to end:

* :meth:`StoryReader.follow` publishes ``story.followed`` on the
  shared event bus — every follow path lands in that one method, so
  every follow is a trigger.
* This engine subscribes.  Its bus handlers never do the work on
  the bus's dispatcher thread: they enqueue, and the engine's own
  worker runs the read — fetch the story into its library
  (``StoryReader.ingest``, resumable), then work through every
  chapter she has not yet read, folding them into the story bible
  (``BibleBuilder``) in reading order.  Her per-story reading
  progress IS the bible's ``chapters_digested`` watermark, persisted
  on disk; the owner's own reading position is never touched.
* New chapters are found by the discovery sweep (a source has no
  push channel, so discovery is a poll — see
  :mod:`nomorals.books.jobs`); the sweep publishes
  ``story.chapters_discovered`` and the SAME consumer runs the read
  immediately on discovery.  The poll finds; it never does the work.
* Every finished run publishes ``story.reading_progress`` with the
  full report, so any other organ (triggers, briefings) can react.

Notifications ride the channel the trigger named: the chat that
asked (when the follow/read request carried one) or the owner's
Notifier channels.  A follow always reports its outcome — read,
partial, or failed, honestly.  A discovery run reports only when it
actually read new chapters; a boot catch-up never pings.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .bible import BibleBuilder
from .failures import record_failure
from .reader import ReaderError, StoryReader

_log = get_logger(__name__)

__all__ = [
    "StoryReadingEngine",
    "discover_new_chapters",
    "ensure_reading_engine",
    "reading_engine_for",
    "TOPIC_FOLLOWED",
    "TOPIC_DISCOVERED",
    "TOPIC_READ_REQUESTED",
    "TOPIC_PROGRESS",
]

#: A story was followed (published by StoryReader.follow).
TOPIC_FOLLOWED = "story.followed"
#: The discovery sweep found chapters the library does not hold.
TOPIC_DISCOVERED = "story.chapters_discovered"
#: An explicit ask to (re)read now — /novel sync rides this.
TOPIC_READ_REQUESTED = "story.read_requested"
#: A read run finished; data carries the full report.
TOPIC_PROGRESS = "story.reading_progress"

#: Chapters folded into the bible per build/update call (the same
#: batch StoryContinuer.prepare digests in).
_DIGEST_BATCH = 24

_REASON_FOLLOW = "follow"
_REASON_DISCOVERY = "discovery"
_REASON_SYNC = "sync"
_REASON_CATCHUP = "catchup"


class StoryReadingEngine:
    """Standing consumer: follows and discoveries become reading."""

    def __init__(self, context: Any, *,
                 chat_notify: Callable[[str, str], None] | None = None,
                 bus: Any = None) -> None:
        self.context = context
        #: (chat_key, text) delivery for pings that belong to a chat;
        #: None means every ping rides the Notifier.
        self._chat_notify = chat_notify
        self._bus = bus if bus is not None else global_bus
        self._queue: queue.Queue = queue.Queue()
        self._queued: set[str] = set()
        self._inflight: set[str] = set()
        self._lock = threading.Lock()
        self._started = False
        self._worker: threading.Thread | None = None
        self._sub_ids: list[str] = []

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> StoryReadingEngine:
        """Subscribe to the story topics (idempotent) and start the
        worker thread.  Handlers only enqueue — the bus dispatcher
        never blocks on a fetch."""
        with self._lock:
            if self._started:
                return self
            self._started = True
        self._sub_ids = [
            self._bus.subscribe(TOPIC_FOLLOWED, self._on_event),
            self._bus.subscribe(TOPIC_DISCOVERED, self._on_event),
            self._bus.subscribe(TOPIC_READ_REQUESTED, self._on_event),
        ]
        self._worker = threading.Thread(
            target=self._work, name="story-reading-engine", daemon=True)
        self._worker.start()
        _log.info("story reading engine started")
        return self

    def stop(self) -> None:
        """Unsubscribe and stop the worker (a stopped runtime must
        not keep consuming story events).  Safe to call twice."""
        with self._lock:
            if not self._started:
                return
            self._started = False
        for sub_id in self._sub_ids:
            with contextlib.suppress(Exception):  # teardown is best-effort
                self._bus.unsubscribe(sub_id)
        self._sub_ids = []
        self._queue.put(None)  # the worker's exit sentinel

    @property
    def started(self) -> bool:
        return self._started

    # ── intake ───────────────────────────────────────────────────────
    def _on_event(self, event: Any) -> None:
        data = getattr(event, "data", None) or {}
        slug = str(data.get("slug") or "")
        if not slug:
            return
        topic = str(getattr(event, "topic", "") or "")
        reason = {
            TOPIC_FOLLOWED: _REASON_FOLLOW,
            TOPIC_DISCOVERED: _REASON_DISCOVERY,
            TOPIC_READ_REQUESTED: _REASON_SYNC,
        }.get(topic, _REASON_SYNC)
        self.request_read(slug, reason=reason,
                          chat_key=str(data.get("chat_key") or ""))

    def request_read(self, slug: str, *, reason: str = _REASON_SYNC,
                     chat_key: str = "") -> bool:
        """Enqueue a read for ``slug``.  Returns False when a read for
        the story is already queued or running (one read per story at
        a time — a second trigger for the same story adds nothing:
        the running read goes to the source's current end anyway)."""
        with self._lock:
            if slug in self._queued or slug in self._inflight:
                return False
            self._queued.add(slug)
        self._queue.put((slug, reason, chat_key))
        return True

    def catch_up_all(self) -> list[str]:
        """Enqueue reads for followed stories whose library already
        holds chapters she has not digested — the restart-safe half
        of the follow contract (a follow made while no engine was
        running still gets read at the next boot).  Pure disk checks:
        source-side lag is the discovery sweep's job, not boot's.
        Returns the slugs enqueued."""
        out: list[str] = []
        try:
            reader = StoryReader(self.context)
            bibles = BibleBuilder(self.context)
            for row in reader.following():
                slug = str(row.get("slug") or "")
                if not slug:
                    continue
                state = reader.ingest_state(slug)
                bible = bibles.load(slug)
                digested = bible.chapters_digested if bible else 0
                if int(state.get("ingested") or 0) > digested and \
                        self.request_read(slug, reason=_REASON_CATCHUP):
                    out.append(slug)
        except Exception as exc:  # noqa: BLE001 - catch-up never breaks boot
            record_failure("reading_engine.catch_up_all", exc)
        return out

    # ── worker ───────────────────────────────────────────────────────
    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # stop() sentinel
                return
            slug, reason, chat_key = item
            with self._lock:
                self._queued.discard(slug)
                self._inflight.add(slug)
            try:
                report = self.read_story(slug, reason=reason)
            except Exception as exc:  # noqa: BLE001 - one bad story
                record_failure("reading_engine.read_story", exc)  # never
                report = {"slug": slug, "ok": False,  # kills the worker
                          "error": f"{type(exc).__name__}: {exc}"}
            finally:
                with self._lock:
                    self._inflight.discard(slug)
            try:
                self._announce(report, reason=reason, chat_key=chat_key)
            except Exception as exc:  # noqa: BLE001
                record_failure("reading_engine.announce", exc)
            try:
                self._bus.publish(Event(
                    topic=TOPIC_PROGRESS, data=dict(report),
                    source=__name__))
            except Exception as exc:  # noqa: BLE001
                record_failure("reading_engine.progress_publish", exc)

    # ── the read itself ──────────────────────────────────────────────
    def read_story(self, slug: str, *, reason: str = _REASON_SYNC,
                   ) -> dict[str, Any]:
        """Fetch + read one followed story, to the source's current end.

        Ingest first (resumable; a partial ingest still leaves a
        readable library), then digest every held chapter past the
        bible's watermark, in order, in batches.  When ingestion
        fails outright the chapters already on disk are still read —
        a source outage must not unmakes yesterday's library.
        Never raises: the report carries the failure honestly.
        """
        reader = StoryReader(self.context)
        bibles = BibleBuilder(self.context)
        try:
            story = reader._load(slug)
        except ReaderError as exc:
            return {"slug": slug, "ok": False, "error": str(exc)}
        slug = story.slug
        report: dict[str, Any] = {
            "slug": slug, "title": story.title, "ok": True,
            "reason": reason, "fetched": [], "failed": [],
            "ingested": 0, "total": 0, "complete": False,
            "digested_through": 0, "new_chapters": [], "error": "",
        }
        ingest_error = ""
        try:
            ing = reader.ingest(slug)
            report["fetched"] = list(ing.get("fetched") or [])
            report["failed"] = list(ing.get("failed") or [])
            report["complete"] = bool(ing.get("complete"))
        except ReaderError as exc:
            ingest_error = str(exc)
            report["error"] = ingest_error
        state = reader.ingest_state(slug)
        watermark = int(state.get("ingested") or 0)
        report["ingested"] = watermark
        report["total"] = int(state.get("total") or 0)
        bible = bibles.load(slug)
        have = bible.chapters_digested if bible else 0
        report["digested_through"] = have
        if watermark <= have:
            if ingest_error and not watermark:
                report["ok"] = False
            return report
        batch: list[tuple[int, str, str]] = []

        def _flush() -> None:
            nonlocal bible, batch
            if not batch:
                return
            if bible is None:
                bible = bibles.build(slug, story.title, batch)
            else:
                bible = bibles.update(slug, batch)
            batch = []

        read_now: list[int] = []
        try:
            for n in range(have + 1, watermark + 1):
                text = reader.chapter_text(slug, n)
                title = text.split("\n", 1)[0].lstrip("# ").strip()[:120]
                batch.append((n, title, text))
                read_now.append(n)
                if len(batch) >= _DIGEST_BATCH:
                    _flush()
            _flush()
        except Exception as exc:  # noqa: BLE001 - partial digestion stands
            record_failure("reading_engine.digest", exc)
            report["error"] = (report["error"] or
                               f"digestion stopped: {exc}")[:200]
            # the batch state matters more than the report here;
            # the on-disk watermark tells the truth either way
            with contextlib.suppress(Exception):
                _flush()
        bible = bibles.load(slug)
        report["digested_through"] = (
            bible.chapters_digested if bible else have)
        report["new_chapters"] = [
            n for n in read_now if n <= report["digested_through"]]
        if ingest_error and not report["new_chapters"]:
            report["ok"] = False
        return report

    # ── telling the owner ────────────────────────────────────────────
    def _announce(self, report: dict[str, Any], *, reason: str,
                  chat_key: str) -> None:
        text = self._announce_text(report, reason=reason)
        if not text:
            return
        if chat_key and self._chat_notify is not None:
            self._chat_notify(chat_key, text)
            return
        from ..agents.notifier import Notifier

        title = str(report.get("title") or report.get("slug") or "story")
        Notifier(self.context).publish(
            "novel", f"📚 {title}", text)

    @staticmethod
    def _announce_text(report: dict[str, Any], *, reason: str) -> str:
        """The ping for a finished run — or "" when the run's contract
        is silence (a catch-up, or a discovery that found her already
        current).  Follow and manual sync runs always report: the
        owner asked, the outcome is the answer."""
        slug = str(report.get("slug") or "")
        title = str(report.get("title") or slug)
        total = int(report.get("total") or 0)
        digested = int(report.get("digested_through") or 0)
        new = [int(n) for n in report.get("new_chapters") or []]
        error = str(report.get("error") or "")
        if reason == _REASON_CATCHUP:
            return ""
        if reason == _REASON_DISCOVERY and not new:
            return ""
        if not report.get("ok") and not new:
            return (f"⚠️ “{title}” is followed, but she couldn't read "
                    f"it yet: {error or 'the source gave nothing'} — "
                    f"/novel sync {slug} retries.")
        if reason == _REASON_DISCOVERY:
            span = (f"chapter {new[0]}" if len(new) == 1 else
                    f"chapters {new[0]}–{new[-1]}")
            return (f"📖 “{title}” — {span} "
                    f"{'is' if len(new) == 1 else 'are'} new, and she's "
                    f"read {'it' if len(new) == 1 else 'them'} "
                    f"(through chapter {digested} of {total or '?'}). "
                    f"/novel read {slug} picks up at your place.")
        if report.get("complete") and digested >= total and total:
            return (f"📚 she's read “{title}” — all {total} chapters "
                    f"are in its library and digested. "
                    f"/novel read {slug} starts at your place; "
                    f"/novel continue {slug} writes from the true ending.")
        return (f"📚 “{title}” — she's read through chapter {digested} "
                f"of {total or '?'} so far"
                + (f" (stopped: {error})" if error else "")
                + f". Reading resumes on the next discovery sweep or "
                  f"/novel sync {slug}.")


def discover_new_chapters(reader: StoryReader,
                          slug: str = "") -> list[dict[str, Any]]:
    """Which followed stories hold chapters their library does not?

    Full enumeration (``chapter_list(limit=0)`` — the interactive
    profile cap that ``check_updates`` reports through must never
    hide a new chapter from discovery: a 500-chapter story's chapter
    401 is exactly the one this exists to find).  Pure discovery:
    fetches chapter LISTS only, never chapter bodies, and reads
    nothing.  Per-story failures land as ``ok: False`` rows, never
    as exceptions."""
    from .sources import SourceError

    targets = [reader._load(slug)] if slug else [
        reader._load(p.parent.name)
        for p in sorted(reader.reader_dir().glob("*/story.json"))]
    out: list[dict[str, Any]] = []
    for story in targets:
        try:
            adapter = reader._adapter(story.url)
            listing = adapter.chapter_list(story.url, limit=0)
        except (SourceError, ReaderError) as exc:
            out.append({"slug": story.slug, "title": story.title,
                        "ok": False, "reason": str(exc)[:160]})
            continue
        cached = reader._cached_numbers(story.slug)
        new = sorted(n for n, _t, _u in listing
                     if n >= 1 and n not in cached)
        out.append({"slug": story.slug, "title": story.title,
                    "ok": True, "cached": len(cached),
                    "live_total": len(listing),
                    "new_chapters": new, "new_count": len(new)})
    return out


# ── one engine per runtime context ───────────────────────────────────

_EXTRAS_KEY = "story_reading_engine"
_fallback: dict[int, StoryReadingEngine] = {}


def reading_engine_for(context: Any) -> StoryReadingEngine | None:
    """The engine bound to ``context``, or None when none was started."""
    extras = getattr(context, "extras", None)
    if isinstance(extras, dict):
        eng = extras.get(_EXTRAS_KEY)
        if isinstance(eng, StoryReadingEngine):
            return eng
    return _fallback.get(id(context))


def ensure_reading_engine(
        context: Any, *,
        chat_notify: Callable[[str, str], None] | None = None,
) -> StoryReadingEngine:
    """Start (once per context) the story reading engine and return it.

    Idempotent: a second call returns the standing engine (a later
    ``chat_notify`` never replaces the first — the runtime wires its
    chat delivery at boot, before any tool path can get here)."""
    existing = reading_engine_for(context)
    if existing is not None:
        if not existing.started:
            existing.start()
        return existing
    engine = StoryReadingEngine(context, chat_notify=chat_notify)
    extras = getattr(context, "extras", None)
    if isinstance(extras, dict):
        extras[_EXTRAS_KEY] = engine
    else:
        _fallback[id(context)] = engine
    engine.start()
    return engine
