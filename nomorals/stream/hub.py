"""Broadcast hub for SSE subscribers (Mercure-style, stdlib-only).

Instead of every subscriber polling the Timeline itself (one SQLite
connection per subscriber per second), one :class:`EventHub` polls once
and fans new events out to per-subscriber bounded queues. This is the
hub pattern from the Mercure protocol, adapted to this codebase's
threading model:

- each event gets a hub-monotonic sequence number (the SSE ``id:``);
- a retained ring buffer backs ``Last-Event-ID`` catch-up on reconnect;
- :meth:`EventHub.inject` lets in-process producers push live events
  without waiting for the next poll (Mercure's "publish to the hub");
- slow subscribers get drop-oldest backpressure with a ``dropped``
  counter instead of wedging the hub.

Threading: one poller thread per hub; ``offer`` runs on the poller,
``get`` runs on the subscriber's connection thread. Locks are
per-subscription; the hub lock is never held while a subscription lock
is held.
"""

from __future__ import annotations

import fnmatch
import queue
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ._poll import (
    PAGE_LIMIT,
    _ts_of,
    classify_one,
    drain_backlog,
)
from .errors import StreamClosed, SubscriberLimitExceeded
from .sse import ServerSentEvent  # noqa: F401  (re-exported for convenience)

__all__ = ["EventHub", "Subscription"]

_log = get_logger(__name__)

#: Sentinel placed on a subscription queue when the hub stops.
_CLOSED: Any = object()


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish telemetry. Best-effort: a broken bus must never break the hub."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


def _normalize_topics(topic: str | None) -> list[str]:
    """Split ``?topic=a.*,b.*`` into glob patterns (``*`` when empty)."""
    if not topic:
        return ["*"]
    patterns = [p.strip() for p in str(topic).split(",")]
    patterns = [p for p in patterns if p]
    return patterns or ["*"]


class Subscription:
    """One subscriber's view of a hub: filter + cursor + bounded queue."""

    def __init__(
        self,
        hub: EventHub,
        topics: list[str],
        cursor: float,
        seen_at_cursor: set[str],
        maxsize: int,
    ) -> None:
        self._hub = hub
        self.topics = list(topics)
        self.id = uuid.uuid4().hex
        self._cursor = cursor
        self._seen = set(seen_at_cursor)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._closed = False
        self.dropped = 0
        self.created_at = time.time()

    # -- matching -----------------------------------------------------
    def matches(self, topic: str | None) -> bool:
        return any(
            fnmatch.fnmatchcase(str(topic or ""), pat)
            for pat in self.topics
        )

    # -- producer side (hub poller thread) -----------------------------
    def offer(self, seq: int, event: dict[str, Any]) -> bool:
        """Route one event to this subscriber. Returns True if queued."""
        with self._lock:
            if self._closed:
                return False
            is_new, self._cursor, self._seen = classify_one(
                event, self._cursor, self._seen)
            if not is_new:
                return False
            item = (seq, event)
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                # Backpressure: drop the oldest, keep the hub moving, and
                # count it so the client can be told to re-sync.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self.dropped += 1
                try:
                    self._queue.put_nowait(item)
                except queue.Full:  # pragma: no cover - vanishingly rare
                    self.dropped += 1
                    return False
            return True

    def absorb(self, event: dict[str, Any]) -> bool:
        """Advance cursor/seen bookkeeping without queueing.

        Used on reconnect: the client already saw this event, so it must
        not be delivered again — but the cursor must move past it so the
        Timeline backfill doesn't re-emit it either.
        """
        with self._lock:
            if self._closed:
                return False
            _, self._cursor, self._seen = classify_one(
                event, self._cursor, self._seen)
            return True

    def close_sentinel(self) -> None:
        """Wake a blocked consumer: the hub is going away."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._queue.put_nowait(_CLOSED)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(_CLOSED)
                except queue.Full:  # pragma: no cover
                    pass

    # -- consumer side (connection thread) -----------------------------
    def get(self, timeout: float | None = None) -> tuple[int, dict[str, Any]]:
        """Next ``(seq, event)``. Raises :class:`StreamClosed` once the hub
        stopped, :exc:`queue.Empty` on timeout."""
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            if self._closed:
                raise StreamClosed("hub stopped") from None
            raise
        if item is _CLOSED:
            raise StreamClosed("hub stopped")
        return item

    def close(self) -> None:
        """Unregister from the hub. Idempotent."""
        self.close_sentinel()
        try:
            self._hub.unsubscribe(self)
        except Exception:  # noqa: BLE001 - best-effort teardown
            _log.debug("subscription close failed", exc_info=True)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def cursor(self) -> float:
        with self._lock:
            return self._cursor


class EventHub:
    """One shared poller fanning Timeline events out to subscribers.

    ``timeline_factory`` is a zero-arg callable returning a Timeline-like
    with ``query(since=..., topic=..., limit=..., until=...)``.
    """

    def __init__(
        self,
        timeline_factory: Callable[[], Any],
        *,
        poll_interval: float = 1.0,
        page_limit: int = PAGE_LIMIT,
        ring_size: int = 10_000,
        default_queue_size: int = 512,
        max_subscribers: int = 1024,
    ) -> None:
        if timeline_factory is None:
            raise ValueError("timeline_factory is required")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if max_subscribers <= 0:
            raise ValueError("max_subscribers must be > 0")
        self.timeline_factory = timeline_factory
        self.poll_interval = poll_interval
        self.page_limit = page_limit
        self.ring_size = ring_size
        self.default_queue_size = default_queue_size
        self.max_subscribers = max_subscribers
        #: Detects hub restarts for clients (seq space resets with the hub).
        self.epoch = uuid.uuid4().hex
        self._seq = 0
        self._ring: deque[tuple[int, dict[str, Any]]] = deque(maxlen=ring_size)
        self._subs: dict[str, Subscription] = {}
        self._poll_cursor = 0.0
        self._poll_seen: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._emitted = 0

    # -- lifecycle ------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("hub already started")
        self._stop.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(
            target=self._poll_loop, name="stream-hub-poller", daemon=True)
        self._thread.start()
        _log.info("stream hub started (poll_interval=%.2fs)", self.poll_interval)

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
        with self._lock:
            subs = list(self._subs.values())
            self._subs.clear()
        for sub in subs:
            sub.close_sentinel()
        _log.info("stream hub stopped (%d subscribers dropped)", len(subs))

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._stop.is_set()

    # -- subscription ---------------------------------------------------
    def subscribe(
        self,
        topic: str | None = None,
        *,
        since: float = 0.0,
        last_event_id: int | None = None,
        queue_size: int | None = None,
    ) -> Subscription:
        """Attach a subscriber.

        ``topic`` is a comma-separated list of fnmatch globs (``*`` =
        everything). ``last_event_id`` (a hub seq, from ``Last-Event-ID``)
        replays the retained ring buffer from that point; otherwise
        ``since`` (epoch) replays ring events with ``ts >= since`` and then
        backfills anything the ring missed from the Timeline. Raises
        :class:`SubscriberLimitExceeded` when full, :class:`StreamClosed`
        when the hub is stopped.
        """
        if self._stop.is_set():
            raise StreamClosed("hub is stopped")
        topics = _normalize_topics(topic)
        with self._lock:
            if len(self._subs) >= self.max_subscribers:
                raise SubscriberLimitExceeded(
                    f"hub already has {len(self._subs)} subscribers "
                    f"(max {self.max_subscribers})")
            sub = Subscription(
                self, topics, since, set(),
                maxsize=queue_size or self.default_queue_size)
            self._subs[sub.id] = sub
        _emit("stream.subscriber_connected", {
            "subscription_id": sub.id,
            "topics": topics,
            "subscribers": self.subscriber_count,
        })
        try:
            self._backfill(sub, since=since, last_event_id=last_event_id)
        except Exception:
            sub.close()
            raise
        return sub

    def _backfill(
        self,
        sub: Subscription,
        *,
        since: float,
        last_event_id: int | None,
    ) -> None:
        """Bring a new subscription up to date: ring replay + Timeline.

        Resume is at-least-once by design (the SSE contract): a
        reconnecting client never misses events, and may see a bounded
        overlap window re-delivered — never an unbounded full-history
        replay, and never a silent gap.
        """
        with self._lock:
            ring = list(self._ring)
        if last_event_id is not None:
            self._backfill_from_id(sub, ring, since, last_event_id)
            return
        # Epoch-based resume: the Timeline is the source of truth, then
        # the ring adds anything the poller hasn't persisted/seen —
        # notably events pushed via inject(), which live only in the ring.
        # The subscription cursor dedups the overlap.
        self._drain_into(sub, floor=since)
        for seq, event in ring:
            if sub.closed:
                return
            if sub.matches(event.get("topic")) and _ts_of(event) >= since:
                sub.offer(seq, event)

    def _backfill_from_id(
        self,
        sub: Subscription,
        ring: list[tuple[int, dict[str, Any]]],
        since: float,
        last_event_id: int,
    ) -> None:
        if not ring:
            # Hub restarted: the seq space is gone. Bounded full backfill
            # rather than silently delivering nothing.
            _log.warning(
                "stream: Last-Event-ID %r with an empty ring (hub restart?)"
                " — bounded full backfill", last_event_id)
            self._drain_into(sub, floor=0.0)
            return
        ring_min, ring_max = ring[0][0], ring[-1][0]
        if last_event_id < ring_min:
            # The ring rotated past the client: fall back to the
            # epoch-based path (bounded, at-least-once).
            _log.warning(
                "stream: Last-Event-ID %r older than retained %r — "
                "falling back to since=%r", last_event_id, ring_min, since)
            self._drain_into(sub, floor=since)
            for seq, event in ring:
                if sub.closed:
                    return
                if (sub.matches(event.get("topic"))
                        and _ts_of(event) >= since):
                    sub.offer(seq, event)
            return
        # Normal case: absorb what the client already saw (cursor
        # bookkeeping only), offer the rest, then catch the at-least-once
        # tail — timeline events at/after the newest absorbed timestamp
        # that never passed through the ring.
        for seq, event in ring:
            if sub.closed:
                return
            if not sub.matches(event.get("topic")):
                continue
            if seq <= last_event_id:
                sub.absorb(event)
            else:
                sub.offer(seq, event)
        self._drain_into(sub, floor=sub.cursor)

    def _drain_into(self, sub: Subscription, floor: float) -> None:
        """Drain Timeline history at/after ``floor`` into one subscriber.

        Backfilled events get hub seqs (so the client can resume from
        them) but are *not* re-retained in the ring — the ring stays
        duplicate-free; per-subscription dedup keeps replay correct.
        """
        drain_backlog(
            lambda ev: self._offer_to_sub(sub, ev, retain=False),
            self.timeline_factory, None, sub.cursor, set(),
            limit=self.page_limit, _floor=floor,
        )

    def _offer_to_sub(
        self,
        sub: Subscription,
        event: dict[str, Any],
        *,
        retain: bool = True,
    ) -> None:
        with self._lock:
            self._seq += 1
            seq = self._seq
            if retain:
                self._ring.append((seq, event))
        if sub.matches(event.get("topic")):
            sub.offer(seq, event)

    def unsubscribe(self, sub: Subscription | str) -> None:
        sub_id = sub.id if isinstance(sub, Subscription) else sub
        with self._lock:
            removed = self._subs.pop(sub_id, None)
        if removed is not None:
            _emit("stream.subscriber_disconnected", {
                "subscription_id": sub_id,
                "subscribers": self.subscriber_count,
            })

    # -- live routing ----------------------------------------------------
    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001 - a bad poll must not kill the hub
                _log.exception("stream hub poll failed")

    def _poll_once(self) -> None:
        with self._lock:
            subs = list(self._subs.values())
        if not subs:
            return  # nobody listening — skip the DB round-trip entirely
        def route(event: dict[str, Any]) -> None:
            self._route_event(event, subs)
        self._poll_cursor, self._poll_seen = drain_backlog(
            route, self.timeline_factory, None,
            self._poll_cursor, self._poll_seen, limit=self.page_limit)

    def _route_event(
        self,
        event: dict[str, Any],
        subs: list[Subscription],
    ) -> None:
        with self._lock:
            self._seq += 1
            seq = self._seq
            self._ring.append((seq, event))
            self._emitted += 1
        topic = event.get("topic")
        for sub in subs:
            if sub.matches(topic):
                sub.offer(seq, event)

    # -- ad-hoc publish ---------------------------------------------------
    def inject(self, topic: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Push a live event to matching subscribers immediately.

        The in-process equivalent of Mercure's "publish to the hub": no
        Timeline write, no waiting for the next poll. Returns the event.
        """
        event: dict[str, Any] = {
            "event_id": uuid.uuid4().hex,
            "ts": time.time(),
            "topic": topic,
        }
        if payload:
            event.update(payload)
        with self._lock:
            subs = list(self._subs.values())
        self._route_event(event, subs)
        return event

    # -- introspection ------------------------------------------------------
    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            dropped = sum(s.dropped for s in self._subs.values())
            return {
                "running": self.running,
                "subscribers": len(self._subs),
                "max_subscribers": self.max_subscribers,
                "events_emitted": self._emitted,
                "events_dropped": dropped,
                "ring_size": len(self._ring),
                "ring_capacity": self.ring_size,
                "poll_interval": self.poll_interval,
                "uptime_s": (
                    time.time() - self._started_at
                    if self._started_at else 0.0),
                "epoch": self.epoch,
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"EventHub(subscribers={self.subscriber_count}, "
            f"emitted={self._emitted}, running={self.running})")
