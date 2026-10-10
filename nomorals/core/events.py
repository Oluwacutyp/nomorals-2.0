"""Thread-safe event bus.

The nervous system of the framework. Every interesting occurrence — a tool call,
a task state change, a memory consolidation, a model hot-swap, a social post — is
published here. Consumers subscribe by topic with glob patterns.

Two delivery modes:

* **sync** — handler runs on the publisher's thread, before ``publish`` returns.
  Use for invariants that must hold before the caller proceeds.
* **async** — handler runs on the bus's dispatcher thread. Use for everything else.

Reliability contract:

* **Per-handler FIFO ordering.** The single dispatcher thread delivers async
  events to each handler in publish order; every event carries a monotonic
  ``seq`` so consumers can verify the order. (Ordering is per handler, never
  global across unrelated topics — anyone who needs global order has a
  throughput problem, not a queue problem.)
* **At-least-once + idempotency.** Enable ``dedupe_window`` and every event's
  ``event_id`` becomes an idempotency key: redeliveries inside the window are
  suppressed instead of double-delivered.
* **No silent loss.** A handler that keeps failing is retried with backoff
  (``retries``/``retry_backoff`` on :meth:`subscribe`) and then parked on the
  **dead-letter queue** — inspect it with :meth:`dlq`, redeliver with
  :meth:`replay_dlq`, or drop it with :meth:`purge_dlq`. Alert on DLQ
  non-empty, not on "queue failing".
* **Handler failures never propagate** to the publisher; they are captured on
  the event, counted, and surfaced through the ``eventbus.error`` topic.
* **Backpressure is a policy** (``on_full``): ``drop`` (count + warn),
  ``block`` (wait up to ``publish_timeout``), or ``raise``.
* **Journal / replay.** With ``journal_path`` set, every published event is
  appended as JSONL; :meth:`journal_replay` reads them back and :meth:`replay`
  re-dispatches them (catch-up after a restart).
"""

from __future__ import annotations

import fnmatch
import heapq
import inspect
import json
import logging
import os
import queue
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import NoMoralsError

__all__ = [
    "BusOverloaded",
    "DeadLetter",
    "Event",
    "EventBus",
    "Subscription",
    "global_bus",
]

_log = logging.getLogger(__name__)

Handler = Callable[["Event"], Any]


class BusOverloaded(NoMoralsError):
    """The dispatch queue is full and ``on_full='raise'`` was configured."""

    code = "eventbus.overloaded"
    retryable = True


@dataclass
class Event:
    """A single occurrence travelling through the bus."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: "")
    errors: list[str] = field(default_factory=list)
    seq: int = 0  #: monotonic sequence assigned by the bus at publish time

    def __post_init__(self) -> None:
        if not self.event_id:
            from .ids import new_short_id

            self.event_id = new_short_id("evt_")

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "source": self.source,
            "ts": self.ts,
            "seq": self.seq,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """Rebuild an event, e.g. from the journal. Never raises."""
        try:
            return cls(
                topic=str(data.get("topic", "")),
                data=dict(data.get("data") or {}),
                source=str(data.get("source", "")),
                ts=float(data.get("ts") or time.time()),
                event_id=str(data.get("event_id", "")),
                seq=int(data.get("seq") or 0),
            )
        except Exception:  # noqa: BLE001 - journal must never break replay
            return cls(topic="eventbus.journal.corrupt",
                       data={"raw": str(data)[:200]})


@dataclass
class DeadLetter:
    """An event a handler could not process, even after retries."""

    event: Event
    sub_id: str
    handler_name: str
    error: str
    attempts: int
    first_failed_at: float = field(default_factory=time.time)
    last_failed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event.event_id,
            "topic": self.event.topic,
            "seq": self.event.seq,
            "sub_id": self.sub_id,
            "handler": self.handler_name,
            "error": self.error,
            "attempts": self.attempts,
            "first_failed_at": self.first_failed_at,
            "last_failed_at": self.last_failed_at,
            "data": self.event.data,
        }


@dataclass
class Subscription:
    """A registered handler."""

    pattern: str
    handler: Handler
    sync: bool = False
    priority: int = 0
    once: bool = False
    sub_id: str = ""
    retries: int = 0  #: redeliveries after a handler failure before DLQ
    retry_backoff: float = 0.5  #: base backoff seconds between retries

    def matches(self, topic: str) -> bool:
        if self.pattern == "*" or self.pattern == topic:
            return True
        if self.pattern.endswith(".*"):
            prefix = self.pattern[:-2]
            return topic == prefix or topic.startswith(prefix + ".")
        return fnmatch.fnmatchcase(topic, self.pattern)


class EventBus:
    """Publish/subscribe hub.

        >>> bus = EventBus()
        >>> seen = []
        >>> bus.subscribe("task.*", lambda e: seen.append(e.topic), sync=True)
        'sub_...'
        >>> bus.publish(Event(topic="task.started", data={"id": 1}))
        >>> seen
        ['task.started']
    """

    def __init__(
        self,
        *,
        max_queue: int = 10_000,
        history: int = 256,
        dispatcher_name: str = "nm-eventbus",
        dead_letter_limit: int = 1_000,
        dedupe_window: int = 0,
        on_full: str = "drop",
        publish_timeout: float = 5.0,
        journal_path: str | None = None,
        journal_max_bytes: int = 50_000_000,
    ) -> None:
        if on_full not in ("drop", "block", "raise"):
            raise ValueError("on_full must be 'drop', 'block' or 'raise'")
        self._subs: list[Subscription] = []
        self._lock = threading.RLock()
        self._queue: queue.Queue[tuple[Subscription, Event] | None] = queue.Queue(
            maxsize=max_queue)
        self._history: list[Event] = []
        self._history_limit = history
        self._counts: dict[str, int] = defaultdict(int)
        self._dropped = 0
        self._dedup_dropped = 0
        self._handler_errors = 0
        self._dlq: list[DeadLetter] = []
        self._dlq_evicted = 0
        self._dead_letter_limit = dead_letter_limit
        self._dedupe_window = dedupe_window
        self._seen_ids: deque[str] = deque()
        self._seen_set: set[str] = set()
        self._on_full = on_full
        self._publish_timeout = publish_timeout
        self._seq = 0
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._dispatcher_name = dispatcher_name
        self._published = 0
        # retry heap: (due_at, order, sub_id, event, attempts_done)
        self._retries: list[tuple[float, int, str, Event, int]] = []
        self._retry_order = 0
        # journal
        self._journal_path = journal_path
        self._journal_max_bytes = journal_max_bytes
        self._journal: Any = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> "EventBus":
        """Start the async dispatcher thread (idempotent)."""
        with self._lock:
            if self._dispatcher is not None and self._dispatcher.is_alive():
                return self
            self._stop.clear()
            self._dispatcher = threading.Thread(
                target=self._dispatch_loop, name=self._dispatcher_name, daemon=True
            )
            self._dispatcher.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the dispatcher, draining pending events first."""
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - queue drained below anyway  # noqa: E103
            pass
        if self._dispatcher is not None:
            self._dispatcher.join(timeout=timeout)
            self._dispatcher = None
        self._journal_close()

    def __enter__(self) -> "EventBus":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- subscription --------------------------------------------------------
    def subscribe(
        self,
        pattern: str,
        handler: Handler,
        *,
        sync: bool = False,
        priority: int = 0,
        once: bool = False,
        retries: int = 0,
        retry_backoff: float = 0.5,
    ) -> str:
        """Register ``handler`` for topics matching ``pattern``.

        ``retries`` redelivers the event to this handler (with exponential
        ``retry_backoff``) when it raises, before the event is parked on the
        dead-letter queue. Returns a subscription id usable with
        :meth:`unsubscribe`.
        """
        from .ids import new_short_id

        sub = Subscription(
            pattern=pattern,
            handler=handler,
            sync=sync,
            priority=priority,
            once=once,
            sub_id=new_short_id("sub_"),
            retries=max(0, retries),
            retry_backoff=max(0.0, retry_backoff),
        )
        with self._lock:
            self._subs.append(sub)
            # Higher priority first; stable for equal priorities.
            self._subs.sort(key=lambda s: -s.priority)
        return sub.sub_id

    def unsubscribe(self, sub_id: str) -> bool:
        with self._lock:
            before = len(self._subs)
            self._subs = [s for s in self._subs if s.sub_id != sub_id]
            return len(self._subs) < before

    def clear(self) -> None:
        with self._lock:
            self._subs.clear()

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

    def subscriptions(self) -> list[dict[str, Any]]:
        """Introspection: every subscription with its reliability settings."""
        with self._lock:
            return [
                {
                    "sub_id": s.sub_id,
                    "pattern": s.pattern,
                    "handler": getattr(s.handler, "__name__", repr(s.handler)),
                    "sync": s.sync,
                    "priority": s.priority,
                    "once": s.once,
                    "retries": s.retries,
                    "retry_backoff": s.retry_backoff,
                }
                for s in self._subs
            ]

    # -- publishing ----------------------------------------------------------
    def _coerce(self, event: Event | str, data: dict[str, Any] | None,
                kwargs: dict[str, Any]) -> Event:
        if isinstance(event, str):
            payload = dict(data or {})
            payload.update(kwargs)
            return Event(topic=event, data=payload)
        if data or kwargs:
            merged = dict(event.data)
            merged.update(data or {})
            merged.update(kwargs)
            return Event(
                topic=event.topic,
                data=merged,
                source=event.source,
                ts=event.ts,
                event_id=event.event_id,
                seq=event.seq,
            )
        return event

    def _note_seen(self, event_id: str) -> None:
        if not self._dedupe_window:
            return
        self._seen_ids.append(event_id)
        self._seen_set.add(event_id)
        while len(self._seen_ids) > self._dedupe_window:
            old = self._seen_ids.popleft()
            self._seen_set.discard(old)

    def publish(self, event: Event | str, data: dict[str, Any] | None = None,
                **kwargs: Any) -> Event:
        """Publish an event. Accepts an :class:`Event` or ``(topic, data)``.

        Assigns a monotonic ``seq``. With ``dedupe_window`` set, an event
        whose ``event_id`` was seen recently is suppressed (at-least-once +
        idempotency) and returned undelivered.
        """
        event = self._coerce(event, data, kwargs)
        with self._lock:
            self._published += 1
            self._seq += 1
            event.seq = self._seq
            if self._dedupe_window and event.event_id in self._seen_set:
                self._dedup_dropped += 1
                _log.debug("eventbus dedup suppressed %s (%s)",
                           event.event_id, event.topic)
                return event
            self._note_seen(event.event_id)
            self._counts[event.topic] += 1
            self._history.append(event)
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
            targets = [s for s in self._subs if s.matches(event.topic)]
        self._journal_write(event)
        self._dispatch(event, targets)
        return event

    def emit(self, topic: str, **data: Any) -> Event:
        """Shorthand for :meth:`publish`."""
        return self.publish(topic, data)

    def replay(self, events: Iterable[Event]) -> int:
        """Re-dispatch events (e.g. from :meth:`journal_replay`).

        Assigns fresh sequence numbers and bypasses dedup — replay is an
        explicit redelivery, not a duplicate. Does not re-journal.
        """
        count = 0
        for event in events:
            with self._lock:
                self._published += 1
                self._seq += 1
                event.seq = self._seq
                self._counts[event.topic] += 1
                self._history.append(event)
                if len(self._history) > self._history_limit:
                    del self._history[: len(self._history) - self._history_limit]
                targets = [s for s in self._subs if s.matches(event.topic)]
            self._dispatch(event, targets)
            count += 1
        return count

    def _dispatch(self, event: Event, targets: list[Subscription]) -> None:
        consumed_once: list[Subscription] = []
        async_targets: list[Subscription] = []
        for sub in targets:
            if sub.sync:
                self._deliver_sync(sub, event)
                if sub.once:
                    consumed_once.append(sub)
            else:
                async_targets.append(sub)

        if async_targets:
            if self._dispatcher is None or not self._dispatcher.is_alive():
                # Bus not started: deliver inline rather than silently dropping.
                for sub in async_targets:
                    self._deliver_sync(sub, event)
                    if sub.once:
                        consumed_once.append(sub)
            else:
                for sub in async_targets:
                    self._enqueue(sub, event)

        if consumed_once:
            with self._lock:
                ids = {s.sub_id for s in consumed_once}
                self._subs = [s for s in self._subs if s.sub_id not in ids]

    def _enqueue(self, sub: Subscription, event: Event) -> None:
        if self._on_full == "block":
            try:
                self._queue.put((sub, event), timeout=self._publish_timeout)
                return
            except queue.Full:
                pass  # fall through to drop accounting
        elif self._on_full == "raise":
            try:
                self._queue.put_nowait((sub, event))
                return
            except queue.Full:
                self._dropped += 1
                raise BusOverloaded(
                    f"eventbus queue full; dropped handler for {event.topic}",
                    details={"topic": event.topic,
                             "max_queue": self._queue.maxsize},
                )
        try:
            self._queue.put_nowait((sub, event))
        except queue.Full:
            self._dropped += 1
            _log.warning("eventbus queue full; dropped handler for %s", event.topic)

    # -- delivery ------------------------------------------------------------
    def _handler_name(self, sub: Subscription) -> str:
        return getattr(sub.handler, "__name__", repr(sub.handler))

    def _invoke(self, sub: Subscription, event: Event) -> str | None:
        """Run one handler. Returns the error string on failure, else None.

        Failures never propagate: they are captured on the event, counted,
        and surfaced on the ``eventbus.error`` topic (which itself never
        re-surfaces, so error storms cannot recurse).
        """
        try:
            result = sub.handler(event)
            if inspect.isawaitable(result):  # pragma: no cover - requires async handler
                import asyncio

                asyncio.run(result)
            return None
        except Exception as exc:  # noqa: BLE001 - never let a handler kill the bus
            detail = f"{type(exc).__name__}: {exc}"
            if len(event.errors) < 10:
                event.errors.append(detail)
            with self._lock:
                self._handler_errors += 1
            _log.error(
                "event handler %r failed for topic %s: %s\n%s",
                self._handler_name(sub),
                event.topic,
                detail,
                traceback.format_exc(limit=4),
            )
            if event.topic != "eventbus.error":
                try:
                    self.publish(Event(
                        topic="eventbus.error",
                        source="eventbus",
                        data={
                            "original_topic": event.topic,
                            "event_id": event.event_id,
                            "seq": event.seq,
                            "handler": self._handler_name(sub),
                            "sub_id": sub.sub_id,
                            "error": detail,
                        },
                    ))
                except Exception:  # noqa: BLE001 - surfacing must not break delivery
                    _log.debug("failed to publish eventbus.error", exc_info=True)
            return detail

    def _dead_letter(self, sub: Subscription, event: Event,
                     error: str, attempts: int) -> None:
        now = time.time()
        with self._lock:
            self._dlq.append(DeadLetter(
                event=event, sub_id=sub.sub_id,
                handler_name=self._handler_name(sub),
                error=error, attempts=attempts,
                first_failed_at=now, last_failed_at=now,
            ))
            if len(self._dlq) > self._dead_letter_limit:
                over = len(self._dlq) - self._dead_letter_limit
                del self._dlq[:over]
                self._dlq_evicted += over
        _log.warning("eventbus: %s parked on DLQ after %d attempts (%s)",
                     event.topic, attempts, error)

    def _deliver_sync(self, sub: Subscription, event: Event) -> bool:
        """Deliver on the caller's thread, with retries. Returns success."""
        attempts = 0
        while True:
            error = self._invoke(sub, event)
            if error is None:
                return True
            attempts += 1
            if attempts > sub.retries:
                self._dead_letter(sub, event, error, attempts)
                return False
            time.sleep(sub.retry_backoff * (2.0 ** (attempts - 1)))

    def _deliver_async(self, sub: Subscription, event: Event,
                       attempts_done: int = 0, final: bool = False) -> None:
        """Deliver on the dispatcher thread; schedule retries on the heap."""
        error = self._invoke(sub, event)
        if error is None:
            if sub.once:
                with self._lock:
                    self._subs = [s for s in self._subs if s.sub_id != sub.sub_id]
            return
        attempts = attempts_done + 1
        if final or attempts > sub.retries:
            self._dead_letter(sub, event, error, attempts)
            return
        due = time.monotonic() + sub.retry_backoff * (2.0 ** (attempts - 1))
        with self._lock:
            self._retry_order += 1
            heapq.heappush(self._retries,
                           (due, self._retry_order, sub.sub_id, event, attempts))

    def _process_due_retries(self, final: bool = False) -> None:
        """Pop due retries off the heap and redeliver them."""
        while True:
            with self._lock:
                if not self._retries:
                    return
                due, _, sub_id, event, attempts = self._retries[0]
                if not final and due > time.monotonic():
                    return
                heapq.heappop(self._retries)
                sub = next((s for s in self._subs if s.sub_id == sub_id), None)
            if sub is None:
                _log.debug("eventbus: retry dropped, subscription %s gone", sub_id)
                continue
            self._deliver_async(sub, event, attempts_done=attempts, final=final)

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            self._process_due_retries()
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            sub, event = item
            self._deliver_async(sub, event)
        # Drain whatever is left so shutdown does not lose events. Pending
        # retries get one final delivery attempt (no more waiting).
        self._process_due_retries(final=True)
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            sub, event = item
            self._deliver_async(sub, event, final=True)

    # -- dead-letter queue ---------------------------------------------------
    def dlq(self) -> list[dict[str, Any]]:
        """Inspect the dead-letter queue (oldest first)."""
        with self._lock:
            return [d.to_dict() for d in self._dlq]

    def purge_dlq(self) -> int:
        """Drop all dead letters. Returns the number removed."""
        with self._lock:
            n = len(self._dlq)
            self._dlq.clear()
            return n

    def replay_dlq(self, sub_id: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """Redeliver dead letters.

        A letter is first offered to its original subscription; when that is
        gone (e.g. the handler was redeployed and re-subscribed under a new
        id), it falls back to the first *currently* subscribed handler whose
        pattern matches the event's topic, and the letter is re-keyed to it.
        A letter is removed when its handler now succeeds; a repeated failure
        updates the letter in place. Letters with no matching subscriber at
        all are skipped (left on the DLQ). Never raises.
        """
        replayed = succeeded = failed = skipped = redirected = 0
        try:
            with self._lock:
                letters = [d for d in self._dlq
                           if sub_id is None or d.sub_id == sub_id]
                if limit is not None:
                    letters = letters[:limit]
            for letter in letters:
                replayed += 1
                try:
                    with self._lock:
                        sub = next((s for s in self._subs
                                    if s.sub_id == letter.sub_id), None)
                        if sub is None:
                            sub = next((s for s in self._subs
                                        if s.matches(letter.event.topic)), None)
                            if sub is not None:
                                letter.sub_id = sub.sub_id
                                letter.handler_name = self._handler_name(sub)
                                redirected += 1
                    if sub is None:
                        skipped += 1
                        continue
                    error = self._invoke(sub, letter.event)
                    with self._lock:
                        if error is None:
                            if letter in self._dlq:
                                self._dlq.remove(letter)
                            succeeded += 1
                        else:
                            letter.attempts += 1
                            letter.error = error
                            letter.last_failed_at = time.time()
                            failed += 1
                except Exception as exc:  # noqa: BLE001 - one bad letter never kills replay
                    _log.debug("eventbus: DLQ replay failed: %s", exc)
                    failed += 1
        except Exception as exc:  # noqa: BLE001 - replay_dlq never raises
            _log.debug("eventbus: replay_dlq failed: %s", exc)
        return {"replayed": replayed, "succeeded": succeeded,
                "failed": failed, "skipped": skipped,
                "redirected": redirected}

    # -- journal / replay ----------------------------------------------------
    def _journal_write(self, event: Event) -> None:
        if not self._journal_path:
            return
        try:
            line = json.dumps(event.to_dict(), default=str) + "\n"
            data = line.encode("utf-8")
            if self._journal is None:
                self._journal = open(self._journal_path, "ab")
            if self._journal_max_bytes:
                self._journal.flush()
                size = os.fstat(self._journal.fileno()).st_size
                if size + len(data) > self._journal_max_bytes:
                    self._journal.close()
                    rotated = self._journal_path + ".1"
                    try:
                        os.replace(self._journal_path, rotated)
                    except OSError:
                        pass
                    self._journal = open(self._journal_path, "ab")
            self._journal.write(data)
            self._journal.flush()
        except Exception:  # noqa: BLE001 - the journal never breaks publish
            _log.debug("eventbus: journal write failed", exc_info=True)

    def _journal_close(self) -> None:
        journal, self._journal = self._journal, None
        if journal is not None:
            try:
                journal.close()
            except Exception:  # noqa: BLE001
                pass

    def journal_replay(self, since_seq: int = 0, topic: str | None = None,
                       limit: int = 1000) -> list[Event]:
        """Read journaled events back (oldest first). Never raises."""
        events: list[Event] = []
        if not self._journal_path:
            return events
        paths = [self._journal_path + ".1", self._journal_path]
        try:
            for path in paths:
                if not os.path.exists(path):
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError:
                            continue
                        if int(data.get("seq") or 0) <= since_seq:
                            continue
                        if topic and not fnmatch.fnmatchcase(
                                str(data.get("topic", "")), topic):
                            continue
                        events.append(Event.from_dict(data))
                        if len(events) >= limit:
                            return events
        except Exception:  # noqa: BLE001 - replay never raises
            _log.debug("eventbus: journal replay failed", exc_info=True)
        return events

    # -- introspection -------------------------------------------------------
    def history(self, limit: int = 50, topic: str | None = None) -> list[Event]:
        with self._lock:
            items = list(self._history)
        if topic:
            items = [e for e in items if fnmatch.fnmatchcase(e.topic, topic)]
        return items[-limit:]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "published": self._published,
                "seq": self._seq,
                "subscribers": len(self._subs),
                "queued": self._queue.qsize(),
                "pending_retries": len(self._retries),
                "dropped": self._dropped,
                "dedup_dropped": self._dedup_dropped,
                "handler_errors": self._handler_errors,
                "dlq_depth": len(self._dlq),
                "dlq_evicted": self._dlq_evicted,
                "topics": dict(sorted(self._counts.items(), key=lambda kv: -kv[1])[:25]),
                "dispatcher_alive": bool(self._dispatcher and self._dispatcher.is_alive()),
            }

    def wait_for(
        self, topic: str, timeout: float = 5.0, predicate: Callable[[Event], bool] | None = None
    ) -> Event | None:
        """Block until an event matching ``topic`` (and ``predicate``) is seen.

        Handy in tests and in orchestrator joins.
        """
        box: dict[str, Event] = {}
        ready = threading.Event()

        def handler(event: Event) -> None:
            if predicate is None or predicate(event):
                box["event"] = event
                ready.set()

        sub_id = self.subscribe(topic, handler, sync=True)
        try:
            return box["event"] if ready.wait(timeout) else None
        finally:
            self.unsubscribe(sub_id)

    def topics(self) -> Iterable[str]:
        with self._lock:
            return sorted(self._counts)


#: Process-wide bus. Started lazily by the system context.
global_bus = EventBus()
