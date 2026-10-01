"""Thread-safe event bus.

The nervous system of the framework. Every interesting occurrence — a tool call,
a task state change, a memory consolidation, a model hot-swap, a social post — is
published here. Consumers subscribe by topic with glob patterns.

Two delivery modes:

* **sync** — handler runs on the publisher's thread, before ``publish`` returns.
  Use for invariants that must hold before the caller proceeds.
* **async** — handler runs on the bus's dispatcher thread. Use for everything else.

Delivery failures never propagate to the publisher; they are captured on the event
and surfaced through the ``eventbus.error`` topic and the logging hook.
"""

from __future__ import annotations

import fnmatch
import inspect
import logging
import queue
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

__all__ = ["Event", "EventBus", "Subscription", "global_bus"]

_log = logging.getLogger(__name__)

Handler = Callable[["Event"], Any]


@dataclass
class Event:
    """A single occurrence travelling through the bus."""

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    ts: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: "")
    errors: list[str] = field(default_factory=list)

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
            "data": self.data,
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
    ) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.RLock()
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=max_queue)
        self._history: list[Event] = []
        self._history_limit = history
        self._counts: dict[str, int] = defaultdict(int)
        self._dropped = 0
        self._stop = threading.Event()
        self._dispatcher: threading.Thread | None = None
        self._dispatcher_name = dispatcher_name
        self._published = 0

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
    ) -> str:
        """Register ``handler`` for topics matching ``pattern``.

        Returns a subscription id usable with :meth:`unsubscribe`.
        """
        from .ids import new_short_id

        sub = Subscription(
            pattern=pattern,
            handler=handler,
            sync=sync,
            priority=priority,
            once=once,
            sub_id=new_short_id("sub_"),
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

    # -- publishing ----------------------------------------------------------
    def publish(self, event: Event | str, data: dict[str, Any] | None = None, **kwargs: Any) -> Event:
        """Publish an event. Accepts an :class:`Event` or ``(topic, data)``."""
        if isinstance(event, str):
            payload = dict(data or {})
            payload.update(kwargs)
            event = Event(topic=event, data=payload)
        elif data or kwargs:
            merged = dict(event.data)
            merged.update(data or {})
            merged.update(kwargs)
            event = Event(
                topic=event.topic,
                data=merged,
                source=event.source,
                ts=event.ts,
                event_id=event.event_id,
            )

        with self._lock:
            self._published += 1
            self._counts[event.topic] += 1
            self._history.append(event)
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
            targets = [s for s in self._subs if s.matches(event.topic)]

        consumed_once: list[Subscription] = []
        async_targets: list[Subscription] = []
        for sub in targets:
            if sub.sync:
                self._invoke(sub, event)
                if sub.once:
                    consumed_once.append(sub)
            else:
                async_targets.append(sub)

        if async_targets:
            if self._dispatcher is None or not self._dispatcher.is_alive():
                # Bus not started: deliver inline rather than silently dropping.
                for sub in async_targets:
                    self._invoke(sub, event)
                    if sub.once:
                        consumed_once.append(sub)
            else:
                for sub in async_targets:
                    try:
                        self._queue.put_nowait((sub, event))
                    except queue.Full:
                        self._dropped += 1
                        _log.warning("eventbus queue full; dropped handler for %s", event.topic)

        if consumed_once:
            with self._lock:
                ids = {s.sub_id for s in consumed_once}
                self._subs = [s for s in self._subs if s.sub_id not in ids]
        return event

    def emit(self, topic: str, **data: Any) -> Event:
        """Shorthand for :meth:`publish`."""
        return self.publish(topic, data)

    # -- internals -----------------------------------------------------------
    def _invoke(self, sub: Subscription, event: Event) -> None:
        try:
            result = sub.handler(event)
            if inspect.isawaitable(result):  # pragma: no cover - requires async handler
                import asyncio

                asyncio.run(result)
        except Exception as exc:  # noqa: BLE001 - never let a handler kill the bus
            detail = f"{type(exc).__name__}: {exc}"
            event.errors.append(detail)
            _log.error(
                "event handler %r failed for topic %s: %s\n%s",
                getattr(sub.handler, "__name__", repr(sub.handler)),
                event.topic,
                detail,
                traceback.format_exc(limit=4),
            )

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            sub, event = item
            self._invoke(sub, event)
            if sub.once:
                with self._lock:
                    self._subs = [s for s in self._subs if s.sub_id != sub.sub_id]
        # Drain whatever is left so shutdown does not lose events.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            sub, event = item
            self._invoke(sub, event)

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
                "subscribers": len(self._subs),
                "queued": self._queue.qsize(),
                "dropped": self._dropped,
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
