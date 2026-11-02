"""Pass tail-118 pins — nomorals/stream/hub.py Subscription (3 exclusions).

Sites pinned (all E103 queue-race swallows):
- 115: ``offer()`` backpressure — queue reports Full, the drop-oldest
  drain attempt races another consumer and raises ``queue.Empty`` ->
  swallowed; the backpressure accounting still runs (``dropped`` is
  incremented, the retry put decides the return value honestly).
- 150: ``close_sentinel()`` — same Empty race on the drain attempt
  while waking a blocked consumer.
- 154: ``close_sentinel()`` — the retry put of the _CLOSED sentinel
  raises Full (queue refilled in the race window) -> the sentinel is
  lost, BUT the loss is fenced: ``_closed`` is already set, and the
  sole consumer (stream/server.py) polls ``get(timeout=heartbeat)``,
  whose Empty branch checks ``_closed`` and raises StreamClosed.
  Pinned in its true shape: queued events drain first, then the next
  timed get raises StreamClosed.
"""

from __future__ import annotations

import queue

import pytest

from nomorals.stream.errors import StreamClosed
from nomorals.stream.hub import Subscription


def _sub(maxsize=1):
    return Subscription(object(), ["*"], 0.0, set(), maxsize=maxsize)


def _empty_boom(self):
    raise queue.Empty()


def test_offer_healthy_queues():
    sub = _sub()
    assert sub.offer(1, {"id": "e1", "ts": 10.0}) is True
    assert sub.get(timeout=0.1)[1]["id"] == "e1"


def test_115_drain_race_dropped_accounting_still_runs(monkeypatch):
    sub = _sub(maxsize=1)
    assert sub.offer(1, {"id": "e1", "ts": 10.0}) is True
    monkeypatch.setattr(queue.Queue, "get_nowait", _empty_boom)
    # Full -> drain raises Empty (115, swallowed) -> dropped += 1 ->
    # retry put still Full -> dropped += 1, honest False.
    assert sub.offer(2, {"id": "e2", "ts": 11.0}) is False
    assert sub.dropped == 2


def test_115_healthy_backpressure_drops_oldest():
    sub = _sub(maxsize=1)
    assert sub.offer(1, {"id": "e1", "ts": 10.0}) is True
    assert sub.offer(2, {"id": "e2", "ts": 11.0}) is True
    assert sub.dropped == 1
    assert sub.get(timeout=0.1)[1]["id"] == "e2"


def test_150_154_sentinel_race_no_raise_and_timeout_fence(monkeypatch):
    sub = _sub(maxsize=1)
    assert sub.offer(1, {"id": "e1", "ts": 10.0}) is True
    monkeypatch.setattr(queue.Queue, "get_nowait", _empty_boom)
    sub.close_sentinel()  # 150 swallowed; retry put Full -> 154 swallowed
    assert sub.closed
    monkeypatch.undo()
    # The fence: the queued event drains, then the timed get sees
    # _closed on Empty and raises StreamClosed despite the lost sentinel.
    assert sub.get(timeout=0.05)[1]["id"] == "e1"
    with pytest.raises(StreamClosed):
        sub.get(timeout=0.05)


def test_150_healthy_sentinel_delivered():
    sub = _sub(maxsize=2)
    sub.close_sentinel()
    with pytest.raises(StreamClosed):
        sub.get(timeout=0.1)
