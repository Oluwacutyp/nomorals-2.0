"""Tail-71 pins: nomorals/core/events.py silent-swallow sites.

Adjudications (see ~/workspace/nm-tail71-verify/SURVEY.md):
- 119 EXCLUSION — Event.from_dict is "Never raises" by contract and
  the failure is encoded in the return value itself: a sentinel
  event on the unique topic "eventbus.journal.corrupt" carrying the
  raw payload.  No consumer can mistake it for a real event.
- 181 FLAGGED (stays exclusion; this pin asserts CURRENT behavior)
  — a raising subscription predicate filters out (False), per the
  site's documented design; the tension (failure invisible on every
  bus surface, permanent for the subscription's life) and a
  within-contract _log.debug sketch are recorded in SURVEY.md for
  parent adjudication.
- 261 EXCLUSION — stop()'s wake-up sentinel put is best-effort; the
  dispatcher polls the stop flag and stop() joins/closes anyway.
- 485 EXCLUSION — the block-policy Full pass falls through to the
  drop accounting, which is where the failure IS recorded
  (_dropped counter + warning log).
- 726 EXCLUSION — journal rotation is housekeeping; on failure the
  write still lands in the un-rotated file, no event is lost.
- 739 EXCLUSION — journal close at teardown is best-effort; the
  handle is detached before the attempt.

Each pin proves the guarded operation was ATTEMPTED (spy) and, for
exclusions, that the failure is NOT routed into the bus's handler-
failure machinery (no handler_errors, no DLQ entries, no warning
logs) where the bus provides such a surface.
"""

from __future__ import annotations

import json
import logging
import os
import queue
from pathlib import Path

import nomorals.core.events as ev


# ── 119: Event.from_dict corrupt sentinel ──────────────────────────────────

def test_119_from_dict_failure_returns_corrupt_sentinel(caplog):
    with caplog.at_level(logging.WARNING, logger="nomorals.core.events"):
        out = ev.Event.from_dict(
            {"topic": "a.b", "data": {"x": 1}, "ts": "not-a-float"})
    assert out.topic == "eventbus.journal.corrupt"
    assert "raw" in out.data and "a.b" in out.data["raw"]
    # Control: a well-formed dict rebuilds the real event.
    ok = ev.Event.from_dict(
        {"topic": "a.b", "data": {"x": 1}, "ts": 123.0, "seq": 7})
    assert ok.topic == "a.b" and ok.data == {"x": 1} and ok.seq == 7
    assert not caplog.records  # the sentinel is the record; nothing logged


# ── 181: Subscription.accepts broken predicate (FLAGGED) ───────────────────

def test_181_accepts_raising_predicate_filters_out(caplog):
    calls = []

    def pred(event):
        calls.append(event.topic)
        raise RuntimeError("broken predicate")

    sub = ev.Subscription(pattern="a.*", handler=lambda e: None,
                          predicate=pred)
    with caplog.at_level(logging.WARNING, logger="nomorals.core.events"):
        assert sub.accepts(ev.Event(topic="a.b")) is False
    assert calls == ["a.b"]  # the predicate was attempted
    # Controls: a truthy predicate accepts; the topic gate runs
    # first, so a non-matching topic never touches the predicate.
    ok = ev.Subscription(pattern="a.*", handler=lambda e: None,
                         predicate=lambda e: True)
    assert ok.accepts(ev.Event(topic="a.b")) is True
    seen = []
    gated = ev.Subscription(pattern="a.*", handler=lambda e: None,
                            predicate=lambda e: seen.append(1) or True)
    assert gated.accepts(ev.Event(topic="zzz")) is False
    assert seen == []
    # Consumer-level (current behavior): dispatch drops the event
    # for this subscription WITHOUT routing the failure into the
    # handler-failure machinery — no error count, no DLQ, no log.
    bus = ev.EventBus()
    delivered = []
    bus.subscribe("a.*", lambda e: delivered.append(e), sync=True,
                  predicate=pred)
    bus.publish(ev.Event(topic="a.b"))
    assert delivered == []
    stats = bus.stats()
    assert stats["handler_errors"] == 0 and stats["dlq_depth"] == 0
    assert not caplog.records


# ── 261: EventBus.stop wake-up sentinel on a full queue ────────────────────

def test_261_stop_full_queue_still_stops():
    bus = ev.EventBus(max_queue=1)
    calls = []
    orig = bus._queue.put_nowait

    def raiser(item):
        calls.append(item)
        raise queue.Full

    bus._queue.put_nowait = raiser
    try:
        bus.stop(timeout=0.1)  # must not raise
    finally:
        bus._queue.put_nowait = orig
    assert calls == [None]  # the wake-up sentinel was attempted
    # Control: on a healthy queue the sentinel is enqueued.
    bus2 = ev.EventBus(max_queue=4)
    bus2.stop(timeout=0.1)
    assert bus2._queue.qsize() == 1
    assert bus2._queue.get_nowait() is None


# ── 485: _enqueue block policy falls through to drop accounting ────────────

def test_485_enqueue_block_full_counts_the_drop(caplog):
    bus = ev.EventBus(max_queue=1, on_full="block", publish_timeout=0.05)
    bus._queue.put_nowait(("filler", ev.Event(topic="f")))
    sub = ev.Subscription(pattern="a.*", handler=lambda e: None)
    with caplog.at_level(logging.WARNING, logger="nomorals.core.events"):
        bus._enqueue(sub, ev.Event(topic="a.b"))  # must not raise
    # The fall-through IS the accounting: counter + warning.
    assert bus._dropped == 1
    assert any("queue full" in r.getMessage() for r in caplog.records)
    # Control: with room, the event is enqueued and nothing drops.
    bus2 = ev.EventBus(max_queue=4, on_full="block", publish_timeout=0.05)
    bus2._enqueue(sub, ev.Event(topic="a.b"))
    assert bus2._dropped == 0 and bus2._queue.qsize() == 1


# ── 726: journal rotation failure keeps writing ────────────────────────────

def test_726_journal_rotation_failure_keeps_writing(tmp_path, monkeypatch):
    path = str(tmp_path / "journal.jsonl")
    bus = ev.EventBus(journal_path=path, journal_max_bytes=10)
    bus._journal_write(ev.Event(topic="first.event"))
    calls = []

    def boom(src, dst):
        calls.append((src, dst))
        raise OSError("cannot rotate")

    monkeypatch.setattr(os, "replace", boom)
    bus._journal_write(ev.Event(topic="second.event"))  # must not raise
    bus._journal_close()
    assert calls and calls[0][1] == path + ".1"  # rotation attempted
    topics = [json.loads(line)["topic"]
              for line in Path(path).read_text().strip().splitlines()]
    assert topics == ["first.event", "second.event"]  # nothing lost


# ── 739: journal close failure is teardown best-effort ─────────────────────

def test_739_journal_close_failure_detaches_handle():
    class BadFile:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
            raise RuntimeError("close boom")

    bus = ev.EventBus()
    bad = BadFile()
    bus._journal = bad
    bus._journal_close()  # must not raise
    assert bad.closed  # the close was attempted
    assert bus._journal is None  # handle detached before the attempt
