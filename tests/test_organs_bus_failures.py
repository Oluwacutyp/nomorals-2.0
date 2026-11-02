"""Pass tail-117 pins: nomorals/organs_bus.py — 3 silent handlers.

- 132 E103 ``_load_state``: missing/corrupt/non-dict state file ->
  the fresh cursor {"consumed": [], "attempts": {}}. Consequence
  pinned honestly: already-consumed events become pending again
  (duplicate re-delivery — the failure degrades toward at-least-once,
  never toward loss; the event log is append-only and untouched) and
  the next drain's save repairs the state file. Not a data-loss
  shape: no durable record is destroyed, only a cursor is reset.
- 159 E103 ``_read_log``: OSError -> the rows read so far ([] when
  the log is absent/unreadable); torn lines are skipped per-line.
- 293 E103 ``_read_dead``: OSError -> [] for the dead-letter review.

The module has no logger at all — non-recording is structural.
"""

import json

from nomorals.organs_bus import PersistedBus


def make_bus(tmp_path):
    return PersistedBus(tmp_path / "bus.jsonl")


# ── 132: _load_state ─────────────────────────────────────────────────

def test_load_state_missing_file_is_fresh(tmp_path):
    bus = make_bus(tmp_path)
    assert bus._load_state() == {"consumed": [], "attempts": {}}


def test_load_state_corrupt_file_is_fresh(tmp_path):
    bus = make_bus(tmp_path)
    bus.state_path.write_text("{corrupt json", encoding="utf-8")
    assert bus._load_state() == {"consumed": [], "attempts": {}}


def test_load_state_non_dict_file_is_fresh(tmp_path):
    bus = make_bus(tmp_path)
    bus.state_path.write_text("[1, 2]", encoding="utf-8")
    assert bus._load_state() == {"consumed": [], "attempts": {}}


def test_load_state_valid_file_round_trips(tmp_path):
    bus = make_bus(tmp_path)
    bus.state_path.write_text(
        json.dumps({"consumed": ["a"], "attempts": {"b": 2}}),
        encoding="utf-8")
    state = bus._load_state()
    assert state["consumed"] == ["a"] and state["attempts"] == {"b": 2}


def test_corrupt_state_redelivers_but_never_loses(tmp_path):
    bus = make_bus(tmp_path)
    bus.publish({"kind": "event", "dst": "x"})
    assert len(bus.drain()) == 1
    assert bus.pending_count() == 0
    bus.state_path.write_text("{corrupt json", encoding="utf-8")
    # consumed cursor forgotten -> the event is pending again (a
    # duplicate, not a loss), and the next drain re-saves valid state
    assert bus.pending_count() == 1
    assert len(bus.drain()) == 1
    assert bus.pending_count() == 0
    assert isinstance(json.loads(bus.state_path.read_text()), dict)
    # the event log itself was never touched
    assert len(bus._read_log()) == 1


# ── 159: _read_log ───────────────────────────────────────────────────

def test_read_log_missing_file_is_empty(tmp_path):
    bus = make_bus(tmp_path)
    assert bus._read_log() == []


def test_read_log_skips_torn_lines(tmp_path):
    bus = make_bus(tmp_path)
    bus.publish({"kind": "event", "dst": "x"})
    with open(bus.path, "a", encoding="utf-8") as fh:
        fh.write("{torn line\n")
        fh.write('["not", "a", "dict"]\n')
    rows = bus._read_log()
    assert len(rows) == 1 and rows[0]["dst"] == "x"


def test_read_log_oserror_is_empty(tmp_path):
    bus = make_bus(tmp_path)
    bus.path = tmp_path  # a directory: open() raises IsADirectoryError
    assert bus._read_log() == []


# ── 293: _read_dead ──────────────────────────────────────────────────

def test_read_dead_missing_file_is_empty(tmp_path):
    bus = make_bus(tmp_path)
    assert bus._read_dead() == []


def test_read_dead_returns_rows(tmp_path):
    bus = make_bus(tmp_path)
    bus.dead_letter({"id": "dead1"}, "test reason")
    rows = bus._read_dead()
    assert len(rows) == 1 and rows[0]["reason"] == "test reason"
