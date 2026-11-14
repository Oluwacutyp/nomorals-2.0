"""Pass #216 pins: games/games/base.py (Room.from_row) — 2 exclusions, 0 converts.

Both sites are in ``Room.from_row``, which rebuilds a Room from a
stored row. A malformed stored timestamp falls back to the field
default (``time.time()`` — now), and the room is still constructed
and usable. The row is legacy data; a bad timestamp does not
invalidate the room:

- 220: ``turn_started`` ``except (TypeError, ValueError): pass``.
  The field keeps its default (now); the room is returned.
- 225: ``last_activity`` ``except (TypeError, ValueError): pass``.
  Same — the field keeps its default (now); the room is returned.

The timestamps ride the state blob (``_turn_started`` /
``_last_activity``); the malformed values below are placed there,
exactly as a corrupt stored row would carry them.

games/games/base.py calls no failures-ledger record function
anywhere, so NOT-recording is pinned with the agency snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file —
the scanner counts test files too. Triggers use malformed stored
timestamps, instead of swallowing.
"""

import json
import time

from nomorals.agents import failures as ag_fail
from nomorals.games.games import base as gb


def _snap():
    return list(ag_fail.recent_failures(limit=200))


def _row(**state):
    return {
        "id": "r1",
        "game": "wordchain",
        "chat_key": "c1",
        "players": "[]",
        "turn": 0,
        "state": json.dumps(state),
        "status": "active",
        "revision": 1,
        "started_at": time.time(),
        "seed": 1,
    }


# ── L220: malformed turn_started falls back to the default ────────

def test_220_malformed_turn_started_keeps_default():
    before = _snap()
    room = gb.Room.from_row(_row(_turn_started="not-a-number"), {})
    # The default is time.time() (now); the room is still built.
    assert room.turn_started > 0
    assert room.id == "r1"
    assert _snap() == before


def test_220_valid_turn_started_is_used():
    # Control: a valid stored turn_started is used, not the default.
    before = _snap()
    room = gb.Room.from_row(_row(_turn_started=1234.5), {})
    assert room.turn_started == 1234.5
    assert _snap() == before


# ── L225: malformed last_activity falls back to the default ───────

def test_225_malformed_last_activity_keeps_default():
    before = _snap()
    room = gb.Room.from_row(_row(_last_activity={"bogus": True}), {})
    assert room.last_activity > 0
    assert room.id == "r1"
    assert _snap() == before


def test_225_valid_last_activity_is_used():
    # Control: a valid stored last_activity is used, not the default.
    before = _snap()
    room = gb.Room.from_row(_row(_last_activity=2345.5), {})
    assert room.last_activity == 2345.5
    assert _snap() == before
