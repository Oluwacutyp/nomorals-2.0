"""Tail-105 pins: nomorals/health/timeline.py census sites.

All four sites are held as exclusions:

385 — ``_row_to_event``: an unparseable ``extra`` JSON column
degrades to ``{}`` — the identical value the same expression
produces for a missing/empty extra (``json.loads(get("extra")
or "{}")``). The row's core fields (id, type, text, ts,
severity) are parsed outside the guard and survive untouched;
``extra`` is a free-form annotation bag whose documented empty
value is ``{}``. Read-path only: nothing is rewritten, so no
durable record is lost or falsified.

399 — ``close()``: teardown. A failing ``db.close()`` during
shutdown has no consumer and no honest failure value to
return; the class is being discarded either way.

525 — ``sparkline``: pure render helper, docstring "Pure;
never raises". The failure value ``""`` is the identical value
the function returns for no data (``if not vals: return ""``)
— degradation to the module's own documented default.

572 — ``parse_measurement``: pure parser guard. With the real
regexes every conversion inside the try runs on digit-only
capture groups (``\\d{2,3}``, ``\\d+(?:\\.\\d+)?``), so the
``float(...)`` calls provably cannot raise on matched text;
the guard exists for pathological inputs only. What the try
returns on a crash is ``out`` — only triples genuinely parsed
so far, never an invented measurement. Guard-for-unreachable-
raise + partial-parse contract.

NOT-recording is pinned with the snapshot method: timeline.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.health.timeline as TL
from nomorals.health.timeline import HealthTimeline, parse_measurement, sparkline


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _bare_timeline(db=None):
    tl = HealthTimeline.__new__(HealthTimeline)
    tl._db = db
    return tl


def _row(extra):
    return {"id": 7, "event_type": "note", "text": "knee hurt",
            "ts": 1700000000.0, "severity": None, "value": "",
            "unit": "", "source": "chat", "extra": extra}


# ── 385: corrupt extra JSON -> {} ───────────────────────────────

def test_385_corrupt_extra_json_degrades_to_empty_dict():
    before = _snap()
    tl = _bare_timeline()
    ev = tl._row_to_event(_row("{not json"))
    assert ev.extra == {}
    # core fields survive untouched
    assert ev.id == "7" and ev.event_type == "note"
    assert ev.text == "knee hurt" and ev.ts == 1700000000.0
    # a non-dict JSON payload degrades identically
    assert tl._row_to_event(_row("[1, 2]")).extra == {}
    # control: valid extra parses through
    assert tl._row_to_event(_row('{"k": 2}')).extra == {"k": 2}
    # control: missing extra is the same {} the crash yields
    assert tl._row_to_event(_row(None)).extra == {}
    assert _snap() == before


# ── 399: close() swallows a failing db.close ────────────────────

def test_399_close_swallows_db_close_failure():
    before = _snap()
    tl = _bare_timeline(db=SimpleNamespace(close=_raise))
    assert tl.close() is None
    # control: a working close really is invoked
    calls = []
    tl2 = _bare_timeline(db=SimpleNamespace(close=lambda: calls.append(1)))
    tl2.close()
    assert calls == [1]
    assert _snap() == before


# ── 525: sparkline crash -> "" (its own no-data value) ──────────

def test_525_sparkline_failure_returns_empty_string():
    before = _snap()
    assert sparkline(["not-a-number"]) == ""
    # the failure value is the documented no-data value
    assert sparkline([]) == ""
    # control: real values render one spark char per point
    out = sparkline([1.0, 2.0, 3.0])
    assert len(out) == 3 and all(c in TL._SPARK for c in out)
    assert _snap() == before


# ── 572: parse guard -> only genuinely parsed triples ───────────

def test_572_parse_measurement_guard_returns_partial_out():
    before = _snap()
    boom = SimpleNamespace(search=_raise)
    with mock.patch.object(TL, "_BP_RE", boom):
        assert parse_measurement("120/80") == []
    # controls: the real regexes parse for real
    assert parse_measurement("120/80") == [
        ("blood_pressure_systolic", 120.0, "mmHg"),
        ("blood_pressure_diastolic", 80.0, "mmHg")]
    assert parse_measurement("72kg") == [("weight", 72.0, "kg")]
    assert parse_measurement("nothing measurable") == []
    assert _snap() == before
