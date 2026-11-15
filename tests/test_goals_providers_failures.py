"""Pass #217 pins: goals/providers.py — 1 exclusion, 0 converts.

- 742: ``_gs_record_ts`` float probe. The ``float(text)`` call is a
  probe for an epoch-number string; when it raises ValueError the
  code falls through to ISO-8601 parsing (a naive ISO value is read
  as UTC — GS never writes one, and assuming local time would
  silently shift the goal window). A value that parses neither way
  returns None. The swallow is the correct behavior: the probe's
  next rung runs, and the function's contract already produces None
  for unparseable values on a non-error path.

goals/providers.py calls no failures-ledger record function anywhere,
so NOT-recording is pinned with the agency snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use plain values, never a
swallow.
"""

from nomorals.agents import failures as ag_fail
from nomorals.goals.providers import _gs_record_ts


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L742: the float probe falls through to ISO parsing ──────────────

def test_742_epoch_string_returns_the_float():
    before = _snap()
    assert _gs_record_ts({"ts": "1700000000"}) == 1700000000.0
    assert _gs_record_ts({"ts": 1700000000}) == 1700000000.0
    assert _gs_record_ts({"ts": 1700000000.5}) == 1700000000.5
    assert _snap() == before


def test_742_zoned_iso_falls_through_the_float_probe():
    # float("2020-01-01T00:00:00+00:00") raises ValueError; the ISO
    # rung runs and the timestamp lands.
    before = _snap()
    assert _gs_record_ts({"ts": "2020-01-01T00:00:00+00:00"}) == 1577836800.0
    assert _gs_record_ts({"ts": "2020-01-01T00:00:00Z"}) == 1577836800.0
    assert _snap() == before


def test_742_naive_iso_is_read_as_utc_not_local():
    # A naive ISO value is read as UTC (GS never writes one); the
    # probe still runs first — float() refuses it, ISO parses it.
    before = _snap()
    assert _gs_record_ts({"ts": "2020-01-01T00:00:00"}) == 1577836800.0
    assert _snap() == before


def test_742_unparseable_returns_none():
    before = _snap()
    assert _gs_record_ts({"ts": "garbage"}) is None
    assert _gs_record_ts({"ts": ""}) is None
    assert _gs_record_ts({"ts": "   "}) is None
    assert _gs_record_ts({"ts": None}) is None
    assert _gs_record_ts({}) is None
    assert _gs_record_ts({"ts": True}) is None
    assert _snap() == before
