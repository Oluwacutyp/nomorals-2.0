"""Pass #216 pins: agents/betdesk/collectors.py (_coerce_ts) — 1 exclusion, 0 converts.

- 171: ``_coerce_ts`` float parse ``except ValueError: pass``.
  The float parse is a probe; on failure the ISO-8601 parse is
  tried next, and 0.0 is returned if that fails too (the
  documented "unknown timestamp" value). The failure does not
  hide anything — the next parse runs, and a timestamp that
  parses as neither a number nor ISO-8601 is honestly 0.0.

agents/betdesk/collectors.py calls no failures-ledger record
function anywhere, so NOT-recording is pinned with the agency
snapshot method: agency recent_failures() before/after each
trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file —
the scanner counts test files too. Triggers use timestamp strings
that are neither numbers nor ISO-8601, instead of swallowing.
"""

from nomorals.agents import failures as ag_fail
from nomorals.agents.betdesk import collectors as col


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L171: float parse failure falls through to ISO-8601 ───────────

def test_171_iso_timestamp_parses_after_float_fails():
    # An ISO-8601 timestamp is not a float, so the float probe
    # fails and the ISO parse runs next.
    before = _snap()
    assert col._coerce_ts("2026-11-14T15:00:00Z") == 1794668400.0
    assert _snap() == before


def test_171_unparseable_timestamp_is_zero():
    # A timestamp that is neither a number nor ISO-8601 is honestly
    # 0.0 (unknown), not a fabricated time.
    before = _snap()
    assert col._coerce_ts("not-a-timestamp") == 0.0
    assert _snap() == before


def test_171_numeric_timestamp_is_used():
    # Control: a numeric timestamp is used directly.
    before = _snap()
    assert col._coerce_ts(1794668400) == 1794668400.0
    assert col._coerce_ts("1794668400") == 1794668400.0
    assert _snap() == before


def test_171_millisecond_timestamp_is_scaled():
    # Control: an epoch-ms value is scaled to seconds.
    before = _snap()
    assert col._coerce_ts(1794668400000) == 1794668400.0
    assert _snap() == before


def test_171_none_and_empty_are_zero():
    # Control: None and empty text are 0.0 without reaching the probe.
    before = _snap()
    assert col._coerce_ts(None) == 0.0
    assert col._coerce_ts("") == 0.0
    assert col._coerce_ts("   ") == 0.0
    assert _snap() == before
