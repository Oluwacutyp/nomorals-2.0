"""Pass #218 pins: storage/rowstore.py — 1 exclusion, 0 converts.

- 74: ``_default_literal`` int probe. A PRAGMA default value is parsed
  as int first; when ``int(text)`` raises ValueError the code falls
  through to the float probe, and when that refuses too the function
  returns None (no default). The swallow is the correct behavior: the
  probe's next rung runs, and an unparseable default honestly means
  "this column has no usable default" — the caller (column_defaults)
  simply omits the column.

storage/rowstore.py DOES record failures elsewhere (the repository
decode seam), so NOT-recording is pinned against the storage ledger:
storage recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use plain values, never a swallow.
"""

from nomorals.storage import failures as st_fail
from nomorals.storage.rowstore import _default_literal


def _snap():
    return list(st_fail.recent_failures(limit=200))


# ── L74: the int probe falls through to the float probe ───────────────


def test_74_int_string_returns_int():
    before = _snap()
    assert _default_literal("42") == 42
    assert isinstance(_default_literal("42"), int)
    assert _default_literal("-7") == -7
    assert _default_literal("(42)") == 42
    assert _snap() == before


def test_74_float_string_falls_through_the_int_probe():
    # int("3.5") raises ValueError; the float rung runs and the
    # value lands.
    before = _snap()
    assert _default_literal("3.5") == 3.5
    assert isinstance(_default_literal("3.5"), float)
    assert _default_literal("-0.25") == -0.25
    assert _default_literal("1e3") == 1000.0
    assert _snap() == before


def test_74_unparseable_returns_none():
    # Both probes refuse; None is the honest "no usable default".
    before = _snap()
    assert _default_literal("garbage") is None
    assert _default_literal("") is None
    assert _default_literal("   ") is None
    assert _snap() == before


def test_74_null_and_none_and_quoted():
    before = _snap()
    assert _default_literal(None) is None
    assert _default_literal("NULL") is None
    assert _default_literal("null") is None
    assert _default_literal("'quoted'") == "quoted"
    assert _default_literal("'it''s'") == "it's"
    assert _snap() == before
