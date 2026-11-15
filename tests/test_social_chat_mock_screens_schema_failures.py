"""Pass #218 pins: social/chat/mock_screens/schema.py — 1 exclusion, 0 converts.

- 525: ``_hex_color`` validation probe. The value is parsed as hex to
  prove it is a ``#rrggbb`` string; when ``int(text[1:], 16)`` raises
  ValueError the code falls through to ``raise ValueError`` with a
  message naming the field and the offending value. The error is NOT
  swallowed — the except is the probe that lets the raise carry the
  field name. A valid colour returns unchanged; an empty value returns
  "" (no colour).

mock_screens/schema.py calls no failures-ledger record function anywhere,
so NOT-recording is pinned with the agency snapshot method: agency
recent_failures() before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use plain values, never a swallow.
"""

import pytest

from nomorals.agents import failures as ag_fail
from nomorals.social.chat.mock_screens.schema import _hex_color


def _snap():
    return list(ag_fail.recent_failures(limit=200))


# ── L525: the hex probe falls through to the ValueError ───────────────


def test_525_valid_hex_returns_unchanged():
    before = _snap()
    assert _hex_color("#ff0000", "status bg") == "#ff0000"
    assert _hex_color("#00F0FF", "story bg") == "#00F0FF"
    assert _snap() == before


def test_525_empty_returns_empty():
    before = _snap()
    assert _hex_color("", "status bg") == ""
    assert _hex_color("   ", "status bg") == ""
    assert _hex_color(None, "status bg") == ""
    assert _snap() == before


def test_525_malformed_hex_raises_with_field_and_value():
    # "#zzzzzz" enters the probe (starts with "#" and is 7 chars),
    # int() refuses it, the except passes, and the raise below fires.
    before = _snap()
    with pytest.raises(ValueError) as excinfo:
        _hex_color("#zzzzzz", "status bg")
    assert "status bg" in str(excinfo.value)
    assert "must be #rrggbb" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo2:
        _hex_color("#gggggg", "story bg")
    assert "story bg" in str(excinfo2.value)
    assert _snap() == before


def test_525_non_hex_and_wrong_length_raise():
    before = _snap()
    with pytest.raises(ValueError):
        _hex_color("zzzzzz", "status bg")
    with pytest.raises(ValueError):
        _hex_color("#fff", "status bg")
    with pytest.raises(ValueError):
        _hex_color("#ff00000", "status bg")
    assert _snap() == before
