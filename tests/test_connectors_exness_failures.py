"""Tail-101 pins: nomorals/connectors/exness.py census sites.

All four sites are held as exclusions:

229 — _parse_private_seed(): a decoder raising -> try the next
encoding. This is a fallback ladder over three accepted seed
encodings, and it ends in a loud ExnessError naming what was
decodable when nothing yields 32 bytes — refusal-shaped, no key
is ever fabricated from a partial decode.
335 — _learn_clock_offset(): any failure parsing the server's
Date header -> pass, the previous offset stands. Clock sync is
best-effort by the site's own comment; a stale offset risks a
rejected request (loud at the API layer), never a forged one.
561 — test_connection(): any exception -> False. Fail-CLOSED:
the error direction claims *less* access than reality, the safe
side for a connection probe.
681 — _format_prices(): the instrument-condition read raises
ExnessError -> digits stay at the documented 2dp fallback
("fall back to 2dp rather than refusing the trade"). The except
is narrow (ExnessError only); the trade still goes through the
_gate confirmation path afterwards.

NOT-recording is pinned with the snapshot method: none of these
sites call a failures-ledger record function, so both ledgers
must be unchanged across each trigger.
"""
from __future__ import annotations

import base64
import time
from email.utils import formatdate
from types import SimpleNamespace

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.connectors.exness import (
    ExnessConnector,
    ExnessError,
    _parse_private_seed,
)


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _bare_connector():
    return ExnessConnector.__new__(ExnessConnector)


# ── 229: undecodable seed -> loud refusal, never a fake key ────

def test_229_undecodable_seed_raises():
    before = _snap()
    # 45 data chars ≡ 1 (mod 4): both base64 decoders raise
    # binascii.Error, both are caught, the ladder ends in refusal.
    with pytest.raises(ExnessError):
        _parse_private_seed("A" * 45)
    # pure noise: decoders discard it, nothing reaches 32 bytes.
    with pytest.raises(ExnessError):
        _parse_private_seed("!!!")
    assert _snap() == before


def test_229_seed_controls_all_encodings():
    seed = bytes(range(32))
    assert _parse_private_seed(seed.hex()) == seed
    b64url = base64.urlsafe_b64encode(seed).rstrip(b"=").decode()
    assert _parse_private_seed(b64url) == seed
    assert _parse_private_seed(
        base64.b64encode(seed).decode()) == seed


# ── 335: clock-sync failure -> previous offset stands ──────────

def test_335_bad_date_header_keeps_offset():
    before = _snap()
    conn = _bare_connector()
    conn._clock_offset_ms = 1234
    conn._learn_clock_offset(
        SimpleNamespace(headers={"date": "not a real date"}))
    assert conn._clock_offset_ms == 1234
    conn._learn_clock_offset(SimpleNamespace(headers={}))
    assert conn._clock_offset_ms == 1234
    assert _snap() == before


def test_335_clock_control_learns_offset():
    conn = _bare_connector()
    conn._clock_offset_ms = 1234
    conn._learn_clock_offset(SimpleNamespace(
        headers={"date": formatdate(time.time(), usegmt=True)}))
    # server ≈ our clock, biased into the past by the safety margin
    assert conn._clock_offset_ms != 1234
    assert -60_000 < conn._clock_offset_ms <= 0


# ── 561: test_connection fails closed ───────────────────────────

def _raise_status():
    raise RuntimeError("offline")


def test_561_test_connection_failure_is_false():
    before = _snap()
    conn = _bare_connector()
    conn.status = _raise_status
    assert conn.test_connection() is False
    assert _snap() == before


def test_561_test_connection_controls():
    conn = _bare_connector()
    conn.status = lambda: SimpleNamespace(connected=True)
    assert conn.test_connection() is True
    conn.status = lambda: SimpleNamespace(connected=False)
    assert conn.test_connection() is False


# ── 681: price digits fall back to documented 2dp ──────────────

def _no_condition(instrument):
    raise ExnessError("condition unavailable")


def test_681_format_prices_fallback_two_digits():
    before = _snap()
    conn = _bare_connector()
    conn.get_instrument_condition = _no_condition
    out = conn._format_prices("XAUUSD", sl=1900.126, tp=1950.5,
                              price=None)
    assert out == {"sl": "1900.13", "tp": "1950.50"}
    assert _snap() == before


def test_681_format_prices_control_uses_point_digits():
    conn = _bare_connector()
    conn.get_instrument_condition = (
        lambda instrument: {"point_digits": 3})
    out = conn._format_prices("XAUUSD", sl=1900.126)
    assert out == {"sl": "1900.126"}
