"""Tail-107 pins: nomorals/tools/typed_encoding.py census sites
(93, 99, 104, 119).

All four sites are held as EXCLUSIONS. Every site sits inside
``_decode_typed``, whose docstring prints the contract up front:
"Never raises on malformed data: an unresolvable tag degrades to
the decoded payload." The consumer is the tool registry's
idempotent-replay restore (``tools/registry.py`` decodes recorded
outcomes): a corrupt stored payload must degrade, never crash the
replay path. The module has no logger and no failure ledger —
non-recording is structural, pinned below.

93 — bytes/bytearray tag, corrupt base64 -> ``b""``. The weakest
of the four (named for parent adjudication): the sibling fallbacks
return the raw payload as residue, but bytes has no honest
in-type residue for undecodable data — ``b""`` is the degradation
the docstring promises, and the alternative (returning the raw
string) would silently hand a ``str`` to a consumer promised
``bytes``. Pinned: corrupt payloads (both tags) -> ``b""``; valid
payload -> the real bytes.

99 — datetime tag, unparseable ISO text -> the raw payload string
(the decoded residue, honestly not a datetime). Control: a valid
ISO string restores a real ``datetime``.

104 — decimal tag, unparseable number text -> the raw payload
string. Control: "1.5" restores ``Decimal("1.5")``.

119 — custom tag, class unresolvable (module gone / renamed) ->
the decoded STATE DICT: the object's data survives as plain data
instead of vanishing or crashing the replay. Control: a live
class (argparse.Namespace) round-trips through encode/decode back
to a real instance.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import decimal

from nomorals.tools import typed_encoding as temod


def test_module_has_no_recording_surface():
    # Non-recording is structural: no logger, no failure ledger.
    assert not hasattr(temod, "_log")
    assert not hasattr(temod, "record_failure")


# ── 93: corrupt bytes payload degrades to b"" ─────────────────────

def test_93_corrupt_bytes_payload_returns_empty():
    assert temod._decode_typed({"__type__": "bytes", "data": "a"}) == b""


def test_93_corrupt_bytearray_payload_returns_empty():
    assert temod._decode_typed(
        {"__type__": "bytearray", "data": "a"}) == b""


def test_93_control_valid_bytes_roundtrip():
    good = base64.b64encode(b"hi").decode("ascii")
    assert temod._decode_typed({"__type__": "bytes", "data": good}) == b"hi"


# ── 99: corrupt datetime payload degrades to the raw string ───────

def test_99_corrupt_datetime_returns_raw_payload():
    assert temod._decode_typed(
        {"__type__": "datetime", "data": "not-a-date"}) == "not-a-date"


def test_99_control_valid_datetime_restored():
    assert temod._decode_typed(
        {"__type__": "datetime", "data": "2026-01-02T03:04:05"}
    ) == datetime.datetime(2026, 1, 2, 3, 4, 5)


# ── 104: corrupt decimal payload degrades to the raw string ───────

def test_104_corrupt_decimal_returns_raw_payload():
    assert temod._decode_typed(
        {"__type__": "decimal", "data": "not-a-number"}) == "not-a-number"


def test_104_control_valid_decimal_restored():
    assert temod._decode_typed(
        {"__type__": "decimal", "data": "1.5"}) == decimal.Decimal("1.5")


# ── 119: unresolvable custom class degrades to the state dict ─────

def test_119_gone_class_returns_state_dict():
    out = temod._decode_typed({"__type__": "custom",
                               "class": "no.such.module.Gone",
                               "state": {"a": 1}})
    assert out == {"a": 1}


def test_119_control_live_class_restored():
    enc = temod._encode_typed(argparse.Namespace(x=7))
    dec = temod._decode_typed(enc)
    assert isinstance(dec, argparse.Namespace)
    assert dec.x == 7
