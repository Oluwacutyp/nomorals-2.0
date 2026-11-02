"""Tail-93 pins: nomorals/security/exif.py census sites.

The module's law is printed in its section header: the scan is
"(awareness, not removal)". Every parser is best-effort with
absent-value defaults — a positive (has_gps True, a coordinate,
a tag) exists only when bytes actually parsed. Removal never
consults the scan: ``strip_metadata_bytes`` does its own
segment surgery by marker/chunk, so a scan shortfall cannot
bless a dirty file clean. All five sites are held as
exclusions:

148 — ``_parse_exif_app1`` orientation conversion crash ->
the tag is skipped and orientation keeps its default 1,
which is the EXIF spec's own default orientation ("normal"),
not an invented reading.
171 — GPS block crash (a RATIONAL whose promised bytes run
off the end of the TIFF -> struct.error) -> gps stays None
and has_gps stays False. Corrupt GPS yields NO coordinates,
never wrong ones — the only honest failure direction for a
location field.
454 — ``scan_metadata`` whole-scan guard: an unexpected
crash mid-scan returns the report accumulated so far plus
absent-value defaults instead of raising into the caller.
State set before the crash (has_exif) survives; nothing
unexamined is claimed present. WEAKEST EXCLUSION of this
batch — see the batch report: fields after the crash point
default to absent with no "scan incomplete" marker.
485 — ``_pil_exif_tags`` per-tag str() crash -> that tag is
skipped (continue); every other tag is still listed.
488 — ``_pil_exif_tags`` outer: Pillow absent/broken -> {}.
PIL is the optional accelerator; the module's own stdlib
parser has already produced the report this listing merely
enriches.

NOT-recording is pinned with the snapshot method: exif.py
calls no failures-ledger record function on these paths, so
both the agency and agents ledgers must be unchanged across
each trigger.
"""
from __future__ import annotations

import struct
import sys
import types
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.security import exif as X


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _ifd(entries, next_off=0):
    return (struct.pack("<H", len(entries)) + b"".join(entries)
            + struct.pack("<I", next_off))


def _entry(tag, typ, cnt, raw4):
    return struct.pack("<HHI", tag, typ, cnt) + raw4


def _exif_payload(ifd0: bytes, extra: bytes = b"") -> bytes:
    tiff = (b"II" + struct.pack("<H", 42) + struct.pack("<I", 8) + ifd0)
    return b"Exif\x00\x00" + tiff + extra


# ── 148: orientation value unconvertible -> spec default 1 ────

def test_148_orientation_bad_type_keeps_default():
    before = _snap()
    # BYTE-typed orientation: _ifd_value returns raw bytes and
    # int(bytes) raises TypeError -> tag skipped, default kept
    payload = _exif_payload(_ifd([_entry(0x0112, 1, 1, b"\x03\x00\x00\x00")]))
    assert X._parse_exif_app1(payload) == {"orientation": 1, "gps": None}
    # control: a SHORT orientation parses to its real value
    payload = _exif_payload(
        _ifd([_entry(0x0112, 3, 1, struct.pack("<H", 6) + b"\x00\x00")]))
    assert X._parse_exif_app1(payload)["orientation"] == 6
    assert _snap() == before


# ── 171: GPS block crash -> gps None, never wrong coords ─────

def _gps_ifd0(gps_off: int) -> bytes:
    return _ifd([_entry(0x8825, 4, 1, struct.pack("<I", gps_off))])


_GPS_OFF = 8 + 2 + 12 + 4  # tiff header + ifd0(count+entry+next)
_LAT_OFF = _GPS_OFF + 2 + 4 * 12 + 4


def test_171_gps_truncated_rational_yields_no_gps():
    before = _snap()
    gps_entries = [
        _entry(0x0001, 2, 2, b"N\x00\x00\x00"),
        _entry(0x0002, 5, 3, struct.pack("<I", _LAT_OFF)),
        _entry(0x0003, 2, 2, b"E\x00\x00\x00"),
        _entry(0x0004, 5, 3, struct.pack("<I", _LAT_OFF)),
    ]
    # only 4 of the 24 promised rational bytes are present:
    # _ifd_value's struct.unpack raises struct.error mid-block
    payload = _exif_payload(_gps_ifd0(_GPS_OFF),
                            _ifd(gps_entries) + b"\x01\x00\x00\x00")
    assert X._parse_exif_app1(payload) == {"orientation": 1, "gps": None}
    assert _snap() == before


def test_171_gps_valid_block_parses_coordinates():
    before = _snap()
    lat = struct.pack("<IIIIII", 1, 1, 30, 1, 0, 1)   # 1° 30' 0" = 1.5
    lon = struct.pack("<IIIIII", 2, 1, 15, 1, 0, 1)   # 2° 15' 0" = 2.25
    gps_entries = [
        _entry(0x0001, 2, 2, b"N\x00\x00\x00"),
        _entry(0x0002, 5, 3, struct.pack("<I", _LAT_OFF)),
        _entry(0x0003, 2, 2, b"E\x00\x00\x00"),
        _entry(0x0004, 5, 3, struct.pack("<I", _LAT_OFF + len(lat))),
    ]
    payload = _exif_payload(_gps_ifd0(_GPS_OFF),
                            _ifd(gps_entries) + lat + lon)
    assert X._parse_exif_app1(payload)["gps"] == {"lat": 1.5, "lon": 2.25}
    assert _snap() == before


# ── 454: whole-scan crash -> partial report, no raise ─────────

def _jpeg_with_app1(payload: bytes) -> bytes:
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return b"\xff\xd8" + seg + b"\xff\xd9"


def test_454_scan_crash_returns_partial_report():
    before = _snap()
    payload = _exif_payload(
        _ifd([_entry(0x0112, 3, 1, struct.pack("<H", 6) + b"\x00\x00")]))
    with mock.patch.object(X, "_parse_exif_app1",
                           side_effect=RuntimeError("parse boom")):
        out = X.scan_metadata(_jpeg_with_app1(payload))
    # the report returns instead of raising; state set before
    # the crash (has_exif) survives, unexamined fields default
    assert out["format"] == "jpeg"
    assert out["has_exif"] is True
    assert out["has_gps"] is False
    assert out["gps"] is None
    assert out["orientation"] == 1
    # control: with the real parser the same file reports fully
    out = X.scan_metadata(_jpeg_with_app1(payload))
    assert out["has_exif"] is True
    assert out["orientation"] == 6
    assert _snap() == before


# ── 485/488: PIL tag listing — bad tag skipped, no PIL -> {} ──

class _BadStr:
    def __str__(self):
        raise RuntimeError("tag str boom")


def _fake_pil():
    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    tags = types.ModuleType("PIL.ExifTags")

    class _Im:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getexif(self):
            return {1: _BadStr(), 2: "Canon"}

    image.open = lambda buf: _Im()
    tags.TAGS = {1: "Make", 2: "Model"}
    pil.Image = image
    pil.ExifTags = tags
    return pil, image, tags


def test_485_pil_bad_tag_skipped_others_kept():
    before = _snap()
    pil, image, tags = _fake_pil()
    with mock.patch.dict(sys.modules, {"PIL": pil, "PIL.Image": image,
                                       "PIL.ExifTags": tags}):
        assert X._pil_exif_tags(b"jpeg-bytes") == {"Model": "Canon"}
    assert _snap() == before


def test_488_pil_unavailable_returns_empty():
    before = _snap()
    with mock.patch.dict(sys.modules, {"PIL": None}):
        assert X._pil_exif_tags(b"jpeg-bytes") == {}
    assert _snap() == before
