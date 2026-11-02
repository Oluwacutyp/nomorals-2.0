"""Tail-94 pins: nomorals/tools/metadata.py census sites.

The module's contract is printed in its header: "Every parser
is defensive — a truncated or hostile file yields partial
metadata plus a warning, never a crash." All five sites are
E103 with NARROW exception tuples (specific struct/decode
errors, never broad Exception), each isolating one unit of a
best-effort parse so the rest survives. Held as exclusions:

158 — one EXIF tag's value bytes are corrupt (offset points
outside the TIFF) -> that tag is skipped; every other tag
still parses. Per-tag isolation is the parser's granularity
contract.
204 — the GPS sub-IFD is corrupt (a rational offset points
outside the TIFF) -> the GPS block is skipped whole; the
already-collected IFD0 tags survive in the result.
246 — a PNG text chunk is undecodable -> the chunk is
skipped; later chunks still parse. NOTE: with the body's
current decode arguments (latin-1 keys, utf-8/"replace"
values, a length-guarded parts[-1]) the guarded
UnicodeDecodeError/IndexError cannot actually arise — the
handler is defensive depth around the decode, and the pin
proves the contract it guards (chunk isolation + loop
continuation), not a reachable raise.
409 — a FLAC VORBIS_COMMENT block is corrupt mid-entry ->
the block's comments are dropped whole (the merge into the
result happens only after the full block parses, so partial
comments never leak in half-parsed); all other metadata
survives. Block granularity, disclosed here honestly.
607 — temp-file unlink in extract_metadata's finally ->
OSError swallowed. NOTE: the site is currently UNREACHABLE —
``temp_path`` is initialized to None (line 568) and never
reassigned anywhere in the function (the URL branch reads
into memory; the module header's "downloaded to a temp file
first" is stale). The pin proves the finally never disturbs
a result even with unlink sabotaged; the dead guard and the
stale header line are reported as a cleanup follow-up, not
fixed in a pins batch.

No recording is claimed: nomorals/tools has no failures
ledger and the module imports none; the pins assert behavior
only. Observability of skipped tags/chunks is a shelved
bind+debug sketch, not a convert — no site here reports a
success that did not happen; partial results are the stated
product.

Each test pins the degraded behavior AND a healthy control.
No bare ``except: pass`` anywhere; no sys.path hacks.
"""
from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

from nomorals.tools import metadata as M


# ── TIFF builders (little-endian) ──────────────────────────────

def _entry(tag, typ, num, value4):
    return struct.pack("<HHI", tag, typ, num) + value4


def _tiff(ifd0_entries, gps_ifd=b""):
    hdr = b"II" + struct.pack("<HI", 42, 8)
    ifd0 = (struct.pack("<H", len(ifd0_entries)) + b"".join(ifd0_entries)
            + struct.pack("<I", 0))
    return hdr + ifd0 + gps_ifd


_MAKE = _entry(0x010F, 2, 3, b"EOS\x00")          # Make = "EOS"
_GPS_OFF = 8 + 2 + 2 * 12 + 4                     # after hdr + IFD0(2 entries)


# ── 158: corrupt numeric tag skipped, other tags survive ───────

def test_158_corrupt_exif_tag_skipped_others_survive():
    bad_iso = _entry(0x8827, 4, 2, struct.pack("<I", 0x7FFF0000))
    found = M._parse_exif_ifd(_tiff([_MAKE, bad_iso]))
    assert found == {"Make": "EOS"}
    # control: the same tag well-formed parses to its value
    good_iso = _entry(0x8827, 4, 1, struct.pack("<I", 200))
    found2 = M._parse_exif_ifd(_tiff([_MAKE, good_iso]))
    assert found2 == {"Make": "EOS", "ISO": 200}


# ── 204: corrupt GPS block skipped, IFD0 tags survive ──────────

def test_204_corrupt_gps_block_skipped_ifd0_survives():
    gps_ptr = _entry(0x8825, 4, 1, struct.pack("<I", _GPS_OFF))
    bad_gps = (struct.pack("<H", 1)
               + _entry(0x0002, 5, 3, struct.pack("<I", 0x7FFF0000))
               + struct.pack("<I", 0))
    found = M._parse_exif_ifd(_tiff([_MAKE, gps_ptr], bad_gps))
    assert found == {"Make": "EOS"}
    # control: well-formed GPS rationals produce the decimal pair
    rats = b"".join(struct.pack("<II", n, d)
                    for n, d in ((51, 1), (30, 1), (0, 1)))
    rat_off = _GPS_OFF + 2 + 2 * 12 + 4
    good_gps = (struct.pack("<H", 2)
                + _entry(0x0001, 2, 2, b"N\x00\x00\x00")
                + _entry(0x0002, 5, 3, struct.pack("<I", rat_off))
                + struct.pack("<I", 0) + rats)
    found2 = M._parse_exif_ifd(_tiff([_MAKE, gps_ptr], good_gps))
    assert found2["Make"] == "EOS"
    assert found2["GPS"]["decimal"] == (51.5, None)
    assert found2["GPS"]["lat"] == [51.0, 30.0, 0.0]


# ── 246: text-chunk isolation — neighbours still parse ─────────

def _chunk(typ, payload):
    return struct.pack(">I", len(payload)) + typ + payload + b"\x00" * 4


def _png():
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
    return (b"\x89PNG\r\n\x1a\n" + ihdr
            + _chunk(b"tEXt", b"Title\x00Hello")
            + _chunk(b"tEXt", b"Separatorless")
            + _chunk(b"tEXt", b"Author\x00Devon"))


def test_246_text_chunks_isolated_and_loop_continues():
    out = M._png_meta(_png())
    assert out["dimensions"] == "3x2"
    assert out["Title"] == "Hello"
    assert out["Author"] == "Devon"     # chunk after the odd one parsed
    # a separator-less chunk degrades to an empty-valued key,
    # never a crash and never a lost neighbour
    assert out["Separatorless"] == ""


# ── 409: corrupt comment block dropped whole, format survives ──

def _flac_comment(entries, count=None):
    body = struct.pack("<I", 3) + b"abc"
    body += struct.pack("<I", len(entries) if count is None else count)
    for e in entries:
        body += struct.pack("<I", len(e)) + e
    return body


def _flac(block_body, tail=b""):
    block = bytes([0x80 | 4]) + len(block_body).to_bytes(3, "big") + block_body
    return b"fLaC" + block + tail


def test_409_corrupt_comment_block_dropped_whole():
    good = _flac(_flac_comment([b"title=Song", b"artist=Devon"]))
    out = M._flac_meta(good)                                # control
    assert out["title"] == "Song" and out["artist"] == "Devon"
    # count promises a 2nd entry; only 2 bytes of its length
    # field remain -> the block is dropped whole, title included
    body = _flac_comment([b"title=Song"], count=2) + b"\x05\x00"
    out2 = M._flac_meta(_flac(body))
    assert out2 == {"format": "FLAC"}


# ── 607: finally cleanup never disturbs a result ───────────────

def test_607_finally_cleanup_never_disturbs_result(tmp_path, monkeypatch):
    target = tmp_path / "pic.png"
    target.write_bytes(_png())
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    meta = M.extract_metadata(ctx, str(target))
    assert meta["format"] == "PNG" and meta["Title"] == "Hello"
    assert len(meta["sha256"]) == 64

    class _UnlinkBoom(Path):
        def unlink(self, *a, **k):
            raise OSError("cannot remove")

    monkeypatch.setattr(M, "Path", _UnlinkBoom)
    meta2 = M.extract_metadata(ctx, str(target))
    assert meta2["format"] == "PNG" and meta2["sha256"] == meta["sha256"]
