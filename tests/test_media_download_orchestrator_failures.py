"""Pass #216 pins: media/download_orchestrator.py — 2 exclusions, 0 converts.

- 191: ``sniff_format`` UTF-8 decode ``except UnicodeDecodeError:
  pass``. The UTF-8 decode is a probe; on failure the latin-1
  decode is tried next, and the function returns "unknown" if
  neither works. The failure does not hide anything — the next
  probe runs, and the outer handler records and returns "unknown"
  (a sniff never kills a download).
- 229: ``image_dimensions`` Pillow ``except ImportError: pass``.
  Pillow is the preferred rung; on ImportError the PNG/JPEG/GIF
  header fallback runs, and the function returns None when neither
  can read it ("None when neither can read it. Never raises.").

media/download_orchestrator.py records failures to its own ledger
(nomorals.media.download_kit_failures), so NOT-recording is pinned
by snapshotting that ledger's recent_failures() before/after each
trigger. Note the latin-1 branch DOES record ("sniff_format.text")
when it fails — the UTF-8 probe itself records nothing, which is
what the pins below prove.

Pins rule (#169): no bare-swallow handler anywhere in this file —
the scanner counts test files too. Triggers use a non-UTF-8 byte
sample and a missing Pillow, instead of swallowing.
"""

import struct
from unittest.mock import patch

from nomorals.media import download_kit_failures as dkf
from nomorals.media import download_orchestrator as do


def _snap():
    return list(dkf.recent_failures(500))


# ── L191: UTF-8 decode failure falls through to latin-1 ──────────

def test_191_non_utf8_sample_falls_through_to_latin1(tmp_path):
    # A sample that is not valid UTF-8 but is valid latin-1 text
    # (printable latin-1) is sniffed as "text" via the latin-1 probe.
    before = _snap()
    f = tmp_path / "a.txt"
    f.write_bytes(b"caf\xe9 na\xefve \xfcmlaut " * 200)
    assert do.sniff_format(f) == "text"
    assert _snap() == before


def test_191_binary_sample_is_unknown(tmp_path):
    # Control: a sample that is neither UTF-8 nor latin-1 text
    # (binary with NULs) is "unknown", not "text".
    before = _snap()
    f = tmp_path / "b.bin"
    f.write_bytes(bytes(range(256)) * 20)
    assert do.sniff_format(f) == "unknown"
    assert _snap() == before


def test_191_valid_utf8_text_is_text(tmp_path):
    # Control: a valid UTF-8 text sample is "text" via the first probe.
    before = _snap()
    f = tmp_path / "c.txt"
    f.write_bytes(b"hello world, plain text\n" * 100)
    assert do.sniff_format(f) == "text"
    assert _snap() == before


# ── L229: missing Pillow falls through to header parsing ──────────

def test_229_no_pillow_png_header_still_reads_dimensions(tmp_path):
    # A PNG whose header carries 320x240 reads (320, 240) with
    # Pillow unimportable — the header fallback runs.
    before = _snap()
    png = tmp_path / "a.png"
    png.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
        + struct.pack(">II", 320, 240) + b"\x00" * 16)
    with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
        assert do.image_dimensions(png) == (320, 240)
    assert _snap() == before


def test_229_no_pillow_unreadable_is_none(tmp_path):
    # Control: a file neither Pillow nor the headers can read is None.
    before = _snap()
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"\x00\x01\x02\x03" * 100)
    with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
        assert do.image_dimensions(junk) is None
    assert _snap() == before
