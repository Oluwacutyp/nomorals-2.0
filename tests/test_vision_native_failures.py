"""Tail-36 pins for nomorals/vision/native.py — 9 exclusions, 0 converts.

Each test pins one site's fallback behavior plus a control half proving the
except path was taken (trigger == fallback AND control != fallback, or the
mock call log proves the failing operation was attempted). The file itself
contains no exception handlers, so it adds zero census sites.

native.py calls no record function anywhere (grep-proven: only _log.debug),
so NOT-recording is pinned with the tail-8 snapshot method:
agency recent_failures() before/after each trigger must be unchanged — a
swallowed failure leaves no trace anywhere. (The module's own debug log is
not a failure record.)

Site 305's pin needs its mechanism note: the except fires only when
exif_data's getexif is the FIRST getexif call on the image. In production
load_image's exif_transpose always calls getexif first (swallowed by site
171), and Pillow's Exif.load sets _loaded_exif BEFORE the parse that raises
— so a corrupt EXIF yields the cached empty Exif, and exif_data reports
absence via the EMPTY path (``if not raw``), never via 305's except. The
pin therefore simulates the only remaining trigger (transpose never ran,
i.e. broken ImageOps import) with a raising getexif double. The real
corrupt-WebP flow (EXIF bytes present, "present: False" reported) is a
genuine defect, but it lives at the empty-path conflation, NOT at site 305
— documented in SURVEY.md as an out-of-batch targeted-fix candidate, not
pinned here as desired behavior.
"""

import io
import struct
import sys
from unittest import mock

from PIL import Image

from nomorals.agents import failures as ag_fail
from nomorals.vision import native as N


def _snap():
    return list(ag_fail.recent_failures())


def _jpeg_with_exif():
    exif = Image.Exif()
    exif[271] = "TestMake"
    exif[272] = "TestModel"
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _webp_with_exif():
    exif = Image.Exif()
    exif[271] = "WMake"
    exif[272] = "WModel"
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(buf, format="WEBP", exif=exif)
    return buf.getvalue()


def _corrupt_webp_exif(data):
    """Flip the EXIF chunk's TIFF byte-order mark; pixels intact."""
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    pos = 12
    while pos + 8 <= len(data):
        ctype = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        payload = pos + 8
        if ctype == b"EXIF":
            out = bytearray(data)
            mark = bytes(out[payload:payload + 2])
            assert mark in (b"II", b"MM"), mark
            out[payload] = ord("M") if mark == b"II" else ord("I")
            return bytes(out)
        pos = payload + size + (size % 2)
    raise AssertionError("no EXIF chunk found")


def test_149_resolver_failure_falls_back_to_which():
    before = _snap()
    with mock.patch("nomorals.llm.providers.ocr.ocr_binary",
                    return_value="/sentinel/tesseract") as res:
        assert N._tesseract_binary() == "/sentinel/tesseract" and res.called
    blocked = "nomorals.llm.providers.ocr"
    saved = sys.modules.get(blocked)
    sys.modules[blocked] = None  # forces ImportError on import
    try:
        with mock.patch("nomorals.vision.native.shutil.which",
                        return_value="/which/tesseract") as wh:
            assert N._tesseract_binary() == "/which/tesseract" and wh.called
    finally:
        if saved is None:
            sys.modules.pop(blocked, None)
        else:
            sys.modules[blocked] = saved
    assert _snap() == before


def test_171_transpose_failure_still_returns_image():
    before = _snap()
    with mock.patch("PIL.ImageOps.exif_transpose",
                    side_effect=RuntimeError("boom")) as tr:
        img = N.load_image(_jpeg_with_exif())
        assert tr.called  # the failing operation itself was attempted
        assert img.size == (32, 32)
    assert N.load_image(_jpeg_with_exif()).size == (32, 32)
    assert _snap() == before


def test_250_rational_fallthrough():
    before = _snap()
    assert N._rational_to_float((3, 2)) == 1.5
    assert N._rational_to_float(2.5) == 2.5
    assert N._rational_to_float((3, 0)) is None
    assert N._rational_to_float("junk") is None
    assert _snap() == before


def test_289_gps_failure_is_none_none():
    before = _snap()
    good = {1: "N", 2: ((10, 1), (30, 1), (0, 1)),
            3: "E", 4: ((20, 1), (0, 1), (0, 1))}
    lat, lon = N._gps_to_decimal(good)
    assert abs(lat - 10.5) < 1e-9 and abs(lon - 20.0) < 1e-9
    assert N._gps_to_decimal({}) == (None, None)
    assert N._gps_to_decimal({1: "N", 2: ("bad",), 3: "E",
                              4: ((20, 1), (0, 1), (0, 1))}) == (None, None)
    assert _snap() == before


def test_305_getexif_failure_reports_absence():
    before = _snap()
    # Raising double: simulates transpose never having run (broken ImageOps).
    with mock.patch.object(Image.Image, "getexif",
                           side_effect=SyntaxError("not a TIFF file")):
        assert N.exif_data(_jpeg_with_exif()).get("present") is False
    # Control: good file reads present.
    assert N.exif_data(_jpeg_with_exif()).get("present") is True
    assert _snap() == before


def test_320_gpsinfo_non_dict_drops_gps_only():
    before = _snap()

    def _run(mapping):
        with mock.patch.object(Image.Image, "getexif", return_value=mapping):
            return N.exif_data(_jpeg_with_exif())

    assert "GPS" not in _run({271: "Make", 34853: 12345})
    assert _run({271: "Make", 34853: 12345}).get("Make") == "Make"
    gps = {1: "N", 2: ((10, 1), (0, 1), (0, 1)),
           3: "E", 4: ((20, 1), (0, 1), (0, 1))}
    ctrl = _run({271: "Make", 34853: gps})
    assert ctrl.get("GPS", {}).get("present") is True
    assert "GPSLatitude" in ctrl["GPS"]["raw"]
    # Latitude is None even here: the out-of-batch key-mismatch defect
    # (SURVEY.md), NOT 320's behavior — not asserted as desired.
    assert _snap() == before


def test_331_bad_tag_skipped_rest_kept():
    before = _snap()

    class BadBytes(bytes):
        def decode(self, *a, **k):
            raise RuntimeError("boom")

    with mock.patch.object(Image.Image, "getexif",
                           return_value={271: "Make", 272: BadBytes(b"x")}):
        result = N.exif_data(_jpeg_with_exif())
    assert result.get("Make") == "Make"
    assert "Model" not in result
    assert _snap() == before


def _tesseract_mocks():
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
           "left\ttop\twidth\theight\tconf\ttext\n"
           "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\thello\n")
    proc = mock.Mock()
    proc.stdout = tsv.encode()
    proc.stderr = b""
    proc.returncode = 0
    return (mock.patch.object(N, "_tesseract_binary",
                              return_value="/bin/tess"),
            mock.patch.object(N.subprocess, "run", return_value=proc))


def _unlink_failer(calls, prefix):
    real_unlink = N.os.unlink

    def failing_unlink(path):
        if prefix in str(path):
            calls.append(path)
            raise OSError("boom")
        return real_unlink(path)

    return failing_unlink


def test_777_layout_unlink_failure_keeps_result():
    before = _snap()
    p1, p2 = _tesseract_mocks()
    calls = []
    with p1, p2, mock.patch.object(
            N.os, "unlink", _unlink_failer(calls, "nm-layout-")):
        result = N.document_layout(_jpeg_with_exif())
    assert calls, "finally must have attempted the unlink"
    assert result["words"] == 1 and result["method"] == "tesseract-tsv"
    assert _snap() == before


def test_875_ocr_unlink_failure_keeps_result():
    before = _snap()
    p1, p2 = _tesseract_mocks()
    calls = []
    with p1, p2, mock.patch.object(
            N.os, "unlink", _unlink_failer(calls, "nm-ocr-")):
        out = N._tesseract_run(_jpeg_with_exif(), language="eng", psm=11,
                               kind="tsv", timeout=30.0)
    assert calls, "finally must have attempted the unlink"
    assert "hello" in out
    assert _snap() == before
