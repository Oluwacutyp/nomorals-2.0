"""Page surgery (Smallpdf #48 + ILovePDF #49): merge/select/rotate/stamp/
compress/repair/protect/unlock/archive/scan/sign over real PDFs.

Fixtures are real PDFs, two kinds: (a) minimal hand-built PDFs with
one distinct text draw per page — page -> text mapping is exact, so
every op's round-trip is measured on extraction, never asserted; (b)
PDFs rendered by Devon's own WinAnsi writer (``core.pdf``) — the
"real-world" corpus.  Rotation is proven on the page box, not the
text; merge is proven as A-then-B extraction order.
"""

from __future__ import annotations

import re
import unittest
import zlib

from nomorals.core.pdf import read_pdf_text, render_pdf
from nomorals.documents.pagewise import (
    ArchiveOp,
    CompressOp,
    CropOp,
    DeleteOp,
    MergeOp,
    PagewiseJob,
    ProtectOp,
    ReorderOp,
    RepairOp,
    RotateOp,
    SelectOp,
    SignOp,
    StampOp,
    UnlockOp,
    archive_pdf,
    extract_page_texts,
    pdf_info,
    protect_pdf,
    repair_pdf,
    run_pagewise,
    scan_to_pdf,
    unlock_pdf,
)


def _pdf_escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("(", "\\(")
                .replace(")", "\\)"))


def make_pdf(pages: list[str], *, compress: bool = False) -> bytes:
    """A minimal valid PDF: one Helvetica text draw per page.

    ``pages[i]`` is the exact text drawn on page i (1-based in the
    proof).  Streams are uncompressed unless ``compress=True``.
    """
    objs: dict[int, bytes] = {}
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    page_ids = [4 + 2 * i for i in range(len(pages))]
    stream_ids = [5 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{n} 0 R" for n in page_ids)
    objs[2] = (f"<< /Type /Pages /Kids [{kids}] "
               f"/Count {len(pages)} >>").encode()
    objs[3] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
               b"/Encoding /WinAnsiEncoding >>")
    for i, text in enumerate(pages):
        stream = (f"BT /F1 24 Tf 72 720 Td "
                  f"({_pdf_escape(text)}) Tj ET").encode("latin-1")
        if compress:
            payload = zlib.compress(stream)
            objs[stream_ids[i]] = (
                f"<< /Length {len(payload)} /Filter /FlateDecode >>\n"
                f"stream\n".encode() + payload + b"\nendstream")
        else:
            objs[stream_ids[i]] = (
                f"<< /Length {len(stream)} >>\nstream\n".encode()
                + stream + b"\nendstream")
        objs[page_ids[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {stream_ids[i]} 0 R >>").encode()
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + objs[num] + b"\nendobj\n"
    xref = len(out)
    top = max(offsets)
    out += f"xref\n0 {top + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, top + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {top + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def _texts(data: bytes) -> list[str]:
    return [t.strip() for t in extract_page_texts(data)]


class TestPagewiseCore(unittest.TestCase):
    def test_merge_extracts_a_then_b(self):
        a = make_pdf(["ALPHA-ONE", "ALPHA-TWO"])
        b = make_pdf(["BETA-ONE"])
        result = run_pagewise(PagewiseJob(
            sources=[a, b], ops=[MergeOp()]))
        self.assertEqual(result.pages, 3)
        self.assertEqual(_texts(result.data),
                         ["ALPHA-ONE", "ALPHA-TWO", "BETA-ONE"])
        # the merge is structural: the output is a fresh object graph
        self.assertTrue(result.data.startswith(b"%PDF-"))
        self.assertIn(b"startxref", result.data)

    def test_select_ranges(self):
        src = make_pdf(["P1", "P2", "P3", "P4", "P5"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[SelectOp("2-3,5")]))
        self.assertEqual(result.pages, 3)
        self.assertEqual(_texts(result.data), ["P2", "P3", "P5"])

    def test_select_reversed_range_is_an_error_not_a_guess(self):
        src = make_pdf(["P1", "P2"])
        with self.assertRaises(ValueError):
            run_pagewise(PagewiseJob(
                sources=[src], ops=[SelectOp("2-1")]))

    def test_delete_pages(self):
        src = make_pdf(["P1", "P2", "P3"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[DeleteOp("2")]))
        self.assertEqual(_texts(result.data), ["P1", "P3"])

    def test_reorder_pages(self):
        src = make_pdf(["P1", "P2", "P3"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[ReorderOp([3, 1, 2])]))
        self.assertEqual(_texts(result.data), ["P3", "P1", "P2"])

    def test_rotate_changes_box_not_text(self):
        src = make_pdf(["ROT-MARKER"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[RotateOp("1", 90)]))
        self.assertEqual(_texts(result.data), ["ROT-MARKER"])
        # the page dict now carries /Rotate 90
        self.assertIn(b"/Rotate 90", result.data)
        # and the box is unchanged (rotation is a dict entry, not a rewrite)
        self.assertIn(b"/MediaBox [0 0 612 792]", result.data)

    def test_rotate_accumulates(self):
        src = make_pdf(["ROT-MARKER"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[RotateOp("1", 90), RotateOp("1", 90)]))
        self.assertIn(b"/Rotate 180", result.data)

    def test_rotate_rejects_non_right_angle(self):
        src = make_pdf(["P1"])
        with self.assertRaises(ValueError):
            run_pagewise(PagewiseJob(
                sources=[src], ops=[RotateOp("1", 45)]))

    def test_crop_sets_cropbox(self):
        src = make_pdf(["CROP-MARKER"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[CropOp("1", (72, 72, 540, 720))]))
        self.assertEqual(_texts(result.data), ["CROP-MARKER"])
        self.assertIn(b"/CropBox [72 72 540 720]", result.data)

    def test_stamp_page_numbers(self):
        src = make_pdf(["STAMP-A", "STAMP-B"])
        result = run_pagewise(PagewiseJob(
            sources=[src],
            ops=[StampOp("{n} of {total}", "all",
                         position="bottom-center")]))
        texts = _texts(result.data)
        self.assertIn("STAMP-A", texts[0])
        self.assertIn("1 of 2", texts[0])
        self.assertIn("STAMP-B", texts[1])
        self.assertIn("2 of 2", texts[1])

    def test_stamp_watermark_text(self):
        src = make_pdf(["WM-MARKER"])
        result = run_pagewise(PagewiseJob(
            sources=[src],
            ops=[StampOp("CONFIDENTIAL", "1", position="center")]))
        self.assertIn("CONFIDENTIAL", _texts(result.data)[0])

    def test_stamp_keeps_original_text(self):
        # the stamp stream is appended, never a re-author of the page
        src = make_pdf(["KEEP-ME"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[StampOp("STAMPED", "1")]))
        self.assertIn("KEEP-ME", _texts(result.data)[0])
        self.assertIn("STAMPED", _texts(result.data)[0])

    def test_compress_shrinks_and_keeps_text(self):
        # a bloated source: the same long draw repeated on 6 pages,
        # streams uncompressed, plus orphan objects the writer drops
        long_text = "COMPRESS-MARKER " * 40
        src = make_pdf([long_text] * 6)
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[CompressOp()]))
        self.assertEqual(_texts(result.data),
                         [long_text.strip()] * 6)
        self.assertLess(result.size_after, result.size_before)
        self.assertGreater(result.size_before - result.size_after, 0)

    def test_compress_reports_measured_delta(self):
        src = make_pdf(["DELTA-MARKER " * 30] * 3)
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[CompressOp()]))
        delta = next(m for m in result.measurements
                     if m["op"] == "compress")
        self.assertEqual(delta["size_before"], result.size_before)
        self.assertEqual(delta["size_after"], result.size_after)
        self.assertEqual(delta["saved_bytes"],
                         result.size_before - result.size_after)

    def test_ops_compose_merge_then_select_then_stamp(self):
        a = make_pdf(["A1", "A2"])
        b = make_pdf(["B1", "B2"])
        result = run_pagewise(PagewiseJob(
            sources=[a, b],
            ops=[MergeOp(), SelectOp("2-3"),
                 StampOp("pg {n}", "all")]))
        texts = _texts(result.data)
        self.assertEqual(len(texts), 2)
        self.assertIn("A2", texts[0])
        self.assertIn("pg 1", texts[0])
        self.assertIn("B1", texts[1])
        self.assertIn("pg 2", texts[1])

    def test_empty_job_is_an_error(self):
        src = make_pdf(["P1"])
        with self.assertRaises(ValueError):
            run_pagewise(PagewiseJob(sources=[src], ops=[]))

    def test_not_a_pdf_is_an_error(self):
        with self.assertRaises(ValueError):
            run_pagewise(PagewiseJob(
                sources=[b"this is not a pdf"], ops=[MergeOp()]))

    def test_select_out_of_range_is_an_error(self):
        src = make_pdf(["P1"])
        with self.assertRaises(ValueError):
            run_pagewise(PagewiseJob(
                sources=[src], ops=[SelectOp("9")]))

    def test_info_reports_pages_and_encryption(self):
        src = make_pdf(["P1", "P2"])
        info = pdf_info(src)
        self.assertEqual(info["pages"], 2)
        self.assertFalse(info["encrypted"])
        locked = protect_pdf(src, "secret")
        self.assertTrue(pdf_info(locked)["encrypted"])

    def test_merge_with_real_writer_pdf(self):
        # "real" corpus: Devon's own WinAnsi writer, multi-page
        text = "\n\n".join(f"Chapter {i}\n\nBody text {i} " * 40
                           for i in range(1, 4))
        real = render_pdf(text, title="Real", chapter_break=True,
                          headings=True)
        self.assertGreaterEqual(pdf_info(real)["pages"], 2)
        other = make_pdf(["EXTRA"])
        result = run_pagewise(PagewiseJob(
            sources=[real, other], ops=[MergeOp()]))
        self.assertEqual(result.pages, pdf_info(real)["pages"] + 1)
        self.assertIn("EXTRA", _texts(result.data)[-1])
        # every real page's text survives the merge
        merged = read_pdf_text(result.data)
        for i in (1, 2, 3):
            self.assertIn(f"Chapter {i}", merged)


class TestPagewiseRepair(unittest.TestCase):
    def _corrupt(self, data: bytes, kind: str) -> bytes:
        if kind == "truncated-xref":
            # cut the xref/trailer off entirely
            cut = data.find(b"xref")
            return data[:cut]
        if kind == "broken-offsets":
            # scramble every xref offset to a wrong value
            return re.sub(rb"(?m)^(\d{10}) 00000 n ",
                          b"0000000001 00000 n ", data)
        if kind == "missing-trailer":
            cut = data.find(b"trailer")
            return data[:cut] + b"%%EOF\n"
        if kind == "garbage-prefix":
            return b"GARBAGE-BYTES\n" + data
        if kind == "truncated-file":
            # cut the last page object out mid-stream
            cut = data.rfind(b"endobj")
            return data[:cut]
        raise ValueError(kind)

    def test_repair_recovers_all_pages_per_corruption_class(self):
        src = make_pdf(["R1", "R2", "R3"])
        for kind in ("truncated-xref", "broken-offsets",
                     "missing-trailer", "garbage-prefix"):
            with self.subTest(kind=kind):
                broken = self._corrupt(src, kind)
                repaired = repair_pdf(broken)
                self.assertEqual(_texts(repaired), ["R1", "R2", "R3"])
                info = pdf_info(repaired)
                self.assertEqual(info["pages"], 3)
                self.assertFalse(info["encrypted"])

    def test_repair_reports_recovered_counts(self):
        src = make_pdf(["R1", "R2", "R3"])
        broken = self._corrupt(src, "truncated-xref")
        result = run_pagewise(PagewiseJob(
            sources=[broken], ops=[RepairOp()]))
        self.assertEqual(result.pages, 3)
        rep = next(m for m in result.measurements if m["op"] == "repair")
        self.assertEqual(rep["recovered_pages"], 3)
        self.assertEqual(rep["recovered_objects"], rep["total_objects"])
        self.assertIn("truncated-xref", rep["corruption"])

    def test_repair_truncated_file_is_honest_about_loss(self):
        src = make_pdf(["R1", "R2", "R3"])
        broken = self._corrupt(src, "truncated-file")
        result = run_pagewise(PagewiseJob(
            sources=[broken], ops=[RepairOp()]))
        # the last page's bytes are gone: recovery is partial and
        # the report says so, never a silent partial presented as whole
        self.assertLess(result.pages, 3)
        rep = next(m for m in result.measurements if m["op"] == "repair")
        self.assertLess(rep["recovered_pages"], 3)
        self.assertIn("recovered_pages", rep)
        self.assertEqual(_texts(result.data)[:2], ["R1", "R2"])

    def test_repair_of_healthy_file_is_a_clean_rewrite(self):
        src = make_pdf(["H1", "H2"])
        repaired = repair_pdf(src)
        self.assertEqual(_texts(repaired), ["H1", "H2"])
        self.assertIn(b"startxref", repaired)


class TestPagewiseProtect(unittest.TestCase):
    def test_protect_then_unlock_round_trip(self):
        src = make_pdf(["SECRET-ONE", "SECRET-TWO"])
        locked = protect_pdf(src, "hunter2")
        self.assertTrue(pdf_info(locked)["encrypted"])
        self.assertIn(b"/Encrypt", locked)
        # the plaintext is really gone from the file bytes
        self.assertNotIn(b"SECRET-ONE", locked)
        opened = unlock_pdf(locked, "hunter2")
        self.assertFalse(pdf_info(opened)["encrypted"])
        self.assertEqual(_texts(opened), ["SECRET-ONE", "SECRET-TWO"])

    def test_unlock_wrong_password_refuses(self):
        src = make_pdf(["P1"])
        locked = protect_pdf(src, "right")
        with self.assertRaises(ValueError):
            unlock_pdf(locked, "wrong")

    def test_unlock_unencrypted_is_an_error(self):
        src = make_pdf(["P1"])
        with self.assertRaises(ValueError):
            unlock_pdf(src, "whatever")

    def test_protect_empty_password_refuses(self):
        src = make_pdf(["P1"])
        with self.assertRaises(ValueError):
            protect_pdf(src, "")

    def test_protect_via_job_and_unlock_via_job(self):
        src = make_pdf(["JOB-SECRET"])
        locked = run_pagewise(PagewiseJob(
            sources=[src], ops=[ProtectOp("pw123")]))
        self.assertTrue(pdf_info(locked.data)["encrypted"])
        opened = run_pagewise(PagewiseJob(
            sources=[locked.data], ops=[UnlockOp("pw123")]))
        self.assertEqual(_texts(opened.data), ["JOB-SECRET"])

    def test_protect_keeps_page_count_and_text_after_unlock(self):
        text = "\n\n".join(f"Section {i} " * 20 for i in range(1, 4))
        real = render_pdf(text, title="Real", chapter_break=True,
                          headings=True)
        locked = protect_pdf(real, "pw")
        opened = unlock_pdf(locked, "pw")
        self.assertEqual(pdf_info(opened)["pages"],
                         pdf_info(real)["pages"])
        self.assertEqual(read_pdf_text(opened), read_pdf_text(real))


class TestPagewiseArchiveAndScan(unittest.TestCase):
    def test_archive_adds_xmp_and_markinfo(self):
        src = make_pdf(["ARCH-ONE"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[ArchiveOp()]))
        self.assertEqual(_texts(result.data), ["ARCH-ONE"])
        self.assertIn(b"/Metadata", result.data)
        self.assertIn(b"pdfaid:part", result.data)
        self.assertIn(b"/MarkInfo", result.data)
        self.assertIn(b"/Marked true", result.data)
        arch = next(m for m in result.measurements if m["op"] == "archive")
        self.assertIn("labels", arch)

    def test_archive_labels_unembedded_fonts(self):
        src = make_pdf(["ARCH-ONE"])
        result = run_pagewise(PagewiseJob(
            sources=[src], ops=[ArchiveOp()]))
        arch = next(m for m in result.measurements if m["op"] == "archive")
        # Helvetica is a base-14 font, not embedded: the label says so
        self.assertTrue(any("font" in label.lower()
                            for label in arch["labels"]))

    def test_archive_refuses_encrypted_without_password(self):
        src = make_pdf(["P1"])
        locked = protect_pdf(src, "pw")
        with self.assertRaises(ValueError):
            archive_pdf(locked)

    def test_scan_jpeg_to_pdf(self):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (200, 100), color=(10, 120, 200)).save(
            buf, format="JPEG")
        out = scan_to_pdf([buf.getvalue()], title="Scan")
        self.assertEqual(pdf_info(out)["pages"], 1)
        self.assertIn(b"/DCTDecode", out)
        self.assertIn(b"/Subtype /Image", out)

    def test_scan_two_images_two_pages(self):
        import io

        from PIL import Image

        images = []
        for color in ((200, 30, 30), (30, 200, 30)):
            buf = io.BytesIO()
            Image.new("RGB", (100, 100), color=color).save(
                buf, format="JPEG")
            images.append(buf.getvalue())
        out = scan_to_pdf(images, title="Two scans")
        self.assertEqual(pdf_info(out)["pages"], 2)

    def test_scan_png_goes_through_pillow(self):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (80, 60), color=(1, 2, 3)).save(
            buf, format="PNG")
        out = scan_to_pdf([buf.getvalue()])
        self.assertEqual(pdf_info(out)["pages"], 1)

    def test_scan_empty_is_an_error(self):
        with self.assertRaises(ValueError):
            scan_to_pdf([])

    def test_scan_non_image_is_an_error(self):
        with self.assertRaises(ValueError):
            scan_to_pdf([b"not an image at all"])

    def test_sign_places_image_on_page(self):
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (120, 40), color=(0, 0, 0)).save(
            buf, format="JPEG")
        src = make_pdf(["SIGN-MARKER"])
        result = run_pagewise(PagewiseJob(
            sources=[src],
            ops=[SignOp(image=buf.getvalue(), page=1,
                        rect=(400, 700, 120, 40))]))
        self.assertEqual(_texts(result.data), ["SIGN-MARKER"])
        self.assertIn(b"/Subtype /Image", result.data)
        self.assertIn(b"/DCTDecode", result.data)


if __name__ == "__main__":
    unittest.main()
