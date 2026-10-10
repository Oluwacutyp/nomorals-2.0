"""OCR for scanned/image-only PDFs and images (L4).

Text extraction (:func:`nomorals.documents.parse_bytes` on a PDF) returns
nothing useful for scanned pages — they are pictures of text.  This module
bridges that gap with Tesseract via ``pytesseract`` (page rendering via
``pdf2image``/poppler).

Beyond page text, the module exposes Tesseract's structured output —
word-level boxes + confidences (:func:`ocr_pdf_words`, the Surya-style
line/word view) and hOCR (:func:`ocr_pdf_hocr`) — plus OCRmyPDF-style
workflow features: multi-core page parallelism (``jobs``), per-page
progress callbacks, PSM/OEM engine knobs, and best-effort auto-rotation
via Tesseract's orientation detection.

Both heavy pieces are *optional*: when they are missing the functions
below fail fast with :class:`DocumentError` carrying an exact install
hint — they never silently return empty text.

Install hints (Debian/Ubuntu; other platforms need the same two binaries):

* ``pip install pytesseract pdf2image``
* ``sudo apt install tesseract-ocr poppler-utils``
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .errors import DocumentError
from .model import Document, Section, new_document

__all__ = ["ocr_available", "ocr_image", "ocr_languages", "ocr_pdf",
           "ocr_pdf_hocr", "ocr_pdf_words"]

_log = get_logger(__name__)

_PIP_HINT = "pip install pytesseract pdf2image"
_TESSERACT_HINT = "tesseract-ocr system package (Debian/Ubuntu: sudo apt install tesseract-ocr)"
_POPPLER_HINT = "poppler-utils system package (Debian/Ubuntu: sudo apt install poppler-utils)"

#: Hard cap on pages per ocr_pdf call unless the caller raises it.
_DEFAULT_MAX_PAGES = 25


def _require_pytesseract() -> Any:
    """Import pytesseract and verify the tesseract binary, or fail fast."""
    try:
        import pytesseract
    except ImportError as exc:
        raise DocumentError(
            "OCR needs the optional 'pytesseract' package: "
            f"{_PIP_HINT}.  Also install {_TESSERACT_HINT}.") from exc
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise DocumentError(
            "OCR needs the Tesseract binary on PATH and it was not found: "
            f"install {_TESSERACT_HINT}.  ({exc})") from exc
    return pytesseract


def _require_pdf2image() -> Any:
    """Import pdf2image and verify the poppler renderer, or fail fast."""
    try:
        import pdf2image
    except ImportError as exc:
        raise DocumentError(
            "OCR of PDFs needs the optional 'pdf2image' package: "
            f"{_PIP_HINT}.  Also install {_POPPLER_HINT}.") from exc
    return pdf2image


def ocr_available() -> bool:
    """True when the full OCR stack (pytesseract + tesseract binary +
    pdf2image) is importable.  Never raises."""
    try:
        _require_pytesseract()
        _require_pdf2image()
    except DocumentError:
        return False
    return True


def ocr_languages() -> list[str]:
    """Installed Tesseract language codes (``tesseract --list-langs``).

    Raises :class:`DocumentError` when the OCR stack is missing.
    """
    pytesseract = _require_pytesseract()
    try:
        return sorted(pytesseract.get_languages(config=""))
    except Exception as exc:
        raise DocumentError(f"could not list Tesseract languages: {exc}") from exc


def _engine_config(psm: int | None, oem: int | None) -> str:
    """Build the ``--psm``/``--oem`` config string (validated)."""
    parts: list[str] = []
    if psm is not None:
        if not 0 <= psm <= 13:
            raise DocumentError(f"psm must be 0-13, got {psm}")
        parts.append(f"--psm {psm}")
    if oem is not None:
        if not 0 <= oem <= 3:
            raise DocumentError(f"oem must be 0-3, got {oem}")
        parts.append(f"--oem {oem}")
    return " ".join(parts)


def _validate_window(data: bytes, first_page: int,
                     last_page: int | None, max_pages: int,
                     dpi: int, jobs: int = 1) -> None:
    if not data:
        raise DocumentError("cannot OCR empty input")
    if not data[:5].startswith(b"%PDF-"):
        raise DocumentError("ocr_pdf expects PDF bytes (not a PDF file)")
    if first_page < 1:
        raise DocumentError(f"first_page must be >= 1, got {first_page}")
    if last_page is not None and last_page < first_page:
        raise DocumentError(
            f"last_page ({last_page}) is before first_page ({first_page})")
    if max_pages < 1:
        raise DocumentError(f"max_pages must be >= 1, got {max_pages}")
    if dpi < 72:
        raise DocumentError(f"dpi {dpi} is too low for OCR (minimum 72)")
    if jobs < 1:
        raise DocumentError(f"jobs must be >= 1, got {jobs}")


def _render_pages(data: bytes, dpi: int, first_page: int,
                  last_page: int | None, max_pages: int) -> list[Any]:
    """Rasterize the PDF page window to PIL images (poppler)."""
    pdf2image = _require_pdf2image()
    try:
        images = pdf2image.convert_from_bytes(
            data, dpi=dpi, first_page=first_page, last_page=last_page)
    except Exception as exc:
        message = str(exc).lower()
        if "poppler" in message or "pdftoppm" in message or "pdfinfo" in message:
            raise DocumentError(
                "OCR page rendering needs the poppler binaries "
                f"({_POPPLER_HINT}): {exc}") from exc
        raise DocumentError(f"OCR page rendering failed: {exc}") from exc
    if not images:
        raise DocumentError("OCR found no pages in the PDF")
    if len(images) > max_pages:
        raise DocumentError(
            f"PDF page window holds {len(images)} pages, above max_pages="
            f"{max_pages}: pass a larger max_pages or a narrower "
            "first_page/last_page window")
    return images


def _auto_rotate(pytesseract: Any, image: Any, lang: str) -> Any:
    """Best-effort deskew-rotation via Tesseract OSD.

    Returns the (possibly rotated) image; never raises — when the OSD
    model is missing the page passes through untouched.
    """
    try:
        osd = pytesseract.image_to_osd(image, lang=lang)
    except Exception:  # noqa: BLE001 - OSD data often not installed
        return image
    angle = 0
    for line in osd.splitlines():
        if line.lower().startswith("orientation in degrees:"):
            try:
                angle = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                angle = 0
            break
    if angle:
        try:
            return image.rotate(angle, expand=True)
        except Exception:  # noqa: BLE001 - keep the original page
            return image
    return image


def _mean_confidence(pytesseract: Any, image: Any, lang: str) -> float:
    """Mean word confidence (0-100) for one page; -1 when unavailable."""
    try:
        data = pytesseract.image_to_data(image, lang=lang,
                                         output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 - confidence is a bonus, not a failure
        return -1.0
    confs = [float(c) for c in data.get("conf", [])
             if isinstance(c, (int, float)) and c >= 0]
    if not confs:
        return -1.0
    return sum(confs) / len(confs)


def ocr_pdf(data: bytes, *, lang: str = "eng", dpi: int = 200,
            first_page: int = 1, last_page: int | None = None,
            max_pages: int = _DEFAULT_MAX_PAGES, jobs: int = 1,
            on_page: Callable[[int, int, int], None] | None = None,
            psm: int | None = None, oem: int | None = None,
            auto_rotate: bool = False) -> Document:
    """OCR a scanned/image-only PDF into a Document (one section per page).

    ``jobs`` > 1 recognizes pages on a thread pool (OCRmyPDF-style
    multi-core; Tesseract releases the GIL in C++).  ``on_page`` is
    called as ``on_page(page_no, total_pages, chars_recognized)`` after
    each page.  ``psm``/``oem`` tune the Tesseract engine (page
    segmentation / OCR engine mode); ``auto_rotate`` deskews pages via
    Tesseract's orientation detection (best-effort: skipped when the OSD
    model is not installed).

    Raises :class:`DocumentError` when the OCR stack is missing (with an
    install hint), when the page window exceeds ``max_pages``, or when no
    page yields any text — never a silently-empty Document.
    """
    _validate_window(data, first_page, last_page, max_pages, dpi, jobs)
    config = _engine_config(psm, oem)
    pytesseract = _require_pytesseract()
    images = _render_pages(data, dpi, first_page, last_page, max_pages)

    if auto_rotate:
        images = [_auto_rotate(pytesseract, image, lang) for image in images]

    def _recognize(item: tuple[int, Any]) -> tuple[int, str, float]:
        page_no, image = item
        try:
            page_text = pytesseract.image_to_string(
                image, lang=lang, config=config)
        except Exception as exc:
            raise DocumentError(
                f"OCR failed on page {page_no}: {exc}") from exc
        page_text = page_text.strip()
        conf = _mean_confidence(pytesseract, image, lang) if page_text else -1.0
        return page_no, page_text, conf

    indexed = [(first_page + offset, image)
               for offset, image in enumerate(images)]
    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_recognize, indexed))
    else:
        results = [_recognize(item) for item in indexed]
    results.sort(key=lambda r: r[0])

    doc = new_document(format="pdf", title="", source="")
    doc.metadata["ocr"] = True
    doc.metadata["ocr_lang"] = lang
    doc.metadata["ocr_dpi"] = dpi
    if psm is not None:
        doc.metadata["ocr_psm"] = psm
    if oem is not None:
        doc.metadata["ocr_oem"] = oem
    confidences: list[float] = []
    for page_no, page_text, conf in results:
        if on_page is not None:
            try:
                on_page(page_no, len(results), len(page_text))
            except Exception:  # noqa: BLE001 - callback must not break OCR
                _log.debug("ocr on_page callback failed", exc_info=True)
        if page_text:
            if conf >= 0:
                confidences.append(conf)
            doc.sections.append(Section(level=1, heading=f"Page {page_no}",
                                        text=page_text, page=page_no))
    if not doc.sections:
        raise DocumentError(
            "OCR extracted no text from any page (blank pages, or the "
            "wrong --lang)")
    doc.metadata["pages"] = len(doc.sections)
    if confidences:
        doc.metadata["ocr_mean_confidence"] = round(
            sum(confidences) / len(confidences), 1)
    _log.info("ocr_pdf: %d page(s), lang=%s", len(doc.sections), lang)
    return doc


def ocr_pdf_words(data: bytes, *, lang: str = "eng", dpi: int = 200,
                  first_page: int = 1, last_page: int | None = None,
                  max_pages: int = _DEFAULT_MAX_PAGES,
                  psm: int | None = None, oem: int | None = None,
                  min_confidence: float = 0.0) -> list[dict[str, Any]]:
    """Word-level OCR: [{page, text, confidence, bbox}].

    The Surya-style structured view — every recognized word with its
    confidence (0-100, -1 when unavailable) and pixel ``bbox``
    ``(x0, y0, x1, y1)`` at the render DPI.  Words below
    ``min_confidence`` are dropped.  Raises :class:`DocumentError` like
    :func:`ocr_pdf` when the stack is missing or nothing is recognized.
    """
    _validate_window(data, first_page, last_page, max_pages, dpi)
    config = _engine_config(psm, oem)
    pytesseract = _require_pytesseract()
    images = _render_pages(data, dpi, first_page, last_page, max_pages)
    words: list[dict[str, Any]] = []
    for offset, image in enumerate(images):
        page_no = first_page + offset
        try:
            data_out = pytesseract.image_to_data(
                image, lang=lang, config=config,
                output_type=pytesseract.Output.DICT)
        except Exception as exc:
            raise DocumentError(
                f"OCR failed on page {page_no}: {exc}") from exc
        texts = data_out.get("text", [])
        confs = data_out.get("conf", [])
        for i, text in enumerate(texts):
            word = str(text or "").strip()
            if not word:
                continue
            try:
                conf = float(confs[i])
            except (IndexError, TypeError, ValueError):
                conf = -1.0
            if conf >= 0 and conf < min_confidence:
                continue
            try:
                bbox = (int(data_out["left"][i]), int(data_out["top"][i]),
                        int(data_out["left"][i]) + int(data_out["width"][i]),
                        int(data_out["top"][i]) + int(data_out["height"][i]))
            except (IndexError, KeyError, TypeError, ValueError):
                bbox = (0, 0, 0, 0)
            words.append({"page": page_no, "text": word,
                          "confidence": round(conf, 1), "bbox": bbox})
    if not words:
        raise DocumentError("OCR recognized no words on any page")
    return words


def ocr_pdf_hocr(data: bytes, *, lang: str = "eng", dpi: int = 200,
                 first_page: int = 1, last_page: int | None = None,
                 max_pages: int = _DEFAULT_MAX_PAGES,
                 psm: int | None = None, oem: int | None = None) -> str:
    """OCR a scanned PDF to hOCR (HTML with word boxes + confidences).

    hOCR is the interoperable structured-OCR format (used by OCRmyPDF,
    Tesseract tooling, and archival pipelines).  Returns one hOCR
    document string.  Raises :class:`DocumentError` like :func:`ocr_pdf`.
    """
    _validate_window(data, first_page, last_page, max_pages, dpi)
    config = _engine_config(psm, oem)
    pytesseract = _require_pytesseract()
    images = _render_pages(data, dpi, first_page, last_page, max_pages)
    parts: list[str] = []
    for offset, image in enumerate(images):
        page_no = first_page + offset
        try:
            page_hocr = pytesseract.image_to_pdf_or_hocr(
                image, lang=lang, config=config, extension="hocr")
        except Exception as exc:
            raise DocumentError(
                f"OCR failed on page {page_no}: {exc}") from exc
        if isinstance(page_hocr, bytes):
            page_hocr = page_hocr.decode("utf-8", "replace")
        parts.append(page_hocr)
    combined = "\n".join(parts)
    if "ocrx_word" not in combined and not combined.strip():
        raise DocumentError("OCR produced no hOCR output")
    return combined


def ocr_image(data: bytes, *, lang: str = "eng") -> str:
    """OCR a single image (PNG/JPEG/…) into plain text.

    Raises :class:`DocumentError` when the OCR stack is missing (with an
    install hint) or when no text is recognized.
    """
    if not data:
        raise DocumentError("cannot OCR empty input")
    pytesseract = _require_pytesseract()
    try:
        from PIL import Image
    except ImportError as exc:
        raise DocumentError(
            "ocr_image needs the optional 'Pillow' package: "
            "pip install Pillow.") from exc
    import io
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        raise DocumentError(f"ocr_image could not decode the image: {exc}") from exc
    try:
        text = pytesseract.image_to_string(image, lang=lang)
    except Exception as exc:
        raise DocumentError(f"OCR failed on the image: {exc}") from exc
    text = text.strip()
    if not text:
        raise DocumentError("OCR recognized no text in the image")
    return text
