"""OCR for scanned/image-only PDFs and images (L4).

Text extraction (:func:`nomorals.documents.parse_bytes` on a PDF) returns
nothing useful for scanned pages — they are pictures of text.  This module
bridges that gap with Tesseract via ``pytesseract`` (page rendering via
``pdf2image``/poppler).

Both heavy pieces are *optional*: when they are missing the functions
below fail fast with :class:`DocumentError` carrying an exact install
hint — they never silently return empty text.

Install hints (Debian/Ubuntu; other platforms need the same two binaries):

* ``pip install pytesseract pdf2image``
* ``sudo apt install tesseract-ocr poppler-utils``
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from .errors import DocumentError
from .model import Document, Section, new_document

__all__ = ["ocr_available", "ocr_image", "ocr_pdf"]

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
            max_pages: int = _DEFAULT_MAX_PAGES) -> Document:
    """OCR a scanned/image-only PDF into a Document (one section per page).

    Raises :class:`DocumentError` when the OCR stack is missing (with an
    install hint), when the page window exceeds ``max_pages``, or when no
    page yields any text — never a silently-empty Document.
    """
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
    pytesseract = _require_pytesseract()
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
    if dpi < 72:
        raise DocumentError(f"dpi {dpi} is too low for OCR (minimum 72)")

    doc = new_document(format="pdf", title="", source="")
    doc.metadata["ocr"] = True
    doc.metadata["ocr_lang"] = lang
    doc.metadata["ocr_dpi"] = dpi
    confidences: list[float] = []
    for offset, image in enumerate(images):
        page_no = first_page + offset
        try:
            page_text = pytesseract.image_to_string(image, lang=lang)
        except Exception as exc:
            raise DocumentError(
                f"OCR failed on page {page_no}: {exc}") from exc
        page_text = page_text.strip()
        if page_text:
            conf = _mean_confidence(pytesseract, image, lang)
            if conf >= 0:
                confidences.append(conf)
            doc.sections.append(Section(level=1, heading=f"Page {page_no}",
                                        text=page_text))
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
