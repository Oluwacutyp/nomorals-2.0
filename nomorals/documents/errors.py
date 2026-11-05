"""Document engine errors.

Wave 2 (Echelon documents port): :class:`DocumentError` joins the
``NoMoralsError`` code system — every raise carries a stable
machine-readable ``code`` plus a ``kind`` mirrored into ``details``
(the Echelon standard: machine-readable kinds separate from human
messages), while remaining a ``ValueError`` so every pre-existing
``except DocumentError`` / ``except ValueError`` / ``except Exception``
handler keeps working unchanged.  The fold is purely additive.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import NoMoralsError

__all__ = ["DocumentError", "DOCUMENT_ERROR_KINDS"]


#: The closed taxonomy of document failure kinds.  ``kind`` is the
#: machine-readable half of the error; the human message stays free
#: text.  Kinds are assigned at raise sites across the documents
#: package; an unclassified raise keeps ``kind="error"``.
DOCUMENT_ERROR_KINDS = frozenset({
    "error",               # unclassified (legacy raise sites)
    "parse",               # input bytes could not be parsed
    "unsupported_format",  # format not recognized / not supported
    "missing_dependency",  # optional package or binary absent
    "empty",               # document has no content for the operation
    "invalid_argument",    # caller passed a bad argument
    "io",                  # file read/write failure
    "write",               # a writer failed to produce its output
    "ocr_unavailable",     # OCR stack absent (tesseract / pytesseract)
    "upstream",            # a backing engine/service the operation depends
                           # on (LLM brain, search backend, index storage)
                           # was present but failed or vanished at runtime —
                           # distinct from missing_dependency (absent)
})


class DocumentError(NoMoralsError, ValueError):
    """Raised when a document cannot be parsed, converted, or written.

    Carries a human-readable reason (callers should surface
    ``str(exc)``), a stable ``code`` (default ``"document.error"``,
    refinable per raise), and a machine-readable ``kind`` from
    :data:`DOCUMENT_ERROR_KINDS`, mirrored into ``details["kind"]``.
    """

    code = "document.error"

    def __init__(
        self,
        message: str = "",
        *,
        kind: str = "error",
        code: str | None = None,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        if kind not in DOCUMENT_ERROR_KINDS:
            # the taxonomy is closed by design — a typo'd kind must
            # fail here, at the raise, not downstream where a consumer
            # switches on it
            raise ValueError(
                f"unknown document error kind {kind!r}; expected one "
                f"of {sorted(DOCUMENT_ERROR_KINDS)}")
        merged: dict[str, Any] = dict(details or {})
        merged.setdefault("kind", kind)
        super().__init__(message, code=code, details=merged,
                         retryable=retryable)
        self.kind = kind
