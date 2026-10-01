"""OCR provider: real, offline "vision" through Tesseract.

Every other vision path asks a *model* to look at an image. This one does the
looking itself: the image bytes are fed to the ``tesseract`` binary and the
recognized text is the answer. It is the floor of the vision chain — no API
key, no model weights, no network — so a phone with ``pkg install tesseract``
can read any screen (a chat, a terminal, a form) even fully offline.

It plugs in as an ordinary :class:`LLMProvider` with the ``vision``
capability, so the router's normal failover handles it: API vision first,
OCR when every model is down (or when you deliberately want the verbatim
text, which is what a screen reader most of the time).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Sequence

from ...core.errors import ProviderError
from ...core.logging_setup import get_logger
from ..base import LLMProvider, LLMResponse, Message, SamplingParams

_log = get_logger(__name__)

__all__ = ["OCRProvider", "ocr_binary", "ocr_bytes"]

#: mime → tesseract-readable extension (it reads png/jpg/bmp/tiff/webp natively)
_MIME_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


def ocr_binary(explicit: str = "") -> str | None:
    """Locate the tesseract binary: explicit path > NM_OCR_BINARY > PATH."""
    for candidate in (explicit, os.environ.get("NM_OCR_BINARY", "")):
        candidate = (candidate or "").strip()
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("tesseract")


def ocr_bytes(
    data: bytes,
    *,
    mime: str = "image/png",
    language: str = "eng",
    binary: str = "",
    timeout: float = 60.0,
) -> str:
    """Run tesseract over raw image bytes. Returns the recognized text.

    Raises :class:`ProviderError` with an install hint when tesseract is not
    available, so callers can degrade to "OCR unavailable" instead of crashing.
    """
    if not data:
        raise ProviderError("no image data to OCR")
    exe = ocr_binary(binary)
    if not exe:
        raise ProviderError(
            "tesseract is not installed — Termux: pkg install tesseract "
            "(or point NM_VISION_OCR_BINARY at a tesseract build)"
        )
    suffix = _MIME_SUFFIX.get((mime or "").lower(), ".png")
    tmp = tempfile.NamedTemporaryFile(prefix="nm-ocr-", suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        argv = [exe, tmp.name, "stdout"]
        lang = (language or "eng").strip()
        if lang and lang != "":
            argv += ["-l", lang]
        proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
        text = (proc.stdout or b"").decode("utf-8", "replace").strip()
        if proc.returncode != 0 and not text:
            stderr = (proc.stderr or b"").decode("utf-8", "replace")[:300]
            raise ProviderError(f"tesseract failed (exit {proc.returncode}): {stderr}")
        return text
    except subprocess.TimeoutExpired as exc:
        raise ProviderError(f"tesseract timed out after {timeout:.0f}s") from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:  # noqa: E103 - temp cleanup is best-effort
            pass


class OCRProvider(LLMProvider):
    """The offline vision floor. ``describe_image`` = tesseract on the bytes."""

    name = "ocr"

    def __init__(
        self,
        *,
        model: str = "tesseract",
        language: str = "eng",
        binary: str = "",
        timeout: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.model = model
        self.language = (language or "eng").strip()
        self.binary = binary

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def capabilities(self) -> set[str]:
        return {"vision"}

    def health(self) -> bool:
        return ocr_binary(self.binary) is not None

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        raise ProviderError("the ocr provider reads images; it does not chat")

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        mime = kw.pop("mime", "image/png")
        try:
            text = ocr_bytes(image, mime=mime, language=self.language, binary=self.binary,
                             timeout=self.timeout)
        except ProviderError as exc:
            return self._record(LLMResponse(text="", model=self.model, error=exc.message),
                                started)
        return self._record(LLMResponse(text=text, model=self.model, provider=self.name,
                                        raw={"ocr": True}), started)
