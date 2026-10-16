"""Progress throttling + long-output-as-document for the Telegram bot.

Two behaviours that used to spam chats:

* Progress edits: a long job edited its message on every tick, hammering
  ``editMessageText`` (and Telegram's rate limits). The throttle allows
  an edit at most once per ``min_interval_s`` (default ~3s) and only
  when the text actually changed — a repeated identical edit is never
  sent.
* Long output: a 40-chunk reply is unreadable and eats the chat. Output
  past ``max_chunks`` chunks goes out as ONE document (``.txt``) with a
  short caption instead — the reader gets the whole thing, the chat
  stays clean.

Both are pure (no network): the adapter owns the actual API calls and
uses these to decide *whether* and *how* to send.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .platform_limits import for_platform

__all__ = [
    "ProgressThrottler",
    "LongOutput",
    "split_or_document",
    "DEFAULT_EDIT_INTERVAL_S",
    "DEFAULT_MAX_CHUNKS",
    "DOCUMENT_SUFFIX",
]

#: Edit-in-place cadence: ~1 edit per 3 seconds, only on change.
DEFAULT_EDIT_INTERVAL_S = 3.0
#: Past this many 4096-char chunks, the output goes out as a document.
DEFAULT_MAX_CHUNKS = 10
#: Document suffix for long output.
DOCUMENT_SUFFIX = ".txt"


class ProgressThrottler:
    """Decide whether a progress edit is worth sending right now.

    Invariant: ``should_edit`` returns True only when the text changed
    AND ``min_interval_s`` elapsed since the last allowed edit. The
    first edit for a new text always passes (no spam, no starvation).
    """

    def __init__(self, min_interval_s: float = DEFAULT_EDIT_INTERVAL_S,
                 clock: Callable[[], float] | None = None) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._clock = clock or time.monotonic
        self._last_text: str | None = None
        self._last_at: float = 0.0
        self.edits_allowed = 0
        self.edits_suppressed = 0

    def should_edit(self, new_text: str) -> bool:
        """True iff this edit should be sent (changed + interval elapsed)."""
        if new_text == self._last_text:
            self.edits_suppressed += 1
            return False
        now = self._clock()
        if self._last_text is not None and (now - self._last_at) < self.min_interval_s:
            self.edits_suppressed += 1
            return False
        self._last_text = new_text
        self._last_at = now
        self.edits_allowed += 1
        return True

    def reset(self) -> None:
        self._last_text = None
        self._last_at = 0.0


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split on newlines (same rule as the adapter's chunker)."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    if rest:
        chunks.append(rest)
    return chunks


class LongOutput:
    """Route long output: chunks for short text, one document for long."""

    def __init__(self, *, chunk_limit: int | None = None,
                 max_chunks: int = DEFAULT_MAX_CHUNKS) -> None:
        # Default chunk bound is the Telegram PlatformLimits cap.
        self.chunk_limit = int(chunk_limit) if chunk_limit is not None \
            else for_platform("telegram").max_text
        self.max_chunks = int(max_chunks)

    def route(self, text: str) -> tuple[str, Any]:
        """``("chunks", [str, ...])`` or ``("document", bytes)``.

        Invariant: the document path is taken iff the text would need
        more than ``max_chunks`` message chunks — never for short text,
        always for very long text. The document bytes are the full
        original text, UTF-8, nothing truncated.
        """
        chunks = _chunk_text(text or "", self.chunk_limit)
        if len(chunks) <= self.max_chunks:
            return ("chunks", chunks)
        return ("document", (text or "").encode("utf-8"))

    def document_name(self, stem: str = "output") -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stem) or "output"
        return f"{safe}{DOCUMENT_SUFFIX}"

    def caption(self, text: str) -> str:
        n = len(text or "")
        return f"📄 long output ({n:,} chars) — full text attached as a document"


def split_or_document(text: str, *, chunk_limit: int | None = None,
                      max_chunks: int = DEFAULT_MAX_CHUNKS) -> tuple[str, Any]:
    """One-shot form of :meth:`LongOutput.route`."""
    return LongOutput(chunk_limit=chunk_limit, max_chunks=max_chunks).route(text)
