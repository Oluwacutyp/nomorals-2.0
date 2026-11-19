"""Texture errors (TextureCraft port, census 3A.15).

Upstream TextureCraft never crashes on bad input: a malformed project,
a bad command or an odd parameter produces an error the user or agent
can act on, never a panic (upstream AGENTS.md §3 "Never crash").
Every failure in this organ raises :class:`TextureError` naming the
step that failed, in the same shape as the other craft organs'
``*Error`` / ``*ErrorKind`` pairs.
"""
from __future__ import annotations


class TextureErrorKind(str):
    """Stable kinds for texture failures."""

    BAD_FILE = "bad_file"
    BAD_FORMAT = "bad_format"
    UNSUPPORTED = "unsupported"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    TOO_LARGE = "too_large"
    RENDER_FAILED = "render_failed"
    OTHER = "other"


class TextureError(Exception):
    """A texture operation failed. ``kind`` names the step; ``detail``
    carries the actionable cause."""

    def __init__(self, kind: str, detail: str = "") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)
