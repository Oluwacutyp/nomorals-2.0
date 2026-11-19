"""Payloads (Craft-Libs port of ai-core ``payload.rs``).

A payload is the data a request carries: text, inline bytes, or a
handle — a reference to data that stays where it is until a
provider adapter streams it. Inline data is capped by
:class:`Limits`; a handle is never capped, because the bytes were
never copied here.
"""

from __future__ import annotations

import base64 as _base64
import binascii
import re
from enum import Enum

from .error import AiError

__all__ = [
    "DEFAULT_MAX_INLINE_BYTES",
    "DEFAULT_MAX_TEXT_BYTES",
    "Limits",
    "Payload",
    "PayloadKind",
    "base64_decode",
    "base64_encode",
    "validate_media_type",
]

DEFAULT_MAX_INLINE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_TEXT_BYTES = 4 * 1024 * 1024
MAX_URI_LEN = 4096

_MEDIA_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+\-]*"
                            r"/[a-z0-9][a-z0-9!#$&^_.+\-]*$")


def validate_media_type(media_type: str) -> None:
    """A media type is ``type/subtype`` in lowercase, no spaces,
    no parameters. Raises :class:`AiError` on anything else."""
    if not _MEDIA_TYPE_RE.match(str(media_type)):
        raise AiError.invalid(f"{media_type!r} is not a media type")


def base64_encode(data: bytes) -> str:
    """Standard base64, no line breaks."""
    return _base64.b64encode(bytes(data)).decode("ascii")


def base64_decode(text: str) -> bytes | None:
    """Strict base64: returns ``None`` on anything malformed
    (bad length, bad padding, bad alphabet, whitespace)."""
    if not isinstance(text, str):
        return None
    if text == "":
        return b""
    if len(text) % 4 != 0:
        return None
    if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", text):
        return None
    if "=" in text[:-2]:
        return None
    try:
        return _base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        return None


class PayloadKind(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    OTHER = "other"


def _kind_for(media_type: str) -> PayloadKind:
    top, _, sub = str(media_type).partition("/")
    if top == "image":
        return PayloadKind.IMAGE
    if top == "video":
        return PayloadKind.VIDEO
    if top == "audio":
        return PayloadKind.AUDIO
    if top == "text":
        return PayloadKind.TEXT
    if top == "application" and sub == "pdf":
        return PayloadKind.DOCUMENT
    return PayloadKind.OTHER


class Limits:
    """Caps on what one request may carry inline."""

    def __init__(self, max_inline_bytes: int = DEFAULT_MAX_INLINE_BYTES,
                 max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES) -> None:
        self.max_inline_bytes = max_inline_bytes
        self.max_text_bytes = max_text_bytes


class Payload:
    """One input value: text, inline bytes, or a handle."""

    def __init__(self, kind: str, media_type: str | None = None,
                 data: bytes | None = None, text: str | None = None,
                 uri: str | None = None,
                 size_hint: int | None = None) -> None:
        self._kind = kind
        self.media_type = media_type
        self.data = data
        self.text = text
        self.uri = uri
        self._size_hint = size_hint

    @classmethod
    def inline(cls, media_type: str, data: bytes) -> Payload:
        validate_media_type(media_type)
        return cls("inline", media_type=media_type, data=bytes(data))

    @classmethod
    def text(cls, text: str) -> Payload:
        return cls("text", media_type="text/plain", text=str(text))

    @classmethod
    def handle(cls, uri: str, media_type: str,
               size_hint: int | None = None) -> Payload:
        validate_media_type(media_type)
        if not uri or len(uri) > MAX_URI_LEN:
            raise AiError.invalid("a handle needs a URI of 1 to "
                                  f"{MAX_URI_LEN} characters")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in uri):
            raise AiError.invalid("a handle URI holds no control characters")
        return cls("handle", media_type=media_type, uri=uri,
                   size_hint=size_hint)

    def kind(self) -> PayloadKind:
        if self._kind == "text":
            return PayloadKind.TEXT
        return _kind_for(self.media_type or "")

    def size_hint(self) -> int | None:
        if self._kind == "handle":
            return self._size_hint
        if self._kind == "inline":
            return len(self.data or b"")
        if self._kind == "text":
            return len((self.text or "").encode("utf-8"))
        return None

    def as_text(self) -> str | None:
        """The text, when this payload is text; ``None``
        otherwise."""
        if self._kind == "text":
            return self.text
        return None

    def check(self, limits: Limits) -> None:
        """Refuse an over-limit payload. An inline payload that is
        too big names the handle as the way out."""
        if (self._kind == "inline"
                and len(self.data or b"") > limits.max_inline_bytes):
            raise AiError.invalid(
                f"inline payload is {len(self.data or b'')} bytes "
                f"(max {limits.max_inline_bytes}); send it as a "
                "handle instead")
        elif (self._kind == "text"
                and len((self.text or "").encode("utf-8"))
                > limits.max_text_bytes):
            raise AiError.invalid(
                f"text payload is over {limits.max_text_bytes} bytes")
        # Handles skip the inline limit: the bytes were never copied.

    def to_dict(self) -> dict:
        if self._kind == "inline":
            return {"type": "inline", "media_type": self.media_type,
                    "data": base64_encode(self.data or b"")}
        if self._kind == "text":
            return {"type": "text", "text": self.text}
        data: dict = {"type": "handle", "uri": self.uri,
                      "media_type": self.media_type}
        if self._size_hint is not None:
            data["size_hint"] = self._size_hint
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Payload:
        kind = data.get("type")
        if kind == "inline":
            raw = base64_decode(str(data.get("data", "")))
            if raw is None:
                raise ValueError("inline payload data is not valid base64")
            return cls.inline(str(data.get("media_type", "")), raw)
        if kind == "text":
            return cls.text(str(data.get("text", "")))
        if kind == "handle":
            return cls.handle(str(data.get("uri", "")),
                              str(data.get("media_type", "")),
                              data.get("size_hint"))
        raise ValueError(f"unknown payload type {kind!r}")

    def __eq__(self, other) -> bool:
        return (isinstance(other, Payload)
                and self.to_dict() == other.to_dict())

    def __repr__(self) -> str:
        return f"Payload({self.to_dict()!r})"
