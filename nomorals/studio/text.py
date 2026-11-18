"""Text a person will read (names, descriptions) from an untrusted package.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
text.rs``, dual MIT/Apache): control characters, bidi overrides and
zero-width characters are refused, because they let a name read
differently from what it is. Joiners stay allowed so scripts that
need them work.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Lines", "TextError", "validate_text"]


class Lines(Enum):
    """Whether line breaks are allowed."""

    ONE = "one"
    MANY = "many"


class TextError(Enum):
    """Why a piece of text was refused."""

    EMPTY = "the text is empty"
    TOO_LONG = "the text is too long"
    CONTROL = "the text contains a control character"
    INVISIBLE = "the text contains invisible formatting characters"
    NOTHING_VISIBLE = "the text has no visible characters"
    NEWLINE = "the text must be on one line"


#: Characters that change how text is drawn or read without being
#: visible themselves. The joiner and non-joiner (U+200C, U+200D)
#: are deliberately absent: some scripts cannot be written without
#: them.
_INVISIBLE = frozenset(
    ["\u00ad", "\u061c", "\u180e", "\u200b", "\u200e", "\u200f",
     "\ufeff"]
    + [chr(c) for c in range(0x202A, 0x202F)]
    + [chr(c) for c in range(0x2060, 0x2065)]
    + [chr(c) for c in range(0x2066, 0x206A)])


def validate_text(text: str, max_chars: int, lines: Lines
                  ) -> TextError | None:
    """Check human-readable text; the refusal reason, or None."""
    if not text:
        return TextError.EMPTY
    if len(text) > max_chars:
        return TextError.TOO_LONG
    for ch in text:
        if ch in ("\n", "\r"):
            if lines is Lines.ONE:
                return TextError.NEWLINE
        elif ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F:
            # Unicode control characters (Cc), tab included.
            return TextError.CONTROL
        elif ch in _INVISIBLE:
            return TextError.INVISIBLE
    if all(ch.isspace() for ch in text):
        return TextError.NOTHING_VISIBLE
    return None
