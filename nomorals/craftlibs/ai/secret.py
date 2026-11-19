"""Secrets (Craft-Libs port of ai-core ``secret.rs``).

``SecretString`` is a secret value that cannot be printed or
serialized by accident: ``repr`` and ``str`` print a placeholder,
and :func:`redact` scrubs a known secret out of text such as a
provider error message before it is shown or logged.
"""

from __future__ import annotations

__all__ = ["SecretString", "redact"]

_MARKER = "[redacted]"


class SecretString:
    """A secret value (API key, token)."""

    def __init__(self, value: str) -> None:
        self._value = str(value)

    @classmethod
    def new(cls, value: str) -> SecretString:
        return cls(value)

    def expose(self) -> str:
        """The raw value. Call this only when building the
        outgoing request."""
        return self._value

    def is_empty(self) -> bool:
        return not self._value

    def __repr__(self) -> str:
        return "SecretString(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"


def redact(text: str, secrets) -> str:
    """Replace every occurrence of each non-empty secret in
    ``text`` with ``[redacted]``. Matches are found in ``text``
    itself, never in the markers already written, so the output
    is at most a fixed multiple of the input whatever the secrets
    are. Overlapping matches of different secrets become one
    marker."""
    values = sorted({s.expose() for s in secrets if not s.is_empty()})
    spans: list[tuple[int, int]] = []
    for value in values:
        start = 0
        while True:
            at = text.find(value, start)
            if at < 0:
                break
            spans.append((at, at + len(value)))
            start = at + 1
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[str] = []
    done = 0
    for start, end in merged:
        out.append(text[done:start])
        out.append(_MARKER)
        done = end
    out.append(text[done:])
    return "".join(out)
