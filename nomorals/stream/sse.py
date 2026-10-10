"""Server-Sent Events wire framing (WHATWG ``text/event-stream`` rules).

A tiny, dependency-free equivalent of ``sse-starlette``'s
``ServerSentEvent``: every event is ``event:``/``data:``/``id:``/``retry:``
lines terminated by a blank line, lines starting with ``:`` are comments
(ignored by ``EventSource`` clients).

Framing rules enforced here:
- comments and multi-line data are split on line boundaries — each line
  gets its own ``: `` / ``data: `` prefix;
- ``id:`` and ``event:`` values must not contain line breaks (they are
  stripped — a broken id would corrupt the client's ``Last-Event-ID``
  resume bookkeeping);
- ``retry`` must be an int (milliseconds);
- the line separator must be one of ``\\r\\n`` / ``\\r`` / ``\\n``.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["DEFAULT_SEP", "ServerSentEvent", "format_sse"]

_LINE_SPLIT = re.compile(r"\r\n|\r|\n")
_VALID_SEPS = ("\r\n", "\r", "\n")

#: Default line separator. ``\n`` (not ``\r\n``) preserves this package's
#: historical wire format — the WHATWG SSE spec accepts ``\r\n`` / ``\r`` /
#: ``\n`` interchangeably.
DEFAULT_SEP = "\n"


class ServerSentEvent:
    """One SSE event, formatted on :meth:`encode`.

    ``data`` may be any object — it is stringified. ``comment`` produces
    client-invisible ``: `` lines (the standard heartbeat shape).
    """

    def __init__(
        self,
        data: Any = None,
        *,
        event: str | None = None,
        id: str | None = None,
        retry: int | None = None,
        comment: str | None = None,
        sep: str = DEFAULT_SEP,
    ) -> None:
        if sep not in _VALID_SEPS:
            raise ValueError(
                f"sep must be one of \\r\\n, \\r, \\n — got {sep!r}")
        if retry is not None and not isinstance(retry, int):
            raise TypeError("retry must be an int (milliseconds)")
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry
        self.comment = comment
        self._sep = sep

    def encode(self) -> bytes:
        """Format the event as UTF-8 bytes, blank-line terminated."""
        lines: list[str] = []
        if self.comment is not None:
            for chunk in _LINE_SPLIT.split(str(self.comment)):
                lines.append(f": {chunk}")
        if self.event is not None:
            lines.append("event: " + _LINE_SPLIT.sub("", str(self.event)))
        if self.id is not None:
            # A newline inside an id would desync Last-Event-ID resume.
            lines.append("id: " + _LINE_SPLIT.sub("", str(self.id)))
        if self.data is not None:
            for chunk in _LINE_SPLIT.split(str(self.data)):
                lines.append(f"data: {chunk}")
        if self.retry is not None:
            lines.append(f"retry: {self.retry}")
        return (self._sep.join(lines) + self._sep * 2).encode("utf-8")


def format_sse(data: Any = None, **kwargs: Any) -> bytes:
    """Format one SSE event. Shortcut for ``ServerSentEvent(...).encode()``."""
    return ServerSentEvent(data, **kwargs).encode()
