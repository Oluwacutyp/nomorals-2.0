"""ULID-style identifiers.

Requirements for ids in this system:

* **Sortable by creation time** — the database is indexed on ids, and time-ordered
  ids keep B-tree inserts append-only (no page-split storms).
* **Globally unique without coordination** — sub-agents run in separate processes
  and must not need a central allocator.
* **Lexicographically sortable as text** — so plain SQLite ``ORDER BY id`` works.
* **Short enough to read in logs.**

Format: 26-character Crockford base32, 48-bit millisecond timestamp + 80 bits of
randomness. Identical layout to the ULID spec, implemented on the stdlib only.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from typing import Iterator

__all__ = ["ULID", "decode_time", "new_id", "new_short_id", "ulid_now", "ulid_range"]

_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODING = {c: i for i, c in enumerate(_ENCODING)}
_TIMESTAMP_LEN = 10
_RANDOM_LEN = 16


def _encode(value: int, length: int) -> str:
    out = ["0"] * length
    for i in range(length - 1, -1, -1):
        out[i] = _ENCODING[value & 0x1F]
        value >>= 5
    return "".join(out)


def _decode(text: str) -> int:
    value = 0
    for ch in text:
        idx = _DECODING.get(ch.upper())
        if idx is None:
            raise ValueError(f"invalid ULID character: {ch!r}")
        value = (value << 5) | idx
    return value


class _MonotonicRandom:
    """ULID monotonic factory.

    Within the same millisecond, the random component is incremented rather than
    regenerated. This guarantees strictly increasing ids inside a single process
    even at >4096 ids/ms, which matters when a fan-out of sub-agents stamps
    records in a tight loop.
    """

    __slots__ = ("_last_ms", "_last_random", "_lock")

    def __init__(self) -> None:
        self._last_ms = -1
        self._last_random = 0
        self._lock = threading.Lock()

    def next(self, now_ms: int | None = None) -> str:
        with self._lock:
            ms = int(now_ms if now_ms is not None else time.time() * 1000)
            if ms <= self._last_ms:
                # Clock went backwards or same ms: keep advancing.
                ms = self._last_ms
                self._last_random += 1
                if self._last_random >= 1 << 80:  # pragma: no cover - unreachable in practice
                    ms += 1
                    self._last_random = secrets.randbits(80)
            else:
                self._last_random = secrets.randbits(80)
            self._last_ms = ms
            return _encode(ms, _TIMESTAMP_LEN) + _encode(self._last_random, _RANDOM_LEN)


_factory = _MonotonicRandom()


def ulid_now(now_ms: int | None = None) -> str:
    """Generate a monotonic ULID string."""
    return _factory.next(now_ms)


# Alias that reads better at call sites.
new_id = ulid_now


def new_short_id(prefix: str = "", length: int = 12) -> str:
    """A short, human-friendly id for logs and CLI references.

    Not monotonic and not intended as a primary key — use :func:`new_id` for that.
    """
    raw = secrets.token_bytes(16).hex()[:length]
    return f"{prefix}{raw}" if prefix else raw


def decode_time(ulid: str) -> float:
    """Return the creation timestamp (unix seconds) encoded in a ULID."""
    if len(ulid) != 26:
        raise ValueError(f"not a ULID (expected 26 chars, got {len(ulid)})")
    return _decode(ulid[:_TIMESTAMP_LEN]) / 1000.0


def ulid_range(start_ms: int, end_ms: int) -> tuple[str, str]:
    """Lexicographic bounds covering every ULID created in ``[start_ms, end_ms)``.

    Useful for range scans without indexing a separate timestamp column::

        lo, hi = ulid_range(day_start, day_end)
        rows = db.query("SELECT * FROM t WHERE id >= ? AND id < ?", (lo, hi))
    """
    low_random = "0" * _RANDOM_LEN
    high_random = "Z" * _RANDOM_LEN
    return (
        _encode(start_ms, _TIMESTAMP_LEN) + low_random,
        _encode(end_ms, _TIMESTAMP_LEN) + high_random,
    )


def ulid_batch(count: int) -> Iterator[str]:
    """Yield ``count`` strictly increasing ULIDs."""
    for _ in range(count):
        yield _factory.next()


class ULID:
    """Object wrapper, mostly for typing and debugging convenience."""

    __slots__ = ("raw",)

    def __init__(self, raw: str) -> None:
        if len(raw) != 26:
            raise ValueError(f"not a ULID: {raw!r}")
        self.raw = raw.upper()

    @classmethod
    def generate(cls) -> "ULID":
        return cls(ulid_now())

    @property
    def datetime_ms(self) -> int:
        return _decode(self.raw[:_TIMESTAMP_LEN])

    @property
    def entropy(self) -> int:
        return _decode(self.raw[_TIMESTAMP_LEN:])

    @property
    def timestamp(self) -> float:
        return self.datetime_ms / 1000.0

    def __str__(self) -> str:
        return self.raw

    def __repr__(self) -> str:
        return f"ULID({self.raw!r})"

    def __lt__(self, other: "ULID") -> bool:
        return self.raw < other.raw

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ULID) and self.raw == other.raw

    def __hash__(self) -> int:
        return hash(self.raw)


def random_bytes(n: int) -> bytes:
    """Cryptographically secure random bytes (re-exported for convenience)."""
    return secrets.token_bytes(n)


def random_token(urlsafe: bool = True) -> str:
    """A 32-byte random token, for API keys and confirmation tokens."""
    raw = os.urandom(32)
    if urlsafe:
        import base64

        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return raw.hex()
