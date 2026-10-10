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
from typing import Iterable, Iterator, NamedTuple

__all__ = ["ULID", "decode_time", "is_ulid", "new_id", "new_short_id", "ulid_at",
           "ulid_now", "ulid_range", "PrefixResolution", "resolve_id_prefix",
           "min_unique_prefix_len"]

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


def new_id(prefix: str = "") -> str:
    """Generate a ULID, optionally with a human-readable prefix.

    Examples:
        new_id()        → "01JFX..."  (plain 26-char ULID)
        new_id("task")  → "task_01JFX..."
        new_id("cron")  → "cron_01JFX..."
    """
    raw = _factory.next()
    return f"{prefix}_{raw}" if prefix else raw


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


def is_ulid(text: str) -> bool:
    """Strict ULID validation per the spec.

    26 chars, Crockford base32 (case-insensitive, no I/L/O/U), and the
    first character must be ``0``–``7`` — anything larger would overflow
    128 bits (the spec's maximum is ``7ZZ…Z``). The classic interop bug is
    accepting a 26-char string that isn't a ULID; this rejects it.
    """
    if not isinstance(text, str) or len(text) != 26:
        return False
    if text[0].upper() not in "01234567":
        return False
    return all(ch.upper() in _DECODING for ch in text)


def ulid_at(when_ms: int) -> str:
    """A ULID stamped at an explicit millisecond timestamp.

    Deterministic given the ms (random part is still fresh entropy) —
    for backfills, migrations, and tests that need ids "from" a moment.
    """
    return _encode(int(when_ms), _TIMESTAMP_LEN) + _encode(
        secrets.randbits(80), _RANDOM_LEN)


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

    @classmethod
    def at(cls, when_ms: int) -> "ULID":
        """A ULID stamped at an explicit millisecond timestamp."""
        return cls(ulid_at(when_ms))

    def next(self) -> "ULID":
        """The next monotonic ULID after this one (same ms → entropy + 1).

        The spec's monotonicity primitive, exposed for id sequences that
        must stay ordered without touching the clock.
        """
        entropy = (self.entropy + 1) % (1 << 80)
        return ULID(_encode(self.datetime_ms, _TIMESTAMP_LEN)
                    + _encode(entropy, _RANDOM_LEN))

    def to_bytes(self) -> bytes:
        """The 128-bit value as 16 big-endian bytes (spec binary layout)."""
        return (_decode(self.raw[:_TIMESTAMP_LEN]) << 80 | self.entropy).to_bytes(16, "big")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ULID":
        """Build from 16 big-endian bytes (inverse of :meth:`to_bytes`)."""
        if len(raw) != 16:
            raise ValueError(f"ULID needs 16 bytes, got {len(raw)}")
        value = int.from_bytes(raw, "big")
        return cls(_encode(value >> 80, _TIMESTAMP_LEN)
                   + _encode(value & ((1 << 80) - 1), _RANDOM_LEN))

    def to_uuid(self) -> str:
        """The 128-bit value as a canonical UUID string."""
        import uuid as _uuid

        return str(_uuid.UUID(bytes=self.to_bytes()))

    @classmethod
    def from_uuid(cls, value: str) -> "ULID":
        """Build from a UUID string (inverse of :meth:`to_uuid`)."""
        import uuid as _uuid

        return cls.from_bytes(_uuid.UUID(str(value)).bytes)

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


# ── short-prefix resolution ────────────────────────────────────────────────
#
# Every chat/CLI resolver that accepts a short id prefix funnels through
# :func:`resolve_id_prefix` so the contract is identical everywhere:
#
# * empty / whitespace-only reference → ``"empty"`` (never match-all)
# * exact full-id match → ``"exact"`` — wins even when the id is also a
#   prefix of another id
# * prefix matching exactly one id → ``"unique"``
# * prefix matching nothing → ``"none"``
# * prefix matching two or more ids → ``"ambiguous"`` — callers must ask
#   the user to disambiguate; they must NEVER pick the first match.
#
# Matching is case-insensitive by default: ULIDs are uppercase Crockford
# but humans type lowercase prefixes.


class PrefixResolution(NamedTuple):
    """Outcome of :func:`resolve_id_prefix`."""

    outcome: str  # "empty" | "none" | "exact" | "unique" | "ambiguous"
    matches: tuple[str, ...]  # matched ids, in candidate order
    min_unique_len: int  # "ambiguous" only: smallest L such that every
    # matched id's first L characters identify it uniquely


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def min_unique_prefix_len(ids: Iterable[str]) -> int:
    """Smallest L such that every id's first L chars are unique among *ids*.

    Returns 0 for fewer than two ids. Duplicate ids can never be
    disambiguated by length, so the answer is capped at the id length —
    callers should then ask for the full id.
    """
    lowered = [str(i).lower() for i in ids]
    if len(lowered) < 2:
        return 0
    best = 0
    for i, a in enumerate(lowered):
        need = 0
        for j, b in enumerate(lowered):
            if i != j:
                need = max(need, _common_prefix_len(a, b) + 1)
        best = max(best, min(need, len(a)))
    return best


def resolve_id_prefix(
    ref: str,
    ids: Iterable[str],
    *,
    case_insensitive: bool = True,
) -> PrefixResolution:
    """Classify a short id reference against candidate ids.

    ``ids`` may contain duplicates; they are de-duplicated preserving order.
    """
    filt = (ref or "").strip()
    if not filt:
        return PrefixResolution("empty", (), 0)
    seen: set[str] = set()
    candidates: list[str] = []
    for cand in ids:
        c = str(cand)
        if c not in seen:
            seen.add(c)
            candidates.append(c)
    if case_insensitive:
        norm = str.lower
    else:
        norm = lambda s: s  # noqa: E731
    want = norm(filt)
    # exact full-id match wins — even when it is also a prefix of another id
    for c in candidates:
        if norm(c) == want:
            return PrefixResolution("exact", (c,), 0)
    matches = tuple(c for c in candidates if norm(c).startswith(want))
    if not matches:
        return PrefixResolution("none", (), 0)
    if len(matches) == 1:
        return PrefixResolution("unique", matches, 0)
    return PrefixResolution(
        "ambiguous", matches, min_unique_prefix_len(matches))
