"""Version metadata."""

from __future__ import annotations

from typing import NamedTuple

__version__ = "0.1.0"


class _VersionInfo(NamedTuple):
    major: int
    minor: int
    patch: int
    releaselevel: str
    serial: int

    def __str__(self) -> str:  # pragma: no cover - trivial
        return __version__


_parts = __version__.split(".")
version_info = _VersionInfo(
    major=int(_parts[0]),
    minor=int(_parts[1]) if len(_parts) > 1 else 0,
    patch=int(_parts[2].split("-")[0]) if len(_parts) > 2 else 0,
    releaselevel="alpha",
    serial=0,
)
