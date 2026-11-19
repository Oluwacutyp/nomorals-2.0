"""The fixed header at the start of every INDD file
(``src/header.rs`` — see ``docs/format/header.md``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import InddError

#: The 16 bytes every INDD file starts with.
SIGNATURE = bytes([0x06, 0x06, 0xED, 0xF5, 0xD8, 0x1D, 0x46, 0xE5,
                   0xBD, 0x31, 0xEF, 0xE7, 0xFE, 0x74, 0xB7, 0x1D])

#: Number of bytes :func:`Header.parse` needs.
HEADER_LEN = 0x25

_KIND_OFFSET = 0x10
_BYTE_ORDER_OFFSET = 0x18
_MAJOR_OFFSET = 0x1D
_MINOR_OFFSET = 0x21
#: Where files from InDesign 1.0 and 1.5 have the ``DOCUMENT`` tag.
#: They do not start with the signature (``docs/format/header.md``).
LEGACY_KIND_OFFSET = 0x5C

#: Number of bytes :func:`Header.parse` needs to recognise every layout.
PROBE_LEN = LEGACY_KIND_OFFSET + 8


class ByteOrder(Enum):
    LITTLE = "little"
    BIG = "big"

    def read_u32(self, data: bytes, offset: int) -> int:
        raw = data[offset:offset + 4]
        if self is ByteOrder.LITTLE:
            return int.from_bytes(raw, "little")
        return int.from_bytes(raw, "big")


@dataclass(frozen=True)
class Version:
    """InDesign application version that last saved the file."""

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def __lt__(self, other: Version) -> bool:
        return (self.major, self.minor) < (other.major, other.minor)


@dataclass(frozen=True)
class Header:
    """ASCII tag after the signature, byte order, version."""

    kind: bytes
    byte_order: ByteOrder
    version: Version

    @classmethod
    def parse(cls, data: bytes) -> Header:
        if len(data) < HEADER_LEN:
            raise InddError.truncated(HEADER_LEN, len(data))
        if data[:16] != SIGNATURE:
            if data[LEGACY_KIND_OFFSET:LEGACY_KIND_OFFSET + 8] == b"DOCUMENT":
                raise InddError.unsupported("InDesign 1.x file layout")
            raise InddError.not_indd()
        flag = data[_BYTE_ORDER_OFFSET]
        if flag == 1:
            byte_order = ByteOrder.LITTLE
        elif flag == 2:
            byte_order = ByteOrder.BIG
        else:
            raise InddError.unknown_byte_order(flag)
        return cls(
            kind=bytes(data[_KIND_OFFSET:_KIND_OFFSET + 8]),
            byte_order=byte_order,
            version=Version(
                major=byte_order.read_u32(data, _MAJOR_OFFSET),
                minor=byte_order.read_u32(data, _MINOR_OFFSET),
            ),
        )
