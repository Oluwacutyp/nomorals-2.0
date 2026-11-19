"""The page and contiguous-object layout of an INDD file
(``src/container.rs`` — see ``docs/format/container.md``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .errors import InddError
from .header import ByteOrder, Header

if TYPE_CHECKING:
    from .database import Database

#: INDD files are made of pages of this size.
PAGE_SIZE = 4096

#: Starts every contiguous object header.
CONTIG_HEADER_GUID = bytes([0xDE, 0x39, 0x39, 0x79, 0x51, 0x88, 0x4B,
                            0x6C, 0x8E, 0x63, 0xEE, 0xF8, 0xAE, 0xE0,
                            0xDD, 0x38])

#: Starts every contiguous object trailer.
CONTIG_TRAILER_GUID = bytes([0xFD, 0xCE, 0xDB, 0x70, 0xF7, 0x86, 0x4B,
                             0x4F, 0xA4, 0xD3, 0xC7, 0x28, 0xB3, 0x41,
                             0x71, 0x06])

_MARKER_LEN = 32
_SEQUENCE_OFFSET = 0x108
_DB_PAGES_OFFSET = 0x118
_XMP_START = b"<?xpacket begin="


def _le_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _le_u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


@dataclass(frozen=True)
class MasterPage:
    """One of the two master pages at the start of the file."""

    #: Generation number. The master page with the higher value is active.
    sequence: int
    #: Number of database pages, including the two master pages. The
    #: contiguous objects start right after them.
    db_pages: int

    @classmethod
    def parse(cls, page: bytes) -> MasterPage:
        return cls(sequence=_le_u64(page, _SEQUENCE_OFFSET),
                   db_pages=_le_u32(page, _DB_PAGES_OFFSET))


@dataclass(frozen=True)
class ContigObject:
    """A copy of one database object, stored as a contiguous byte
    stream after the database pages."""

    #: File offset of the object's header marker.
    offset: int
    uid: int
    class_id: int
    data: bytes


class Container:
    def __init__(self, data: bytes, header: Header, active: int,
                 masters: tuple[MasterPage, MasterPage]) -> None:
        self.bytes = data
        self.header = header
        #: Index (0 or 1) of the active master page.
        self.active = active
        self.masters = masters

    @classmethod
    def parse(cls, data: bytes) -> Container:
        # The signature first, so that a short file of another kind is
        # reported as not an INDD file.
        first = Header.parse(data[:min(len(data), PAGE_SIZE)])
        if len(data) < 2 * PAGE_SIZE:
            raise InddError.truncated(2 * PAGE_SIZE, len(data))
        Header.parse(data[PAGE_SIZE:2 * PAGE_SIZE])
        masters = (MasterPage.parse(data[:PAGE_SIZE]),
                   MasterPage.parse(data[PAGE_SIZE:2 * PAGE_SIZE]))
        active = 1 if masters[1].sequence > masters[0].sequence else 0
        # Every complete file holds at least its database pages
        # (container.md).
        needed = masters[active].db_pages * PAGE_SIZE
        if len(data) < needed:
            raise InddError.truncated(needed, len(data))
        header = first if active == 0 else Header.parse(
            data[PAGE_SIZE:2 * PAGE_SIZE])
        return cls(data, header, active, masters)

    def database(self) -> Database:
        from .database import Database
        return Database.open(self)

    def master(self) -> MasterPage:
        return self.masters[self.active]

    def contig_start(self) -> int:
        """File offset where the contiguous objects start."""
        return self.master().db_pages * PAGE_SIZE

    def contig_objects(self):
        """Iterate the contiguous objects after the database pages."""
        pos = self.contig_start()
        data = self.bytes
        while True:
            start = pos
            # The objects end at the first position without a header
            # marker. What follows is padding, sometimes with stale
            # bytes from earlier saves.
            if (start + _MARKER_LEN > len(data)
                    or data[start:start + 16] != CONTIG_HEADER_GUID):
                return
            uid = _le_u32(data, start + 16)
            class_id = _le_u32(data, start + 20)
            length = _le_u32(data, start + 24)
            data_start = start + _MARKER_LEN
            trailer = data_start + length
            valid = (trailer + _MARKER_LEN <= len(data)
                     and data[trailer:trailer + 16] == CONTIG_TRAILER_GUID
                     and _le_u32(data, trailer + 16) == uid
                     and _le_u32(data, trailer + 24) == length)
            if not valid:
                raise InddError.bad_contig_object(start)
            pos = trailer + _MARKER_LEN
            yield ContigObject(offset=start, uid=uid, class_id=class_id,
                               data=data[data_start:trailer])

    def xmp(self) -> bytes | None:
        """The XMP packet, from ``<?xpacket begin=`` to the end of the
        packet."""
        for obj in self.contig_objects():
            packet = _xmp_packet(obj.data, self.header.byte_order)
            if packet is not None:
                return packet
        return None


def _xmp_packet(data: bytes, order: ByteOrder) -> bytes | None:
    """An XMP object is a u32 packet length followed by the packet.
    Some files store the length in the other byte order, so both are
    accepted. The packet can be shorter than the object: the rest is
    left over from an earlier, longer packet
    (``docs/format/container.md``)."""
    if len(data) < 4 + len(_XMP_START):
        return None
    room = len(data) - 4
    fmt = order.value
    other = "big" if fmt == "little" else "little"
    length = int.from_bytes(data[0:4], fmt)
    swapped = int.from_bytes(data[0:4], other)
    chosen = None
    for candidate in (length, swapped):
        if candidate == room:
            chosen = candidate
            break
    if chosen is None:
        for candidate in (length, swapped):
            if candidate <= room:
                chosen = candidate
                break
    if chosen is None:
        return None
    packet = data[4:4 + chosen]
    return packet if packet.startswith(_XMP_START) else None
