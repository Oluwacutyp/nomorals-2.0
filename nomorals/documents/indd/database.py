"""The object database stored in the database pages
(``src/database.rs`` — see ``docs/format/database.md``).

Every object has a UID and a byte stream. The streams are found
through a B+ tree keyed by (UID, segment). These structures are
little-endian in every file, whatever the byte order of the object
data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .container import PAGE_SIZE
from .errors import InddError
from .object import Encoding

#: Page types, from the u32 at 0xFF4 of every database page.
ALLOC_DIR = 2
ALLOC_BITMAP = 3
LOGICAL_DIR = 4
LOGICAL_TABLE = 5
TREE_LEAF = 6
TREE_INTERIOR = 7
DATA = 8
SLOTTED = 9

_TRAILER_TYPE = 0xFF4
_TABLE_START = 0x80
_TABLE_ENTRIES = 989
_SLOT_FOOTER = 0xFDC

#: Fields of the active master page.
_MASTER_LOGICAL_DIR = 0x3A8
#: Three u32 tree roots (logical pages): objects, classes, unclassed UIDs.
_MASTER_TREE_ROOTS = 0xB7C
#: Three u64 entry counts, in the same order as the roots.
_MASTER_TREE_COUNTS = 0xB88

#: Largest segment stored on a data page.
SEGMENT_MAX = 0xF70

_SLOT_CONTINUES = 0x8000


def _corrupt(message: str) -> InddError:
    return InddError.corrupt(message)


@dataclass(frozen=True)
class Entry:
    """One leaf entry of the object tree: part ``segment`` (1-based)
    of object ``uid``."""

    segment: int
    uid: int
    #: Either a byte count (segment on a data page), or
    #: ``slot << 16 | count`` (segment in a slotted page).
    length: int
    #: Physical page for data-page segments, logical page for slotted ones.
    page: int

    def is_slotted(self) -> bool:
        return self.length >> 16 != 0

    def __len__(self) -> int:
        if self.is_slotted():
            return self.length & 0xFFFF
        return self.length

    def is_empty(self) -> bool:
        return len(self) == 0


class Database:
    def __init__(self, data: bytes) -> None:
        self.bytes = data
        self.logical: dict[int, int] = {}
        self.entries: list[Entry] = []
        #: (UID, class ID), sorted by UID.
        self.classes: list[tuple[int, int]] = []
        #: UIDs listed in the third tree. They have no class entry.
        self.unclassed: list[int] = []
        #: The encoding of object data.
        self.encoding = Encoding.default()

    @classmethod
    def open(cls, container) -> Database:
        from .object import Encoding as _Encoding
        master = container.active * PAGE_SIZE
        db = cls(container.bytes)
        db.encoding = _Encoding.of(container.header)
        directory = db._u32(master + _MASTER_LOGICAL_DIR)
        db_pages = container.master().db_pages
        if directory >= db_pages:
            raise InddError.no_database(db_pages, directory)
        db._read_logical_table(directory)
        trees: list[list[Entry]] = []
        for i in range(3):
            root = db._u32(master + _MASTER_TREE_ROOTS + 4 * i)
            expected = struct.unpack_from(
                "<Q", db.bytes, master + _MASTER_TREE_COUNTS + 8 * i)[0]
            out: list[Entry] = []
            db._walk(root, 0, out)
            if len(out) != expected:
                raise _corrupt(f"tree {i} has {len(out)} entries, "
                               f"master page says {expected}")
            trees.append(out)
        entries, classes, unclassed = trees
        db.entries = entries
        db.classes = [(e.uid, e.length) for e in classes]
        db.unclassed = [e.uid for e in unclassed]
        return db

    def _u16(self, offset: int) -> int:
        raw = self.bytes[offset:offset + 2]
        if len(raw) < 2:
            raise _corrupt(f"read past end of file at {offset:#x}")
        return struct.unpack("<H", raw)[0]

    def _u32(self, offset: int) -> int:
        raw = self.bytes[offset:offset + 4]
        if len(raw) < 4:
            raise _corrupt(f"read past end of file at {offset:#x}")
        return struct.unpack("<I", raw)[0]

    def page_type(self, page: int) -> int:
        return self._u32(page * PAGE_SIZE + _TRAILER_TYPE)

    def _expect_type(self, page: int, expected: int) -> int:
        actual = self.page_type(page)
        if actual != expected:
            raise _corrupt(f"page {page} has type {actual}, "
                           f"expected {expected}")
        return page * PAGE_SIZE

    def _read_logical_table(self, dir_page: int) -> None:
        base = self._expect_type(dir_page, LOGICAL_DIR)
        for j in range(_TABLE_ENTRIES):
            table_page = self._u32(base + _TABLE_START + 4 * j)
            if table_page == 0:
                continue
            table = self._expect_type(table_page, LOGICAL_TABLE)
            for i in range(_TABLE_ENTRIES):
                physical = self._u32(table + _TABLE_START + 4 * i)
                if physical != 0:
                    self.logical[j * _TABLE_ENTRIES + i] = physical

    def physical(self, logical: int) -> int:
        """Physical page for a logical page number."""
        try:
            return self.logical[logical]
        except KeyError:
            raise _corrupt(f"logical page {logical} is not mapped") \
                from None

    def _walk(self, logical: int, depth: int, out: list[Entry]) -> None:
        if depth > 32:
            raise _corrupt("object tree is too deep")
        page = self.physical(logical)
        base = page * PAGE_SIZE
        count = self._u32(base)
        kind = self.page_type(page)
        if kind == TREE_LEAF:
            for i in range(count):
                o = base + 4 + 16 * i
                out.append(Entry(
                    segment=self._u32(o), uid=self._u32(o + 4),
                    length=self._u32(o + 8), page=self._u32(o + 12)))
        elif kind == TREE_INTERIOR:
            # child, then (segment, uid) key and child, repeated.
            children = [self._u32(base + 4)]
            for i in range(1, count):
                children.append(self._u32(base + 4 + 12 * i))
            for child in children:
                self._walk(child, depth + 1, out)
        else:
            raise _corrupt(f"tree page {page} has type {kind}")

    def class_of(self, uid: int) -> int | None:
        """Class ID of object ``uid``."""
        import bisect
        uids = [u for u, _ in self.classes]
        i = bisect.bisect_left(uids, uid)
        if i < len(self.classes) and self.classes[i][0] == uid:
            return self.classes[i][1]
        return None

    def _read_slotted(self, logical: int, slot: int, length: int,
                      out: bytearray) -> None:
        # Each continuation holds at least one byte and fewer than
        # ``length``, so ``length`` falls with every record and a cycle
        # of pointers ends.
        while True:
            base = self._expect_type(self.physical(logical), SLOTTED)
            pointer = base + _SLOT_FOOTER - 4 * slot
            if pointer < base:
                raise _corrupt(f"slot {slot} is outside its page")
            offset = base + self._u32(pointer)
            record_len = self._u16(offset)
            record_id = self._u16(offset + 2)
            if record_id & ~_SLOT_CONTINUES != slot:
                raise _corrupt(f"slot {slot} of logical page {logical} "
                               f"holds record {record_id:#x}")
            record = self.bytes[offset + 4:offset + record_len]
            if len(record) != record_len - 4:
                raise _corrupt(f"record at {offset:#x} runs past end "
                               "of file")
            if record_id & _SLOT_CONTINUES == 0:
                if len(record) < length:
                    raise _corrupt(f"record at {offset:#x} is shorter "
                                   f"than {length}")
                out.extend(record[:length])
                return
            # A continued record starts with a pointer to the rest:
            # ``slot << 16`` and a logical page, as in a leaf entry.
            if len(record) < 8:
                raise _corrupt(f"continued record at {offset:#x} is "
                               "too short")
            nxt = self._u32(offset + 4)
            next_page = self._u32(offset + 8)
            here = record[8:]
            if not here or len(here) >= length or nxt & 0xFFFF != 0:
                raise _corrupt(f"bad continuation at {offset:#x}")
            out.extend(here)
            length -= len(here)
            logical = next_page
            slot = nxt >> 16

    def _read_entry(self, entry: Entry, out: bytearray) -> None:
        if entry.is_slotted():
            self._read_slotted(entry.page, entry.length >> 16,
                               len(entry), out)
            return
        if entry.length > SEGMENT_MAX:
            raise _corrupt(f"segment of {entry.length} bytes on a data "
                           "page")
        base = self._expect_type(entry.page, DATA)
        out.extend(self.bytes[base:base + len(entry)])

    def uids(self):
        """The UIDs of all objects, in ascending order."""
        seen = None
        for entry in self.entries:
            if entry.uid != seen:
                seen = entry.uid
                yield entry.uid

    def object(self, uid: int) -> bytes | None:
        """The complete byte stream of object ``uid``, or ``None`` if
        it doesn't exist."""
        import bisect
        uids = [e.uid for e in self.entries]
        start = bisect.bisect_left(uids, uid)
        out = bytearray()
        found = False
        i = start
        while i < len(self.entries) and self.entries[i].uid == uid:
            entry = self.entries[i]
            if entry.segment != i - start + 1:
                raise _corrupt(f"object {uid} is missing segment "
                               f"{i - start + 1}")
            self._read_entry(entry, out)
            found = True
            i += 1
        return bytes(out) if found else None

    def get(self, uid: int):
        """Object ``uid`` with its class, or ``None`` if it has no data."""
        from .object import Object
        data = self.object(uid)
        if data is None:
            return None
        return Object(uid=uid, class_id=self.class_of(uid), data=data,
                      encoding=self.encoding)
