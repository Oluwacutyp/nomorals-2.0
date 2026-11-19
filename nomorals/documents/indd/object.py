"""The contents of database objects (``src/object.rs`` — see
``docs/format/objects.md``).

Most objects are a sequence of chunks: u32 chunk ID, u32 length,
data. Some objects (embedded files, the XMP packet) are plain byte
streams.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import InddError
from .header import ByteOrder, Header, Version


def _corrupt(message: str) -> InddError:
    return InddError.corrupt(message)


class Encoding:
    """How the object data of one file is encoded: its byte order
    (``docs/format/objects.md``) and the first byte of its in-object
    strings. Every decoder of object data takes it from the
    :class:`Object` or :class:`Cursor` it reads, so files of
    different encodings can be read at the same time."""

    def __init__(self, big_endian: bool, string_tag: int) -> None:
        self.big_endian = big_endian
        self.string_tag = string_tag

    @classmethod
    def default(cls) -> Encoding:
        """Little-endian, from InDesign 3.0 or later."""
        return cls(big_endian=False, string_tag=2)

    @classmethod
    def new(cls, order: ByteOrder, version: Version) -> Encoding:
        """The encoding of a file with this byte order and version.
        In-object strings start with 1 in files from InDesign 2.0,
        with 2 in later files (``docs/format/objects.md``)."""
        return cls(big_endian=order is ByteOrder.BIG,
                   string_tag=1 if version.major <= 2 else 2)

    @classmethod
    def of(cls, header: Header) -> Encoding:
        return cls.new(header.byte_order, header.version)

    def _fmt(self, size: int) -> str:
        return (">" if self.big_endian else "<") + {2: "H", 4: "I",
                                                     8: "Q"}[size]

    def u16_from(self, raw: bytes) -> int:
        return struct.unpack(self._fmt(2), raw)[0]

    def i16_from(self, raw: bytes) -> int:
        return struct.unpack(self._fmt(2).replace("H", "h"), raw)[0]

    def u32_from(self, raw: bytes) -> int:
        return struct.unpack(self._fmt(4), raw)[0]

    def i32_from(self, raw: bytes) -> int:
        return struct.unpack(self._fmt(4).replace("I", "i"), raw)[0]

    def f64_from(self, raw: bytes) -> float:
        return struct.unpack(self._fmt(8).replace("Q", "d"), raw)[0]

    def u16_at(self, data: bytes, offset: int) -> int | None:
        raw = data[offset:offset + 2]
        return self.u16_from(raw) if len(raw) == 2 else None

    def i16_at(self, data: bytes, offset: int) -> int | None:
        raw = data[offset:offset + 2]
        return self.i16_from(raw) if len(raw) == 2 else None

    def u32_at(self, data: bytes, offset: int) -> int | None:
        raw = data[offset:offset + 4]
        return self.u32_from(raw) if len(raw) == 4 else None

    def i32_at(self, data: bytes, offset: int) -> int | None:
        raw = data[offset:offset + 4]
        return self.i32_from(raw) if len(raw) == 4 else None

    def f64_at(self, data: bytes, offset: int) -> float | None:
        raw = data[offset:offset + 8]
        return self.f64_from(raw) if len(raw) == 8 else None

    def u16_bytes(self, value: int) -> bytes:
        return struct.pack(self._fmt(2), value)

    def u32_bytes(self, value: int) -> bytes:
        return struct.pack(self._fmt(4), value)

    def cursor(self, data: bytes) -> Cursor:
        return Cursor(data, 0, False, self)

    def chunks(self, data: bytes) -> list[Chunk] | None:
        """Split an object's bytes into chunks. ``None`` if the bytes
        are not a chunk sequence."""
        out: list[Chunk] = []
        pos = 0
        while pos < len(data):
            head = data[pos:pos + 8]
            if len(head) < 8:
                return None
            chunk_id = self.u32_at(head, 0)
            length = self.u32_at(head, 4)
            if chunk_id is None or length is None:
                return None
            chunk_data = data[pos + 8:pos + 8 + length]
            if len(chunk_data) != length:
                return None
            out.append(Chunk(id=chunk_id, data=chunk_data))
            pos += 8 + length
        return out


@dataclass(frozen=True)
class Chunk:
    id: int
    data: bytes


@dataclass
class Object:
    """A decoded object: its UID, class and chunks."""

    uid: int
    class_id: int | None
    data: bytes
    encoding: Encoding

    @property
    def bytes(self) -> bytes:
        """The raw byte stream (upstream's field name)."""
        return self.data

    def chunks(self) -> list[Chunk] | None:
        return self.encoding.chunks(self.data)

    def cursor(self, data: bytes) -> Cursor:
        """A cursor over ``data`` (normally a chunk of this object)
        in the object's encoding."""
        return self.encoding.cursor(data)

    def chunk(self, chunk_id: int) -> bytes | None:
        """Data of the first chunk with this ID."""
        chunks = self.chunks()
        if chunks is None:
            return None
        for chunk in chunks:
            if chunk.id == chunk_id:
                return chunk.data
        return None


@dataclass
class Name:
    """A name stored as a flag byte and an in-object string. A flag
    of 1 marks an InDesign built-in key, which IDML writes with
    ``$ID/`` before it (``docs/format/objects.md``)."""

    builtin: bool = False
    name: str = ""

    def idml(self) -> str:
        """The name as IDML writes it."""
        return builtin_key(self.name) if self.builtin else self.name


def builtin_key(key: str) -> str:
    """How IDML writes the InDesign built-in key ``key``: ``$ID/key``."""
    return f"$ID/{key}"


def _short(what: str, pos: int) -> InddError:
    return _corrupt(f"chunk data ends while reading {what} at {pos}")


class Cursor:
    """Cursor over object data in one :class:`Encoding`. Make one with
    :meth:`Encoding.cursor` or :meth:`Object.cursor`."""

    def __init__(self, data: bytes, pos: int, tag_read: bool,
                 enc: Encoding) -> None:
        self.data = data
        self.pos = pos
        self.tag_read = tag_read
        self.enc = enc

    def encoding(self) -> Encoding:
        return self.enc

    def sub(self, data: bytes) -> Cursor:
        """A cursor over other data in the same encoding."""
        return self.enc.cursor(data)

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def bytes(self, n: int) -> bytes:
        out = self.data[self.pos:self.pos + n]
        if len(out) != n:
            raise _short("bytes", self.pos)
        self.pos += n
        return out

    def _array(self, n: int) -> bytes:
        out = self.data[self.pos:self.pos + n]
        if len(out) != n:
            raise _short("bytes", self.pos)
        self.pos += n
        return out

    def skip(self, n: int) -> None:
        self.bytes(n)

    def u8(self) -> int:
        return self.bytes(1)[0]

    def u16(self) -> int:
        return self.enc.u16_from(self._array(2))

    def u32(self) -> int:
        return self.enc.u32_from(self._array(4))

    def i32(self) -> int:
        return self.enc.i32_from(self._array(4))

    def f64(self) -> float:
        return self.enc.f64_from(self._array(8))

    def u32_list(self) -> list[int]:
        """A u32 count followed by that many u32 values."""
        n = self.u32()
        if n > self.remaining() // 4:
            raise _corrupt(f"list of {n} at {self.pos} is too long")
        return [self.u32() for _ in range(n)]

    def segments(self, chars: int) -> str:
        """Text made of ``chars`` UTF-16 code units, stored as
        segments. Unpaired surrogates become U+FFFD, as Rust's
        ``String::from_utf16_lossy`` does."""
        units = self.segment_units(chars)
        raw = b"".join(struct.pack(">H" if self.enc.big_endian else "<H",
                                   unit)
                       for unit in units)
        codec = "utf-16-be" if self.enc.big_endian else "utf-16-le"
        return raw.decode(codec, errors="replace")

    def segment_units(self, chars: int) -> list[int]:
        """Like :meth:`segments`, but returns the UTF-16 code units,
        so a surrogate pair split between two runs can be joined by
        the caller."""
        # Each unit takes at least one byte, which bounds the capacity
        # of a damaged count.
        units: list[int] = []
        while len(units) < chars:
            header = self.u16()
            n = header & 0x3FFF
            kind = header & 0xC000
            if kind == 0x4000:
                units.extend(self.bytes(n))
            elif kind == 0x8000:
                for _ in range(n):
                    units.append(self.u16())
            else:
                raise _corrupt(f"unknown text segment header "
                               f"{header:#06x} at {self.pos - 2}")
            if n == 0:
                raise _corrupt(f"empty text segment at {self.pos - 2}")
        if len(units) != chars:
            raise _corrupt(f"text segments hold {len(units)} units, "
                           f"expected {chars}")
        return units

    def flag(self) -> int:
        """The flag byte before a string (see :meth:`string`)."""
        tag = self.enc.string_tag
        if (self.enc.big_endian and self.pos < len(self.data)
                and self.data[self.pos] == tag and self.remaining() >= 2):
            self.tag_read = True
            self.pos += 2
            return self.data[self.pos - 1]
        return self.u8()

    def name(self) -> Name:
        """A flag byte (see :meth:`flag`), 1 for a built-in key, and
        a string."""
        builtin = self.flag() == 1
        return Name(builtin=builtin, name=self.string())

    def string(self) -> str:
        """A string stored inside object data: u8 2 (1 in files from
        InDesign 2.0, see :meth:`Encoding.new`), a u8 whose meaning is
        unknown (usually 0), u16 length in UTF-16 code units, then
        segments.

        The byte before the 2 is often a flag (1 = the string is a
        built-in key), and the two bytes form a u16 with the 2 in the
        high byte. In big-endian data the 2 therefore comes first:
        read such a flag with :meth:`flag`. If the caller skipped the
        byte before the string instead, the 2 is that byte
        (``docs/format/big-endian.md``)."""
        start = self.pos
        expected = self.enc.string_tag
        if self.enc.big_endian:
            if not self.tag_read:
                self.tag_read = False
                before = (self.data[start - 1] if start >= 1 else None)
                if before != expected and self.u8() != expected:
                    raise _corrupt(f"no string tag at {start}")
                # The flag.
                self.u8()
            else:
                self.tag_read = False
        else:
            tag = self.u8()
            if tag != expected:
                raise _corrupt(f"string tag {tag} at {start}, "
                               f"expected {expected}")
        self.u8()
        n = self.u16()
        if n == 0:
            return ""
        return self.segments(n)
