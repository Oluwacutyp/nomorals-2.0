"""Photoshop descriptor (OSType) parser — the ``lfx2`` / ``TySh`` /
``SoLd`` / ``vibA`` / ``blwh`` / ``clrL`` blocks' language.

Adobe's public "Photoshop File Formats Specification" defines the
descriptor format (a UnicodeString name, a class id, then typed
items: ``Objc`` / ``VlLs`` / ``doub`` / ``UntF`` / ``TEXT`` / ``enum``
/ ``long`` / ``comp`` / ``bool`` / ``type`` / ``alis`` / ``ObAr`` and
the reference forms). PhotoCraft's ``crates/psd/src/descriptor.rs``
is the same public layout; the code here is Devon's own Python
written against the spec, not a copy of the Rust.

The parser is strict about bounds (every read is checked, nesting
is capped at 64) and never crashes on a hostile block: a failure
is a :class:`DescError` naming the offset, and callers that meet
one treat the block as unparsed instead of dying.
"""

from __future__ import annotations

import struct
from typing import Any

__all__ = [
    "Desc",
    "DescError",
    "parse_versioned",
    "read_desc",
]

#: Maximum descriptor nesting (the spec's own recursion is shallow;
#: 64 is far past anything Photoshop writes).
MAX_DEPTH = 64
#: Backstop against absurd counts in a hostile block.
MAX_ITEMS = 1_000_000


class DescError(Exception):
    """A descriptor block that cannot be read — always with the
    reason named."""


class _Reader:
    """A bounds-checked cursor over descriptor bytes."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self._data = data
        self._pos = pos

    @property
    def pos(self) -> int:
        return self._pos

    def take(self, count: int) -> bytes:
        if count < 0 or self._pos + count > len(self._data):
            raise DescError(
                f"short read at offset {self._pos}: need "
                f"{count} bytes, "
                f"{len(self._data) - self._pos} left")
        out = self._data[self._pos:self._pos + count]
        self._pos += count
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack(">I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack(">i", self.take(4))[0]

    def i64(self) -> int:
        return struct.unpack(">q", self.take(8))[0]

    def f64(self) -> float:
        return struct.unpack(">d", self.take(8))[0]

    def ustr(self) -> str:
        """A UnicodeString: u32 unit count + UTF-16BE units."""
        count = self.u32()
        if count > MAX_ITEMS:
            raise DescError(
                f"unicode string length {count} is implausible")
        raw = self.take(2 * count)
        try:
            return raw.decode("utf-16-be",
                              errors="replace").rstrip("\x00")
        except UnicodeDecodeError:
            return ""

    def ident(self) -> str:
        """A class/key id: u32 length 0 means a 4-byte code, else
        that many bytes."""
        count = self.u32()
        if count == 0:
            return self.take(4).decode("latin-1")
        if count > 4096:
            raise DescError(f"identifier length {count}")
        return self.take(count).decode("latin-1",
                                       errors="replace")


class Desc:
    """One parsed descriptor: a name, a class id, and ordered
    (key, value) items. Values are plain Python: ``float`` for
    ``doub``, ``int`` for ``long``/``comp``, ``bool`` for ``bool``,
    ``str`` for ``TEXT`` (as ``("text", str)``), ``("unit", (code,
    float))`` for ``UntF``, ``("unitfloats", (code, [float]))`` for
    ``UnFl``, ``("enum", (type, value))`` for ``enum``, ``("class",
    (name, id))`` for ``type``/``GlbC``, ``("raw", (kind, bytes))``
    for ``alis``/``Pth``/``tdta``, ``("obar", (prefix, Desc))`` for
    ``ObAr``, ``("ref", [...])`` for ``obj ``, a ``list`` for
    ``VlLs``, and a nested :class:`Desc` for ``Objc``/``GlbO``."""

    def __init__(self, name: str, class_id: str,
                 items: list[tuple[str, Any]]) -> None:
        self.name = name
        self.class_id = class_id
        self.items = items

    def get(self, key: str) -> Any:
        for item_key, value in self.items:
            if item_key == key:
                return value
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"Desc({self.class_id!r}, "
                f"{[(k, _short(v)) for k, v in self.items]!r})")


def _short(value: Any) -> str:
    if isinstance(value, Desc):
        return f"<Desc {value.class_id}>"
    if isinstance(value, list):
        return f"<List {len(value)}>"
    if (isinstance(value, tuple) and len(value) == 2
            and value[0] == "unit"):
        return f"<{value[1][0]} {value[1][1]}>"
    text = repr(value)
    return text if len(text) < 80 else text[:77] + "..."


def _read_value(reader: _Reader, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise DescError("descriptor nesting too deep")
    kind = reader.take(4)
    if kind == b"obj ":
        count = reader.u32()
        if count > MAX_ITEMS:
            raise DescError(f"reference count {count}")
        return ("ref", [_read_ref(reader, depth + 1)
                        for _ in range(count)])
    if kind in (b"Objc", b"GlbO"):
        return _read_desc(reader, depth + 1)
    if kind == b"VlLs":
        count = reader.u32()
        if count > MAX_ITEMS:
            raise DescError(f"list length {count}")
        return [_read_value(reader, depth + 1)
                for _ in range(count)]
    if kind == b"doub":
        return reader.f64()
    if kind == b"UntF":
        unit = reader.take(4).decode("latin-1")
        return ("unit", (unit, reader.f64()))
    if kind == b"UnFl":
        unit = reader.take(4).decode("latin-1")
        count = reader.u32()
        if count > MAX_ITEMS:
            raise DescError(f"unit-float count {count}")
        return ("unitfloats",
                (unit, [reader.f64() for _ in range(count)]))
    if kind == b"TEXT":
        return ("text", reader.ustr())
    if kind == b"enum":
        return ("enum", (reader.ident(), reader.ident()))
    if kind == b"long":
        return reader.i32()
    if kind == b"comp":
        return reader.i64()
    if kind == b"bool":
        return reader.u8() != 0
    if kind in (b"type", b"GlbC"):
        return ("class", (reader.ustr(), reader.ident()))
    if kind in (b"alis", b"Pth ", b"tdta"):
        count = reader.u32()
        if count > MAX_ITEMS:
            raise DescError(f"raw-data length {count}")
        return ("raw", (kind.decode("latin-1"),
                        reader.take(count)))
    if kind == b"ObAr":
        prefix = reader.u32()
        return ("obar", (prefix, _read_desc(reader, depth + 1)))
    raise DescError(f"unknown OSType {kind!r} at "
                    f"{reader.pos - 4}")


def _read_ref(reader: _Reader, depth: int) -> Any:
    if depth > MAX_DEPTH:
        raise DescError("descriptor nesting too deep")
    kind = reader.take(4)
    if kind == b"prop":
        return ("prop", (reader.ustr(), reader.ident(),
                         reader.ident()))
    if kind == b"Clss":
        return ("class", (reader.ustr(), reader.ident()))
    if kind == b"Enmr":
        return ("enum", (reader.ustr(), reader.ident(),
                         reader.ident(), reader.ident()))
    if kind == b"rele":
        return ("offset", (reader.ustr(), reader.ident(),
                           reader.i32()))
    if kind == b"Idnt":
        return ("ident", (reader.ustr(), reader.ident(),
                          reader.u32()))
    if kind == b"indx":
        return ("index", (reader.ustr(), reader.ident(),
                          reader.u32()))
    if kind == b"name":
        return ("name", (reader.ustr(), reader.ident(),
                         reader.ustr()))
    raise DescError(f"unknown reference type {kind!r}")


def _read_desc(reader: _Reader, depth: int) -> Desc:
    if depth > MAX_DEPTH:
        raise DescError("descriptor nesting too deep")
    name = reader.ustr()
    class_id = reader.ident()
    count = reader.u32()
    if count > MAX_ITEMS:
        raise DescError(f"descriptor item count {count}")
    items = []
    for _ in range(count):
        key = reader.ident()
        items.append((key, _read_value(reader, depth + 1)))
    return Desc(name, class_id, items)


def read_desc(data: bytes, offset: int = 0) -> tuple[Desc, int]:
    """Parse one descriptor at ``offset``; returns the descriptor
    and the offset just past it (trailing bytes allowed)."""
    reader = _Reader(data, offset)
    desc = _read_desc(reader, 0)
    return desc, reader.pos


def parse_versioned(data: bytes, offset: int = 0
                    ) -> tuple[Desc, int]:
    """Parse a versioned descriptor (u32 version 16 + descriptor)
    at ``offset``; returns the descriptor and the offset just past
    it. The version must be 16 — anything else is a :class:`DescError`."""
    reader = _Reader(data, offset)
    version = reader.u32()
    if version != 16:
        raise DescError(
            f"descriptor version {version} — expected 16")
    desc = _read_desc(reader, 0)
    return desc, reader.pos


# -- convenience accessors over parsed values -------------------------

def desc_num(desc: Desc, key: str) -> float | None:
    """A numeric item as a float (``long`` / ``doub`` /
    ``UntF``), or None."""
    value = desc.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if (isinstance(value, tuple) and len(value) == 2
            and value[0] == "unit"):
        return float(value[1][1])
    return None


def desc_bool(desc: Desc, key: str) -> bool | None:
    value = desc.get(key)
    return value if isinstance(value, bool) else None


def desc_text(desc: Desc, key: str) -> str | None:
    value = desc.get(key)
    if (isinstance(value, tuple) and len(value) == 2
            and value[0] == "text"):
        return value[1]
    if isinstance(value, str):
        return value
    return None


def desc_enum(desc: Desc, key: str) -> tuple[str, str] | None:
    value = desc.get(key)
    if (isinstance(value, tuple) and len(value) == 2
            and value[0] == "enum"):
        return value[1]
    return None


def desc_list(desc: Desc, key: str) -> list | None:
    value = desc.get(key)
    return value if isinstance(value, list) else None


def desc_desc(desc: Desc, key: str) -> Desc | None:
    value = desc.get(key)
    return value if isinstance(value, Desc) else None


def color_from_desc(desc: Desc) -> tuple[float, float, float] | None:
    """A colour descriptor as an sRGB triple in 0..1.

    ``RGBC`` (doubles 0..255, or ``redFloat``/``greenFloat``/
    ``blueFloat`` 0..1), ``CMYC`` (percent, inverted inks),
    ``Grsc`` (gray percent), ``LbCl`` (L 0..100, a/b −128..127)
    and ``HSBC`` (degrees, percent, percent) — the same classes
    PhotoCraft's ``color_from_desc`` reads. Anything else falls
    back to any ``Rd  `` / ``Grn `` / ``Bl  `` triple present."""
    from .blend import cmyk_to_rgb, hsb_to_rgb, lab_to_srgb

    cls = desc.class_id.encode("latin-1", errors="replace")
    get = lambda k: desc_num(desc, k)  # noqa: E731 - tiny local
    if cls == b"RGBC" and desc.get("redFloat") is not None:
        red = get("redFloat")
        green = get("greenFloat")
        blue = get("blueFloat")
        if None not in (red, green, blue):
            return (red, green, blue)
    if cls == b"RGBC":
        red, green, blue = get("Rd  "), get("Grn "), get("Bl  ")
        if None not in (red, green, blue):
            return (red / 255.0, green / 255.0, blue / 255.0)
    if cls == b"CMYC":
        cyan, magenta, yellow, black = (
            get("Cyn "), get("Mgnt"), get("Ylw "), get("Blck"))
        if None not in (cyan, magenta, yellow, black):
            return cmyk_to_rgb((cyan / 100.0, magenta / 100.0,
                                yellow / 100.0, black / 100.0))
    if cls == b"Grsc":
        gray = get("Gry ")
        if gray is not None:
            value = 1.0 - gray / 100.0
            return (value, value, value)
    if cls == b"LbCl":
        lum, a, b = get("Lmnc"), get("A   "), get("B   ")
        if None not in (lum, a, b):
            return lab_to_srgb((lum / 100.0, a + 128.0,
                                b + 128.0))
    if cls == b"HSBC":
        hue, sat, bri = get("H   "), get("Strt"), get("Brgh")
        if None not in (hue, sat, bri):
            return hsb_to_rgb(hue, sat / 100.0, bri / 100.0)
    if desc.get("redFloat") is not None:
        red = get("redFloat")
        green = get("greenFloat")
        blue = get("blueFloat")
        if None not in (red, green, blue):
            return (red, green, blue)
    red, green, blue = get("Rd  "), get("Grn "), get("Bl  ")
    if None not in (red, green, blue):
        return (red / 255.0, green / 255.0, blue / 255.0)
    return None
