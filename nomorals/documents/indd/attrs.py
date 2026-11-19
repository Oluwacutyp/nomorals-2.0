"""Attribute lists (chunk 0x6E03 and others) — ``src/model/attrs.rs``.

A list is a u32 count, then records: u32 attribute ID, u16 payload
size, payload. The payload is a u16 value count and that many values,
each a u32 type, u16 length and data. The first value is the
attribute's value.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InddError
from .object import Cursor, Encoding


class Ty:
    """Value type codes."""

    DOUBLE = 0x6E68
    INT = 0x6E67
    ENUM = 0x6E65
    POINT = 0x6E69
    REF = 0x117
    #: A built-in item code, the second value after a reference of 0.
    CODE = 0x6E64
    #: A flag byte and an in-object string.
    STRING = 0x1B02
    FORM_STRING = 0x1451F
    #: The u16 caption position code of a table style (tables.md).
    CAPTION_POSITION = 0x22E1F


#: Page item attributes whose value is a list of opacity gradient
#: stops: u32 count, then three f64 per stop. See
#: ``docs/format/transparency.md``.
OPACITY_STOPS = (0x1EB8C, 0x1EB95, 0x1EB9E)

#: The page item attribute of a stroke's dashes and gaps: 8 bytes (0 in
#: every sample), u32 count, that many f64, u16 corner adjustment. See
#: ``docs/format/attributes.md``, dashes.
DASHES = 0x5A35


@dataclass(frozen=True)
class TabStop:
    """A tab stop. See ``docs/format/attributes.md``, tab lists."""

    position: float
    #: 0 left, 1 centre, 2 right, 3 on a character.
    alignment: int
    #: The character of a stop aligned on one (code 3).
    character: str | None
    leader: str


@dataclass(frozen=True)
class GrepStyle:
    """A GREP style: the character style it applies (0 for none) to
    the text that the expression matches."""

    style: int
    expression: str


@dataclass(frozen=True)
class LineStyle:
    """A line style: the character style it applies (0 for none) to a
    number of lines."""

    style: int
    lines: int


@dataclass(frozen=True)
class NestedStyle:
    """A nested style: the character style it applies (0 for none)
    and up to where."""

    style: int
    delimiter: str
    repetition: int
    inclusive: bool


class Value:
    """One decoded attribute value. ``kind`` names the variant;
    ``data`` holds its payload (a number, a string, a tuple, a list,
    or ``(type_code, bytes)`` for :meth:`other`)."""

    def __init__(self, kind: str, data) -> None:
        self.kind = kind
        self.data = data

    def __eq__(self, other) -> bool:
        return (isinstance(other, Value) and self.kind == other.kind
                and self.data == other.data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Value({self.kind!r}, {self.data!r})"

    @classmethod
    def double(cls, value: float) -> Value:
        return cls("double", value)

    @classmethod
    def int(cls, value: int) -> Value:
        return cls("int", value)

    @classmethod
    def enum(cls, value: int) -> Value:
        return cls("enum", value)

    @classmethod
    def ref(cls, value: int) -> Value:
        return cls("ref", value)

    @classmethod
    def point(cls, x: float, y: float) -> Value:
        return cls("point", (x, y))

    @classmethod
    def ref_or_code(cls, ref: int, code: int) -> Value:
        return cls("ref_or_code", (ref, code))

    @classmethod
    def words(cls, first: int, second: int) -> Value:
        return cls("words", (first, second))

    @classmethod
    def words3(cls, values) -> Value:
        return cls("words3", tuple(values))

    @classmethod
    def string(cls, value: str) -> Value:
        return cls("string", value)

    @classmethod
    def text(cls, value: str) -> Value:
        return cls("text", value)

    @classmethod
    def tab_list(cls, stops) -> Value:
        return cls("tab_list", list(stops))

    @classmethod
    def style_list(cls, count: int, items) -> Value:
        return cls("style_list", (count, items))

    @classmethod
    def bullet_char(cls, kind: int, value: int) -> Value:
        return cls("bullet_char", (kind, value))

    @classmethod
    def dashes(cls, lengths, corner: int) -> Value:
        return cls("dashes", (list(lengths), corner))

    @classmethod
    def glyph(cls, name: str) -> Value:
        return cls("glyph", name)

    @classmethod
    def features(cls, values) -> Value:
        return cls("features", list(values))

    @classmethod
    def stops(cls, values) -> Value:
        return cls("stops", [list(v) for v in values])

    @classmethod
    def other(cls, type_code: int, data: bytes) -> Value:
        return cls("other", (type_code, bytes(data)))

    def as_f64(self) -> float | None:
        if self.kind == "double":
            return self.data
        if self.kind == "int":
            return float(self.data)
        if self.kind == "enum":
            return float(self.data)
        return None

    def as_ref(self) -> int | None:
        return self.data if self.kind == "ref" else None

    def as_u32(self) -> int | None:
        if self.kind in ("ref", "int", "enum"):
            return self.data
        return None

    def as_string(self) -> str | None:
        return self.data if self.kind == "string" else None

    def code(self) -> int | None:
        """The code ``indd audit`` reports for a value without an
        IDML value: its integer, the code of a reference-or-code or
        the first of two or three words."""
        if self.kind in ("ref_or_code", "words"):
            return self.data[0]
        if self.kind == "words3":
            return self.data[0]
        return self.as_u32()


def nested_delimiter(code: str):
    """Delimiter, repetition and inclusiveness of a nested style's
    delimiter code: ``^c``, or ``(d)`` / ``[d]`` followed by an
    optional count. ``None`` for codes without evidence. See
    ``docs/format/attributes.md``, nested styles."""
    if code == "^c":
        return ("dropcap", 1, True)
    if not code:
        return None
    first = code[0]
    if first == "(":
        inclusive, close = True, ")"
    elif first == "[":
        inclusive, close = False, "]"
    else:
        return None
    end = code.rfind(close)
    if end < 0:
        return None
    inner = code[1:end]
    count = code[end + 1:]
    try:
        repetition = 1 if not count else int(count)
    except ValueError:
        return None
    delimiter = {
        "^c": "dropcap", "^w": "any_word", "^?": "any_character",
        "^t": "tabs", "^p": "end_nested_style", "^9": "digits",
        "^S": "sentence", "^L": "repeat",
    }.get(inner)
    if delimiter is None:
        if len(inner) == 1 and inner != "^":
            delimiter = f"char:{inner}"
        else:
            return None
    return (delimiter, repetition, inclusive)


class Attrs:
    """Attribute IDs and values."""

    def __init__(self, values: list[tuple[int, Value]] | None = None):
        self.values: list[tuple[int, Value]] = list(values or [])

    def __eq__(self, other) -> bool:
        return isinstance(other, Attrs) and self.values == other.values

    def get(self, attr_id: int) -> Value | None:
        for found_id, value in self.values:
            if found_id == attr_id:
                return value
        return None

    def unknown_code(self, attr_id: int, value: Value) -> None:
        """A value without an IDML value, for ``indd audit``. The
        writer records nothing here: audit reporting lives in the
        audit pass, not in the attribute list, so this is a no-op
        that keeps the writer's call sites faithful in shape."""
        return None

    @classmethod
    def parse(cls, enc: Encoding, data: bytes) -> Attrs:
        """A page item list (chunk 0x6E03): u32 count, then records."""
        cursor = enc.cursor(data)
        n = cursor.u32()
        return cls._records(cursor, n, _decode)

    @classmethod
    def parse_at(cls, cursor: Cursor) -> Attrs:
        """A page item attribute list (u32 count, records) at the
        cursor, which is left after the list."""
        n = cursor.u32()
        return cls._records(cursor, n, _decode)

    @classmethod
    def parse_short(cls, enc: Encoding, data: bytes) -> Attrs:
        """A page item attribute list with a u16 count, as object
        styles hold."""
        cursor = enc.cursor(data)
        n = cursor.u16()
        return cls._records(cursor, n, _decode)

    @classmethod
    def parse_text(cls, cursor: Cursor, count: int) -> Attrs:
        """A text attribute list: ``count`` records at the cursor.
        Values are decoded by their attribute's layout (see
        ``text_layout``)."""
        return cls._records(cursor, count, _decode_text)

    @classmethod
    def _records(cls, cursor: Cursor, n: int, decode) -> Attrs:
        enc = cursor.encoding()
        out: list[tuple[int, Value]] = []
        for _ in range(n):
            attr_id = cursor.u32()
            size = cursor.u16()
            payload = enc.cursor(cursor.bytes(size))
            # A record without values (for example a merged-cell
            # marker).
            if size < 2:
                continue
            count = payload.u16()
            first = None
            for i in range(count):
                type_code = payload.u32()
                length = payload.u16()
                data = payload.bytes(length)
                if i == 0:
                    first = decode(enc, attr_id, type_code, data)
                elif (i == 1 and first is not None
                        and first.kind == "ref"
                        and type_code == Ty.CODE and length == 4):
                    code = enc.u32_at(data, 0)
                    if code is not None:
                        first = Value.ref_or_code(first.data, code)
            if first is not None:
                out.append((attr_id, first))
        return cls(out)


def _text_layout(attr_id: int) -> str | None:
    """How the value of a text, table or cell attribute is laid out,
    for the attributes whose value is not a single number. See
    ``docs/format/attributes.md`` and ``tables.md``."""
    if attr_id in (0x1B49, 0x1B4A):
        return "point"
    if attr_id in (0x1B02, 0x1A414, 0x1A423, 0x1A424, 0x4234, 0x424B):
        return "string"
    if attr_id in (0x1B71, 0x1B72, 0x1B95, 0x1B9D, 0x1BFD):
        return "ref_or_code"
    if attr_id == 0x422E:
        return "text"
    if attr_id == 0x1B29:
        return "tab_list"
    if attr_id == 0x1B75:
        return "nested_styles"
    if attr_id == 0x1BBA:
        return "grep_styles"
    if attr_id == 0x1BBB:
        return "line_styles"
    if attr_id == 0x1A406:
        return "bullet_char"
    if attr_id == 0x1A41E:
        return "words3"
    if attr_id == 0x1B5E:
        return "glyph"
    if attr_id == 0x42AE:
        return "features"
    if attr_id in (0xB64D, 0xB64E, 0xB64F, 0xB650, 0xB655, 0xB658,
                   0xB65B, 0xB65E, 0xB688, 0xB689, 0xB68E, 0xB68F):
        return "words"
    return None


def _features(cursor: Cursor):
    """The features of attribute 0x42AE (``docs/format/attributes.md``,
    alternate glyphs): the tag bytes are in reading order in either
    byte order. ``None`` unless every tag is printable ASCII and the
    features fill the value exactly."""
    try:
        n = cursor.u16()
        out = []
        for _ in range(n):
            tag = cursor.bytes(4)
            if not all(0x20 <= b < 0x7F for b in tag):
                return None
            out.append((tag.decode("ascii"), cursor.u32()))
        return out if cursor.remaining() == 0 else None
    except InddError:
        return None


def _tab_list(cursor: Cursor):
    """A tab list: u16 count, then per stop f64 position, u16
    alignment (for code 3 followed by the alignment character as a
    UTF-16 unit), u16 leader length and the leader in UTF-16 code
    units. ``None`` unless the stops fill the value exactly."""
    try:
        n = cursor.u16()
        out = []
        for _ in range(n):
            position = cursor.f64()
            alignment = cursor.u16()
            character = None
            if alignment == 3:
                character = chr(cursor.u16())
            length = cursor.u16()
            units = [cursor.u16() for _ in range(length)]
            leader = "".join(chr(u) for u in units)
            out.append(TabStop(position=position, alignment=alignment,
                               character=character, leader=leader))
        return out if cursor.remaining() == 0 else None
    except InddError:
        return None


def _style_items(layout: str, cursor: Cursor):
    """The records of a style list: u32 count, then per record a u32
    character style and
    - nested styles: u32 length and the delimiter code as text segments;
    - GREP styles: u32 length and the expression as text segments;
    - line styles: u32 line count.

    ``None`` unless every record can be read (every nested delimiter
    is known) and the records fill the value exactly."""
    def text(c: Cursor) -> str | None:
        try:
            length = c.u32()
        except InddError:
            return None
        if length == 0:
            return ""
        try:
            return c.segments(length)
        except InddError:
            return None
    try:
        n = cursor.u32()
        nested, grep, line = [], [], []
        for _ in range(n):
            style = cursor.u32()
            if layout == "nested_styles":
                code = text(cursor)
                if code is None:
                    return None
                parsed = nested_delimiter(code)
                if parsed is None:
                    return None
                delimiter, repetition, inclusive = parsed
                nested.append(NestedStyle(
                    style=style, delimiter=delimiter,
                    repetition=repetition, inclusive=inclusive))
            elif layout == "grep_styles":
                expression = text(cursor)
                if expression is None:
                    return None
                grep.append(GrepStyle(style=style,
                                      expression=expression))
            else:
                line.append(LineStyle(style=style, lines=cursor.u32()))
        if cursor.remaining() != 0:
            return None
        if layout == "nested_styles":
            return ("nested", nested)
        if layout == "grep_styles":
            return ("grep", grep)
        return ("line", line)
    except InddError:
        return None


def _decode_text(enc: Encoding, attr_id: int, type_code: int,
                 data: bytes) -> Value:
    """A text, table or cell attribute value. Values of the attributes
    in :func:`_text_layout` are decoded by their layout (``other`` if
    they do not fit it); the others are numbers of 8, 4 or 2 bytes."""
    layout = _text_layout(attr_id)
    if layout is None:
        if len(data) == 8:
            value = enc.f64_at(data, 0)
            return Value.double(value) if value is not None else Value.other(type_code, data)
        if len(data) == 4:
            value = enc.u32_at(data, 0)
            return Value.ref(value) if value is not None else Value.other(type_code, data)
        if len(data) == 2:
            value = enc.u16_at(data, 0)
            return Value.enum(value) if value is not None else Value.other(type_code, data)
        return Value.other(type_code, data)
    cursor = enc.cursor(data)
    value = None
    try:
        if layout == "point" and len(data) == 16:
            value = Value.point(cursor.f64(), cursor.f64())
        elif layout == "string":
            cursor.flag()
            value = Value.string(cursor.string())
        elif layout == "text":
            n = cursor.u32()
            value = Value.text(cursor.segments(n))
        elif layout == "tab_list":
            stops = _tab_list(cursor)
            value = Value.tab_list(stops) if stops is not None else None
        elif layout in ("nested_styles", "grep_styles", "line_styles"):
            count = cursor.u32()
            items = _style_items(layout, enc.cursor(data))
            value = (Value.style_list(count, items)
                     if items is not None else None)
        elif layout == "bullet_char" and len(data) == 8:
            value = Value.bullet_char(cursor.u32(), cursor.u32())
        elif layout == "words":
            value = Value.words(cursor.u32(), cursor.u32())
        elif layout == "words3" and len(data) == 12:
            value = Value.words3([cursor.u32(), cursor.u32(),
                                  cursor.u32()])
        elif layout == "ref_or_code" and len(data) == 8:
            value = Value.ref_or_code(cursor.u32(), cursor.u32())
        elif layout == "glyph":
            n = cursor.u16()
            name = cursor.bytes(n)
            cursor.u32()
            if cursor.remaining() == 0:
                value = Value.glyph(name.decode("utf-8", errors="strict"))
        elif layout == "features":
            found = _features(cursor)
            value = Value.features(found) if found is not None else None
    except (InddError, UnicodeDecodeError):
        value = None
    return value if value is not None else Value.other(type_code, data)


def _decode(enc: Encoding, attr_id: int, type_code: int,
            data: bytes) -> Value:
    def f(offset):
        return enc.f64_at(data, offset)

    if attr_id == DASHES:
        n = enc.u32_at(data, 8)
        if (n is not None and len(data) - 14 == n * 8
                and all(f(12 + 8 * i) is not None for i in range(n))):
            corner = enc.u16_at(data, 12 + 8 * n)
            if corner is not None:
                return Value.dashes(
                    [f(12 + 8 * i) for i in range(n)], corner)
    if attr_id in OPACITY_STOPS:
        n = enc.u32_at(data, 0)
        if (n is not None and n > 0 and len(data) == 4 + n * 24):
            stops = []
            ok = True
            for i in range(n):
                stop = [f(4 + (3 * i + k) * 8) for k in range(3)]
                if any(v is None for v in stop):
                    ok = False
                    break
                stops.append(stop)
            if ok:
                return Value.stops(stops)
    if type_code == Ty.DOUBLE and len(data) == 8:
        value = f(0)
        if value is not None:
            return Value.double(value)
    if type_code == Ty.INT and len(data) == 4:
        value = enc.i32_at(data, 0)
        if value is not None:
            return Value.int(value)
    if type_code in (Ty.ENUM, Ty.CAPTION_POSITION) and len(data) == 2:
        value = enc.u16_at(data, 0)
        if value is not None:
            return Value.enum(value)
    if type_code == Ty.REF and len(data) == 4:
        value = enc.u32_at(data, 0)
        if value is not None:
            return Value.ref(value)
    if type_code == Ty.POINT and len(data) == 16:
        x, y = f(0), f(8)
        if x is not None and y is not None:
            return Value.point(x, y)
    if type_code in (Ty.STRING, Ty.FORM_STRING):
        try:
            cursor = enc.cursor(data)
            cursor.flag()
            return Value.string(cursor.string())
        except InddError:
            pass
    if len(data) == 4:
        value = enc.u32_at(data, 0)
        if value is not None:
            return Value.ref(value)
    return Value.other(type_code, data)
