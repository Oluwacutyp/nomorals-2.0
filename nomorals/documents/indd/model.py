"""A document model built from database objects (``src/model/``).

``Reader`` reads typed objects; ``Reader.document`` builds the
``Document``. The areas it reads are in spreads (spreads, pages,
layers, sections), items (page items and graphics), stories, styles,
settings (document-level settings and lists) and the other sections
of this file; ``ids`` holds the class and chunk IDs.

Evidence: ``docs/format/objects.md`` for the class and chunk layouts
used here.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from datetime import UTC
from decimal import Decimal
from typing import ClassVar

from . import ids as _ids
from .attrs import Attrs
from .database import Database
from .errors import InddError
from .header import Version
from .object import Cursor, Encoding, Name, builtin_key

Class = _ids.Class
Chunk = _ids.Chunk
Strand = _ids.Strand
WrapMode = _ids.WrapMode
RootKind = _ids.RootKind
Numbering = _ids.Numbering

DOC = 1
MAX_ITEM_DEPTH = 100
MAX_GROUP_DEPTH = 100


def _corrupt(message: str) -> InddError:
    return InddError.corrupt(message)


def _axial_shading_be(enc, data: bytes) -> bytes:
    """The shading data of an axial shade, re-encoded with every field
    big-endian, as IDML ``Contents`` holds it."""
    c = enc.cursor(data)
    out = bytearray()

    def u16s(n):
        for _ in range(n):
            out.extend(struct.pack(">H", c.u16()))

    def u32_be():
        v = c.u32()
        out.extend(struct.pack(">I", v))
        return v

    def f64s(n):
        if n > c.remaining() // 8:
            raise _corrupt(f"{n} shading numbers")
        for _ in range(n):
            out.extend(struct.pack(">d", c.f64()))

    def expect(what, v):
        got = u32_be()
        if got != v:
            raise _corrupt(f"shading {what} {got} is not known")

    f64s(6)
    u16s(4)
    expect("colour space", 2)
    functions = u32_be()
    if functions > c.remaining() // 50:
        raise _corrupt(f"{functions} shading functions")
    for _ in range(functions):
        expect("function type", 0)
        expect("function inputs", 1)
        expect("function outputs", 1)
        # Domain and range: u16, u32 count, pairs of f64; encode: u32,
        # u32 count, pairs; decode: u32 count, pairs.
        for lead in (2, 2, 4, 0):
            if lead == 2:
                u16s(1)
            elif lead == 4:
                u32_be()
            n = u32_be()
            f64s(n * 2)
        u16s(1)
        u32_be()
        expect("bits per sample", 8)
        samples = u32_be()
        out.extend(c.bytes(samples))
    expect("extend code", 4)
    f64s(6)
    expect("end code", 2)
    u16s(2)
    if c.remaining() != 0:
        raise _corrupt(f"{c.remaining()} bytes after the shading data")
    return bytes(out)


def uid_or_none(value: int) -> int | None:
    return value if value != 0 else None


def percent_decode(s: str) -> str:
    """``s`` with ``%xx`` escapes replaced by their bytes, read as UTF-8."""
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "%" and i + 3 <= len(s):
            try:
                out.append(int(s[i + 1:i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(ord(s[i]))
        i += 1
    return bytes(out).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Matrix:
    """A 2D affine transform ``[a b c d tx ty]``, as in IDML
    ``ItemTransform``."""

    values: tuple[float, float, float, float, float, float] = (
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    IDENTITY: ClassVar[Matrix]  # set after the class

    @classmethod
    def read(cls, cursor: Cursor) -> Matrix:
        return cls(tuple(cursor.f64() for _ in range(6)))

    def then(self, outer: Matrix) -> Matrix:
        """This transform followed by ``outer`` (points are row vectors,
        as in IDML)."""
        a, b, c, d, e, f = self.values
        p, q, r, s, t, u = outer.values
        return Matrix((a * p + b * r, a * q + b * s,
                       c * p + d * r, c * q + d * s,
                       e * p + f * r + t, e * q + f * s + u))


Matrix.IDENTITY = Matrix()


@dataclass(frozen=True)
class PathPoint:
    anchor: tuple[float, float]
    left: tuple[float, float]
    right: tuple[float, float]
    #: The stored point type: 2 a corner without direction points, 0 or
    #: 1 a point with them.
    kind: int


@dataclass
class Path:
    points: list[PathPoint] = field(default_factory=list)
    open: bool = False


class Shape:
    RECTANGLE = "rectangle"
    OVAL = "oval"
    POLYGON = "polygon"
    GRAPHIC_LINE = "graphic_line"


def _oval_anchors(path: Path) -> bool:
    """Whether the second and fourth anchors of a four-point path lie on
    the midpoint of the first and third, along the axis on which those
    two differ (within 1e-8)."""
    a = [p.anchor for p in path.points]
    if len(a) != 4:
        return False
    p0, p1, p2, p3 = a
    def on(m: float, u: float, v: float) -> bool:
        return abs(u - m) <= 1e-8 and abs(v - m) <= 1e-8
    if abs(p0[0] - p2[0]) < abs(p0[1] - p2[1]):
        return on((p0[1] + p2[1]) / 2.0, p1[1], p3[1])
    return on((p0[0] + p2[0]) / 2.0, p1[0], p3[0])


def classify(paths: list[Path], shape) -> str:
    """The IDML element of a frame or shape. ``shape`` is the u16 flag
    and the u32 shape code of chunk 0x6204: with flag 1 the code
    decides; with flag 0 or without the chunk, the path."""
    if shape is not None and shape[0] == 1:
        code = shape[1]
        if code == 1:
            return Shape.GRAPHIC_LINE
        if code in (2, 3):
            return Shape.RECTANGLE
        if code in (4, 5):
            return Shape.OVAL
        return Shape.POLYGON
    if len(paths) != 1:
        return Shape.POLYGON
    path = paths[0]
    corners = all(p.left == p.anchor and p.right == p.anchor
                  for p in path.points)
    if path.open and len(path.points) == 2 and corners:
        return Shape.GRAPHIC_LINE
    if not path.open and len(path.points) == 4 and corners:
        xs = [p.anchor[0] for p in path.points]
        ys = [p.anchor[1] for p in path.points]
        def distinct(values):
            rounded = sorted({round(v * 1e6) for v in values})
            return len(rounded)
        if distinct(xs) == 2 and distinct(ys) == 2:
            return Shape.RECTANGLE
    if (not path.open and len(path.points) == 4
            and all(p.left != p.anchor and p.kind == 0
                    for p in path.points)
            and _oval_anchors(path)):
        return Shape.OVAL
    return Shape.POLYGON


# ---------------------------------------------------------------------------
# Names stored as in-object strings (src/model/strings.rs)
# ---------------------------------------------------------------------------

def find_flagged_string(enc: Encoding, data: bytes, start: int,
                        accept=None):
    """Find a flag byte (1 = built-in name) followed by an in-object
    string that ``accept`` takes, at or after ``start``. Returns the
    offset of the pair, whether the flag is 1, and the string. In
    big-endian data the flag and the string's tag (2) are swapped."""
    big = enc.big_endian
    if accept is None:
        def accept(s):
            return True
    for i in range(start, max(start, len(data) - 6)):
        header = data[i] == 2 and data[i + 1] <= 2 if big else \
            data[i] <= 2 and data[i + 1] == 2
        if not header:
            continue
        try:
            name = enc.cursor(data[i:]).name()
        except InddError:
            continue
        if accept(name.name):
            return (i, name.builtin, name.name)
    return None


def find_string(enc: Encoding, data: bytes, start: int) -> str:
    """Find the first in-object string at or after ``start``. The byte
    after the tag can have any value."""
    if enc.big_endian:
        for i in range(start, max(start, len(data) - 5)):
            if (data[i] == 2 and data[i + 1] <= 2
                    and enc.u16_from(bytes([data[i + 3], data[i + 4]])) > 0):
                try:
                    return enc.cursor(data[i:]).string()
                except InddError:
                    continue
        return ""
    for i in range(start, max(start, len(data) - 4)):
        if data[i] == 2:
            n = enc.u16_from(bytes([data[i + 2], data[i + 3]]))
            if (n > 0 and i + 6 <= len(data)
                    and (enc.u16_from(bytes([data[i + 4], data[i + 5]]))
                         & 0xC000) != 0):
                try:
                    return enc.cursor(data[i:]).string()
                except InddError:
                    continue
    return ""


# ---------------------------------------------------------------------------
# Colours (src/model/color.rs)
# ---------------------------------------------------------------------------

class ColorClass:
    COLOR = 0x1F05
    SWATCH_NONE = 0x6E0B
    GRADIENT = 0x5503
    INK = 0x1F07
    COLOR_GROUP = 0x1F39


class ColorChunk:
    MIXED_INK_SPOTS = 0x102D
    COLOR_VALUE = 0x1F01
    COLOR_MODEL = 0x1F09
    COLOR_NAME = 0x1F10
    COLOR_ALTERNATE = 0x1F0A
    COLOR_OVERRIDE = 0x1F24
    TINT_BASE = 0x117
    GRADIENT_STOPS = 0x5503
    GRADIENT_NAME = 0x5505
    INK = 0x1F0D
    COLOR_GROUP_NAME = 0x13C
    COLOR_GROUP_SWATCHES = 0x1F60
    COLOR_GROUPS = 0x1F61


@dataclass
class GradientStop:
    color: int
    #: 0–1.
    location: float
    #: Position of the midpoint between this stop and the next one, 0–1,
    #: measured from the start of the gradient.
    midpoint: float


@dataclass
class Gradient:
    uid: int
    name: str
    builtin_name: bool
    #: 1 linear, 2 radial.
    kind: int
    stops: list[GradientStop] = field(default_factory=list)
    editable: bool = False
    removable: bool = False
    visible: bool = False

    def reference(self) -> str:
        if not self.name:
            return f"Gradient/u{self.uid:x}"
        return "Gradient/" + _escape_ref(self.name)

    def idml_midpoint(self, i: int) -> float | None:
        """IDML ``Midpoint`` of stop ``i`` (from 1): the midpoint between
        stops ``i``−1 and ``i``, in percent of the distance between
        them. ``None`` for the first stop, and when the value is outside
        the 13–87 that IDML allows."""
        if i < 1 or i >= len(self.stops):
            return None
        prev, stop = self.stops[i - 1], self.stops[i]
        if stop.location == prev.location:
            return None
        m = ((prev.midpoint - prev.location)
             / (stop.location - prev.location) * 100.0)
        m = _round6(m)
        return m if 13.0 <= m <= 87.0 else None

    def idml_name(self) -> str:
        return builtin_key(self.name) if self.builtin_name else self.name

    @classmethod
    def read(cls, uid: int, obj) -> Gradient | None:
        enc = obj.encoding
        stops = obj.chunk(ColorChunk.GRADIENT_STOPS)
        name = obj.chunk(ColorChunk.GRADIENT_NAME)
        if stops is None or name is None:
            return None
        c = enc.cursor(stops)
        n = c.u16()
        colors = [c.u32() for _ in range(n)]
        locations = [c.f64() for _ in range(n)]
        midpoints = [c.f64() for _ in range(n)]
        kind = c.u32()
        nc = enc.cursor(name)
        builtin_name = nc.flag() == 1
        name_str = nc.string()
        flags = nc.u32()
        return cls(uid=uid, name=name_str, builtin_name=builtin_name,
                   kind=kind,
                   stops=[GradientStop(color=colors[i],
                                       location=locations[i],
                                       midpoint=midpoints[i])
                          for i in range(n)],
                   removable=bool(flags & 1), visible=bool(flags & 2),
                   editable=bool(flags & 4))


def _escape_ref(name: str) -> str:
    return (name.replace("%", "%25").replace(":", "%3a")
            .replace("\r", "%0d"))


def _round6(value: float) -> float:
    """Round to 6 significant digits, as IDML writes numbers."""
    if value == 0:
        return 0.0
    import math as _math
    scale = 10 ** (6 - 1 - _math.floor(_math.log10(abs(value))))
    return round(value * scale) / scale


def _override_name(code: int) -> str:
    return {1: "Specialpaper", 2: "Specialblack",
            3: "Specialregistration", 4: "Hiddenreserved"}.get(code,
                                                                "Normal")


@dataclass
class Tint:
    """A tint swatch: a colour object with a base colour and a tint
    value instead of a name and colour values."""

    uid: int
    base: int
    #: Percent.
    value: float
    color_override: int

    @classmethod
    def read(cls, uid: int, obj) -> Tint | None:
        enc = obj.encoding
        if obj.chunk(ColorChunk.COLOR_NAME) is not None:
            return None
        base = obj.chunk(ColorChunk.TINT_BASE)
        tint = obj.chunk(ColorChunk.COLOR_OVERRIDE)
        if base is None or tint is None:
            return None
        base_uid = enc.cursor(base).u32()
        c = enc.cursor(tint)
        value = c.f64()
        color_override = c.u32()
        if base_uid == 0 or value < 0.0:
            return None
        return cls(uid=uid, base=base_uid, value=value,
                   color_override=color_override)

    def idml_name(self, base: Color) -> str:
        if base.color_override == 2:
            return self.named(f"[{base.name}]")
        return self.named(base.name)

    def named(self, base: str) -> str:
        return f"{base} {format_number(self.value)}%"

    def reference(self, base: Color) -> str:
        return self.reference_of(self.idml_name(base))

    @staticmethod
    def reference_of(name: str) -> str:
        return "Tint/" + _escape_ref(name)

    def override_name(self) -> str:
        return _override_name(self.color_override)


def format_number(value: float) -> str:
    """A number as IDML writes it: 6 significant digits, no trailing
    zeros, no exponent for the values documents hold."""
    if value == 0:
        return "0"
    rounded = _round6(value)
    if rounded == int(rounded) and abs(rounded) < 1e15:
        return str(int(rounded))
    text = f"{rounded:.6g}"
    return text


@dataclass
class MixedInk:
    uid: int
    name: str
    group: bool
    base: int | None
    inks: list[int] = field(default_factory=list)
    percentages: list[float] = field(default_factory=list)
    spots: list[int] = field(default_factory=list)
    editable: bool = False
    removable: bool = False
    visible: bool = False
    creator: int | None = None

    @classmethod
    def read(cls, color: Color, obj, group) -> MixedInk | None:
        enc = obj.encoding
        base = None
        base_chunk = obj.chunk(ColorChunk.TINT_BASE)
        if base_chunk is not None and color.space == ("other", 9):
            base = enc.cursor(base_chunk).u32()
        if color.model != 3 and base is None:
            return None
        source = group if base is not None else obj
        if source is None:
            return None
        model = source.chunk(ColorChunk.COLOR_MODEL)
        if model is None:
            return None
        c = enc.cursor(model)
        if c.u32() != 3:
            return None
        inks = c.u32_list()
        spots_chunk = source.chunk(ColorChunk.MIXED_INK_SPOTS)
        spots = (enc.cursor(spots_chunk).u32_list()
                 if spots_chunk is not None else [])
        return cls(uid=color.uid, name=color.name,
                   group=base is None
                   and obj.chunk(ColorChunk.COLOR_OVERRIDE) is not None,
                   base=base, inks=inks,
                   percentages=list(color.values), spots=spots,
                   editable=color.editable, removable=color.removable,
                   visible=color.visible, creator=color.creator)

    def reference(self) -> str:
        kind = "MixedInkGroup" if self.group else "MixedInk"
        return f"{kind}/" + _escape_ref(self.name)


def _hsb_to_rgb(h: float, s: float, b: float) -> list[float]:
    h6 = (h % 1.0) * 6.0
    sector = math.floor(h6)
    f = h6 - sector
    p, q, t = (b * (1.0 - s), b * (1.0 - s * f),
               b * (1.0 - s * (1.0 - f)))
    return [[b, t, p], [q, b, p], [p, b, t],
            [p, q, b], [t, p, b], [b, p, q]][sector % 6]


@dataclass
class Color:
    uid: int
    #: Empty for unnamed colours, which IDML names by UID.
    name: str = ""
    #: The name is an InDesign key, written with ``$ID/`` in IDML.
    builtin_name: bool = False
    model: int = 0
    #: ("rgb",) | ("cmyk",) | ("lab",) | ("hsb",) | ("other", code)
    space: tuple = ("cmyk",)
    #: Component values as fractions (0–1) for RGB and CMYK.
    values: list[float] = field(default_factory=list)
    editable: bool = False
    removable: bool = False
    visible: bool = False
    color_override: int = 0
    #: ``SwatchCreatorID``: the u32 at the end of chunk 0x1F10.
    creator: int | None = None
    #: The alternate colour (chunk 0x1F0A): space code and components.
    alternate: tuple | None = None

    def reference(self) -> str:
        if not self.name:
            return f"Color/u{self.uid:x}"
        return "Color/" + _escape_ref(self.name)

    def idml_name(self) -> str:
        return builtin_key(self.name) if self.builtin_name else self.name

    def model_name(self) -> str | None:
        return {0: "Process", 1: "Spot", 2: "Registration"}.get(self.model)

    def space_name(self) -> str:
        if self.space in (("rgb",), ("hsb",)):
            return "RGB"
        if self.space == ("lab",):
            return "LAB"
        return "CMYK"

    def idml_values(self) -> list[float]:
        if self.space == ("cmyk",):
            return [v * 100.0 for v in self.values]
        if self.space == ("rgb",):
            return [v * 255.0 for v in self.values]
        if self.space == ("hsb",) and len(self.values) == 3:
            return [v * 255.0 for v in _hsb_to_rgb(*self.values)]
        return list(self.values)

    def override_name(self) -> str:
        return _override_name(self.color_override)

    @classmethod
    def read(cls, uid: int, obj) -> Color | None:
        enc = obj.encoding
        name = obj.chunk(ColorChunk.COLOR_NAME)
        value = obj.chunk(ColorChunk.COLOR_VALUE)
        if name is None or value is None:
            return None
        c = enc.cursor(name)
        builtin_name = c.flag() == 1
        name_str = c.string()
        flags = c.u32()
        creator = None
        if c.remaining() >= 12:
            c.skip(8)
            creator = c.u32()
        alternate = None
        alt = obj.chunk(ColorChunk.COLOR_ALTERNATE)
        if alt is not None:
            a = enc.cursor(alt)
            space = a.u32()
            n = a.u16()
            alternate = (space, [a.f64() for _ in range(n)])
        v = enc.cursor(value)
        space_code = v.u32()
        space = {5: ("rgb",), 6: ("cmyk",), 7: ("lab",),
                 14: ("hsb",)}.get(space_code, ("other", space_code))
        n = v.u16()
        values = [v.f64() for _ in range(n)]
        model_chunk = obj.chunk(ColorChunk.COLOR_MODEL)
        model = enc.cursor(model_chunk).u32() if model_chunk is not None else 0
        override_chunk = obj.chunk(ColorChunk.COLOR_OVERRIDE)
        color_override = 0
        if override_chunk is not None and len(override_chunk) >= 12:
            color_override = enc.cursor(override_chunk[8:]).u32()
        return cls(uid=uid, name=name_str, builtin_name=builtin_name,
                   model=model, space=space, values=values,
                   removable=bool(flags & 1), visible=bool(flags & 2),
                   editable=bool(flags & 4),
                   color_override=color_override, creator=creator,
                   alternate=alternate)

    def idml_alternate(self):
        if self.alternate is None or self.alternate[0] == 3:
            return ("NoAlternateColor", [])
        code, values = self.alternate
        if code == 6:
            return ("CMYK", [x * 100.0 for x in values])
        if code == 7:
            return ("LAB", list(values))
        return None


@dataclass
class Ink:
    """An ink (class 0x1F07). Chunk 0x1F0D: flag byte and name, then
    fields at offsets from the end of the name: f64 neutral density at
    14, u32 trap order − 1 at 26, f64 frequency at 32, f64 angle at 40,
    u8 convert to process at 56."""

    uid: int
    name: Name = field(default_factory=Name)
    neutral_density: float | None = None
    trap_order: int = 0
    frequency: float = 0.0
    angle: float = 0.0
    convert_to_process: bool | None = None

    @classmethod
    def read(cls, uid: int, obj) -> Ink | None:
        enc = obj.encoding
        d = obj.chunk(ColorChunk.INK)
        if d is None:
            return None
        c = enc.cursor(d)
        name = c.name()
        end = c.pos
        if len(d) < end + 48:
            return None
        def f(o):
            return enc.cursor(d[end + o:]).f64()
        density = f(14)
        if density is not None and not (0.001 <= density <= 10.0):
            density = None
        convert = d[end + 56] if len(d) > end + 56 else None
        return cls(uid=uid, name=name, neutral_density=density,
                   trap_order=enc.cursor(d[end + 26:]).u32() + 1,
                   frequency=f(32), angle=f(40),
                   convert_to_process={0: False, 1: True}.get(convert))


@dataclass
class ColorGroup:
    """A colour group (class 0x1F39): chunk 0x13C is a flag byte and the
    name, chunk 0x1F60 a UID list of its swatches."""

    uid: int
    name: str = ""
    swatches: list[int] = field(default_factory=list)

    @classmethod
    def read(cls, uid: int, obj) -> ColorGroup | None:
        enc = obj.encoding
        d = obj.chunk(ColorChunk.COLOR_GROUP_NAME)
        if d is None:
            return None
        name = enc.cursor(d[1:]).string()
        swatches_chunk = obj.chunk(ColorChunk.COLOR_GROUP_SWATCHES)
        swatches = (enc.cursor(swatches_chunk).u32_list()
                    if swatches_chunk is not None else [])
        return cls(uid=uid, name=name, swatches=swatches)


# ---------------------------------------------------------------------------
# Fonts (src/model/font.rs)
# ---------------------------------------------------------------------------

class FontChunk:
    FAMILY = 0x3E05
    TYPEKIT_IDS = 0x3EEB


@dataclass
class Font:
    style: str = ""
    #: The style is a built-in key (a missing font's ``FontStyleName``).
    style_builtin: bool = False
    postscript_name: str = ""
    full_name: str = ""
    #: ``FontStyleNameNative`` as IDML writes it.
    style_native: str = ""
    full_name_native: str = ""
    #: Font type code (see :meth:`type_name`).
    font_type: int = 0
    version: str = ""
    #: IDML ``TypekitID``; the empty built-in key when the font has none.
    typekit_id: Name = field(default_factory=lambda: Name(True, ""))
    #: IDML ``PlatformName`` of a font in a missing-font record.
    platform_name: str | None = None

    def type_name(self) -> str | None:
        return {0: "Type1", 1: "TrueType", 3: "ATC", 6: "OpenTypeCFF",
                7: "OpenTypeCID", 8: "OpenTypeTT",
                0xFFFFFFFF: "Unknown"}.get(self.font_type)


@dataclass
class FontFamily:
    """A font family and the fonts of it that the document records."""

    uid: int
    name: str = ""
    #: The name is a built-in key (written with ``$ID/``).
    builtin: bool = False
    #: The family name in the font's own script.
    native_name: str = ""
    fonts: list[Font] = field(default_factory=list)
    #: IDML ``WritingScript`` of every font in the family.
    writing_script: int = 0

    def unescaped_name(self) -> str:
        """The family name with each ``<hhhh>`` escape (four hexadecimal
        digits of a UTF-16 code unit) replaced by its character."""
        units: list[int] = []
        rest = self.name
        while True:
            i = rest.find("<")
            if i < 0:
                break
            units.extend(_utf16_units(rest[:i]))
            text = rest[i + 1:i + 6]
            unit = None
            if (len(text) == 5 and text.endswith(">")
                    and all(c in "0123456789abcdefABCDEF"
                            for c in text[:4])):
                try:
                    unit = int(text[:4], 16)
                except ValueError:
                    unit = None
            if unit is not None:
                units.append(unit)
                rest = rest[i + 6:]
            else:
                units.append(ord("<"))
                rest = rest[i + 1:]
        units.extend(_utf16_units(rest))
        return _utf16_lossy(units)

    def referring(self, uid: int) -> FontFamily:
        """The family that a record of kind 2 makes of the family it
        refers to: the same name, fonts and writing script, with the
        font type ``Unknown``, no version and no Typekit ID."""
        return FontFamily(
            uid=uid, name=self.name, builtin=self.builtin,
            native_name=self.native_name,
            fonts=[Font(style=f.style, style_builtin=f.style_builtin,
                        postscript_name=f.postscript_name,
                        full_name=f.full_name,
                        style_native=f.style_native,
                        full_name_native=f.full_name_native,
                        font_type=0xFFFFFFFF, version="",
                        typekit_id=Name(True, ""),
                        platform_name=f.platform_name)
                   for f in self.fonts],
            writing_script=self.writing_script)

    @classmethod
    def read(cls, uid: int, obj) -> FontFamily | None:
        data = obj.chunk(FontChunk.FAMILY)
        if data is None:
            return None
        # Files from InDesign 3.0 and 4.0 use older font records; one is
        # taken only if it reads the chunk to its end.
        try:
            family = _parse_font_family(obj.encoding, uid, data, "current")
        except InddError:
            try:
                family = _parse_font_family(obj.encoding, uid, data, "v4")
            except InddError:
                family = _parse_font_family(obj.encoding, uid, data, "v3")
        ids = obj.chunk(FontChunk.TYPEKIT_IDS)
        if ids is not None:
            for font, typekit_id in zip(family.fonts,
                                        _typekit_ids(obj.encoding, ids),
                                        strict=False):
                font.typekit_id = typekit_id
        return family


def _utf16_units(text: str) -> list[int]:
    return list(struct.unpack(f"<{len(text.encode('utf-16-le')) // 2}H",
                              text.encode("utf-16-le"))) if text else []


def _utf16_lossy(units: list[int]) -> str:
    raw = b"".join(struct.pack("<H", u) for u in units)
    return raw.decode("utf-16-le", errors="replace")


def _parse_font_family(enc: Encoding, uid: int, data: bytes,
                       record: str) -> FontFamily:
    c = enc.cursor(data)
    kind = c.u8()
    c.u8()
    builtin = c.flag() == 1
    name = c.string()
    c.flag()
    native_name = c.string()
    if kind == 3 and record == "current":
        return FontFamily(uid=uid, name=name, builtin=builtin,
                          native_name=native_name,
                          fonts=_missing_fonts(c), writing_script=0)
    c.skip(6)
    count = c.u16()
    fonts = []
    for _ in range(count):
        c.skip(1)
        style = c.string()
        if record != "current":
            c.skip(1)
            postscript_name = c.string()
        else:
            n = c.u16()
            postscript_name = "".join(chr(b) for b in c.bytes(n))
        c.skip(1)
        full_name = c.string()
        native_builtin = c.flag() == 1
        style_native = c.string()
        if native_builtin:
            style_native = builtin_key(style_native)
        c.skip(1)
        full_name_native = c.string()
        font_type = c.u32()
        n = 0 if record == "v3" else c.u32()
        version = "" if n == 0 else c.segments(n)
        fonts.append(Font(style=style, postscript_name=postscript_name,
                          full_name=full_name,
                          style_native=style_native,
                          full_name_native=full_name_native,
                          font_type=font_type, version=version))
    writing_script = c.u32()
    if record != "current" and c.remaining() != 0:
        raise _corrupt(f"font family {uid}: unknown record")
    return FontFamily(uid=uid, name=name, builtin=builtin,
                      native_name=native_name, fonts=fonts,
                      writing_script=writing_script)


def _missing_fonts(c: Cursor) -> list[Font]:
    if c.remaining() == 6:
        c.skip(6)
        return []
    c.skip(4)
    count = c.u16()
    fonts = []
    for _ in range(count):
        style = c.name()
        platform = c.name()
        c.name()
        c.skip(10)
        fonts.append(Font(style=style.name, style_builtin=style.builtin,
                          style_native=platform.idml(),
                          font_type=0xFFFFFFFF,
                          platform_name=platform.idml()))
    if c.remaining() != 0:
        raise _corrupt("missing-font record: bytes left")
    return fonts


def _typekit_ids(enc: Encoding, data: bytes) -> list[Name]:
    c = enc.cursor(data)
    n = c.u32()
    return [c.name() for _ in range(n)]


def font_reference(enc: Encoding, data: bytes):
    """A record of kind 2 in chunk 0x3E05: u8 2, u8, u32 UID of another
    font family, u16 count of the font entries that follow. Returns the
    UID and the count."""
    c = enc.cursor(data)
    if c.u8() != 2:
        return None
    c.u8()
    return (c.u32(), c.u16())


# ---------------------------------------------------------------------------
# Styles (src/model/style.rs)
# ---------------------------------------------------------------------------

@dataclass
class ExportTagMap:
    export_type: str = ""
    tag: str = ""
    class_name: str = ""
    attributes: list = field(default_factory=list)


@dataclass
class StyleExport:
    """The export settings of a paragraph, character or object style
    (chunk 0x28F0): tag maps, then u16 flags."""

    maps: list[ExportTagMap] = field(default_factory=list)
    flags: list[int] = field(default_factory=list)

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> StyleExport:
        c = enc.cursor(d)

        def string(cursor: Cursor) -> str:
            n = cursor.u32()
            if n == 0:
                return ""
            if n > cursor.remaining():
                raise _corrupt(f"export tag map string of {n}")
            return cursor.segments(n)

        n = c.u32()
        if n > c.remaining() // 16:
            raise _corrupt(f"{n} export tag maps")
        maps = []
        for _ in range(n):
            export_type = string(c)
            tag = string(c)
            class_name = string(c)
            k = c.u32()
            if k > c.remaining() // 8:
                raise _corrupt(f"{k} export tag attributes")
            attributes = [(string(c), string(c)) for _ in range(k)]
            maps.append(ExportTagMap(export_type, tag, class_name,
                                     attributes))
        flags = []
        while c.remaining() >= 2:
            flags.append(c.u16())
        return cls(maps=maps, flags=flags)


@dataclass
class Style:
    uid: int
    name: str = ""
    #: The name is an InDesign built-in key, written with ``$ID/``.
    builtin: bool = False
    paragraph: bool = False
    based_on: int | None = None
    next: int | None = None
    attrs: Attrs = field(default_factory=Attrs)
    #: The u16 after the kind, before the name (IDML ``Imported``).
    imported: bool = False
    #: The GUID string after the name in newer files.
    unique_id: str | None = None
    #: Keyboard shortcut: u32 key 10 bytes before the name's flag byte,
    #: then the two bytes 6 and 5 before it.
    shortcut: tuple | None = None
    export: StyleExport | None = None
    #: ``PreviewColor``: the u32 14 bytes before the name's flag byte, 0
    #: for none or an interface colour. ``None`` for another value or a
    #: shorter header.
    preview_color: object = None


@dataclass
class TocEntry:
    name: str = ""
    builtin: bool = False
    format_style: int = 0
    level: int = 0
    page_number_position: int = 0
    page_number_style: int = 0
    separator: str = ""
    separator_style: int = 0


@dataclass
class TocStyle:
    uid: int
    name: str = ""
    builtin: bool = False
    title: str = ""
    title_style: int = 0
    include_book_documents: bool = False
    create_bookmarks: bool = False
    flags: list[int] = field(default_factory=list)
    entries: list[TocEntry] = field(default_factory=list)


@dataclass
class StrokeStyle:
    """A custom stroke style."""

    uid: int
    #: Dashed (class 0x5A43) or striped (class 0xB016).
    dashed: bool = True
    name: str = ""
    #: Dashed: dash and gap lengths in points. Striped: start and width
    #: of each stripe, as fractions of the stroke weight.
    values: list[float] = field(default_factory=list)
    #: Dashed: corner adjustment code and end cap code.
    corner: int = 0
    cap: int = 0


@dataclass
class StyleGroup:
    uid: int
    #: Empty for root groups.
    name: str = ""
    #: Root groups: what they hold, from chunk 0x28C2.
    root: int | None = None
    children: list[int] = field(default_factory=list)


@dataclass
class AnchorSettings:
    """Anchored object settings (chunk 0x2800, of an anchor, an object
    style or the preferences). A field is ``None`` where the chunk is
    too short for it."""

    y_offset: float | None = None
    x_offset: float | None = None
    position: int | None = None
    horizontal_reference: int | None = None
    vertical_reference: int | None = None
    point: tuple | None = None
    horizontal_alignment: int | None = None
    vertical_alignment: int | None = None
    spine_relative: int | None = None
    pin: int | None = None

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> AnchorSettings:
        def u(o):
            return enc.u16_at(d, o)
        return cls(y_offset=enc.f64_at(d, 0), x_offset=enc.f64_at(d, 16),
                   position=u(40), horizontal_reference=u(42),
                   vertical_reference=u(44),
                   point=(u(46), u(50)) if u(46) is not None
                   and u(50) is not None else None,
                   horizontal_alignment=u(48),
                   vertical_alignment=u(52), spine_relative=u(54),
                   pin=u(56))

    @classmethod
    def absent(cls) -> AnchorSettings:
        return cls(y_offset=0.0, x_offset=0.0, position=0,
                   horizontal_reference=1, vertical_reference=4,
                   point=(0, 2), horizontal_alignment=2,
                   vertical_alignment=0, spine_relative=0, pin=1)


@dataclass
class BaselineGrid:
    """Baseline frame grid settings: chunk 0x2834 of a multi-column
    frame, and the text frame settings of an object style at 82."""

    use_custom: bool = False
    start: float = 0.0
    relative: int = 3
    increment: float = 12.0
    color: int = 0

    DEFAULT: ClassVar[BaselineGrid] = None

    @classmethod
    def read(cls, enc: Encoding, d: bytes, at: int):
        use_custom = enc.u16_at(d, at)
        start = enc.f64_at(d, at + 2)
        relative = enc.u16_at(d, at + 10)
        increment = enc.f64_at(d, at + 12)
        color = enc.u32_at(d, at + 20)
        if None in (use_custom, start, relative, increment, color):
            return None
        return cls(use_custom=use_custom == 1, start=start,
                   relative=relative, increment=increment, color=color)


BaselineGrid.DEFAULT = BaselineGrid()


@dataclass
class ObjectStyleFrame:
    """Text frame settings of an object style (chunk 0x1B924), at the
    offsets known for its sizes (106, 142, 162 and 222 bytes). Each
    field is ``None`` if the chunk is too short for it."""

    column_fixed_width: float | None = None
    column_gutter: float | None = None
    insets: list | None = None
    column_count: int | None = None
    footnotes: tuple | None = None
    column_rule: tuple | None = None
    first_baseline_offset: int | None = None
    vertical_balance_columns: bool | None = None
    use_fixed_width: bool | None = None
    vertical_justification: int | None = None
    auto_sizing: tuple | None = None
    minimum_sizes: tuple | None = None
    no_line_breaks: bool | None = None
    baseline_grid: BaselineGrid | None = None

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> ObjectStyleFrame:
        def f(o):
            return enc.f64_at(d, o)
        long = len(d) >= 142
        def flag(o):
            return (enc.u16_at(d, o) == 1) if long else None
        def code(o):
            return enc.u16_at(d, o) if long else None
        insets = None
        values = [f(34), f(42), f(50), f(58)]
        if all(v is not None for v in values):
            insets = values
        footnotes = None
        if (enc.u16_at(d, 144) is not None and f(146) is not None
                and f(154) is not None):
            footnotes = (enc.u16_at(d, 144), f(146), f(154))
        column_rule = None
        if (f(190) is not None and enc.u32_at(d, 198) is not None
                and f(210) is not None):
            column_rule = (f(190), enc.u32_at(d, 198), f(210))
        auto_sizing = None
        if code(116) is not None and code(118) is not None:
            auto_sizing = (code(116), code(118))
        minimum_sizes = None
        if (flag(120) is not None and flag(130) is not None
                and f(122) is not None and f(132) is not None):
            minimum_sizes = ([flag(120), flag(130)], [f(122), f(132)])
        return cls(column_fixed_width=f(0), column_gutter=f(8),
                   insets=insets, column_count=enc.u32_at(d, 66),
                   footnotes=footnotes, column_rule=column_rule,
                   first_baseline_offset=enc.u16_at(d, 76),
                   vertical_balance_columns=flag(32),
                   use_fixed_width=flag(70),
                   vertical_justification=code(74),
                   auto_sizing=auto_sizing,
                   minimum_sizes=minimum_sizes,
                   no_line_breaks=flag(140),
                   baseline_grid=BaselineGrid.read(enc, d, 82))


@dataclass
class StorySettings:
    """Story settings of an object style (chunk 0x285B), if the chunk
    has 16 bytes: u16 story orientation at 0, f64 optical margin size
    at 2, u16 optical margin alignment at 12 and u16 frame type at 14."""

    orientation: int = 0
    optical_size: float = 0.0
    optical_alignment: int = 0
    frame_type: int = 0

    @classmethod
    def read(cls, enc: Encoding, d: bytes):
        if len(d) < 16:
            return None
        orientation = enc.u16_at(d, 0)
        optical_size = enc.f64_at(d, 2)
        optical_alignment = enc.u16_at(d, 12)
        frame_type = enc.u16_at(d, 14)
        if None in (orientation, optical_size, optical_alignment,
                    frame_type):
            return None
        return cls(orientation=orientation, optical_size=optical_size,
                   optical_alignment=optical_alignment,
                   frame_type=frame_type)


@dataclass
class ObjectStyle:
    uid: int
    name: str = ""
    builtin: bool = False
    based_on: int | None = None
    fitting: Attrs = field(default_factory=Attrs)
    attrs: Attrs = field(default_factory=Attrs)
    flex_spacing: list | None = None
    transparency: Attrs = field(default_factory=Attrs)
    frame: ObjectStyleFrame | None = None
    story: StorySettings | None = None
    direction: int | None = None
    text_wrap: object = None
    contour: object = None
    enabled: list | None = None
    paragraph_style: int | None = None
    shortcut_key: int | None = None
    apply_next: int | None = None
    anchor: AnchorSettings | None = None
    named_grid: object = None
    export: StyleExport | None = None
    export_options: object = None
    transform: list = field(default_factory=lambda: [None] * 4)


def _is_guid(s: str) -> bool:
    groups = s.split("-")
    return ([len(g) for g in groups] == [8, 4, 4, 4, 12]
            and all(all(c.isdigit() or c in "abcdef" for c in g)
                    for g in groups))


def prune_style_groups(groups: dict, warn) -> None:
    """Make the style groups a tree under their root groups, so that
    walking them ends: a group listed a second time (in a cycle, or
    under a second parent) or nested deeper than MAX_GROUP_DEPTH is
    removed from the children of the group that lists it, with a
    warning."""
    roots = [uid for uid, g in groups.items() if g.root is not None]
    seen = set(roots)
    stack = [(r, 0) for r in roots]
    while stack:
        uid, depth = stack.pop()
        group = groups.get(uid)
        if group is None:
            continue
        children = list(group.children)
        kept = []
        for child in children:
            if child in groups:
                if child in seen:
                    warn(f"style group {child} is listed twice; "
                         f"left out of group {uid}")
                    continue
                if depth + 1 > MAX_GROUP_DEPTH:
                    warn(f"style group {child} is nested more than "
                         f"{MAX_GROUP_DEPTH} deep; left out")
                    continue
                stack.append((child, depth + 1))
            kept.append(child)
        groups[uid].children = kept


def _toc_entries(c: Cursor) -> list[TocEntry]:
    n = c.u16()
    out = []
    for _ in range(n):
        builtin = c.flag() == 1
        name = c.string()
        c.flag()
        c.string()
        format_style = c.u32()
        level = c.u16()
        c.u16()
        page_number_position = c.u16()
        page_number_style = c.u32()
        length = c.u32()
        separator = c.segments(length)
        separator_style = c.u32()
        c.u16()
        length = c.u32()
        c.segments(length)
        c.u16()
        out.append(TocEntry(name=name, builtin=builtin,
                            format_style=format_style, level=level,
                            page_number_position=page_number_position,
                            page_number_style=page_number_style,
                            separator=separator,
                            separator_style=separator_style))
    if c.remaining() != 0:
        raise _corrupt("TOC entries do not fill the chunk")
    return out


# ---------------------------------------------------------------------------
# Tables (src/model/table.rs)
# ---------------------------------------------------------------------------

class TableClass:
    CELL_STYLE = 0x2021A
    TABLE_STYLE = 0xB63F
    TABLE_ANCHOR = 0xB651
    TABLE = 0xB608
    CELL_STRAND_OWNER = 0xB606
    GRAPHIC_CELL = 0x1046B


class TableChunk:
    ANCHOR_TABLE = 0xB67D
    TABLE_SIZE = 0xB608
    TABLE_ROWS = 0xB616
    TABLE_COLUMNS = 0xB617
    TABLE_PARTS = 0xB6FB
    TABLE_STYLE = 0xB6FC
    TABLE_ATTRS = 0xB668
    CELL_STYLE_ATTRS = 0x20253
    TABLE_STYLE_ATTRS = 0xB667
    CELL_STYLE_ROOT_CHILDREN = 0x2024E
    TABLE_STYLE_ROOT_CHILDREN = 0x104E5
    TABLE_DIRECTION = 0x50F65
    GRAPHIC_CELL_ITEMS = 0x10466


class TableAttr:
    COLUMN_WIDTH = 0xB60D
    ROW_HEIGHT = 0xB60C
    ROW_MIN_HEIGHT = 0xB66E
    CELL = 0xB666
    GRAPHIC_CELL = 0x10469
    COVERED = 0xB614


@dataclass
class Row:
    height: float | None = None
    min_height: float | None = None
    attrs: Attrs = field(default_factory=Attrs)


class CellKind:
    TEXT = "text"
    GRAPHIC = "graphic"


@dataclass
class Cell:
    id: int = 0
    row: int = 0
    column: int = 0
    row_span: int = 1
    column_span: int = 1
    kind: str = CellKind.TEXT
    width: float | None = None
    runs: list = field(default_factory=list)
    items: list = field(default_factory=list)
    format: int | None = None


@dataclass
class CellFormat:
    attrs: Attrs = field(default_factory=Attrs)
    style_priority: int = 0
    style: int = 0


@dataclass
class TableStyle:
    uid: int
    name: str = ""
    builtin: bool = False
    based_on: int | None = None
    attrs: Attrs = field(default_factory=Attrs)


@dataclass
class Table:
    uid: int
    style: int | None = None
    attrs: Attrs = field(default_factory=Attrs)
    right_to_left: bool = False
    header_rows: int = 0
    footer_rows: int = 0
    rows: list[Row] = field(default_factory=list)
    columns: list[float] = field(default_factory=list)
    cells: list[Cell] = field(default_factory=list)
    formats: list[CellFormat] = field(default_factory=list)
    grid: list = field(default_factory=list)

    def format(self, cell: Cell):
        if cell.format is None:
            return None
        return (self.formats[cell.format]
                if cell.format < len(self.formats) else None)

    def edge_formats(self, cell: Cell):
        def at(r, c):
            if r >= len(self.grid):
                return None
            row = self.grid[r]
            if c >= len(row):
                return None
            index = row[c]
            if index is None or index >= len(self.formats):
                return None
            return self.formats[index]
        rows = range(cell.row, cell.row + cell.row_span)
        columns = range(cell.column, cell.column + cell.column_span)
        last_row = cell.row + cell.row_span - 1
        last_column = cell.column + cell.column_span - 1
        return [[at(r, cell.column) for r in rows],
                [at(r, last_column) for r in rows],
                [at(cell.row, c) for c in columns],
                [at(last_row, c) for c in columns]]

    def value(self, attr_id: int, styles):
        found = self.attrs.get(attr_id)
        if found is not None:
            return found
        for style in styles.table_chain(self.style):
            found = style.attrs.get(attr_id)
            if found is not None:
                return found
        return None

    def _region_style(self, row, column, span, styles):
        chain = styles.table_chain(self.style)
        def get(attr_id):
            for style in chain:
                value = style.attrs.get(attr_id)
                if value is not None:
                    return value.as_u32()
            return None
        nrows = len(self.rows)
        if row < self.header_rows:
            region = (0x10450, 0x10457)
        elif self.footer_rows > 0 and row + self.footer_rows >= nrows:
            region = (0x10451, 0x10458)
        elif column == 0:
            region = (0x10453, 0x10459)
        elif column + span >= len(self.columns):
            region = (0x10454, 0x1045A)
        else:
            region = None
        attr_id = region[0] if region is not None and \
            get(region[1]) != 1 else 0x10452
        value = get(attr_id)
        return value if value not in (None, 0) else None

    def cell_value(self, cell: Cell, attr_id: int, styles):
        return self._cell_value_at(cell, cell.column_span, attr_id, styles)

    def _cell_value_at(self, cell, span, attr_id, styles):
        fmt = self.format(cell)
        if fmt is not None:
            found = fmt.attrs.get(attr_id)
            if found is not None:
                return found
            if fmt.style != 0:
                found = styles.cell_style_value(fmt.style, attr_id)
                if found is not None:
                    return found
        region = self._region_style(cell.row, cell.column, span, styles)
        if region is not None:
            found = styles.cell_style_value(region, attr_id)
            if found is not None:
                return found
        return self.value(attr_id, styles)

    def resolve_spans(self, styles) -> None:
        """Set the cells' column and row spans. A cell's column span is
        the number of columns, up to the next cell in its row, whose
        widths best match its laid-out width plus its left and right
        insets; its row span takes the rows below whose positions under
        it are covered and not yet taken."""
        nrows = min(len(self.rows), len(self.grid))
        starts = [[False] * len(r) for r in self.grid]
        for cell in self.cells:
            starts[cell.row][cell.column] = True
        taken = [list(row) for row in starts]
        spans = []
        for cell in self.cells:
            row = starts[cell.row]
            run = 1
            for start in row[cell.column + 1:]:
                if start:
                    break
                run += 1
            if cell.width is not None and cell.width > 0:
                def inset(n, attr_id, cell=cell):
                    value = self._cell_value_at(cell, n, attr_id, styles)
                    return value.as_f64() if value is not None else 0.0
                best = (1, math.inf)
                total = 0.0
                for n in range(1, run + 1):
                    total += (self.columns[cell.column + n - 1]
                              if cell.column + n - 1 < len(self.columns)
                              else 0.0)
                    target = (cell.width + inset(n, 0xB62B)
                              + inset(n, 0xB62D))
                    d = abs(total - target)
                    if d < best[1]:
                        best = (n, d)
                span = best[0]
            else:
                span = run
            for i in range(cell.column,
                           min(cell.column + span, len(taken[cell.row]))):
                taken[cell.row][i] = True
            spans.append(span)
        for cell, span in zip(self.cells, spans, strict=False):
            cell.column_span = span
            rows = 1
            for row in taken[cell.row + 1:nrows]:
                part = row[cell.column:cell.column + span]
                if len(part) == span and all(not t for t in part):
                    for i in range(cell.column, cell.column + span):
                        row[i] = True
                    rows += 1
                else:
                    break
            cell.row_span = rows


class TableStyles:
    """The cell and table styles of a document, for values that tables
    and cells inherit."""

    def __init__(self, cells, tables):
        self.cells = cells
        self.tables = tables
        self.root = next((s for s in tables.values()
                          if s.builtin and s.based_on is None
                          and s.name == "[No table style]"), None)

    @staticmethod
    def _chain(styles, uid):
        out = []
        nxt = uid
        while nxt is not None:
            style = styles.get(nxt)
            if style is None or any(s.uid == nxt for s in out):
                break
            out.append(style)
            nxt = style.based_on
        return out

    def table_chain(self, style):
        out = self._chain(self.tables, style)
        if self.root is not None and all(s.uid != self.root.uid
                                         for s in out):
            out.append(self.root)
        return out

    def cell_style_value(self, uid, attr_id):
        for style in self._chain(self.cells, uid):
            value = style.attrs.get(attr_id)
            if value is not None:
                return value
        return None


def _table_groups(enc: Encoding, data: bytes):
    """Groups of rows or columns sharing attributes: u32 group count;
    per group u32 count, u16, u16 attribute count, attributes, 8
    bytes."""
    c = enc.cursor(data)
    n = c.u32()
    out = []
    for _ in range(n):
        count = c.u32()
        c.u16()
        na = c.u16()
        attrs = Attrs.parse_text(c, na)
        c.skip(8)
        out.append((count, attrs))
    return out


def _raw_attrs(c: Cursor, n: int):
    """Attribute records keeping the raw payload of each value."""
    enc = c.encoding()
    out = []
    for _ in range(n):
        attr_id = c.u32()
        size = c.u16()
        payload = c.bytes(size)
        values = []
        if size >= 2:
            p = enc.cursor(payload)
            count = p.u16()
            for _ in range(count):
                p.u32()
                length = p.u16()
                values.append(p.bytes(length))
        out.append((attr_id, values))
    return out


def _layout_width(enc: Encoding, v: bytes):
    """Text area width of a cell's layout record: u32 flags, u32 parcel
    count (and one more u32 if the low four flag bits are 0xF), then per
    parcel u32 flags (and one more u32 if bit 31 is set), 16 bytes, f64
    width, f64 height. ``None`` for a cell never laid out (no parcels)
    or a record too short."""
    try:
        c = enc.cursor(v)
        flags = c.u32()
        if c.u32() == 0:
            return None
        if flags & 0xF == 0xF:
            c.u32()
        if c.u32() & 0x80000000 != 0:
            c.u32()
        c.skip(16)
        return c.f64()
    except InddError:
        return None


# ---------------------------------------------------------------------------
# Spreads, pages, layers, guides, sections (src/model/spread.rs)
# ---------------------------------------------------------------------------

@dataclass
class LayerSettings:
    """Layer settings in chunk 0x304."""

    printable: bool = True
    lock_guides: bool = False
    ui: bool = False
    ignore_wrap: bool = False
    #: The layer colour (an interface colour object).
    color: tuple | None = None

    @classmethod
    def read(cls, enc: Encoding, d: bytes):
        if len(d) < 17:
            return None
        printable = enc.u16_at(d, 0) == 1
        lock_guides = enc.u16_at(d, 2) == 1
        ui = enc.u16_at(d, 4) == 1
        ignore_wrap = enc.u16_at(d, 6) == 1
        color = None
        if len(d) >= 40:
            values = [enc.f64_at(d, 16), enc.f64_at(d, 24),
                      enc.f64_at(d, 32)]
            if all(v is not None for v in values):
                color = tuple(values)
        return cls(printable=printable, lock_guides=lock_guides, ui=ui,
                   ignore_wrap=ignore_wrap, color=color)


@dataclass
class Layer:
    uid: int
    name: str = ""
    visible: bool = True
    locked: bool = False
    #: The hidden layer that holds pages; not written to IDML.
    internal: bool = False
    #: The other settings, when chunk 0x304 has the layout of the
    #: samples.
    settings: LayerSettings | None = None


class UiColorRef:
    """A colour setting that names an interface colour (class 0x1F11),
    or one of two codes."""

    USE_MASTER = "use_master"
    NOTHING = "nothing"
    RGB = "rgb"
    UNKNOWN = "unknown"


@dataclass
class PageSettings:
    """Settings of a page."""

    #: Master page items overridden on the page, each with its override
    #: (0 for none), from chunk 0x1404.
    overrides: list = field(default_factory=list)
    #: Chunk 0xCD04: whether the page uses its master's layout grid.
    use_master_grid: bool | None = None
    #: Chunk 0xCD04: the layout grid starting point code (u32 at 0).
    grid_start: int | None = None
    #: Layout rule code (chunk 0x563); ``None`` without the chunk.
    layout_rule: int | None = None
    color: tuple = (UiColorRef.USE_MASTER, None)
    #: Page items in tab order (the spread's chunk 0x14580).
    tab_order: list = field(default_factory=list)


@dataclass
class Margins:
    """Page margins from chunk 0x51A: four f64 (left, top, right,
    bottom), then u16 1 if the page has its own margins, 0 if it uses
    its master page's."""

    left: float = 0.0
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    own: bool = False


@dataclass
class Columns:
    """Page columns from chunk 0x528: u32 count *n*, *n* f64 column edge
    positions, f64 gutter, then u16 1 if the page has its own columns."""

    positions: list = field(default_factory=list)
    gutter: float = 0.0
    own: bool = False
    #: The last u16 of the chunk: 1 for vertical columns, 0 horizontal.
    direction: int | None = None


@dataclass
class GridData:
    """Layout grid settings of a page (chunk 0xCD02): u32 font family,
    a flag byte, the font style as an in-object string, five f64 and
    four u32."""

    font: int = 0
    #: The font style as IDML writes it (``$ID/`` for a built-in key).
    font_style: str = ""
    numbers: list = field(default_factory=lambda: [0.0] * 5)
    codes: list = field(default_factory=lambda: [0] * 4)

    @classmethod
    def read(cls, c: Cursor) -> GridData:
        return cls(font=c.u32(), font_style=c.name().idml(),
                   numbers=[c.f64() for _ in range(5)],
                   codes=[c.u32() for _ in range(4)])


@dataclass
class NamedGrid:
    """A named grid (class 0xCD12)."""

    uid: int
    builtin: bool = False
    name: str = ""
    #: Its own layout grid settings (chunk 0xCD02), if it has them.
    grid: GridData | None = None


@dataclass
class Page:
    uid: int
    #: Left, top, right, bottom in page coordinates.
    bounds: list = field(default_factory=lambda: [0.0] * 4)
    transform: Matrix = field(default_factory=Matrix)
    master: int | None = None
    master_transform: Matrix = field(default_factory=Matrix)
    #: Margins and columns (chunks 0x51A and 0x528). After
    #: :func:`resolve_page_layout`, these are the values in effect: the
    #: page's own, or its master page's.
    margins: Margins | None = None
    columns: Columns | None = None
    #: Layout grid settings (chunk 0xCD02).
    grid: GridData | None = None
    settings: PageSettings = field(default_factory=PageSettings)


@dataclass
class Spread:
    uid: int
    #: Master spreads only: name prefix (for example "A") and base name.
    master_name: tuple | None = None
    transform: Matrix = field(default_factory=Matrix)
    binding_location: int = 0
    pages: list = field(default_factory=list)
    items: list = field(default_factory=list)
    guides: list = field(default_factory=list)
    #: Chunk 0x1A8: ``None`` without it.
    shuffle: int | None = None
    #: The line art and text and the gradient and mesh resolutions in
    #: chunk 0x10833, when it is there.
    flattener_resolution: tuple | None = None
    #: Master spreads: chunk 0x140D.
    show_master_items: int | None = None
    #: Master spreads: the story of the primary text frame (chunk
    #: 0x140A); ``None`` without the chunk.
    primary_story: int | None = None
    #: Page transition type and direction codes (chunk 0x2050C, u32 at 0
    #: and 8); ``None`` without the chunk.
    transition: tuple | None = None
    #: Timing lists (chunk 0x2056D).
    timing: list = field(default_factory=list)


@dataclass
class Guide:
    """A ruler guide (class 0x3301, chunk 0x3308)."""

    uid: int
    horizontal: bool = False
    #: Position in spread coordinates: y for a horizontal guide, x for
    #: a vertical one.
    position: float = 0.0
    #: The page the guide belongs to, or the spread.
    owner: int = 0
    fit_to_page: bool = False
    #: Stored view threshold (0.05 in every sample).
    view_threshold: float = 0.05
    #: Colour code (6 in every sample).
    color: int = 6
    #: 0 ruler guide, 1 liquid guide; none in 40-byte records.
    guide_type: int | None = None
    #: Document layer.
    layer: int = 0
    #: Chunk 0x2C2D, as for page items.
    locked: bool = False
    #: f64 at 44 of the 52-byte record (``GuideZone``).
    zone: float | None = None
    #: Chunk 0x1424, as for page items.
    overridden: tuple | None = None


@dataclass
class Section:
    """Numbering of a section."""

    uid: int
    #: The section's first page; ``None`` for the first section, which
    #: starts at the first page of the document.
    page: int | None = None
    continue_numbering: bool = False
    #: ``IncludeSectionPrefix``: page names start with the prefix.
    include_prefix: bool = False
    start: int = 1
    #: Page number style code (see :func:`numbering`).
    style: int = 0
    #: ``SectionPrefix`` and ``Marker``; empty if the chunk is missing.
    prefix: str = ""
    marker: str = ""
    #: The alternate layout name after the numbers, with the flag byte
    #: before it (1: a built-in key); ``None`` if the chunk ends before
    #: it.
    alternate_layout: tuple | None = None


def _left_to_right(spread: Spread) -> list[int]:
    order = list(range(len(spread.pages)))
    order.sort(key=lambda i: spread.pages[i].transform.values[4])
    return order


def resolve_page_layout(spreads: list, masters: list) -> None:
    """Replace the margins and columns that pages take from their
    master page (flag 0) with the values in effect on that master page.
    A page's master page is the page at the same position, counted from
    the left, in its applied master spread, or that spread's only page."""

    def master_page(raw, master, index):
        if master is None:
            return None
        pages = raw.get(master)
        if pages is None:
            return None
        i = 0 if len(pages) == 1 else index
        if i >= len(pages):
            return None
        return (i, pages[i])

    def effective(raw, get, page, index, depth):
        own, is_own = get(page)
        if is_own or depth > 8:
            return own
        found = master_page(raw, page[0], index)
        if found is not None:
            i, m = found
            value = effective(raw, get, m, i, depth + 1)
            return value if value is not None else own
        return own

    raw = {}
    for s in masters:
        pages = []
        for i in _left_to_right(s):
            p = s.pages[i]
            pages.append((p.master, p.margins, p.columns))
        raw[s.uid] = pages

    def margins_of(page):
        m = page[1]
        return (m, m is not None and m.own)

    def columns_of(page):
        c = page[2]
        return (c, c is not None and c.own)

    for s in list(spreads) + list(masters):
        for index, i in enumerate(_left_to_right(s)):
            p = s.pages[i]
            layout = (p.master, p.margins, p.columns)
            p.margins = effective(raw, margins_of, layout, index, 0)
            p.columns = effective(raw, columns_of, layout, index, 0)


# ---------------------------------------------------------------------------
# Settings (src/model/settings.rs)
# ---------------------------------------------------------------------------

@dataclass
class NumberingList:
    uid: int
    name: str = ""
    builtin: bool = False
    across_stories: bool = False
    across_documents: bool = False


@dataclass
class ConstantShade:
    """A constant shade: chunk 0x3C1 (the name, as chunk 0x5531 reads
    it) and chunk 0x3C4 (u16 type and a f64)."""

    uid: int
    name: str | None = None
    builtin: bool = False
    type_code: int = 0
    amount: float = 0.0
    #: The three f64 of the shade, in order (the writer's ``Contents``
    #: is the count followed by these, all big-endian).
    values: list = field(default_factory=list)


@dataclass
class AxialShade:
    """An axial shade: chunk 0x3C2 (the name), chunk 0x3C4 (u16 type and
    a f64), chunk 0x3C3 (u16 count, then the stops), chunk 0x3C5 (the
    angle, a f64) and chunk 0x3C6 (the two f64 of the start point)."""

    uid: int
    name: str | None = None
    builtin: bool = False
    type_code: int = 0
    amount: float = 0.0
    stops: list = field(default_factory=list)
    angle: float = 0.0
    start: tuple = (0.0, 0.0)
    #: The six f64 at offset 4 of chunk 0x5532.
    matrix: list = field(default_factory=list)
    #: The shading data with every field big-endian, as IDML
    #: ``Contents`` holds it.
    data: bytes = b""


@dataclass
class DocumentPreferences:
    """The document preferences (class 0xB3A8, chunk 0xB3F0)."""

    #: Page width and height in points.
    page_size: tuple = (0.0, 0.0)
    #: 0 for a single page, 1 for facing pages.
    facing_pages: int = 0
    #: Page width and height of a page in points.
    page_width: float = 0.0
    page_height: float = 0.0
    columns: int = 1
    column_gutter: float = 0.0
    column_direction: int = 0
    bleed: tuple = (0.0, 0.0, 0.0, 0.0)
    slug: tuple = (0.0, 0.0, 0.0, 0.0)
    #: The four f64 of the page margins: top, bottom, left, right.
    margins: tuple = (0.0, 0.0, 0.0, 0.0)
    #: The intent code of the setup layout (0 print, 1 web, 2 mobile).
    intent: int | None = None
    #: The page binding code (0 left-to-right, 1 right-to-left).
    page_binding: int | None = None
    bleed_inside: float = 0.0
    bleed_outside: float = 0.0
    slug_inside: float = 0.0
    slug_outside: float = 0.0
    master_text_frame: bool = False
    master_text_frame_width: float = 0.0
    master_text_frame_height: float = 0.0
    document_slug_offset: float = 0.0
    slug_top: float = 0.0
    slug_left: float = 0.0
    slug_bottom: float = 0.0
    slug_right: float = 0.0
    document_bleed_top: float = 0.0
    document_bleed_left: float = 0.0
    document_bleed_bottom: float = 0.0
    document_bleed_right: float = 0.0
    preserve_layout: bool = False
    allow_page_shuffle: bool = False
    create_primary_text_frame: bool = False


@dataclass
class SetupLayout:
    """The setup layout (class 0xB3A8, chunk 0xB3F1)."""

    page_size: tuple = (0.0, 0.0)
    facing_pages: int = 0
    columns: int = 1
    column_gutter: float = 0.0
    column_direction: int = 0
    bleed: tuple = (0.0, 0.0, 0.0, 0.0)
    slug: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass
class Language:
    """A language (class 0x2D07): chunk 0x2D08 holds the name, chunk
    0x2D09 the language code and chunk 0x2D0A the hyphenation vendor
    code."""

    uid: int
    name: str = ""
    builtin: bool = False
    code: int = 0
    hyphenation: int = 0
    #: The primary and secondary names from chunk 0x2D0F.
    primary: str = ""
    sub: str = ""
    #: The u16 ID from chunk 0x2D0F.
    id: int = 0
    #: Two (flag, text) vendor pairs from chunk 0x2D0F: spelling,
    #: then hyphenation. ``None`` when the rest of the chunk cannot
    #: be read.
    vendors: list | None = None
    #: (single, double) quotes from chunk 0x2D26; ``None`` without
    #: the chunk.
    quotes: tuple | None = None


@dataclass
class Bullet:
    """A bullet (class 0x2A0B): chunk 0x2A0C holds the character, chunk
    0x2A0D the font UID, chunk 0x2A0E the font style as an in-object
    string and chunk 0x2A0F the built-in name."""

    uid: int
    character: str = ""
    font: int = 0
    font_style: Name = field(default_factory=lambda: Name(True, ""))
    name: str = ""
    builtin: bool = False
    #: 0 ``UnicodeOnly``, 1 ``UnicodeWithFont``, 2 ``GlyphWithFont``.
    kind: int = 0
    #: The character code (the value ``character`` is derived from).
    value: int = 0


# ---------------------------------------------------------------------------
# Stories (src/model/story.rs)
# ---------------------------------------------------------------------------

@dataclass
class TextRun:
    """A stretch of story text with one paragraph style and one
    character style."""

    #: UTF-16 offset of the run in the story text.
    start: int = 0
    text: str = ""
    paragraph_style: int | None = None
    character_style: int | None = None
    #: Local paragraph formatting.
    paragraph_attrs: Attrs = field(default_factory=Attrs)
    #: Local character formatting.
    character_attrs: Attrs = field(default_factory=Attrs)
    #: Index of the paragraph-style run record that holds the run. IDML
    #: starts a paragraph range at every record, even between records
    #: with the same formatting.
    paragraph_run: int = 0


@dataclass
class IndexMarker:
    """An index marker: a page reference (class 0x13006) owned by a
    U+FEFF."""

    uid: int = 0
    #: ``Id``: the u32 at 26 of a 30-byte chunk 0x13009.
    id: int | None = None
    #: The first 14 bytes of chunk 0x13009 are those of every sample,
    #: whose IDML has ``PageReferenceType="CurrentPage"``.
    current_page: bool = False


@dataclass
class StoryChunks:
    """Settings of a story stored in its own chunks. ``None`` where the
    story has no chunk."""

    #: ``StoryTitle`` as IDML writes it (chunk 0xA44C).
    title: str | None = None
    #: Chunk 0x50F96: u16.
    direction: int | None = None
    #: Chunk 0x2EE: ``OpticalMarginSize`` and the alignment u16.
    optical_margin: tuple | None = None
    #: Chunk 0x2EE: the text orientation u16 at 0 and the frame type
    #: u16 at 14.
    layout: tuple | None = None
    #: Chunk 0xCD02: the layout grid of a frame grid story.
    grid: GridData | None = None
    #: Chunk 0xCD32: the applied named grid.
    named_grid: object = None
    #: Link, style mappings and import options of a linked or placed
    #: story.
    linked: object = None


@dataclass
class Footnote:
    """A footnote: its text runs, at their offsets in the story text,
    which holds the footnote text after the story's own."""

    uid: int
    runs: list = field(default_factory=list)


@dataclass
class Note:
    """An InDesign note: its text runs, at their offsets in the story
    text, and chunk 0xA412."""

    uid: int
    runs: list = field(default_factory=list)
    user: str = ""
    #: Creation and modification times (FILETIME, UTC).
    created: int = 0
    modified: int = 0


@dataclass
class Deletion:
    """Deleted text of a tracked change (class 0xA40A): its text runs,
    at their offsets in the story text."""

    uid: int
    runs: list = field(default_factory=list)


@dataclass
class ChangeRun:
    """A run of the tracked-change strand (run kind 0xA466)."""

    #: UTF-16 offset and length.
    start: int = 0
    len: int = 0
    entries: list = field(default_factory=list)


@dataclass
class ChangeEntry:
    """One entry of a tracked-change run."""

    #: 1: a deletion at the start of the run; 2: the run was inserted.
    kind: int = 0
    user: str = ""
    #: FILETIME, UTC.
    time: int = 0

    def is_deletion(self) -> bool:
        return self.kind == 1

    def is_insertion(self) -> bool:
        return self.kind == 2


@dataclass
class EndnoteRange:
    """The text of an endnote in the endnote story."""

    #: UTF-16 offset in the story text.
    start: int = 0
    len: int = 0
    #: The range object (class 0x2804).
    uid: int = 0
    #: The endnote it belongs to (class 0x2805).
    endnote: int = 0


class Orientation:
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


@dataclass
class XmlMarker:
    """What an XML marker character (U+FEFF) in story text stands for:
    ("start"|"end"|"placeholder", key) or ("hidden", None)."""

    kind: str = "hidden"
    key: tuple | None = None


@dataclass
class XmlElement:
    """An XML element that can be written, with its IDML values."""

    #: IDML ``Self``.
    name: str = ""
    #: Tag name (IDML ``MarkupTag`` without ``XMLTag/``).
    tag: str = ""
    #: UID written as ``XMLContent``.
    content: int | None = None
    #: The content is a story.
    story_content: bool = False
    #: Attribute names and values, in stored order.
    attributes: list = field(default_factory=list)


@dataclass
class Story:
    uid: int
    runs: list = field(default_factory=list)
    #: Page items anchored in the text, by UTF-16 offset of their
    #: U+FFFC.
    anchors: dict = field(default_factory=dict)
    #: Tables, by UTF-16 offset of their U+0016.
    tables: dict = field(default_factory=dict)
    #: Text variable instances, by UTF-16 offset of their U+0018.
    text_variables: dict = field(default_factory=dict)
    #: Hyperlink text sources, sorted by start.
    sources: list = field(default_factory=list)
    #: XML markers, by UTF-16 offset of their U+FEFF.
    xml_markers: dict = field(default_factory=dict)
    #: Footnotes, by UTF-16 offset of their reference (U+0004).
    footnotes: dict = field(default_factory=dict)
    #: InDesign notes, by UTF-16 offset of their anchor (U+FEFF).
    notes: dict = field(default_factory=dict)
    #: Deleted texts of tracked changes, by UTF-16 offset of the
    #: character that follows the deletion.
    deletions: dict = field(default_factory=dict)
    #: Tracked-change runs (strand run kind 0xA466) that have entries,
    #: sorted by start.
    changes: list = field(default_factory=list)
    #: Index markers (``PageReference``), by UTF-16 offset of their
    #: U+FEFF.
    index_markers: dict = field(default_factory=dict)
    #: Endnote references, by UTF-16 offset of their U+0004: the
    #: endnote UID and the UID of its range.
    endnotes: dict = field(default_factory=dict)
    #: The story is the endnote story (class 0x2801).
    is_endnote: bool = False
    #: Endnote text ranges (in the endnote story), sorted by start;
    #: each lies within one text run. ``source`` is the range UID.
    endnote_ranges: list = field(default_factory=list)
    #: Text and paragraph destinations, by UTF-16 offset of their
    #: U+FEFF.
    text_destinations: dict = field(default_factory=dict)
    #: The XML element whose content is this story.
    xml_element: tuple | None = None
    #: Text orientation, from the frames that show the story; ``None``
    #: when it has no frame or its frames disagree.
    orientation: str | None = None
    #: The TOC style that made the story (chunk 0x8C40).
    toc_style: int | None = None
    #: Story settings in their own chunks.
    settings: StoryChunks = field(default_factory=StoryChunks)

    def text(self) -> str:
        return "".join(run.text for run in self.runs)


def char_offsets(text: list) -> list:
    """The UTF-16 offset of each character of ``text``, where a
    surrogate pair is one character, followed by the length of
    ``text``."""
    out = []
    i = 0
    while i < len(text):
        out.append(i)
        pair = (0xD800 <= text[i] < 0xDC00 and i + 1 < len(text)
                and 0xDC00 <= text[i + 1] < 0xE000)
        i += 2 if pair else 1
    out.append(len(text))
    return out


def split_runs(utf16: list, para, chars, extra) -> list:
    """Combine text with paragraph-style and character-style run lengths
    (counted in UTF-16 code units) into runs with both styles
    constant."""
    cuts = [0, len(utf16)]
    cuts.extend(min(c, len(utf16)) for c in extra)
    for runs in (para, chars):
        pos = 0
        for length, _style, _attrs in runs:
            pos += length
            cuts.append(min(pos, len(utf16)))
    cuts = sorted(set(cuts))

    def run_at(runs, at):
        pos = 0
        for k, (length, style, attrs) in enumerate(runs):
            if at < pos + length:
                return (uid_or_none(style), attrs, k)
            pos += length
        return None

    out = []
    for a, b in zip(cuts, cuts[1:], strict=False):
        if a >= b:
            continue
        para_run = run_at(para, a)
        char_run = run_at(chars, a)
        text = _utf16_lossy(utf16[a:b])
        out.append(TextRun(
            start=a, text=text,
            paragraph_style=(para_run[0] if para_run else None),
            character_style=(char_run[0] if char_run else None),
            paragraph_attrs=(para_run[1] if para_run else Attrs()),
            character_attrs=(char_run[1] if char_run else Attrs()),
            paragraph_run=(para_run[2] if para_run else 0)))
    return out


# ---------------------------------------------------------------------------
# Page items (src/model/item.rs)
# ---------------------------------------------------------------------------

class FormKind:
    BUTTON = "button"
    CHECK_BOX = "check_box"
    RADIO_BUTTON = "radio_button"
    TEXT_BOX = "text_box"
    COMBO_BOX = "combo_box"
    SIGNATURE_FIELD = "signature_field"
    MULTI_STATE = "multi_state"

    @classmethod
    def of(cls, class_id):
        return {0x1450C: cls.BUTTON, 0x1450D: cls.CHECK_BOX,
                0x1450E: cls.RADIO_BUTTON, 0x1450F: cls.TEXT_BOX,
                0x14511: cls.COMBO_BOX, 0x14512: cls.SIGNATURE_FIELD,
                0x145D8: cls.MULTI_STATE}.get(class_id)

    @classmethod
    def has_states(cls, kind) -> bool:
        return kind in (cls.BUTTON, cls.CHECK_BOX, cls.RADIO_BUTTON,
                        cls.MULTI_STATE)


@dataclass
class FormState:
    """One state of a form field."""

    #: The state ID, which gives the state type.
    id: int = 0
    active: bool = False
    enabled: bool = False
    items: list = field(default_factory=list)


@dataclass
class Form:
    """A form field or multi-state object. Its fill, stroke and other
    page item values are in the item's attribute list."""

    kind: str = FormKind.BUTTON
    states: list = field(default_factory=list)
    #: The bounds of the paths of the items it shows, in its own
    #: coordinates: left, top, right, bottom.
    bounds: list | None = None
    name: str | None = None
    description: str | None = None
    export_value: str | None = None
    required: bool | None = None
    multiline: bool | None = None
    scrollable: bool | None = None
    font_size: float | None = None
    #: Font family and style of a text box.
    font: tuple | None = None
    #: Timing lists of a button (chunk 0x2056E).
    timing: list = field(default_factory=list)
    #: Behaviours in the order of chunk 0x15601.
    behaviors: list = field(default_factory=list)


@dataclass
class EpsText:
    """The settings of EPS text."""

    #: Chunk 0x6611 as IDML writes it (``EPSTextData``): big-endian,
    #: with two fields left out.
    data: bytes = b""
    #: Chunk 0x154: left, top, right, bottom (``PathBoundingBox``).
    path_bounds: list = field(default_factory=lambda: [0.0] * 4)
    #: The first four f64 of chunk 0x6612, in the same order
    #: (``EPSTextAttributeBounds``).
    attr_bounds: list = field(default_factory=lambda: [0.0] * 4)


@dataclass
class TextPath:
    """Text on the path of a shape or frame (class 0xB320)."""

    uid: int
    story: int | None = None
    previous: int | None = None
    next: int | None = None
    #: ``FlipPathEffect``.
    flipped: bool = False
    start: float = 0.0
    end: float = 0.0
    #: The three codes at 8 of chunk 0xB30A are those of every sample
    #: (``CenterPathAlignment``, ``BaselineTextAlignment``,
    #: ``RainbowPathEffect``, ``PathSpacing`` 0).
    default_codes: bool = True


@dataclass
class ExportOptions:
    """Alternative text and tagging settings of a page item (chunk
    0x1E206)."""

    #: Source codes: 0 custom, 5 XML structure, 6 XMP (actual text
    #: only), 8 decorative image (alternative text only).
    alt_source: int = 0
    alt_text: Name = field(default_factory=lambda: Name(True, ""))
    #: The namespace prefix and property path of the metadata property.
    alt_metadata: list = field(
        default_factory=lambda: [Name(True, ""), Name(True, "")])
    actual_source: int = 0
    actual_text: Name = field(default_factory=lambda: Name(True, ""))
    actual_metadata: list = field(
        default_factory=lambda: [Name(True, ""), Name(True, "")])
    #: 0 tag from structure, 1 artifact.
    tag_type: int = 0


@dataclass
class GeneratedAltText:
    """Generated alternative text of a page item (chunks 0x1E260 to
    0x1E262)."""

    #: Chunk 0x1E260 is 1.
    error: bool = False
    #: Chunk 0x1E262 is 1.
    ai: bool = False
    #: The four f64 of chunk 0x1E261, unless all are 0.
    crop: list | None = None


@dataclass
class ItemName:
    """A page item name: a built-in key or a name given by the user."""

    builtin: bool = False
    name: str = ""


@dataclass
class ItemProps:
    """Settings every page item has."""

    #: The name (chunk 0x2C10, of a group 0x418); ``None`` without the
    #: chunk.
    name: ItemName | None = None
    #: Chunk 0x2C32 is 0.
    hidden: bool = False
    #: Chunk 0x2C2D is 1.
    locked: bool = False
    #: ``AllowOverrides``, for items of master spreads only: ``false``
    #: when chunk 0x142D is 1.
    allow_overrides: bool | None = None
    #: Layout constraint flags (chunk 0x22228).
    layout_constraints: int | None = None
    #: The parent, target and last updated interface change counts
    #: (chunks 0x21D4E, 0x21D50, 0x21D53), each a list of numbers.
    change_counts: list = field(default_factory=lambda: [[], [], []])
    #: The overridden master page item (0 for none) and the IDs of the
    #: attributes overridden (chunk 0x1424).
    overridden: tuple | None = None
    #: Chunk 0x1623 is 1: a frame for a graphic (``ContentType``
    #: ``GraphicType`` without a graphic).
    graphic_frame: bool = False
    #: The i32 at 4 of chunk 0x324: the layer position before grouping.
    grouping_layer_position: int | None = None
    #: The layer named by the u32 at 0 of chunk 0x324, a spread layer
    #: resolved to its layer.
    grouping_layer: int | None = None
    generated_alt: GeneratedAltText = field(default_factory=GeneratedAltText)
    #: The value of key ``CustomSizeType`` of chunk 0x1E22B.
    export_size_type: int | None = None


@dataclass
class TextWrap:
    """Text wrap settings of a page item or graphic (chunk 0x3703)."""

    #: Wrap mode code (see :func:`wrap_mode`).
    mode: int = 0
    #: The four offsets: left, top, right, bottom.
    offsets: list = field(default_factory=lambda: [0.0] * 4)
    #: The u16 at offset 40: 0 for ``ApplyToMasterPageOnly``, 1
    #: otherwise.
    master_flag: int = 1
    #: The u16 at offset 42: the ``TextWrapSide`` code.
    side: int = 0
    #: The u32 at offset 4: the wrap path object (class 0x3702), 0 for
    #: none.
    path_uid: int = 0
    #: The wrap path when its object is flagged custom; empty otherwise
    #: and for object styles.
    custom_path: list = field(default_factory=list)


def _key_values(c: Cursor):
    """A list of keys with string values (chunk 0x1E22B): u32 count; per
    entry a u32-counted key in text segments, a flag byte, two bytes, a
    u16 value count and the values, each a u8 length, a u8 and that many
    bytes."""
    n = c.u32()
    out = []
    for _ in range(n):
        length = c.u32()
        key = c.segments(length)
        c.skip(3)
        count = c.u16()
        values = []
        for _ in range(count):
            m = c.u8()
            c.u8()
            values.append(c.bytes(m).decode("utf-8", errors="replace"))
        out.append((key, values))
    return out


class GraphicKind:
    IMAGE = "image"
    PDF = "pdf"
    EPS = "eps"
    SVG = "svg"
    #: A placed InDesign page (class 0x6607).
    IMPORTED_PAGE = "imported_page"


@dataclass
class Link:
    """A link (class 0x8C42, 0x21D03 or 0x21D32) and its link resource
    (class 0x8C41)."""

    uid: int = 0
    #: The class of the link object (``LinkClassID``); 0 for unknown.
    class_id: int = 0
    #: ``LinkResourceURI``, for example ``file:/Users/me/image.jpg``.
    uri: str = ""
    #: The linked file is stored in the document
    #: (``StoredState="Embedded"``).
    embedded: bool = False
    #: Chunk 0x8C92, u32 at URI+8 is 3 (``StoredState="Contained"``).
    contained: bool = False
    #: Chunk 0x8C9B, u32 at 12: 0 when the linked file was modified.
    modified: bool | None = None
    #: Chunk 0x8C9B, u16 at 20: 1 when the link is shown.
    shown: bool | None = None
    #: The import stamp (``file <time> <size>``); ``None`` when empty.
    stamp: str | None = None
    #: The file's modification time and the time it was placed or
    #: updated, as FILETIME (100 ns since 1601-01-01 UTC).
    times: list | None = None
    #: Chunk 0x1B6 of the link.
    pdf_identifier: int | None = None
    #: The file size of the resource: high and low u32.
    size: list | None = None
    #: The format name of the resource (``JPEG``, ``Photoshop``, ...).
    format: str | None = None


@dataclass
class Contour:
    """The contour of a text wrap: chunk 0x373D of placed graphics,
    0x373F of the preferences and 0x3777 of object styles (29 bytes
    each)."""

    #: Contour type code (u32 at 0).
    kind: int = 5
    #: ``IncludeInsideEdges`` (u8 at 17); ``None`` for a byte other than
    #: 0 or 1.
    inside_edges: bool | None = False

    DEFAULT: ClassVar[Contour] = None

    @classmethod
    def read(cls, enc: Encoding, data: bytes):
        if len(data) < 18:
            return None
        kind = enc.u32_at(data, 0)
        if kind is None:
            return None
        inside = data[17]
        return cls(kind=kind,
                   inside_edges={0: False, 1: True}.get(inside))


Contour.DEFAULT = Contour(kind=5, inside_edges=False)


@dataclass
class ImageProperties:
    """Image properties (chunk 0x1708): u32 count, then records of u32
    key, u32 length *n*, u8, *n* bytes."""

    #: Key 0x67: 1 grey, 2 RGB, 4 CMYK.
    color_space: int | None = None
    #: Key 0x6F (a colour table) is present.
    indexed: bool = False
    #: Keys 0x6C and 0x6D, 16.16 fixed point.
    resolution: list | None = None
    #: Key 0x7D: the name of the image's clipping path, when its bytes
    #: are printable ASCII.
    clipping_path_name: str | None = None


@dataclass
class GraphicLayer:
    name: Name = field(default_factory=lambda: Name(True, ""))
    id: int = 0
    original_visibility: bool = False
    current_visibility: bool = False
    #: The parent layer's ID, -1 for a top-level layer.
    parent: int = -1
    #: 0x01 separator, 0x04 effects layer, 0x08 locked.
    flags: int = 0


@dataclass
class GraphicLayers:
    """Layers of a placed graphic (chunk 0x177A)."""

    #: The u16 at 0 is 1.
    option: bool = False
    #: In IDML order.
    layers: list = field(default_factory=list)


@dataclass
class ClippingPath:
    """Clipping path settings of a graphic (chunk 0x2C1A, 37 bytes):
    u32 type at 0, f64 tolerance at 4, f64 inset at 12, u8 threshold at
    20, i16 index at 23, u8 flags at 25."""

    kind: int = 0
    tolerance: float = 0.0
    inset: float = 0.0
    threshold: int = 0
    index: int = 0
    flags: int = 0


@dataclass
class TextFramePreferences:
    """Text frame settings, from the frame's multi-column frame
    object."""

    column_count: int = 1
    column_gutter: float = 0.0
    column_fixed_width: float = 0.0
    #: First baseline offset code (chunk 0x2CE, u16 at 0): 0
    #: LeadingOffset, 1 AscentOffset, 2 CapHeight, 3 EmboxHeight.
    first_baseline_offset: int = 0
    vertical_justification: int = 0
    vertical_balance_columns: bool = False
    #: Chunk 0x2CE u16 at 22 and 24; ``None`` for a chunk of 22 to 27
    #: bytes, which has no auto-sizing fields.
    auto_sizing_type: int | None = None
    auto_sizing_reference_point: int | None = None
    #: Chunk 0x2D1: u8 at 12 and, in 40-byte chunks, f64 at 32.
    use_fixed_width: bool = False
    max_width: float | None = None
    #: Chunk 0x2CE (48 bytes): u16 at 26 and f64 at 28 (minimum
    #: height), u16 at 36 and f64 at 38 (minimum width), u16 at 46 (no
    #: line breaks).
    minimum_sizes: tuple | None = None
    #: Chunk 0x3730: u16 ignore text wrap.
    ignore_wrap: bool | None = None
    #: Column rule (chunk 0x22646): f64 width at 28, u32 colour at 36.
    column_rule: tuple | None = None
    #: Column rule override: u32 at 0 of chunk 0x22646, 1 for true.
    column_rule_override: bool | None = None
    #: Footnote options (chunk 0x22608): u16, u16 span across columns,
    #: f64 minimum spacing, f64 space between.
    footnotes: list | None = None
    #: Chunk 0x22608, u16 at 2: footnotes span across columns.
    footnote_span: bool = False
    #: Baseline frame grid (chunk 0x2834); ``None`` without the chunk.
    baseline_grid: BaselineGrid | None = None
    #: The interface colour that the grid's colour UID names.
    baseline_rgb: list | None = None
    #: Inset spacing (the frame's chunk 0x3723): top, left, bottom,
    #: right.
    inset: list = field(default_factory=lambda: [0.0] * 4)
    #: The f64 at 0 of the frame's chunk 0x3723: the inset spacing of a
    #: frame that is not a plain rectangle.
    inset_single: float = 0.0
    #: A frame grid (chunk 0xCD41 starts with u16 1).
    frame_grid: bool = False


@dataclass
class Graphic:
    uid: int
    kind: str = GraphicKind.IMAGE
    transform: Matrix = field(default_factory=Matrix)
    #: Left, top, right, bottom in the graphic's own coordinates.
    bounds: list = field(default_factory=lambda: [0.0] * 4)
    link: Link | None = None
    #: The graphic's file, when the document holds it: an embedded link
    #: or a graphic pasted without a link. IDML writes it as
    #: ``Contents``.
    contents: bytes | None = None
    #: The raw data object of InDesign's screen preview of a graphic
    #: whose file the document does not hold (chunk 0x170D, then chunk
    #: 0x119 of the preview object). IDML has no place for it.
    preview: int | None = None
    text_wrap: TextWrap | None = None
    #: The contour of the text wrap (chunk 0x373D).
    contour: Contour | None = None
    #: Clipping path settings (chunk 0x2C1A), if stored.
    clipping: ClippingPath | None = None
    #: The clipping path (the graphic's chunk 0x162B), read only for a
    #: clipping type other than 0.
    clipping_paths: list = field(default_factory=list)
    #: Name, visibility, change counts and the other page item settings.
    props: ItemProps = field(default_factory=ItemProps)
    #: Chunk 0x1B916.
    object_style: int | None = None
    #: The attribute list (chunk 0x6E03).
    attrs: Attrs = field(default_factory=Attrs)
    #: Image properties (chunk 0x1708 of an image).
    image: ImageProperties | None = None
    #: Colour profile code (chunk 0x7C0F, u32 at 0).
    profile: int | None = None
    #: Grey, RGB, grey again and CMYK vector policy codes of a PDF or
    #: EPS (chunk 0x7C42, four u32); only the RGB and CMYK codes are
    #: known.
    vector_policies: list | None = None
    #: PDF placement (chunk 0x251B): page number, transparent
    #: background byte, crop code.
    pdf: tuple | None = None
    #: Layers of an image, PDF or imported page (chunk 0x177A).
    layers: GraphicLayers | None = None
    #: Applied layer comp (chunk 0x9209, i32 at 4).
    layer_comp: int | None = None
    #: Image import options (chunk 0x1714): apply the Photoshop
    #: clipping path, alpha channel name.
    import_options: tuple | None = None
    #: The name of the image's first alpha channel (chunk 0x177C),
    #: which an ``AlphaChannel`` contour uses.
    alpha_channel: str | None = None
    #: The index of a placed InDesign page (chunk 0x2505, u32 at 0).
    page_index: int | None = None


@dataclass
class PageItem:
    uid: int
    #: Placed graphics (images, PDF, EPS, SVG) inside a frame.
    graphics: list = field(default_factory=list)
    #: Placed movies and sounds inside a frame. They are not graphics:
    #: the frame's content type and contour do not depend on them.
    media: list = field(default_factory=list)
    #: ("text_frame", dict) | ("shape", shape) | ("group",) |
    #: ("eps_text", EpsText) | ("form", Form)
    kind: tuple = ("group",)
    transform: Matrix = field(default_factory=Matrix)
    paths: list = field(default_factory=list)
    #: Document layer; ``None`` for items anchored in text.
    layer: int | None = None
    #: Local formatting (fill, stroke, corners, ...).
    attrs: Attrs = field(default_factory=Attrs)
    object_style: int | None = None
    children: list = field(default_factory=list)
    #: Text wrap (chunk 0x3703); ``None`` if the item has none.
    text_wrap: TextWrap | None = None
    #: Anchored object settings (chunk 0x2800 of the anchor), for an
    #: item anchored in text.
    anchor: AnchorSettings | None = None
    props: ItemProps = field(default_factory=ItemProps)
    #: Export options (chunk 0x1E206); ``None`` without the chunk.
    export: ExportOptions | None = None
    #: Text on the item's path (chunk 0xB30A).
    text_paths: list = field(default_factory=list)
    #: Script label: key and value pairs, in stored order.
    label: list = field(default_factory=list)
    #: Animation (chunk 0x2055F).
    animation: object = None
    #: A group's article children list (chunk 0x21B05), in stored
    #: order; ``None`` without the chunk.
    article_children: list | None = None

    def kind_name(self) -> str:
        """The kind as a plain string for tests and the tool surface:
        ``text_frame`` for a text frame, ``graphic`` for a frame holding
        a graphic, ``spline``/``oval``/``rectangle``/``graphic_line`` for
        a shape, ``group`` for a group, ``eps_text`` / ``form``."""
        kind = self.kind[0]
        if self.graphics:
            return "graphic"
        if kind == "text_frame":
            return "text_frame"
        if kind == "shape":
            shape = self.kind[1]
            if shape == Shape.RECTANGLE:
                return "rectangle"
            if shape == Shape.OVAL:
                return "oval"
            if shape == Shape.GRAPHIC_LINE:
                return "graphic_line"
            return "spline"
        if kind == "group":
            return "group"
        if kind == "eps_text":
            return "eps_text"
        if kind == "form":
            return "form"
        return kind


# ---------------------------------------------------------------------------
# IDML number format (src/idml/format.rs::num)
# ---------------------------------------------------------------------------

def _shortest_digits(v: float):
    """The digits of the shortest decimal that reads back as ``v``,
    and the exponent of its leading digit. Python's ``repr`` is the
    shortest decimal; ``Decimal`` keeps its significant digits. (Do
    NOT use ``format(v, "e")`` here: without a precision it writes
    six digits after the point, not the shortest form — Rust's
    ``{v:e}`` is shortest, and the first version of this function
    silently rounded every number to six digits.)"""
    dec = Decimal(repr(v))
    digits = "".join(str(d) for d in dec.as_tuple().digits).rstrip("0")
    return (digits or "0", dec.adjusted())


def idml_num(v: float) -> str:
    """A number as IDML text: the shortest decimal that reads back as the
    same value. Numbers below 1e-4 are written with an exponent of at
    least two digits; all others in plain notation. NaN or infinity are
    written as 0; negative zero is kept."""
    if not math.isfinite(v):
        return "0"
    if v == 0.0:
        return "-0" if math.copysign(1.0, v) < 0 else "0"
    # Shortest digits that read back as v, then the decimal of that
    # length nearest to the exact value (Python's format rounds ties
    # to even, as Rust's does), if it also reads back.
    digits, exp_short = _shortest_digits(v)
    length = len(digits)
    nearest = format(v, f".{max(length - 1, 0)}e")
    if float(nearest) == v:
        sci = nearest
    else:
        sci = (f"{'-' if v < 0 else ''}{digits[0]}"
               f"{'.' + digits[1:] if len(digits) > 1 else ''}"
               f"e{exp_short:+d}")
    mantissa, exp = sci.split("e")
    exp = int(exp)
    sign = ""
    if mantissa.startswith("-"):
        sign = "-"
        mantissa = mantissa[1:]
    digits = "".join(ch for ch in mantissa if ch.isdigit()).rstrip("0")
    if not digits:
        digits = "0"
    if exp < -4:
        first, rest = digits[0], digits[1:]
        point = "." if rest else ""
        text = f"{first}{point}{rest}e-{(-exp):02d}"
    elif exp < 0:
        text = "0." + "0" * (-exp - 1) + digits
    else:
        point = exp + 1
        if point >= len(digits):
            text = digits + "0" * (point - len(digits))
        else:
            text = digits[:point] + "." + digits[point:]
    return sign + text


def num_half_even(v: float) -> str:
    """A number as IDML writes the paper rectangles of the print
    settings: the digits of the shortest form that reads back, the last
    one rounded half to even from the exact value."""
    if v == 0.0 or not math.isfinite(v):
        return idml_num(v)
    shortest = _shortest_digits(v)[0]
    n = max(len(shortest), 1)
    e = format(v, f".{n - 1}e")
    mant, exp = e.split("e") if "e" in e else (e, "0")
    exp = int(exp)
    neg = mant.startswith("-")
    digits = "".join(ch for ch in mant if ch.isdigit())
    point = exp + 1
    if point <= 0:
        out = "0." + "0" * (-point) + digits
    elif point >= len(digits):
        out = digits + "0" * (point - len(digits))
    else:
        out = digits[:point] + "." + digits[point:]
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    if neg:
        out = "-" + out
    return out


# ---------------------------------------------------------------------------
# Hyperlinks (src/model/hyperlink.rs)
# ---------------------------------------------------------------------------

class HyperlinkClass:
    HYPERLINK = 0x13501
    TEXT_SOURCE = 0x13502
    PAGE_ITEM_SOURCE = 0x13503
    TEXT_DESTINATION = 0x13504
    PAGE_DESTINATION = 0x13505
    URL_DESTINATION = 0x13506
    EXTERNAL_PAGE_DESTINATION = 0x13552
    BOOKMARK = 0x1354C
    DESTINATION_OWNER = 0x1353C
    SOURCE_START = 0x13508
    SOURCE_END = 0x1353B
    RANGE_STRAND = 0xCA1C
    RANGE_NODE = 0xCA1E


class HyperlinkChunk:
    DOCUMENT_LISTS = 0x13501
    HYPERLINK = 0x13502
    HYPERLINK_APPEARANCE = 0x13553
    OTHER_DOCUMENT = 0x1359F
    TEXT_SOURCE = 0x13504
    TEXT_SOURCE_RANGE = 0x1352E
    TEXT_SOURCE_ALTERNATIVE = 0x135B7
    CROSS_REFERENCE = 0x135A0
    PAGE_ITEM_SOURCE = 0x13505
    PAGE_ITEM_SOURCE_ITEM = 0x13525
    PAGE_DESTINATION = 0x13507
    PAGE_DESTINATION_VIEW = 0x13527
    URL_DESTINATION = 0x13509
    EXTERNAL_PAGE_DESTINATION = 0x13580
    EXTERNAL_PAGE_INDEX = 0x1B8
    TEXT_DESTINATION = 0x13508
    TEXT_DESTINATION_OWNER = 0x13526
    OWNER_DESTINATION = 0x1352B
    OWNER_DESTINATION_5 = 0x1351E
    OWNER_DESTINATION_5 = 0x1351E
    SOURCE_MARKERS = 0x13524
    URL = 0x100B
    BOOKMARK = 0x13547
    RANGE_TREE = 0x16126
    RANGE_PAGE = 0x16127


class HyperlinkKind:
    #: No destination in this document.
    NONE = 2000
    #: A page in another document.
    EXTERNAL_PAGE = 2004


_APPEARANCE = bytes([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def _appearance(data):
    """Whether chunk 0x13553 has the corpus pattern; its ``Highlight``
    byte if so."""
    if data is None or len(data) != len(_APPEARANCE):
        return None
    if all(i in (6, 10) or b == a
           for i, (b, a) in enumerate(zip(data, _APPEARANCE,
                                     strict=False))):
        return data[10]
    return None


@dataclass
class Hyperlink:
    uid: int
    name: str = ""
    source: int = 0
    hidden: bool = False
    #: ``DestinationUniqueKey``, shared with the destination; ``None``
    #: in InDesign 5.0 files.
    key: int | None = None
    #: The destination UID (InDesign 5.0 files, which have no keys).
    destination: int | None = None
    #: Destination kind (u32 at offset 12 of chunk 0x13502).
    kind: int = 0
    #: Chunk 0x13553 has the pattern of every corpus hyperlink, whose
    #: ``Visible``, ``Width``, ``BorderStyle`` and ``BorderColor`` are
    #: one value.
    appearance_known: bool = False
    #: Byte 10 of chunk 0x13553 (``Highlight``), when the pattern holds.
    highlight: int | None = None
    #: Chunk 0x1359F: the destination is in another document.
    other_document: bool = False

    def kind_is_none(self) -> bool:
        return self.kind == HyperlinkKind.NONE

    def to_external_page(self) -> bool:
        return self.kind == HyperlinkKind.EXTERNAL_PAGE

    @classmethod
    def read(cls, uid: int, obj, major: int):
        data = obj.chunk(HyperlinkChunk.HYPERLINK)
        if data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        source = c.u32()
        c.skip(2)
        hidden = c.u16() != 0
        if major <= 5:
            dest = c.u32()
            c.skip(4)
            key, destination = None, dest if dest != 0 else None
            kind = c.u32()
        else:
            key = c.u32()
            kind = c.u32()
            c.skip(4)
            destination = None
        c.skip(4)
        name = c.name().name
        highlight = _appearance(obj.chunk(HyperlinkChunk.HYPERLINK_APPEARANCE))
        return cls(uid=uid, name=name, source=source, hidden=hidden,
                   key=key, destination=destination, kind=kind,
                   appearance_known=highlight is not None,
                   highlight=highlight,
                   other_document=obj.chunk(HyperlinkChunk.OTHER_DOCUMENT)
                   is not None)


@dataclass
class Alternative:
    """An alternative destination of a text source (chunk 0x135B7)."""

    #: Type code; 2 is a table of contents anchor, the only one seen.
    kind: int = 0
    anchor_name: str = ""
    index_marker: int = 0
    page_number: str = ""
    level: int = 0

    TOC_ANCHOR = 2

    def is_toc_anchor(self) -> bool:
        return self.kind == self.TOC_ANCHOR

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> Alternative:
        c = enc.cursor(d)
        kind = c.u32()
        anchor_name = c.name().name
        index_marker = c.u32()
        n = c.u32()
        page_number = "" if n == 0 else c.segments(n)
        return cls(kind=kind, anchor_name=anchor_name,
                   index_marker=index_marker, page_number=page_number,
                   level=c.u32())


@dataclass
class TextSource:
    uid: int
    name: str = ""
    hidden: bool = False
    character_style: int | None = None
    #: Chunk 0x135B7: ``AlternativeDestination``.
    alternative: Alternative | None = None
    #: Chunk 0x135A0: the source is a ``CrossReferenceSource`` with
    #: this cross-reference format.
    format: int | None = None

    @classmethod
    def read(cls, uid: int, obj):
        data = obj.chunk(HyperlinkChunk.TEXT_SOURCE)
        if data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        hidden = c.u8() != 0
        c.skip(5)
        name = c.name().name
        character_style = None
        range_data = obj.chunk(HyperlinkChunk.TEXT_SOURCE_RANGE)
        if range_data is not None:
            character_style = uid_or_none(enc.u32_at(range_data, 0))
        alternative = None
        alt = obj.chunk(HyperlinkChunk.TEXT_SOURCE_ALTERNATIVE)
        if alt is not None:
            alternative = Alternative.read(enc, alt)
        fmt = None
        xref = obj.chunk(HyperlinkChunk.CROSS_REFERENCE)
        if xref is not None:
            fmt = enc.u32_at(xref, 0)
        return cls(uid=uid, name=name, hidden=hidden,
                   character_style=character_style,
                   alternative=alternative, format=fmt)


@dataclass
class PageItemSource:
    """A page item that is a hyperlink source (class 0x13503)."""

    uid: int
    name: str = ""
    hidden: bool = False
    item: int = 0

    @classmethod
    def read(cls, uid: int, obj):
        data = obj.chunk(HyperlinkChunk.PAGE_ITEM_SOURCE)
        item_data = obj.chunk(HyperlinkChunk.PAGE_ITEM_SOURCE_ITEM)
        if data is None or item_data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        hidden = c.u8() != 0
        c.skip(5)
        name = c.name().name
        return cls(uid=uid, name=name, hidden=hidden,
                   item=enc.cursor(item_data).u32())


class DestinationKind:
    PAGE = "page"
    URL = "url"
    EXTERNAL_PAGE = "external_page"
    TEXT = "text"
    PARAGRAPH = "paragraph"


@dataclass
class Destination:
    uid: int
    name: str = ""
    hidden: bool = False
    #: ``DestinationUniqueKey``; ``None`` in InDesign 5.0 files, whose
    #: destination chunks end after the name.
    key: int | None = None
    #: ("page", page, zoom, view, bounds) | ("url", url) |
    #: ("external_page", page_index, zoom, view, bounds, link, file_name)
    #: | ("text",) | ("paragraph",)
    kind: tuple = ("text",)

    def reference(self) -> str:
        """The IDML ``Self`` (and reference) of the destination."""
        escaped = (self.name.replace("%", "%25")
                   .replace(":", "%3a").replace("\r", "%0d"))
        kind = self.kind[0]
        if kind == DestinationKind.PAGE:
            return f"HyperlinkPageDestination/{escaped}"
        if kind == DestinationKind.URL:
            return f"HyperlinkURLDestination/{escaped}"
        if kind == DestinationKind.EXTERNAL_PAGE:
            return f"u{self.uid:x}"
        if kind == DestinationKind.PARAGRAPH:
            return f"ParagraphDestination/{escaped}"
        return f"HyperlinkTextDestination/{escaped}"

    @classmethod
    def read(cls, uid: int, class_id: int, obj):
        enc = obj.encoding
        if class_id == HyperlinkClass.PAGE_DESTINATION:
            data = obj.chunk(HyperlinkChunk.PAGE_DESTINATION)
            if data is None:
                return None
            c = enc.cursor(data)
            hidden = c.u8() != 0
            c.skip(5)
            name = c.name().name
            key = None
            if c.remaining() >= 4:
                key = c.u32()
            page = None
            zoom = None
            view = 0
            bounds = None
            view_data = obj.chunk(HyperlinkChunk.PAGE_DESTINATION_VIEW)
            if view_data is not None:
                v = enc.cursor(view_data)
                page = v.u32()
                raw_zoom = v.f64()
                zoom = raw_zoom if 0.05 <= raw_zoom <= 40.0 else None
                view = v.u32()
                bounds = [v.f64() for _ in range(4)]
            return cls(uid=uid, name=name, hidden=hidden, key=key,
                       kind=(DestinationKind.PAGE, page, zoom, view,
                             bounds))
        if class_id == HyperlinkClass.URL_DESTINATION:
            data = obj.chunk(HyperlinkChunk.URL_DESTINATION)
            if data is None:
                return None
            c = enc.cursor(data)
            hidden = c.u8() != 0
            c.skip(5)
            name = c.name().name
            key = c.u32() if c.remaining() >= 4 else None
            url = ""
            url_data = obj.chunk(HyperlinkChunk.URL)
            if url_data is not None:
                url = enc.cursor(url_data).string()
            return cls(uid=uid, name=name, hidden=hidden, key=key,
                       kind=(DestinationKind.URL, url))
        if class_id == HyperlinkClass.EXTERNAL_PAGE_DESTINATION:
            data = obj.chunk(HyperlinkChunk.EXTERNAL_PAGE_DESTINATION)
            if data is None:
                return None
            c = enc.cursor(data)
            hidden = c.u8() != 0
            c.skip(5)
            name = c.name().name
            key = c.u32() if c.remaining() >= 4 else None
            page_index = None
            index_data = obj.chunk(HyperlinkChunk.EXTERNAL_PAGE_INDEX)
            if index_data is not None:
                page_index = enc.u32_at(index_data, 0)
            zoom = None
            view = 0
            bounds = None
            view_data = obj.chunk(HyperlinkChunk.PAGE_DESTINATION_VIEW)
            if view_data is not None:
                v = enc.cursor(view_data)
                raw_zoom = v.f64()
                zoom = raw_zoom if 0.05 <= raw_zoom <= 40.0 else None
                view = v.u32()
                bounds = [v.f64() for _ in range(4)]
            return cls(uid=uid, name=name, hidden=hidden, key=key,
                       kind=(DestinationKind.EXTERNAL_PAGE, page_index,
                             zoom, view, bounds, None, None))
        if class_id == HyperlinkClass.TEXT_DESTINATION:
            data = obj.chunk(HyperlinkChunk.TEXT_DESTINATION)
            if data is None:
                return None
            c = enc.cursor(data)
            hidden = c.u8() != 0
            c.skip(5)
            name = c.name().name
            key = c.u32() if c.remaining() >= 4 else None
            return cls(uid=uid, name=name, hidden=hidden, key=key,
                       kind=(DestinationKind.TEXT,))
        return None


@dataclass
class Bookmark:
    uid: int
    name: str = ""
    #: The parent bookmark, or the document (UID 1).
    parent: int = 1
    children: list = field(default_factory=list)
    destination: int = 0

    @classmethod
    def read(cls, uid: int, obj):
        data = obj.chunk(HyperlinkChunk.BOOKMARK)
        if data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        c.skip(4)
        parent = c.u32()
        name = c.name().name
        destination = c.u32()
        return cls(uid=uid, name=name, parent=parent,
                   destination=destination)


def document_bookmarks(enc: Encoding, data: bytes, major: int):
    """The bookmark UIDs of a document (chunk 0x13501): u32 count, then
    the UIDs."""
    c = enc.cursor(data)
    n = c.u32()
    return [c.u32() for _ in range(n)]


@dataclass
class SourceRange:
    """A hyperlink text source's range in a story: the source UID, the
    story UID, start and length in UTF-16 units, and the cross-reference
    format UID (0 for a plain hyperlink source)."""

    source: int = 0
    story: int = 0
    start: int = 0
    length: int = 0
    format: int = 0
    #: Written as a child of the paragraph range, around character
    #: ranges, rather than inside one character range.
    paragraph: bool = False


# ---------------------------------------------------------------------------
# Text variables (src/model/variable.rs)
# ---------------------------------------------------------------------------

VARIABLE_TYPES = [
    (0xCAA1, "OutputDateType"),
    (0xCAA3, "CustomTextType"),
    (0xCAA6, "FileNameType"),
    (0xCAA8, "LastPageNumberType"),
    (0xCAA9, "ChapterNumberType"),
    (0xCAAA, "MatchParagraphStyleType"),
    (0xCAAB, "ModificationDateType"),
    (0xCAAC, "CreationDateType"),
    (0xCAB5, "XrefPageNumberType"),
    (0xCAB6, "XrefChapterNumberType"),
    (0xCAC0, "LiveCaptionType"),
]


@dataclass
class VariableSettings:
    """The settings of a definition: three u32, then the text before
    and after the value."""

    a: int = 0
    b: int = 0
    c: int = 0
    before: str = ""
    after: str = ""


@dataclass
class TextVariable:
    uid: int
    name: str = ""
    #: Type code (see :data:`VARIABLE_TYPES`).
    kind: int = 0
    #: The date format of date variables; the contents of custom text
    #: variables (not proven, see the format notes).
    text: str = ""
    #: Paragraph style of a running header
    #: (``MatchParagraphStyleType``).
    style: int | None = None
    #: The fields not identified hold the values every corpus sample of
    #: this type has, so the IDML settings are those of the samples.
    as_in_samples: bool = False
    #: The settings after the type code, where they parse to the end of
    #: the chunk.
    settings: VariableSettings | None = None

    def type_name(self):
        for code, name in VARIABLE_TYPES:
            if code == self.kind:
                return name
        return None

    @classmethod
    def read(cls, uid: int, obj):
        data = obj.chunk(0x28BD)
        if data is None:
            return None
        return _parse_text_variable(obj.encoding, uid, data)


def _counted(c: Cursor) -> str:
    n = c.u32()
    return "" if n == 0 else c.segments(n)


def _parse_text_variable(enc: Encoding, uid: int, data: bytes):
    c = enc.cursor(data)
    name = _counted(c)
    text = _counted(c)
    kind = c.u32()
    settings = None
    try:
        s = enc.cursor(data[c.pos:])
        a, b, c3 = s.u32(), s.u32(), s.u32()
        before = _counted(s)
        after = _counted(s)
        if s.remaining() == 0:
            settings = VariableSettings(a=a, b=b, c=c3, before=before,
                                        after=after)
    except InddError:
        settings = None
    rest = data[c.pos:]
    zero = all(b == 0 for b in rest)
    style = None
    if len(rest) == 20:
        if kind in (0xCAA1, 0xCAAB, 0xCAAC):
            as_in_samples = ((all(b == 0 for b in rest[:4])
                              or enc.u32_at(rest, 0) == 0xCAB3)
                             and all(b == 0 for b in rest[4:]))
        elif kind == 0xCAC0:
            as_in_samples = (enc.u32_at(rest, 0) == 0x8C64
                             and all(b == 0 for b in rest[4:]))
        elif kind == 0xCAAA:
            style = enc.u32_at(rest, 4)
            as_in_samples = (all(b == 0 for b in rest[:4])
                             and all(b == 0 for b in rest[8:]))
        else:
            as_in_samples = zero and text == ""
    else:
        as_in_samples = False
    return TextVariable(uid=uid, name=name, kind=kind, text=text,
                        style=(style if style not in (None, 0) else None),
                        as_in_samples=as_in_samples, settings=settings)


@dataclass
class VariableInstance:
    """A text variable placed in text (U+0018 owning a class 0xCA64
    object)."""

    uid: int
    #: Name of the variable it shows.
    name: str = ""

    @classmethod
    def read(cls, uid: int, obj):
        enc = obj.encoding
        name = ""
        data = obj.chunk(0x100B)
        if data is not None and len(data) > 1:
            c = enc.cursor(data)
            c.flag()
            name = c.string()
        return cls(uid=uid, name=name)


# ---------------------------------------------------------------------------
# XML (src/model/xml.rs)
# ---------------------------------------------------------------------------

class XmlClass:
    DOCUMENT_NODE = 0xBF2F
    ELEMENT = 0xBF34
    MARKER_STRAND = 0x2B8
    CONTENT_HOLDER = 0x2C65


class XmlChunk:
    STORE = 0x16126
    PAGE = 0x16127
    NODE = 0xBF0D
    MARKERS = 0x1032
    MARKER_TREE = 0xCA57
    NODE_REF = 0xBF14


@dataclass
class XmlNode:
    #: (story UID, node ID) naming the node.
    key: tuple = (0, 0)
    #: The document node (the root element's parent), not an element.
    document: bool = False
    #: Tag UID (class 0xBF19); 0 for the document node.
    tag: int = 0
    #: UID of the tagged content (a story or a content holder), or 0.
    content: int = 0
    parent: tuple = (0, 0)
    children: list = field(default_factory=list)
    #: Start marker ID.
    start: int | None = None
    #: End marker ID; ``None`` for an element that is a single marker.
    end: int | None = None
    #: The element's attributes: name and value.
    attributes: list = field(default_factory=list)


def _xml_attributes(c: Cursor):
    """The attributes of an element node, after tag and content: 18
    bytes not identified, u32 count, then per attribute u32 name length
    and the name as text segments, u16 (not identified), u32 value
    length and the value. Returns them and the size read, when one or
    more attributes are followed by a child list that ends the part."""
    try:
        c.skip(18)
        n = c.u32()
        if n == 0 or n > c.remaining() // 10:
            return None
        start = c.pos

        def text(cursor):
            length = cursor.u32()
            if length > cursor.remaining():
                return None
            return cursor.segments(length)

        out = []
        for _ in range(n):
            name = text(c)
            if name is None:
                return None
            c.u16()
            value = text(c)
            if value is None:
                return None
            out.append((name, value))
        size = 22 + (c.pos - start)
        c.skip(16)
        k = c.u32()
        if k * 8 != c.remaining():
            return None
        return (out, size)
    except InddError:
        return None


def _parse_xml_node(enc: Encoding, kind: int, body: bytes) -> XmlNode:
    c = enc.cursor(body)
    n = c.u16()
    node_data = None
    markers = (0xFFFFFFFF, 0xFFFFFFFF)
    for _ in range(n):
        part_id = c.u32()
        length = c.u32()
        data = c.bytes(length)
        if part_id == XmlChunk.NODE:
            node_data = data
        elif part_id == XmlChunk.MARKERS and length >= 8:
            m = enc.cursor(data)
            markers = (m.u32(), m.u32())
    if node_data is None:
        raise _corrupt("XML node without chunk 0xBF0D")
    # u32 tag, u32 content, 22 bytes that end with the attribute count
    # (20 bytes in a file from InDesign 4.0, 4 in one from 3.0, without
    # attributes), the attributes, parent, self, child list.
    tag = enc.u32_at(node_data, 0)
    content = enc.u32_at(node_data, 4)
    if tag is None or content is None:
        raise _corrupt("XML node is too short")
    found = _xml_attributes(enc.cursor(node_data[8:]))
    if found is not None:
        attrs, size = found
        c = enc.cursor(node_data)
        c.skip(8 + size)
    else:
        attrs = []
        fixed = None
        for candidate in (22, 20, 4):
            probe = enc.cursor(node_data)
            try:
                probe.skip(candidate + 8)
                probe.skip(16)
                k = probe.u32()
                if k * 8 == probe.remaining():
                    fixed = candidate
                    break
            except InddError:
                continue
        if fixed is None:
            raise _corrupt("XML node of unknown layout")
        c = enc.cursor(node_data)
        c.skip(8)
        c.skip(fixed)
    parent = (c.u32(), c.u32())
    me = (c.u32(), c.u32())
    count = c.u32()
    if count > c.remaining() // 8:
        raise _corrupt("XML node child list is too long")
    children = [(c.u32(), c.u32()) for _ in range(count)]
    def marker(m):
        return m if m != 0xFFFFFFFF else None
    return XmlNode(key=me, document=kind == XmlClass.DOCUMENT_NODE,
                   tag=tag, content=content, parent=parent,
                   children=children, start=marker(markers[0]),
                   end=marker(markers[1]), attributes=attrs)


def read_xml_store(enc: Encoding, first_page: int, page_data) -> list:
    """Read the nodes of a store, following the pages from
    ``first_page``. ``page_data`` maps a page UID to its chunk bytes."""
    out = []
    page = first_page
    seen = []
    while page != 0 and page not in seen:
        seen.append(page)
        data = page_data(page)
        if data is None:
            break
        c = enc.cursor(data)
        next_page = c.u32()
        c.skip(12)
        n = c.u32()
        c.skip(4 * n)
        count = c.u32()
        for _ in range(count):
            kind = c.u32()
            size = c.u32()
            body = c.bytes(size)
            if kind in (XmlClass.DOCUMENT_NODE, XmlClass.ELEMENT):
                out.append(_parse_xml_node(enc, kind, body))
        page = next_page
    return out


def xml_marker_positions(enc: Encoding, data: bytes) -> dict:
    """The marker tree (chunk 0xCA57): marker ID to character
    position. A u16 that is 0 for an empty tree, then the nodes in
    preorder, each u16, u32 value, u32 marker ID, u32 length (1), u16
    flags (1 = has a left child, 2 = has a right child). The root's
    value is its position; a left child is *value* before its parent, a
    right child *value* after it."""
    out = {}
    c = enc.cursor(data)
    if c.u16() == 0:
        return out
    # (parent position, side): side 0 for the root, -1 left, 1 right.
    stack = [(0, 0)]
    while stack:
        parent, side = stack.pop()
        c.skip(2)
        value = c.u32()
        marker = c.u32()
        c.skip(4)
        flags = c.u16()
        if side == 0:
            pos = value
        elif side == -1:
            pos = parent - value
        else:
            pos = parent + value
        if pos < 0 or flags > 3:
            raise _corrupt("XML marker tree is inconsistent")
        out[marker] = pos
        if flags & 2:
            stack.append((pos, 1))
        if flags & 1:
            stack.append((pos, -1))
    return out


# ---------------------------------------------------------------------------
# Story links (src/model/story_link.rs)
# ---------------------------------------------------------------------------

@dataclass
class StyleMapping:
    """A style mapping (chunk 0x21D30): names as stored and the rule
    code."""

    source: str = ""
    destination: str = ""
    #: 0 ``StyleMappingRule``, 1 ``GroupMappingRule``, 2
    #: ``StyleToGroupMappingRule``.
    rule: int = 0


@dataclass
class StyleMappings:
    """The style mappings of a story or of the document (chunk
    0x21D30)."""

    paragraph: list = field(default_factory=list)
    character: list = field(default_factory=list)
    #: Entries in lists 3 to 5, which no sample fills.
    other: int = 0


class ImportOptionsKind:
    WORD = "word"
    EXCEL = "excel"


@dataclass
class ExcelOptions:
    """Excel import options (chunk 0x12213 of the options object)."""

    #: u32 at 16: 0 formatted, 1 unformatted table.
    table: int = 0
    #: The sheet name, range and the string after them, as IDML writes
    #: them.
    sheet: str = ""
    range: str = ""
    view: str = ""


@dataclass
class StoryLinkChunks:
    """The settings of a linked or placed story."""

    #: The story's link (chunk 0x8CBC).
    link: Link | None = None
    #: Chunk 0x21D12: ``ApplyStyleMappings``.
    apply_style_mappings: bool | None = None
    mappings: StyleMappings = field(default_factory=StyleMappings)
    import_options: tuple | None = None
    #: Chunk 0x12216: paragraph and character style names.
    mapping_prefs: tuple | None = None


_MAPPING_MIN = 1 + 4 + 1 + 4 + 4


def _style_mappings(c: Cursor) -> StyleMappings:
    """Chunk 0x21D30: five lists of mappings, each bounded by the data
    left."""
    out = StyleMappings()
    for list_index in range(5):
        n = c.u32()
        if n > c.remaining() // _MAPPING_MIN:
            raise _corrupt(f"{n} style mappings")
        for _ in range(n):
            c.flag()
            source = c.string()
            c.flag()
            destination = c.string()
            rule = c.u32()
            mapping = StyleMapping(source=source, destination=destination,
                                   rule=rule)
            if list_index == 0:
                out.paragraph.append(mapping)
            elif list_index == 1:
                out.character.append(mapping)
            else:
                out.other += 1
    return out


# ---------------------------------------------------------------------------
# Media (src/model/media.rs)
# ---------------------------------------------------------------------------

class MediaKind:
    MOVIE = "movie"
    SOUND = "sound"


@dataclass
class MediaSettings:
    """Playback settings (chunk 0x14004). A flag is true when its key
    is stored with value 1."""

    #: Key 0x14000.
    name: str | None = None
    #: Key 0x14002; empty without it.
    description: str = ""
    #: Key 0x14008; empty without it.
    controller_skin: str = ""
    #: Key 0x14003.
    custom_poster: bool = False
    #: Key 0x14006.
    play_on_page_turn: bool = False
    #: Key 0x1400C.
    show_controller: bool = False
    #: Key 0x1400E: ``MovieLoop`` or ``SoundLoop``.
    looped: bool = False
    #: Key 0x14004.
    embed_in_pdf: bool = False
    #: Key 0x1400D.
    stop_on_page_turn: bool = False


def _media_settings(enc: Encoding, d: bytes):
    """Chunk 0x14004: list 1 (u32 count, then u32 key and u16 value per
    entry), list 2 (u32 values), four lists that are empty in every
    sample, then list 7 (u32 key, flagged string). Returns (settings,
    complete): ``complete`` is False when one of lists 3 to 6 is not
    empty — the flags read so far, without the strings."""
    c = enc.cursor(d)
    s = MediaSettings()
    for _ in range(c.u32()):
        key = c.u32()
        on = c.u16() == 1
        if key == 0x14003:
            s.custom_poster = on
        elif key == 0x14004:
            s.embed_in_pdf = on
        elif key == 0x14006:
            s.play_on_page_turn = on
        elif key == 0x1400C:
            s.show_controller = on
        elif key == 0x1400D:
            s.stop_on_page_turn = on
        elif key == 0x1400E:
            s.looped = on
    for _ in range(c.u32()):
        c.skip(8)
    complete = True
    for _ in range(4):
        if c.u32() != 0:
            complete = False
    if not complete:
        return (s, False)
    for _ in range(c.u32()):
        key = c.u32()
        c.flag()
        text = c.string()
        if key == 0x14000:
            s.name = text
        elif key == 0x14002:
            s.description = text
        elif key == 0x14008:
            s.controller_skin = text
    return (s, True)


@dataclass
class Media:
    """A placed movie or sound."""

    uid: int
    kind: str = MediaKind.MOVIE
    transform: Matrix = field(default_factory=Matrix)
    paths: list = field(default_factory=list)
    #: Chunk 0x1633: left, top, right, bottom (``GraphicBounds``).
    bounds: list = field(default_factory=lambda: [0.0] * 4)
    link: Link | None = None
    settings: MediaSettings = field(default_factory=MediaSettings)
    props: ItemProps = field(default_factory=ItemProps)
    object_style: int | None = None
    #: The attribute list (chunk 0x6E03).
    attrs: Attrs = field(default_factory=Attrs)
    text_wrap: TextWrap | None = None
    label: list = field(default_factory=list)
    #: The poster: the first child that is a placed graphic.
    poster: Graphic | None = None


# ---------------------------------------------------------------------------
# Animation (src/model/animation.rs)
# ---------------------------------------------------------------------------

@dataclass
class MotionPreset:
    """A motion preset (class 0x20517, chunk 0x20532)."""

    uid: int
    name: Name = field(default_factory=lambda: Name(True, ""))
    #: The preset's XML (IDML ``Properties/Contents``).
    contents: str = ""


@dataclass
class TimingGroup:
    plays: int = 0
    targets: list = field(default_factory=list)


@dataclass
class TimingList:
    """One timing list of a spread or button (chunks 0x2056D and
    0x2056E)."""

    #: Trigger code: 1 on page load, 5 on click, 6 on release, 7 on
    #: rollover.
    trigger: int = 0
    groups: list = field(default_factory=list)


@dataclass
class TimingTarget:
    uid: int = 0
    role: int = 0
    action: int = 0
    #: Delay in seconds.
    delay: float = 0.0


@dataclass
class Animation:
    """The animation of a page item (chunk 0x2055F)."""

    custom: bool = False
    #: The motion preset's UID.
    preset: int = 0
    #: Seconds.
    duration: float = 0.0
    #: Design option code: 0 from current appearance, 2 to current
    #: location.
    design: int = 0
    plays: int = 0
    #: Ease code: 0 none, 4 custom.
    ease: int = 0
    offsets: list = field(default_factory=lambda: [0.0, 0.0])
    #: Motion path key frames: frame, x, y.
    path: list = field(default_factory=list)
    #: Points of a motion path without key frames.
    points: list = field(default_factory=list)
    #: Animated properties by stored name, with their keys (frame,
    #: value).
    properties: list = field(default_factory=list)


def _parse_timing(c: Cursor) -> list:
    lists = c.u32()
    out = []
    for _ in range(lists):
        trigger = c.u32()
        groups = c.u32()
        timing = TimingList(trigger=trigger)
        for _ in range(groups):
            n = c.u32()
            targets = []
            for _ in range(n):
                uid = c.u32()
                role = c.u32()
                action = c.u32()
                c.u32()
                delay = c.f64()
                targets.append(TimingTarget(uid=uid, role=role,
                                            action=action, delay=delay))
            plays = c.u32()
            c.u16()
            timing.groups.append(TimingGroup(plays=plays, targets=targets))
        out.append(timing)
    if c.remaining() != 0:
        raise _corrupt(f"timing: {c.remaining()} bytes left")
    return out


def _parse_animation(c: Cursor) -> Animation:
    c.u16()
    custom = c.u16() == 1
    preset = c.u32()
    duration = c.f64()
    # Frames per second and frames.
    c.skip(8)
    design = c.u32()
    c.u32()
    plays = c.u32()
    c.u16()
    ease = c.u32()
    offsets = [c.f64(), c.f64()]
    path = []
    for _ in range(c.u32()):
        frame = c.u32()
        c.u32()
        path.append((frame, c.f64(), c.f64()))
    points = []
    for _ in range(c.u32()):
        c.u32()
        points.append((c.f64(), c.f64()))
    properties = []
    for _ in range(c.u32()):
        name = _counted(c)
        _counted(c)
        keys = []
        for _ in range(c.u32()):
            frame = c.u32()
            c.u32()
            keys.append((frame, c.f64()))
        properties.append((name, keys))
    if c.remaining() != 0:
        raise _corrupt(f"animation: {c.remaining()} bytes left")
    return Animation(custom=custom, preset=preset, duration=duration,
                     design=design, plays=plays, ease=ease,
                     offsets=offsets, path=path, points=points,
                     properties=properties)


# ---------------------------------------------------------------------------
# Behaviours (src/model/behavior.rs)
# ---------------------------------------------------------------------------

class BehaviorClass:
    FORM_BEHAVIORS = 0x15601
    BEHAVIOR_LABEL = 0x156E1


_BEHAVIOR_CLASSES = [
    (0x15613, 0x15614),
    (0x15615, 0x1561A),
    (0x15616, 0x1561B),
    (0x1561B, 0x15622),
    (0x1562C, 0x1562C),
    (0x15636, 0x15634),
    (0x15664, 0x15650),
    (0x1566E, 0x1565A),
    (0x15678, 0x15664),
    (0x156D3, 0x156F0),
    (0x156DD, 0x15703),
    (0x156F4, 0x1571C),
    (0x15625, 0x156DE),
    (0x1563F, 0x1566D),
    (0x15643, 0x1567D),
    (0x15660, 0x15680),
    (0x156F9, 0x15721),
]


@dataclass
class Behavior:
    uid: int
    #: Event code of its group: 0x15602 mouse down, 0x15603 mouse up,
    #: 0x15604 mouse enter.
    event: int = 0
    enabled: bool = False
    #: The name: a built-in key (``$$$/Forms Action/…``) in every
    #: sample.
    name: Name = field(default_factory=lambda: Name(True, ""))
    #: Chunk 0x156E1.
    label: list = field(default_factory=list)
    #: ("first_page", zoom) | ("next_page", zoom) |
    #: ("previous_page", zoom) | ("next_view", zoom) |
    #: ("page", page, zoom) | ("anchor", file, anchor, zoom) |
    #: ("url", url|None) | ("open_file", path) |
    #: ("submit_form", url|None) | ("clear_form",) | ("print_form",) |
    #: ("show_hide", [(uid, shown)]) | ("state", mso, state) |
    #: ("next_state", mso) | ("previous_state", mso) |
    #: ("movie", item, operation, point|None) |
    #: ("sound", item, operation)
    kind: tuple = ("clear_form",)

    def element(self) -> str:
        return {
            "first_page": "GotoFirstPageBehavior",
            "next_page": "GotoNextPageBehavior",
            "previous_page": "GotoPreviousPageBehavior",
            "next_view": "GotoNextViewBehavior",
            "page": "GotoPageBehavior",
            "anchor": "GotoAnchorBehavior",
            "url": "GotoURLBehavior",
            "open_file": "OpenFileBehavior",
            "submit_form": "SubmitFormBehavior",
            "clear_form": "ClearFormBehavior",
            "print_form": "PrintFormBehavior",
            "show_hide": "ShowHideFieldsBehavior",
            "state": "GotoStateBehavior",
            "next_state": "GotoNextStateBehavior",
            "previous_state": "GotoPreviousStateBehavior",
            "movie": "MovieBehavior",
            "sound": "SoundBehavior",
        }[self.kind[0]]


def _parse_behavior_list(c: Cursor):
    """The groups of chunk 0x15601: u32 group count, then per group u32
    event code, u32 count and the behaviour UIDs. Returns (event, UID)
    in group order."""
    groups = c.u32()
    if groups > c.remaining() // 8:
        raise _corrupt(f"{groups} behaviour groups")
    out = []
    for _ in range(groups):
        event = c.u32()
        n = c.u32()
        if n > c.remaining() // 4:
            raise _corrupt(f"{n} behaviours in a group")
        for _ in range(n):
            out.append((event, c.u32()))
    if c.remaining() != 0:
        raise _corrupt(f"behaviour list: {c.remaining()} bytes left")
    return out


def _behavior_navigation(c: Cursor, page: bool):
    """Page navigation fields: u32 navigation kind, u32 page index (go
    to page only), u32 zoom. Returns the page index and the zoom."""
    c.u32()
    index = c.u32() if page else 0
    return (index, c.u32())


def _behavior_url(c: Cursor):
    """A URL: flag byte and in-object string; ``None`` for a built-in
    key."""
    name = c.name()
    return None if name.builtin else name.name


def _parse_behavior_data(c: Cursor, cls: int):
    """A behaviour's data chunk, of class ``cls``: u32 owner, u32 type
    code, the name (flag byte, string), u16 enabled, then the type's
    fields."""
    c.u32()
    c.u32()
    name = c.name()
    enabled = c.u16() != 0
    if cls == 0x15613:
        kind = ("first_page", _behavior_navigation(c, False)[1])
    elif cls == 0x15615:
        kind = ("next_page", _behavior_navigation(c, False)[1])
    elif cls == 0x15616:
        kind = ("previous_page", _behavior_navigation(c, False)[1])
    elif cls == 0x156DD:
        kind = ("next_view", _behavior_navigation(c, False)[1])
    elif cls == 0x15625:
        page, zoom = _behavior_navigation(c, True)
        kind = ("page", page, zoom)
    elif cls == 0x156D3:
        c.flag()
        file = c.string()
        anchor = c.u32()
        zoom = c.u32()
        kind = ("anchor", file, anchor, zoom)
    elif cls == 0x1561B:
        kind = ("url", _behavior_url(c))
    elif cls == 0x15678:
        c.flag()
        kind = ("open_file", c.string())
    elif cls == 0x1566E:
        c.u32()
        kind = ("submit_form", _behavior_url(c))
    elif cls == 0x15664:
        kind = ("clear_form",)
    elif cls == 0x156F9:
        kind = ("print_form",)
    elif cls == 0x15636:
        n = c.u32()
        if n > c.remaining() // 8:
            raise _corrupt(f"{n} fields to show or hide")
        fields = []
        for _ in range(n):
            uid = c.u32()
            action = c.u32()
            if action > 1:
                raise _corrupt(f"show/hide action {action}")
            fields.append((uid, action == 1))
        kind = ("show_hide", fields)
    elif cls in (0x1563F, 0x15643, 0x15660):
        mso = c.u32()
        c.u32()
        c.u16()
        c.u16()
        state = c.i32()
        if cls == 0x1563F:
            kind = ("state", mso, state)
        elif cls == 0x15643:
            kind = ("next_state", mso)
        else:
            kind = ("previous_state", mso)
    elif cls == 0x1562C:
        item = c.u32()
        operation = c.u32()
        point = c.i32() if c.remaining() > 0 else None
        kind = ("movie", item, operation, point)
    elif cls == 0x156F4:
        item = c.u32()
        operation = c.u32()
        c.f64()
        c.u32()
        kind = ("sound", item, operation)
    else:
        raise _corrupt(f"class {cls:#x} is not a behaviour")
    if c.remaining() != 0:
        raise _corrupt(f"behaviour data: {c.remaining()} bytes left")
    return (name, enabled, kind)


# ---------------------------------------------------------------------------
# CJK (src/model/cjk.rs)
# ---------------------------------------------------------------------------

class CjkClass:
    COMPOSITE_FONT = 0xCB02
    COMPOSITE_FONT_ENTRY = 0xCB03
    #: Kinsoku tables: hard, soft, Korean, Simplified Chinese,
    #: Traditional Chinese and custom.
    KINSOKU = [0x4209, 0x420A, 0x42B4, 0x42B5, 0x42B6, 0x4204]
    CUSTOM_KINSOKU = 0x4204
    MOJIKUMI = 0x4206
    CUSTOM_MOJIKUMI = 0x4203


class CjkChunk:
    TABLE_NAME = 0x100B
    COMPOSITE_FONT = 0xCB02
    COMPOSITE_FONT_ENTRY = 0xCB03
    KINSOKU_CHARS = 0x4214
    MOJIKUMI_AKI = 0x420A


def _code_point(c: Cursor) -> int:
    """A code point stored as one UTF-16 unit or a surrogate pair."""
    hi = c.u16()
    if 0xD800 <= hi < 0xDC00:
        lo = c.u16()
        return 0x10000 + ((hi - 0xD800) << 10) + ((lo - 0xDC00) & 0x3FF)
    return hi


@dataclass
class CompositeFontEntry:
    """One entry of a composite font."""

    uid: int
    name: Name = field(default_factory=lambda: Name(True, ""))
    font_family: int = 0
    font_style: Name = field(default_factory=lambda: Name(True, ""))
    #: Four f64: relative size, baseline shift, then the horizontal and
    #: vertical scale (100 in every sample, so their order is not shown).
    numbers: list = field(default_factory=lambda: [0.0] * 4)
    #: The u16 after the numbers: 1 for ``Locked="true"``, 0 for false.
    locked: int = 0
    #: Character ranges as (first, last) code points.
    ranges: list = field(default_factory=list)
    #: The four u16 after the ranges: all 1 for ``ScaleOption="true"``,
    #: all 0 for false.
    scale: list = field(default_factory=lambda: [0] * 4)

    def characters(self) -> str:
        """The characters of the ranges, in order (IDML
        ``CustomCharacters``)."""
        out = []
        for first, last in self.ranges:
            for code in range(first, last + 1):
                if 0 <= code <= 0x10FFFF and not 0xD800 <= code < 0xE000:
                    out.append(chr(code))
        return "".join(out)

    @classmethod
    def read(cls, uid: int, obj):
        """Chunk 0xCB03: name, u32 font family, font style, four f64,
        u16 locked, u16 range count, ranges (first, last, first), four
        u16."""
        data = obj.chunk(CjkChunk.COMPOSITE_FONT_ENTRY)
        if data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        name = c.name()
        font_family = c.u32()
        font_style = c.name()
        numbers = [c.f64() for _ in range(4)]
        locked = c.u16()
        n = c.u16()
        ranges = []
        for _ in range(n):
            first = _code_point(c)
            last = _code_point(c)
            _code_point(c)
            ranges.append((first, last))
        scale = [c.u16() for _ in range(4)]
        return cls(uid=uid, name=name, font_family=font_family,
                   font_style=font_style, numbers=numbers, locked=locked,
                   ranges=ranges, scale=scale)


@dataclass
class CompositeFont:
    """A composite font: its name and the UIDs of its entries, in IDML
    order."""

    uid: int
    name: Name = field(default_factory=lambda: Name(True, ""))
    entries: list = field(default_factory=list)

    @classmethod
    def read(cls, uid: int, obj):
        """Chunk 0xCB02: four flagged strings (the name first), a u32
        and a u16 not identified, then a u16 count and the entry UIDs,
        which end the chunk. In files from InDesign 3.0 four zero bytes
        come before the name. A chunk of another layout gives the first
        u16 count after the name whose UIDs end the chunk."""
        data = obj.chunk(CjkChunk.COMPOSITE_FONT)
        if data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        try:
            name = c.name()
        except InddError as first_error:
            if data[:4] == bytes(4):
                c = enc.cursor(data)
                c.skip(4)
                try:
                    name = c.name()
                except InddError:
                    raise first_error from None
            else:
                raise first_error
        start = c.pos
        entries = None
        try:
            for _ in range(3):
                c.name()
            c.u32()
            c.u16()
            n = c.u16()
            if c.remaining() == 4 * n:
                entries = [c.u32() for _ in range(n)]
        except InddError:
            entries = None
        if entries is None:
            entries = []
            for p in range(start, max(len(data) - 1, start)):
                n = enc.u16_at(data, p)
                if n is not None and n > 0 and p + 2 + 4 * n == len(data):
                    sub = enc.cursor(data[p:])
                    count = sub.u16()
                    entries = [sub.u32() for _ in range(count)]
                    break
        return cls(uid=uid, name=name, entries=entries)


@dataclass
class AkiEntry:
    """One spacing entry of a custom mojikumi table."""

    target_class: int = 0
    side_class: int = 0
    minimum: float = 0.0
    desired: float = 0.0
    maximum: float = 0.0
    compression_priority: int = 0
    aki_does_not_float: bool = False
    side_is_after_target: bool = False


@dataclass
class MojikumiSettings:
    """The settings of a custom mojikumi table (chunk 0x420A)."""

    #: The ``kMojikumiDefaultName<n>`` code of the table it is based on;
    #: 0 for none.
    based_on: int = 0
    entries: list = field(default_factory=list)

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> MojikumiSettings:
        c = enc.cursor(d)
        c.skip(8)
        n = c.u32()
        if len(d) != n * 34 + 22:
            raise _corrupt(f"mojikumi settings of {len(d)} bytes "
                           f"for {n} entries")
        entries = []
        for _ in range(n):
            target_class = c.u16()
            side_class = c.u16()
            minimum = c.f64()
            desired = c.f64()
            maximum = c.f64()
            compression_priority = c.u16()
            aki_does_not_float = _bool16(c)
            side_is_after_target = _bool16(c)
            entries.append(AkiEntry(
                target_class=target_class, side_class=side_class,
                minimum=minimum, desired=desired, maximum=maximum,
                compression_priority=compression_priority,
                aki_does_not_float=aki_does_not_float,
                side_is_after_target=side_is_after_target))
        c.skip(8)
        based_on = c.u16()
        return cls(based_on=based_on, entries=entries)


def _bool16(c: Cursor) -> bool:
    v = c.u16()
    if v == 0:
        return False
    if v == 1:
        return True
    raise _corrupt(f"mojikumi flag {v} is not 0 or 1")


@dataclass
class CjkTable:
    uid: int
    mojikumi: bool = False
    name: Name = field(default_factory=lambda: Name(True, ""))
    #: The character lists of a custom kinsoku table: cannot begin a
    #: line, cannot end a line, (not identified), hanging punctuation,
    #: cannot be separated.
    chars: list | None = None
    #: The settings of a custom mojikumi table.
    custom_mojikumi: MojikumiSettings | None = None

    @classmethod
    def read(cls, uid: int, class_id: int, obj):
        data = obj.chunk(CjkChunk.TABLE_NAME)
        if data is None:
            return None
        enc = obj.encoding
        name = enc.cursor(data).name()
        chars = None
        kinsoku = obj.chunk(CjkChunk.KINSOKU_CHARS)
        if kinsoku is not None and class_id == CjkClass.CUSTOM_KINSOKU:
            c = enc.cursor(kinsoku)
            counts = [c.u16() for _ in range(5)]
            lists = []
            for n in counts:
                units = [c.u16() for _ in range(n)]
                lists.append(_utf16_lossy(units))
            chars = lists
        custom_mojikumi = None
        if class_id == CjkClass.CUSTOM_MOJIKUMI:
            settings_data = obj.chunk(CjkChunk.MOJIKUMI_AKI)
            if settings_data is None:
                raise _corrupt("mojikumi table without settings")
            custom_mojikumi = MojikumiSettings.read(enc, settings_data)
        return cls(uid=uid,
                   mojikumi=class_id in (CjkClass.MOJIKUMI,
                                         CjkClass.CUSTOM_MOJIKUMI),
                   name=name, chars=chars,
                   custom_mojikumi=custom_mojikumi)


# ---------------------------------------------------------------------------
# Index (src/model/index.rs)
# ---------------------------------------------------------------------------

class IndexClass:
    INDEX = 0x13004
    SECTION = 0x13005
    PAGE_REFERENCE = 0x13006


class IndexChunk:
    SECTIONS = 0x13005
    TOPICS = 0x13007


@dataclass
class Topic:
    """An index topic."""

    name: str = ""
    sort_order: str = ""
    #: The page references (class 0x13006) of this topic.
    refs: list = field(default_factory=list)
    children: list = field(default_factory=list)


@dataclass
class Index:
    """The document's index."""

    uid: int
    #: The level-1 topics, section by section.
    topics: list = field(default_factory=list)


def _read_topics(c: Cursor) -> list:
    """The topic records of a section (chunk 0x13007), as a tree. The
    records are depth first; each has its level (1 for a top topic)."""
    top = c.u32()
    c.u16()
    n = c.u32()
    stack = []
    out = []

    def close(depth):
        while len(stack) > depth:
            topic = stack.pop()
            if stack:
                stack[-1].children.append(topic)
            else:
                out.append(topic)

    for _ in range(n):
        c.u16()
        level = c.u16()
        if level == 0 or level > len(stack) + 1:
            raise _corrupt(f"index topic of level {level}")
        c.u32()
        name = _counted(c)
        sort_order = _counted(c)
        m = c.u32()
        if m > c.remaining() // 4:
            raise _corrupt(f"{m} page references")
        refs = [c.u32() for _ in range(m)]
        # The cross references of the topic (class 0x13007), then a u32.
        k = c.u32()
        if k > c.remaining() // 4:
            raise _corrupt(f"{k} cross references")
        c.skip(4 * k + 4)
        close(level - 1)
        stack.append(Topic(name=name, sort_order=sort_order, refs=refs))
    close(0)
    if len(out) != top or c.remaining() != 0:
        raise _corrupt(f"{len(out)} top topics, {top} expected, "
                       f"{c.remaining()} bytes left")
    return out


# ---------------------------------------------------------------------------
# Cross-reference formats (src/model/xref.rs)
# ---------------------------------------------------------------------------

XREF_CLASS = 0x1355E
XREF_CHUNK = 0x13593

#: Building block type codes and their IDML ``BlockType``.
XREF_BLOCK_TYPES = [
    (0, "CustomStringBuildingBlock"),
    (3, "PageNumberBuildingBlock"),
    (4, "FullParagraphBuildingBlock"),
    (5, "ParagraphNumberBuildingBlock"),
    (6, "ParagraphTextBuildingBlock"),
    (7, "BookmarkNameBuildingBlock"),
]


@dataclass
class BuildingBlock:
    kind: int = 0
    #: The 10 bytes after the type are all 0, as in every sample.
    zero_fields: bool = True
    #: ``CustomText``.
    text: Name = field(default_factory=lambda: Name(True, ""))

    def type_name(self):
        for code, name in XREF_BLOCK_TYPES:
            if code == self.kind:
                return name
        return None


@dataclass
class CrossReferenceFormat:
    uid: int
    name: str = ""
    #: The format's u32 character style field; 0 in every sample.
    character_style: int = 0
    blocks: list = field(default_factory=list)

    @classmethod
    def read(cls, uid: int, obj):
        data = obj.chunk(XREF_CHUNK)
        if data is None:
            return None
        enc = obj.encoding
        c = enc.cursor(data)
        c.u8()
        name = c.string()
        character_style = c.u32()
        c.skip(2)
        n = c.u32()
        blocks = []
        for _ in range(n):
            kind = c.u32()
            zero_fields = all(b == 0 for b in c.bytes(10))
            text = c.name()
            blocks.append(BuildingBlock(kind=kind,
                                        zero_fields=zero_fields,
                                        text=text))
        return cls(uid=uid, name=name,
                   character_style=character_style, blocks=blocks)


# ---------------------------------------------------------------------------
# XMP (src/model/xmp.rs)
# ---------------------------------------------------------------------------

@dataclass
class XmpDate:
    """An XMP date: the UTC second and the offset from UTC in seconds."""

    utc: int = 0
    offset: int = 0


@dataclass
class XmpColorant:
    """An XMP colourant: name, mode, type, RGB and CMYK values."""

    name: str = ""
    mode: str = ""
    type_code: str = ""
    rgb: list = field(default_factory=list)
    cmyk: list = field(default_factory=list)


def _xmp_text(packet: bytes, tag: str) -> str | None:
    import re
    pattern = re.compile(rb"<" + re.escape(tag.encode())
                         + rb"[^>]*>([^<]*)</" + re.escape(tag.encode())
                         + rb">")
    m = pattern.search(packet)
    if m is None:
        return None
    return m.group(1).decode("utf-8", errors="replace")


def xmp_dates(packet: bytes) -> list:
    """The XMP dates of a packet: every ``xmp:CreateDate`` and
    ``xmp:ModifyDate`` value, as (UTC second, offset seconds)."""
    import re
    from datetime import datetime
    out = []
    for tag in (b"xmp:CreateDate", b"xmp:ModifyDate"):
        for m in re.finditer(rb"<" + tag + rb"[^>]*>([^<]*)</" + tag + rb">",
                             packet):
            text = m.group(1).decode("utf-8", errors="replace").strip()
            try:
                if text.endswith("Z"):
                    dt = datetime.fromisoformat(text[:-1] + "+00:00")
                else:
                    dt = datetime.fromisoformat(text)
                utc = int(dt.astimezone(UTC).timestamp())
                offset = int(dt.utcoffset().total_seconds()) \
                    if dt.utcoffset() is not None else 0
                out.append(XmpDate(utc=utc, offset=offset))
            except ValueError:
                continue
    return out


def xmp_colorants(packet: bytes) -> list:
    """The XMP colourants of a packet (``xmpG:Colorant`` elements)."""
    import re
    out = []
    for m in re.finditer(rb"<xmpG:Colorant[^>]*>(.*?)</xmpG:Colorant>",
                         packet, re.DOTALL):
        body = m.group(1)
        name = _xmp_text(body, "xmpG:swatchName") or ""
        mode = _xmp_text(body, "xmpG:mode") or ""
        type_code = _xmp_text(body, "xmpG:type") or ""
        rgb = []
        for tag in ("xmpG:red", "xmpG:green", "xmpG:blue"):
            value = _xmp_text(body, tag)
            rgb.append(float(value) if value else 0.0)
        cmyk = []
        for tag in ("xmpG:cyan", "xmpG:magenta", "xmpG:yellow",
                    "xmpG:black"):
            value = _xmp_text(body, tag)
            cmyk.append(float(value) if value else 0.0)
        out.append(XmpColorant(name=name, mode=mode, type_code=type_code,
                               rgb=rgb, cmyk=cmyk))
    return out


def xmp_black_name(packet: bytes) -> str | None:
    """The name of the black colourant of a packet, if it has one."""
    for colorant in xmp_colorants(packet):
        if colorant.cmyk == [0.0, 0.0, 0.0, 1.0]:
            return colorant.name
    return None


def days_from_civil(year: int, month: int, day: int) -> int:
    """Days since 1970-01-01 for a civil date."""
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = (month + 9) % 12
    doy = (153 * mp + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def civil_from_days(days: int):
    """The civil date of days since 1970-01-01."""
    z = days + 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    day = doy - (153 * mp + 2) // 5 + 1
    month = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if month <= 2 else 0), month, day)


def day_of_year(utc: int) -> int:
    """The day of the year of a UTC second."""
    year, month, day = civil_from_days(utc // 86400)
    return days_from_civil(year, month, day) - days_from_civil(year, 1, 1)


# ---------------------------------------------------------------------------
# Preferences (src/model/prefs.rs)
# ---------------------------------------------------------------------------

@dataclass
class PrefValue:
    """A value for an element of ``Resources/Preferences.xml`` or a
    document setting in ``designmap.xml``."""

    #: IDML element (``ViewPreference``, ``Document``, ...).
    element: str = ""
    #: Attribute name.
    name: str = ""
    value: str = ""


class RulerOrigin:
    """The point guide locations are measured from."""

    SPREAD = "spread"
    PAGE = "page"
    SPINE = "spine"

    @staticmethod
    def idml(origin: str) -> str:
        return {"spread": "SpreadOrigin", "page": "PageOrigin",
                "spine": "SpineOrigin"}[origin]


class FrameGridColor:
    """The colour of baseline frame grids."""

    RGB = "rgb"
    CHARCOAL = "charcoal"
    LIGHT_BLUE = "light_blue"


@dataclass
class EndnoteOptions:
    """The document's endnote options (preferences chunk 0x2261E)."""

    title: str = ""
    title_style: int = 0
    #: Numbering style code.
    numbering: int = 0
    start_at: int = 0
    #: Restart code.
    restart: int = 0
    #: Marker positioning code.
    positioning: int = 0
    marker_style: int = 0
    text_style: int = 0
    separator: str = ""
    #: The last five fields: two u32, two strings, a u32. They hold one
    #: value in every sample.
    rest: tuple = (0, 0, "", "", 0)

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> EndnoteOptions:
        c = enc.cursor(d)
        title = _counted(c)
        title_style = c.u32()
        numbering = c.u32()
        start_at = c.u32()
        restart = c.u32()
        positioning = c.u32()
        marker_style = c.u32()
        text_style = c.u32()
        separator = _counted(c)
        rest = (c.u32(), c.u32(), _counted(c), _counted(c), c.u32())
        return cls(title=title, title_style=title_style,
                   numbering=numbering, start_at=start_at,
                   restart=restart, positioning=positioning,
                   marker_style=marker_style, text_style=text_style,
                   separator=separator, rest=rest)


@dataclass
class FootnoteRule:
    """A footnote rule block (78 bytes)."""

    on: bool = False
    #: Stroke type code.
    stroke: int = 0
    color: int = 0
    weight: float = 0.0
    tint: float = 0.0
    gap_color: int = 0
    gap_tint: float = 0.0
    left_indent: float = 0.0
    width: float = 0.0
    offset: float = 0.0


@dataclass
class FootnoteTail:
    no_splitting: bool = False
    #: f64 at 4: ``FootnoteMinimumFirstBaselineOffset``.
    minimum_first_baseline: float = 0.0
    #: ``EnableStraddling``; only in the 174-byte tail.
    straddling: bool | None = None
    #: The rule and the continuing rule.
    rules: list = field(default_factory=lambda: [FootnoteRule(),
                                                 FootnoteRule()])


@dataclass
class FootnoteOptions:
    """The document's footnote options (preferences chunk 0x2820)."""

    marker_style: int = 0
    text_style: int = 0
    #: Numbering style code.
    numbering: int = 0
    start_at: int = 0
    #: Restart code.
    restart: int = 0
    #: Superscript and ruby flags of the marker.
    marker: tuple = (0, 0)
    space_between: float = 0.0
    spacer: float = 0.0
    #: u16 at 40: ``EosPlacement``.
    eos_placement: int = 0
    #: Prefix and suffix display code.
    prefix_suffix: int = 0
    prefix: str = ""
    suffix: str = ""
    separator: str = ""
    #: The fields after the strings, when the tail has a known length.
    tail: FootnoteTail | None = None

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> FootnoteOptions:
        c = enc.cursor(d)
        marker_style = c.u32()
        text_style = c.u32()
        numbering = c.u32()
        start_at = c.u32()
        restart = c.u16()
        superscript = c.u16()
        c.skip(2)
        ruby = c.u16()
        space_between = c.f64()
        spacer = c.f64()
        eos_placement = c.u16()
        prefix_suffix = c.u16()
        prefix = _counted(c)
        suffix = _counted(c)
        separator = _counted(c)
        t = c.remaining()
        tail = None
        if t in (172, 174):
            tail_bytes = c.bytes(t)
            rule_at = 18 if t == 174 else 16

            def rule(at):
                r = enc.cursor(tail_bytes[at:at + 78])
                on = r.u16() != 0
                stroke = r.u32()
                r.skip(4)
                color = r.u32()
                weight = r.f64()
                tint = r.f64()
                r.skip(2)
                gap_color = r.u32()
                gap_tint = r.f64()
                r.skip(2)
                return FootnoteRule(
                    on=on, stroke=stroke, color=color, weight=weight,
                    tint=tint, gap_color=gap_color, gap_tint=gap_tint,
                    left_indent=r.f64(), width=r.f64(),
                    offset=r.f64())

            tail = FootnoteTail(
                no_splitting=enc.u16_at(tail_bytes, 0) != 0,
                minimum_first_baseline=enc.f64_at(tail_bytes, 4) or 0.0,
                straddling=(enc.u16_at(tail_bytes, 12) != 0
                            if t == 174 else None),
                rules=[rule(rule_at), rule(rule_at + 78)])
        return cls(marker_style=marker_style, text_style=text_style,
                   numbering=numbering, start_at=start_at,
                   restart=restart, marker=(superscript, ruby),
                   space_between=space_between, spacer=spacer,
                   eos_placement=eos_placement,
                   prefix_suffix=prefix_suffix, prefix=prefix,
                   suffix=suffix, separator=separator, tail=tail)


@dataclass
class PrefNode:
    """An element in a tree of preference values."""

    tag: str = ""
    attrs: list = field(default_factory=list)
    children: list = field(default_factory=list)


@dataclass
class IndexOptions:
    """The index options (preferences chunk 0x13010)."""

    title: str = ""
    #: Name of the title's paragraph style.
    title_style: str = ""
    replace_existing: bool = False
    include_book_documents: bool = False
    include_section_headings: bool = False
    #: ``FollowingTopicSeparator``, ``BetweenPageNumbersSeparator``,
    #: ``BetweenEntriesSeparator``, ``BeforeCrossReferenceSeparator``,
    #: ``PageRangeSeparator``, ``EntryEndSeparator``.
    separators: list = field(default_factory=lambda: [""] * 6)
    #: Style names: ``Level1Style`` to ``Level4Style`` and
    #: ``SectionHeadingStyle`` (paragraph styles), then
    #: ``PageNumberStyle``, ``CrossReferenceStyle`` and
    #: ``CrossReferenceTopicStyle`` (character styles).
    styles: list = field(default_factory=lambda: [""] * 8)

    @classmethod
    def read(cls, enc: Encoding, d: bytes) -> IndexOptions:
        c = enc.cursor(d)
        title = _counted(c)
        title_style = _flagged(c)[1]
        replace_existing = c.u8() == 1
        include_book_documents = c.u8() == 1
        include_section_headings = c.bytes(14)[10] == 1
        separators = [_counted(c) for _ in range(6)]
        styles = [_flagged(c)[1] for _ in range(8)]
        return cls(title=title, title_style=title_style,
                   replace_existing=replace_existing,
                   include_book_documents=include_book_documents,
                   include_section_headings=include_section_headings,
                   separators=separators, styles=styles)


@dataclass
class Prefs:
    """Preference values read from the INDD."""

    values: list = field(default_factory=list)
    #: Interface colours: element, ``Properties`` child, red, green,
    #: blue.
    colors: list = field(default_factory=list)
    #: The text defaults (chunk 0x23F, a text attribute list).
    text_defaults: Attrs | None = None
    #: Default anchored object settings (chunk 0x2800, as object styles
    #: have).
    anchor: AnchorSettings | None = None
    #: Page item defaults (class 0x6E07), a page item attribute list.
    item_defaults: Attrs | None = None
    #: The values of page item attributes that an item does not store
    #: (class 0x6E02, chunk 0x6E03).
    item_base: Attrs | None = None
    #: The entries of the two tables around that list: class, UID, UID.
    item_default_entries: list = field(default_factory=list)
    #: The swatches that chunk 0x6E06 of the page item defaults names:
    #: class and UID.
    item_default_swatches: list = field(default_factory=list)
    #: ``Properties`` children: element, name, value.
    props: list = field(default_factory=list)
    #: Records of the print settings, written in base64.
    print_records: list = field(default_factory=list)
    #: Footnote options (chunk 0x2820).
    footnotes: FootnoteOptions | None = None
    #: Endnote options (chunk 0x2261E).
    endnotes: EndnoteOptions | None = None
    #: Index options (chunk 0x13010).
    index_options: IndexOptions | None = None
    #: Default paragraph and character style UIDs (chunks 0x28D4 and
    #: 0x28D5, u32 at 8).
    default_styles: list = field(default_factory=lambda: [None, None])
    #: The contour of new placed graphics (chunk 0x373F, else
    #: ``Contour.DEFAULT``).
    contour: Contour | None = None
    #: Default graphic, text and grid object style UIDs (chunk 0x1B959).
    default_object_styles: list | None = None
    #: Layout and story grid defaults: element, grid settings and, for
    #: the story grid, ``CharacterCountSize``.
    grids: list = field(default_factory=list)
    #: The colour of baseline frame grids, for the preferences and every
    #: object style.
    baseline_frame_grid_color: tuple | None = None
    #: Column rule of the default text frame settings (chunk 0x22646,
    #: as frames): f64 width at 28, u32 swatch UID at 36.
    frame_column_rule: tuple | None = None
    #: Footnote settings of the default text frame settings (chunk
    #: 0x22608, as frames): u16 span at 2, f64 minimum spacing at 4,
    #: f64 space between at 12.
    frame_footnotes: tuple | None = None
    #: ``ViewPreference`` ``RulerOrigin`` (chunk 0x808).
    ruler_origin: str = RulerOrigin.SPREAD


class PrefsId:
    VIEW = 0x1202
    BASELINE_GRID = 0x55F
    DOCUMENT_GRID = 0x545
    BASELINE_GRID_SHOWN = 0x56A
    GUIDES_LOCKED = 0x569
    COLUMN_GUIDES_LOCKED = 0x56B
    SHOW_FRAME_EDGES = 0x593
    SHOW_RULERS = 0x803
    RULER_ORIGIN = 0x808
    SHOW_INVISIBLES = 0xCAC7
    COLOR_PROFILES = 0x7C04
    COLOR_INTENT = 0x7C08
    COLOR_POLICIES = 0x7C44
    WATERMARK = 0x16344
    TEXT = 0x280
    BASELINE_FRAME_GRID = 0x2834
    SMART_TEXT_REFLOW = 0x28BE
    MARGINS = 0x550
    COLUMNS = 0x555
    ANCHOR = 0x2800
    FOOTNOTE_OPTIONS = 0x2820
    ENDNOTE_OPTIONS = 0x2261E
    PRINT = 0xA4C
    PRINT_BOOKLET = 0xAF2
    BOOKLET_OPTIONS = 0xAF0
    PASTEBOARD = 0x5D2
    XML_TAGS = 0xBF4F
    GRIDS_IN_BACK = 0x567
    TEXT_WRAP = 0x3768
    ITEM_DEFAULTS = 0x6E07
    ITEM_DEFAULT_SWATCHES = 0x6E06
    PROFILE_NAME = 0x13C
    INDEX_OPTIONS = 0x13010
    INDEX_HEADER = 0x1300E
    CHAPTER_NUMBER = 0x1A4C4
    DICTIONARY = 0x2806
    EPUB = 0x21A1A
    HTML = 0x21A19
    EPUB_FIXED_LAYOUT = 0x21A25
    PUBLISH = 0x21A20
    ADJUST_LAYOUT = 0x7020
    DEFAULT_PARAGRAPH_STYLE = 0x28D4
    DEFAULT_CHARACTER_STYLE = 0x28D5
    DEFAULT_OBJECT_STYLES = 0x1B959
    LAYOUT_GRID = 0xCD2F
    STORY_GRID = 0xCD2E
    MASTER_TEXT_FRAME = 0x59C
    BLENDING_SPACE = 0x1081F
    ALLOW_PAGE_SHUFFLE = 0x5A6
    GUIDES_SHOWN = 0x568
    SNAP = 0x55A
    GUIDES = 0x53F
    ZERO_POINT = 0x54A
    LAYOUT_ADJUSTMENT = 0x7006
    SHOW_TEXT_THREADS = 0xCA0B
    DOCUMENT_GRID_SHOWN = 0x566
    STYLE_PREVIEW = 0xCAF3
    HIGHLIGHT_HJ_VIOLATIONS = 0xCA62
    HIGHLIGHT_CUSTOM_SPACING = 0xCAD3
    HIGHLIGHT_SUBSTITUTED_GLYPHS = 0xCAD4
    SNAP_TO_LAYOUT_GRID = 0xCD14
    SHOW_ALL_LAYOUT_GRIDS = 0xCD1C
    SHOW_ALL_FRAME_GRIDS = 0xCD1D
    SHOW_CHARACTER_COUNT = 0xCD1E
    MOJIKUMI_UI = 0x42DB
    ARTICLE_ORDER = 0x11C69
    XML_IMPORT = 0x1BC0B
    STORY = 0x2EE
    ITEM_TEXT_WRAP = 0x3720
    ITEM_CONTOUR = 0x373F


_UI_LIGHT_GRAY = (0.73, 0.73, 0.73)
_UI_MAGENTA = (1.0, 0.31, 1.0)
_UI_FIESTA = (0.97, 0.35, 0.42)
_UI_GRID_BLUE = (0.48, 0.73, 0.85)
_UI_LAVENDER = (0.6, 0.6, 1.0)
_UI_VIOLET = (0.6, 0.2, 1.0)


def _flagged(c: Cursor):
    """A flagged in-object string: a flag byte (1 for a built-in key)
    and the string."""
    flag = c.u8()
    return (flag, c.string())


def _chapter_format(code: int):
    return {0x1A477: "1, 2, 3, 4...", 0x1A47A: "A, B, C, D...",
            0x1A479: "i, ii, iii, iv...",
            0x1A473: "001,002,003..."}.get(code)


def _key_or_text(flagged_pair):
    """A built-in key or plain text: flag 1 gives ``$ID/`` and the key,
    flag 0 the text; other flags give nothing."""
    flag, text = flagged_pair
    if flag == 1:
        return builtin_key(text)
    if flag == 0:
        return text
    return None


def _xml_import_link(enc: Encoding, d: bytes):
    """``CreateLinkToXML`` from the XML import options (chunk 0x1BC0B):
    a u32 count of key and value pairs, each a u32 length in UTF-16
    units and text segments. No pairs is ``false``; key
    ``XMediaUI_CreateLink`` with value ``1`` is ``true``. ``None``
    where the pairs cannot be read or lack the key."""
    try:
        c = enc.cursor(d)
        n = c.u32()
        for _ in range(n):
            length = c.u32()
            key = c.segments(length)
            length = c.u32()
            value = c.segments(length)
            if key == "XMediaUI_CreateLink":
                return value == "1"
        return False if n == 0 else None
    except InddError:
        return None


_EPUB_METADATA = ["EpubTitle", "EpubCreator", "EpubDate",
                  "EpubDescription", "EpubRights", "EpubSubject"]
_EPUB_ACCESSIBILITY = [
    "EpubAccessibilityFeature", "EpubAccessibilityHazard",
    "EpubAccessibilityMode", "EpubAccessibilityModeSufficient",
    "EpubAccessibilitySummary", "EpubAccessibilityConformsTo",
    "EpubAccessibilityCertifiedBy", "EpubAccessibilityCredentials",
    "EpubAccessibilityReportLink"]
_EPUB_ID = "urn:uuid:29d919dd-24f5-4384-be78-b447c9dc299b"


def _style_name(flagged_pair) -> str:
    """A flagged string as IDML writes a style name: ``$ID/`` before
    the text of a built-in name (flag 1)."""
    flag, text = flagged_pair
    return f"$ID/{text}" if flag == 1 else text


def _resolution(ppi: int):
    return {72: "Ppi72", 150: "Ppi150"}.get(ppi)


def _epub_export(enc: Encoding, d: bytes):
    """Values of the EPUB export options (chunk 0x21A1A), or ``None``
    for a layout the samples do not show. Two blocks have lengths that
    differ between versions; the layout read is the one that ends
    exactly at the chunk's end."""
    def read(b2, b3):
        c = enc.cursor(d)
        out = []

        def code(name, v, codes):
            for key, text in codes:
                if key == v:
                    out.append((name, text))

        code("Version", c.u32(), [(0, "Epub2"), (1, "Epub3")])
        code("ExportOrder", c.u32(),
             [(1, "LayoutOrder"), (2, "ArticlePanelOrder")])
        code("EpubCover", c.u32(),
             [(0, "None"), (1, "FirstPage"), (2, "ExternalImage")])
        out.append(("CoverImageFile", _flagged(c)[1]))
        c.u16()
        toc = _flagged(c)
        out.append(("UseTocStyle", str(bool(toc[1])).lower()))
        out.append(("TocStyleName", _style_name(toc)))
        match c.u16():
            case 0:
                out.append(("BreakDocument", "false"))
            case 1:
                out.append(("BreakDocument", "true"))
        out.append(("ParagraphStyleName", _style_name(_flagged(c))))
        c.skip(46)
        out.append(("EpubPublisher", _flagged(c)[1]))
        out.append(("Id", _flagged(c)[1]))
        c.skip(b2 - 50)
        b = enc.cursor(c.bytes(50))
        match b.u16():
            case 0:
                out.extend([
                    ("PreserveLayoutAppearence", "false"),
                    ("FootnotePlacement", "FootnoteInsidePopup"),
                    ("UseExistingImageOnExport", "true"),
                    ("UseOriginalImageOnExport", "true")])
            case 1:
                out.extend([
                    ("PreserveLayoutAppearence", "true"),
                    ("FootnotePlacement", "FootnoteAfterStory"),
                    ("UseExistingImageOnExport", "false"),
                    ("UseOriginalImageOnExport", "false")])
        resolution = _resolution(b.u16())
        if resolution is not None:
            out.append(("ImageExportResolution", resolution))
        b.u16()
        match b.u16():
            case 0:
                out.append(("CustomImageSizeOption", "SizeFixed"))
            case 1:
                out.append(("CustomImageSizeOption",
                            "SizeRelativeToTextFlow"))
        if _flagged(c)[1] != "":
            raise _corrupt("EPUB export: string not empty")
        block = c.bytes(b3)
        font = enc.u16_at(block, 14 if b3 == 26 else 10)
        if font == 0:
            out.append(("EmbedFont", "false"))
        elif font == 1:
            out.append(("EmbedFont", "true"))
        if c.remaining() > 0:
            for name in _EPUB_METADATA:
                out.append((name, _flagged(c)[1]))
        if c.remaining() > 0:
            raise _corrupt("EPUB export: bytes left")
        return out

    for b2, b3 in [(50, 20), (50, 22), (50, 26),
                   (52, 20), (52, 22), (52, 26)]:
        try:
            return read(b2, b3)
        except InddError:
            continue
    return None


def _html_export(enc: Encoding, d: bytes):
    """Values of the HTML export options (chunk 0x21A19): u32 at 4, u16
    at 48 and 54, and after byte 98 a string, 6 bytes (10 from
    InDesign 17), two strings, a u16 and the u16
    ``PreserveLocalOverride``."""
    out = []
    match enc.u32_at(d, 4):
        case 1:
            out.append(("ExportOrder", "LayoutOrder"))
        case 2:
            out.append(("ExportOrder", "ArticlePanelOrder"))
    for offset, name in [(48, "ViewDocumentAfterExport"),
                         (54, "PreserveLayoutAppearence")]:
        match enc.u16_at(d, offset):
            case 0:
                out.append((name, "false"))
            case 1:
                out.append((name, "true"))

    def local(gap):
        c = enc.cursor(d)
        c.skip(98)
        _flagged(c)
        c.skip(gap)
        _flagged(c)
        _flagged(c)
        c.u16()
        return c.u16()

    try:
        value = local(6)
    except InddError:
        try:
            value = local(10)
        except InddError:
            value = None
    if value == 0:
        out.append(("PreserveLocalOverride", "false"))
    elif value == 1:
        out.append(("PreserveLocalOverride", "true"))
    return out


def _fixed_layout_export(enc: Encoding, d: bytes):
    """Values of the fixed layout EPUB export options (chunk 0x21A25),
    read to the chunk's end; ``None`` if the layout does not end
    there."""
    try:
        c = enc.cursor(d)
        out = []
        match c.u32():
            case 0:
                out.append(("EpubCover", "None"))
            case 1:
                out.append(("EpubCover", "FirstPage"))
            case 2:
                out.append(("EpubCover", "ExternalImage"))
        out.append(("CoverImageFile", _flagged(c)[1]))
        c.u16()
        out.append(("TocStyleName", _style_name(_flagged(c))))
        c.u16()
        out.append(("EpubPublisher", _flagged(c)[1]))
        out.append(("Id", _flagged(c)[1]))
        c.skip(16)
        _flagged(c)
        c.skip(12)
        for name in _EPUB_METADATA:
            out.append((name, _flagged(c)[1]))
        # The page range: empty in every sample.
        _flagged(c)
        tail = c.bytes(12)
        match enc.u32_at(tail, 8):
            case 0:
                out.append(("EpubNavigationStyles", "NoNavigation"))
            case 2:
                out.append(("EpubNavigationStyles", "TocStyleNavigation"))
            case 3:
                out.append(("EpubNavigationStyles", "BookmarksNavigation"))
        if c.remaining() > 0:
            raise _corrupt("fixed layout export: bytes left")
        return out
    except InddError:
        return None


def _publish_export(enc: Encoding, d: bytes):
    """Values of the Publish Online export options (chunk 0x21A20),
    read to the chunk's end; ``None`` if the layout does not end
    there."""
    try:
        c = enc.cursor(d)
        c.skip(8)
        out = []
        for name in ["PublishFileName", "PublishDescription",
                     "PublishPageRange"]:
            out.append((name, _flagged(c)[1]))
        c.skip(16)
        _flagged(c)
        c.skip(7)
        _flagged(c)
        c.skip(5)
        out.append(("CoverPage", _style_name(_flagged(c))))
        if c.remaining() > 0:
            raise _corrupt("publish export: bytes left")
        return out
    except InddError:
        return None


def _unit(code: int):
    """Measurement unit codes (``ViewPreference``)."""
    return {0x1201: "Points", 0x1202: "Picas", 0x1203: "Inches",
            0x1205: "Millimeters", 0x1206: "Centimeters",
            0x120B: "Pixels", 0x5101: "Ha",
            0x510F: "Q"}.get(code)


def _policy(code: bytes):
    """Colour management policy codes: four characters, stored
    reversed."""
    return {b" ffo": "ColorPolicyOff",
            b"serp": "PreserveEmbeddedProfiles",
            b"lidp": "CombinationOfPreserveAndSafeCmyk",
            b"vnoc": "ConvertToWorkingSpace"}.get(code)


_NOT_IN_BOOKLET = ["PrintSpreads", "Thumbnails", "Tile",
                   "TilingOverlap", "ThumbnailsPerPage",
                   "IncludeSlugToPrint"]


def _print_prefs(enc: Encoding, d: bytes, element: str, values: list,
                 props: list, records: list) -> None:
    """Print settings (chunk 0xA4C, and 0xAF2 for booklets). The two
    chunks have one layout: a head of fields and strings, then fixed
    blocks with strings between them, then the paper size selector."""
    booklet = element == "PrintBookletPrintPreference"

    def set_value(name, value):
        if not (booklet and name in _NOT_IN_BOOKLET):
            values.append(PrefValue(element=element, name=name,
                                    value=value))

    def prop(name, ty, text):
        props.append((element, name, ("text", ty, text)))

    c = enc.cursor(d)
    has_record = c.u8() != 0
    c.u8()
    n = c.u32()
    records.append((element, "PrintRecord", c.bytes(n)))
    if has_record:
        c.u16()
        device = c.u32()
    else:
        c.u16()
        device = 0
    set_value("DeviceType", str(device))
    preset = _flagged(c)
    to = c.u32()
    printer = _flagged(c)
    _flagged(c)
    ppd = _flagged(c)
    ppd_file = _flagged(c)
    if preset == (1, "kPrSt_DefaultName"):
        prop("ActivePrinterPreset", "enumeration", "Default")
    elif preset == (1, ""):
        prop("ActivePrinterPreset", "enumeration", "Custom")
    elif preset[0] != 1:
        prop("ActivePrinterPreset", "string", preset[1])
    prepress = printer == (1, "kPrepress File")
    if printer == (1, "kPrepress File"):
        prop("Printer", "enumeration", "PostscriptFile")
    elif printer == (1, ""):
        prop("Printer", "string", "$ID/")
    elif printer[0] != 1:
        prop("Printer", "string", printer[1])
    if ppd == (1, "kDevice Independent") and prepress:
        prop("PPD", "enumeration", "DeviceIndependent")
    elif ppd == (1, "kDevice Independent"):
        prop("PPD", "string", builtin_key("kDevice Independent"))
    elif ppd == (1, ""):
        prop("PPD", "string", "$ID/")
    elif ppd[0] != 1:
        prop("PPD", "string", ppd[1])
    set_value("PrintTo", str(to))
    set_value("PrintToDisk", str(to == 2).lower())
    file = builtin_key(ppd_file[1]) if ppd_file[0] == 1 else ppd_file[1]
    set_value("PPDFile", file)
    match c.u32():
        case 2:
            set_value("PostScriptLevel", "Level2")
        case 3:
            set_value("PostScriptLevel", "Level3")
    set_value("PrintResolution", idml_num(c.f64()))

    def rect(name):
        attrs = []
        for side in ["Left", "Top", "Right", "Bottom"]:
            attrs.append((side, num_half_even(c.f64())))
        props.append((element, name, ("attrs", attrs)))

    rect("PaperSizeRect")
    rect("ImageablePaperSizeRect")
    size = c.i32()
    name = _flagged(c)
    paper = None
    if size == -3:
        paper = ("string", name[1])
    elif size == -2:
        paper = ("enumeration", "DefinedByDriver")
    elif size == -1:
        paper = ("enumeration", "Custom")
    if paper is not None:
        props.append((element, "PaperSize",
                      ("text", paper[0], paper[1])))
    enums = []

    def f(block, offset):
        return idml_num(enc.cursor(block[offset:]).f64())

    def flag(v):
        return {0: "false", 1: "true"}.get(v)

    def set_flag(name, v):
        value = flag(v)
        if value is not None:
            set_value(name, value)

    # Block A: paper ranges, orientation, copies.
    a = c.bytes(126)
    set_value("PaperWidthRange", f"{f(a, 8)} {f(a, 16)}")
    set_value("PaperHeightRange", f"{f(a, 32)} {f(a, 40)}")
    set_value("PaperOffsetRange", f"{f(a, 56)} {f(a, 64)}")
    match enc.cursor(a[84:]).u16():
        case 0:
            set_value("PrintPageOrientation", "Portrait")
        case 1:
            set_value("PrintPageOrientation", "Landscape")
    set_value("Copies", str(enc.cursor(a[104:]).u32()))
    set_flag("PrintBlankPages", a[118])
    # The page range: 0 at A+122 is `AllPages`; with 2 IDML has none.
    if not booklet and enc.cursor(a[122:]).u16() == 0:
        props.append((element, "PageRange",
                      ("text", "enumeration", "AllPages")))
    _flagged(c)

    # Block B: spreads, colour output, composite screening.
    b = c.bytes(30)
    set_flag("PrintSpreads", b[4])
    output = {0: "CompositeGray", 1: "CompositeRGB",
              2: "CompositeCMYK",
              5: "CompositeLeaveUnchanged"}.get(b[6])
    if output is not None:
        set_value("ColorOutput", output)
        set_value("PreserveColorNumbers",
                  str(output != "CompositeRGB").lower())
    set_flag("TextAsBlack", b[10])
    set_value("CompositeAngle", f(b, 14))
    set_value("CompositeFrequency", f(b, 22))
    value = _key_or_text(_flagged(c))
    if value is not None:
        set_value("CompositeScreening", value)
    value = _key_or_text(_flagged(c))
    if value is not None:
        set_value("SeparationScreening", value)

    # Block C: scaling and page position.
    s = c.bytes(25)
    if s[0] == 0:
        enums.append(("ScaleMode", "ScaleToFit"))
    elif s[0] == 1:
        enums.append(("ScaleMode", "ScaleWidthHeight"))
    set_flag("ScaleProportional", s[4])
    set_value("ScaleWidth", f(s, 6))
    set_value("ScaleHeight", f(s, 14))
    if s[22] == 0:
        enums.append(("PagePosition", "UpperLeft"))
    elif s[22] == 3:
        enums.append(("PagePosition", "Centered"))
    _flagged(c)

    # Block D: tiling, thumbnails, graphics and fonts.
    g = c.bytes(48)
    if g[0] <= 2:
        set_value("Tile", str(g[0] == 1).lower())
        set_value("Thumbnails", str(g[0] == 2).lower())
    set_value("TilingOverlap", f(g, 8))
    if g[16] == 2:
        enums.append(("ThumbnailsPerPage", "K1x2"))
    elif g[16] == 4:
        enums.append(("ThumbnailsPerPage", "K2x2"))
    elif g[16] == 9:
        enums.append(("ThumbnailsPerPage", "K3x3"))
    if g[20] == 0:
        enums.append(("SendImageData", "AllImageData"))
    elif g[20] == 1:
        enums.append(("SendImageData", "OptimizedSubsampling"))
    if g[24] == 0:
        enums.append(("DataFormat", "Binary"))
    elif g[24] == 1:
        enums.append(("DataFormat", "ASCII"))
    set_value("BitmapResolution",
              str(enc.cursor(g[30:]).u16()))
    if g[42] == 0:
        enums.append(("FontDownloading", "None"))
    elif g[42] == 1:
        enums.append(("FontDownloading", "Complete"))
    elif g[42] == 2:
        enums.append(("FontDownloading", "Subset"))
    set_flag("DownloadPPDFonts", g[46])
    mark_flagged = _flagged(c)
    if mark_flagged == (1, ""):
        mark_type = "Default"
    elif mark_flagged == (1, "kJMarksWithCircle"):
        mark_type = "JMarkWithCircle"
    else:
        mark_type = None

    # Block E: printer marks, bleed, slug, colour profile.
    e = c.bytes(72)
    if e[0] == 1:
        enums.append(("MarkLineWeight", "P25pt"))
    elif e[0] == 2:
        enums.append(("MarkLineWeight", "P50pt"))
    elif e[0] == 4:
        enums.append(("MarkLineWeight", "P07mm"))
    elif e[0] == 5:
        enums.append(("MarkLineWeight", "P10mm"))
    set_value("MarkOffset", f(e, 4))
    marks = [("CropMarks", e[12]), ("PageInformationMarks", e[14]),
             ("ColorBars", e[16]), ("RegistrationMarks", e[18]),
             ("BleedMarks", e[20])]
    for mark_name, v in marks:
        set_flag(mark_name, v)
    if all(v <= 1 for _, v in marks):
        set_value("AllPrinterMarks",
                  str(all(v == 1 for _, v in marks)).lower())
    set_flag("UseDocumentBleedToPrint", e[22])
    set_value("BleedTop", f(e, 24))
    set_flag("BleedChain", e[32])
    set_value("BleedInside", f(e, 34))
    set_value("BleedBottom", f(e, 42))
    set_value("BleedOutside", f(e, 50))
    set_flag("IncludeSlugToPrint", e[58])
    profile = {0: "PostScriptCMS", 1: "UseDocument"}.get(e[68])
    _flagged(c)
    c.skip(4)
    _flagged(c)

    # Block G: screen frequencies and angles of the inks.
    k = c.bytes(98)
    for offset, ink_name in [(12, "CyanFrequency"), (20, "CyanAngle"),
                             (30, "MagentaFrequency"),
                             (38, "MagentaAngle"),
                             (48, "YellowFrequency"),
                             (56, "YellowAngle"),
                             (66, "BlackFrequency"),
                             (74, "BlackAngle"),
                             (82, "SpotFrequency"), (90, "SpotAngle")]:
        set_value(ink_name, f(k, offset))
    preset2 = _flagged(c)
    if preset2[0] == 1:
        flattener = builtin_key(preset2[1])
    elif preset2[0] == 3 and preset2[1] in ("[High Resolution]",
                                            "[Vysoké rozlišení]"):
        flattener = builtin_key("kFlSt_HighDefaultName")
    else:
        flattener = None
    if flattener is not None:
        set_value("FlattenerPresetName", flattener)

    # The paper size selector.
    c.skip(6)
    n = c.u32()
    records.append((element, "PaperSizeSelector", c.bytes(n)))

    for enum_name, v in enums:
        set_value(enum_name, v)
    for prop_name, v in [("MarkType", mark_type), ("Profile", profile)]:
        if v is not None:
            props.append((element, prop_name,
                          ("text", "enumeration", v)))


def _index_header(enc: Encoding, d: bytes):
    """Index header setting (chunk 0x1300E): the header set name,
    language and handler, then groups of sections."""
    element = "IndexHeaderSetting"

    def key(flagged_pair):
        flag, text = flagged_pair
        return builtin_key(text) if flag == 1 else text

    c = enc.cursor(d)
    values = []

    def set_value(name, value):
        values.append(PrefValue(element=element, name=name, value=value))

    set_value("HeaderSetName", key(_flagged(c)))
    set_value("HeaderSetLanguage", str(c.u16()))
    set_value("IndexHeaderSetHandler", str(c.u32()))
    set_value("IndexHeaderSetGroupValue", str(c.u32()))
    set_value("IndexHeaderSetGroupOptionValue", str(c.u32()))
    c.u16()
    groups = c.u32()
    nodes = []
    for _ in range(groups):
        internal = key(_flagged(c))
        ui = key(_flagged(c))
        document = builtin_key(_counted(c))
        visible = c.u16() == 1
        n = c.u32()
        sections = []
        for _ in range(n):
            sorting = builtin_key(_counted(c))
            header = builtin_key(_counted(c))
            ui_text = key(_flagged(c))
            language = c.u16()
            sections.append(PrefNode(
                tag="SectionHeaderType",
                attrs=[("SortingHeaderString", sorting),
                       ("DocumentHeaderString", header),
                       ("UIHeaderString", ui_text),
                       ("Language", str(language))]))
        nodes.append(PrefNode(
            tag="IndexHeaderGroupType",
            attrs=[("InternalName", internal), ("UIString", ui),
                   ("DocumentString", document),
                   ("Visibility", str(visible).lower())],
            children=[PrefNode(tag="SectionHeaderArray",
                               children=sections)]))
    if c.remaining() != 0:
        raise _corrupt(f"index header setting: {c.remaining()} bytes left")
    return (values, nodes)

# ---------------------------------------------------------------------------
# Document + Reader (src/model/document.rs, src/model/reader.rs)
# ---------------------------------------------------------------------------

#: The UID of the document object.
DOC = 1


@dataclass
class XmlStructure:
    """The XML structure of the document."""

    #: The backing story, IDML ``XmlStory``.
    story: Story | None = None
    elements: dict = field(default_factory=dict)


@dataclass
class Article:
    """An article (class 0x21B01): its name (chunk 0x13C) and the UIDs
    of its member page items (chunk 0x21B02), in order."""

    uid: int
    name: str = ""
    members: list = field(default_factory=list)


@dataclass
class ClassObjects:
    """The objects read from the class list, before they are put
    together into the ``Document``."""

    styles: dict = field(default_factory=dict)
    colors: list = field(default_factory=list)
    mixed_inks: list = field(default_factory=list)
    tints: list = field(default_factory=list)
    gradients: list = field(default_factory=list)
    swatches: dict = field(default_factory=dict)
    fonts: dict = field(default_factory=dict)
    languages: dict = field(default_factory=dict)
    language_list: list = field(default_factory=list)
    style_groups: dict = field(default_factory=dict)
    object_styles: dict = field(default_factory=dict)
    cell_styles: dict = field(default_factory=dict)
    table_styles: dict = field(default_factory=dict)
    text_variables: list = field(default_factory=list)
    hyperlinks: list = field(default_factory=list)
    text_sources: dict = field(default_factory=dict)
    page_item_sources: list = field(default_factory=list)
    destinations: list = field(default_factory=list)
    bookmarks: dict = field(default_factory=dict)
    cross_reference_formats: dict = field(default_factory=dict)
    #: Composite fonts, with their entry UIDs still unresolved.
    composite_fonts: list = field(default_factory=list)
    composite_entries: dict = field(default_factory=dict)
    cjk_tables: list = field(default_factory=list)
    xml_tags: list = field(default_factory=list)
    xml_tag_names: dict = field(default_factory=dict)
    inks: list = field(default_factory=list)
    color_groups: dict = field(default_factory=dict)


@dataclass
class Document:
    """The document model: everything the converter knows, read from
    the document object and the class list."""

    version: Version = field(default_factory=lambda: Version(0, 0))
    layers: list = field(default_factory=list)
    active_layer: int | None = None
    #: Document users (chunk 0xA443): flag byte and name.
    users: list = field(default_factory=list)
    #: The script byte of the last document user's name.
    users_script: int | None = None
    #: The code of the last session in the save history (chunk 0x1D8):
    #: 0x0101 for a session of a Japanese or Chinese edition.
    last_session_code: int | None = None
    #: Major and minor version of the application of the last session
    #: in the save history (``16.1.0.20`` gives 16, 1).
    last_session_version: tuple | None = None
    #: The third number of that version (``16.1.0.20`` gives 0).
    last_session_patch: int | None = None
    #: The fourth number of that version, the build (``16.1.0.20``
    #: gives 20).
    last_session_build: int | None = None
    #: The platform of the last session: 0 or 3.
    last_session_platform: int | None = None
    #: The document's label (chunk 0x1630B): key and value, as IDML
    #: writes them.
    label: list = field(default_factory=list)
    #: The numbering lists, the default list first.
    numbering_lists: list = field(default_factory=list)
    #: Index sort groups (preferences chunk 0x1307E), in order: name,
    #: include flag and the name of the header variant.
    index_groups: list = field(default_factory=list)
    #: The index (class 0x13004) and its topics.
    index: Index | None = None
    #: The constant shade the document lists (the class 0x5533 object
    #: of lowest UID with a constant shade in chunk 0x5532).
    constant_shade: ConstantShade | None = None
    #: The pasted smooth shades with an axial shading, in UID order.
    axial_shades: list = field(default_factory=list)
    #: Assignment objects (class 0x1BE01), in UID order.
    assignments: list = field(default_factory=list)
    #: The name of the first assignment, in the language of the edition
    #: that made the document.
    assignment_name: str | None = None
    #: Articles, in the order of document chunk 0x21B03.
    articles: list = field(default_factory=list)
    #: Named grids (class 0xCD12), in UID order.
    named_grids: list = field(default_factory=list)
    #: Motion presets, in the order of chunk 0x2053E of the preferences.
    motion_presets: list = field(default_factory=list)
    spreads: list = field(default_factory=list)
    master_spreads: list = field(default_factory=list)
    stories: list = field(default_factory=list)
    styles: dict = field(default_factory=dict)
    toc_styles: list = field(default_factory=list)
    #: Custom stroke styles, in UID order.
    stroke_styles: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    #: Mixed inks and mixed ink groups, in UID order.
    mixed_inks: list = field(default_factory=list)
    #: Tint swatches, with their IDML reference and name.
    tints: list = field(default_factory=list)
    gradients: list = field(default_factory=list)
    #: IDML reference (``Color/...``, ``Swatch/None``) for each swatch
    #: UID.
    swatches: dict = field(default_factory=dict)
    #: Font families by UID.
    fonts: dict = field(default_factory=dict)
    #: The language objects (class 0x2D07) in UID order.
    language_list: list = field(default_factory=list)
    #: Language name (IDML ``AppliedLanguage`` without ``$ID/``) for
    #: each language UID.
    languages: dict = field(default_factory=dict)
    #: Style groups, including the root groups (empty name).
    style_groups: dict = field(default_factory=dict)
    object_styles: dict = field(default_factory=dict)
    cell_styles: dict = field(default_factory=dict)
    table_styles: dict = field(default_factory=dict)
    sections: list = field(default_factory=list)
    text_variables: list = field(default_factory=list)
    hyperlinks: list = field(default_factory=list)
    text_sources: dict = field(default_factory=dict)
    page_item_sources: list = field(default_factory=list)
    destinations: list = field(default_factory=list)
    bookmarks: dict = field(default_factory=dict)
    #: All bookmarks in document order (document chunk 0x13501).
    bookmark_order: list = field(default_factory=list)
    #: Cross-reference formats, by UID.
    cross_reference_formats: dict = field(default_factory=dict)
    #: Problems that did not stop the conversion (content left out).
    warnings: list = field(default_factory=list)
    preferences: DocumentPreferences | None = None
    #: Other preference values (``prefs.rs``).
    prefs: Prefs = field(default_factory=Prefs)
    composite_fonts: list = field(default_factory=list)
    #: Kinsoku and mojikumi tables, in UID order.
    cjk_tables: list = field(default_factory=list)
    #: Inks, in UID order.
    inks: list = field(default_factory=list)
    #: Colour groups in document order, the root group first.
    color_groups: list = field(default_factory=list)
    #: The bullet characters offered for lists (preferences chunk
    #: 0x1A488).
    bullets: list = field(default_factory=list)
    #: XML tags (class 0xBF19).
    xml_tags: list = field(default_factory=list)
    xml: XmlStructure = field(default_factory=XmlStructure)
    #: The dates of the XMP packet that have a UTC offset; link times
    #: are written with their offsets.
    xmp_dates: list = field(default_factory=list)
    #: The black swatch's name in the XMP packet: the language of the
    #: edition that saved the document.
    black_name: str | None = None
    #: ``LinkResourceId`` by story or text frame UID (document chunk
    #: 0x21D06).
    link_source_ids: dict = field(default_factory=dict)
    #: Document style mappings (chunk 0x21D30 of the preferences).
    style_mappings: StyleMappings = field(default_factory=StyleMappings)

    def read_xmp(self, packet: bytes) -> None:
        """Read the dates and the black name of the XMP packet. IDML
        names a tint of black with the black name (``objects.md``,
        saving edition).

        Upstream matches the black colourant against the black swatch's
        space and values; this port reads the black colourant of the
        packet directly (``xmp_black_name``), which names the same
        swatch whenever the packet carries a black colourant.
        """
        self.xmp_dates = xmp_dates(packet)
        black = next((c for c in self.colors
                      if c.reference() == "Color/Black"), None)
        if black is None:
            return
        name = xmp_black_name(packet)
        self.black_name = name
        if name is None or black.color_override != 2:
            return
        for tint, _reference, _idml in self.tints:
            if tint.base == black.uid:
                self.swatches[tint.uid] = Tint.reference_of(
                    tint.named(f"[{name}]"))
        self.tints = [(t, t.reference(b), t.idml_name(b))
                      if t.base == black.uid
                      and (b := next((c for c in self.colors
                                      if c.uid == t.base), None)) is not None
                      else (t, r, n)
                      for (t, r, n) in self.tints]


def _anchored_stories(story: Story) -> list:
    """The stories of the text frames anchored in ``story``, in text
    order (nested items included)."""

    out = []

    def walk(items, out):
        for item in items:
            if item.kind[0] == "text_frame" and item.kind[1].get("story"):
                out.append(item.kind[1]["story"])
            walk(item.children, out)

    for items in story.anchors.values():
        walk(items, out)
    return out


def _setup_layout_of(length: int):
    """Where the fields of the document setup chunk (0x533) are: the
    146-byte layout (or longer) and the 126-byte layout of InDesign 7.0
    and 7.5, which has 16 bytes before the page size and its fields
    from the facing pages flag on 20 bytes earlier. Returns the
    offsets (page, facing, binding, bleed, bleed_uniform, slug,
    slug_uniform, intent), or ``None`` for an unknown length."""
    if length == 126:
        page, shift = 16, 20
    elif length >= 146:
        page, shift = 0, 0
    else:
        return None
    return (page, 58 - shift, 64 - shift, 70 - shift, 102 - shift,
            104 - shift, 136 - shift, 142 - shift)


class Reader:
    """Reads typed objects from a database, caching them. An object
    that cannot be read is left out with a warning; the warnings are
    in ``Document.warnings``."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self._cache: dict = {}
        self.warnings: list = []
        #: XML nodes found in the stories read so far.
        self.xml_nodes: dict = {}
        #: Text orientation of each text frame read so far, by story.
        self.frame_orientations: dict = {}
        #: The groups whose page items are being read, outermost first.
        self.item_path: list = []
        #: The major version of the document, set when it is read.
        self.major: int = 0
        #: InDesign 5.0 text sources by their start marker: (source,
        #: end marker). Built on first use.
        self._source_markers = None

    def object(self, uid: int):
        """Object ``uid`` with its class, cached; raises if it has no
        data."""
        if uid in self._cache:
            return self._cache[uid]
        obj = self.db.get(uid)
        if obj is None:
            raise _corrupt(f"object {uid} has no data")
        self._cache[uid] = obj
        return obj

    def enc(self) -> Encoding:
        """The encoding of the document's object data."""
        return self.db.encoding

    def cursor(self, data: bytes) -> Cursor:
        """A cursor over object data (normally a chunk) of this
        document."""
        return self.db.encoding.cursor(data)

    def attrs_or_warn(self, what: str, fn):
        """The attribute list ``fn()`` returns, or ``None`` with a
        warning that names the list's owner (``what``) if it cannot be
        parsed."""
        try:
            return fn()
        except InddError as exc:
            self.warn(f"{what}: attribute list left out: {exc}")
            return None

    def warn(self, msg: str) -> None:
        """Record a problem that does not stop the conversion."""
        self.warnings.append(msg)

    def class_of(self, uid: int):
        return self.db.class_of(uid)

    def chunk(self, uid: int, chunk_id: int):
        """Data of chunk ``chunk_id`` of object ``uid``, or ``None``."""
        return self.object(uid).chunk(chunk_id)

    def required(self, uid: int, chunk_id: int) -> bytes:
        data = self.chunk(uid, chunk_id)
        if data is None:
            raise _corrupt(f"object {uid} has no chunk {chunk_id:#x}")
        return data

    def uid_list(self, uid: int, chunk_id: int) -> list:
        data = self.chunk(uid, chunk_id)
        if data is None:
            return []
        return self.cursor(data).u32_list()

    def uid_list_or_none(self, uid: int, chunk_id: int):
        """The UID list of chunk ``chunk_id``, or ``None`` without the
        chunk."""
        data = self.chunk(uid, chunk_id)
        if data is None:
            return None
        return self.cursor(data).u32_list()

    def children(self, uid: int, chunk_id: int) -> list:
        """Children listed in a hierarchy chunk: parent, owner, then a
        u32 list."""
        data = self.chunk(uid, chunk_id)
        if data is None:
            return []
        c = self.cursor(data)
        c.skip(8)
        return c.u32_list()

    def document(self, version: Version) -> Document:
        """Read the document object and every class the converter
        knows. An object that cannot be read is left out with a
        warning; the warnings are in ``Document.warnings``."""
        self.major = version.major
        # The first layer of the list is the internal layer, whatever
        # its name (objects.md, layers).
        layers = [self.layer(uid, i == 0)
                  for i, uid in enumerate(
                      self.uid_list(DOC, Chunk.DOC_LAYERS))]
        active_layer = None
        data = self.chunk(DOC, Chunk.DOC_ACTIVE_LAYER)
        if data is not None:
            active_layer = self.cursor(data).u32()
        spreads, master_spreads = self.document_spreads()
        stories = self.document_stories()
        xml_story = self.xml_story()
        # Stories of text frames anchored in the XML backing story are
        # not in the document's list; IDML writes them after the
        # listed ones (objects.md, stories).
        if xml_story is not None:
            known = {s.uid for s in stories}
            known.add(xml_story.uid)
            nxt = list(reversed(_anchored_stories(xml_story)))
            while nxt:
                uid = nxt.pop()
                if uid in known or self.class_of(uid) != Class.STORY:
                    continue
                known.add(uid)
                try:
                    story = self.story(uid)
                except InddError as exc:
                    self.warn(f"story {uid} left out: {exc}")
                    continue
                nxt.extend(reversed(_anchored_stories(story)))
                stories.append(story)
        objects = self.class_objects()
        # Cell spans depend on insets that cells inherit from styles.
        table_styles = TableStyles(objects.cell_styles,
                                   objects.table_styles)
        for story in stories:
            for table in story.tables.values():
                table.resolve_spans(table_styles)
        for story in stories:
            story.orientation = self.story_orientation(story.uid)
        xml = XmlStructure(
            story=xml_story,
            elements=self.xml_elements(objects.xml_tag_names))
        tints = self.tints(objects)
        prune_style_groups(objects.style_groups, self.warn)
        preferences = self.document_preferences()
        prefs = self.prefs(version)
        toc_styles = self.toc_styles(version)
        stroke_styles = self.stroke_styles()
        named_grids = self.named_grids()
        motion_presets = self.motion_presets()
        index_groups = self.index_groups()
        try:
            index = self.index()
        except InddError as exc:
            self.warn(f"index left out: {exc}")
            index = None
        constant_shade = self.constant_shade()
        axial_shades = self.axial_shades()
        # Page item fills refer to axial shades by UID (objects.md,
        # pasted smooth shades).
        for shade in axial_shades:
            objects.swatches[shade.uid] = \
                f"PastedSmoothShade/u{shade.uid:x}"
        assignments = self.assignments()
        # The name of the first assignment: a flag byte and a string at
        # the start of chunk 0x1BE1B (objects.md, assignments).
        assignment_name = None
        if assignments:
            data = None
            try:
                data = self.chunk(assignments[0], Chunk.ASSIGNMENT_NAME)
            except InddError:
                data = None
            if data is not None:
                c = self.cursor(data)
                try:
                    c.flag()
                    assignment_name = c.string()
                except InddError:
                    assignment_name = None
        articles = self.articles()
        users_script = self.users_script(DOC)
        last_session = self.last_session(DOC)
        last_session_platform = self.last_session_platform(DOC)
        last_session_code = (last_session[0]
                             if last_session is not None else None)

        def session_number(i):
            if last_session is None:
                return None
            parts = last_session[1].split(".")
            if i >= len(parts):
                return None
            try:
                return int(parts[i])
            except ValueError:
                return None

        last_session_version = None
        first = session_number(0)
        second = session_number(1)
        if first is not None and second is not None:
            last_session_version = (first, second)
        last_session_patch = session_number(2)
        last_session_build = session_number(3)
        try:
            numbering_lists = self.numbering_lists()
        except InddError as exc:
            self.warn(f"numbering lists left out: {exc}")
            numbering_lists = []
        try:
            label = self.label(DOC)
        except InddError as exc:
            self.warn(f"document label left out: {exc}")
            label = []
        try:
            users = self.users(DOC)
        except InddError as exc:
            self.warn(f"document users left out: {exc}")
            users = []
        sections = [self.section(uid) for uid in
                    self.uid_list(DOC, Chunk.DOC_SECTIONS)]
        bookmark_order = []
        data = None
        try:
            data = self.chunk(DOC, HyperlinkChunk.DOCUMENT_LISTS)
        except InddError:
            data = None
        if data is not None:
            bookmark_order = document_bookmarks(self.enc(), data,
                                                version.major)
        for font in objects.composite_fonts:
            font.entries = [objects.composite_entries[e]
                            for e in font.entries
                            if e in objects.composite_entries]
        color_groups = [objects.color_groups[u]
                        for u in self.color_group_order()
                        if u in objects.color_groups]
        bullets = self.bullets()
        try:
            link_source_ids = self.link_source_ids()
        except InddError as exc:
            self.warn(f"link source IDs left out: {exc}")
            link_source_ids = {}
        style_mappings = self.document_style_mappings()
        return Document(
            version=version,
            layers=layers,
            active_layer=active_layer,
            spreads=spreads,
            master_spreads=master_spreads,
            stories=stories,
            styles=objects.styles,
            colors=objects.colors,
            mixed_inks=objects.mixed_inks,
            tints=tints,
            gradients=objects.gradients,
            swatches=objects.swatches,
            fonts=objects.fonts,
            languages=objects.languages,
            language_list=objects.language_list,
            toc_styles=toc_styles,
            stroke_styles=stroke_styles,
            named_grids=named_grids,
            motion_presets=motion_presets,
            index_groups=index_groups,
            index=index,
            constant_shade=constant_shade,
            axial_shades=axial_shades,
            assignments=assignments,
            assignment_name=assignment_name,
            articles=articles,
            style_groups=objects.style_groups,
            object_styles=objects.object_styles,
            cell_styles=objects.cell_styles,
            table_styles=objects.table_styles,
            users=users,
            users_script=users_script,
            last_session_code=last_session_code,
            last_session_version=last_session_version,
            last_session_patch=last_session_patch,
            last_session_build=last_session_build,
            last_session_platform=last_session_platform,
            label=label,
            numbering_lists=numbering_lists,
            sections=sections,
            text_variables=objects.text_variables,
            hyperlinks=objects.hyperlinks,
            text_sources=objects.text_sources,
            page_item_sources=objects.page_item_sources,
            destinations=objects.destinations,
            bookmarks=objects.bookmarks,
            bookmark_order=bookmark_order,
            cross_reference_formats=objects.cross_reference_formats,
            preferences=preferences,
            prefs=prefs,
            composite_fonts=objects.composite_fonts,
            cjk_tables=objects.cjk_tables,
            inks=objects.inks,
            color_groups=color_groups,
            bullets=bullets,
            xml_tags=objects.xml_tags,
            xml=xml,
            xmp_dates=[],
            black_name=None,
            link_source_ids=link_source_ids,
            style_mappings=style_mappings,
            # Last, so that it holds every warning of the model.
            warnings=list(self.warnings),
        )

    def document_spreads(self):
        """The spreads and master spreads of the document, with the
        page layout of each page resolved. A spread that cannot be
        read is left out with a warning."""
        spreads = self.each_or_warn(
            "spread", Chunk.DOC_SPREADS, lambda u: self.spread(u))
        master_spreads = self.each_or_warn(
            "master spread", Chunk.DOC_MASTER_SPREADS,
            lambda u: self.spread(u))
        resolve_page_layout(spreads, master_spreads)
        return spreads, master_spreads

    def document_stories(self):
        """The stories the document lists. A story that cannot be read
        is left out with a warning."""
        def read(uid):
            cls = self.class_of(uid)
            if cls in (Class.STORY, Class.ENDNOTE_STORY):
                return self.story(uid)
            return None

        return self.each_or_warn("story", Chunk.DOC_STORIES, read)

    def each_or_warn(self, what, list_id, read):
        """Read each object of the document's UID list ``list_id``
        with ``read``. An object that cannot be read is left out with
        a warning that calls it ``what``."""
        out = []
        for uid in self.uid_list(DOC, list_id):
            try:
                item = read(uid)
            except InddError as exc:
                self.warn(f"{what} {uid} left out: {exc}")
                continue
            if item is not None:
                out.append(item)
        return out

    def xml_story(self):
        """The backing story of the XML structure: the document's
        chunk 0xBF14 names it and the document node."""
        data = self.chunk(DOC, XmlChunk.NODE_REF)
        if data is None or len(data) < 4:
            return None
        story_uid = self.cursor(data).u32()
        if self.class_of(story_uid) != Class.STORY:
            return None
        try:
            return self.story(story_uid)
        except InddError as exc:
            self.warn(f"XML story {story_uid} left out: {exc}")
            return None

    def assignments(self):
        """Assignment objects (class 0x1BE01), in UID order."""
        return sorted(uid for uid, cls in self.db.classes
                      if cls == Class.ASSIGNMENT)

    def articles(self):
        """The articles listed in document chunk 0x21B03, each once.
        A listed object that is not an article is skipped; an article
        without a readable name or member list is left out with a
        warning."""
        try:
            order = self.uid_list(DOC, Chunk.DOC_ARTICLES)
        except InddError as exc:
            self.warn(f"articles left out: {exc}")
            return []
        seen = set()
        out = []
        for uid in order:
            if self.class_of(uid) != Class.ARTICLE or uid in seen:
                continue
            seen.add(uid)
            try:
                article = self.article(uid)
            except InddError as exc:
                self.warn(f"article {uid} left out: {exc}")
                continue
            if article is None:
                self.warn(f"article {uid} left out: no name")
            else:
                out.append(article)
        return out

    def article(self, uid):
        obj = self.object(uid)
        data = obj.chunk(Chunk.ARTICLE_NAME)
        if data is None:
            return None
        c = self.cursor(data)
        c.flag()
        name = c.string()
        members = []
        data = obj.chunk(Chunk.ARTICLE_MEMBERS)
        if data is not None:
            members = self.cursor(data).u32_list()
        return Article(uid=uid, name=name, members=members)

    def class_objects(self) -> ClassObjects:
        """Read every object of the class list whose class the
        converter knows. An object that cannot be read is left out
        with a warning."""
        out = ClassObjects()
        for uid, cls in self.db.classes:
            if self.db.object(uid) is None:
                continue
            try:
                self.class_object(out, uid, cls)
            except InddError as exc:
                self.warn(f"object {uid} (class {cls:#x}) left out: {exc}")
        return out

    def class_object(self, out, uid, cls):
        """Read one object of the class list into ``out``."""
        if cls in (Class.STYLE_ROOT_GROUP, Class.OBJECT_STYLE_ROOT_GROUP,
                   Class.CELL_STYLE_ROOT_GROUP,
                   Class.TABLE_STYLE_ROOT_GROUP):
            out.style_groups[uid] = self.style_root_group(uid)
        elif cls == Class.STYLE_GROUP:
            out.style_groups[uid] = self.style_group(uid)
        elif cls == Class.OBJECT_STYLE:
            style = self.object_style(uid)
            if style is not None:
                out.object_styles[uid] = style
        elif cls == Class.STYLE:
            style = self.style(uid)
            if style is not None:
                out.styles[uid] = style
        elif cls == TableClass.CELL_STYLE:
            style = self.table_style(uid, TableChunk.CELL_STYLE_ATTRS)
            if style is not None:
                out.cell_styles[uid] = style
        elif cls == TableClass.TABLE_STYLE:
            style = self.table_style(uid, TableChunk.TABLE_STYLE_ATTRS)
            if style is not None:
                out.table_styles[uid] = style
        elif cls == ColorClass.COLOR:
            self.color_object(out, uid)
        elif cls == ColorClass.SWATCH_NONE:
            out.swatches[uid] = "Swatch/None"
        elif cls == ColorClass.GRADIENT:
            gradient = self.gradient(uid)
            if gradient is not None:
                out.swatches[uid] = gradient.reference()
                out.gradients.append(gradient)
        elif cls == Class.FONT_FAMILY:
            font = self.font_family(uid)
            if font is not None:
                out.fonts[uid] = font
        elif cls == Class.TEXT_VARIABLE:
            variable = TextVariable.read(uid, self.object(uid))
            if variable is not None:
                out.text_variables.append(variable)
        elif cls == HyperlinkClass.HYPERLINK:
            hyperlink = Hyperlink.read(uid, self.object(uid),
                                       self.major)
            if hyperlink is not None:
                out.hyperlinks.append(hyperlink)
        elif cls == HyperlinkClass.TEXT_SOURCE:
            source = TextSource.read(uid, self.object(uid))
            if source is not None:
                out.text_sources[uid] = source
        elif cls == HyperlinkClass.PAGE_ITEM_SOURCE:
            source = PageItemSource.read(uid, self.object(uid))
            if source is not None:
                out.page_item_sources.append(source)
        elif cls in (HyperlinkClass.PAGE_DESTINATION,
                     HyperlinkClass.URL_DESTINATION,
                     HyperlinkClass.EXTERNAL_PAGE_DESTINATION,
                     HyperlinkClass.TEXT_DESTINATION):
            destination = self.destination(uid, cls)
            if destination is not None:
                out.destinations.append(destination)
        elif cls == XREF_CLASS:
            fmt = CrossReferenceFormat.read(uid, self.object(uid))
            if fmt is not None:
                out.cross_reference_formats[uid] = fmt
        elif cls == HyperlinkClass.BOOKMARK:
            bookmark = Bookmark.read(uid, self.object(uid))
            if bookmark is not None:
                out.bookmarks[uid] = bookmark
        elif cls == CjkClass.COMPOSITE_FONT:
            font = CompositeFont.read(uid, self.object(uid))
            if font is not None:
                out.composite_fonts.append(font)
        elif cls == CjkClass.COMPOSITE_FONT_ENTRY:
            entry = CompositeFontEntry.read(uid, self.object(uid))
            if entry is not None:
                out.composite_entries[uid] = entry
        elif cls in (CjkClass.MOJIKUMI, CjkClass.CUSTOM_MOJIKUMI) \
                or cls in CjkClass.KINSOKU:
            table = CjkTable.read(uid, cls, self.object(uid))
            if table is not None:
                out.cjk_tables.append(table)
        elif cls == ColorClass.INK:
            ink = self.ink(uid)
            if ink is not None:
                out.inks.append(ink)
        elif cls == ColorClass.COLOR_GROUP:
            group = ColorGroup.read(uid, self.object(uid))
            if group is not None:
                out.color_groups[uid] = group
        elif cls == Class.XML_TAG:
            tag = self.xml_tag(uid)
            if tag is not None:
                name, _color = tag
                out.xml_tag_names[uid] = name
                out.xml_tags.append(tag)
        elif cls == Class.LANGUAGE:
            res = self.language(uid)
            if res is not None:
                name, language = res
                out.languages[uid] = name
                if language is not None:
                    out.language_list.append(language)

    def color_object(self, out, uid):
        """A colour object (class 0x200): a colour or a tint."""
        obj = self.object(uid)
        color = Color.read(uid, obj)
        if color is not None:
            # A member of a mixed ink group takes its inks from the
            # group.
            group = None
            base_data = obj.chunk(ColorChunk.TINT_BASE)
            if base_data is not None and color.space == ("other", 9):
                group_uid = self.cursor(base_data).u32()
                if group_uid != uid and \
                        self.class_of(group_uid) == ColorClass.COLOR:
                    group = self.object(group_uid)
            mixed = MixedInk.read(color, obj, group)
            if mixed is not None:
                out.swatches[uid] = mixed.reference()
                out.mixed_inks.append(mixed)
                return
            if color.model_name() is None:
                self.warn(f"colour {uid}: colour model code "
                          f"{color.model} is not known; left out")
            out.swatches[uid] = color.reference()
            out.colors.append(color)
        else:
            tint = Tint.read(uid, obj)
            if tint is not None:
                out.tints.append(tint)

    def gradient(self, uid):
        """A gradient, with a warning for each midpoint IDML cannot
        hold."""
        gradient = Gradient.read(uid, self.object(uid))
        if gradient is None:
            return None
        for i in range(1, len(gradient.stops)):
            if gradient.idml_midpoint(i) is None:
                self.warn(f"gradient {uid}: midpoint before stop {i} "
                          f"is not within 13–87 %; left out")
        return gradient

    def font_family(self, uid):
        """A font family. If its fonts cannot be read, the family keeps
        its name, which text formatting refers to."""
        # A family that refers to another (fonts.md).
        data = self.chunk(uid, FontChunk.FAMILY)
        ref = font_reference(self.enc(), data) if data is not None else None
        if ref is not None:
            target, count = ref
            family = None
            if self.class_of(target) == Class.FONT_FAMILY \
                    and target != uid:
                target_obj = self.object(target)
                target_data = target_obj.chunk(FontChunk.FAMILY)
                if target_data is not None and \
                        font_reference(self.enc(), target_data) is None:
                    try:
                        family = FontFamily.read(target, target_obj)
                    except InddError:
                        family = None
            if family is None:
                self.warn(f"font family {uid}: refers to family "
                          f"{target}, which cannot be read; left out")
                return None
            if count != 0:
                self.warn(f"font family {uid}: refers to family "
                          f"{target} and lists {count} fonts of its "
                          f"own; fonts left out")
                referred = family.referring(uid)
                referred.fonts = []
                return referred
            return family.referring(uid)
        try:
            family = FontFamily.read(uid, self.object(uid))
        except InddError as exc:
            self.warn(f"font family {uid}: fonts left out: {exc}")
            data = self.chunk(uid, FontChunk.FAMILY)
            if data is None:
                return None
            return FontFamily(uid=uid, name=find_string(self.enc(), data, 0),
                              builtin=False, native_name="", fonts=[],
                              writing_script=0)
        if family is not None and not (-2 ** 31 <= family.writing_script < 2 ** 31):
            self.warn(f"font family {uid}: writing script "
                      f"{family.writing_script:#x} is not an IDML "
                      f"integer; left out")
        return family

    def destination(self, uid, cls):
        """A page, URL, external page or text destination."""
        dest = Destination.read(uid, cls, self.object(uid))
        if dest is None:
            return None
        # The file name of an external page destination is the last
        # part of its link's URI (hyperlinks.md).
        if dest.kind[0] == DestinationKind.EXTERNAL_PAGE \
                and dest.kind[5] is not None:
            linked = self.link(dest.kind[5])
            if linked is not None:
                link, _data = linked
                file_name = None
                parts = link.uri.rsplit("/", 1)
                if parts and parts[-1]:
                    file_name = percent_decode(parts[-1])
                kind = list(dest.kind)
                kind[6] = file_name
                dest.kind = tuple(kind)
        if dest.kind[0] == DestinationKind.PAGE and dest.kind[2] is None:
            self.warn(f"destination {uid}: view zoom is outside "
                      f"5–4000 %; left out")
        return dest

    def ink(self, uid):
        """An ink, with a warning if its neutral density cannot be
        written."""
        ink = Ink.read(uid, self.object(uid))
        if ink is not None and ink.neutral_density is None:
            self.warn(f"ink {uid}: neutral density is outside "
                      f"0.001–10; left out")
        return ink

    def tints(self, objects):
        """The tint swatches, with their IDML reference and name. Each
        tint is added to ``objects.swatches``; a tint whose base colour
        has no name is left out with a warning."""
        out = []
        for tint in objects.tints:
            base = next((c for c in objects.colors
                         if c.uid == tint.base and c.name), None)
            if base is None:
                self.warn(f"tint {tint.uid}: left out: base colour "
                          f"{tint.base} has no name")
                continue
            reference = tint.reference(base)
            objects.swatches[tint.uid] = reference
            out.append((tint, reference, tint.idml_name(base)))
        objects.tints = []
        return out

    def xml_tag(self, uid):
        """An XML tag (class 0xBF19): its name and colour. Chunk
        0xBF2F: u32 length, then the name as text segments; chunk
        0x117: the UID of the tag's colour."""
        data = self.chunk(uid, Chunk.XML_TAG_NAME)
        if data is None:
            return None
        c = self.cursor(data)
        n = c.u32()
        try:
            name = c.segments(n)
        except InddError:
            c = self.cursor(data)
            c.flag()
            name = c.string()
        color = None
        data = self.chunk(uid, Chunk.XML_TAG_COLOR)
        if data is not None and len(data) >= 4:
            color = self.ui_color(self.cursor(data).u32())
        return (name, color)

    # -- Spreads (src/model/spread.rs) ----------------------------------

    def ui_color(self, uid):
        """The RGB of an interface colour object (class 0x1F11), or
        ``None``."""
        if self.class_of(uid) != Class.UI_COLOR:
            return None
        data = self.chunk(uid, ColorChunk.COLOR_VALUE)
        if data is None:
            return None
        c = self.cursor(data)
        if c.u32() != 5 or c.u16() != 3:
            return None
        return [c.f64(), c.f64(), c.f64()]

    def section(self, uid):
        section = Section(uid=uid, page=None, continue_numbering=True,
                          include_prefix=False, start=1,
                          style=Numbering.ARABIC, prefix="", marker="",
                          alternate_layout=None)
        data = self.chunk(uid, Chunk.SECTION_INFO)
        if data is not None:
            c = self.cursor(data)
            try:
                c.u8()
                prefix = c.string()
                c.u8()
                marker = c.string()
                page, start, style, cont = (c.u32(), c.u32(), c.u32(),
                                            c.u32())
            except InddError:
                page = start = style = cont = None
                prefix = marker = None
            if prefix is not None:
                section.page = uid_or_none(page)
                section.start = start
                section.style = style
                section.continue_numbering = cont != 0
                section.prefix = prefix
                section.marker = marker
                try:
                    c.u16()
                    section.include_prefix = c.u16() != 0
                    flag = c.flag()
                    section.alternate_layout = (flag, c.string())
                except InddError:
                    section.alternate_layout = None
        if section.style not in (Numbering.ARABIC,
                                 Numbering.LOWER_ROMAN, Numbering.KANJI,
                                 Numbering.DOUBLE_LEADING_ZEROS,
                                 Numbering.UPPER_ROMAN,
                                 Numbering.LOWER_LETTERS,
                                 Numbering.SINGLE_LEADING_ZEROS):
            self.warn(f"section {uid}: page number style "
                      f"{section.style:#x} is not known; left out")
        return section

    def layer(self, uid, internal):
        """A document layer; ``internal`` for the first layer of the
        document's list."""
        data = self.required(uid, Chunk.LAYER_PROPS)
        c = self.cursor(data)
        locked = c.u16() != 0
        visible = c.u16() != 0
        c.skip(14)
        # A string follows, at offset 19 in all corpus pairs;
        # elsewhere search for the tag (objects.md, layers).
        if len(data) > 19 and data[19] == 2 and not self.enc().big_endian:
            name = self.cursor(data[19:]).string()
        else:
            name = find_string(self.enc(), data, 18)
        settings = self.layer_settings(data)
        return Layer(uid=uid, internal=internal, name=name,
                     visible=visible, locked=locked, settings=settings)

    def layer_settings(self, data):
        """The settings in chunk 0x304 when the name is at offset 19,
        as in all corpus pairs: u16 printable at 4, u16 lock guides at
        6, u32 colour at 10, u16 UI at 14; after the name, u16 ignore
        wrap."""
        if self.enc().big_endian or len(data) < 23 or data[19] != 2:
            return None
        c = self.cursor(data[19:])
        c.string()
        ignore_wrap = c.u16() != 0
        def flag(at):
            return self.enc().u16_at(data, at) not in (None, 0)
        color = None
        u = self.enc().u32_at(data, 10)
        if u is not None:
            color = self.ui_color(u)
        return LayerSettings(printable=flag(4), lock_guides=flag(6),
                             ui=flag(14), ignore_wrap=ignore_wrap,
                             color=color)

    def ui_color_ref(self, code):
        """A colour setting: 0 none, 1 the master's, else an interface
        colour."""
        if code == 0:
            return (UiColorRef.NOTHING, None)
        if code == 1:
            return (UiColorRef.USE_MASTER, None)
        rgb = self.ui_color(code)
        if rgb is not None:
            return (UiColorRef.RGB, rgb)
        return (UiColorRef.UNKNOWN, None)

    def spread(self, uid):
        transform = None
        data = self.chunk(uid, Chunk.SPREAD_TRANSFORM)
        transform = Matrix.read(self.cursor(data)) if data is not None \
            else Matrix.IDENTITY
        binding_location = 0
        data = self.chunk(uid, Chunk.SPREAD_BINDING)
        if data is not None:
            binding_location = self.cursor(data).u32()
        master_name = None
        data = self.chunk(uid, Chunk.MASTER_NAME)
        if data is not None:
            c = self.cursor(data)
            c.u8()
            prefix = c.string()
            c.u8()
            base = c.string()
            master_name = (prefix, base)
        children = self.required(uid, Chunk.SPREAD_CHILDREN)
        c = self.cursor(children)
        c.skip(8)
        spread_layers = c.u32_list()
        pages = []
        items = []
        guides = []
        for sl in spread_layers:
            layer = 0
            data = self.chunk(sl, Chunk.SPREAD_LAYER_LAYER)
            if data is not None:
                layer = self.cursor(data).u32()
            for child in self.children(sl, Chunk.SPREAD_LAYER_CHILDREN):
                cls = self.class_of(child)
                if cls == Class.PAGE:
                    pages.append(self.page(child))
                elif cls == Class.GUIDE:
                    guide = self.guide(child, layer)
                    if guide is not None:
                        guides.append(guide)
                else:
                    item = self.page_item(child, layer)
                    if item is not None:
                        items.append(item)
        data = self.chunk(uid, Chunk.SPREAD_TAB_ORDERS)
        if data is not None:
            c = self.cursor(data)
            n = c.u32()
            for _ in range(n):
                page_uid = c.u32()
                order = c.u32_list()
                for page in pages:
                    if page.uid == page_uid:
                        page.settings.tab_order = order
        flattener_resolution = None
        data = self.chunk(uid, Chunk.SPREAD_FLATTENER)
        if data is not None:
            # f64 at 28 line art and text, at 20 gradient and mesh.
            line_art = self.enc().f64_at(data, 28)
            gradient = self.enc().f64_at(data, 20)
            if line_art is not None and gradient is not None:
                flattener_resolution = [line_art, gradient]
        transition = None
        data = self.chunk(uid, Chunk.SPREAD_TRANSITION)
        if data is not None:
            first = self.enc().u32_at(data, 0)
            second = self.enc().u32_at(data, 8)
            if first is not None and second is not None:
                transition = (first, second)

        def short(chunk_id):
            data = self.chunk(uid, chunk_id)
            if data is None:
                return None
            return self.enc().u16_at(data, 0)

        primary_story = None
        data = self.chunk(uid, Chunk.MASTER_PRIMARY_STORY)
        if data is not None:
            primary_story = self.enc().u32_at(data, 0)
        return Spread(
            transition=transition,
            timing=self.timing(uid, Chunk.SPREAD_TIMING),
            uid=uid,
            master_name=master_name,
            transform=transform,
            binding_location=binding_location,
            pages=pages,
            items=items,
            guides=guides,
            shuffle=short(Chunk.SPREAD_SHUFFLE),
            flattener_resolution=flattener_resolution,
            show_master_items=short(Chunk.MASTER_SHOW_ITEMS),
            primary_story=primary_story,
        )

    def guide(self, uid, layer):
        """A ruler guide from chunk 0x3308: f64 position, u32 owner
        (page or spread), u16 orientation (1 horizontal), f64 view
        threshold, u32 colour, u16 fit to page, f64, u32, u32 guide
        type, f64. Records of 40 bytes, from InDesign 3.0 to 7.0, end
        before the guide type."""
        data = self.chunk(uid, Chunk.GUIDE)
        if data is None:
            return None
        if len(data) != 40 and len(data) < 52:
            self.warn(f"guide {uid}: guide record of {len(data)} bytes "
                      f"is not known; left out")
            return None
        def f(o):
            return self.cursor(data[o:]).f64()

        def u(o):
            return self.cursor(data[o:]).u32()

        def h(o):
            return self.cursor(data[o:]).u16()
        locked = False
        data_locked = self.chunk(uid, Chunk.ITEM_LOCKED)
        if data_locked is not None:
            locked = self.enc().u32_at(data_locked, 0) == 1
        overridden = None
        data_over = self.chunk(uid, Chunk.ITEM_OVERRIDE)
        if data_over is not None:
            c = self.cursor(data_over)
            master = c.u32()
            overridden = (master, c.u32_list())
        return Guide(
            uid=uid,
            position=f(0),
            owner=u(8),
            horizontal=h(12) == 1,
            view_threshold=f(14),
            color=u(22),
            fit_to_page=h(26) == 1,
            guide_type=u(40) if len(data) >= 52 else None,
            layer=layer,
            locked=locked,
            zone=f(44) if len(data) >= 52 else None,
            overridden=overridden,
        )

    def page(self, uid):
        # Files from InDesign 3.0 and 4.0 store the transform and
        # bounds in the chunks page items use (docs/format/big-endian.md).
        data = self.chunk(uid, Chunk.PAGE_TRANSFORM)
        if data is None:
            data = self.required(uid, Chunk.ITEM_TRANSFORM)
        transform = Matrix.read(self.cursor(data))
        data = self.chunk(uid, Chunk.PAGE_BOUNDS)
        if data is None:
            data = self.required(uid, Chunk.OLD_PAGE_BOUNDS)
        c = self.cursor(data)
        bounds = [c.f64(), c.f64(), c.f64(), c.f64()]
        master = None
        master_transform = Matrix.IDENTITY
        data = self.chunk(uid, Chunk.PAGE_MASTER)
        if data is not None:
            c = self.cursor(data)
            master = uid_or_none(c.u32())
            c.skip(2)
            # The matrix is missing in files from InDesign 3.0 and 4.0.
            if c.remaining() != 0:
                master_transform = Matrix.read(c)
        margins = None
        data = self.chunk(uid, Chunk.PAGE_MARGINS)
        if data is not None and len(data) >= 34:
            c = self.cursor(data)
            margins = Margins(left=c.f64(), top=c.f64(), right=c.f64(),
                              bottom=c.f64(), own=c.u16() == 1)
        columns = None
        data = self.chunk(uid, Chunk.PAGE_COLUMNS)
        if data is not None:
            c = self.cursor(data)
            n = c.u32()
            if n > len(data) // 8:
                raise _corrupt(f"page {uid}: {n} column positions")
            positions = [c.f64() for _ in range(n)]
            gutter = c.f64()
            own = c.u16() == 1
            # A u16, then the column direction.
            direction = None
            try:
                c.u16()
                direction = c.u16()
            except InddError:
                direction = None
            columns = Columns(positions=positions, gutter=gutter,
                              own=own, direction=direction)
        grid = None
        data = self.chunk(uid, Chunk.PAGE_GRID)
        if data is not None:
            grid = GridData.read(self.cursor(data))
        # u32 count, then (unless it is 0) the two lists.
        overrides = []
        data = self.chunk(uid, Chunk.PAGE_OVERRIDES)
        if data is not None and self.enc().u32_at(data, 0) not in (None, 0):
            c = self.cursor(data[4:])
            items = c.u32_list()
            with_overrides = c.u32_list()
            overrides = list(zip(items, with_overrides, strict=False))
        grid_use = self.chunk(uid, Chunk.PAGE_GRID_USE)
        settings = PageSettings(
            overrides=overrides,
            use_master_grid=(None if grid_use is None else
                             self.enc().u16_at(grid_use, 4) != 0
                             if self.enc().u16_at(grid_use, 4) is not None
                             else None),
            grid_start=(None if grid_use is None else
                        self.enc().u32_at(grid_use, 0)),
            layout_rule=None,
            color=(UiColorRef.USE_MASTER, None),
            tab_order=[],
        )
        data = self.chunk(uid, Chunk.PAGE_LAYOUT_RULE)
        if data is not None:
            settings.layout_rule = self.enc().u32_at(data, 4)
        data = self.chunk(uid, Chunk.PAGE_COLOR)
        if data is not None:
            code = self.enc().u32_at(data, 0)
            if code is not None:
                settings.color = self.ui_color_ref(code)
        return Page(uid=uid, bounds=bounds, transform=transform,
                    master=master, master_transform=master_transform,
                    margins=margins, columns=columns, grid=grid,
                    settings=settings)

    # -- Stories (src/model/story.rs) ----------------------------------

    def story_orientation(self, story):
        """The text orientation shared by all frames of a story. A
        story whose frames disagree, or have an orientation that is not
        known, gets none, with a warning."""
        frames = self.frame_orientations.get(story)
        if frames is None:
            return None
        first = frames[0]
        if first is None or any(o != first for o in frames):
            self.warn(f"story {story}: text orientation of its frames "
                      f"is not known or differs; written as horizontal")
            return None
        return first

    def story(self, uid):
        data = self.required(uid, Chunk.STORY_STRANDS)
        c = self.cursor(data)
        c.skip(6)
        strands = [c.u32()]
        strands.extend(c.u32_list())
        text = []
        para = []
        chars = []
        owned = []
        # (start, length, entries) of the tracked-change runs.
        changes = []
        # (start, length, owner, cell) for each stretch of text.
        owners = []
        sources = []
        marker_strand = None
        for strand in strands:
            # A strand without data occurs in a file from InDesign 3.0.
            if self.db.object(strand) is None:
                continue
            if self.class_of(strand) == XmlClass.MARKER_STRAND:
                marker_strand = strand
            if self.class_of(strand) == HyperlinkClass.RANGE_STRAND:
                tree = self.chunk(strand, HyperlinkChunk.RANGE_TREE)
                first = None
                if tree is not None:
                    first = self.enc().u32_at(tree, 0)
                if first is not None:
                    def pages(u):
                        return self.chunk(u, HyperlinkChunk.RANGE_PAGE)
                    try:
                        sources.extend(
                            _source_ranges(self.enc(), first, pages))
                    except InddError as exc:
                        self.warn(f"story {uid}: hyperlink sources "
                                  f"left out: {exc}")
            chunk_data = self.chunk(strand, Chunk.STRAND_DATA)
            if chunk_data is None:
                continue
            c = self.cursor(chunk_data)
            n = c.u16()
            base = 0
            for _ in range(n):
                data_len = c.u32()
                data_uid = c.u32()
                runs = self.required(data_uid, Chunk.STRAND_RUNS)
                r = self.cursor(runs)
                kind = r.u32()
                r.skip(4)
                count = r.u16()
                pos = base
                base += data_len
                for _ in range(count):
                    size = r.u32()
                    rec = r.bytes(size)
                    rc = self.cursor(rec)
                    length = rc.u32()
                    start = pos
                    pos += length
                    if kind == Strand.OWNED_ITEMS:
                        n_owned = rc.u16()
                        for _ in range(n_owned):
                            owned_class = rc.u32()
                            item = rc.u32()
                            owned.append((start, owned_class, item))
                    elif kind == Strand.TEXT:
                        text.extend(rc.segment_units(length))
                    elif kind == Strand.TEXT_OWNER:
                        owner = rc.u32()
                        cell = rc.u32()
                        owners.append((start, length, owner, cell))
                    elif kind == Strand.CHANGES:
                        n_entries = rc.u16()
                        entries = []
                        for _ in range(n_entries):
                            rc.flag()
                            user = rc.string()
                            change_kind = rc.u16()
                            time = (rc.u32() << 32) | rc.u32()
                            rc.skip(2)
                            entries.append(ChangeEntry(
                                kind=change_kind, user=user, time=time))
                        if entries:
                            changes.append((start, length, entries))
                    elif kind in (Strand.PARAGRAPH_STYLE,
                                  Strand.CHARACTER_STYLE):
                        style = rc.u32()
                        n_attrs = rc.u16()
                        attrs = self.attrs_or_warn(
                            f"story {uid}, text formatting",
                            lambda rc=rc, n_attrs=n_attrs: Attrs.parse_text(rc, n_attrs))
                        if attrs is None:
                            attrs = Attrs()
                        target = para if kind == Strand.PARAGRAPH_STYLE \
                            else chars
                        target.append((length, style, attrs))
                    elif kind == Strand.RUBY:
                        pass
                    else:
                        pass
        # InDesign 5.0 text sources: items owned at their first and
        # last character (hyperlinks.md).
        if self.major <= 5:
            ends = {item: pos for pos, cls, item in owned
                    if cls == HyperlinkClass.SOURCE_END}
            markers = self.source_markers()
            for pos, cls, item in owned:
                if cls != HyperlinkClass.SOURCE_START:
                    continue
                pair = markers.get(item)
                if pair is None:
                    continue
                source, end = pair
                last = ends.get(end)
                if last is not None and last >= pos:
                    sources.append(SourceRange(
                        start=pos, length=last + 1 - pos, source=source,
                        story=uid, format=0, paragraph=False))
        # Text records count UTF-16 code units, every other strand
        # counts characters (a surrogate pair is one position).
        # Convert the positions to UTF-16 offsets.
        offsets = char_offsets(text)

        def unit(p):
            if p < len(offsets):
                return offsets[p]
            return len(text) + (p + 1 - len(offsets))

        def to_units(runs):
            out = []
            at = 0
            for length, style, attrs in runs:
                out.append((unit(at + length) - unit(at), style, attrs))
                at += length
            return out

        para = to_units(para)
        chars = to_units(chars)
        owned = [(unit(p), cls, item) for p, cls, item in owned]
        owners = [(unit(p), unit(p + length) - unit(p), owner, cell)
                  for p, length, owner, cell in owners]
        changes = [ChangeRun(start=unit(p),
                             len=unit(p + length) - unit(p),
                             entries=entries)
                   for p, length, entries in changes]
        for source in sources:
            start = unit(source.start)
            source.length = unit(source.start + source.length) - start
            source.start = start
        try:
            xml_markers = self.story_xml(uid, marker_strand, unit)
        except InddError as exc:
            self.warn(f"story {uid}: XML structure left out: {exc}")
            xml_markers = {}
        xml_element = None
        data = self.chunk(uid, XmlChunk.NODE_REF)
        if data is not None and len(data) >= 8:
            c = self.cursor(data)
            xml_element = (c.u32(), c.u32())
        anchors = {}
        tables = {}
        text_variables = {}
        text_destinations = {}
        footnotes = {}
        endnotes = {}
        notes = {}
        index_markers = {}
        deletions = {}
        for pos, cls, item in owned:
            if cls == Class.ENDNOTE:
                # Chunk 0x22616: the endnote's range.
                range_uid = None
                data = self.chunk(item, Chunk.ENDNOTE_RANGE_OF)
                if data is not None:
                    range_uid = uid_or_none(
                        self.enc().u32_at(data, 0) or 0)
                if range_uid is not None:
                    endnotes[pos] = (item, range_uid)
            elif cls == Class.NOTE:
                d = self.required(item, Chunk.NOTE)
                c = self.cursor(d)
                n = c.u32()
                user = "" if n == 0 else c.segments(n)

                def filetime(c=c):
                    return (c.u32() << 32) | c.u32()

                created = filetime()
                modified = filetime()
                notes[pos] = Note(uid=item, runs=[], user=user,
                                  created=created, modified=modified)
            elif cls == Class.DELETED_TEXT:
                deletions.setdefault(pos, []).append(
                    Deletion(uid=item, runs=[]))
            elif cls == Class.INDEX_MARKER:
                # ``Id`` is the last u32 of chunk 0x13009 when the
                # chunk has 30 bytes; IDML has no ``Id`` for a 26-byte
                # chunk. Its first 14 bytes give the type (index.md).
                d = self.chunk(item, Chunk.INDEX_MARKER)
                marker_id = None
                if d is not None and len(d) == 30:
                    marker_id = self.enc().u32_at(d, 26)
                current_page = False
                if d is not None:
                    enc = self.enc()
                    current_page = (
                        enc.u32_at(d, 0), enc.u16_at(d, 4),
                        enc.u32_at(d, 6), enc.u32_at(d, 10)) == \
                        (1, 0, 1, 0)
                index_markers[pos] = IndexMarker(
                    uid=item, id=marker_id, current_page=current_page)
            elif cls == Class.FOOTNOTE:
                footnotes[pos] = Footnote(uid=item, runs=[])
            elif cls == HyperlinkClass.DESTINATION_OWNER:
                # The owner names its destination (hyperlinks.md).
                dest = None
                data = self.chunk(item, HyperlinkChunk.OWNER_DESTINATION)
                if data is not None:
                    dest = self.enc().u32_at(data, 0)
                else:
                    data = self.chunk(item,
                                      HyperlinkChunk.OWNER_DESTINATION_5)
                    if data is not None:
                        dest = self.enc().u32_at(data, 8)
                dest = uid_or_none(dest or 0)
                if dest is not None:
                    read = Destination.read(
                        dest, HyperlinkClass.TEXT_DESTINATION,
                        self.object(dest))
                    if read is not None:
                        text_destinations.setdefault(pos, []).append(read)
            elif cls == Class.TEXT_VARIABLE_INSTANCE:
                variable = VariableInstance.read(item,
                                                 self.object(item))
                text_variables[pos] = variable
            elif cls == Class.ANCHOR:
                settings = None
                data = self.chunk(item, Chunk.ANCHOR_SETTINGS)
                if data is not None:
                    settings = AnchorSettings.read(self.enc(), data)
                for child in self.children(item, Chunk.ANCHOR_CHILDREN):
                    page_item = self.page_item(child, None)
                    if page_item is not None:
                        page_item.anchor = settings
                        anchors.setdefault(pos, []).append(page_item)
            elif cls == TableClass.TABLE_ANCHOR:
                # A table that cannot be read is left out, with a
                # warning, rather than failing the whole document.
                try:
                    table = self.table_of_anchor(item)
                except InddError as exc:
                    self.warn(f"story {uid}: table left out: {exc}")
                    table = None
                if table is not None:
                    tables[pos] = table
        cuts = [o[0] for o in owners]
        all_runs = split_runs(text, para, chars, cuts)

        def owner_at(at):
            for start, length, owner, cell in owners:
                if start <= at < start + length:
                    return (owner, cell)
            return None

        runs = []
        for run in all_runs:
            found = owner_at(run.start)
            if found is None:
                runs.append(run)
                continue
            owner, cell = found
            footnote = next((f for f in footnotes.values()
                             if f.uid == owner), None)
            if footnote is not None:
                footnote.runs.append(run)
                continue
            note = next((n for n in notes.values()
                         if n.uid == owner), None)
            if note is not None:
                note.runs.append(run)
                continue
            deletion = next((d for ds in deletions.values() for d in ds
                             if d.uid == owner), None)
            if deletion is not None:
                deletion.runs.append(run)
                continue
            if owner != uid:
                table = next((t for t in tables.values()
                              if t.uid == owner), None)
                cell_obj = None
                if table is not None:
                    cell_obj = next((c for c in table.cells
                                     if c.id == cell), None)
                if cell_obj is not None:
                    cell_obj.runs.append(run)
                    continue
            runs.append(run)
        # The range tree of the endnote story also holds the endnotes'
        # ranges (footnotes.md).
        ranges = [r for r in sources
                  if self.class_of(r.source) == Class.ENDNOTE_RANGE]
        sources = [r for r in sources
                   if self.class_of(r.source) != Class.ENDNOTE_RANGE]
        endnote_ranges = []
        for r in ranges:
            endnote = None
            data = self.chunk(r.source, Chunk.RANGE_ENDNOTE)
            if data is not None:
                endnote = uid_or_none(self.enc().u32_at(data, 0) or 0)
            inside = any(run.start <= r.start and
                         r.start + r.length <=
                         run.start + len(run.text.encode("utf-16-le")) // 2
                         for run in runs)
            if endnote is not None and inside and r.length > 0:
                endnote_ranges.append(EndnoteRange(
                    start=r.start, len=r.length, uid=r.source,
                    endnote=endnote))
            else:
                self.warn(f"story {uid}: endnote range {r.source} spans "
                          f"several text ranges; left out")
        endnote_ranges.sort(key=lambda r: r.start)
        # IDML writes a text source inside one character range, or as a
        # child of a paragraph range around several character ranges
        # (hyperlinks.md, extent and placement).
        sources.sort(key=lambda r: r.start)
        lists = [runs]
        lists.extend(cell.runs for t in tables.values() for cell in t.cells)
        lists.extend(f.runs for f in footnotes.values())
        lists.extend(n.runs for n in notes.values())
        lists.extend(d.runs for ds in deletions.values() for d in ds)
        kept = []
        for r in sources:
            if r.length == 0:
                continue
            end = r.start + r.length

            def run_end(t):
                return t.start + len(t.text.encode("utf-16-le")) // 2

            span = None
            for run_list in lists:
                a = next((i for i, t in enumerate(run_list)
                          if r.start >= t.start and r.start < run_end(t)),
                         None)
                b = next((i for i, t in enumerate(run_list)
                          if end > t.start and end <= run_end(t)), None)
                if a is not None and b is not None and a <= b:
                    span = run_list[a:b + 1]
                    break
            if span is None:
                self.warn(f"story {uid}: hyperlink source {r.source} "
                          f"spans several text ranges; left out")
                continue
            first = span[0]
            one_paragraph = all(run_end(span[i]) == span[i + 1].start
                                for i in range(len(span) - 1)) and \
                all((t.paragraph_style, t.paragraph_attrs) ==
                    (first.paragraph_style, first.paragraph_attrs)
                    for t in span)
            if not one_paragraph:
                self.warn(f"story {uid}: hyperlink source {r.source} "
                          f"spans several paragraph style ranges; "
                          f"left out")
                continue
            # The schema allows no text directly in a cross-reference
            # source: one that holds more than a text variable
            # instance is written around a character range.
            cross_reference = False
            try:
                cross_reference = self.chunk(
                    r.source, HyperlinkChunk.CROSS_REFERENCE) is not None
            except InddError:
                cross_reference = False
            variable_only = r.length == 1 and r.start in text_variables
            r.paragraph = len(span) > 1 or \
                (cross_reference and not variable_only)
            if r.paragraph:
                if any(isinstance(m, XmlMarker) and m.kind != "hidden"
                       for pos, m in xml_markers.items()
                       if r.start <= pos < end):
                    self.warn(f"story {uid}: hyperlink source "
                              f"{r.source} holds an XML marker; left out")
                    continue
                kept.append(r)
                continue
            # The IDML schema allows no page item inside a text source
            # written in a character range.
            if any(r.start <= pos < end for pos in anchors):
                self.warn(f"story {uid}: hyperlink source {r.source} "
                          f"holds an anchored object; left out")
                continue
            kept.append(r)
        sources = kept
        margin = self.chunk(uid, Chunk.STORY_OPTICAL_MARGIN)
        title = None
        data = self.chunk(uid, Chunk.STORY_TITLE)
        if data is not None:
            title = self.cursor(data).name().idml()
        direction = None
        data = self.chunk(uid, Chunk.STORY_DIRECTION)
        if data is not None:
            direction = self.enc().u16_at(data, 0)
        optical_margin = None
        layout = None
        if margin is not None:
            optical_margin = (self.enc().f64_at(margin, 2),
                              self.enc().u16_at(margin, 12))
            layout = (self.enc().u16_at(margin, 0),
                      self.enc().u16_at(margin, 14))
        grid = None
        data = self.chunk(uid, Chunk.PAGE_GRID)
        if data is not None:
            grid = GridData.read(self.cursor(data))
        try:
            named_grid = self.applied_named_grid(uid)
        except InddError:
            named_grid = None
        try:
            linked = self.story_link_chunks(uid)
        except InddError as exc:
            self.warn(f"story {uid}: link and import options left out: "
                      f"{exc}")
            linked = StoryLinkChunks()
        toc_style = None
        data = self.chunk(uid, Chunk.STORY_TOC)
        if data is not None:
            toc_uid = uid_or_none(self.enc().u32_at(data, 0) or 0)
            if toc_uid is not None:
                data = self.chunk(toc_uid, Chunk.TOC_STYLE_OF)
                if data is not None:
                    toc_style = uid_or_none(
                        self.enc().u32_at(data, 0) or 0)
        return Story(
            uid=uid,
            runs=runs,
            anchors=anchors,
            tables=tables,
            text_variables=text_variables,
            sources=sources,
            xml_markers=xml_markers,
            footnotes=footnotes,
            notes=notes,
            deletions=deletions,
            changes=changes,
            index_markers=index_markers,
            endnotes=endnotes,
            is_endnote=self.class_of(uid) == Class.ENDNOTE_STORY,
            endnote_ranges=endnote_ranges,
            text_destinations=text_destinations,
            xml_element=xml_element,
            orientation=None,
            # The story names an object (class 0x8C20) whose chunk
            # 0x11613 is the TOC style.
            settings=StoryChunks(
                title=title,
                direction=direction,
                optical_margin=optical_margin,
                layout=layout,
                grid=grid,
                named_grid=named_grid,
                linked=linked,
            ),
            toc_style=toc_style,
        )

    def source_markers(self):
        """InDesign 5.0 text sources by their start marker: (source
        UID, end marker UID), from chunk 0x13524 of every text
        source."""
        if self._source_markers is not None:
            return self._source_markers
        out = {}
        for uid, cls in self.db.classes:
            if cls != HyperlinkClass.TEXT_SOURCE:
                continue
            if self.db.object(uid) is None:
                continue
            data = self.chunk(uid, HyperlinkChunk.SOURCE_MARKERS)
            if data is None:
                continue
            start = self.enc().u32_at(data, 0)
            end = self.enc().u32_at(data, 4)
            if start is not None and end is not None:
                out[start] = (uid, end)
        self._source_markers = out
        return out

    def story_xml(self, uid, marker_strand, unit):
        """Read the XML nodes stored with story ``uid`` and place their
        markers. ``unit`` converts a character position to a UTF-16
        offset."""
        out = {}
        data = self.chunk(uid, XmlChunk.STORE)
        if data is None or len(data) < 4:
            return out
        first = self.cursor(data).u32()
        positions = {}
        if marker_strand is not None:
            data = self.chunk(marker_strand, XmlChunk.MARKER_TREE)
            if data is not None:
                positions = xml_marker_positions(self.enc(), data)
        try:
            nodes = read_xml_store(
                self.enc(), first,
                lambda p: self.chunk(p, XmlChunk.PAGE))
        except InddError as exc:
            # The markers are not text even when their nodes cannot be
            # read (xml.md).
            self.warn(f"story {uid}: XML structure left out: {exc}")
            return {unit(p): XmlMarker(kind="hidden", key=None)
                    for p in positions.values()}
        if not nodes:
            return out
        for node in nodes:
            def at(m):
                if m is None or m not in positions:
                    return None
                return unit(positions[m])

            start = at(node.start)
            end = at(node.end)
            if (node.start is not None) != (start is not None) or \
                    (node.end is not None) != (end is not None):
                self.warn(f"story {uid}: XML node {node.key!r} left out: "
                          f"marker not found")
            elif node.document:
                if start is not None:
                    out[start] = XmlMarker(kind="hidden", key=None)
            elif start is not None and end is not None:
                out[start] = XmlMarker(kind="start", key=node.key)
                out[end] = XmlMarker(kind="end", key=node.key)
            elif start is not None:
                out[start] = XmlMarker(kind="placeholder", key=node.key)
            self.xml_nodes[node.key] = node
        return out

    def xml_elements(self, tags):
        """The XML elements that can be written: those whose parents
        lead to the document node and whose tag is known. See
        ``docs/format/xml.md``."""
        out = {}
        for key, node in self.xml_nodes.items():
            if node.document:
                continue
            # ``Self``: "d", then "i" and the hex ID of each element
            # from the root down to this one.
            path = [key[1]]
            parent = node.parent
            reached = False
            while parent in self.xml_nodes:
                parent_node = self.xml_nodes[parent]
                if parent_node.document:
                    reached = True
                    break
                if len(path) > len(self.xml_nodes):
                    break
                path.append(parent_node.key[1])
                parent = parent_node.parent
            tag = tags.get(node.tag)
            if tag is None:
                self.warn(f"XML element {key!r} left out: tag "
                          f"{node.tag} unknown")
                continue
            if not reached:
                self.warn(f"XML element {key!r} left out: no path to "
                          f"the root")
                continue
            name = "d"
            for part in reversed(path):
                name += f"i{part:x}"
            content = None
            story_content = False
            if node.content == 0:
                pass
            elif self.class_of(node.content) == Class.STORY:
                content = node.content
                story_content = True
            elif self.class_of(node.content) == XmlClass.CONTENT_HOLDER:
                data = self.chunk(node.content, Chunk.ITEM_HIERARCHY)
                if data is not None and len(data) >= 8:
                    content = uid_or_none(self.cursor(data[4:]).u32())
            else:
                self.warn(f"XML element {key!r}: content "
                          f"{node.content} of class "
                          f"{self.class_of(node.content)} left out")
            out[key] = XmlElement(name=name, tag=tag, content=content,
                                  story_content=story_content,
                                  attributes=node.attributes)
        return out


    # -- Page items (src/model/item.rs) --------------------------------

    def frame_orientation(self, mcf):
        """Text orientation of a multi-column frame (chunk 0x2DE): the
        identity gives horizontal text, the rotation 0 1 −1 0 vertical
        text. Returns ``_ABSENT`` without the chunk, ``None`` for
        another matrix, else ``"horizontal"`` / ``"vertical"``."""
        data = self.chunk(mcf, Chunk.FRAME_TEXT_TRANSFORM)
        if data is None:
            return _ABSENT
        if len(data) < 48:
            return None
        matrix = Matrix.read(self.cursor(data))
        a, b, c, d = matrix.values[0], matrix.values[1], \
            matrix.values[2], matrix.values[3]
        if (a, b, c, d) == (1.0, 0.0, 0.0, 1.0):
            return Orientation.HORIZONTAL
        if (a, b, c, d) == (0.0, 1.0, -1.0, 0.0):
            return Orientation.VERTICAL
        return None

    def text_frame_preferences(self, frame, mcf):
        """Text frame settings of frame ``frame``, from its
        multi-column frame ``mcf`` and the frame itself."""
        cols = self.chunk(mcf, Chunk.FRAME_COLUMNS)
        if cols is None:
            return None
        # A frame without chunk 0x2CE has codes 0, reference point 4
        # and no minimum sizes (objects.md, text frame preferences).
        just = self.chunk(mcf, Chunk.FRAME_JUSTIFICATION)
        if just is None:
            just = bytes(48)
            just = just[:24] + self.enc().u16_bytes(4) + just[26:]
        if len(cols) < 22 or len(just) < 22:
            return None
        auto_sizing = len(just) >= 26
        baseline_grid = None
        data = self.chunk(mcf, Chunk.FRAME_BASELINE_GRID)
        if data is not None:
            baseline_grid = BaselineGrid.read(self.enc(), data, 0)
        baseline_rgb = None
        if baseline_grid is not None and baseline_grid.color != 0:
            baseline_rgb = self.ui_color(baseline_grid.color)
        minimum_sizes = None
        if len(just) >= 48:
            def flag(at):
                return self.enc().u16_at(just, at) not in (None, 0)
            minimum_sizes = (
                [flag(26), flag(36)],
                [self.enc().f64_at(just, 28) or 0.0,
                 self.enc().f64_at(just, 38) or 0.0],
                flag(46),
            )
        rule_chunk = self.chunk(mcf, Chunk.FRAME_COLUMN_RULE)
        column_rule = None
        if rule_chunk is not None:
            w = self.enc().f64_at(rule_chunk, 28)
            c = self.enc().u32_at(rule_chunk, 36)
            if w is not None and c is not None:
                column_rule = (w, c)
        footnote_chunk = self.chunk(mcf, Chunk.FRAME_FOOTNOTES)
        footnotes = None
        if footnote_chunk is not None:
            a = self.enc().f64_at(footnote_chunk, 4)
            b = self.enc().f64_at(footnote_chunk, 12)
            if a is not None and b is not None:
                footnotes = [a, b]
        # The frame's chunk 0x3723: f64, u32, then four f64: left, top,
        # right, bottom.
        inset = [0.0, 0.0, 0.0, 0.0]
        inset_single = 0.0
        data = self.chunk(frame, Chunk.FRAME_INSET)
        if data is not None and len(data) >= 44:
            def f(at):
                return self.enc().f64_at(data, at) or 0.0
            inset = [f(20), f(12), f(36), f(28)]
            inset_single = f(0)
        ignore_wrap = None
        data = self.chunk(mcf, Chunk.FRAME_IGNORE_WRAP)
        if data is not None:
            value = self.enc().u16_at(data, 0)
            if value is not None:
                ignore_wrap = value != 0
        return TextFramePreferences(
            column_count=self.cursor(cols).u32(),
            column_gutter=self.cursor(cols[4:]).f64(),
            column_fixed_width=self.cursor(cols[14:]).f64(),
            first_baseline_offset=self.cursor(just).u16(),
            vertical_justification=self.cursor(just[2:]).u16(),
            vertical_balance_columns=self.cursor(just[20:]).u16() != 0,
            auto_sizing_type=(self.enc().u16_at(just, 22)
                              if auto_sizing else None),
            auto_sizing_reference_point=(self.enc().u16_at(just, 24)
                                         if auto_sizing else None),
            use_fixed_width=len(cols) > 12 and cols[12] != 0,
            max_width=(self.enc().f64_at(cols, 32)
                       if len(cols) >= 40 else None),
            minimum_sizes=minimum_sizes,
            ignore_wrap=ignore_wrap,
            column_rule=column_rule,
            column_rule_override=(self.enc().u32_at(rule_chunk, 0) == 1
                                  if rule_chunk is not None else None),
            footnotes=footnotes,
            footnote_span=(self.enc().u16_at(footnote_chunk, 2) == 1
                           if footnote_chunk is not None else False),
            baseline_grid=baseline_grid,
            baseline_rgb=baseline_rgb,
            inset=inset,
            inset_single=inset_single,
            frame_grid=(self.enc().u16_at(
                self.chunk(mcf, Chunk.FRAME_GRID) or b"", 0) == 1
                if self.chunk(mcf, Chunk.FRAME_GRID) is not None
                else False),
        )

    def item_props(self, uid):
        """The settings every page item has (``ItemProps``)."""
        name_chunk = Chunk.GROUP_NAME \
            if self.class_of(uid) == Class.GROUP else Chunk.ITEM_NAME
        name = None
        data = self.chunk(uid, name_chunk)
        if data is not None:
            c = self.cursor(data)
            builtin = c.flag() == 1
            name = ItemName(builtin=builtin, name=c.string())

        def counts(chunk_id):
            data = self.chunk(uid, chunk_id)
            if data is None:
                return []
            c = self.cursor(data)
            n = c.u32()
            if n > len(data) // 8:
                raise _corrupt(f"item {uid}: {n} change counts")
            return [c.u32() for _ in range(2 * n)]

        overridden = None
        data = self.chunk(uid, Chunk.ITEM_OVERRIDE)
        if data is not None:
            c = self.cursor(data)
            master = c.u32()
            overridden = (master, c.u32_list())
        # A form field with states names its spread in chunk 0x14526.
        data = self.chunk(uid, Chunk.ITEM_HIERARCHY)
        if data is None:
            data = self.chunk(uid, Chunk.FORM_STATE_ITEMS)
        on_master = False
        if data is not None:
            spread = self.enc().u32_at(data, 0)
            on_master = spread is not None and \
                self.class_of(spread) == Class.MASTER_SPREAD
        allow_overrides = None
        if on_master:
            data = self.chunk(uid, Chunk.ITEM_ALLOW_OVERRIDES)
            allow_overrides = not (data is not None and
                                   self.enc().u32_at(data, 0) == 1)
        hidden = False
        data = self.chunk(uid, Chunk.ITEM_VISIBLE)
        if data is not None and len(data) >= 2:
            hidden = self.cursor(data).u16() == 0
        locked = False
        data = self.chunk(uid, Chunk.ITEM_LOCKED)
        if data is not None and len(data) >= 4:
            locked = self.cursor(data).u32() == 1
        layout_constraints = None
        data = self.chunk(uid, Chunk.ITEM_LAYOUT_CONSTRAINTS)
        if data is not None and len(data) > 0:
            layout_constraints = data[0]
        export_size_type = None
        data = self.chunk(uid, Chunk.ITEM_EXPORT_SIZE)
        if data is not None:
            try:
                pairs = _key_values(self.cursor(data))
            except InddError:
                pairs = []
            for key, values in pairs:
                if key == "CustomSizeType" and values:
                    try:
                        export_size_type = int(values[0])
                    except ValueError:
                        export_size_type = None
                    break
        return ItemProps(
            name=name,
            allow_overrides=allow_overrides,
            hidden=hidden,
            locked=locked,
            layout_constraints=layout_constraints,
            change_counts=[counts(Chunk.ITEM_PARENT_CHANGES),
                           counts(Chunk.ITEM_TARGET_CHANGES),
                           counts(Chunk.ITEM_UPDATED_CHANGES)],
            overridden=overridden,
            graphic_frame=(self.chunk(uid, Chunk.ITEM_CONTENT) is not None
                           and len(self.chunk(uid, Chunk.ITEM_CONTENT)) >= 2
                           and self.cursor(
                               self.chunk(uid, Chunk.ITEM_CONTENT)).u16() == 1),
            grouping_layer_position=None,
            grouping_layer=self.grouping_layer(uid),
            generated_alt=self.generated_alt(uid),
            export_size_type=export_size_type,
        )

    def generated_alt(self, uid):
        def flag(chunk_id):
            data = self.chunk(uid, chunk_id)
            if data is None:
                return False
            return self.enc().u16_at(data, 0) == 1

        crop = None
        data = self.chunk(uid, Chunk.ITEM_ALT_TEXT_CROP)
        if data is not None:
            rect = [self.enc().f64_at(data, o)
                    for o in (0, 8, 16, 24)]
            if all(v is not None for v in rect) and \
                    any(v != 0.0 for v in rect):
                crop = rect
        return GeneratedAltText(
            error=flag(Chunk.ITEM_ALT_TEXT_ERROR),
            ai=flag(Chunk.ITEM_AI_ALT_TEXT),
            crop=crop,
        )

    def grouping_layer(self, uid):
        """The layer of chunk 0x324: a layer UID itself, a spread
        layer's layer (its chunk 0x302), else none."""
        data = self.chunk(uid, Chunk.ITEM_GROUPING_LAYER)
        if data is None:
            return None
        layer = self.enc().u32_at(data, 0)
        if layer is None:
            return None
        if layer != 0 and self.class_of(layer) == Class.SPREAD_LAYER:
            data = self.chunk(layer, Chunk.SPREAD_LAYER_LAYER)
            layer = (self.enc().u32_at(data, 0) or 0) \
                if data is not None else 0
        if layer != 0 and self.class_of(layer) == Class.LAYER:
            return layer
        return None

    def export_options(self, uid):
        """The export options of a page item (chunk 0x1E206): the
        alternative text and actual text, each a u32 source and three
        flagged strings (text, metadata namespace prefix, property
        path), then the u32 tagging type. The rest of the chunk is not
        decoded."""
        data = self.chunk(uid, Chunk.ITEM_EXPORT)
        if data is None:
            return None
        c = self.cursor(data)
        alt_source = c.u32()
        alt_text = c.name()
        alt_metadata = [c.name(), c.name()]
        actual_source = c.u32()
        actual_text = c.name()
        actual_metadata = [c.name(), c.name()]
        return ExportOptions(
            alt_source=alt_source,
            alt_text=alt_text,
            alt_metadata=alt_metadata,
            actual_source=actual_source,
            actual_text=actual_text,
            actual_metadata=actual_metadata,
            tag_type=c.u32(),
        )

    def text_wrap(self, uid):
        """Text wrap of a page item or graphic, from chunk 0x3703:
        u32 mode, u32 contour path object, four f64 offsets, u16
        master page flag, u16 side."""
        wrap = self.wrap_chunk(uid, Chunk.TEXT_WRAP)
        if wrap is not None:
            wrap.custom_path = self.custom_wrap_path(uid, wrap.path_uid)
        return wrap

    def custom_wrap_path(self, uid, path):
        """The paths (chunk 0x162B) of wrap path object ``path`` when
        chunk 0x3709 flags it custom (u16 at offset 8 = 1). A path that
        cannot be read is left out with a warning."""
        if path == 0 or self.class_of(path) != Class.TEXT_WRAP_PATH:
            return []
        data = self.chunk(path, Chunk.TEXT_WRAP_PATH_INFO)
        if data is None:
            return []
        if len(data) != 10 or self.cursor(data[8:]).u16() != 1:
            return []
        try:
            paths = self.paths(path)
        except InddError as exc:
            self.warn(f"item {uid}: text wrap path not read ({exc})")
            return []
        return [p for p in paths if p.points]

    def wrap_chunk(self, uid, chunk_id):
        """A text wrap record in chunk ``chunk_id`` of object ``uid``."""
        data = self.chunk(uid, chunk_id)
        if data is None or len(data) < 44:
            return None
        c = self.cursor(data)
        mode = c.u32()
        path_uid = c.u32()
        offsets = [c.f64(), c.f64(), c.f64(), c.f64()]
        master_flag = c.u16()
        side = c.u16()
        if mode not in (WrapMode.NONE, WrapMode.JUMP_OBJECT,
                        WrapMode.BOUNDING_BOX, WrapMode.CONTOUR):
            self.warn(f"item {uid}: text wrap mode {mode} is not "
                      f"known; left out")
        return TextWrap(mode=mode, offsets=offsets,
                        master_flag=master_flag, side=side,
                        path_uid=path_uid, custom_path=[])

    def paths(self, uid):
        data = self.chunk(uid, Chunk.ITEM_PATHS)
        if data is None:
            return []
        c = self.cursor(data)
        n = c.u32()
        out = []
        for _ in range(n):
            count = c.u32()
            points = []
            for _ in range(count):
                kind = c.u32()
                if kind == 2:
                    anchor = (c.f64(), c.f64())
                    points.append(PathPoint(anchor=anchor, left=anchor,
                                            right=anchor, kind=kind))
                elif kind in (0, 1):
                    left = (c.f64(), c.f64())
                    anchor = (c.f64(), c.f64())
                    right = (c.f64(), c.f64())
                    points.append(PathPoint(anchor=anchor, left=left,
                                            right=right, kind=kind))
                else:
                    raise _corrupt(f"object {uid}: unknown path point "
                                   f"type {kind}")
            open_path = c.u16() != 0
            out.append(Path(points=points, open=open_path))
        return out

    def page_item(self, uid, layer):
        """A page item and the items it contains. An item that cannot
        be read, contains itself or is nested more than
        ``MAX_ITEM_DEPTH`` deep is left out with a warning."""
        cls = self.class_of(uid)
        if cls not in (Class.SPLINE_ITEM, Class.GROUP, Class.EPS_TEXT) \
                and FormKind.of(cls) is None:
            return None
        if uid in self.item_path:
            self.warn(f"item {uid} contains itself; left out")
            return None
        if len(self.item_path) >= MAX_ITEM_DEPTH:
            self.warn(f"item {uid} is nested more than "
                      f"{MAX_ITEM_DEPTH} deep; left out")
            return None
        self.item_path.append(uid)
        try:
            item = self.page_item_contents(uid, cls, layer)
        except InddError as exc:
            self.warn(f"item {uid} left out: {exc}")
            item = None
        finally:
            self.item_path.pop()
        return item

    def page_item_contents(self, uid, cls, layer):
        form = FormKind.of(cls)
        states = form is not None and FormKind.has_states(form)
        # Groups without chunk 0x151 have their transform in chunk
        # 0x40D, form fields with states in chunk 0x14527.
        transform = None
        data = self.chunk(uid, Chunk.ITEM_TRANSFORM)
        if data is not None:
            transform = Matrix.read(self.cursor(data))
        else:
            other = Chunk.FORM_TRANSFORM if states else \
                Chunk.GROUP_TRANSFORM
            data = self.chunk(uid, other)
            if data is not None and (states or cls == Class.GROUP) \
                    and len(data) >= 48:
                transform = Matrix.read(self.cursor(data))
            else:
                transform = Matrix.IDENTITY
        child_uids = [] if states else \
            self.children(uid, Chunk.ITEM_HIERARCHY)
        children = []
        graphics = []
        media = []
        text_column = None
        frame_prefs = None
        frame_mcf = None
        for child in child_uids:
            graphic = self.graphic(child)
            if graphic is not None:
                graphics.append(graphic)
                continue
            medium = self.media(child)
            if medium is not None:
                media.append(medium)
                continue
            if self.class_of(child) == Class.MULTI_COLUMN_FRAME:
                frame_prefs = self.text_frame_preferences(uid, child)
                frame_mcf = child
                for col in self.children(child, Chunk.ITEM_HIERARCHY):
                    if self.class_of(col) == Class.FRAME_COLUMN:
                        text_column = col
                        break
                continue
            item = self.page_item(child, layer)
            if item is not None:
                children.append(item)
        paths = self.paths(uid)
        attrs = None
        data = self.chunk(uid, Chunk.ITEM_ATTRS)
        if data is not None:
            attrs = self.attrs_or_warn(
                f"item {uid}",
                lambda: Attrs.parse(self.enc(), data))
        if attrs is None:
            attrs = Attrs()
        object_style = None
        data = self.chunk(uid, Chunk.ITEM_OBJECT_STYLE)
        if data is not None:
            object_style = uid_or_none(self.cursor(data).u32())
        if form is not None:
            form_states = self.form_states(uid, form, layer) \
                if states else []
            form_obj = _form_field(form, form_states, attrs, children)
            if form == FormKind.BUTTON:
                form_obj.timing = self.timing(uid, Chunk.BUTTON_TIMING)
            form_obj.behaviors = self.behaviors(uid)
            kind = ("form", form_obj)
        elif cls == Class.GROUP:
            kind = ("group",)
        elif cls == Class.EPS_TEXT:
            kind = ("eps_text", self.eps_text(uid))
        elif text_column is not None:
            kind = self.text_frame_links(text_column)
            kind[1]["preferences"] = frame_prefs
            if kind[1].get("story") is not None and frame_mcf is not None:
                orientation = self.frame_orientation(frame_mcf)
                if orientation is not _ABSENT:
                    self.frame_orientations.setdefault(
                        kind[1]["story"], []).append(orientation)
        else:
            shape = None
            data = self.chunk(uid, Chunk.ITEM_SHAPE)
            if data is not None:
                shape = (self.enc().u16_at(data, 0),
                         self.enc().u32_at(data, 2))
                if shape[0] is None or shape[1] is None:
                    shape = None
            kind = ("shape", classify(paths, shape))
        try:
            props = self.item_props(uid)
        except InddError as exc:
            self.warn(f"item {uid}: settings left out: {exc}")
            props = ItemProps()
        try:
            export = self.export_options(uid)
        except InddError as exc:
            self.warn(f"item {uid}: export options left out: {exc}")
            export = None
        try:
            text_path = self.text_path(uid)
        except InddError as exc:
            self.warn(f"item {uid}: text on its path left out: {exc}")
            text_path = None
        try:
            label = self.item_label(uid, cls)
        except InddError as exc:
            self.warn(f"item {uid}: label left out: {exc}")
            label = []
        article_children = None
        if cls == Class.GROUP:
            try:
                article_children = self.uid_list_or_none(
                    uid, Chunk.GROUP_ARTICLE_CHILDREN)
            except InddError as exc:
                self.warn(f"group {uid}: article children list left "
                          f"out: {exc}")
                article_children = None
        return PageItem(
            uid=uid,
            graphics=graphics,
            media=media,
            kind=kind,
            transform=transform,
            paths=paths,
            layer=layer,
            attrs=attrs,
            object_style=object_style,
            children=children,
            text_wrap=self.text_wrap(uid),
            anchor=None,
            props=props,
            export=export,
            text_paths=[text_path] if text_path is not None else [],
            label=label,
            animation=self.animation(uid),
            article_children=article_children,
        )

    def form_states(self, uid, kind, layer):
        """The states of a form field (chunks 0x14521 and 0x14526)
        with the items each shows. A state whose key has no entry in
        chunk 0x14526 shows nothing."""
        data = self.chunk(uid, Chunk.FORM_STATES)
        if data is None:
            return []
        c = self.cursor(data)
        active = c.u32()
        n = c.u32()
        if n > c.remaining() // 10:
            raise _corrupt(f"form field {uid}: {n} states")
        entries = []
        for _ in range(n):
            state_id = c.u32()
            enabled = c.u16()
            key = c.u16()
            c.u16()
            entries.append((state_id, enabled, key))
        lists = []
        data = self.chunk(uid, Chunk.FORM_STATE_ITEMS)
        if data is not None:
            c = self.cursor(data)
            c.skip(8)
            n_lists = c.u32()
            if n_lists > c.remaining() // 8:
                raise _corrupt(f"form field {uid}: {n_lists} state lists")
            for _ in range(n_lists):
                key = c.u32()
                lists.append((key, c.u32_list()))
        out = []
        for state_id, enabled, key in entries:
            items = []
            for list_key, uids in lists:
                if list_key != key:
                    continue
                for child in uids:
                    item = self.page_item(child, layer)
                    if item is None:
                        continue
                    # A state's group without a name of its own has
                    # the name of its state type.
                    if item.kind[0] == "group" and (
                            item.props.name is None or
                            (item.props.name.builtin and
                             not item.props.name.name)):
                        name = _state_group_name(kind, state_id)
                        if name is not None:
                            item.props.name = ItemName(builtin=True,
                                                       name=name)
                    items.append(item)
            out.append(FormState(id=state_id, active=state_id == active,
                                 enabled=enabled != 0, items=items))
        return out

    def text_path(self, uid):
        """The text on the path of an item (chunk 0xB30A): u32 text
        path UID, u32, three codes (8 bytes), u8 flip at 16, f64 start
        at 18 and end at 26. The story and thread come from the column
        (class 0xB318) of the text path's multi-column frame (chunk
        0xB334)."""
        data = self.chunk(uid, Chunk.ITEM_TEXT_PATH)
        if data is None:
            return None
        c = self.cursor(data)
        path = c.u32()
        if path == 0 or self.class_of(path) != Class.TEXT_PATH:
            return None
        c.u32()
        codes = c.bytes(8)
        default_codes = self.enc().u16_at(codes, 0) == 1 and \
            self.enc().u16_at(codes, 2) == 4 and \
            self.enc().u32_at(codes, 4) == 3
        flipped = c.u8() == 1
        c.u8()
        start = c.f64()
        end = c.f64()
        column = None
        for mcf in self.children(path, Chunk.TEXT_PATH_FRAME):
            if self.class_of(mcf) != Class.MULTI_COLUMN_FRAME:
                continue
            for col in self.children(mcf, Chunk.ITEM_HIERARCHY):
                if self.class_of(col) == Class.TEXT_PATH_COLUMN:
                    column = col
                    break
            if column is not None:
                break
        story = previous = next_uid = None
        if column is not None:
            kind = self.text_frame_links(column)
            story = kind[1].get("story")
            previous = kind[1].get("previous")
            next_uid = kind[1].get("next")
        return TextPath(uid=path, story=story, previous=previous,
                        next=next_uid, flipped=flipped, start=start,
                        end=end, default_codes=default_codes)

    def eps_text(self, uid):
        """The settings of EPS text (class 0x660B)."""
        def four(chunk_id):
            data = self.chunk(uid, chunk_id)
            if data is None:
                raise _corrupt(f"EPS text {uid}: no chunk {chunk_id:#x}")
            c = self.cursor(data)
            return [c.f64(), c.f64(), c.f64(), c.f64()]

        data = self.chunk(uid, Chunk.EPS_TEXT_DATA)
        if data is None:
            raise _corrupt(f"EPS text {uid}: no text record")
        return EpsText(
            data=_eps_text_data(self.enc(), data),
            path_bounds=four(Chunk.OLD_PAGE_BOUNDS),
            attr_bounds=four(Chunk.EPS_TEXT_ATTR_BOUNDS),
        )

    def graphic(self, uid):
        cls = self.class_of(uid)
        kind = {Class.IMAGE: GraphicKind.IMAGE,
                Class.PDF: GraphicKind.PDF,
                Class.EPS: GraphicKind.EPS,
                Class.SVG: GraphicKind.SVG,
                Class.IMPORTED_PAGE: GraphicKind.IMPORTED_PAGE}.get(cls)
        if kind is None:
            return None
        transform = Matrix.IDENTITY
        data = self.chunk(uid, Chunk.ITEM_TRANSFORM)
        if data is not None:
            transform = Matrix.read(self.cursor(data))
        bounds = [0.0, 0.0, 0.0, 0.0]
        data = self.chunk(uid, Chunk.GRAPHIC_BOUNDS)
        if data is not None:
            c = self.cursor(data)
            bounds = [c.f64(), c.f64(), c.f64(), c.f64()]
        link = None
        contents = None
        data = self.chunk(uid, Chunk.GRAPHIC_LINK)
        if data is not None and len(data) >= 12:
            linked = self.link(self.cursor(data[8:]).u32())
            if linked is not None:
                link, contents = linked
        if link is None:
            data_id = None
            if kind in (GraphicKind.IMAGE, GraphicKind.EPS):
                data_id = Chunk.IMAGE_DATA
            elif kind == GraphicKind.PDF:
                data_id = Chunk.PDF_DATA
            if data_id is not None:
                data = self.chunk(uid, data_id)
                if data is not None and len(data) >= 4:
                    contents = self.raw_data(self.cursor(data).u32())
            # An image that names no raw data object holds its file in
            # its preview, unless it names an old link object.
            if contents is None and kind == GraphicKind.IMAGE \
                    and not self.old_link(uid):
                preview = self.preview(uid)
                if preview is not None:
                    contents = self.raw_data(preview)
        clipping = None
        data = self.chunk(uid, Chunk.CLIPPING_PATH)
        if data is not None and len(data) >= 26:
            clipping = ClippingPath(
                kind=self.cursor(data).u32(),
                tolerance=self.cursor(data[4:]).f64(),
                inset=self.cursor(data[12:]).f64(),
                threshold=data[20],
                index=self.enc().i16_from(bytes(data[23:25])),
                flags=data[25],
            )
        clipping_paths = []
        if clipping is not None and clipping.kind != 0:
            try:
                clipping_paths = self.paths(uid)
            except InddError as exc:
                self.warn(f"graphic {uid}: clipping path not read ({exc})")
                clipping_paths = []
        # A preview that can't be read is left out; the graphic is
        # still read.
        preview = None
        if contents is None:
            try:
                preview = self.preview(uid)
            except InddError:
                preview = None
        try:
            props = self.item_props(uid)
        except InddError as exc:
            self.warn(f"graphic {uid}: settings left out: {exc}")
            props = ItemProps()
        object_style = None
        data = self.chunk(uid, Chunk.ITEM_OBJECT_STYLE)
        if data is not None and len(data) >= 4:
            object_style = uid_or_none(self.cursor(data).u32())
        attrs = None
        data = self.chunk(uid, Chunk.ITEM_ATTRS)
        if data is not None:
            attrs = self.attrs_or_warn(
                f"graphic {uid}",
                lambda: Attrs.parse(self.enc(), data))
        if attrs is None:
            attrs = Attrs()
        image = None
        data = self.chunk(uid, Chunk.IMAGE_PROPERTIES)
        if data is not None and kind == GraphicKind.IMAGE:
            image = self.image_properties(data)
        profile = None
        data = self.chunk(uid, Chunk.IMAGE_PROFILE)
        if data is not None and len(data) >= 4:
            profile = self.cursor(data).u32()
        vector_policies = None
        data = self.chunk(uid, Chunk.VECTOR_POLICIES)
        if data is not None and len(data) >= 16:
            c = self.cursor(data)
            vector_policies = [c.u32(), c.u32(), c.u32(), c.u32()]
        pdf = None
        data = self.chunk(uid, Chunk.PDF_PLACEMENT)
        if data is not None and len(data) >= 12:
            pdf = (self.cursor(data).u32(), data[6],
                   self.cursor(data[8:]).u32())
        layers = None
        data = self.chunk(uid, Chunk.GRAPHIC_LAYERS)
        if data is not None:
            try:
                layers = self.graphic_layers(data)
            except InddError as exc:
                self.warn(f"graphic {uid}: layers left out: {exc}")
                layers = None
        layer_comp = None
        data = self.chunk(uid, Chunk.LAYER_COMP)
        if data is not None and len(data) >= 8:
            layer_comp = self.cursor(data[4:]).i32()
        import_options = None
        data = self.chunk(uid, Chunk.IMAGE_IMPORT)
        if data is not None and len(data) >= 9:
            c = self.cursor(data[4:])
            clipping_flag = c.u16()
            c.u16()
            name = c.name()
            if clipping_flag in (0, 1):
                import_options = (clipping_flag == 1, name)
        alpha = None
        data = self.chunk(uid, Chunk.IMAGE_CHANNELS)
        if data is not None and kind == GraphicKind.IMAGE:
            alpha = _alpha_channel(self.enc(), data)
        page_index = None
        data = self.chunk(uid, Chunk.IMPORTED_PAGE_INFO)
        if data is not None and kind == GraphicKind.IMPORTED_PAGE:
            page_index = self.enc().u32_at(data, 0)
        return Graphic(
            uid=uid,
            kind=kind,
            transform=transform,
            bounds=bounds,
            link=link,
            contents=contents,
            preview=preview,
            text_wrap=self.text_wrap(uid),
            contour=(Contour.read(self.enc(), data)
                     if (data := self.chunk(uid, Chunk.CONTOUR_OPTION))
                     is not None else None),
            clipping=clipping,
            clipping_paths=clipping_paths,
            props=props,
            object_style=object_style,
            attrs=attrs,
            image=image,
            profile=profile,
            vector_policies=vector_policies,
            pdf=pdf,
            layers=layers,
            layer_comp=layer_comp,
            import_options=import_options,
            alpha_channel=alpha,
            page_index=page_index,
        )

    def image_properties(self, data):
        """Image properties from chunk 0x1708; ``None`` if the record
        list is cut short."""
        c = self.cursor(data)
        try:
            n = c.u32()
        except InddError:
            return None
        props = ImageProperties()
        resolution = [None, None]
        for _ in range(n):
            try:
                key = c.u32()
                length = c.u32()
                c.u8()
                record = c.bytes(length)
            except InddError:
                return None
            value = None
            if length >= 4:
                try:
                    value = self.cursor(record).u32()
                except InddError:
                    value = None
            if key == 0x67:
                props.color_space = value
            elif key == 0x6C:
                if value is not None:
                    resolution[0] = value / 65536.0
            elif key == 0x6D:
                if value is not None:
                    resolution[1] = value / 65536.0
            elif key == 0x6F:
                props.indexed = True
            elif key == 0x7D and record:
                # u8 length n, n bytes of name, then 6 bytes.
                count = record[0]
                name = record[1:1 + count]
                if name and all(0x20 <= b < 0x7F for b in name):
                    props.clipping_path_name = "".join(
                        chr(b) for b in name)
        if resolution[0] is not None and resolution[1] is not None:
            props.resolution = resolution
        return props

    def graphic_layers(self, data):
        """Layers of a placed graphic (chunk 0x177A): u16 option flag,
        8 bytes, u32 count, then per layer a flagged name, u32 ID, u32
        original and current visibility, i32 parent ID and u32
        flags."""
        c = self.cursor(data)
        option = c.u16() == 1
        c.skip(8)
        n = c.u32()
        if n > len(data) // 20:
            raise _corrupt(f"{n} graphic layers")
        layers = []
        for _ in range(n):
            name = c.name()
            layers.append(GraphicLayer(
                name=name,
                id=c.u32(),
                original_visibility=c.u32() == 1,
                current_visibility=c.u32() == 1,
                parent=c.i32(),
                flags=c.u32(),
            ))
        return GraphicLayers(option=option, layers=layers)

    def raw_data(self, uid):
        """The bytes of raw data object ``uid`` (``None`` if ``uid`` is
        not one)."""
        if uid == 0 or self.class_of(uid) != Class.RAW_DATA:
            return None
        return self.db.object(uid)

    def old_link(self, uid):
        """Whether image ``uid`` names an object in chunk 0x8C1B, as
        images of InDesign 2.0 to 5.0 do. Their preview is not used as
        ``Contents``."""
        data = self.chunk(uid, Chunk.OLD_GRAPHIC_LINK)
        if data is not None and len(data) >= 4:
            return self.cursor(data).u32() != 0
        return False

    def preview(self, uid):
        """The raw data object of graphic ``uid``'s preview, if it
        names one. The data object itself is not read."""
        data = self.chunk(uid, Chunk.GRAPHIC_PREVIEW)
        if data is None or len(data) < 4:
            return None
        obj = self.cursor(data).u32()
        if obj == 0 or self.class_of(obj) != Class.PREVIEW:
            return None
        data = self.chunk(obj, Chunk.PREVIEW_DATA)
        if data is None or len(data) < 4:
            return None
        raw = self.cursor(data).u32()
        if self.class_of(raw) == Class.RAW_DATA:
            return raw
        return None

    def link(self, uid):
        """A link, the URI of its resource and, for an embedded link,
        the embedded file. Fields after the URI that are cut short are
        left out (``None``)."""
        if uid == 0 or self.db.object(uid) is None:
            return None
        info = self.chunk(uid, Chunk.LINK_INFO)
        if info is None or len(info) < 12:
            return None
        resource = self.cursor(info[8:]).u32()
        link = Link(uid=uid, class_id=self.class_of(uid) or 0)
        data = None
        uri_data = self.chunk(resource, Chunk.LINK_RESOURCE_URI)
        if uri_data is not None and len(uri_data) >= 5:
            c = self.cursor(uri_data)
            c.u8()
            n = c.u32()
            link.uri = c.bytes(min(n, c.remaining())).decode(
                "utf-8", errors="replace")
            # After the URI: 8 bytes, the stored state (3 for
            # contained content), then the UID of the embedded file's
            # raw data object (0 for a normal link).
            if c.remaining() >= 16:
                c.skip(8)
                link.contained = c.u32() == 3
                data = self.raw_data(c.u32())
                # Then the stamp, the modification time, the file
                # size, a byte and the format name.
                try:
                    n = c.u32()
                    c.segments(n)
                    c.skip(8)
                    size = [c.u32(), c.u32()]
                    fmt = c.name().name
                    link.size = size
                    link.format = fmt
                except InddError:
                    pass
        link.embedded = data is not None
        c = self.cursor(info)
        try:
            c.skip(12)
            modified = c.u32()
            c.skip(4)
            shown = c.u16()
            c.skip(10)
        except InddError:
            modified = shown = None
        if modified is not None and shown is not None:
            link.modified = {0: True, 1: False}.get(modified)
            link.shown = {0: False, 1: True}.get(shown)
            try:
                n = c.u32()
                stamp = c.segments(n)
                def time():
                    return (c.u32() << 32) | c.u32()
                times = [time(), time()]
                link.stamp = stamp if stamp else None
                link.times = times
            except InddError:
                pass
        data = self.chunk(uid, Chunk.LINK_PDF_IDENTIFIER)
        if data is not None and len(data) >= 4:
            link.pdf_identifier = self.cursor(data).u32()
        return (link, data)

    def text_frame_links(self, column):
        """Story and threading of the text frame owning ``column``."""
        none_kind = ("text_frame", {"story": None, "previous": None,
                                    "next": None, "preferences": None})
        data = self.chunk(column, Chunk.COLUMN_FRAME_LIST)
        if data is None:
            return none_kind
        frame_list = self.cursor(data).u32()
        data = self.chunk(frame_list, Chunk.FRAME_LIST_FRAMES)
        if data is None:
            return none_kind
        c = self.cursor(data)
        story = uid_or_none(c.u32())
        columns = c.u32_list()

        def frame_of(col):
            data = self.chunk(col, Chunk.ITEM_HIERARCHY)
            if data is None:
                return None
            mcf = self.cursor(data[4:]).u32()
            data = self.chunk(mcf, Chunk.ITEM_HIERARCHY)
            if data is None:
                return None
            return uid_or_none(self.cursor(data[4:]).u32())

        frames = []
        for col in columns:
            frame = frame_of(col)
            if frame is not None and (not frames or frames[-1] != frame):
                frames.append(frame)
        me = frame_of(column)
        pos = None
        if me is not None and me in frames:
            pos = frames.index(me)
        return ("text_frame", {
            "story": story,
            "previous": (frames[pos - 1] if pos is not None and pos > 0
                         else None),
            "next": (frames[pos + 1] if pos is not None
                     and pos + 1 < len(frames) else None),
            "preferences": None,
        })

    # -- Animation, behaviours, media, index --------------------------

    def timing(self, uid, chunk_id):
        try:
            data = self.chunk(uid, chunk_id)
        except InddError as exc:
            self.warn(f"object {uid}: timing left out: {exc}")
            return []
        if data is None:
            return []
        try:
            return _parse_timing(self.cursor(data))
        except InddError as exc:
            self.warn(f"object {uid}: timing left out: {exc}")
            return []

    def animation(self, uid):
        """The animation of a page item; ``None`` without one, or with
        a warning if chunk 0x2055F does not parse."""
        try:
            data = self.chunk(uid, Chunk.ANIMATION)
        except InddError as exc:
            self.warn(f"item {uid}: animation left out: {exc}")
            return None
        if data is None or len(data) <= 2:
            return None
        try:
            return _parse_animation(self.cursor(data))
        except InddError as exc:
            self.warn(f"item {uid}: animation left out: {exc}")
            return None

    def motion_presets(self):
        """The motion presets listed in chunk 0x2053E of the
        preferences object, in that order. A preset that cannot be
        read is left out with a warning."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return []
        try:
            uids = self.uid_list(prefs_uid, Chunk.PRESET_LIST)
        except InddError as exc:
            self.warn(f"motion presets left out: {exc}")
            return []
        out = []
        for uid in uids:
            try:
                if self.class_of(uid) != Class.MOTION_PRESET:
                    raise _corrupt("not a motion preset")
                data = self.required(uid, Chunk.PRESET_DATA)
                c = self.cursor(data)
                contents = _parse_preset_text(c)
                c.skip(4)
                c.name()
                name = c.name()
                if c.remaining() != 0:
                    raise _corrupt("motion preset: bytes left")
                out.append(MotionPreset(uid=uid, name=name,
                                        contents=contents))
            except InddError as exc:
                self.warn(f"motion preset {uid} left out: {exc}")
        return out

    def behaviors(self, uid):
        try:
            data = self.chunk(uid, Chunk.FORM_BEHAVIORS)
        except InddError as exc:
            self.warn(f"form field {uid}: behaviours left out: {exc}")
            return []
        if data is None:
            return []
        try:
            groups = _parse_behavior_list(self.cursor(data))
        except InddError as exc:
            self.warn(f"form field {uid}: behaviours left out: {exc}")
            return []
        out = []
        for event, behaviour_uid in groups:
            try:
                behaviour = self.behavior(behaviour_uid, event)
            except InddError as exc:
                self.warn(f"behaviour {behaviour_uid} left out: {exc}")
                continue
            if behaviour is not None:
                out.append(behaviour)
        return out

    def behavior(self, uid, event):
        """Behaviour ``uid``; ``None``, with a warning, for a class
        that is not read."""
        cls = self.class_of(uid)
        entry = next((e for e in _BEHAVIOR_CLASSES if e[0] == cls), None)
        if entry is None:
            if cls is None:
                self.warn(f"behaviour {uid} left out: no object")
            else:
                self.warn(f"behaviour {uid} left out: class {cls:#x} "
                          f"is not read")
            return None
        data_id = entry[1]
        data = self.required(uid, data_id)
        name, enabled, kind = _parse_behavior_data(self.cursor(data), cls)
        label = []
        data = self.chunk(uid, Chunk.BEHAVIOR_LABEL)
        if data is not None:
            label = self.label_at(data, 0)
        return Behavior(uid=uid, event=event, enabled=enabled,
                        name=name, label=label, kind=kind)

    def media(self, uid):
        cls = self.class_of(uid)
        kind = {Class.MOVIE: MediaKind.MOVIE,
                Class.SOUND: MediaKind.SOUND}.get(cls)
        if kind is None:
            return None
        transform = Matrix.IDENTITY
        data = self.chunk(uid, Chunk.ITEM_TRANSFORM)
        if data is not None:
            transform = Matrix.read(self.cursor(data))
        bounds = [0.0, 0.0, 0.0, 0.0]
        data = self.chunk(uid, Chunk.GRAPHIC_BOUNDS)
        if data is not None:
            c = self.cursor(data)
            bounds = [c.f64(), c.f64(), c.f64(), c.f64()]
        link = None
        data = self.chunk(uid, Chunk.GRAPHIC_LINK)
        if data is not None and len(data) >= 12:
            linked = self.link(self.cursor(data[8:]).u32())
            if linked is not None:
                link = linked[0]
        settings = MediaSettings()
        data = self.chunk(uid, Chunk.MEDIA_SETTINGS)
        if data is not None:
            try:
                settings, ok = _media_settings(self.enc(), data)
                if not ok:
                    self.warn(f"media {uid}: playback settings of "
                              f"unknown layout; strings left out")
            except InddError as exc:
                self.warn(f"media {uid}: playback settings left out: "
                          f"{exc}")
                settings = MediaSettings()
        poster = None
        for child in self.children(uid, Chunk.ITEM_HIERARCHY):
            graphic = self.graphic(child)
            if graphic is not None:
                poster = graphic
                break
        try:
            props = self.item_props(uid)
        except InddError as exc:
            self.warn(f"media {uid}: settings left out: {exc}")
            props = ItemProps()
        object_style = None
        data = self.chunk(uid, Chunk.ITEM_OBJECT_STYLE)
        if data is not None:
            object_style = uid_or_none(self.cursor(data).u32())
        attrs = None
        data = self.chunk(uid, Chunk.ITEM_ATTRS)
        if data is not None:
            attrs = self.attrs_or_warn(
                f"media {uid}",
                lambda: Attrs.parse(self.enc(), data))
        if attrs is None:
            attrs = Attrs()
        try:
            label = self.item_label(uid, cls)
        except InddError as exc:
            self.warn(f"media {uid}: label left out: {exc}")
            label = []
        return Media(
            uid=uid,
            kind=kind,
            transform=transform,
            paths=self.paths(uid),
            bounds=bounds,
            link=link,
            settings=settings,
            props=props,
            object_style=object_style,
            attrs=attrs,
            text_wrap=self.text_wrap(uid),
            label=label,
            poster=poster,
        )

    def index(self):
        if self.enc().big_endian:
            return None
        found = [u for u, c in self.db.classes
                 if c == IndexClass.INDEX]
        if not found:
            return None
        uid = found[0]
        if len(found) > 1:
            self.warn("more than one index; only the first is written")
        data = self.chunk(uid, IndexChunk.SECTIONS)
        if data is None:
            return Index(uid=uid, topics=[])
        c = self.cursor(data)
        c.u16()
        n = c.u16()
        sections = [c.u32() for _ in range(n)]
        topics = []
        for section_uid in sections:
            if self.class_of(section_uid) != IndexClass.SECTION:
                continue
            data = self.chunk(section_uid, IndexChunk.TOPICS)
            if data is not None:
                try:
                    topics.extend(_read_topics(self.cursor(data)))
                except InddError as exc:
                    self.warn(f"index section {section_uid} left out: "
                              f"{exc}")
        return Index(uid=uid, topics=topics)


    # -- Tables (src/model/table.rs) ----------------------------------

    def counted_attrs(self, uid, chunk_id):
        """An attribute list stored as a u16 count and records."""
        data = self.chunk(uid, chunk_id)
        if data is None or len(data) < 2:
            return Attrs()
        c = self.cursor(data)
        n = c.u16()
        return Attrs.parse_text(c, n)

    def table_style(self, uid, attrs_chunk):
        """A cell or table style: name and base from chunk 0x230 (as for
        text styles), attributes from ``attrs_chunk``."""
        data = self.chunk(uid, Chunk.STYLE_INFO)
        if data is None:
            return None
        based_on = self.cursor(data[min(4, len(data)):]).u32()
        found = find_flagged_string(self.enc(), data, 12,
                                    lambda n: n != "")
        if found is None:
            return None
        _at, builtin, name = found
        return TableStyle(
            uid=uid,
            name=name,
            builtin=builtin,
            based_on=uid_or_none(based_on),
            attrs=self.counted_attrs(uid, attrs_chunk),
        )

    def table_of_anchor(self, anchor):
        """The table owned by a table anchor object, without cell text."""
        data = self.chunk(anchor, TableChunk.ANCHOR_TABLE)
        if data is None:
            return None
        uid = self.cursor(data).u32()
        if self.class_of(uid) != TableClass.TABLE:
            return None
        return self.table(uid)

    def table(self, uid):
        size = self.required(uid, TableChunk.TABLE_SIZE)
        c = self.cursor(size)
        nrows = c.u32()
        ncols = c.u32()
        header_rows = c.u32()
        footer_rows = c.u32()
        column_groups = []
        data = self.chunk(uid, TableChunk.TABLE_COLUMNS)
        if data is not None:
            column_groups = _table_groups(self.enc(), data)
        row_groups = []
        data = self.chunk(uid, TableChunk.TABLE_ROWS)
        if data is not None:
            row_groups = _table_groups(self.enc(), data)
        # The cell data has a row of grid positions for every table row,
        # and a column count no larger than its widest row (every corpus
        # table). Larger counts are damage; checking them before
        # allocating keeps a damaged count from sizing the lists below.
        grid, formats = self.cell_grid(uid)
        widest = max((len(row) for row in grid), default=0)
        if nrows > len(grid) or ncols > widest:
            raise _corrupt(f"table {uid} has {nrows} rows and {ncols} "
                           f"columns, but its cells fill {len(grid)} rows "
                           f"and {widest} columns")
        columns = []
        for count, attrs in column_groups:
            width = attrs.get(TableAttr.COLUMN_WIDTH)
            width = width.as_f64() if width is not None else 0.0
            count = min(count, ncols - len(columns))
            columns.extend([width] * count)
        columns.extend([columns[-1] if columns else 0.0] *
                       (ncols - len(columns)))
        rows = []
        for count, attrs in row_groups:
            height = attrs.get(TableAttr.ROW_HEIGHT)
            min_height = attrs.get(TableAttr.ROW_MIN_HEIGHT)
            row = Row(
                height=height.as_f64() if height is not None else None,
                min_height=min_height.as_f64()
                if min_height is not None else None,
                attrs=attrs,
            )
            count = min(count, nrows - len(rows))
            rows.extend([row] * count)
        rows.extend([Row(height=None, min_height=None, attrs=Attrs())] *
                    (nrows - len(rows)))
        cells = _cells_from_grid(grid)
        # Cells are in the order of the positions that start them.
        contents = [pos for row in grid for pos in row
                    if pos["starts"] is not None]
        for cell, pos in zip(cells, contents, strict=False):
            if pos["content"] is not None:
                cell.items = self.graphic_cell_items(pos["content"])
        grid = [[pos["format"] for pos in row] for row in grid]
        style = None
        data = self.chunk(uid, TableChunk.TABLE_STYLE)
        if data is not None:
            style = uid_or_none(self.cursor(data).u32())
        attrs = self.counted_attrs(uid, TableChunk.TABLE_ATTRS)
        right_to_left = False
        data = self.chunk(uid, TableChunk.TABLE_DIRECTION)
        if data is not None:
            right_to_left = self.cursor(data).u16() == 1
        return Table(
            uid=uid,
            style=style,
            attrs=attrs,
            right_to_left=right_to_left,
            header_rows=header_rows,
            footer_rows=footer_rows,
            rows=rows,
            columns=columns,
            cells=cells,
            formats=formats,
            grid=grid,
        )

    def graphic_cell_items(self, content):
        """The page items of a graphic cell object (class 0x1046B),
        listed in its chunk 0x10466."""
        if self.class_of(content) != TableClass.GRAPHIC_CELL:
            return []
        items = []
        for child in self.children(content, TableChunk.GRAPHIC_CELL_ITEMS):
            item = self.page_item(child, None)
            if item is not None:
                items.append(item)
        return items

    def cell_grid(self, table):
        """Grid positions by row, from the cell strand's row groups, and
        the attribute sets they refer to."""
        parts = self.required(table, TableChunk.TABLE_PARTS)
        c = self.cursor(parts)
        n = c.u32()
        owner = None
        for _ in range(n):
            uid = c.u32()
            if c.u32() == TableClass.CELL_STRAND_OWNER:
                owner = uid
        grid = []
        all_formats = []
        if owner is None:
            return (grid, all_formats)
        data = self.chunk(owner, Chunk.STRAND_DATA)
        if data is None:
            return (grid, all_formats)
        lc = self.cursor(data)
        count = lc.u16()
        for _ in range(count):
            lc.u32()
            data = self.required(lc.u32(), Chunk.STRAND_RUNS)
            r = self.cursor(data)
            r.skip(8)
            groups = r.u16()
            for _ in range(groups):
                size = r.u32()
                g = self.cursor(r.bytes(size))
                # Attribute sets, each shared by a run of columns: u32
                # number of columns, u16 1 if attributes follow (u16
                # count, attributes), u32 cell style priority, u32 cell
                # style.
                sets = g.u32()
                formats = []
                for _ in range(sets):
                    columns = g.u32()
                    if g.u16() != 0:
                        na = g.u16()
                        attrs = Attrs.parse_text(g, na)
                    else:
                        attrs = Attrs()
                    style_priority = g.u32()
                    style = g.u32()
                    all_formats.append(CellFormat(
                        attrs=attrs,
                        style_priority=style_priority,
                        style=style,
                    ))
                    # A row of this group has at most one position per
                    # four bytes, so later formats would never be used.
                    room = max(0, size // 4 - len(formats))
                    formats.extend([len(all_formats) - 1] *
                                   min(columns, room))
                nr = g.u32()
                ids = []
                for _ in range(nr):
                    k = g.u32()
                    ids.append([g.u32() for _ in range(k)])
                nrec = g.u32()
                for i, row_ids in enumerate(ids):
                    if i >= nrec:
                        break
                    n = g.u16()
                    raw = _raw_attrs(g, n)
                    # A row record with fewer attribute records than
                    # positions leaves the last positions covered.
                    row = []
                    for k, cell_id in enumerate(row_ids):
                        if k < len(raw):
                            aid, values = raw[k]
                        else:
                            aid, values = (0, [])
                        if aid == TableAttr.CELL:
                            starts = CellKind.TEXT
                        elif aid == TableAttr.GRAPHIC_CELL:
                            starts = CellKind.GRAPHIC
                        else:
                            starts = None
                        width = None
                        if aid == TableAttr.CELL and len(values) > 1:
                            width = _layout_width(self.enc(), values[1])
                        content = None
                        if aid == TableAttr.GRAPHIC_CELL and values:
                            content = self.enc().u32_at(values[0], 70)
                        row.append({
                            "id": cell_id,
                            "starts": starts,
                            "width": width,
                            "content": content,
                            "format": formats[k]
                            if k < len(formats) else None,
                        })
                    grid.append(row)
        return (grid, all_formats)

    # -- Styles (src/model/style.rs) ----------------------------------

    def style(self, uid):
        data = self.chunk(uid, Chunk.STYLE_INFO)
        if data is None:
            return None
        c = self.cursor(data)
        next_uid = c.u32()
        based_on = c.u32()
        # The header before the name is 4 bytes shorter in files from
        # InDesign 13 and earlier, so locate the name by its structure:
        # a flag byte (1 = built-in name), then an in-object string.
        # Four bytes before the flag, a u16 is 1 for paragraph styles
        # and 0 for character styles; the u16 after it is not
        # identified.
        found = find_flagged_string(self.enc(), data, 12,
                                    lambda _: True)
        if found is None:
            raise _corrupt(f"style {uid}: no name")
        at, builtin, name = found
        paragraph = self.cursor(data[at - 4:]).u16() != 0
        imported = self.cursor(data[at - 2:]).u16() != 0
        shortcut = None
        if at >= 10:
            shortcut = (self.cursor(data[at - 10:]).u32(),
                        data[at - 6], data[at - 5])
        # A GUID in a 36-character in-object string after the name's
        # string. The byte after the tag can have any value.
        unique_id = None
        if not self.enc().big_endian:
            c = self.cursor(data[at:])
            try:
                c.name()
                end = at + c.pos
            except InddError:
                end = at
            for i in range(len(data) - end - 5):
                w = data[end + i:end + i + 6]
                if w[0] == 2 and w[2:] == bytes([36, 0, 36, 0x40]):
                    try:
                        guid = self.cursor(data[end + i:]).string()
                    except InddError:
                        guid = None
                    if guid is not None and _is_guid(guid):
                        unique_id = guid
                    break
        preview_color = None
        if at >= 14:
            code = self.enc().u32_at(data, at - 14)
            if code == 0:
                preview_color = False
            elif code is not None:
                rgb = self.ui_color(code)
                preview_color = rgb if rgb is not None else None
        attrs = Attrs()
        data = self.chunk(uid, Chunk.STYLE_ATTRS)
        if data is not None and len(data) >= 2:
            c = self.cursor(data)
            n = c.u16()
            parsed = self.attrs_or_warn(
                f"style {uid}",
                lambda: Attrs.parse_text(c, n))
            if parsed is not None:
                attrs = parsed
        export = self.style_export(uid)
        return Style(
            uid=uid,
            name=name,
            builtin=builtin,
            paragraph=paragraph,
            based_on=uid_or_none(based_on),
            next=uid_or_none(next_uid),
            attrs=attrs,
            imported=imported,
            unique_id=unique_id,
            shortcut=shortcut,
            export=export,
            preview_color=preview_color,
        )

    def toc_styles(self, version):
        """The table of contents styles, in UID order."""
        out = []
        for uid, cls in self.db.classes:
            if cls != Class.TOC_STYLE:
                continue
            try:
                data = self.chunk(uid, Chunk.TOC_STYLE)
                if data is None:
                    continue
                c = self.cursor(data)
                builtin = c.flag() == 1
                name = c.string()
                c.u16()
                include_book_documents = c.u16() == 1
                create_bookmarks = c.u32() == 1
                title_style = c.u32()
                c.flag()
                title = c.string()
                c.flag()
                c.string()
                # Two u16 before version 9, three before 13, four from
                # 13, then the entries.
                if version.major <= 8:
                    count = 2
                elif version.major <= 12:
                    count = 3
                else:
                    count = 4
                flags = []
                while len(flags) < count and c.remaining() >= 2:
                    flags.append(c.u16())
                entries = []
                if c.remaining() >= 2:
                    try:
                        entries = _toc_entries(c)
                    except InddError:
                        self.warn(f"TOC style {uid}: entries left out")
                        entries = []
                out.append(TocStyle(
                    uid=uid,
                    name=name,
                    builtin=builtin,
                    title=title,
                    title_style=title_style,
                    include_book_documents=include_book_documents,
                    create_bookmarks=create_bookmarks,
                    flags=flags,
                    entries=entries,
                ))
            except InddError as exc:
                self.warn(f"TOC style {uid} left out: {exc}")
        out.sort(key=lambda t: t.uid)
        return out

    def stroke_styles(self):
        """The custom stroke styles, in UID order. A style whose chunks
        do not have the lengths of the documented layout is left out
        with a warning."""
        out = []
        for uid, cls in self.db.classes:
            if cls == Class.DASHED_STROKE_STYLE:
                dashed = True
            elif cls == Class.STRIPED_STROKE_STYLE:
                dashed = False
            else:
                continue
            try:
                if dashed:
                    name_id, data_id = (Chunk.DASHED_STROKE_NAME,
                                        Chunk.DASHED_STROKE)
                else:
                    name_id, data_id = (Chunk.STRIPED_STROKE_NAME,
                                        Chunk.STRIPED_STROKE)
                n = self.chunk(uid, name_id)
                d = self.chunk(uid, data_id)
                if n is None or d is None:
                    continue
                c = self.cursor(n)
                c.flag()
                name = c.string()
                c = self.cursor(d)
                count = c.u32()
                values, extra = (count, 6) if dashed else (2 * count, 0)
                if values * 8 + 4 + extra != len(d):
                    continue
                parsed = [c.f64() for _ in range(values)]
                if dashed:
                    corner = c.u16()
                    cap = c.u32()
                else:
                    corner = 0
                    cap = 0
                out.append(StrokeStyle(
                    uid=uid,
                    dashed=dashed,
                    name=name,
                    values=parsed,
                    corner=corner,
                    cap=cap,
                ))
            except InddError:
                self.warn(f"stroke style {uid} left out")
        out.sort(key=lambda s: s.uid)
        return out

    def style_root_group(self, uid):
        """A root style group (the root of paragraph and character,
        object, cell or table styles): its kind (chunk 0x28C2) and
        children."""
        data = self.chunk(uid, Chunk.ROOT_GROUP_KIND)
        kind = self.cursor(data).u32() if data is not None else 0
        children = []
        for chunk_id in (Chunk.STYLE_ROOT_CHILDREN,
                         Chunk.OBJECT_STYLE_ROOT_CHILDREN,
                         TableChunk.CELL_STYLE_ROOT_CHILDREN,
                         TableChunk.TABLE_STYLE_ROOT_CHILDREN):
            if not children:
                children = self.children(uid, chunk_id)
        return StyleGroup(uid=uid, name="", root=kind,
                          children=children)

    def style_group(self, uid):
        """A named style group: a flag byte and the name, and its
        children."""
        name = ""
        data = self.chunk(uid, Chunk.STYLE_GROUP_NAME)
        if data is not None and len(data) > 1:
            c = self.cursor(data)
            c.flag()
            name = c.string()
        children = self.children(uid, Chunk.STYLE_GROUP_CHILDREN)
        return StyleGroup(uid=uid, name=name, root=None,
                          children=children)

    def style_export(self, uid):
        """The export settings of a style (chunk 0x28F0), with a warning
        when they do not parse."""
        data = self.chunk(uid, Chunk.STYLE_EXPORT)
        if data is None:
            return None
        try:
            return StyleExport.read(self.enc(), data)
        except InddError as exc:
            self.warn(f"style {uid}: export settings: {exc}; left out")
            return None

    def object_style(self, uid):
        """An object style. ``None`` if it has no info chunk (based-on
        style, flag byte and name)."""
        data = self.chunk(uid, Chunk.OBJECT_STYLE_INFO)
        if data is None:
            return None
        c = self.cursor(data)
        based_on = c.u32()
        builtin = c.flag() == 1
        name = c.string()
        # The u32 after the name: the keyboard shortcut key.
        try:
            shortcut_key = c.u32()
        except InddError:
            shortcut_key = None

        def u32_chunk(chunk_id):
            d = self.chunk(uid, chunk_id)
            if d is not None and len(d) >= 4:
                return self.cursor(d).u32()
            return None

        def short_attrs(chunk_id, what=None):
            d = self.chunk(uid, chunk_id)
            if d is None:
                return Attrs()
            message = f"object style {uid}"
            if what:
                message = f"{message}, {what}"
            parsed = self.attrs_or_warn(
                message,
                lambda: Attrs.parse_short(self.enc(), d))
            return parsed if parsed is not None else Attrs()

        frame = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_FRAME)
        if data is not None:
            if len(data) not in (106, 142, 162, 222):
                self.warn(f"object style {uid}: text frame settings of "
                          f"{len(data)} bytes are not known; left out")
            else:
                frame = ObjectStyleFrame.read(self.enc(), data)
        story = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_STORY)
        if data is not None:
            story = StorySettings.read(self.enc(), data)
        direction = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_DIRECTION)
        if data is not None and len(data) >= 2:
            direction = self.cursor(data).u16()
        flex_spacing = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_FLEX)
        if data is not None and len(data) == 78:
            values = [self.enc().f64_at(data, 30 + 8 * i)
                      for i in range(6)]
            if all(v is not None for v in values):
                flex_spacing = values
        contour = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_CONTOUR)
        if data is not None:
            contour = Contour.read(self.enc(), data)
        enabled = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_ENABLED)
        if data is not None:
            enabled = self.cursor(data).u32_list()
        anchor = None
        data = self.chunk(uid, Chunk.ANCHOR_SETTINGS)
        if data is not None:
            anchor = AnchorSettings.read(self.enc(), data)
        else:
            anchor = AnchorSettings.absent()
        try:
            export_options = self.export_options(uid)
        except InddError as exc:
            self.warn(f"object style {uid}: export options left out: "
                      f"{exc}")
            export_options = None
        apply_next = None
        data = self.chunk(uid, Chunk.OBJECT_STYLE_APPLY_NEXT)
        if data is not None:
            apply_next = self.enc().u16_at(data, 0)
        transform = [None, None, None, None]
        data = self.chunk(uid, Chunk.OBJECT_STYLE_TRANSFORM)
        if data is not None:
            transform = _transform_options(self.enc(), data)
        return ObjectStyle(
            uid=uid,
            name=name,
            builtin=builtin,
            based_on=uid_or_none(based_on),
            fitting=short_attrs(Chunk.OBJECT_STYLE_FITTING,
                                "frame fitting"),
            attrs=short_attrs(Chunk.OBJECT_STYLE_ATTRS),
            transparency=short_attrs(Chunk.OBJECT_STYLE_TRANSPARENCY,
                                     "transparency"),
            frame=frame,
            story=story,
            direction=direction,
            text_wrap=self.wrap_chunk(uid, Chunk.OBJECT_STYLE_WRAP),
            flex_spacing=flex_spacing,
            contour=contour,
            enabled=enabled,
            paragraph_style=u32_chunk(
                Chunk.OBJECT_STYLE_PARAGRAPH_STYLE),
            shortcut_key=shortcut_key,
            apply_next=apply_next,
            anchor=anchor,
            named_grid=self.applied_named_grid(uid),
            export=self.style_export(uid),
            export_options=export_options,
            transform=transform,
        )

    # -- Hyperlinks and story links (src/model/hyperlink.rs,
    #    src/model/story_link.rs) ------------------------------------

    def document_style_mappings(self):
        """Chunk 0x21D30 of the preferences object."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return StyleMappings()
        try:
            data = self.chunk(prefs_uid, Chunk.STYLE_MAPPINGS)
        except InddError:
            return StyleMappings()
        if data is None:
            return StyleMappings()
        try:
            return _style_mappings(self.cursor(data))
        except InddError as exc:
            self.warn(f"document style mappings left out: {exc}")
            return StyleMappings()

    def link_source_ids(self):
        """Document chunk 0x21D06: link source ID by story or page item
        UID."""
        out = {}
        data = self.chunk(DOC, Chunk.LINK_SOURCE_IDS)
        if data is None:
            return out
        c = self.cursor(data)
        c.skip(4)
        n = min(c.u32(), c.remaining() // 8)
        for _ in range(n):
            link_id = c.u32()
            out[c.u32()] = link_id
        return out

    def story_link_chunks(self, uid):
        """The link of a story, its options, style mappings and import
        options. A part that cannot be read is left out with a
        warning."""
        out = StoryLinkChunks()
        data = self.chunk(uid, Chunk.GRAPHIC_LINK)
        if data is not None and len(data) >= 12:
            c = self.cursor(data[4:])
            if c.u32() >= 1:
                link_uid = c.u32()
                linked = self.link(link_uid)
                if linked is not None:
                    out.link = linked[0]
                else:
                    self.warn(f"story {uid}: link {link_uid} cannot be "
                              f"read; left out")
        data = self.chunk(uid, Chunk.STORY_LINK_OPTIONS)
        if data is not None:
            value = self.enc().u32_at(data, 6)
            if value is not None:
                out.apply_style_mappings = value != 0
        data = self.chunk(uid, Chunk.STYLE_MAPPINGS)
        if data is not None:
            try:
                out.mappings = _style_mappings(self.cursor(data))
            except InddError as exc:
                self.warn(f"story {uid}: style mappings left out: {exc}")
        try:
            out.import_options = self.import_options(uid)
        except InddError as exc:
            self.warn(f"story {uid}: import options left out: {exc}")
            out.import_options = None
        data = self.chunk(uid, Chunk.STYLE_MAPPING_PREFS)
        if data is not None:
            try:
                out.mapping_prefs = self.mapping_prefs(data)
            except InddError as exc:
                self.warn(f"story {uid}: style mapping preferences "
                          f"left out: {exc}")
        return out

    def mapping_prefs(self, data):
        """Chunk 0x12216: u32 p, u32 c, then 2p and 2c names."""
        c = self.cursor(data)
        p = c.u32()
        n = c.u32()
        # A name takes at least a flag byte and 4 bytes of string
        # header.
        if p + n > c.remaining() // 10:
            raise _corrupt(f"{p} and {n} style mappings")

        def names(count):
            return [c.name().name for _ in range(2 * count)]

        return (names(p), names(n))

    def import_options(self, uid):
        """The import options object that story chunk 0x12202 names."""
        options = None
        data = self.chunk(uid, Chunk.STORY_IMPORT_OPTIONS)
        if data is not None:
            options = self.enc().u32_at(data, 0)
        if options is None or options == 0:
            return None
        cls = self.class_of(options)
        if cls == Class.WORD_IMPORT_OPTIONS:
            data = self.chunk(options, Chunk.WORD_IMPORT_OPTIONS)
            if data is None:
                return None
            c = self.cursor(data)
            return ("word", [c.u32() for _ in range(19)])
        if cls == Class.EXCEL_IMPORT_OPTIONS:
            data = self.chunk(options, Chunk.EXCEL_IMPORT_OPTIONS)
            if data is None:
                return None
            c = self.cursor(data)
            c.skip(16)
            table = c.u32()
            c.skip(4)
            sheet = c.name().idml()
            range_text = c.name().idml()
            view = c.name().idml()
            return ("excel", ExcelOptions(
                table=table, sheet=sheet, range=range_text, view=view))
        return None


    # -- Settings (src/model/settings.rs) -------------------------------

    def document_preferences(self):
        """Document preferences (chunk 0x533 of the preferences
        object). Only the page size, facing pages, bleeds, intent and
        page binding are kept."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return None
        data = self.chunk(prefs_uid, Chunk.DOCUMENT_PREFERENCES)
        if data is None:
            return None
        layout = _setup_layout_of(len(data))
        if layout is None:
            self.warn(f"document preferences of {len(data)} bytes are "
                      f"not known; left out")
            return None
        page, facing, binding, bleed, _bleed_uniform, _slug, \
            _slug_uniform, _intent = layout
        binding_code = self.cursor(data[binding:]).u16()
        if binding_code > 1:
            self.warn(f"document preferences: page binding code "
                      f"{binding_code} is not known; left out")

        def f(o):
            return self.cursor(data[o:]).f64()

        def u(o):
            return self.cursor(data[o:]).u32()

        intent = u(_intent)
        return DocumentPreferences(
            page_width=f(page),
            page_height=f(page + 8),
            facing_pages=data[facing] == 2,
            bleed=[f(bleed + 8), f(bleed + 24), f(bleed),
                   f(bleed + 16)],
            intent=intent if intent in (0, 1, 2) else None,
            page_binding=binding_code if binding_code <= 1 else None,
        )

    def users(self, doc):
        """The document's users, from chunk 0xA443 of the document."""
        data = self.chunk(doc, Chunk.DOC_USERS)
        if data is None:
            return []
        c = self.cursor(data)
        n = c.u32()
        if n > len(data) // 9:
            raise _corrupt(f"{n} document users")
        out = []
        for _ in range(n):
            flag = c.flag()
            name = c.string()
            c.u32()
            out.append((flag, name))
        return out

    def numbering_lists(self):
        """The numbering lists (preferences chunk 0x1A49B): the default
        list first, then the others in stored order. A list whose chunk
        cannot be read is left out with a warning."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return []
        data = self.chunk(prefs_uid, Chunk.NUMBERING_LISTS)
        if data is None:
            return []
        c = self.cursor(data)
        default = c.u32()
        n = c.u16()
        uids = [default]
        for _ in range(n):
            uids.append(c.u32())
        out = []
        for uid in uids:
            if self.class_of(uid) != Class.NUMBERING_LIST:
                continue
            data = self.chunk(uid, Chunk.NUMBERING_LIST_INFO)
            if data is None:
                continue
            c = self.cursor(data)
            try:
                name_obj = c.name()
                flags = c.bytes(3)
            except InddError as exc:
                self.warn(f"numbering list {uid} left out: {exc}")
                continue
            out.append(NumberingList(
                uid=uid,
                name=name_obj.idml(),
                builtin=name_obj.builtin,
                across_stories=flags[0] == 1,
                across_documents=flags[2] == 1,
            ))
        return out

    def label(self, uid):
        """The label of an object (chunk 0x1630B of the document): u16
        0x7B7B, u32 count, then key and value of each pair as a flag
        byte and an in-object string. IDML writes the strings as
        stored, whatever the flag."""
        data = self.chunk(uid, Chunk.DOC_LABEL)
        if data is None:
            return []
        return self.label_at(data, 0)

    def item_label(self, uid, cls):
        """The label of a page item: in chunk 0x6204 after the shape
        code (u16, u32) for frames and shapes, in chunk 0x2C0B for
        groups and in chunk 0x14582 for buttons and multi-state
        objects, with the layout of the document label. No label
        without the bytes."""
        if cls == Class.SPLINE_ITEM:
            chunk_id, at = Chunk.ITEM_SHAPE, 6
        elif cls == Class.GROUP:
            chunk_id, at = Chunk.GROUP_LABEL, 0
        elif cls in (Class.BUTTON, Class.MULTI_STATE):
            chunk_id, at = Chunk.FORM_LABEL, 0
        elif cls == Class.MOVIE:
            chunk_id, at = Chunk.MOVIE_LABEL, 0
        elif cls == Class.SOUND:
            chunk_id, at = Chunk.SOUND_LABEL, 0
        else:
            return []
        data = self.chunk(uid, chunk_id)
        if data is not None and len(data) > at:
            return self.label_at(data, at)
        return []

    def label_at(self, data, at):
        """A label from byte ``at`` of ``data``: u16 0x7B7B, u32 count,
        then flagged key and value strings."""
        c = self.cursor(data)
        c.skip(at)
        if c.u16() != 0x7B7B:
            raise _corrupt("label of unknown layout")
        n = c.u32()
        if n > len(data) // 8:
            raise _corrupt(f"{n} label pairs")
        return [(c.name().name, c.name().name) for _ in range(n)]

    def last_session(self, doc):
        """The code and application version string of the last record of
        the save history (chunk 0x1D8 of the document)."""
        record = self._last_record(doc)
        if record is None:
            return None
        _platform, code, version = record
        return (code, version)

    def last_session_platform(self, doc):
        """The platform (the second u16) of the last record of the save
        history: 0 or 3."""
        record = self._last_record(doc)
        if record is None:
            return None
        return record[0]

    def _last_record(self, doc):
        """Platform, code and version string of the last save history
        record."""
        try:
            data = self.chunk(doc, Chunk.DOC_HISTORY)
        except InddError:
            return None
        if data is None:
            return None
        c = self.cursor(data)
        try:
            n = c.u32()
        except InddError:
            return None
        # A record with an empty string has 25 bytes.
        if n == 0 or n > len(data) // 25:
            return None
        last = None
        for _ in range(n):
            try:
                c.u16()
                platform = c.u16()
                c.u16()
                c.u16()
                code = c.u16()
                c.flag()
                version = c.string()
                c.u16()
                c.u32()
                c.u32()
            except InddError:
                return None
            last = (platform, code, version)
        return last

    def users_script(self, doc):
        """The script byte of the last document user's name: the second
        byte of its in-object string. Only read in little-endian
        files."""
        if self.enc().big_endian:
            return None
        try:
            data = self.chunk(doc, Chunk.DOC_USERS)
        except InddError:
            return None
        if data is None:
            return None
        c = self.cursor(data)
        try:
            n = c.u32()
        except InddError:
            return None
        last = None
        for _ in range(n):
            try:
                c.flag()
                last = data[c.pos + 1] if c.pos + 1 < len(data) else None
                c.string()
                c.u32()
            except InddError:
                return None
        return last

    def index_groups(self):
        """The index sort groups of the preferences object, chunk
        0x1307E: name, include flag and header variant of each group,
        in stored order. Empty when the chunk does not parse to its
        end."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return []
        try:
            data = self.chunk(prefs_uid, Chunk.INDEX_GROUPS)
        except InddError:
            return []
        if data is None or self.enc().big_endian:
            return []
        try:
            return _index_groups(self.cursor(data))
        except InddError as exc:
            self.warn(f"index sort options left out: {exc}")
            return []

    def constant_shade(self):
        """The first pasted smooth shade (class 0x5533) whose chunk
        0x5532 holds a constant shade: 92 bytes, with u32 28 at offset
        60 and then a u32 and three f64."""
        uids = sorted(u for u, c in self.db.classes
                      if c == Class.SMOOTH_SHADE)
        for uid in uids:
            try:
                data = self.chunk(uid, Chunk.SMOOTH_SHADE)
            except InddError:
                continue
            if data is None:
                continue
            if len(data) != 92 or \
                    self.enc().u32_at(data, 60) != 28:
                continue
            count = self.enc().u32_at(data, 64)
            values = [self.enc().f64_at(data, o)
                      for o in (68, 76, 84)]
            if count is None or any(v is None for v in values):
                continue
            name = self._shade_name(uid)
            return ConstantShade(
                uid=uid,
                name=name[1] if name is not None else None,
                builtin=name[0] if name is not None else False,
                type_code=count,
                amount=values[0],
                values=values,
            )
        return None

    def _shade_name(self, uid):
        """The name of a pasted smooth shade (chunk 0x5531): flag and
        name."""
        try:
            data = self.chunk(uid, Chunk.SMOOTH_SHADE_NAME)
        except InddError:
            return None
        if data is None:
            return None
        c = self.cursor(data)
        try:
            builtin = c.flag() == 1
            return (builtin, c.string())
        except InddError:
            return None

    def axial_shades(self):
        """The pasted smooth shades (class 0x5533) whose chunk 0x5532
        holds an axial shading (type 5 at offset 56), in UID order. A
        shading whose data does not follow the known grammar is left
        out with a warning. The shading data is kept big-endian, as
        IDML ``Contents`` holds it."""
        uids = sorted(u for u, c in self.db.classes
                      if c == Class.SMOOTH_SHADE)
        out = []
        for uid in uids:
            try:
                data = self.chunk(uid, Chunk.SMOOTH_SHADE)
            except InddError:
                continue
            if data is None:
                continue
            if len(data) < 64 or \
                    self.enc().u32_at(data, 56) != 5:
                continue
            n = self.enc().u32_at(data, 60) or 0
            try:
                if len(data) != 64 + n:
                    raise _corrupt(f"{n} bytes of shading data in a "
                                   f"chunk of {len(data)}")
                c = self.cursor(data[4:])
                matrix = [c.f64() for _ in range(6)]
                converted = _axial_shading_be(self.enc(), data[64:])
            except InddError as exc:
                self.warn(f"pasted smooth shade {uid} left out: {exc}")
                continue
            name = self._shade_name(uid)
            out.append(AxialShade(
                uid=uid,
                name=name[1] if name is not None else None,
                builtin=name[0] if name is not None else False,
                type_code=5,
                amount=None,
                stops=None,
                angle=None,
                start=(matrix[4], matrix[5]),
                matrix=matrix,
                data=converted,
            ))
        return out

    def named_grids(self):
        """The named grids, in UID order: built-in key, name and the
        grid's own settings (chunk 0xCD02, which only grids made by
        the user have)."""
        out = []
        for uid, cls in self.db.classes:
            if cls != Class.NAMED_GRID:
                continue
            try:
                data = self.chunk(uid, Chunk.NAMED_GRID)
                if data is None:
                    continue
                c = self.cursor(data)
                c.u32()
                builtin = c.flag() == 1
                name = c.string()
                grid = None
                data = self.chunk(uid, Chunk.PAGE_GRID)
                if data is not None:
                    grid = GridData.read(self.cursor(data))
                out.append(NamedGrid(uid=uid, builtin=builtin,
                                     name=name, grid=grid))
            except InddError as exc:
                self.warn(f"named grid {uid} left out: {exc}")
        return out

    def applied_named_grid(self, uid):
        """The named grid applied to a story or object style (chunk
        0xCD32): ``("none",)`` for none (no chunk or u16 0),
        ``("uid", uid)`` for a grid, ``None`` for another layout code
        (no attribute is written)."""
        data = self.chunk(uid, Chunk.APPLIED_NAMED_GRID)
        if data is None:
            return ("none",)
        code = self.enc().u16_at(data, 0)
        if code == 0:
            return ("none",)
        if code == 1:
            return ("uid", self.enc().u32_at(data, 2))
        return None

    def language(self, uid):
        """A language object (class 0x2D07): its name, and the full
        language if the rest of chunk 0x2D0F can be read. ``None`` if
        it has no name. ``code`` is the u16 ID and ``hyphenation``
        the hyphenation vendor's flag."""
        data = self.chunk(uid, Chunk.LANGUAGE_NAME)
        if data is None or len(data) <= 1:
            return None
        c = self.cursor(data)
        builtin = c.flag() == 1
        name = c.string()
        # Then the primary and secondary names, u16 ID, and two
        # vendors (flag, u32, string).
        rest = None
        try:
            c.flag()
            primary = c.string()
            c.flag()
            sub = c.string()
            code = c.u16()

            def vendor():
                flag = c.u8()
                c.u32()
                return (flag, c.string())

            spelling = vendor()
            hyphenation = vendor()
            rest = (primary, sub, code, [spelling, hyphenation])
        except InddError:
            rest = None
        # Quotes (chunk 0x2D26): flag byte, the language name, then
        # the two single and the two double quotes as UTF-16 units.
        quotes = None
        qdata = self.chunk(uid, Chunk.LANGUAGE_QUOTES)
        if qdata is not None:
            try:
                qc = self.cursor(qdata)
                qc.flag()
                qc.string()

                def unit():
                    value = qc.u16()
                    ch = chr(value) if 0 <= value < 0x110000 else None
                    if ch is None:
                        raise InddError.corrupt("quote is not a character")
                    return ch

                quotes = (unit() + unit(), unit() + unit())
            except InddError:
                quotes = None
        language = None
        if rest is not None:
            primary, sub, code, vendors = rest
            language = Language(uid=uid, name=name, builtin=builtin,
                                code=code,
                                hyphenation=vendors[1][0],
                                primary=primary, sub=sub, id=code,
                                vendors=vendors, quotes=quotes)
        return (name, language)

    def bullets(self):
        """The bullet characters in the preferences object (chunk
        0x1A488): u16 1, u32 count, then the bullets."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return []
        data = self.chunk(prefs_uid, Chunk.BULLETS)
        if data is None:
            return []
        c = self.cursor(data)
        if c.u16() != 1:
            return []
        n = c.u32()
        out = []
        for i in range(n):
            kind = c.u32()  # 0 UnicodeOnly, 1 UnicodeWithFont, 2 GlyphWithFont
            value = c.u32()
            font = c.u32()
            font_style = c.name()
            # 0, or 1 and four bytes of unknown meaning.
            marker = c.u8()
            if marker == 0:
                pass
            elif marker == 1:
                c.skip(4)
            else:
                raise _corrupt(f"bullet font style followed by "
                               f"{marker} at {c.pos - 1}")
            character = ""
            if 0 <= value < 0x110000 and not (0xD800 <= value < 0xE000):
                character = chr(value)
            out.append(Bullet(
                uid=i,
                character=character,
                font=font,
                font_style=font_style,
                name="",
                builtin=False,
                kind=kind,
                value=value,
            ))
        return out

    def color_group_order(self):
        """The colour groups listed in the preferences object (chunk
        0x1F61)."""
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return []
        return self.uid_list(prefs_uid, ColorChunk.COLOR_GROUPS)

    # -- Preferences (src/model/prefs.rs) ------------------------------

    def prefs(self, version):
        """Preference values of the document. Only InDesign CS5 (7.0)
        and later in little-endian files are read; anything else gets
        the defaults. Any read error leaves the defaults with a
        warning, since ``document`` calls this unguarded."""
        major = version.major
        if major < 7 or self.enc().big_endian:
            return Prefs()
        prefs_uid = next((u for u, c in self.db.classes
                          if c == Class.PREFERENCES), None)
        if prefs_uid is None:
            return Prefs()
        try:
            return self._prefs(version, prefs_uid)
        except InddError as exc:
            self.warn(f"preferences left out: {exc}")
            return Prefs()

    def _prefs(self, version, prefs_uid):
        major = version.major
        minor = version.minor
        values = []
        colors = []
        text_defaults = None
        props = []

        def get(chunk_id):
            return self.chunk(prefs_uid, chunk_id)

        def set_value(element, name, value):
            values.append(PrefValue(element=element, name=name,
                                    value=value))

        def flag(chunk_id, absent):
            data = get(chunk_id)
            if data is None:
                return ("true" if absent else "false")
            if len(data) != 2:
                return None
            code = self.cursor(data).u16()
            if code == 0:
                return "false"
            if code == 1:
                return "true"
            return None

        def num(x):
            return idml_num(x)

        # The code of the last session of the save history (document
        # UID 1): 0x0101 for a Japanese or other CJK edition. Values
        # without a chunk follow it.
        session = self.last_session(1)
        cjk = session[0] == 0x0101 if session is not None else None
        # Colour settings.
        data = get(PrefsId.COLOR_POLICIES)
        if data is not None and len(data) == 12:
            rgb = _policy(data[4:8])
            if rgb is not None:
                set_value("Document", "RGBPolicy", rgb)
            cmyk = _policy(data[8:12])
            if cmyk is not None:
                set_value("Document", "CMYKPolicy", cmyk)
        data = get(PrefsId.COLOR_PROFILES)
        if data is not None:
            c = self.cursor(data)
            c.u16()
            n = c.u32()
            c.skip(4 * n)
            if c.u32() >= 2:
                for name in ("RGBProfile", "CMYKProfile"):
                    profile = c.u32()
                    if profile == 0:
                        value = "$ID/"
                    elif self.class_of(profile) == Class.COLOR_PROFILE:
                        s = self.chunk(profile, PrefsId.PROFILE_NAME)
                        if s is not None and len(s) > 1:
                            try:
                                value = self.cursor(s[1:]).string()
                            except InddError:
                                value = None
                        else:
                            value = None
                    else:
                        value = None
                    if value is not None:
                        set_value("Document", name, value)
        intent = None
        data = get(PrefsId.COLOR_INTENT)
        if data is None:
            intent = "UseColorSettings"
        elif len(data) >= 4:
            code = self.cursor(data[2:]).u16()
            intent = {0: "Perceptual", 2: "RelativeColorimetric",
                      11: "UseColorSettings"}.get(code)
        if intent is not None:
            for name in ("SolidColorIntent", "AfterBlendingIntent",
                         "DefaultImageIntent"):
                set_value("Document", name, intent)
        # View settings.
        data = get(PrefsId.VIEW)
        view_data = data
        if data is not None and len(data) == 56:
            c = self.cursor(data)
            units = [c.u32() for _ in range(6)]
            for i, names in enumerate([
                    ["HorizontalMeasurementUnits"],
                    ["VerticalMeasurementUnits"],
                    ["TypographicMeasurementUnits"],
                    ["TextSizeMeasurementUnits"],
                    ["PrintDialogMeasurementUnits"],
                    ["LineMeasurementUnits", "StrokeMeasurementUnits"]]):
                unit = _unit(units[i])
                if unit is not None:
                    for name in names:
                        set_value("ViewPreference", name, unit)
            for name in ("HorizontalCustomPoints",
                         "VerticalCustomPoints", "CursorKeyIncrement",
                         "PointsPerInch"):
                set_value("ViewPreference", name, num(c.f64()))
        for chunk_id, absent, element, name in [
                (PrefsId.SHOW_FRAME_EDGES, True, "ViewPreference",
                 "ShowFrameEdges"),
                (PrefsId.SHOW_RULERS, True, "ViewPreference",
                 "ShowRulers"),
                (PrefsId.BASELINE_GRID_SHOWN, False, "GridPreference",
                 "BaselineGridShown"),
                (PrefsId.GUIDES_LOCKED, False, "GuidePreference",
                 "GuidesLocked"),
                (PrefsId.COLUMN_GUIDES_LOCKED, True,
                 "DocumentPreference", "ColumnGuideLocked"),
                (PrefsId.SHOW_INVISIBLES, False, "TextPreference",
                 "ShowInvisibles"),
                (PrefsId.GRIDS_IN_BACK, True, "GridPreference",
                 "GridsInBack"),
                (PrefsId.DOCUMENT_GRID_SHOWN, False, "GridPreference",
                 "DocumentGridShown"),
                (PrefsId.STYLE_PREVIEW, False, "TextPreference",
                 "EnableStylePreviewMode"),
                (PrefsId.SHOW_ALL_FRAME_GRIDS, True,
                 "CjkGridPreference", "ShowAllFrameGrids"),
                (PrefsId.SHOW_CHARACTER_COUNT, True,
                 "CjkGridPreference", "ShowCharacterCount")]:
            value = flag(chunk_id, absent)
            if value is not None:
                set_value(element, name, value)
        # The highlight flags are `false` without their chunk, except
        # in a few documents that no stored value separates.
        for chunk_id, element, name in [
                (PrefsId.HIGHLIGHT_HJ_VIOLATIONS, "TextPreference",
                 "HighlightHjViolations"),
                (PrefsId.HIGHLIGHT_CUSTOM_SPACING, "TextPreference",
                 "HighlightCustomSpacing"),
                (PrefsId.HIGHLIGHT_SUBSTITUTED_GLYPHS, "TextPreference",
                 "HighlightSubstitutedGlyphs")]:
            value = flag(chunk_id, False)
            if value is not None:
                set_value(element, name, value)
        # Flags whose documents without the chunk do not have one
        # value.
        for chunk_id, element, name in [
                (PrefsId.SNAP_TO_LAYOUT_GRID, "CjkGridPreference",
                 "SnapToLayoutGrid"),
                (PrefsId.SHOW_ALL_LAYOUT_GRIDS, "CjkGridPreference",
                 "ShowAllLayoutGrids")]:
            if get(chunk_id) is not None:
                value = flag(chunk_id, False)
                if value is not None:
                    set_value(element, name, value)
        data = get(PrefsId.MOJIKUMI_UI)
        if data is not None:
            value = self.enc().u16_at(data, 0)
            if value is not None:
                set_value("MojikumiUiPreference", "MojikumiUiSettings",
                          str(value if value < 32768 else value - 65536))
        else:
            set_value("MojikumiUiPreference", "MojikumiUiSettings",
                      "16383")
        # Chunk 0x11C69 changes both values; which one it stores is
        # not known.
        articles = None
        data = get(PrefsId.ARTICLE_ORDER)
        if data is None:
            articles = False
        elif self.enc().u32_at(data, 0) == 1:
            articles = True
        if articles is not None:
            order = "UseArticles" if articles else "UseXMLStructure"
            set_value("TaggedPDFPreference", "StructureOrder", order)
            set_value("DictionaryPreference", "RecomposeWhenChanged",
                      "false" if articles else "true")
        data = get(PrefsId.XML_IMPORT)
        if data is not None:
            link = _xml_import_link(self.enc(), data)
            if link is not None:
                set_value("XMLImportPreference", "CreateLinkToXML",
                          "true" if link else "false")
        # Story settings of new text frames: u16 at 12 optical margin
        # alignment, u16 at 14 frame type, f64 at 2 optical margin
        # size.
        data = get(PrefsId.STORY)
        if data is not None and len(data) == 16:
            def u(o):
                return self.enc().u16_at(data, o)

            if u(12) == 0:
                set_value("StoryPreference", "OpticalMarginAlignment",
                          "false")
            elif u(12) == 1:
                set_value("StoryPreference", "OpticalMarginAlignment",
                          "true")
            if u(14) == 0:
                set_value("StoryPreference", "FrameType",
                          "TextFrameType")
            elif u(14) == 1:
                set_value("StoryPreference", "FrameType",
                          "FrameGridType")
            size = self.enc().f64_at(data, 2)
            if size is not None:
                set_value("StoryPreference", "OpticalMarginSize",
                          num(size))
        elif data is None:
            set_value("StoryPreference", "OpticalMarginAlignment",
                      "false")
            set_value("StoryPreference", "FrameType", "TextFrameType")
            # 13 Q where text sizes are in Q, 12 pt otherwise.
            q = None
            if view_data is not None and len(view_data) == 56:
                unit_code = self.enc().u32_at(view_data, 12)
                if unit_code is not None:
                    q = _unit(unit_code) == "Q"
            if q is True:
                set_value("StoryPreference", "OpticalMarginSize",
                          "9.2125984251969")
            elif q is False:
                set_value("StoryPreference", "OpticalMarginSize", "12")
        # Text wrap of new page items: u32 mode, four f64 offsets
        # (left, top, right, bottom), u32 1 where the wrap applies
        # everywhere.
        wrap = None
        data = get(PrefsId.ITEM_TEXT_WRAP)
        if data is not None and len(data) == 40:
            c = self.cursor(data)
            mode = c.u32()
            offsets = [c.f64(), c.f64(), c.f64(), c.f64()]
            everywhere = c.u32()
            wrap = (mode, offsets, everywhere)
        elif data is None:
            wrap = (0, [0.0, 0.0, 0.0, 0.0], 1)
        if wrap is not None:
            mode, offsets, everywhere = wrap
            left, top, right, bottom = offsets
            mode_name = {0: "None", 1: "JumpObjectTextWrap",
                         3: "BoundingBoxTextWrap",
                         6: "Contour"}.get(mode)
            if mode_name is not None:
                set_value("TextWrapPreference", "TextWrapMode",
                          mode_name)
            if everywhere == 0:
                set_value("TextWrapPreference", "ApplyToMasterPageOnly",
                          "true")
            elif everywhere == 1:
                set_value("TextWrapPreference", "ApplyToMasterPageOnly",
                          "false")
            props.append(("TextWrapPreference", "TextWrapOffset",
                          ("attrs", [("Top", num(top)),
                                     ("Left", num(left)),
                                     ("Bottom", num(bottom)),
                                     ("Right", num(right))])))
        # The story orientation of new text frames: byte 116 of the
        # text preferences.
        data = get(PrefsId.TEXT)
        if data is not None and len(data) > 116:
            if data[116] == 0:
                set_value("StoryPreference", "StoryOrientation",
                          "Horizontal")
            elif data[116] == 1:
                set_value("StoryPreference", "StoryOrientation",
                          "Vertical")
        # Document setup beyond `DocumentPreferences`.
        data = get(Chunk.DOCUMENT_PREFERENCES)
        if data is not None:
            layout = _setup_layout_of(len(data))
            if layout is not None:
                _page, _facing, _binding, bleed, bleed_uniform, slug, \
                    slug_uniform, _intent = layout

                def f(o):
                    return self.cursor(data[o:]).f64()

                set_value("DocumentPreference",
                          "DocumentBleedUniformSize",
                          "true" if data[bleed_uniform] != 0 else "false")
                set_value("DocumentPreference", "SlugInsideOrLeftOffset",
                          num(f(slug)))
                set_value("DocumentPreference", "SlugTopOffset",
                          num(f(slug + 8)))
                set_value("DocumentPreference",
                          "SlugRightOrOutsideOffset", num(f(slug + 16)))
                set_value("DocumentPreference", "SlugBottomOffset",
                          num(f(slug + 24)))
                set_value("DocumentPreference", "DocumentSlugUniformSize",
                          "true" if data[slug_uniform] != 0 else "false")
        # Text preferences and the default text frame columns.
        baseline_frame_grid_color = None
        data = get(PrefsId.TEXT)
        if data is not None and len(data) >= 174:
            # Bytes 168 and 170 are `UseCidMojikumi` and
            # `UseNewVerticalScaling` in an order not known; they are
            # equal in every sample.
            if data[168] == data[170] and data[168] <= 1:
                on = "true" if data[168] == 1 else "false"
                set_value("TextPreference", "UseCidMojikumi", on)
                set_value("TextPreference", "UseNewVerticalScaling", on)
            grid_data = get(PrefsId.BASELINE_FRAME_GRID)
            ui = self.enc().u32_at(grid_data, 20) \
                if grid_data is not None else None
            if ui is not None and ui != 0:
                rgb = self.ui_color(ui)
                if rgb is not None:
                    baseline_frame_grid_color = ("rgb", rgb)
            elif cjk is True:
                baseline_frame_grid_color = ("charcoal", None)
            elif cjk is False:
                baseline_frame_grid_color = ("light_blue", None)

            def f(o):
                return self.cursor(data[o:]).f64()

            def b(o):
                return "true" if data[o] != 0 else "false"

            for o, name in [(0, "SmallCap"), (8, "SuperscriptSize"),
                            (16, "SubscriptSize"),
                            (24, "SuperscriptPosition"),
                            (32, "SubscriptPosition"),
                            (88, "LeadingKeyIncrement"),
                            (96, "BaselineShiftKeyIncrement")]:
                set_value("TextPreference", name, num(f(o)))
            set_value("TextPreference", "KerningKeyIncrement",
                      num(f(104) * 1000.0))
            set_value("TextPreference", "TypographersQuotes", b(118))
            set_value("TextPreference", "LinkTextFilesWhenImporting",
                      b(130))
            set_value("TextPreference", "UseParagraphLeading", b(162))
            if len(data) >= 212:
                set_value("TextPreference",
                          "QuoteCharactersRotatedInVertical", b(184))
            if len(data) >= 214:
                set_value("TextPreference",
                          "ShapeIndicAndLatinWithHarbuzz", b(212))
            set_value("TextFramePreference", "TextColumnGutter",
                      num(f(40)))
            set_value("TextFramePreference", "TextColumnCount",
                      str(self.cursor(data[112:]).u32()))
            baseline = ["LeadingOffset", "AscentOffset", "CapHeight",
                        "EmboxHeight", "XHeight"]
            if data[142] < len(baseline):
                set_value("TextFramePreference", "FirstBaselineOffset",
                          baseline[data[142]])
        data = get(PrefsId.SMART_TEXT_REFLOW)
        if data is not None and len(data) >= 14:
            def b(o):
                return "true" if data[o] != 0 else "false"

            set_value("TextPreference", "SmartTextReflow", b(4))
            set_value("TextPreference", "LimitToMasterTextFrames", b(6))
            set_value("TextPreference", "DeleteEmptyPages", b(8))
            set_value("TextPreference", "PreserveFacingPageSpreads",
                      b(10))
            if len(data) >= 16:
                set_value("TextPreference", "SmartTextReflowSync", b(12))
        elif data is None:
            set_value("TextPreference", "SmartTextReflow", "false")
            set_value("TextPreference", "LimitToMasterTextFrames",
                      "true")
            set_value("TextPreference", "DeleteEmptyPages", "false")
            set_value("TextPreference", "PreserveFacingPageSpreads",
                      "false")
        # Default margins and columns of new pages.
        margins = None
        data = get(PrefsId.MARGINS)
        if data is not None and len(data) >= 32:
            def f(o):
                return self.cursor(data[o:]).f64()

            margins = [f(0), f(8), f(16), f(24)]
        elif data is None:
            margins = [36.0, 36.0, 36.0, 36.0]
        if margins is not None:
            left, top, right, bottom = margins
            set_value("MarginPreference", "Left", num(left))
            set_value("MarginPreference", "Top", num(top))
            set_value("MarginPreference", "Right", num(right))
            set_value("MarginPreference", "Bottom", num(bottom))
        columns = None
        data = get(PrefsId.COLUMNS)
        if data is not None and len(data) >= 12:
            columns = (self.cursor(data).u32(),
                       self.cursor(data[4:]).f64())
        elif data is None:
            # 5 mm after a CJK session, 12 pt otherwise.
            columns = (1, 14.173228346456694 if cjk is True else 12.0)
        if columns is not None:
            count, gutter = columns
            set_value("MarginPreference", "ColumnCount", str(count))
            set_value("MarginPreference", "ColumnGutter", num(gutter))
        # Column direction: u16 at 12 of the column chunk; without
        # it, the story orientation of new frames.
        direction = None
        data = get(PrefsId.COLUMNS)
        if data is not None and len(data) >= 14:
            code = self.enc().u16_at(data, 12)
            direction = code
        elif data is None:
            text = get(PrefsId.TEXT)
            if text is not None and len(text) > 116:
                direction = text[116]
        if direction == 0:
            direction_name = "Horizontal"
        elif direction == 1:
            direction_name = "Vertical"
        else:
            direction_name = None
        if direction_name is not None:
            set_value("DocumentPreference", "ColumnDirection",
                      direction_name)
            set_value("MarginPreference", "ColumnDirection",
                      direction_name)
        anchor_data = get(PrefsId.ANCHOR)
        # Pasteboard: f64 horizontal and vertical margins.
        pasteboard = None
        data = get(PrefsId.PASTEBOARD)
        if data is not None and len(data) >= 16:
            def f(o):
                return self.cursor(data[o:]).f64()

            pasteboard = (f(0), f(8))
        elif data is None:
            pasteboard = (-1.0, 72.0)
        if pasteboard is not None:
            h, v = pasteboard
            set_value("PasteboardPreference", "PasteboardMargins",
                      f"{num(h)} {num(v)}")
            set_value("PasteboardPreference",
                      "MinimumSpaceAboveAndBelow", num(v))
        # From DOM 9 the chunk has 34 bytes, byte 32 the theme colour
        # flag.
        data = get(PrefsId.PASTEBOARD)
        if data is not None and len(data) == 34:
            if data[32] == 0:
                set_value("PasteboardPreference",
                          "MatchPreviewBackgroundToThemeColor", "false")
            elif data[32] == 1:
                set_value("PasteboardPreference",
                          "MatchPreviewBackgroundToThemeColor", "true")
        elif data is None and major >= 9:
            set_value("PasteboardPreference",
                      "MatchPreviewBackgroundToThemeColor", "false")
        # Text wrap: u8 at 0, 2 and, in 6-byte chunks, 4.
        data = get(PrefsId.TEXT_WRAP)
        if data is not None and len(data) >= 3:
            set_value("TextPreference", "AbutTextToTextWrap",
                      "true" if data[0] != 0 else "false")
            set_value("TextPreference", "ZOrderTextWrap",
                      "true" if data[2] != 0 else "false")
            if len(data) >= 6:
                set_value("TextPreference",
                          "HonourTextIndentsWithTextWrap",
                          "true" if data[4] != 0 else "false")
        # Default XML tags: name (u32 length, text segments) and u32
        # UID of an interface colour, for story, table, (untagged),
        # cell, image.
        data = get(PrefsId.XML_TAGS)
        if data is not None:
            c = self.cursor(data)
            for name, color in [
                    ("DefaultStoryTagName", "DefaultStoryTagColor"),
                    ("DefaultTableTagName", "DefaultTableTagColor"),
                    (None, ""),
                    ("DefaultCellTagName", "DefaultCellTagColor"),
                    ("DefaultImageTagName", "DefaultImageTagColor")]:
                n = c.u32()
                tag = c.segments(n)
                ui = c.u32()
                if name is not None:
                    set_value("XMLPreference", name, tag)
                    rgb = self.ui_color(ui)
                    if rgb is not None:
                        colors.append(("XMLPreference", color, rgb))
        # Page item defaults: a table (u32, u32, u32 n, n 12-byte
        # entries), a page item attribute list, and a second table of
        # the same layout.
        item_default_entries = []
        item_defaults = None
        defaults_uid = next((u for u, c in self.db.classes
                             if c == Class.ITEM_DEFAULTS), None)
        if defaults_uid is not None:
            data = self.chunk(defaults_uid, PrefsId.ITEM_DEFAULTS)
            if data is not None:
                c = self.cursor(data)
                try:
                    def read_table(cursor):
                        cursor.skip(8)
                        n = cursor.u32()
                        for _ in range(n):
                            item_default_entries.append(
                                (cursor.u32(), cursor.u32(),
                                 cursor.u32()))

                    read_table(c)
                    item_defaults = self.attrs_or_warn(
                        f"page item defaults {defaults_uid}",
                        lambda: Attrs.parse_at(c))
                    if item_defaults is not None and c.remaining() > 0:
                        try:
                            read_table(c)
                        except InddError:
                            self.warn(f"page item defaults "
                                      f"{defaults_uid}: the second "
                                      f"table is not complete")
                except InddError:
                    item_defaults = None
        # The attributes of items that do not store them.
        item_base = None
        base_uid = next((u for u, c in self.db.classes
                         if c == Class.ITEM_BASE), None)
        if base_uid is not None:
            data = self.chunk(base_uid, Chunk.ITEM_ATTRS)
            if data is not None:
                item_base = self.attrs_or_warn(
                    f"page item attribute base {base_uid}",
                    lambda: Attrs.parse(self.enc(), data))
        # The swatches of chunk 0x6E06 of the page item defaults.
        item_default_swatches = []
        if defaults_uid is not None:
            data = self.chunk(defaults_uid,
                              PrefsId.ITEM_DEFAULT_SWATCHES)
            if data is not None and len(data) == 34:
                c = self.cursor(data[16:])
                for _ in range(3):
                    c.u16()
                    uid = c.u32()
                    if self.class_of(uid) is not None:
                        item_default_swatches.append(
                            (self.class_of(uid), uid))
        # Print settings.
        print_records = []
        for chunk_id, element in [(PrefsId.PRINT, "PrintPreference"),
                                  (PrefsId.PRINT_BOOKLET,
                                   "PrintBookletPrintPreference")]:
            data = get(chunk_id)
            if data is None:
                continue
            v = []
            p = []
            r = []
            try:
                _print_prefs(self.enc(), data, element, v, p, r)
            except InddError:
                continue
            for pv in v:
                set_value(pv.element, pv.name, pv.value)
            # Stored Japanese marks are `Default` unless the last
            # session was a CJK edition's.
            if cjk is False:
                for i, (el, name, value) in enumerate(p):
                    if name == "MarkType" and value[0] == "text" \
                            and value[2] == "JMarkWithCircle":
                        p[i] = (el, name, ("text", value[1], "Default"))
            props.extend(p)
            print_records.extend(r)
        # Booklet options: u32, a flagged string (not mapped), 4
        # bytes, then the four margins.
        booklet = None
        data = get(PrefsId.BOOKLET_OPTIONS)
        if data is not None:
            c = self.cursor(data)
            try:
                c.u32()
                _flagged(c)
                c.skip(4)
                booklet = [c.f64(), c.f64(), c.f64(), c.f64()]
            except InddError:
                booklet = None
        else:
            booklet = [36.0, 36.0, 36.0, 36.0]
        if booklet is not None:
            for name, value in zip(("TopMargin", "BottomMargin",
                                    "LeftMargin", "RightMargin"),
                                   booklet, strict=False):
                set_value("PrintBookletOption", name, num(value))
        # Default styles: u32 UIDs; the default is the third.
        default_styles = [None, None]
        for i, chunk_id in enumerate([PrefsId.DEFAULT_PARAGRAPH_STYLE,
                                      PrefsId.DEFAULT_CHARACTER_STYLE]):
            data = get(chunk_id)
            if data is not None:
                default_styles[i] = self.enc().u32_at(data, 8)
        default_object_styles = None
        data = get(PrefsId.DEFAULT_OBJECT_STYLES)
        if data is not None:
            def u(o):
                return self.enc().u32_at(data, o)

            trio = [u(8), u(12), u(16)]
            if all(v is not None for v in trio):
                default_object_styles = trio
        # Layout and story grids: as page chunk 0xCD02; the story grid
        # continues with u32 0 and f64 `CharacterCountSize`.
        grids = []
        for chunk_id, element in [(PrefsId.LAYOUT_GRID,
                                   "LayoutGridDataInformation"),
                                  (PrefsId.STORY_GRID,
                                   "StoryGridDataInformation")]:
            data = get(chunk_id)
            if data is None:
                continue
            c = self.cursor(data)
            try:
                grid = GridData.read(c)
                count = None
                if element == "StoryGridDataInformation":
                    c.u32()
                    count = c.f64()
                grids.append((element, grid, count))
            except InddError:
                continue
        # Master text frame: the chunk is there when it is on.
        master = "true" if get(PrefsId.MASTER_TEXT_FRAME) is not None \
            else "false"
        set_value("DocumentPreference", "MasterTextFrame", master)
        if major >= 8:
            set_value("DocumentPreference", "CreatePrimaryTextFrame",
                      master)
        data = get(PrefsId.BLENDING_SPACE)
        if data is None:
            set_value("TransparencyPreference", "BlendingSpace", "CMYK")
        else:
            code = self.enc().u32_at(data, 0)
            if code == 2:
                set_value("TransparencyPreference", "BlendingSpace",
                          "RGB")
            elif code == 3:
                set_value("TransparencyPreference", "BlendingSpace",
                          "CMYK")
        # Guides, grids, pasteboard: flags and interface colours.
        def bool_at(d, o):
            if o >= len(d):
                return None
            if d[o] == 0:
                return "false"
            if d[o] == 1:
                return "true"
            return None

        data = get(PrefsId.ALLOW_PAGE_SHUFFLE)
        if data is not None:
            value = bool_at(data, 0)
            if value is not None:
                set_value("DocumentPreference", "AllowPageShuffle",
                          value)
        data = get(PrefsId.GUIDES_SHOWN)
        if data is None:
            set_value("GuidePreference", "GuidesShown", "true")
        else:
            value = bool_at(data, 0)
            if value is not None:
                set_value("GuidePreference", "GuidesShown", value)
        data = get(PrefsId.SNAP)
        if data is None:
            set_value("GuidePreference", "GuidesSnapto", "true")
            set_value("GridPreference", "DocumentGridSnapto", "false")
        else:
            value = bool_at(data, 0)
            if value is not None:
                set_value("GuidePreference", "GuidesSnapto", value)
            value = bool_at(data, 2)
            if value is not None:
                set_value("GridPreference", "DocumentGridSnapto", value)

        def ui_color_push(element, name, uid):
            if uid is None:
                return
            rgb = self.ui_color(uid)
            if rgb is not None:
                colors.append((element, name, rgb))

        data = get(PrefsId.GUIDES)
        if data is not None:
            value = bool_at(data, 0)
            if value is not None:
                set_value("GuidePreference", "GuidesInBack", value)
            ui_color_push("GuidePreference", "RulerGuidesColor",
                          self.enc().u32_at(data, 18))
        data = get(PrefsId.BASELINE_GRID)
        if data is not None:
            ui_color_push("GridPreference", "BaselineColor",
                          self.enc().u32_at(data, 30))
            if len(data) > 34:
                if data[34] == 0:
                    set_value("GridPreference",
                              "BaselineGridRelativeOption",
                              "TopOfPageOfBaselineGridRelativeOption")
                elif data[34] == 1:
                    set_value("GridPreference",
                              "BaselineGridRelativeOption",
                              "TopOfMarginOfBaselineGridRelativeOption")
        for chunk_id, at, element, name, absent in [
                (PrefsId.DOCUMENT_GRID, 32, "GridPreference",
                 "GridColor", _UI_LIGHT_GRAY),
                (PrefsId.MARGINS, 36, "DocumentPreference",
                 "MarginGuideColor", _UI_MAGENTA),
                (PrefsId.COLUMNS, 18, "DocumentPreference",
                 "ColumnGuideColor",
                 _UI_LAVENDER if cjk is True else _UI_VIOLET)]:
            data = get(chunk_id)
            if data is not None:
                ui_color_push(element, name, self.enc().u32_at(data, at))
            elif absent is not None:
                colors.append((element, name, absent))
        data = get(PrefsId.PASTEBOARD)
        if data is not None:
            for at, name in [(20, "BleedGuideColor"),
                             (24, "SlugGuideColor"),
                             (28, "PreviewBackgroundColor")]:
                ui_color_push("PasteboardPreference", name,
                              self.enc().u32_at(data, at))
        else:
            for name, rgb in [("BleedGuideColor", _UI_FIESTA),
                              ("SlugGuideColor", _UI_GRID_BLUE),
                              ("PreviewBackgroundColor", _UI_LIGHT_GRAY)]:
                colors.append(("PasteboardPreference", name, rgb))
        data = get(PrefsId.ZERO_POINT)
        if data is None:
            set_value("Document", "ZeroPoint", "0 0")
        else:
            x = self.enc().f64_at(data, 0)
            y = self.enc().f64_at(data, 8)
            if x is not None and y is not None:
                set_value("Document", "ZeroPoint", f"{num(x)} {num(y)}")
        data = get(PrefsId.LAYOUT_ADJUSTMENT)
        if data is not None:
            value = bool_at(data, 0)
            if value is not None:
                set_value("LayoutAdjustmentPreference",
                          "EnableLayoutAdjustment", value)
            z = self.enc().f64_at(data, 12)
            if z is not None:
                set_value("LayoutAdjustmentPreference", "SnapZone",
                          num(z))
        else:
            set_value("LayoutAdjustmentPreference",
                      "EnableLayoutAdjustment", "false")
        if (major, minor) >= (21, 1):
            data = get(PrefsId.SHOW_TEXT_THREADS)
            shown = data is not None and \
                self.enc().u16_at(data, 0) == 1
            set_value("ViewPreference", "ShowTextThreads",
                      "true" if shown else "false")
        # Index header setting.
        data = get(PrefsId.INDEX_HEADER)
        if data is not None:
            try:
                v, header_list = _index_header(self.enc(), data)
                for pv in v:
                    set_value(pv.element, pv.name, pv.value)
                props.append(("IndexHeaderSetting",
                              "ListOfIndexHeaderGroup",
                              ("nodes", header_list)))
            except InddError as exc:
                self.warn(f"index header setting left out: {exc}")
        # Chapter numbering: u32 format code, u32 source, u32 number,
        # u16.
        chapter = get(PrefsId.CHAPTER_NUMBER)
        if chapter is None:
            props.append(("ChapterNumberPreference",
                          "ChapterNumberFormat",
                          ("text", "string", "1, 2, 3, 4...")))
            set_value("ChapterNumberPreference", "ChapterNumberSource",
                      "ContinueFromPreviousDocument")
            set_value("ChapterNumberPreference", "ChapterNumber", "1")
        if chapter is not None and len(chapter) >= 12:
            c = self.cursor(chapter)
            fmt, source, number = c.u32(), c.u32(), c.u32()
            fmt_name = _chapter_format(fmt)
            if fmt_name is not None:
                props.append(("ChapterNumberPreference",
                              "ChapterNumberFormat",
                              ("text", "string", fmt_name)))
            if source == 1:
                set_value("ChapterNumberPreference",
                          "ChapterNumberSource", "UserDefined")
            elif source == 3:
                set_value("ChapterNumberPreference",
                          "ChapterNumberSource",
                          "ContinueFromPreviousDocument")
            set_value("ChapterNumberPreference", "ChapterNumber",
                      str(number))
        # Dictionary: the chunk is there when composition uses the
        # document's dictionary.
        composition = "UseDocument" \
            if get(PrefsId.DICTIONARY) is not None else "Both"
        set_value("DictionaryPreference", "Composition", composition)
        # EPUB and HTML export, from DOM 8.
        if major >= 8:
            v = (major, minor)
            # Every IDML of DOM 8 and 9 has it.
            if major <= 9:
                set_value("EPubExportPreference",
                          "ViewDocumentAfterExport", "true")
            data = get(PrefsId.EPUB)
            if data is not None:
                exported = _epub_export(self.enc(), data)
                if exported is not None:
                    for name, value in exported:
                        if name in ("FootnotePlacement",
                                    "UseOriginalImageOnExport"):
                            since = (9, 2)
                        elif name in _EPUB_METADATA or \
                                name == "UseExistingImageOnExport":
                            since = (10, 0)
                        else:
                            since = (8, 0)
                        if v >= since:
                            set_value("EPubExportPreference", name,
                                      value)
                else:
                    # A layout the samples do not show: the version
                    # and the identifier, the string that starts
                    # `urn:uuid:`.
                    version_code = self.cursor(data).u32()
                    if version_code == 0:
                        set_value("EPubExportPreference", "Version",
                                  "Epub2")
                    elif version_code == 1:
                        set_value("EPubExportPreference", "Version",
                                  "Epub3")
                    found = find_flagged_string(
                        self.enc(), data, 4,
                        lambda s: s.startswith("urn:uuid:"))
                    if found is not None:
                        set_value("EPubExportPreference", "Id",
                                  found[2])
            else:
                epub3 = v >= (18, 1)
                epub_version = "Epub3" if epub3 else "Epub2"
                size = "SizeRelativeToTextFlow" if v >= (21, 1) \
                    else "SizeFixed"
                for name, value in [
                        ("Version", epub_version),
                        ("Id", _EPUB_ID),
                        ("TocStyleName", "$ID/"),
                        ("UseTocStyle", "false"),
                        ("ExportOrder", "LayoutOrder"),
                        ("EpubCover", "FirstPage"),
                        ("CoverImageFile", ""),
                        ("BreakDocument", "false"),
                        ("ParagraphStyleName",
                         "$ID/NormalParagraphStyle"),
                        ("EpubPublisher", ""),
                        ("PreserveLayoutAppearence", "true"),
                        ("ImageExportResolution", "Ppi150"),
                        ("CustomImageSizeOption", size),
                        ("EmbedFont", "true")]:
                    set_value("EPubExportPreference", name, value)
                if v >= (9, 2):
                    set_value("EPubExportPreference",
                              "FootnotePlacement", "FootnoteAfterStory")
                    set_value("EPubExportPreference",
                              "UseOriginalImageOnExport", "false")
                if v >= (10, 0):
                    for name in _EPUB_METADATA:
                        set_value("EPubExportPreference", name, "")
                    set_value("EPubExportPreference",
                              "UseExistingImageOnExport", "false")
            data = get(PrefsId.HTML)
            if data is not None:
                for name, value in _html_export(self.enc(), data):
                    set_value("HTMLExportPreference", name, value)
            else:
                for name, value in [
                        ("ExportOrder", "LayoutOrder"),
                        ("ViewDocumentAfterExport", "true"),
                        ("PreserveLayoutAppearence", "false"),
                        ("PreserveLocalOverride", "true")]:
                    set_value("HTMLExportPreference", name, value)
        # EPUB accessibility from INDD 19.3, page navigation from
        # 19.4. No sample has a value.
        v = (major, minor)
        if v >= (19, 3):
            for name in _EPUB_ACCESSIBILITY:
                set_value("EPubExportPreference", name, "")
                set_value("EPubFixedLayoutExportPreference", name, "")
        if v >= (19, 4):
            set_value("EPubExportPreference",
                      "EpubCreatePageNavigation", "true")
        # Fixed layout EPUB export, from DOM 10.
        if major >= 10:
            data = get(PrefsId.EPUB_FIXED_LAYOUT)
            if data is not None:
                exported = _fixed_layout_export(self.enc(), data)
                if exported is not None:
                    for name, value in exported:
                        set_value("EPubFixedLayoutExportPreference",
                                  name, value)
            else:
                for name, value in [
                        ("EpubCover", "FirstPage"),
                        ("CoverImageFile", ""),
                        ("TocStyleName", "$ID/"),
                        ("EpubPublisher", ""),
                        ("Id", _EPUB_ID),
                        ("EpubNavigationStyles", "NoNavigation")]:
                    set_value("EPubFixedLayoutExportPreference", name,
                              value)
                for name in _EPUB_METADATA:
                    set_value("EPubFixedLayoutExportPreference", name,
                              "")
        # Adjust layout (designmap), from DOM 14: u16 flags at 0, 4
        # and 24 of the 26-byte chunk. Without it the writer decides
        # `EnableAdjustLayout` by the save history.
        if major >= 14:
            data = get(PrefsId.ADJUST_LAYOUT)
            if data is not None and len(data) == 26:
                for o, name in [
                        (0, "EnableAdjustLayout"),
                        (4, "AllowFontSizeAndLeadingAdjustment"),
                        (24, "EnableAutoAdjustMargins")]:
                    code = self.enc().u16_at(data, o)
                    if code == 0:
                        set_value("AdjustLayoutPreference", name,
                                  "false")
                    elif code == 1:
                        set_value("AdjustLayoutPreference", name, "true")
        # Publish Online export (designmap), from DOM 11.
        if major >= 11:
            data = get(PrefsId.PUBLISH)
            if data is not None:
                exported = _publish_export(self.enc(), data)
                if exported is not None:
                    for name, value in exported:
                        set_value("PublishExportPreference", name,
                                  value)
            else:
                for name, value in [
                        ("PublishFileName", ""),
                        ("PublishDescription", ""),
                        ("PublishPageRange", ""),
                        ("CoverPage", "$ID/")]:
                    set_value("PublishExportPreference", name, value)
        # Grids.
        data = get(PrefsId.BASELINE_GRID)
        if data is not None and len(data) >= 26:
            def f(o):
                return self.cursor(data[o:]).f64()

            set_value("GridPreference", "BaselineDivision", num(f(2)))
            set_value("GridPreference", "BaselineStart", num(f(10)))
            set_value("GridPreference", "BaselineViewThreshold",
                      num(f(18) * 100.0))
        grid_values = None
        data = get(PrefsId.DOCUMENT_GRID)
        if data is not None and len(data) >= 32:
            def f(o):
                return self.cursor(data[o:]).f64()

            def u(o):
                return self.cursor(data[o:]).u32()

            grid_values = (f(8), u(16), f(20), u(28))
        elif data is not None:
            grid_values = (float("nan"), 0, float("nan"), 0)
        elif cjk is True:
            # 20 mm and 10 after a CJK session, 72 pt and 8
            # otherwise.
            grid_values = (56.69291338582678, 10, 56.69291338582678, 10)
        else:
            grid_values = (72.0, 8, 72.0, 8)
        if grid_values is not None:
            h, hs, vv, vs = grid_values
            if h == h:  # not NaN
                set_value("GridPreference", "HorizontalGridlineDivision",
                          num(h))
                set_value("GridPreference",
                          "HorizontalGridSubdivision", str(hs))
                set_value("GridPreference", "VerticalGridlineDivision",
                          num(vv))
                set_value("GridPreference", "VerticalGridSubdivision",
                          str(vs))
        # Watermark: u32 and u16 fields, then the font family and
        # style (u32 length, then text segments), the point size and
        # the UID of an interface colour.
        data = get(PrefsId.WATERMARK)
        if data is not None and len(data) > 10:
            c = self.cursor(data[10:])
            n = c.u32()
            family = c.segments(n)
            n = c.u32()
            style = c.segments(n)
            c.u32()
            color = c.u32()
            set_value("WatermarkPreference", "WatermarkFontFamily",
                      family)
            set_value("WatermarkPreference", "WatermarkFontStyle",
                      style)
            rgb = self.ui_color(color)
            if rgb is not None:
                colors.append(("WatermarkPreference",
                               "WatermarkFontColor", rgb))
        # Text defaults: a text attribute list, as styles have.
        data = get(Chunk.STYLE_ATTRS)
        if data is not None and len(data) >= 2:
            c = self.cursor(data)
            n = c.u16()
            text_defaults = self.attrs_or_warn(
                "text defaults",
                lambda: Attrs.parse_text(c, n))
        frame_column_rule = None
        data = get(Chunk.FRAME_COLUMN_RULE)
        if data is not None:
            width = self.enc().f64_at(data, 28)
            swatch = self.enc().u32_at(data, 36)
            if width is not None and swatch is not None:
                frame_column_rule = (width, swatch)
        frame_footnotes = None
        data = get(Chunk.FRAME_FOOTNOTES)
        if data is not None:
            span = self.enc().u16_at(data, 2)
            minimum = self.enc().f64_at(data, 4)
            space = self.enc().f64_at(data, 12)
            if span is not None and minimum is not None \
                    and space is not None:
                frame_footnotes = (span == 1, minimum, space)
        # u16 code; without the chunk (or with a code the samples do
        # not show) the origin is the spread.
        ruler_origin = RulerOrigin.SPREAD
        data = get(PrefsId.RULER_ORIGIN)
        if data is not None and len(data) == 2:
            code = self.cursor(data).u16()
            if code == 1:
                ruler_origin = RulerOrigin.PAGE
            elif code == 2:
                ruler_origin = RulerOrigin.SPINE
        footnotes = None
        data = get(PrefsId.FOOTNOTE_OPTIONS)
        if data is not None:
            try:
                parsed = FootnoteOptions.read(self.enc(), data)
                if parsed.tail is None:
                    self.warn(f"footnote options of {len(data)} bytes "
                              f"are not known; rule settings left out")
                footnotes = parsed
            except InddError as exc:
                self.warn(f"footnote options left out: {exc}")
                footnotes = None
        endnotes = None
        data = get(PrefsId.ENDNOTE_OPTIONS)
        if data is not None:
            try:
                endnotes = EndnoteOptions.read(self.enc(), data)
            except InddError as exc:
                self.warn(f"endnote options left out: {exc}")
                endnotes = None
        index_options = None
        data = get(PrefsId.INDEX_OPTIONS)
        if data is not None:
            try:
                index_options = IndexOptions.read(self.enc(), data)
            except InddError as exc:
                self.warn(f"index options left out: {exc}")
                index_options = None
        contour = None
        data = get(PrefsId.ITEM_CONTOUR)
        if data is not None:
            contour = Contour.read(self.enc(), data)
        if contour is None:
            contour = Contour.DEFAULT
        return Prefs(
            ruler_origin=ruler_origin,
            baseline_frame_grid_color=baseline_frame_grid_color,
            frame_column_rule=frame_column_rule,
            frame_footnotes=frame_footnotes,
            values=values,
            colors=colors,
            text_defaults=text_defaults,
            anchor=(AnchorSettings.read(self.enc(), anchor_data)
                    if anchor_data is not None
                    else AnchorSettings.absent()),
            item_defaults=item_defaults,
            item_base=item_base,
            item_default_entries=item_default_entries,
            item_default_swatches=item_default_swatches,
            props=props,
            print_records=print_records,
            footnotes=footnotes,
            endnotes=endnotes,
            index_options=index_options,
            default_styles=default_styles,
            contour=contour,
            default_object_styles=default_object_styles,
            grids=grids,
        )


def _source_ranges(enc, first_page, page_data):
    """The text ranges held in the range tree of a strand. Each node's
    start is stored relative to its parent: a left child starts
    ``value`` before its parent, a right child ``value`` after it; the
    root's value is its start."""
    nodes = {}
    page = first_page
    seen = []
    while page != 0 and page not in seen:
        seen.append(page)
        data = page_data(page)
        if data is None:
            break
        page = _read_range_page(enc, data, nodes)
    out = []
    roots = [k for k, n in nodes.items() if n["parent"] == 0]

    def bad():
        return _corrupt("range tree is inconsistent")

    for root in roots:
        stack = [(root, nodes[root]["value"])]
        while stack:
            key, pos = stack.pop()
            if len(out) > len(nodes):
                raise bad()
            node = nodes.get(key)
            if node is None or pos < 0:
                raise bad()
            out.append(SourceRange(start=pos, length=node["len"],
                                   source=node["source"], story=0,
                                   format=0, paragraph=False))
            if node["left"] != 0:
                left = nodes.get(node["left"])
                if left is None:
                    raise bad()
                stack.append((node["left"], pos - left["value"]))
            if node["right"] != 0:
                right = nodes.get(node["right"])
                if right is None:
                    raise bad()
                stack.append((node["right"], pos + right["value"]))
    out.sort(key=lambda r: r.start)
    return out


def _read_range_page(enc, data, nodes):
    """Read the nodes of one range tree page (chunk 0x16127) into
    ``nodes``, by node index. Returns the next page's UID (0 for
    none)."""
    c = enc.cursor(data)
    nxt = c.u32()
    c.skip(12)
    n = c.u32()
    c.skip(4 * n)
    count = c.u32()
    for _ in range(count):
        cls = c.u32()
        size = c.u32()
        body = c.bytes(size)
        if cls != HyperlinkClass.RANGE_NODE or size != 60:
            continue
        b = enc.cursor(body)
        b.skip(12)
        refs = []
        for _ in range(4):
            obj = b.u32()
            index = b.u32()
            refs.append(0 if obj == 0 else index)
        parent, me, left, right = refs
        nodes[me] = {"parent": parent, "left": left, "right": right,
                     "value": b.u32(), "len": b.u32(),
                     "source": b.u32()}
    return nxt

#: Sentinel for "no chunk": distinct from ``None`` ("chunk present,
#: value unknown"), used by ``frame_orientation``.
_ABSENT = object()


def _alpha_channel(enc, data):
    c = enc.cursor(data)
    try:
        if c.u32() == 0:
            return None
        c.flag()
        name = c.string()
    except InddError:
        return None
    return name if name else None


def _parse_preset_text(c):
    """The contents of a motion preset (chunk 0x20532): text as a
    flagged string. Upstream reads it with its ``text`` helper, which
    reads a counted string; a flagged string is the same shape here."""
    return c.name().name


def _state_group_name(kind, state_id):
    """The built-in name of a state's group that has no name of its
    own, by state ID (``docs/format/objects.md``, form fields)."""
    if kind == FormKind.MULTI_STATE:
        return None
    return {0: "$$$/StateType/Normal",
            1: "$$$/StateType/Over",
            2: "$$$/StateType/Down",
            3: "$$$/StateType/NormalOn",
            4: "$$$/StateType/NormalOff",
            6: "$$$/StateType/OverOff",
            7: "$$$/StateType/DownOn",
            8: "$$$/StateType/DownOff"}.get(state_id)


def _form_field(kind, states, attrs, children):
    """The settings of a form field from its attribute list, and the
    bounds of the items it shows (``docs/format/objects.md``, form
    fields)."""
    def text(attr_id):
        value = attrs.get(attr_id)
        if value is None:
            return None
        return value.as_string()

    def flag(attr_id):
        value = attrs.get(attr_id)
        if value is None:
            return None
        as_u32 = value.as_u32()
        return None if as_u32 is None else as_u32 != 0

    text_box = kind == FormKind.TEXT_BOX
    if FormKind.has_states(kind):
        shown = [item for state in states for item in state.items]
    else:
        shown = list(children)
    bounds = None
    for item in shown:
        item_bounds = _item_bounds(item, 0)
        if item_bounds is None:
            continue
        left, top, right, bottom = item_bounds
        if bounds is None:
            bounds = [left, top, right, bottom]
        else:
            bounds[0] = min(bounds[0], left)
            bounds[1] = min(bounds[1], top)
            bounds[2] = max(bounds[2], right)
            bounds[3] = max(bounds[3], bottom)
    font = None
    family = text(0x1455F)
    if family is not None and text_box:
        font = (family, text(0x14542))
    font_size = None
    value = attrs.get(0x14540)
    if value is not None and kind in (FormKind.TEXT_BOX,
                                      FormKind.COMBO_BOX):
        font_size = value.as_f64()
    return Form(
        kind=kind,
        states=states,
        bounds=bounds,
        name=text(0x14534),
        description=text(0x14535),
        export_value=text(0x14550) if kind in (FormKind.CHECK_BOX,
                                               FormKind.RADIO_BUTTON)
        else None,
        required=flag(0x1453D),
        multiline=flag(0x14552) if text_box else None,
        scrollable=flag(0x14553) if text_box else None,
        font_size=font_size,
        font=font,
        timing=[],
        behaviors=[],
    )


def _curve_extrema(p):
    """The parameters in (0, 1) where a cubic Bézier curve has a
    horizontal or vertical tangent."""
    out = []
    for k in range(2):
        def value(i, k=k):
            return p[i][0] if k == 0 else p[i][1]
        # The derivative divided by 3: a t² + b t + c.
        a = -value(0) + 3.0 * value(1) - 3.0 * value(2) + value(3)
        b = 2.0 * (value(0) - 2.0 * value(1) + value(2))
        c = value(1) - value(0)
        if abs(a) < 1e-12:
            if abs(b) > 1e-12:
                out.append(-c / b)
        else:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                root = disc ** 0.5
                out.append((-b + root) / (2.0 * a))
                out.append((-b - root) / (2.0 * a))
    return [t for t in out if 0.0 < t < 1.0]


def _cubic_point(p, t):
    u = 1.0 - t
    w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t]
    return (sum(w[i] * p[i][0] for i in range(4)),
            sum(w[i] * p[i][1] for i in range(4)))


def _item_bounds(item, depth):
    """The bounds of ``item`` in its parent's coordinates: the box
    around its paths (their curves, not only the anchor points) and,
    for a group or a form field, the boxes of the items it shows, with
    its corners through the item's transform."""
    if depth > MAX_ITEM_DEPTH:
        return None
    bounds = None

    def add(x, y):
        nonlocal bounds
        if bounds is None:
            bounds = [x, y, x, y]
        else:
            bounds[0] = min(bounds[0], x)
            bounds[1] = min(bounds[1], y)
            bounds[2] = max(bounds[2], x)
            bounds[3] = max(bounds[3], y)

    for path in item.paths:
        n = len(path.points)
        segments = n - 1 if path.open else n
        for point in path.points:
            add(*point.anchor)
        # The curve between two anchors can reach beyond them.
        for i in range(segments):
            p = path.points[i]
            q = path.points[(i + 1) % n]
            ctrl = [p.anchor, p.right, q.left, q.anchor]
            for t in _curve_extrema(ctrl):
                add(*_cubic_point(ctrl, t))
    # The items in a frame do not count; those of a group or a form
    # field do.
    inner = []
    if item.kind[0] == "group":
        inner = item.children
    elif item.kind[0] == "form":
        form = item.kind[1]
        if FormKind.has_states(form.kind):
            inner = [i for state in form.states for i in state.items]
        else:
            inner = item.children
    for child in inner:
        child_bounds = _item_bounds(child, depth + 1)
        if child_bounds is not None:
            left, top, right, bottom = child_bounds
            add(left, top)
            add(right, bottom)
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    a, b, c, d, tx, ty = item.transform.values
    def at(x, y):
        return (a * x + c * y + tx, b * x + d * y + ty)
    corners = [at(left, top), at(right, top), at(left, bottom),
               at(right, bottom)]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def _eps_text_data(enc, data):
    """The text record of EPS text (chunk 0x6611) as IDML
    ``EPSTextData`` writes it: in big-endian byte order, with the u32
    at 72 and at 78 of the 210-byte tail left out. Layout: u32
    version, flagged in-object string (font name), nine f64, flagged
    in-object string (text), tail."""
    c = enc.cursor(data)
    out = bytearray(c.u32().to_bytes(4, "big"))

    def string(cursor, out):
        if enc.big_endian:
            tag = cursor.u8()
            flag = cursor.u8()
        else:
            flag = cursor.u8()
            tag = cursor.u8()
        if tag != enc.string_tag:
            raise _corrupt(f"EPS text: string tag {tag}")
        script = cursor.u8()
        n = cursor.u16()
        out.extend([tag, flag, script])
        out.extend(n.to_bytes(2, "big"))
        left = n
        while left > 0:
            header = cursor.u16()
            count = header & 0x3FFF
            if count == 0 or count > left:
                raise _corrupt("EPS text: bad text segment")
            out.extend(header.to_bytes(2, "big"))
            if header & 0xC000 == 0x4000:
                out.extend(cursor.bytes(count))
            elif header & 0xC000 == 0x8000:
                for _ in range(count):
                    out.extend(cursor.u16().to_bytes(2, "big"))
            else:
                raise _corrupt("EPS text: bad text segment")
            left -= count

    string(c, out)
    for _ in range(9):
        out.extend(struct.pack(">d", c.f64()))
    string(c, out)
    if c.remaining() != 210:
        raise _corrupt(f"EPS text: tail of {c.remaining()} bytes, "
                       f"expected 210")
    tail = c.bytes(210)
    t = enc.cursor(tail)
    out.extend(t.u32().to_bytes(4, "big"))
    out.extend(t.bytes(68))
    t.skip(4)
    out.extend(t.u16().to_bytes(2, "big"))
    out.extend(t.bytes(132))
    return bytes(out)

def _cells_from_grid(grid):
    """The cells of the grid, each 1 x 1 until ``Table.resolve_spans``."""
    cells = []
    for r, row in enumerate(grid):
        for c, pos in enumerate(row):
            if pos["starts"] is None:
                continue
            cells.append(Cell(
                id=pos["id"],
                row=r,
                column=c,
                row_span=1,
                column_span=1,
                kind=pos["starts"],
                width=pos["width"],
                runs=[],
                items=[],
                format=pos["format"],
            ))
    return cells


def _toc_entries(c):
    """The entries of a TOC style chunk after its fields: u16 count,
    then per entry the style name (flag byte, string), a document path
    (flag byte, string), u32 format style, u16 level, u16, u16 page
    number position, u32 page number style, u32-counted separator text,
    u32 separator style, u16, a u32-counted text and a u16. The chunk
    must end after them."""
    n = c.u16()
    out = []
    for _ in range(n):
        builtin = c.flag() == 1
        name = c.string()
        c.flag()
        c.string()
        format_style = c.u32()
        level = c.u16()
        # 0 in every entry of the pairs; 1 in some entries of files
        # without one.
        c.u16()
        page_number_position = c.u16()
        page_number_style = c.u32()
        length = c.u32()
        separator = c.segments(length)
        separator_style = c.u32()
        c.u16()
        length = c.u32()
        c.segments(length)
        c.u16()
        out.append(TocEntry(
            name=name,
            builtin=builtin,
            format_style=format_style,
            level=level,
            page_number_position=page_number_position,
            page_number_style=page_number_style,
            separator=separator,
            separator_style=separator_style,
        ))
    if c.remaining() != 0:
        raise _corrupt("TOC entries do not fill the chunk")
    return out


def _transform_options(enc, data):
    """Chunk 0x1E234 of an object style: u16 flags at 8, 10, 12 and 14
    (1 when the style sets X, Y, height and width), and those values as
    f64 at 16, 24, 32 and 40."""
    out = []
    for i in range(4):
        if enc.u16_at(data, 8 + 2 * i) == 1:
            out.append(enc.f64_at(data, 16 + 8 * i))
        else:
            out.append(None)
    return out

def _index_groups(c):
    """The groups of chunk 0x1307E: for each, a flagged name, u8
    include, u8, u16 header variant (an index into the variant list),
    u16 and its header variants with their sections. Returns name,
    include and the name of the chosen variant (the first string of
    its record) of each group."""
    count = c.u32()
    out = []
    for _ in range(count):
        c.u8()
        name = c.string()
        include = c.u8() != 0
        c.u8()
        variant = c.u16()
        c.u16()
        chosen = None
        for i in range(c.u32()):
            c.u8()
            variant_name = c.string()
            if i == variant:
                chosen = variant_name
            c.u8()
            c.string()
            _counted_string(c)
            c.skip(2)
            for _ in range(c.u32()):
                _counted_string(c)
                _counted_string(c)
                c.u8()
                c.string()
                c.u16()
        out.append((name, include, chosen))
    if c.remaining() != 0:
        raise _corrupt(f"{c.remaining()} bytes left")
    return out


def _counted_string(c):
    """A u32 length in UTF-16 units, then text segments."""
    n = c.u32()
    if n == 0:
        return ""
    return c.segments(n)
