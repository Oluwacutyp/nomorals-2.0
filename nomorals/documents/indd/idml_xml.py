"""IDML package writing: the small XML writer, the stored-only ZIP
writer, and the text forms of IDML values.

A faithful port of ``src/idml/xml.rs``, ``src/idml/zip.rs`` and
``src/idml/format.rs`` from storytold/indd-utils (commit 136f973, dual
MIT/Apache-2.0 — see attribution.py). ``num`` is the model's
``idml_num`` (the same shortest-round-trip form, ported from
``format.rs::num``).
"""

from __future__ import annotations

import binascii
import io
import struct

from .idml_values import Node
from .model import civil_from_days, day_of_year
from .model import idml_num as num
from .object import builtin_key

__all__ = [
    "CDATA_SECTION",
    "Xml",
    "ZipWriter",
    "base64_lines",
    "escape_attr",
    "frame_grid_color",
    "link_time",
    "matrix",
    "num",
    "nums",
    "page_color",
    "prop_node",
    "round",
    "self_name",
    "set_property",
    "style_name",
    "style_path",
    "ui_color_name",
    "ui_color_property",
    "uref",
    "variable_name",
    "xml_char",
    "xml_tag_color",
]


def xml_char(c):
    """Whether XML 1.0 allows ``c`` (control characters aside): not
    U+FFFE or U+FFFF."""
    return c not in ("\uFFFE", "\uFFFF")


def escape_attr(s):
    out = []
    for ch in s:
        if ch == "&":
            out.append("&amp;")
        elif ch == "<":
            out.append("&lt;")
        elif ch == ">":
            out.append("&gt;")
        elif ch == '"':
            out.append("&quot;")
        elif ch == "\t":
            out.append("&#x9;")
        elif ch == "\n":
            out.append("&#xa;")
        elif ch == "\r":
            out.append("&#xd;")
        elif ord(ch) < 0x20 or not xml_char(ch):
            pass
        else:
            out.append(ch)
    return "".join(out)


class Xml:
    """A small XML writer that indents with tabs, like InDesign's IDML
    export. The attributes of an open start tag are kept until the tag
    is closed, so an element can never get an attribute twice: ``attr``
    sets a value (replacing an earlier one in place), and
    ``attrs_missing`` adds observed values only for attributes the
    element does not have. Values set with ``attr`` therefore take
    precedence over observed values whatever the order of the calls."""

    def __init__(self):
        self.out = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n']
        self.stack = []
        # The current element's start tag is still open (``<x a="b"``).
        self.open = False
        # The current element has text content, so its end tag is not
        # indented.
        self.inline = False
        # Attributes of the open start tag, in order, with their
        # escaped values.
        self.attrs = []

    def _flush_attrs(self):
        for k, v in self.attrs:
            self.out.append(f' {k}="{v}"')
        self.attrs = []

    def _close_start(self):
        if self.open:
            self._flush_attrs()
            self.out.append(">")
            self.open = False

    def _indent(self):
        self.out.append("\n")
        self.out.append("\t" * len(self.stack))

    def pi(self, text):
        """A processing instruction at the current position."""
        self._close_start()
        if not self.stack:
            self.out.append(f"<?{text}?>\n")
        else:
            self.out.append(f"<?{text}?>")
        return self

    def start(self, name):
        self._close_start()
        if self.stack and not self.inline:
            self._indent()
        self.out.append(f"<{name}")
        self.stack.append(name)
        self.open = True
        self.inline = False
        self.attrs = []
        return self

    def attr(self, name, value):
        """Set attribute ``name`` of the open start tag. An attribute
        the tag already has keeps its position and gets this value."""
        value = escape_attr(value)
        for i, (k, _) in enumerate(self.attrs):
            if k == name:
                self.attrs[i] = (name, value)
                return self
        self.attrs.append((name, value))
        return self

    def attr_opt(self, name, value):
        """Set attribute ``name`` when there is a value."""
        if value is not None:
            self.attr(name, value)
        return self

    def remove_attr(self, name):
        self.attrs = [(k, v) for k, v in self.attrs if k != name]
        return self

    def has_attr(self, name):
        return self.open and any(k == name for k, _ in self.attrs)

    def attrs_missing(self, attrs):
        """Add the attributes the open start tag does not have
        (observed values). A later ``attr`` with the same name replaces
        them."""
        for k, v in attrs:
            if not self.has_attr(k):
                self.attr(k, v)
        return self

    def text(self, s):
        """Text content. Characters below U+0020 other than tab and
        line feed become ``<?ACE n?>``, with the code in hex as IDML
        writes it (U+0018, the automatic page number, is
        ``<?ACE 18?>``)."""
        self._close_start()
        self.inline = True
        for ch in s:
            if ch == "&":
                self.out.append("&amp;")
            elif ch == "<":
                self.out.append("&lt;")
            elif ch == ">":
                self.out.append("&gt;")
            elif ch in ("\t", "\n"):
                self.out.append(ch)
            elif ord(ch) < 0x20:
                self.out.append(f"<?ACE {ord(ch):x}?>")
            elif not xml_char(ch):
                pass
            else:
                self.out.append(ch)
        return self

    def cdata(self, s, section):
        """Text content as CDATA sections of at most ``section``
        characters each. ``s`` must not contain ``]]>``."""
        self._close_start()
        self.inline = True
        rest = s
        while rest:
            n = min(section, len(rest))
            while n > 0 and (ord(rest[n - 1]) & 0xC0) == 0x80:
                n -= 1
            if n == 0:
                n = 1
            self.out.append(f"<![CDATA[{rest[:n]}]]>")
            rest = rest[n:]
        return self

    def end(self):
        """Close the current element."""
        if not self.stack:
            return self
        name = self.stack.pop()
        if self.open:
            self._flush_attrs()
            self.out.append(" />")
            self.open = False
        else:
            if not self.inline:
                self._indent()
            self.out.append(f"</{name}>")
        self.inline = False
        return self

    def empty(self, name, attrs):
        self.start(name)
        for k, v in attrs:
            self.attr(k, v)
        self.end()
        return self

    def finish(self):
        while self.stack:
            self.end()
        return "".join(self.out)


def _crc32(data):
    return binascii.crc32(data) & 0xFFFFFFFF


class ZipWriter:
    """A minimal ZIP writer: stored (uncompressed) entries only, which
    is all an IDML package needs. The first entry must be ``mimetype``,
    stored."""

    def __init__(self, out=None):
        self.out = out if out is not None else io.BytesIO()
        self.written = 0
        self.entries = []

    def _put(self, data):
        self.out.write(data)
        self.written += len(data)
        if self.written > 0xFFFFFFFF:
            raise ValueError("package larger than 4 GB")

    def add(self, name, data):
        crc = _crc32(data)
        size = len(data)
        if size > 0xFFFFFFFF:
            raise ValueError("entry too large")
        offset = self.written
        name_b = name.encode("utf-8")
        h = struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            20,  # version needed
            0x0800,  # UTF-8 names
            0,  # stored
            0,  # time
            0x21,  # date: 1980-01-01
            crc,
            size,
            size,
            len(name_b),
            0,
        )
        self._put(h + name_b)
        self._put(data)
        self.entries.append((name, crc, size, offset))

    def finish(self):
        start = self.written
        entries = self.entries
        self.entries = []
        for name, crc, size, offset in entries:
            name_b = name.encode("utf-8")
            h = struct.pack(
                "<IHHHHHHIIIHHHHHII",
                0x02014B50,
                20,  # made by
                20,  # needed
                0x0800,
                0,
                0,
                0x21,
                crc,
                size,
                size,
                len(name_b),
                0,  # extra
                0,  # comment
                0,  # disk
                0,  # internal attrs
                0,  # external attrs
                offset,
            )
            self._put(h + name_b)
        size = self.written - start
        end = struct.pack(
            "<IHHHHIIH",
            0x06054B50,
            0,
            0,
            len(entries),
            len(entries),
            size,
            start,
            0,
        )
        self._put(end)
        return self.out


# --- format.rs ---------------------------------------------------------

_UI_COLOR_NAMES = [
    ((0.31, 0.6, 1.0), "LightBlue"),
    ((1.0, 0.0, 0.0), "Red"),
    ((0.31, 1.0, 0.31), "Green"),
    ((0.0, 0.0, 1.0), "Blue"),
    ((1.0, 1.0, 0.31), "Yellow"),
    ((1.0, 0.31, 1.0), "Magenta"),
    ((0.0, 1.0, 1.0), "Cyan"),
    ((0.5, 0.5, 0.5), "Gray"),
    ((0.0, 0.0, 0.0), "Black"),
    ((0.6, 0.0, 0.0), "BrickRed"),
    ((1.0, 0.6, 0.0), "Gold"),
    ((1.0, 0.4, 0.0), "Orange"),
    ((0.0, 0.33, 0.0), "DarkGreen"),
    ((0.6, 0.6, 1.0), "Lavender"),
    ((0.67, 0.64, 0.71), "Charcoal"),
    ((0.6, 0.2, 1.0), "Violet"),
    ((1.0, 1.0, 1.0), "White"),
    ((1.0, 0.6, 0.8), "Pink"),
    ((0.61, 0.87, 0.61), "GridGreen"),
    ((0.0, 0.0, 0.53), "DarkBlue"),
    ((0.6, 0.8, 0.0), "GrassGreen"),
    ((0.81, 0.51, 0.71), "Lipstick"),
    ((1.0, 0.71, 0.42), "GridOrange"),
    ((0.97, 0.35, 0.42), "Fiesta"),
    ((0.0, 0.6, 0.6), "Teal"),
    ((0.73, 0.73, 0.73), "LightGray"),
    ((0.48, 0.73, 0.85), "GridBlue"),
    ((1.0, 0.6, 0.6), "Peach"),
    ((0.6, 0.0, 0.2), "Burgundy"),
]


def ui_color_name(rgb):
    """The IDML name of an interface colour (class 0x1F11) by its red,
    green and blue fractions."""
    for c, name in _UI_COLOR_NAMES:
        if all(abs(a - b) < 1e-4 for a, b in zip(c, rgb, strict=True)):
            return name
    return None


def xml_tag_color(rgb):
    """The IDML name of an XML tag colour (red, green, blue
    fractions); ``None`` for colours without evidence."""
    return ui_color_name(rgb)


def _ui_color_node(tag, name):
    return Node(tag=tag, attrs=[("type", "enumeration")], text=name)


def ui_color_property(tag, rgb):
    """A ``Properties`` child for an interface colour: its name as an
    enumeration, or, for a colour of whole 255ths without a name, the
    three numbers as a list. ``None`` for other colours."""
    name = ui_color_name(rgb)
    if name is not None:
        return _ui_color_node(tag, name)
    whole = [round(v * 255.0) for v in rgb]
    if any(abs(w / 255.0 - v) > 1e-6 for w, v in zip(whole, rgb,
                                                    strict=True)):
        return None
    node = Node(tag=tag, attrs=[("type", "list")])
    node.children = [
        Node(tag="ListItem", attrs=[("type", "double")], text=num(w))
        for w in whole
    ]
    return node


def page_color(c):
    """The ``PageColor`` property of a page colour. ``c`` is the
    ``(kind, rgb)`` tuple the model's ``ui_color_ref`` returns."""
    kind, rgb = c
    if kind == "use_master":
        return _ui_color_node("PageColor", "UseMasterColor")
    if kind == "nothing":
        return _ui_color_node("PageColor", "Nothing")
    if kind == "rgb":
        return ui_color_property("PageColor", rgb)
    return None


def frame_grid_color(c):
    """The ``BaselineFrameGridColor`` property of a baseline frame grid
    colour. ``c`` is the ``(kind, rgb)`` tuple the model's prefs
    reader stores."""
    if c is None:
        return None
    kind, rgb = c
    if kind == "rgb":
        return ui_color_property("BaselineFrameGridColor", rgb)
    name = {"charcoal": "Charcoal", "light_blue": "LightBlue"}.get(kind)
    if name is None:
        return None
    return _ui_color_node("BaselineFrameGridColor", name)


def set_property(node, prop):
    """Put ``property`` in the ``Properties`` of ``node``, in place of
    a property of the same name."""
    props = None
    for c in node.children:
        if c.tag == "Properties":
            props = c
            break
    if props is None:
        props = Node(tag="Properties")
        node.children.insert(0, props)
    for i, c in enumerate(props.children):
        if c.tag == prop.tag:
            props.children[i] = prop
            return
    props.children.append(prop)


def prop_node(p):
    """A ``Properties`` child as a values node."""
    name, ty, value = p
    n = Node(tag=name)
    if ty:
        n.attrs.append(("type", ty))
    kind = value[0]
    if kind == "text":
        n.text = value[1]
    elif kind == "attrs":
        n.attrs.extend((k, v) for k, v in value[1])
    elif kind == "list":
        for item in value[1]:
            li = Node(tag="ListItem", attrs=[("type", "record")])
            li.children = [
                Node(tag=f, attrs=[("type", t)], text=text)
                for f, t, text in item
            ]
            n.children.append(li)
    elif kind == "lists":
        for lst in value[1]:
            li = Node(tag="ListItem", attrs=[("type", "list")])
            li.children = [
                Node(tag="ListItem", attrs=[("type", t)], text=text)
                for t, text in lst
            ]
            n.children.append(li)
    return n


def link_time(filetime, dates):
    """A link time (FILETIME: 100 ns intervals since 1601-01-01 UTC)
    as IDML writes it, ``YYYY-MM-DDTHH:MM:SS`` in the local time of the
    computer that exported the IDML. The INDD does not store that time
    zone; the offset is taken from the XMP date nearest in day of the
    year (ties: the latest date), or UTC without XMP dates."""
    utc = (filetime // 10_000_000) - 11_644_473_600
    day = day_of_year(utc)
    best = None
    best_key = None
    for d in dates:
        distance = abs(day_of_year(d.utc) - day)
        # min_by: nearest day of year, ties to the latest date.
        key = (distance, -d.utc)
        if best_key is None or key < best_key:
            best_key = key
            best = d
    offset = best.offset if best is not None else 0
    local = utc + offset
    year, month, dnum = civil_from_days(local // 86_400)
    secs = local % 86_400
    return (f"{year:04}-{month:02}-{dnum:02}T"
            f"{secs // 3600:02}:{secs // 60 % 60:02}:{secs % 60:02}")


def round(v):
    """Round away binary noise from scaled values
    (0.8 * 100 = 80.00000000000001)."""
    import math
    return math.floor(v * 1e9 + 0.5) / 1e9 if v >= 0 else -math.floor(-v * 1e9 + 0.5) / 1e9


def nums(v):
    return " ".join(num(x) for x in v)


def matrix(m):
    return nums(m.values)


CDATA_SECTION = 262_144


def base64_lines(data):
    """Base64 (RFC 4648, with padding) in lines of 76 characters
    separated by line feeds, as IDML writes embedded file data."""
    import base64
    encoded = base64.b64encode(data).decode("ascii")
    lines = [encoded[i:i + 76] for i in range(0, len(encoded), 76)]
    return "\n".join(lines)


def uref(uid):
    return "n" if uid is None else f"u{uid:x}"


def style_name(s):
    """Name as used in a style's ``Self`` and in references: ``$ID/``
    for built-ins."""
    if s.builtin:
        return builtin_key(s.name)
    return s.name


def variable_name(name):
    """A text variable name as IDML writes it: control characters
    (U+001B in the built-in cross-reference variables) become
    ``<?AID 00xx?>``."""
    out = []
    for c in name:
        if ord(c) < 0x20:
            out.append(f"<?AID {ord(c):04x}?>")
        else:
            out.append(c)
    return "".join(out)


def style_path(names):
    """Names joined into the IDML name of a style or style group:
    ``:`` between them, and ``\\:`` for a ``:`` inside one."""
    return ":".join(n.replace(":", "\\:") for n in names)


def self_name(name):
    """Escape a name for use in a ``Self`` reference (``:`` separates
    style groups there)."""
    return name.replace("%", "%25").replace(":", "%3a").replace("\r", "%0d")


def write_properties(x, props, extra=()):
    """``<Properties>`` with ``props``, then the elements in
    ``extra``."""
    if not props and not extra:
        return
    x.start("Properties")
    for name, ty, value in props:
        x.start(name)
        if ty:
            x.attr("type", ty)
        kind = value[0]
        if kind == "attrs":
            for k, v in value[1]:
                x.attr(k, v)
        elif kind == "text":
            x.text(value[1])
        elif kind == "list":
            for item in value[1]:
                x.start("ListItem").attr("type", "record")
                for field, fty, text in item:
                    x.start(field).attr("type", fty).text(text).end()
                x.end()
        elif kind == "lists":
            for lst in value[1]:
                x.start("ListItem").attr("type", "list")
                for t, text in lst:
                    x.start("ListItem").attr("type", t).text(text).end()
                x.end()
        x.end()
    for n in extra:
        n.write(x)
    x.end()
