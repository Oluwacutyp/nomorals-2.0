"""OOXML package layer for the WordCraft port — stdlib only.

Reads and writes .docx packages (ZIP + WordprocessingML) following the
upstream ``crates/docx`` design (``package.rs`` + ``xml.rs``): the
reader is lenient (unknown markup is skipped, bad numbers fall back to
defaults) and bounded (zip entry sizes, total decompressed size, XML
depth and element counts are capped), so hostile input yields an
error, never a blow-up. No third-party dependency: ``zipfile`` +
``xml.etree`` only — this also frees Devon's .docx reading from the
optional python-docx gate (see ``parsers._parse_docx``).

Preservation-first: :class:`DocxPackage` keeps every part's raw
bytes; callers re-serialize only the parts they dirtied, so unedited
parts stay byte-identical in the output file.
"""

from __future__ import annotations

import contextlib
import io
import xml.etree.ElementTree as ET
import zipfile

from ..errors import DocumentError

__all__ = [
    "DocxPackage",
    "MAX_PACKAGE_BYTES",
    "MAX_PART_BYTES",
    "MAX_XML_DEPTH",
    "MAX_XML_ELEMENTS",
    "W_NS",
    "XML_NS",
    "local",
    "parse_xml_bounded",
    "qname",
    "read_package",
    "register_namespaces",
    "serialize_xml",
    "write_package",
]

#: WordprocessingML main namespace.
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
#: Relationships namespace.
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
#: DrawingML wordprocessing-drawing namespace.
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
#: DrawingML main namespace.
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
#: DrawingML picture namespace.
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
#: Package content-types namespace.
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
#: Package relationships namespace.
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
#: Core-properties namespace.
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
#: Dublin Core elements.
DC_NS = "http://purl.org/dc/elements/1.1/"
#: Dublin Core terms.
DCT_NS = "http://purl.org/dc/terms/"
#: Extended properties namespace.
AP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
#: XML namespace (xml:space).
XML_NS = "http://www.w3.org/XML/1998/namespace"

#: Largest total decompressed size of a package (zip-bomb guard).
MAX_PACKAGE_BYTES = 64 * 1024 * 1024
#: Largest single part.
MAX_PART_BYTES = 32 * 1024 * 1024
#: Deepest XML nesting accepted.
MAX_XML_DEPTH = 100
#: Most elements in one XML part.
MAX_XML_ELEMENTS = 500_000

_PREFIXES = {
    "w": W_NS, "r": R_NS, "wp": WP_NS, "a": A_NS, "pic": PIC_NS,
    "w14": "http://schemas.microsoft.com/office/word/2010/wordml",
    "w15": "http://schemas.microsoft.com/office/word/2012/wordml",
    "w16": "http://schemas.microsoft.com/office/word/2018/wordml",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "wne": "http://schemas.microsoft.com/office/word/2006/wordml",
    "wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    "cp": CP_NS, "dc": DC_NS, "dcterms": DCT_NS,
}


def register_namespaces() -> None:
    """Register the OOXML prefixes so re-serialization keeps them.

    Without this, ElementTree invents ``ns0``/``ns1`` prefixes and
    untouched paragraphs would serialize differently.
    """
    for prefix, uri in _PREFIXES.items():
        with contextlib.suppress(ValueError):
            ET.register_namespace(prefix, uri)


register_namespaces()


def qname(ns: str, local_name: str) -> str:
    """``{namespace}local`` — ElementTree's qualified tag form."""
    return f"{{{ns}}}{local_name}"


def local(tag: str) -> str:
    """The local part of a qualified tag (``{ns}name`` → ``name``)."""
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[1]
    return tag


def parse_xml_bounded(data: bytes, what: str) -> ET.Element:
    """Parse one XML part with depth/element caps.

    Raises :class:`DocumentError` (kind ``parse``) on malformed XML
    or a cap breach — never a raw ``ParseError`` leaking out.
    """
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise DocumentError(f"malformed XML in {what}: {exc}",
                            kind="parse") from exc
    count = 0
    stack: list[tuple[ET.Element, int]] = [(root, 1)]
    while stack:
        element, depth = stack.pop()
        if depth > MAX_XML_DEPTH:
            raise DocumentError(
                f"{what} nests deeper than {MAX_XML_DEPTH} levels",
                kind="parse")
        count += 1
        if count > MAX_XML_ELEMENTS:
            raise DocumentError(
                f"{what} has more than {MAX_XML_ELEMENTS} elements",
                kind="parse")
        stack.extend((child, depth + 1) for child in element)
    return root


def serialize_xml(root: ET.Element) -> bytes:
    """Serialize an XML tree the way the reader canonicalizes it."""
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


class DocxPackage:
    """One .docx package: ordered part name → raw bytes.

    ``dirty`` names the parts whose bytes were re-serialized by an
    edit; every other part is written back byte-identical.
    """

    def __init__(self, parts: dict[str, bytes]) -> None:
        self.parts: dict[str, bytes] = dict(parts)
        self.dirty: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self.parts

    def get(self, name: str, default: bytes | None = None) -> bytes | None:
        return self.parts.get(name, default)

    def part_names(self) -> list[str]:
        return list(self.parts)

    def mark_dirty(self, name: str) -> None:
        self.dirty.add(name)

    def set_part(self, name: str, data: bytes) -> None:
        self.parts[name] = data
        self.dirty.add(name)

    def to_bytes(self) -> bytes:
        return write_package(self.parts)


def read_package(data: bytes) -> DocxPackage:
    """Open .docx bytes into a :class:`DocxPackage`.

    Bad zip, absurd sizes, and a missing ``word/document.xml`` all
    raise :class:`DocumentError` (kind ``parse``).
    """
    if not data:
        raise DocumentError("empty .docx file", kind="parse")
    if len(data) > MAX_PACKAGE_BYTES:
        raise DocumentError(
            f".docx file of {len(data)} bytes exceeds the "
            f"{MAX_PACKAGE_BYTES} byte cap", kind="parse")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"not a .docx (zip) file: {exc}",
                            kind="parse") from exc
    try:
        infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"damaged .docx (zip) file: {exc}",
                            kind="parse") from exc
    total = 0
    parts: dict[str, bytes] = {}
    for info in infos:
        if info.is_dir():
            continue
        if info.file_size > MAX_PART_BYTES:
            raise DocumentError(
                f"part {info.filename!r} of {info.file_size} bytes "
                f"exceeds the {MAX_PART_BYTES} byte cap", kind="parse")
        total += info.file_size
        if total > MAX_PACKAGE_BYTES:
            raise DocumentError(
                "decompressed .docx exceeds the "
                f"{MAX_PACKAGE_BYTES} byte cap", kind="parse")
        try:
            parts[info.filename] = zf.read(info.filename)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise DocumentError(
                f"cannot read part {info.filename!r}: {exc}",
                kind="parse") from exc
    if "word/document.xml" not in parts:
        raise DocumentError(
            "not a .docx file: word/document.xml is missing",
            kind="parse")
    return DocxPackage(parts)


def write_package(parts: dict[str, bytes]) -> bytes:
    """Serialize an ordered part map back into .docx bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return buf.getvalue()


def parse_part(package: DocxPackage, name: str) -> ET.Element | None:
    """Parse one XML part, or None when the part is absent."""
    data = package.get(name)
    if data is None:
        return None
    return parse_xml_bounded(data, name)


def children(element: ET.Element, tag_local: str) -> list[ET.Element]:
    """Direct children whose local tag name matches."""
    return [c for c in element if local(c.tag) == tag_local]


def first_child(element: ET.Element | None, tag_local: str
                ) -> ET.Element | None:
    """First direct child with the local tag name, or None.

    None-tolerant: a missing parent simply has no children.
    """
    if element is None:
        return None
    for child in element:
        if local(child.tag) == tag_local:
            return child
    return None


def find_all(element: ET.Element, tag_local: str) -> list[ET.Element]:
    """All descendants (in document order) with the local tag name."""
    return [e for e in element.iter() if local(e.tag) == tag_local]


def text_of(element: ET.Element) -> str:
    """The concatenated ``w:t`` text of an element (tabs/newlines kept)."""
    out: list[str] = []
    for node in element.iter():
        name = local(node.tag)
        if name == "t":
            out.append(node.text or "")
        elif name == "tab":
            out.append("\t")
        elif name in ("br", "cr"):
            out.append("\n")
    return "".join(out)
