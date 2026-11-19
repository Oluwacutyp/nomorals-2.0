"""OOXML package layer for the GridCraft port — stdlib only.

Reads and writes .xlsx packages (ZIP + SpreadsheetML) following the
upstream ``crates/xlsx`` design (a zip of XML parts, workbook +
worksheets + shared strings + styles): the reader is lenient
(unknown markup is skipped, bad numbers fall back to defaults) and
bounded (zip entry sizes, total decompressed size, XML depth and
element counts are capped), so hostile input yields an error, never
a blow-up. No third-party dependency: ``zipfile`` + ``xml.etree``
only — this also frees Devon's .xlsx reading from the optional
openpyxl gate (see ``datasci/workspace.py``).

Preservation-first: :class:`XlsxPackage` keeps every part's raw
bytes; callers re-serialize only the parts they dirtied, so unedited
parts stay byte-identical in the output file.

The bounded XML helpers (``parse_xml_bounded``, ``serialize_xml``,
``qname``, ``local``) and the size caps are shared with the word
organ (``documents.word.ooxml``) rather than duplicated — one seam
for OOXML parsing across both organs.
"""

from __future__ import annotations

import contextlib
import io
import xml.etree.ElementTree as ET
import zipfile

from ..errors import DocumentError
from ..word.ooxml import (
    MAX_PACKAGE_BYTES,
    MAX_PART_BYTES,
    local,
    parse_xml_bounded,
    qname,
    serialize_xml,
)

__all__ = [
    "MAX_PACKAGE_BYTES",
    "MAX_PART_BYTES",
    "XlsxPackage",
    "children",
    "first_child",
    "local",
    "parse_part",
    "parse_xml_bounded",
    "qname",
    "read_package",
    "register_namespaces",
    "serialize_xml",
    "write_package",
]

#: SpreadsheetML main namespace.
X_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
#: Relationships namespace.
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
#: Package relationships namespace.
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
#: Package content-types namespace.
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
#: Core-properties namespace.
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
#: Dublin Core elements.
DC_NS = "http://purl.org/dc/elements/1.1/"
#: Dublin Core terms.
DCT_NS = "http://purl.org/dc/terms/"
#: Extended properties namespace.
AP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"

_PREFIXES = {
    "x": X_NS, "r": R_NS, "pr": PR_NS, "ct": CT_NS,
    "cp": CP_NS, "dc": DC_NS, "dcterms": DCT_NS,
}


def register_namespaces() -> None:
    """Register the SpreadsheetML prefixes so re-serialization keeps them.

    Without this, ElementTree invents ``ns0``/``ns1`` prefixes and
    untouched rows would serialize differently.
    """
    for prefix, uri in _PREFIXES.items():
        with contextlib.suppress(ValueError):
            ET.register_namespace(prefix, uri)


register_namespaces()


class XlsxPackage:
    """One .xlsx package: ordered part name → raw bytes.

    ``dirty`` names the parts whose bytes were re-serialized by an
    edit; every other part is written back byte-identical.
    """

    def __init__(self, parts: dict[str, bytes]) -> None:
        self.parts: dict[str, bytes] = dict(parts)
        self.dirty: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self.parts

    def get(self, name: str, default: bytes | None = None
            ) -> bytes | None:
        return self.parts.get(name, default)

    def part_names(self) -> list[str]:
        return list(self.parts)

    def mark_dirty(self, name: str) -> None:
        self.dirty.add(name)

    def set_part(self, name: str, data: bytes) -> None:
        self.parts[name] = data
        self.dirty.add(name)

    def remove_part(self, name: str) -> None:
        self.parts.pop(name, None)
        self.dirty.discard(name)

    def to_bytes(self) -> bytes:
        return write_package(self.parts)


def read_package(data: bytes) -> XlsxPackage:
    """Open .xlsx bytes into an :class:`XlsxPackage`.

    Bad zip, absurd sizes, and a missing ``xl/workbook.xml`` all
    raise :class:`DocumentError` (kind ``parse``).
    """
    if not data:
        raise DocumentError("empty .xlsx file", kind="parse")
    if len(data) > MAX_PACKAGE_BYTES:
        raise DocumentError(
            f".xlsx file of {len(data)} bytes exceeds the "
            f"{MAX_PACKAGE_BYTES} byte cap", kind="parse")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"not a .xlsx (zip) file: {exc}",
                            kind="parse") from exc
    try:
        infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"damaged .xlsx (zip) file: {exc}",
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
                "decompressed .xlsx exceeds the "
                f"{MAX_PACKAGE_BYTES} byte cap", kind="parse")
        try:
            parts[info.filename] = zf.read(info.filename)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise DocumentError(
                f"cannot read part {info.filename!r}: {exc}",
                kind="parse") from exc
    if "xl/workbook.xml" not in parts:
        raise DocumentError(
            "not a .xlsx file: xl/workbook.xml is missing",
            kind="parse")
    return XlsxPackage(parts)


def write_package(parts: dict[str, bytes]) -> bytes:
    """Serialize an ordered part map back into .xlsx bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return buf.getvalue()


def parse_part(package: XlsxPackage, name: str) -> ET.Element | None:
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
