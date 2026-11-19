"""OOXML package layer for the DeckCraft port — stdlib only.

Reads and writes .pptx packages (ZIP + PresentationML) following
the upstream ``crates/pptx`` design (``opc.rs`` + ``xml.rs`` +
``read.rs``): the reader is lenient (unknown markup is skipped, bad
numbers fall back to defaults) and bounded (zip entry sizes, total
decompressed size, XML depth and element counts are capped), so
hostile input yields an error, never a blow-up.

This layer SHARES the word organ's XML seam instead of duplicating
it: ``parse_xml_bounded``, ``serialize_xml``, ``qname``, ``local``,
``children``, ``first_child`` and ``find_all`` are imported from
:mod:`nomorals.documents.word.ooxml` — the two organs differ only in
the package shape (``ppt/presentation.xml`` is the root part here,
not ``word/document.xml``) and the PresentationML namespaces.

Preservation-first: :class:`PptxPackage` keeps every part's raw
bytes; callers re-serialize only the parts they dirtied, so unedited
parts stay byte-identical in the output file.
"""

from __future__ import annotations

import contextlib
import io
import xml.etree.ElementTree as ET
import zipfile

from ..errors import DocumentError
from ..word.ooxml import (
    A_NS,
    CP_NS,
    CT_NS,
    DC_NS,
    DCT_NS,
    MAX_PACKAGE_BYTES,
    MAX_PART_BYTES,
    MAX_XML_DEPTH,
    MAX_XML_ELEMENTS,
    PR_NS,
    R_NS,
    children,
    find_all,
    first_child,
    local,
    parse_xml_bounded,
    qname,
    serialize_xml,
)

__all__ = [
    "A_NS",
    "C_NS",
    "CP_NS",
    "CT_NS",
    "DC_NS",
    "DCT_NS",
    "MAX_PACKAGE_BYTES",
    "MAX_PART_BYTES",
    "MAX_XML_DEPTH",
    "MAX_XML_ELEMENTS",
    "P_NS",
    "PIC_NS",
    "PR_NS",
    "PptxPackage",
    "R_NS",
    "children",
    "find_all",
    "first_child",
    "local",
    "parse_part",
    "parse_xml_bounded",
    "qname",
    "read_package",
    "serialize_xml",
    "write_package",
]

#: PresentationML main namespace.
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
#: DrawingML chart namespace.
C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
#: DrawingML picture namespace.
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"

#: Relationship types the organ resolves.
REL_OFFICE_DOCUMENT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/officeDocument")
REL_SLIDE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/slide")
REL_SLIDE_MASTER = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/slideMaster")
REL_SLIDE_LAYOUT = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/slideLayout")
REL_THEME = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/theme")
REL_NOTES_SLIDE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/notesSlide")
REL_IMAGE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/image")
REL_CHART = (
    "http://schemas.openxmlformats.org/officeDocument/2006/"
    "relationships/chart")

_PREFIXES = {
    "p": P_NS,
    "c": C_NS,
    "a": A_NS,
    "r": R_NS,
    "pic": PIC_NS,
}


def register_namespaces() -> None:
    """Register the PresentationML prefixes so re-serialization keeps
    them (ElementTree would otherwise invent ``ns0``/``ns1``). The
    relationships namespace is the default namespace — that is the
    standard spelling of ``.rels`` parts."""
    for prefix, uri in _PREFIXES.items():
        with contextlib.suppress(ValueError):
            ET.register_namespace(prefix, uri)
    with contextlib.suppress(ValueError):
        ET.register_namespace("", PR_NS)


register_namespaces()


class PptxPackage:
    """One .pptx package: ordered part name → raw bytes.

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


def read_package(data: bytes) -> PptxPackage:
    """Open .pptx bytes into a :class:`PptxPackage`.

    Bad zip, absurd sizes, and a missing ``ppt/presentation.xml``
    all raise :class:`DocumentError` (kind ``parse``).
    """
    if not data:
        raise DocumentError("empty .pptx file", kind="parse")
    if len(data) > MAX_PACKAGE_BYTES:
        raise DocumentError(
            f".pptx file of {len(data)} bytes exceeds the "
            f"{MAX_PACKAGE_BYTES} byte cap", kind="parse")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"not a .pptx (zip) file: {exc}",
                            kind="parse") from exc
    try:
        infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise DocumentError(f"damaged .pptx (zip) file: {exc}",
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
                "decompressed .pptx exceeds the "
                f"{MAX_PACKAGE_BYTES} byte cap", kind="parse")
        try:
            parts[info.filename] = zf.read(info.filename)
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise DocumentError(
                f"cannot read part {info.filename!r}: {exc}",
                kind="parse") from exc
    if "ppt/presentation.xml" not in parts:
        raise DocumentError(
            "not a .pptx file: ppt/presentation.xml is missing",
            kind="parse")
    return PptxPackage(parts)


def write_package(parts: dict[str, bytes]) -> bytes:
    """Serialize an ordered part map back into .pptx bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return buf.getvalue()


def parse_part(package: PptxPackage, name: str) -> ET.Element | None:
    """Parse one XML part, or None when the part is absent."""
    data = package.get(name)
    if data is None:
        return None
    return parse_xml_bounded(data, name)


def rels_part_for(part: str) -> str:
    """The ``.rels`` part name for a package part."""
    directory, _, name = part.rpartition("/")
    return f"{directory}/_rels/{name}.rels" if directory \
        else f"_rels/{name}.rels"


def read_rels(package: PptxPackage, part: str) -> dict[str, dict[str, str]]:
    """Relationships of ``part``: id → {type, target}.

    Targets resolve relative to the part's directory (``../media/x``
    from ``ppt/slides/`` becomes ``ppt/media/x``); external targets
    keep their raw form.
    """
    root = parse_part(package, rels_part_for(part))
    if root is None:
        return {}
    base = part.rpartition("/")[0]
    out: dict[str, dict[str, str]] = {}
    for rel in root:
        rid = rel.get("Id", "")
        if not rid:
            continue
        target = rel.get("Target", "")
        raw = target
        if target and not target.startswith(("http://", "https://",
                                            "mailto:", "/")):
            segments = (base.split("/") if base else []) \
                + target.split("/")
            resolved: list[str] = []
            for segment in segments:
                if segment in ("", "."):
                    continue
                if segment == "..":
                    if resolved:
                        resolved.pop()
                    continue
                resolved.append(segment)
            target = "/".join(resolved)
        out[rid] = {"type": rel.get("Type", ""), "target": target,
                    "raw": raw}
    return out


def next_rel_id(rels: dict[str, dict[str, str]]) -> str:
    """The next free ``rIdN`` for a relationships map."""
    taken = 0
    for rid in rels:
        if rid.startswith("rId"):
            try:
                taken = max(taken, int(rid[3:]))
            except ValueError:
                continue
    return f"rId{taken + 1}"


def write_rels(rels: dict[str, dict[str, str]]) -> bytes:
    """Serialize a relationships map to a ``.rels`` part.

    Targets are written back in their original relative form
    (``raw`` when the map came from :func:`read_rels`); writing the
    resolved package path would break every target on the next
    read, since rels resolve relative to their own part.
    """
    root = ET.Element(qname(PR_NS, "Relationships"))
    for rid in sorted(rels, key=lambda r: (len(r), r)):
        entry = rels[rid]
        child = ET.SubElement(root, qname(PR_NS, "Relationship"))
        child.set("Id", rid)
        child.set("Type", entry["type"])
        child.set("Target", entry.get("raw") or entry["target"])
        if entry.get("target_mode"):
            child.set("TargetMode", entry["target_mode"])
    return serialize_xml(root)
