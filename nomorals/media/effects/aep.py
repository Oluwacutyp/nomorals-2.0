"""After Effects project fixtures — RIFX structure probe.

Devon has no After Effects importer: the ``.aep`` / ``.aepx`` / ``.ffx``
files in ``tests/fixtures/effectcraft/`` are oracles, and this module
is the reader that exercises them. What it really does:

* validates the RIFX container — magic ``RIFX``, big-endian size,
  form type ``Egg!`` for projects and ``FaFX`` for presets — and
  walks the chunk tree inside the declared size, counting chunks
  and recording the top-level inventory. A truncated or garbled
  file fails with a named reason, never a silent partial read.
  Bytes past the declared end are reported as ``trailing_bytes``;
  on these fixtures the tail is the XMP metadata packet, which the
  probe checks starts with ``<?xpacket`` instead of guessing.
  One real After Effects shape: a LIST whose payload is not
  sub-chunks — ``btdk`` holds a PostScript blob (CoolType font
  data) — counts as one opaque leaf, not a parse failure.
* recovers the project's item names. For ``.aepx`` it parses the
  XML and reads the ``<Item><string>NAME</string>`` entries. For
  ``.aep`` / ``.ffx`` it reads every ``Utf8`` chunk payload and
  scans the RIFX region for printable UTF-8 strings, then checks
  the expected composition names (``docs/aep/CASES.md`` in the
  corpus) are among them.

What it does NOT do: full project import. Turning a project into
a Devon Composition needs the item / property stream decoding
(``tdb4`` / ``cdat`` payloads) — not built, and the probe says so
(``import_supported`` is False with the reason named) instead of
pretending a structure inventory is an import.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AepError",
    "Chunk",
    "ProjectProbe",
    "missing_compositions",
    "parse_aepx_items",
    "probe_file",
    "recover_strings",
]

#: Form types seen in the corpus (RIFX bytes 8..12).
FORM_AEP = "Egg!"
FORM_FFX = "FaFX"

#: Hard bounds: no fixture in the corpus comes near these.
MAX_FILE_SIZE = 64 * 1024 * 1024
MAX_CHUNKS = 200_000

_XMP_HEAD = b'<?xpacket'
_PRINTABLE = re.compile(rb"[\x20-\x7e]{3,}")


class AepError(Exception):
    """A fixture file that is not a readable After Effects project."""


class _Opaque(Exception):
    """A LIST whose payload is not sub-chunks (e.g. a PostScript blob)."""


@dataclass(frozen=True)
class Chunk:
    """One RIFX chunk header, as walked."""

    id: str
    size: int
    offset: int
    depth: int
    list_type: str = ""


@dataclass(frozen=True)
class ProjectProbe:
    """The structure recovered from one fixture file."""

    path: str
    kind: str  # "aep" | "aepx" | "ffx"
    form: str  # "Egg!" | "FaFX" | "" for aepx
    file_size: int
    declared_size: int  # RIFX-declared bytes after the size field
    trailing_bytes: int
    trailing_is_xmp: bool
    top_chunks: tuple[Chunk, ...]
    chunk_count: int
    max_depth: int
    names: tuple[str, ...] = ()
    items: tuple[str, ...] = ()
    item_count: int = 0  # <Item> elements in an .aepx (named or not)
    import_supported: bool = False
    import_refusal: str = (
        "aep_import_not_implemented: full project import needs the "
        "item/property stream decoding (tdb4/cdat payloads); this "
        "probe reads structure and names only"
    )


def recover_strings(data: bytes) -> list[str]:
    """Printable UTF-8 runs of length >= 3, in first-seen order."""
    seen: set[str] = set()
    found: list[str] = []
    for match in _PRINTABLE.finditer(data):
        text = match.group().decode("utf-8", "replace")
        if text not in seen:
            seen.add(text)
            found.append(text)
    return found


def _walk_region(data: bytes, start: int, end: int, depth: int,
                 *, what: str) -> list[Chunk]:
    """The chunk headers of one region, or raise :class:`_Opaque`.

    A region parses only when its headers consume it exactly (an
    odd-sized last chunk may overhang by its single pad byte).
    """
    out: list[Chunk] = []
    pos = start
    while pos + 8 <= end:
        cid = data[pos:pos + 4]
        if not all(32 <= b < 127 for b in cid):
            raise _Opaque(f"{what}: implausible chunk id {cid!r} at {pos}")
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if body + size > end:
            raise _Opaque(
                f"{what}: chunk {cid.decode()} at {pos} runs past its "
                f"region ({size} bytes, {end - body} left)"
            )
        list_type = ""
        if cid == b"LIST" and size >= 4:
            list_type = data[body:body + 4].decode("latin-1")
        out.append(Chunk(
            id=cid.decode("latin-1"),
            size=size,
            offset=pos,
            depth=depth,
            list_type=list_type,
        ))
        pos = body + size + (size % 2)
    if pos > end + 1:
        raise _Opaque(f"{what}: region overrun at {pos} (end {end})")
    if pos < end:
        raise _Opaque(f"{what}: {end - pos} unparsed bytes at {pos}")
    return out


def _walk_tree(data: bytes, *, what: str) -> tuple[list[Chunk], int, int]:
    """Walk the chunk tree inside the declared RIFX region.

    Returns (top-level chunks, total chunk count, max depth). A
    LIST whose payload is not sub-chunks — After Effects stores
    PostScript blobs (``btdk``) this way — counts as one opaque
    leaf; the container around it must still parse, so a truncated
    or garbled file fails instead of yielding a partial walk.
    """
    declared = struct.unpack(">I", data[4:8])[0]
    end = 8 + declared
    if end > len(data):
        raise AepError(
            f"{what}: truncated: RIFX declares {declared} bytes after "
            f"the size field, file holds {len(data) - 8}"
        )
    try:
        top = _walk_region(data, 12, end, 0, what=what)
    except _Opaque as exc:
        raise AepError(
            f"{what}: top-level chunk tree unreadable: {exc}"
        ) from exc
    total = len(top)
    max_depth = 0
    stack: list[tuple[Chunk, int]] = [
        (chunk, 0) for chunk in top if chunk.id == "LIST"
    ]
    while stack:
        chunk, depth = stack.pop()
        try:
            kids = _walk_region(
                data, chunk.offset + 12, chunk.offset + 8 + chunk.size,
                depth + 1, what=what,
            )
        except _Opaque:
            continue
        total += len(kids)
        if total > MAX_CHUNKS:
            raise AepError(f"{what}: more than {MAX_CHUNKS} chunks")
        max_depth = max(max_depth, depth + 1)
        stack.extend(
            (kid, depth + 1) for kid in kids if kid.id == "LIST"
        )
    return top, total, max_depth


def _structured_regions(data: bytes, *, what: str) -> list[tuple[int, int]]:
    """Every (start, end) region whose bytes parse as chunks."""
    declared = struct.unpack(">I", data[4:8])[0]
    end = 8 + declared
    try:
        top = _walk_region(data, 12, end, 0, what=what)
    except _Opaque as exc:
        raise AepError(
            f"{what}: top-level chunk tree unreadable: {exc}"
        ) from exc
    regions = [(12, end)]
    stack: list[Chunk] = [chunk for chunk in top if chunk.id == "LIST"]
    seen: set[tuple[int, int]] = set()
    while stack:
        chunk = stack.pop()
        region = (chunk.offset + 12, chunk.offset + 8 + chunk.size)
        if region in seen:
            continue
        seen.add(region)
        try:
            kids = _walk_region(
                data, region[0], region[1], 0, what=what)
        except _Opaque:
            continue
        regions.append(region)
        stack.extend(kid for kid in kids if kid.id == "LIST")
    return regions


def _utf8_payloads(data: bytes, *, what: str) -> list[str]:
    """Payloads of every ``Utf8`` chunk in the structured regions."""
    out: list[str] = []
    for start, region_end in _structured_regions(data, what=what):
        pos = start
        while pos + 8 <= region_end:
            cid = data[pos:pos + 4]
            size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
            body = pos + 8
            if cid == b"Utf8":
                out.append(
                    data[body:body + size].decode("utf-8", "replace")
                )
            pos = body + size + (size % 2)
    return out


def parse_aepx_items(data: bytes, *, what: str) -> tuple[list[str], int]:
    """Item names from an ``.aepx`` file, plus the <Item> count.

    Names come from ``<Item><string>NAME</string>``; the count
    covers every ``<Item>`` element, named or not.
    """
    try:
        root = ET.fromstring(data.decode("utf-8"))
    except (ET.ParseError, UnicodeDecodeError) as exc:
        raise AepError(f"{what}: not well-formed AEPX XML: {exc}") from exc
    ns = {"ae": "http://www.adobe.com/products/aftereffects"}
    items = root.findall(".//ae:Item", ns)
    names: list[str] = []
    for item in items:
        name = item.find("ae:string", ns)
        if name is not None and name.text:
            names.append(name.text)
    if not names:
        raise AepError(f"{what}: AEPX holds no <Item><string> entries")
    return names, len(items)


def _probe_rifx(path: str, data: bytes, *, kind: str,
                form: str) -> ProjectProbe:
    what = f"{kind} {Path(path).name}"
    if len(data) < 12 or data[0:4] != b"RIFX":
        raise AepError(f"{what}: not RIFX (magic is {data[:4]!r})")
    actual_form = data[8:12].decode("latin-1")
    if actual_form != form:
        raise AepError(
            f"{what}: form type is {actual_form!r}, expected {form!r}"
        )
    declared = struct.unpack(">I", data[4:8])[0]
    end = 8 + declared
    trailing = len(data) - end
    trailing_is_xmp = (
        trailing > 0 and data[end:end + len(_XMP_HEAD)] == _XMP_HEAD
    )
    top, total, max_depth = _walk_tree(data, what=what)
    names: list[str] = []
    seen: set[str] = set()
    for text in _utf8_payloads(data, what=what) + recover_strings(
        data[:end]
    ):
        if text not in seen:
            seen.add(text)
            names.append(text)
    return ProjectProbe(
        path=path,
        kind=kind,
        form=form,
        file_size=len(data),
        declared_size=declared,
        trailing_bytes=trailing,
        trailing_is_xmp=trailing_is_xmp,
        top_chunks=tuple(top),
        chunk_count=total,
        max_depth=max_depth,
        names=tuple(names),
    )


def _probe_aepx(path: str, data: bytes) -> ProjectProbe:
    what = f"aepx {Path(path).name}"
    if not data.lstrip().startswith(b"<?xml"):
        raise AepError(f"{what}: not XML (starts with {data[:16]!r})")
    items, item_count = parse_aepx_items(data, what=what)
    names: list[str] = []
    seen: set[str] = set()
    for text in items + recover_strings(data):
        if text not in seen:
            seen.add(text)
            names.append(text)
    return ProjectProbe(
        path=path,
        kind="aepx",
        form="",
        file_size=len(data),
        declared_size=0,
        trailing_bytes=0,
        trailing_is_xmp=False,
        top_chunks=(),
        chunk_count=0,
        max_depth=0,
        names=tuple(names),
        items=tuple(items),
        item_count=item_count,
    )


def probe_file(path: str | Path) -> ProjectProbe:
    """Probe one fixture file; fail closed with a named reason."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".aep":
        kind, form = "aep", FORM_AEP
    elif suffix == ".ffx":
        kind, form = "ffx", FORM_FFX
    elif suffix == ".aepx":
        kind, form = "aepx", ""
    else:
        raise AepError(f"{path.name}: unsupported fixture kind {suffix!r}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AepError(f"{path.name}: unreadable: {exc}") from exc
    if len(data) > MAX_FILE_SIZE:
        raise AepError(
            f"{path.name}: {len(data)} bytes exceeds the "
            f"{MAX_FILE_SIZE} byte bound"
        )
    if len(data) < 12:
        raise AepError(f"{path.name}: {len(data)} bytes, too small to read")
    if kind == "aepx":
        return _probe_aepx(str(path), data)
    return _probe_rifx(str(path), data, kind=kind, form=form)


def missing_compositions(probe: ProjectProbe,
                         expected: list[str]) -> list[str]:
    """Expected composition names the probe did not recover."""
    have = set(probe.names)
    return [name for name in expected if name not in have]
