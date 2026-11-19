"""Convert InDesign files to IDML packages.

A faithful port of ``src/lib.rs`` (``convert``, ``convert_file``,
``convert_into``, ``read_header``) from storytold/indd-utils (commit
136f973, dual MIT/Apache-2.0 — see attribution.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .container import Container
from .errors import InddError
from .header import PROBE_LEN, Header
from .idml import write as write_idml
from .model import Reader


@dataclass
class Preview:
    """InDesign's screen preview of placed graphics whose file the
    document does not hold, such as linked images. IDML has no place
    for it, so it is not in the package; an application can show it
    when the linked file is missing."""

    #: The `Self` in the package of each graphic with this preview.
    graphics: list = field(default_factory=list)
    #: The preview file as stored: TIFF, JPEG, PNG or GIF.
    data: bytes = b""


@dataclass
class Conversion:
    """The result of a conversion: the IDML package, the warnings and
    the previews of linked graphics."""

    #: The IDML package (a ZIP file).
    idml: bytes = b""
    #: A problem that did not stop the conversion: content that was
    #: left out or could not be converted exactly.
    warnings: list = field(default_factory=list)
    #: InDesign's previews of the graphics whose file the document does
    #: not hold.
    previews: list = field(default_factory=list)


def read_header(source):
    """Read only the header of an INDD file: ``source`` is the file's
    bytes, or a path to it."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source[:PROBE_LEN])
    else:
        with open(source, "rb") as f:
            data = f.read(PROBE_LEN)
    return Header.parse(data)


def read_document(data):
    """Read INDD bytes into a model :class:`Document` without
    converting it."""
    doc, _db = _read(data)
    return doc


def _read(data):
    """Read INDD bytes into ``(Document, Database)``."""
    container = Container.parse(data)
    db = container.database()
    doc = Reader(db).document(container.header.version)
    # A damaged XMP packet only loses the time zone of link times and
    # the black name.
    try:
        packet = container.xmp()
    except InddError:
        packet = None
    if packet is not None:
        doc.read_xmp(packet)
    return doc, db


def convert(data, name):
    """Convert INDD bytes to an IDML package. ``name`` is the document
    name the package records (normally the INDD file name)."""
    idml, warnings, previews = _convert_with(data, name, True)
    return Conversion(idml=idml, warnings=warnings, previews=previews)


def convert_file(path):
    """Convert the INDD file at ``path``. The package records the file
    name as the document name."""
    with open(path, "rb") as f:
        data = f.read()
    name = os.path.basename(path)
    return convert(data, name)


def convert_into(data, name, out):
    """Convert INDD bytes to an IDML package written to ``out`` (a path
    or a binary file object), and return the warnings. The previews are
    not returned, and their data is not read."""
    idml, warnings, _previews = _convert_with(data, name, False)
    if isinstance(out, (str, os.PathLike)):
        with open(out, "wb") as f:
            f.write(idml)
    else:
        out.write(idml)
    return warnings


def _convert_with(data, name, want_previews):
    doc, db = _read(data)
    warnings = list(doc.warnings)
    idml, write_warnings = write_idml(doc, name)
    warnings.extend(write_warnings)
    previews = []
    if want_previews:
        previews = read_previews(db, doc, warnings)
    return idml, warnings, previews


def read_previews(db, doc, warnings):
    """The previews of the document's graphics, one per preview file.
    The graphics are those of every page item the package holds (on
    spreads and master spreads, anchored in text (cell text included),
    in graphic cells, in groups and in the states of form fields) and
    the posters of their movies and sounds."""
    graphics = {}
    items = []
    for s in list(doc.spreads) + list(doc.master_spreads):
        items.extend(s.items)
    for s in doc.stories:
        for anchors in s.anchors.values():
            items.extend(anchors)
        for t in s.tables.values():
            for c in t.cells:
                items.extend(c.items)
    while items:
        item = items.pop()
        for g in list(item.graphics) + [m.poster for m in item.media
                                        if m.poster is not None]:
            if g.preview is not None:
                graphics.setdefault(g.preview, []).append(f"u{g.uid:x}")
        items.extend(item.children)
        if item.kind[0] == "form":
            for state in item.kind[1].states:
                items.extend(state.items)
    out = []
    for uid, names in sorted(graphics.items()):
        try:
            data = db.object(uid)
        except InddError as exc:
            warnings.append(f"preview {uid}: left out: {exc}")
            continue
        if data is not None:
            out.append(Preview(graphics=names, data=data))
    return out
