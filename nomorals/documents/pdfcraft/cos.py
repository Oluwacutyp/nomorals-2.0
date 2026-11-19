"""COS layer — the PDF object model, lexer, xref reader, serializer.

Port of the *design* of storytold/pdfcraft ``crates/cos`` (``object.rs``,
``parser.rs``, ``writer.rs``, ``document/xref.rs``): one typed object per
PDF value, a lexer that reads objects from bytes, an xref reader that
follows trailer chains, and a serializer with the
``write_incremental`` / ``write_full`` split.  Devon's own code — no
upstream text copied; the shape (Object variants, Lexer, incremental vs
full save) is the adopted design.  Attribution for the upstream units
this derives from: ``nomorals/documents/pdfcraft-attribution.toml``.

What this is: the low-level layer.  It knows objects, xref tables,
trailers, streams and filters — nothing about pages, fonts, or edits.
The document model lives in :mod:`model`, edits in :mod:`edit`.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any

from ..errors import DocumentError

__all__ = [
    "PdfName", "PdfRef", "PdfString", "PdfStream",
    "XrefEntry", "Xref",
    "parse_object", "parse_indirect", "serialize", "serialize_indirect",
    "parse_xref", "resolve", "decode_stream",
    "build_incremental", "build_full",
]

# ── errors ──────────────────────────────────────────────────────────────


def _fail(message: str, *, kind: str = "parse") -> DocumentError:
    return DocumentError(message, kind=kind)


# ── object model (upstream crates/cos object.rs: Object) ─────────────────


class PdfName(str):
    """A PDF name, stored WITHOUT the leading slash (``/Type`` → ``"Type"``).

    ``#`` hex escapes are decoded on parse; serializing re-escapes every
    delimiter byte so the name round-trips exactly.
    """

    def __new__(cls, value: str = "") -> PdfName:
        return super().__new__(cls, value)


@dataclass(frozen=True)
class PdfRef:
    """An indirect reference: object number + generation (``12 0 R``)."""

    num: int
    gen: int = 0


class PdfString(bytes):
    """A PDF string's raw bytes (literal or hex — the distinction is not
    kept; bytes are bytes).  Decoded to text by callers (latin-1 for
    WinAnsi strings, UTF-16BE when the BOM says so)."""


@dataclass
class PdfStream:
    """A stream object: its dictionary plus its raw (still encoded)
    bytes.  ``dict`` keys are plain strings with the slash (``"/Length"``),
    matching the rest of this layer."""

    dict: dict[str, Any]
    data: bytes


# A parsed PDF value: None | bool | int | float | PdfName | PdfString |
# list | dict[str, Any] | PdfRef | PdfStream.
PdfObject = Any

_DELIMITERS = b"()<>[]{}/%"
_WS = b"\x00\t\n\x0c\r "


def _is_ws(byte: int) -> bool:
    return byte in _WS


def _is_delimiter(byte: int) -> bool:
    return byte in _DELIMITERS


# ── lexer (upstream crates/cos parser.rs: Lexer) ─────────────────────────


class _Lexer:
    """Byte cursor that reads one COS object at a time."""

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def skip_ws(self) -> None:
        data, n = self.data, len(self.data)
        while self.pos < n:
            byte = data[self.pos]
            if _is_ws(byte):
                self.pos += 1
            elif byte == 0x25:  # '%' — comment to end of line
                while self.pos < n and data[self.pos] not in (0x0A, 0x0D):
                    self.pos += 1
            else:
                return

    def peek(self) -> bytes:
        self.skip_ws()
        if self.pos >= len(self.data):
            return b""
        return self.data[self.pos:self.pos + 1]

    def _token(self) -> bytes:
        self.skip_ws()
        start = self.pos
        while self.pos < len(self.data):
            byte = self.data[self.pos]
            if _is_ws(byte) or _is_delimiter(byte):
                break
            self.pos += 1
        return self.data[start:self.pos]

    def _read_literal_string(self) -> PdfString:
        # self.pos is at '('; nesting counts, escapes decoded.
        data, n = self.data, len(self.data)
        self.pos += 1
        out = bytearray()
        depth = 1
        while self.pos < n:
            byte = data[self.pos]
            if byte == 0x5C:  # backslash
                self.pos += 1
                if self.pos >= n:
                    break
                nxt = data[self.pos]
                if nxt in (0x0A, 0x0D):  # line continuation
                    self.pos += 1
                    if nxt == 0x0D and self.pos < n \
                            and data[self.pos] == 0x0A:
                        self.pos += 1
                    continue
                if 0x30 <= nxt <= 0x37:  # octal, up to 3 digits
                    digits = bytearray()
                    while self.pos < n and len(digits) < 3 \
                            and 0x30 <= data[self.pos] <= 0x37:
                        digits.append(data[self.pos])
                        self.pos += 1
                    out.append(int(bytes(digits), 8) & 0xFF)
                    continue
                simple = {0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09,
                          0x62: 0x08, 0x66: 0x0C, 0x28: 0x28,
                          0x29: 0x29, 0x5C: 0x5C}
                out.append(simple.get(nxt, nxt))
                self.pos += 1
                continue
            if byte == 0x28:
                depth += 1
                out.append(byte)
                self.pos += 1
                continue
            if byte == 0x29:
                depth -= 1
                self.pos += 1
                if depth == 0:
                    return PdfString(bytes(out))
                out.append(byte)
                continue
            out.append(byte)
            self.pos += 1
        raise _fail("unterminated literal string in PDF object")

    def _read_hex_string(self) -> PdfString:
        # self.pos is at '<' (not '<<').
        data, n = self.data, len(self.data)
        self.pos += 1
        hexs = bytearray()
        while self.pos < n and data[self.pos] != 0x3E:  # '>'
            byte = data[self.pos]
            if not _is_ws(byte):
                hexs.append(byte)
            self.pos += 1
        if self.pos >= n:
            raise _fail("unterminated hex string in PDF object")
        self.pos += 1  # '>'
        if len(hexs) % 2:
            hexs.append(0x30)  # odd nibble pads with 0 (spec §7.3.4.3)
        try:
            return PdfString(bytes.fromhex(hexs.decode("ascii")))
        except ValueError:
            raise _fail("bad hex string in PDF object") from None

    def _read_name(self) -> PdfName:
        # self.pos is at '/'.
        self.pos += 1
        out = bytearray()
        while self.pos < len(self.data):
            byte = self.data[self.pos]
            if _is_ws(byte) or _is_delimiter(byte):
                break
            if byte == 0x23:  # '#' hex escape
                hexs = self.data[self.pos + 1:self.pos + 3]
                try:
                    out.append(int(hexs.decode("ascii"), 16))
                except ValueError:
                    raise _fail("bad # escape in PDF name") from None
                self.pos += 3
                continue
            out.append(byte)
            self.pos += 1
        return PdfName(out.decode("latin-1"))

    def _read_number_or_ref(self) -> PdfObject:
        # At a digit / sign / dot.  Try "N G R" first (indirect ref).
        match = re.match(rb"([+-]?\d+)\s+(\d+)\s+R(?![\w])",
                         self.data[self.pos:])
        if match:
            self.pos += match.end()
            return PdfRef(int(match.group(1)), int(match.group(2)))
        token = self._token()
        try:
            return int(token)
        except ValueError:
            pass
        try:
            value = float(token)
        except ValueError:
            raise _fail(f"bad number {token!r} in PDF object") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise _fail(f"bad number {token!r} in PDF object")
        return value

    def object(self) -> PdfObject:
        """Parse one object at the cursor; returns (value)."""
        self.skip_ws()
        if self.pos >= len(self.data):
            raise _fail("unexpected end of data while reading PDF object")
        byte = self.data[self.pos]
        if byte == 0x3C:  # '<'
            if self.data[self.pos:self.pos + 2] == b"<<":
                self.pos += 2
                result: dict[str, Any] = {}
                while True:
                    self.skip_ws()
                    if self.data[self.pos:self.pos + 2] == b">>":
                        self.pos += 2
                        # A stream keyword directly after a dictionary
                        # is the stream object itself (PDF 32000
                        # §7.3.7) — parse_object must see it too, not
                        # only parse_indirect.
                        saved = self.pos
                        self.skip_ws()
                        found = _stream_bytes(
                            self.data, self.pos,
                            result.get("/Length"))
                        if found is not None:
                            payload, end = found
                            self.pos = end
                            self.skip_ws()
                            if self.data[self.pos:self.pos + 9] \
                                    == b"endstream":
                                self.pos += 9
                            return PdfStream(result, payload)
                        self.pos = saved
                        return result
                    if self.pos >= len(self.data):
                        raise _fail("unterminated dictionary in PDF object")
                    key = self._read_name()
                    result["/" + str(key)] = self.object()
            return self._read_hex_string()
        if byte == 0x5B:  # '['
            self.pos += 1
            result_list: list[Any] = []
            while True:
                self.skip_ws()
                if self.pos >= len(self.data):
                    raise _fail("unterminated array in PDF object")
                if self.data[self.pos] == 0x5D:  # ']'
                    self.pos += 1
                    return result_list
                result_list.append(self.object())
        if byte == 0x28:  # '('
            return self._read_literal_string()
        if byte == 0x2F:  # '/'
            return self._read_name()
        if self.data[self.pos:self.pos + 4] == b"true":
            self.pos += 4
            return True
        if self.data[self.pos:self.pos + 5] == b"false":
            self.pos += 5
            return False
        if self.data[self.pos:self.pos + 4] == b"null":
            self.pos += 4
            return None
        if byte in b"0123456789+-.":
            return self._read_number_or_ref()
        raise _fail(
            f"unexpected byte {chr(byte)!r} at offset {self.pos} "
            f"while reading PDF object")


def parse_object(data: bytes, pos: int = 0) -> tuple[PdfObject, int]:
    """Parse one COS object at ``pos``; returns (value, next position)."""
    lexer = _Lexer(data, pos)
    value = lexer.object()
    return value, lexer.pos


def _stream_bytes(data: bytes, after_dict: int, length: Any
                  ) -> tuple[bytes, int] | None:
    """Raw stream payload after a dict ending at ``after_dict``.

    Returns (payload, end position) — ``end`` is where the payload
    stops in the file, so the caller can skip the EOL and consume
    ``endstream``.  ``stream`` must follow (EOL after the keyword is
    part of the syntax, not the data).  A trustworthy /Length wins —
    it is exact; otherwise scan for ``endstream``.  A declared length
    that does not land on ``endstream`` is treated as wrong and the
    scan wins (upstream's indirect-stream-with-wrong-length recovery).
    """
    match = re.match(rb"\s*stream\r?\n", data[after_dict:])
    if not match:
        return None
    start = after_dict + match.end()
    if isinstance(length, int) and length >= 0:
        end = start + length
        tail = data[end:end + 12]
        if tail.startswith(b"\r\nendstream") \
                or tail.startswith(b"\nendstream") \
                or tail.startswith(b"endstream"):
            return data[start:end], end
    end = data.find(b"endstream", start)
    if end < 0:
        return None
    payload = data[start:end]
    if payload.endswith(b"\r\n"):
        payload = payload[:-2]
    elif payload.endswith(b"\n") or payload.endswith(b"\r"):
        payload = payload[:-1]
    return payload, end


def parse_indirect(data: bytes, offset: int
                   ) -> tuple[PdfRef, PdfObject, int]:
    """Parse ``N G obj ... endobj`` at ``offset``.

    Returns (ref, value, end position).  A stream value is returned as
    :class:`PdfStream` (dict + raw bytes).  Raises
    :class:`~nomorals.documents.errors.DocumentError` (kind ``parse``)
    when the header does not parse.
    """
    match = re.match(rb"(\d+)\s+(\d+)\s+obj\b", data[offset:])
    if not match:
        raise _fail(f"no indirect object at offset {offset}")
    ref = PdfRef(int(match.group(1)), int(match.group(2)))
    lexer = _Lexer(data, offset + match.end())
    value = lexer.object()
    if isinstance(value, dict):
        found = _stream_bytes(data, lexer.pos, value.get("/Length"))
        if found is not None:
            value = PdfStream(value, found[0])
            lexer.skip_ws()
            if data[lexer.pos:lexer.pos + 9] == b"endstream":
                lexer.pos += 9
    lexer.skip_ws()
    if data[lexer.pos:lexer.pos + 6] == b"endobj":
        lexer.pos += 6
    return ref, value, lexer.pos


# ── serializer (upstream crates/cos writer.rs: serialize) ─────────────────


def _escape_name(name: str) -> bytes:
    out = bytearray()
    for byte in name.encode("latin-1", "replace"):
        if _is_ws(byte) or _is_delimiter(byte) or byte == 0x23 \
                or byte < 0x21 or byte > 0x7E:
            out += f"#{byte:02X}".encode("ascii")
        else:
            out.append(byte)
    return bytes(out)


def _escape_string(raw: bytes) -> bytes:
    out = bytearray(b"(")
    for byte in raw:
        if byte == 0x28:
            out += b"\\("
        elif byte == 0x29:
            out += b"\\)"
        elif byte == 0x5C:
            out += b"\\\\"
        elif byte == 0x0A:
            out += b"\\n"
        elif byte == 0x0D:
            out += b"\\r"
        elif byte == 0x09:
            out += b"\\t"
        elif byte < 0x20 or byte > 0x7E:
            out += f"\\{byte:03o}".encode("ascii")
        else:
            out.append(byte)
    out += b")"
    return bytes(out)


def _num(value: float) -> bytes:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return (text or "0").encode("ascii")


def serialize(obj: PdfObject) -> bytes:
    """One COS object to its canonical bytes (deterministic: dict keys
    sorted, so the same object always serializes the same way)."""
    if obj is None:
        return b"null"
    if obj is True:
        return b"true"
    if obj is False:
        return b"false"
    if isinstance(obj, int):
        return str(obj).encode("ascii")
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise _fail(f"cannot serialize non-finite number {obj!r}",
                        kind="write")
        return _num(obj)
    if isinstance(obj, PdfName):
        return b"/" + _escape_name(str(obj))
    if isinstance(obj, PdfString):
        return _escape_string(bytes(obj))
    if isinstance(obj, PdfRef):
        return f"{obj.num} {obj.gen} R".encode("ascii")
    if isinstance(obj, PdfStream):
        body = dict(obj.dict)
        body["/Length"] = len(obj.data)
        return (serialize(body) + b"\nstream\n" + obj.data
                + b"\nendstream")
    if isinstance(obj, list):
        return b"[" + b" ".join(serialize(v) for v in obj) + b"]"
    if isinstance(obj, dict):
        parts = []
        for key in sorted(obj):
            parts.append(b"/" + _escape_name(key.lstrip("/"))
                         + b" " + serialize(obj[key]))
        return b"<< " + b" ".join(parts) + b" >>"
    raise _fail(f"cannot serialize {type(obj).__name__} as a PDF object",
                kind="write")


def serialize_indirect(ref: PdfRef, obj: PdfObject) -> bytes:
    """``N G obj ... endobj`` for one indirect object."""
    return (f"{ref.num} {ref.gen} obj\n".encode("ascii")
            + serialize(obj) + b"\nendobj\n")


# ── xref (upstream crates/cos document/xref.rs) ───────────────────────────


@dataclass
class XrefEntry:
    """One xref entry: object number, generation, and where to find it.

    ``kind`` is ``"n"`` (plain, ``offset`` is the byte offset),
    ``"f"`` (free), or ``"compressed"`` (``offset`` is the object-stream
    number, ``index`` the position inside it)."""

    num: int
    gen: int
    offset: int
    kind: str = "n"
    index: int = 0


@dataclass
class Xref:
    """A parsed cross-reference: entries by object number, the winning
    trailer dictionary, the ``startxref`` position, whether the table
    was reconstructed by scanning (``repaired``), and how many trailer
    generations were followed (``generations``)."""

    entries: dict[int, XrefEntry] = field(default_factory=dict)
    trailer: dict[str, Any] = field(default_factory=dict)
    startxref: int = 0
    repaired: bool = False
    generations: int = 0


def _parse_classic_table(data: bytes, pos: int, xref: Xref
                         ) -> dict[str, Any]:
    """Parse ``xref`` + subsections + ``trailer`` at ``pos``.

    Returns the trailer dictionary.  Entries from a newer generation
    win — the chain is walked newest-first, so an object number
    already present is never overwritten.
    """
    lexer = _Lexer(data, pos)
    token = lexer._token()
    if token != b"xref":
        raise _fail(f"expected xref at offset {pos}")
    while True:
        lexer.skip_ws()
        head = re.match(rb"(\d+)\s+(\d+)", data[lexer.pos:])
        if not head:
            break
        first, count = int(head.group(1)), int(head.group(2))
        lexer.pos += head.end()
        lexer.skip_ws()
        for i in range(count):
            line = data[lexer.pos:lexer.pos + 20]
            match = re.match(
                rb"(\d{10})\s+(\d{5})\s+([nf])", line)
            if not match:
                raise _fail("malformed xref entry")
            num = first + i
            if num not in xref.entries:
                xref.entries[num] = XrefEntry(
                    num, int(match.group(2)), int(match.group(1)),
                    kind=match.group(3).decode("ascii"))
            # Entries are exactly 20 bytes including the end-of-line
            # marker: 18 bytes matched above + a 2-byte EOL (CRLF or
            # SP+LF); a lone CR or LF (19-byte entry) is tolerated.
            lexer.pos += match.end()
            if data[lexer.pos:lexer.pos + 2] == b"\r\n":
                lexer.pos += 2
            elif lexer.pos < len(data) \
                    and data[lexer.pos] in (0x0A, 0x0D):
                lexer.pos += 1
            elif lexer.pos < len(data) and data[lexer.pos] == 0x20:
                lexer.pos += 1
                if lexer.pos < len(data) \
                        and data[lexer.pos] in (0x0A, 0x0D):
                    lexer.pos += 1
        lexer.skip_ws()
        if data[lexer.pos:lexer.pos + 7] == b"trailer":
            lexer.pos += 7
            trailer, _ = parse_object(data, lexer.pos)
            if not isinstance(trailer, dict):
                raise _fail("trailer is not a dictionary")
            return trailer
    raise _fail("xref table without a trailer")


def _parse_xref_stream(data: bytes, pos: int, xref: Xref
                       ) -> dict[str, Any]:
    """Parse an xref stream object at ``pos``; returns its trailer
    dictionary (only the trailer keys — ``/W``, ``/Index`` and the
    stream's own keys are not trailer entries)."""
    ref, value, end = parse_indirect(data, pos)
    if not isinstance(value, PdfStream):
        raise _fail("xref position is not a stream object")
    stream_dict = value.dict
    width = stream_dict.get("/W")
    if not isinstance(width, list) or len(width) != 3:
        raise _fail("xref stream without a /W array of 3 widths")
    widths = [int(w) for w in width]
    index = stream_dict.get("/Index")
    if isinstance(index, list) and len(index) % 2 == 0 and index:
        ranges = [(int(index[i]), int(index[i + 1]))
                  for i in range(0, len(index), 2)]
    else:
        size = stream_dict.get("/Size")
        ranges = [(0, int(size))] if isinstance(size, int) else []
    raw = decode_stream(value)
    cursor = 0
    for first, count in ranges:
        for i in range(count):
            fields = []
            for w in widths:
                if w == 0:
                    fields.append(1 if len(fields) == 0 else 0)
                else:
                    chunk = raw[cursor:cursor + w]
                    if len(chunk) < w:
                        raise _fail("xref stream truncated")
                    fields.append(int.from_bytes(chunk, "big"))
                    cursor += w
            num = first + i
            if num in xref.entries:
                continue
            kind, field2, field3 = fields
            if kind == 0:
                xref.entries[num] = XrefEntry(num, field3, 0, kind="f")
            elif kind == 1:
                xref.entries[num] = XrefEntry(num, field3, field2,
                                              kind="n")
            elif kind == 2:
                xref.entries[num] = XrefEntry(num, 0, field2,
                                              kind="compressed",
                                              index=field3)
    return {k: v for k, v in stream_dict.items()
            if k in ("/Size", "/Root", "/Info", "/Prev", "/Encrypt",
                     "/ID")}


def _scan_objects(data: bytes) -> dict[int, XrefEntry]:
    """Reconstruct an xref by scanning for ``N G obj`` markers — the
    repair fallback when the table is unusable (upstream's
    ``damaged_xref_is_reconstructed``).  Generation is always 0 here;
    a file whose objects disagree with its table is read from the
    scan, never from the lie."""
    entries: dict[int, XrefEntry] = {}
    for match in re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", data):
        num = int(match.group(1))
        if num not in entries:
            entries[num] = XrefEntry(num, int(match.group(2)),
                                     match.start(), kind="n")
    return entries


def parse_xref(data: bytes) -> Xref:
    """Parse the xref of ``data`` following the ``/Prev`` chain.

    Classic tables and xref streams both work; a file whose table is
    missing or corrupt falls back to scanning (``repaired=True``) and
    still opens.  ``/Encrypt`` in any trailer is NOT resolved here —
    the document model refuses the file, this layer only reports it.
    """
    xref = Xref()
    start_match = None
    for match in re.finditer(rb"startxref\s*(\d+)", data):
        start_match = match
    if start_match is None:
        _repair_by_scan(data, xref)
        return xref
    pos = int(start_match.group(1))
    xref.startxref = pos
    seen: set[int] = set()
    while pos >= 0 and pos < len(data) and pos not in seen:
        seen.add(pos)
        try:
            if data[pos:pos + 4] == b"xref":
                trailer = _parse_classic_table(data, pos, xref)
            else:
                trailer = _parse_xref_stream(data, pos, xref)
        except DocumentError:
            break
        if xref.generations == 0:
            # The newest trailer wins outright.
            xref.trailer = trailer
        else:
            # Older generations only fill keys the newest lacks.
            for key in ("/Root", "/Info", "/ID", "/Encrypt"):
                if key not in xref.trailer and key in trailer:
                    xref.trailer[key] = trailer[key]
        xref.generations += 1
        prev = trailer.get("/Prev")
        if isinstance(prev, int) and prev >= 0:
            pos = prev
            continue
        break
    if not _xref_is_sound(data, xref):
        _repair_by_scan(data, xref)
    return xref


def _xref_is_sound(data: bytes, xref: Xref) -> bool:
    """True when the parsed table can be trusted: the trailer names a
    /Root and the entry for it resolves to the object it claims.
    A table of zeroed offsets parses cleanly but points nowhere —
    that is damage, not a table, and the scan fallback must win."""
    if not xref.entries:
        return False
    root = xref.trailer.get("/Root")
    if not isinstance(root, PdfRef):
        return False
    entry = xref.entries.get(root.num)
    if entry is None or entry.kind != "n":
        return False
    if entry.offset <= 0 or entry.offset >= len(data):
        return False
    try:
        found, _value, _ = parse_indirect(data, entry.offset)
    except DocumentError:
        return False
    return found.num == root.num


def _repair_by_scan(data: bytes, xref: Xref) -> None:
    """Rebuild the xref by scanning for object markers (upstream's
    ``damaged_xref_is_reconstructed``).  The parsed trailer is kept
    when it names a /Root; otherwise /Root and /Info are recovered by
    searching the body."""
    xref.entries = _scan_objects(data)
    xref.repaired = True
    xref.generations = 0
    if "/Root" not in xref.trailer:
        root_match = re.search(rb"/Root\s+(\d+)\s+(\d+)\s+R", data)
        if root_match:
            xref.trailer["/Root"] = PdfRef(int(root_match.group(1)),
                                           int(root_match.group(2)))
    if "/Info" not in xref.trailer:
        info_match = re.search(rb"/Info\s+(\d+)\s+(\d+)\s+R", data)
        if info_match:
            xref.trailer["/Info"] = PdfRef(int(info_match.group(1)),
                                           int(info_match.group(2)))
    if xref.entries:
        xref.trailer.setdefault("/Size", max(xref.entries) + 1)


def _resolve_compressed(data: bytes, xref: Xref, entry: XrefEntry
                        ) -> PdfObject | None:
    """An object living inside an object stream (PDF 1.5)."""
    stm_entry = xref.entries.get(entry.offset)
    if stm_entry is None or stm_entry.kind != "n":
        return None
    try:
        _ref, value, _ = parse_indirect(data, stm_entry.offset)
    except DocumentError:
        return None
    if not isinstance(value, PdfStream):
        return None
    stream_dict = value.dict
    count = stream_dict.get("/N")
    first = stream_dict.get("/First")
    if not isinstance(count, int) or not isinstance(first, int):
        return None
    raw = decode_stream(value)
    header = raw[:first].split()
    if len(header) < 2 * count:
        return None
    pairs = [(int(header[2 * i]), int(header[2 * i + 1]))
             for i in range(count)]
    for i, (_num, offset) in enumerate(pairs):
        if i == entry.index:
            start = first + offset
            end = first + (pairs[i + 1][1] if i + 1 < count
                           else len(raw))
            try:
                obj, _ = parse_object(raw, start)
            except DocumentError:
                return None
            if end < start:
                return None
            return obj
    return None


def resolve(data: bytes, xref: Xref, ref: PdfRef) -> PdfObject | None:
    """Resolve ``ref`` to its object, or None when it cannot be found.

    Follows plain and compressed (object-stream) entries.  A missing
    object is None, never an exception — callers decide whether the
    absence matters (a page tree tolerates it; a catalog does not).
    """
    entry = xref.entries.get(ref.num)
    if entry is None:
        return None
    if entry.kind == "f":
        return None
    if entry.kind == "compressed":
        return _resolve_compressed(data, xref, entry)
    try:
        found, value, _ = parse_indirect(data, entry.offset)
    except DocumentError:
        return None
    if found.num != ref.num:
        # The table lied about this offset — the entry is stale.
        return None
    return value


# ── stream decoding ───────────────────────────────────────────────────────


def decode_stream(stream: PdfStream) -> bytes:
    """Decode a stream's bytes: ``FlateDecode`` only (with PNG predictor
    unprediction when ``/DecodeParms`` carries one — the form real
    producers emit).  Any other filter raises ``unsupported_format``
    naming the filter; nothing is silently passed through as decoded.
    """
    raw = stream.data
    filters = stream.dict.get("/Filter")
    names: list[str] = []
    if isinstance(filters, PdfName):
        names = [str(filters)]
    elif isinstance(filters, list):
        names = [str(f) for f in filters if isinstance(f, PdfName)]
    elif filters is None:
        return raw
    else:
        raise _fail(f"unsupported stream filter {filters!r}",
                    kind="unsupported_format")
    for name in names:
        if name == "FlateDecode":
            try:
                raw = zlib.decompress(raw)
            except zlib.error:
                # A trailing newline inside the declared length is the
                # usual cause; one retry on the stripped payload.
                try:
                    raw = zlib.decompress(raw.rstrip(b"\r\n"))
                except zlib.error:
                    raise _fail("FlateDecode stream does not decompress",
                                kind="parse") from None
            parms = stream.dict.get("/DecodeParms")
            if isinstance(parms, dict):
                predictor = parms.get("/Predictor")
                if isinstance(predictor, int) and predictor > 1:
                    columns = parms.get("/Columns")
                    colors = parms.get("/Colors", 1)
                    bpc = parms.get("/BitsPerComponent", 8)
                    if isinstance(columns, int) and isinstance(colors, int) \
                            and isinstance(bpc, int):
                        raw = _unpredict(raw, predictor, columns,
                                         colors, bpc)
        else:
            raise _fail(f"unsupported stream filter /{name}",
                        kind="unsupported_format")
    return raw


def _unpredict(raw: bytes, predictor: int, columns: int, colors: int,
               bpc: int) -> bytes:
    """PNG predictor unprediction (PDF spec §7.4.4.3): predictors 10–15.
    Predictor 1 is a no-op; 2 is TIFF (unsupported — the bytes are
    returned as-is and the caller sees the raw form, never a lie)."""
    if predictor == 1:
        return raw
    if predictor == 2:
        return raw
    row_bytes = (columns * colors * bpc + 7) // 8
    out = bytearray()
    prev = bytearray(row_bytes)
    pos = 0
    bpp = max(1, (colors * bpc + 7) // 8)
    while pos < len(raw):
        if pos + row_bytes + 1 > len(raw):
            break
        filter_type = raw[pos]
        row = bytearray(raw[pos + 1:pos + 1 + row_bytes])
        pos += 1 + row_bytes
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(bpp, len(row)):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif filter_type == 2:
            for i in range(len(row)):
                row[i] = (row[i] + prev[i]) & 0xFF
        elif filter_type == 3:
            for i in range(len(row)):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(len(row)):
                a = row[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (
                    b if pb <= pc else c)
                row[i] = (row[i] + pred) & 0xFF
        else:
            # Unknown filter byte: keep the row as-is rather than
            # corrupting the rest of the stream.
            pass
        out += row
        prev = row
    return bytes(out)


# ── writers (upstream crates/cos writer.rs) ───────────────────────────────


def build_incremental(original: bytes,
                      objects: dict[PdfRef, PdfObject],
                      trailer: dict[str, Any],
                      prev_startxref: int) -> bytes:
    """Append an update to ``original`` (upstream ``write_incremental``):
    the new/changed objects, a fresh xref subsection, a trailer with
    ``/Prev`` pointing at the old ``startxref``, and a new ``startxref``.

    ``original`` is returned byte-for-byte as the prefix — the whole
    point of an incremental save.  ``trailer`` carries the winning
    entries (``/Root``, ``/Info``); ``/Size`` and ``/Prev`` are set
    here.  An empty ``objects`` with no trailer change is a no-op that
    returns ``original`` unchanged.
    """
    if not objects and not trailer:
        return original
    out = bytearray(original)
    if not out.endswith(b"\n"):
        out += b"\n"
    max_num = 0
    for ref in objects:
        max_num = max(max_num, ref.num)
    size = trailer.get("/Size")
    size = (max_num + 1 if not isinstance(size, int)
            else max(size, max_num + 1))
    offsets: dict[int, int] = {}
    for ref in sorted(objects, key=lambda r: (r.num, r.gen)):
        offsets[ref.num] = len(out)
        out += serialize_indirect(ref, objects[ref])
    xref_pos = len(out)
    out += b"xref\n"
    nums = sorted(offsets)
    start = nums[0]
    run = [start]
    for num in nums[1:]:
        if num == run[-1] + 1:
            run.append(num)
        else:
            _write_xref_run(out, run, offsets)
            run = [num]
    _write_xref_run(out, run, offsets)
    new_trailer = {k: v for k, v in trailer.items()
                   if k not in ("/Size", "/Prev")}
    new_trailer["/Size"] = size
    if prev_startxref > 0:
        new_trailer["/Prev"] = prev_startxref
    out += b"trailer\n" + serialize(new_trailer) + b"\n"
    out += f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    return bytes(out)


def _write_xref_run(out: bytearray, run: list[int],
                    offsets: dict[int, int]) -> None:
    out += f"{run[0]} {len(run)}\n".encode("ascii")
    for num in run:
        out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")


def build_full(objects: dict[PdfRef, PdfObject],
               root_ref: PdfRef,
               info_ref: PdfRef | None) -> bytes:
    """Write every object fresh with a complete xref table and trailer
    (upstream ``write_full``).  Used after a repair, or whenever the
    source's xref cannot be trusted: nothing of the original file's
    layout survives, only its objects."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    nums = sorted({ref.num for ref in objects})
    by_num: dict[int, PdfObject] = {}
    for ref, obj in objects.items():
        by_num[ref.num] = obj
    for num in nums:
        ref = next(r for r in objects if r.num == num)
        offsets[num] = len(out)
        out += serialize_indirect(ref, by_num[num])
    xref_pos = len(out)
    max_id = max(nums) if nums else 0
    out += f"xref\n0 {max_id + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for num in range(1, max_id + 1):
        if num in offsets:
            out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
        else:
            out += b"0000000000 65535 f \n"
    trailer: dict[str, Any] = {"/Size": max_id + 1, "/Root": root_ref}
    if info_ref is not None:
        trailer["/Info"] = info_ref
    out += b"trailer\n" + serialize(trailer) + b"\n"
    out += f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    return bytes(out)
