"""Page surgery over PDFs — the Smallpdf (#48) + ILovePDF (#49) core.

One organ serves both sites (they are the same system shape with
different catalogs; building two suites would violate the
no-duplicate-systems rule).  The page-surgery half Devon lacked:
merge, split, extract/delete/reorder pages, rotate, crop, stamp
(watermark / page numbers), compress, repair, protect/unlock,
PDF/A archival shape, scan-to-PDF, and signature-image placement.

The writer reassembles a fresh object graph from selected pages of
one or more sources, on ``core.pdf``'s tolerant object parser
(``_parse_objects`` — it scans for ``N 0 obj`` markers, so broken
xref tables and truncated files still load).  Content streams are
copied as objects, never re-authored, so fidelity is structural,
not re-rendered — except stamp/sign streams, which are new objects
appended to a page's ``/Contents``.

Honest limits, carried in every result's ``labels``:
- protect/unlock is the 40-bit RC4 standard handler (V=1, R=3) —
  the only PDF encryption the stdlib can do (AES needs a crypto
  library).  A Devon-protected PDF opens in any spec reader; the
  cross-reader proof is live evidence, owed after the pull.
- repair cannot recover content whose bytes are gone — the result
  reports ``recovered_pages`` / ``recovered_objects`` honestly,
  never a silent partial presented as whole.
- PDF/A is the archival *shape* (XMP metadata, MarkInfo, Lang,
  non-archival features stripped); conformance *validation*
  (veraPDF-grade) is a separate, unbuilt rung.
- OCR-backed conversion stays gated on the Tesseract stack
  (``document.ocr_unavailable``); e-sign *ceremonies* are out of
  scope — the native slice is placing a signature image/annotation.
"""

from __future__ import annotations

import hashlib
import io
import re
import secrets
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.pdf import (
    _page_text,
    _parse_objects,
    _stream_bytes,
)
from .errors import DocumentError

__all__ = [
    "ArchiveOp",
    "CompressOp",
    "CropOp",
    "DeleteOp",
    "MergeOp",
    "PagewiseJob",
    "PagewiseResult",
    "ProtectOp",
    "RepairOp",
    "ReorderOp",
    "RotateOp",
    "ScanOp",
    "SelectOp",
    "SignOp",
    "StampOp",
    "UnlockOp",
    "archive_pdf",
    "extract_page_texts",
    "format_pagewise_reply",
    "pdf_info",
    "protect_pdf",
    "repair_pdf",
    "run_pagewise",
    "scan_to_pdf",
    "unlock_pdf",
]

# ── small PDF value parser ──────────────────────────────────────────────────
#
# A dict body is parsed into Python values so every edit (rotate,
# crop, resource merge, key stripping) is structural, never regex
# surgery on nested dictionaries.  Names stay ``str`` (leading ``/``),
# strings stay ``bytes`` (latin-1), refs are ("ref", num, gen).

_WS = b" \t\r\n\x00\x0c"


def _skip_ws(data: bytes, i: int) -> int:
    while i < len(data):
        ch = data[i:i + 1]
        if ch in _WS:
            i += 1
            continue
        if ch == b"%" :  # comment to end of line
            while i < len(data) and data[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        break
    return i


def _parse_literal(data: bytes, i: int) -> tuple[bytes, int]:
    """Parse a (...) literal string starting at ``data[i] == '('``."""
    assert data[i:i + 1] == b"("
    i += 1
    depth = 1
    out = bytearray()
    while i < len(data) and depth:
        ch = data[i:i + 1]
        if ch == b"\\":
            nxt = data[i + 1:i + 2]
            if nxt.isdigit():
                j = i + 1
                digits = b""
                while (j < len(data) and data[j:j + 1].isdigit()
                       and len(digits) < 3):
                    digits += data[j:j + 1]
                    j += 1
                out.append(int(digits, 8) & 0xFF if digits else 0)
                i = j
                continue
            simple = {b"n": 0x0A, b"r": 0x0D, b"t": 0x09, b"b": 0x08,
                      b"f": 0x0C, b"(": 0x28, b")": 0x29, b"\\": 0x5C}
            if nxt in simple:
                out.append(simple[nxt])
                i += 2
                continue
            if nxt in (b"\n", b"\r"):
                i += 2
                if nxt == b"\r" and data[i:i + 1] == b"\n":
                    i += 1
                continue
            out += nxt
            i += 2
            continue
        if ch == b"(":
            depth += 1
            out += ch
            i += 1
            continue
        if ch == b")":
            depth -= 1
            if depth:
                out += ch
            i += 1
            continue
        out += ch
        i += 1
    return bytes(out), i


def _parse_value(data: bytes, i: int) -> tuple[Any, int]:
    i = _skip_ws(data, i)
    if i >= len(data):
        raise ValueError("unexpected end of PDF value")
    two = data[i:i + 2]
    if two == b"<<":
        out: dict[str, Any] = {}
        i += 2
        while True:
            i = _skip_ws(data, i)
            if data[i:i + 2] == b">>":
                return out, i + 2
            if data[i:i + 1] != b"/":
                raise ValueError("expected a name key in PDF dict")
            key, i = _parse_value(data, i)
            val, i = _parse_value(data, i)
            out[key] = val
    if data[i:i + 1] == b"[":
        out_list: list[Any] = []
        i += 1
        while True:
            i = _skip_ws(data, i)
            if data[i:i + 1] == b"]":
                return out_list, i + 1
            val, i = _parse_value(data, i)
            out_list.append(val)
    if data[i:i + 1] == b"(":
        return _parse_literal(data, i)
    if data[i:i + 1] == b"<":
        end = data.find(b">", i)
        if end < 0:
            raise ValueError("unterminated hex string")
        raw = re.sub(rb"\s+", b"", data[i + 1:end])
        try:
            return bytes.fromhex(raw.decode("ascii")), end + 1
        except ValueError:
            return raw, end + 1
    if data[i:i + 1] == b"/":
        m = re.match(rb"/[^\s\[\]<>(){}%/\x00-\x1f]+", data[i:])
        if not m:
            raise ValueError("bad PDF name")
        return m.group(0).decode("latin-1"), i + len(m.group(0))
    m = re.match(rb"[^\s\[\]<>(){}%/]+", data[i:])
    if not m:
        raise ValueError("bad PDF token")
    tok = m.group(0).decode("latin-1")
    i += len(m.group(0))
    if tok in ("true", "false"):
        return tok == "true", i
    if tok == "null":
        return None, i
    # an indirect reference "<num> <gen> R" — check before the number
    # parses, or "2 0 R" comes back as a bare 2.  Negative numbers are
    # the writer's own local ids (-1, -2 for stamp/scan objects).
    if tok.lstrip("-").isdigit():
        m2 = re.match(rb"\s+(\d+)\s+R", data[i:])
        if m2:
            return ("ref", int(tok), int(m2.group(1))), i + len(m2.group(0))
        return int(tok), i
    try:
        return float(tok), i
    except ValueError:
        pass
    return tok, i


def _parse_dict(body: bytes) -> dict[str, Any]:
    """Parse the leading ``<< ... >>`` of a dict body, or {}."""
    i = _skip_ws(body, 0)
    if body[i:i + 2] != b"<<":
        return {}
    val, _ = _parse_value(body, i)
    return val if isinstance(val, dict) else {}


def _escape_literal(raw: bytes) -> bytes:
    out = bytearray(b"(")
    for byte in raw:
        if byte in (0x28, 0x29, 0x5C):
            out.append(0x5C)
            out.append(byte)
        elif 0x20 <= byte < 0x7F:
            out.append(byte)
        else:
            out += f"\\{byte:03o}".encode()
    out.append(0x29)
    return bytes(out)


def _render_value(val: Any) -> bytes:
    if val is None:
        return b"null"
    if val is True:
        return b"true"
    if val is False:
        return b"false"
    if isinstance(val, tuple) and len(val) == 3 and val[0] == "ref":
        return f"{val[1]} {val[2]} R".encode()
    if isinstance(val, int):
        return str(val).encode()
    if isinstance(val, float):
        if val.is_integer():
            return str(int(val)).encode()
        return repr(val).encode()
    if isinstance(val, bytes):
        # hex strings: always valid, and the form the spec wants for
        # /ID and the /O /U entries of the encryption dictionary
        return b"<" + val.hex().encode() + b">"
    if isinstance(val, str):
        return val.encode("latin-1", "replace")
    if isinstance(val, list):
        return b"[" + b" ".join(_render_value(v) for v in val) + b"]"
    if isinstance(val, dict):
        parts = []
        for key, sub in val.items():
            parts.append(f"{key} ".encode() + _render_value(sub))
        return b"<< " + b" ".join(parts) + b" >>"
    raise ValueError(f"cannot render PDF value {val!r}")


def _split_stream(body: bytes) -> tuple[bytes, bytes | None]:
    """(dict part, stream payload or None) of an object body."""
    m = re.search(rb"stream\r?\n", body)
    if not m:
        return body, None
    end = body.rfind(b"endstream")
    payload = body[m.end():end if end > 0 else len(body)]
    return body[:m.start()], payload


def _dict_refs(d: Any, out: set[int]) -> None:
    """Collect every indirect object number referenced by a value."""
    if isinstance(d, tuple) and len(d) == 3 and d[0] == "ref":
        out.add(d[1])
    elif isinstance(d, dict):
        for val in d.values():
            _dict_refs(val, out)
    elif isinstance(d, list):
        for val in d:
            _dict_refs(val, out)


def _remap_value(val: Any, mapping: dict[int, int]) -> Any:
    if isinstance(val, tuple) and len(val) == 3 and val[0] == "ref":
        return ("ref", mapping.get(val[1], val[1]), val[2])
    if isinstance(val, dict):
        return {k: _remap_value(v, mapping) for k, v in val.items()}
    if isinstance(val, list):
        return [_remap_value(v, mapping) for v in val]
    return val


# ── RC4 standard security handler (V=1, R=3, 40-bit) ───────────────────────
#
# PDF 1.4 spec, Algorithms 3.2–3.5.  The only PDF encryption the
# stdlib can do (AES needs a crypto library) — and it is real,
# interoperable cryptography: a Devon-protected PDF opens in any
# spec-conformant reader.

_PADDING = bytes([
    0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41,
    0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
    0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80,
    0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
])


def _pad_password(password: str) -> bytes:
    raw = password.encode("latin-1", "replace")[:32]
    return raw + _PADDING[:32 - len(raw)]


def _rc4(key: bytes, data: bytes) -> bytes:
    state = list(range(256))
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % len(key)]) & 0xFF
        state[i], state[j] = state[j], state[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) & 0xFF
        j = (j + state[i]) & 0xFF
        state[i], state[j] = state[j], state[i]
        out.append(byte ^ state[(state[i] + state[j]) & 0xFF])
    return bytes(out)


def _rc4_file_key(password: str, owner: bytes, perms: int,
                  file_id: bytes) -> bytes:
    digest = hashlib.md5(
        _pad_password(password) + owner + struct.pack("<i", perms)
        + file_id).digest()
    return digest[:5]


def _rc4_owner_key(owner_password: str, user_password: str) -> bytes:
    key = hashlib.md5(_pad_password(owner_password)).digest()[:5]
    return _rc4(key, _pad_password(user_password))


def _rc4_object_key(file_key: bytes, obj_num: int) -> bytes:
    seed = file_key + struct.pack("<I", obj_num)[:3] + b"\x00\x00"
    return hashlib.md5(seed).digest()[:10]


def _encrypt_strings_in_dict(body: bytes, obj_num: int,
                             file_key: bytes) -> bytes:
    """Encrypt every literal/hex string in a dict body (not streams)."""
    key = _rc4_object_key(file_key, obj_num)
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i:i + 1]
        if ch == b"(":
            raw, nxt = _parse_literal(body, i)
            enc = _rc4(key, raw)
            out += b"<" + enc.hex().encode() + b">"
            i = nxt
            continue
        if ch == b"<":
            if body[i + 1:i + 2] == b"<":
                # a dictionary open, not a hex string
                out += b"<<"
                i += 2
                continue
            end = body.find(b">", i)
            if end < 0:
                out += body[i:]
                break
            raw = re.sub(rb"\s+", b"", body[i + 1:end])
            try:
                decoded = bytes.fromhex(raw.decode("ascii"))
            except ValueError:
                decoded = raw
            out += b"<" + _rc4(key, decoded).hex().encode() + b">"
            i = end + 1
            continue
        out += ch
        i += 1
    return bytes(out)


def _decrypt_strings_in_dict(body: bytes, obj_num: int,
                             file_key: bytes) -> bytes:
    """Decrypt every literal/hex string in a dict body (not streams)."""
    key = _rc4_object_key(file_key, obj_num)
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i:i + 1]
        if ch == b"(":
            raw, nxt = _parse_literal(body, i)
            out += _escape_literal(_rc4(key, raw))
            i = nxt
            continue
        if ch == b"<":
            if body[i + 1:i + 2] == b"<":
                # a dictionary open, not a hex string
                out += b"<<"
                i += 2
                continue
            end = body.find(b">", i)
            if end < 0:
                out += body[i:]
                break
            raw = re.sub(rb"\s+", b"", body[i + 1:end])
            try:
                decoded = bytes.fromhex(raw.decode("ascii"))
            except ValueError:
                decoded = raw
            out += _escape_literal(_rc4(key, decoded))
            i = end + 1
            continue
        out += ch
        i += 1
    return bytes(out)


# ── loading ─────────────────────────────────────────────────────────────────


def _fail(message: str, kind: str = "parse") -> DocumentError:
    return DocumentError(message, kind=kind)


@dataclass
class _Page:
    num: int
    body: bytes
    mediabox: tuple[float, float, float, float] = (0, 0, 612, 792)
    resources: dict[str, Any] | None = None


@dataclass
class _Source:
    objects: dict[int, bytes]
    root_num: int | None
    pages: list[_Page]
    encrypt: dict[str, Any] | None
    info: dict[str, Any] | None
    file_id: bytes | None
    corruption: list[str] = field(default_factory=list)
    lost_pages: int = 0

    @property
    def page_refs_total(self) -> int:
        return len(self.pages)


def _find_root_num(data: bytes,
                   objects: dict[int, bytes]) -> int | None:
    m = re.search(rb"/Root\s+(\d+)\s+0\s+R", data)
    if m and int(m.group(1)) in objects:
        return int(m.group(1))
    for num, body in objects.items():
        if b"/Catalog" in body:
            return num
    return None


def _parse_box(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, list) and len(value) == 4:
        try:
            nums = [float(v) for v in value]
        except (TypeError, ValueError):
            return None
        return (nums[0], nums[1], nums[2], nums[3])
    return None


def _walk_tree(objects: dict[int, bytes], node_num: int,
               out: list[_Page], seen: set[int],
               inherited_box: tuple[float, float, float, float] | None,
               inherited_res: dict[str, Any] | None,
               stats: dict[str, int]) -> None:
    body = objects.get(node_num)
    if body is None:
        stats["lost"] += 1
        return
    d = _parse_dict(body)
    box = _parse_box(d.get("/MediaBox")) or inherited_box
    res = d.get("/Resources")
    if isinstance(res, dict):
        inherited_res = res
    elif isinstance(res, tuple):
        target = objects.get(res[1])
        if target is not None:
            inherited_res = _parse_dict(target)
    kids = d.get("/Kids")
    if isinstance(kids, list) and kids:
        for kid in kids:
            if isinstance(kid, tuple) and kid[0] == "ref":
                if kid[1] in seen:
                    continue
                seen.add(kid[1])
                _walk_tree(objects, kid[1], out, seen, box,
                           inherited_res, stats)
        return
    if "/Contents" in d or _stream_bytes(body) is not None:
        # a page whose content streams are gone is lost, not kept as
        # an empty shell — repair reports it honestly
        contents = d.get("/Contents")
        refs: set[int] = set()
        _dict_refs(contents, refs)
        if refs and not refs <= set(objects):
            stats["lost"] += 1
            return
        out.append(_Page(num=node_num, body=body,
                         mediabox=box or (0, 0, 612, 792),
                         resources=inherited_res))


def _load_source(data: bytes) -> _Source:
    head = data[:1024]
    header_at = head.find(b"%PDF-")
    if header_at < 0:
        raise _fail("not a PDF file (missing %PDF- header)")
    objects = _parse_objects(data)
    corruption: list[str] = []
    if header_at > 0:
        corruption.append("garbage-prefix")
    if b"xref" not in data:
        corruption.append("truncated-xref")
    if b"trailer" not in data:
        corruption.append("missing-trailer")
    if b"startxref" not in data:
        corruption.append("missing-startxref")
    if not data.rstrip().endswith(b"%%EOF"):
        corruption.append("missing-eof-marker")
    if not objects:
        raise _fail("no PDF objects found — file is empty or destroyed",
                    kind="parse")
    root_num = _find_root_num(data, objects)
    pages: list[_Page] = []
    stats: dict[str, int] = {"lost": 0}
    if root_num is not None:
        root_body = objects[root_num]
        root_d = _parse_dict(root_body)
        pages_ref = root_d.get("/Pages")
        if isinstance(pages_ref, tuple) and pages_ref[0] == "ref":
            seen: set[int] = {pages_ref[1]}
            _walk_tree(objects, pages_ref[1], pages, seen, None, None,
                       stats)
        elif isinstance(pages_ref, dict):
            # inline page tree — rare; walk it as a node body
            _walk_inline(objects, pages_ref, pages, set(), None, None,
                         stats)
        if (not pages and ("/Contents" in root_d
                           or _stream_bytes(root_body) is not None)):
            # the /Pages node itself may be a single page
            pages.append(_Page(num=root_num, body=root_body))
    if not pages:
        # last resort, mirroring core.pdf's reader: every object with
        # a /Contents and no /Kids, in object-number order
        for num in sorted(objects):
            d = _parse_dict(objects[num])
            if "/Contents" in d and "/Kids" not in d:
                contents = d.get("/Contents")
                refs: set[int] = set()
                _dict_refs(contents, refs)
                if refs and not refs <= set(objects):
                    stats["lost"] += 1
                    continue
                pages.append(_Page(num=num, body=objects[num]))
    if not pages:
        raise _fail("no pages found in this PDF", kind="parse")
    encrypt: dict[str, Any] | None = None
    m = re.search(rb"/Encrypt\s+(\d+)\s+0\s+R", data)
    if m and int(m.group(1)) in objects:
        encrypt = _parse_dict(objects[int(m.group(1))])
    elif m:
        encrypt = {}
    info: dict[str, Any] | None = None
    m = re.search(rb"/Info\s+(\d+)\s+0\s+R", data)
    if m and int(m.group(1)) in objects:
        info = _parse_dict(objects[int(m.group(1))])
    file_id: bytes | None = None
    m = re.search(rb"/ID\s*\[\s*<([0-9A-Fa-f]+)>", data)
    if m:
        try:
            file_id = bytes.fromhex(m.group(1).decode())
        except ValueError:
            file_id = None
    return _Source(objects=objects, root_num=root_num, pages=pages,
                   encrypt=encrypt, info=info, file_id=file_id,
                   corruption=corruption, lost_pages=stats["lost"])


def _walk_inline(objects: dict[int, bytes], node: dict[str, Any],
                 out: list[_Page], seen: set[int],
                 inherited_box: tuple[float, float, float, float] | None,
                 inherited_res: dict[str, Any] | None,
                 stats: dict[str, int]) -> None:
    kids = node.get("/Kids")
    if isinstance(kids, list) and kids:
        for kid in kids:
            if isinstance(kid, tuple) and kid[0] == "ref":
                if kid[1] in seen:
                    continue
                seen.add(kid[1])
                _walk_tree(objects, kid[1], out, seen, inherited_box,
                           inherited_res, stats)
        return
    if "/Contents" in node:
        return  # an inline node without a number cannot be emitted


def _closure(objects: dict[int, bytes],
             seeds: list[int]) -> dict[int, bytes]:
    """Every object reachable from ``seeds`` via dict references.

    Stream payloads are never scanned for references; a stream's
    ``/Length`` reference is not followed (the length is recomputed
    at emit time).
    """
    keep: dict[int, bytes] = {}
    queue = list(seeds)
    while queue:
        num = queue.pop()
        if num in keep or num not in objects:
            continue
        body = objects[num]
        keep[num] = body
        dpart, payload = _split_stream(body)
        d = _parse_dict(dpart)
        refs: set[int] = set()
        _dict_refs(d, refs)
        if payload is not None and isinstance(d.get("/Length"), tuple):
            refs.discard(d["/Length"][1])
        for ref in refs:
            if ref not in keep:
                queue.append(ref)
    return keep


def _object_stream_payload(body: bytes) -> bytes | None:
    return _stream_bytes(body)


# ── writer ──────────────────────────────────────────────────────────────────


@dataclass
class _OutPage:
    """One page in the output graph: dict body + closure objects."""
    body: bytes
    closure: dict[int, bytes]
    mediabox: tuple[float, float, float, float]
    extra: dict[int, bytes] = field(default_factory=dict)
    """Per-page objects (stamp streams, extracted inline content) —
    negative local ids, never shared between pages."""


def _emit_pdf(pages: list[_OutPage], *,
              title: str | None = None,
              author: str | None = None,
              extra_catalog: dict[str, Any] | None = None,
              metadata_obj: bytes | None = None,
              encrypt: bytes | None = None,
              file_key: bytes | None = None,
              file_id: bytes | None = None,
              compress_level: int = 0,
              jpeg_quality: int | None = None,
              output_intent_icc: bytes | None = None) -> bytes:
    """Assemble a fresh PDF from output pages.

    Object 1 is the catalog, 2 the page tree; every other object is
    renumbered from 3.  ``compress_level`` > 0 re-compresses Flate
    streams at that zlib level and flate-compresses uncompressed
    streams (0 = copy streams untouched).  ``encrypt`` is the raw
    /Encrypt dictionary body; it is never itself encrypted.
    """
    if file_id is None:
        file_id = secrets.token_bytes(16)
    slots: dict[int, bytes] = {}
    next_id = 3
    page_ids: list[int] = []
    # first pass: hand out new numbers (page dicts first, then closures)
    for _ in pages:
        page_ids.append(next_id)
        next_id += 1
    # closure objects, in deterministic order per page; pages from the
    # same source share one closure dict, so shared objects (fonts,
    # images) are emitted once, not once per page
    closure_maps: list[dict[int, int]] = []
    seen_closures: dict[int, dict[int, int]] = {}
    for page in pages:
        key = id(page.closure)
        if key not in seen_closures:
            cmap: dict[int, int] = {}
            for old in sorted(page.closure):
                cmap[old] = next_id
                next_id += 1
            seen_closures[key] = cmap
        closure_maps.append(seen_closures[key])
    metadata_id = None
    if metadata_obj is not None:
        metadata_id = next_id
        next_id += 1
    encrypt_id = None
    if encrypt is not None:
        encrypt_id = next_id
        next_id += 1
    info_id = None
    info_d: dict[str, Any] = {}
    if title is not None:
        info_d["/Title"] = title.encode("latin-1", "replace")
    if author is not None:
        info_d["/Author"] = author.encode("latin-1", "replace")
    if info_d:
        info_id = next_id
        next_id += 1
    profile_id = None
    profile_body: bytes | None = None
    if output_intent_icc is not None:
        profile_id = next_id
        next_id += 1
        profile_body = (
            b"<< /N 3 /Length "
            + str(len(output_intent_icc)).encode()
            + b" >>\nstream\n" + output_intent_icc + b"\nendstream")
    catalog_d: dict[str, Any] = {"/Type": "/Catalog",
                                 "/Pages": ("ref", 2, 0)}
    if metadata_id is not None:
        catalog_d["/Metadata"] = ("ref", metadata_id, 0)
    if profile_id is not None:
        catalog_d["/OutputIntent"] = [{
            "/Type": "/OutputIntent",
            "/S": "/GTS_PDFA1",
            "/OutputConditionIdentifier": b"sRGB IEC61966-2.1",
            "/DestOutputProfile": ("ref", profile_id, 0),
        }]
    if extra_catalog:
        catalog_d.update(extra_catalog)
    slots[1] = _render_value(catalog_d)
    children = " ".join(f"{pid} 0 R" for pid in page_ids)
    slots[2] = (f"<< /Type /Pages /Kids [{children}] "
                f"/Count {len(page_ids)} >>").encode()
    if encrypt_id is not None and encrypt is not None:
        # the /Encrypt dictionary is never itself encrypted
        slots[encrypt_id] = encrypt

    def emit_object(old_body: bytes, cmap: dict[int, int],
                    new_num: int) -> bytes:
        dpart, payload = _split_stream(old_body)
        d = _parse_dict(dpart)
        if payload is not None and isinstance(d.get("/Length"), tuple):
            d.pop("/Length")
        d = _remap_value(d, cmap)
        if payload is not None:
            if compress_level and d.get("/Filter") == "/FlateDecode":
                try:
                    raw = zlib.decompress(payload.rstrip(b"\r\n"))
                    payload = zlib.compress(raw, compress_level)
                except zlib.error:
                    pass
            elif (compress_level and jpeg_quality is not None
                    and _is_jpeg_image(d) and _pillow_available()):
                recompressed = _reencode_jpeg(payload, jpeg_quality)
                if recompressed is not None:
                    payload = recompressed
            elif compress_level and d.get("/Filter") is None:
                # uncompressed stream: flate it (the compress rung —
                # this is what shrinks plain-text fixture PDFs)
                try:
                    payload = zlib.compress(payload.rstrip(b"\r\n"),
                                            compress_level)
                    d["/Filter"] = "/FlateDecode"
                except zlib.error:
                    pass
            d["/Length"] = len(payload)
            out = _render_value(d) + b"\nstream\n" + payload + b"\nendstream"
        else:
            out = _render_value(d)
        if file_key is not None:
            dpart2, payload2 = _split_stream(out)
            if payload2 is not None:
                enc_payload = _rc4(_rc4_object_key(file_key, new_num),
                                   payload2)
                enc_dpart = _encrypt_strings_in_dict(dpart2, new_num,
                                                     file_key)
                out = (enc_dpart + b"\nstream\n" + enc_payload
                       + b"\nendstream")
            else:
                out = _encrypt_strings_in_dict(out, new_num, file_key)
        return out

    # metadata / info / ICC profile go through emit_object too, so a
    # protected file encrypts them like every other object
    if metadata_id is not None and metadata_obj is not None:
        slots[metadata_id] = emit_object(metadata_obj, {}, metadata_id)
    if info_id is not None:
        slots[info_id] = emit_object(_render_value(info_d), {}, info_id)
    if profile_id is not None and profile_body is not None:
        slots[profile_id] = emit_object(profile_body, {}, profile_id)

    emitted_closures: set[int] = set()
    for page, pid, cmap in zip(pages, page_ids, closure_maps,
                               strict=True):
        # per-page extra objects (stamp streams, extracted inline
        # content) get their own ids; the page dict remaps through the
        # combined map so its /Contents can point at them
        extra_map: dict[int, int] = {}
        for old in sorted(page.extra):
            extra_map[old] = next_id
            next_id += 1
        combined = dict(cmap)
        combined.update(extra_map)
        slots[pid] = emit_object(page.body, combined, pid)
        for old_num, extra_body in sorted(page.extra.items()):
            slots[extra_map[old_num]] = emit_object(extra_body, combined,
                                                    extra_map[old_num])
        if id(page.closure) in emitted_closures:
            continue
        emitted_closures.add(id(page.closure))
        for old_num, old_body in sorted(page.closure.items()):
            slots[cmap[old_num]] = emit_object(old_body, cmap,
                                               cmap[old_num])

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(slots):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode()
        out += slots[obj_id]
        out += b"\nendobj\n"
    xref_pos = len(out)
    max_id = max(offsets)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        if obj_id in offsets:
            out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
        else:
            out += b"0000000000 65535 f \n"
    trailer: dict[str, Any] = {"/Size": max_id + 1, "/Root": ("ref", 1, 0)}
    if encrypt_id is not None:
        trailer["/Encrypt"] = ("ref", encrypt_id, 0)
    if info_id is not None:
        trailer["/Info"] = ("ref", info_id, 0)
    trailer["/ID"] = [file_id, file_id]
    out += b"trailer\n" + _render_value(trailer) + b"\n"
    out += f"startxref\n{xref_pos}\n%%EOF\n".encode()
    return bytes(out)


def _is_jpeg_image(d: dict[str, Any]) -> bool:
    return (d.get("/Subtype") == "/Image"
            and d.get("/Filter") == "/DCTDecode")


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _reencode_jpeg(payload: bytes, quality: int) -> bytes | None:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(payload))
        img.load()
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        return data if len(data) < len(payload) else None
    except Exception:
        return None


# ── page preparation ────────────────────────────────────────────────────────


def _merge_into(res_d: dict[str, Any],
                additions: dict[str, dict[str, tuple]],
                closure: dict[int, bytes]) -> None:
    """Merge ``additions`` ({kind: {name: ref}}) into a resources dict.

    When a kind's entry lives in a shared closure object, the entry is
    added to that object once — idempotent, so stamping or signing
    several pages that share resources is safe.
    """
    for kind, entries in additions.items():
        existing = res_d.get(kind)
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.update(entries)
            res_d[kind] = merged
        elif (isinstance(existing, tuple) and existing[0] == "ref"
              and closure is not None):
            target = closure.get(existing[1])
            if target is not None:
                fd = _parse_dict(target)
                fd.update(entries)
                closure[existing[1]] = _render_value(fd)
            else:
                res_d[kind] = dict(entries)
        else:
            res_d[kind] = dict(entries)


def _merge_resources(page_d: dict[str, Any], closure: dict[int, bytes],
                     additions: dict[str, dict[str, tuple]]) -> None:
    """Make ``additions`` visible to a page through its /Resources."""
    res = page_d.get("/Resources")
    if isinstance(res, dict):
        res_d = dict(res)
        _merge_into(res_d, additions, closure)
        page_d["/Resources"] = res_d
        return
    if isinstance(res, tuple) and res[0] == "ref":
        target = closure.get(res[1])
        if target is None:
            page_d["/Resources"] = {k: dict(v)
                                    for k, v in additions.items()}
            return
        res_d = _parse_dict(target)
        _merge_into(res_d, additions, closure)
        closure[res[1]] = _render_value(res_d)
        return
    page_d["/Resources"] = {k: dict(v) for k, v in additions.items()}


def _prepare_page(page: _Page, closure: dict[int, bytes],
                  rotate_deg: int = 0,
                  crop: tuple[float, float, float, float] | None = None,
                  appends: list[tuple[bytes, dict[str, dict[str, bytes]]]]
                  | None = None) -> _OutPage:
    """Render one working page into its output dict body.

    ``appends`` is a list of (content-stream payload, resources):
    each stream is appended to /Contents, each resource object
    (font, image XObject) becomes a per-page extra object under a
    negative local id, remapped at emit.
    """
    dpart, inline_payload = _split_stream(page.body)
    d = _parse_dict(dpart)
    d["/Parent"] = ("ref", 2, 0)
    # materialize inherited resources onto the page so the output is
    # self-contained (the reader resolves fonts from the page body)
    if "/Resources" not in d and page.resources is not None:
        d["/Resources"] = page.resources
    if rotate_deg:
        current = d.get("/Rotate")
        base = current if isinstance(current, int) else 0
        d["/Rotate"] = (base + rotate_deg) % 360
    if crop is not None:
        d["/CropBox"] = [int(v) if float(v).is_integer() else v
                         for v in crop]
    extra: dict[int, bytes] = {}
    next_local = -1

    def alloc(body: bytes) -> tuple:
        nonlocal next_local
        num = next_local
        next_local -= 1
        extra[num] = body
        return ("ref", num, 0)

    if appends:
        contents = d.get("/Contents")
        if inline_payload is not None and "/Contents" not in d:
            contents = [alloc(b"<< /Length "
                              + str(len(inline_payload)).encode()
                              + b" >>\nstream\n" + inline_payload
                              + b"\nendstream")]
        elif isinstance(contents, tuple):
            contents = [contents]
        elif isinstance(contents, list):
            contents = list(contents)
        else:
            contents = []
        additions: dict[str, dict[str, tuple]] = {}
        for payload, resources in appends:
            contents.append(alloc(b"<< /Length "
                                  + str(len(payload)).encode()
                                  + b" >>\nstream\n" + payload
                                  + b"\nendstream"))
            for kind, entries in resources.items():
                for name, body in entries.items():
                    additions.setdefault(kind, {})[name] = alloc(body)
        d["/Contents"] = contents
        _merge_resources(d, closure, additions)
    return _OutPage(body=_render_value(d), closure=closure,
                    mediabox=page.mediabox, extra=extra)


def _stamp_stream(text: str, box: tuple[float, float, float, float],
                  position: str, *, size: int = 10) -> bytes:
    """A content stream drawing ``text`` at ``position`` on the page."""
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    safe = text.encode("latin-1", "replace").decode("latin-1")
    escaped = safe.replace("\\", "\\\\").replace("(", "\\(").replace(
        ")", "\\)")
    # Helvetica averages ~0.5 em per glyph — good enough for a stamp
    est = len(safe) * size * 0.5
    if position == "bottom-center":
        x, y = x0 + max(0.0, (width - est) / 2), y0 + 36
    elif position == "bottom-right":
        x, y = x1 - est - 36, y0 + 36
    elif position == "bottom-left":
        x, y = x0 + 36, y0 + 36
    elif position == "top-center":
        x, y = x0 + max(0.0, (width - est) / 2), y1 - 48
    elif position == "top-right":
        x, y = x1 - est - 36, y1 - 48
    elif position == "top-left":
        x, y = x0 + 36, y1 - 48
    elif position == "center":
        x, y = x0 + max(0.0, (width - est) / 2), y0 + height / 2
        size = 48
    else:
        raise _fail(f"unknown stamp position {position!r}; use "
                    "bottom-center, bottom-right, bottom-left, "
                    "top-center, top-right, top-left, or center",
                    kind="invalid_argument")
    gray = 0.5 if position == "center" else 0.3
    return (f"q\nBT /Fpw {size} Tf {gray} g\n"
            f"{x:.1f} {y:.1f} Td\n({escaped}) Tj\nET\nQ").encode("latin-1")


_STAMP_FONT = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
               b"/Encoding /WinAnsiEncoding >>")


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Width/height out of a JPEG's SOF marker, stdlib only."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > len(data):
            return None
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            if i + 9 > len(data):
                return None
            height = struct.unpack(">H", data[i + 5:i + 7])[0]
            width = struct.unpack(">H", data[i + 7:i + 9])[0]
            return width, height
        i += 2 + length
    return None


def _image_xobject(image: bytes, *, jpeg_quality: int = 85
                   ) -> tuple[bytes, int, int]:
    """(image XObject body, width, height) for JPEG/PNG bytes.

    JPEG embeds directly (DCTDecode); anything else goes through
    Pillow to JPEG when available, else an honest refusal.
    """
    if image[:2] == b"\xff\xd8":
        dims = _jpeg_dimensions(image)
        if dims is None and _pillow_available():
            from PIL import Image
            with Image.open(io.BytesIO(image)) as img:
                dims = img.size
        if dims is None:
            raise _fail("could not read this JPEG's dimensions",
                        kind="parse")
        width, height = dims
        body = (f"<< /Type /XObject /Subtype /Image /Width {width} "
                f"/Height {height} /ColorSpace /DeviceRGB "
                f"/BitsPerComponent 8 /Filter /DCTDecode "
                f"/Length {len(image)} >>\nstream\n").encode() \
            + image + b"\nendstream"
        return body, width, height
    if not _pillow_available():
        raise _fail("this image needs Pillow to embed (only JPEG embeds "
                    "directly); Pillow is not installed here",
                    kind="missing_dependency")
    from PIL import Image
    try:
        with Image.open(io.BytesIO(image)) as img:
            img = img.convert("RGB")
            width, height = img.size
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            payload = buf.getvalue()
    except Exception as exc:
        raise _fail(f"could not read this image: {exc}",
                    kind="parse") from None
    body = (f"<< /Type /XObject /Subtype /Image /Width {width} "
            f"/Height {height} /ColorSpace /DeviceRGB "
            f"/BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(payload)} >>\nstream\n").encode() \
        + payload + b"\nendstream"
    return body, width, height


# ── ops (Studio D1 idiom: source + ordered op list, replayable) ───────────


@dataclass(frozen=True)
class MergeOp:
    """Merge every page of every source, in source order."""


@dataclass(frozen=True)
class SelectOp:
    ranges: str  # "1-3,5" / "all", 1-based, inclusive
    source: int = 0


@dataclass(frozen=True)
class DeleteOp:
    pages: str  # ranges to drop, 1-based
    source: int = 0


@dataclass(frozen=True)
class ReorderOp:
    order: list[int]  # 1-based page numbers in new order
    source: int = 0


@dataclass(frozen=True)
class RotateOp:
    pages: str  # ranges, 1-based
    degrees: int  # multiple of 90
    source: int = 0


@dataclass(frozen=True)
class CropOp:
    pages: str
    box: tuple[float, float, float, float]
    source: int = 0


@dataclass(frozen=True)
class StampOp:
    text: str  # may contain {n} page number and {total} page count
    pages: str = "all"
    position: str = "bottom-center"
    source: int = 0


@dataclass(frozen=True)
class CompressOp:
    level: int = 9
    jpeg_quality: int | None = 75


@dataclass(frozen=True)
class RepairOp:
    source: int = 0


@dataclass(frozen=True)
class ProtectOp:
    password: str
    owner_password: str | None = None
    permissions: int = -4


@dataclass(frozen=True)
class UnlockOp:
    password: str
    source: int = 0


@dataclass(frozen=True)
class ArchiveOp:
    level: str = "1b"
    password: str | None = None


@dataclass(frozen=True)
class SignOp:
    image: bytes | str  # image bytes, or a path to one
    page: int = 1
    rect: tuple[float, float, float, float] = (400, 700, 120, 40)
    source: int = 0


@dataclass(frozen=True)
class ScanOp:
    images: tuple[bytes | str, ...] = ()


def _parse_ranges(spec: str, total: int, what: str) -> list[int]:
    """1-based page ranges ("1-3,5", "all") -> sorted page indices."""
    if not spec or not str(spec).strip():
        raise _fail(f"{what}: empty page range", kind="invalid_argument")
    if str(spec).strip().lower() == "all":
        return list(range(1, total + 1))
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                raise _fail(f"{what}: bad range {part!r}",
                            kind="invalid_argument") from None
            if lo < 1 or hi > total or lo > hi:
                raise _fail(f"{what}: range {part!r} is outside 1..{total}",
                            kind="invalid_argument")
            out.extend(range(lo, hi + 1))
        else:
            try:
                num = int(part)
            except ValueError:
                raise _fail(f"{what}: bad page {part!r}",
                            kind="invalid_argument") from None
            if num < 1 or num > total:
                raise _fail(f"{what}: page {num} is outside 1..{total}",
                            kind="invalid_argument")
            out.append(num)
    if not out:
        raise _fail(f"{what}: empty page range", kind="invalid_argument")
    return sorted(set(out))


# ── job runner ──────────────────────────────────────────────────────────────


@dataclass
class PagewiseJob:
    sources: list[bytes | str]
    ops: list[Any] = field(default_factory=list)
    title: str | None = None
    author: str | None = None


@dataclass
class PagewiseResult:
    data: bytes
    pages: int
    size_before: int
    size_after: int
    measurements: list[dict[str, Any]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


@dataclass
class _WorkPage:
    page: _Page
    closure: dict[int, bytes]
    source_index: int
    rotate: int = 0
    crop: tuple[float, float, float, float] | None = None
    appends: list[tuple[bytes, dict[str, dict[str, bytes]]]] = field(
        default_factory=list)


def _read_source_bytes(source: bytes | str) -> bytes:
    if isinstance(source, bytes):
        return source
    try:
        return Path(source).read_bytes()
    except OSError as exc:
        raise _fail(f"cannot read {source}: {exc}", kind="io") from None


def _decrypt_source(data: bytes, src: _Source,
                    password: str) -> _Source:
    """Return a clean (decrypted) copy of an encrypted source."""
    enc = src.encrypt
    if not enc:
        raise _fail("this PDF is not encrypted", kind="invalid_argument")
    owner = enc.get("/O")
    user = enc.get("/U")
    perms = enc.get("/P", -4)
    version = enc.get("/V", 1)
    revision = enc.get("/R", 3)
    if version != 1 or revision != 3:
        raise _fail("unsupported encryption handler — only the 40-bit "
                    "RC4 standard handler (V=1, R=3) is supported",
                    kind="parse")
    if not isinstance(owner, bytes) or not isinstance(user, bytes):
        raise _fail("malformed /Encrypt dictionary", kind="parse")
    file_id = src.file_id or b"\x00" * 16
    file_key = _rc4_file_key(password, owner, int(perms), file_id)
    if _rc4(file_key, _PADDING) != user:
        raise _fail("wrong password for this PDF", kind="invalid_argument")
    objects: dict[int, bytes] = {}
    for num, body in src.objects.items():
        d = _parse_dict(body)
        if d.get("/Filter") == "/Standard" and "/O" in d:
            objects[num] = body  # the /Encrypt dict itself, untouched
            continue
        dpart, payload = _split_stream(body)
        clean_dpart = _decrypt_strings_in_dict(dpart, num, file_key)
        if payload is not None:
            key = _rc4_object_key(file_key, num)
            objects[num] = (clean_dpart + b"\nstream\n"
                            + _rc4(key, payload) + b"\nendstream")
        else:
            objects[num] = clean_dpart
    pages: list[_Page] = []
    stats: dict[str, int] = {"lost": 0}
    if src.root_num is not None:
        root_d = _parse_dict(objects[src.root_num])
        pages_ref = root_d.get("/Pages")
        if isinstance(pages_ref, tuple) and pages_ref[0] == "ref":
            seen: set[int] = {pages_ref[1]}
            _walk_tree(objects, pages_ref[1], pages, seen, None, None,
                       stats)
    if not pages:
        for num in sorted(objects):
            d = _parse_dict(objects[num])
            if "/Contents" in d and "/Kids" not in d:
                pages.append(_Page(num=num, body=objects[num]))
    return _Source(objects=objects, root_num=src.root_num, pages=pages,
                   encrypt=None, info=src.info, file_id=src.file_id,
                   corruption=src.corruption, lost_pages=stats["lost"])


def _ensure_work(work: list[_WorkPage], sources: list[_Source],
                 closures: list[dict[int, bytes]],
                 source: int) -> list[_WorkPage]:
    """The working set a page op acts on.

    Page ops act on the working set built so far; when it is empty,
    a single-source job implicitly starts from that source's pages,
    so "rotate page 1 of this PDF" needs no explicit MergeOp.
    """
    if work:
        return work
    if len(sources) > 1:
        raise _fail("merge the sources first (MergeOp) — page ops need "
                    "a working set", kind="invalid_argument")
    if source >= len(sources):
        raise _fail(f"no source {source + 1}", kind="invalid_argument")
    for page in sources[source].pages:
        work.append(_WorkPage(page=page, closure=closures[source],
                              source_index=source))
    return work


def _count_markers(data: bytes) -> int:
    return len({int(m.group(1)) for m in
                re.finditer(rb"(\d+)\s+0\s+obj\b", data)})


_XMP_TEMPLATE = """<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Devon pagewise">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:pdfaid="http://www.aiim.org/pdfa/ns/id/">
   <pdfaid:part>1</pdfaid:part>
   <pdfaid:conformance>B</pdfaid:conformance>
  </rdf:Description>
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/">
   <dc:format>application/pdf</dc:format>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


def _find_srgb_icc() -> bytes | None:
    """An sRGB ICC profile from the system, when one is installed."""
    import glob

    for pattern in ("/usr/share/color/icc/*.icc",
                    "/usr/share/color/icc/colord/*.icc",
                    "/usr/share/color/icc/*/*.icc"):
        for path in sorted(glob.glob(pattern)):
            if "srgb" in Path(path).name.lower():
                try:
                    return Path(path).read_bytes()
                except OSError:
                    continue
    return None


def _unembedded_fonts(closures: list[dict[int, bytes]]) -> list[str]:
    """Base font names with no embedded font program."""
    names: list[str] = []
    seen: set[int] = set()
    for closure in closures:
        for num, body in closure.items():
            if num in seen:
                continue
            seen.add(num)
            d = _parse_dict(body)
            base = d.get("/BaseFont")
            if not isinstance(base, str):
                continue
            descriptor = d.get("/FontDescriptor")
            embedded = False
            if isinstance(descriptor, tuple):
                target = closure.get(descriptor[1])
                if target is not None:
                    dd = _parse_dict(target)
                    embedded = any(k in dd for k in
                                   ("/FontFile", "/FontFile2",
                                    "/FontFile3"))
            if not embedded and base not in names:
                names.append(base.lstrip("/"))
    return names


def run_pagewise(job: PagewiseJob) -> PagewiseResult:
    if not job.ops:
        raise _fail("a pagewise job needs at least one op",
                    kind="invalid_argument")
    if not job.sources and not any(isinstance(op, ScanOp)
                                   for op in job.ops):
        raise _fail("a pagewise job needs at least one source",
                    kind="invalid_argument")
    raw = [_read_source_bytes(s) for s in job.sources]
    size_before = sum(len(b) for b in raw)
    passwords: dict[int, str] = {}
    for op in job.ops:
        if isinstance(op, UnlockOp):
            passwords[op.source] = op.password
        elif isinstance(op, ArchiveOp) and op.password is not None:
            passwords.setdefault(0, op.password)
    sources: list[_Source] = []
    for i, data in enumerate(raw):
        src = _load_source(data)
        if src.encrypt is not None:
            pw = passwords.get(i)
            if pw is None:
                raise _fail(f"source {i + 1} is password-protected; "
                            "unlock it first (UnlockOp)",
                            kind="invalid_argument")
            src = _decrypt_source(data, src, pw)
        elif any(isinstance(op, UnlockOp) and op.source == i
                 for op in job.ops):
            raise _fail(f"source {i + 1} is not encrypted",
                        kind="invalid_argument")
        sources.append(src)
    closures: list[dict[int, bytes]] = []
    for src in sources:
        seeds = [p.num for p in src.pages]
        seed_set = set(seeds)
        for page in src.pages:
            if page.resources is not None:
                _dict_refs(page.resources, seed_set)
        closures.append(_closure(src.objects, sorted(seed_set)))

    work: list[_WorkPage] = []
    measurements: list[dict[str, Any]] = []
    labels: list[str] = []
    protect: ProtectOp | None = None
    compress: CompressOp | None = None
    archive: ArchiveOp | None = None
    for op in job.ops:
        if isinstance(op, MergeOp):
            for si, src in enumerate(sources):
                for page in src.pages:
                    work.append(_WorkPage(page=page, closure=closures[si],
                                          source_index=si))
        elif isinstance(op, SelectOp):
            target = _ensure_work(work, sources, closures, op.source)
            idxs = _parse_ranges(op.ranges, len(target), "select")
            work = [target[i - 1] for i in idxs]
        elif isinstance(op, DeleteOp):
            target = _ensure_work(work, sources, closures, op.source)
            drop = set(_parse_ranges(op.pages, len(target), "delete"))
            work = [p for i, p in enumerate(target, 1) if i not in drop]
        elif isinstance(op, ReorderOp):
            target = _ensure_work(work, sources, closures, op.source)
            if sorted(op.order) != list(range(1, len(target) + 1)):
                raise _fail("reorder must list every page exactly once",
                            kind="invalid_argument")
            if any(i < 1 or i > len(target) for i in op.order):
                raise _fail("reorder page outside 1.."
                            f"{len(target)}", kind="invalid_argument")
            work = [target[i - 1] for i in op.order]
        elif isinstance(op, RotateOp):
            if op.degrees % 90:
                raise _fail("rotate needs a multiple of 90 degrees",
                            kind="invalid_argument")
            target = _ensure_work(work, sources, closures, op.source)
            for i in _parse_ranges(op.pages, len(target), "rotate"):
                page = target[i - 1]
                page.rotate = (page.rotate + op.degrees) % 360
        elif isinstance(op, CropOp):
            target = _ensure_work(work, sources, closures, op.source)
            for i in _parse_ranges(op.pages, len(target), "crop"):
                target[i - 1].crop = op.box
        elif isinstance(op, StampOp):
            target = _ensure_work(work, sources, closures, op.source)
            total = len(target)
            for i in _parse_ranges(op.pages, total, "stamp"):
                text = op.text.replace("{n}", str(i)).replace(
                    "{total}", str(total))
                payload = _stamp_stream(text, target[i - 1].page.mediabox,
                                        op.position)
                target[i - 1].appends.append(
                    (payload, {"/Font": {"/Fpw": _STAMP_FONT}}))
        elif isinstance(op, SignOp):
            target = _ensure_work(work, sources, closures, op.source)
            if op.page < 1 or op.page > len(target):
                raise _fail(f"sign page {op.page} is outside 1.."
                            f"{len(target)}", kind="invalid_argument")
            image = op.image
            if isinstance(image, (str, Path)):
                image = _read_source_bytes(image)
            body, _w, _h = _image_xobject(image)
            x, y, rw, rh = op.rect
            draw = (f"q {rw} 0 0 {rh} {x} {y} cm /ImS Do Q"
                    ).encode("latin-1")
            target[op.page - 1].appends.append(
                (draw, {"/XObject": {"/ImS": body}}))
        elif isinstance(op, CompressOp):
            compress = op
        elif isinstance(op, RepairOp):
            if work:
                raise _fail("repair applies to a source before other ops",
                            kind="invalid_argument")
            src = sources[op.source]
            measurements.append({
                "op": "repair",
                "source": op.source,
                "recovered_pages": len(src.pages),
                "lost_pages": src.lost_pages,
                "recovered_objects": len(src.objects),
                "total_objects": _count_markers(raw[op.source]),
                "corruption": list(src.corruption),
            })
            for page in src.pages:
                work.append(_WorkPage(page=page,
                                      closure=closures[op.source],
                                      source_index=op.source))
        elif isinstance(op, ProtectOp):
            if not op.password:
                raise _fail("protect needs a non-empty password",
                            kind="invalid_argument")
            protect = op
        elif isinstance(op, UnlockOp):
            pass  # handled at load time
        elif isinstance(op, ArchiveOp):
            archive = op
        elif isinstance(op, ScanOp):
            for image in op.images:
                if isinstance(image, (str, Path)):
                    image = _read_source_bytes(image)
                body, width, height = _image_xobject(image)
                content = (f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q"
                           ).encode()
                content_body = (
                    b"<< /Length " + str(len(content)).encode()
                    + b" >>\nstream\n" + content + b"\nendstream")
                page_d = {
                    "/Type": "/Page",
                    "/Parent": ("ref", 2, 0),
                    "/MediaBox": [0, 0, width, height],
                    "/Resources": {"/XObject": {"/Im0": ("ref", -1, 0)}},
                    "/Contents": ("ref", -2, 0),
                }
                closure = {-1: body, -2: content_body}
                page = _Page(num=-1, body=_render_value(page_d),
                             mediabox=(0, 0, width, height))
                work.append(_WorkPage(page=page, closure=closure,
                                      source_index=-1))
        else:
            raise _fail(f"unknown pagewise op {op!r}",
                        kind="invalid_argument")

    if not work:
        # no page op touched anything: the job is its sources, whole
        for si, src in enumerate(sources):
            for page in src.pages:
                work.append(_WorkPage(page=page, closure=closures[si],
                                      source_index=si))
    if not work:
        raise _fail("this job produced no pages", kind="invalid_argument")

    out_pages: list[_OutPage] = []
    for wp in work:
        out_pages.append(_prepare_page(
            wp.page, wp.closure, rotate_deg=wp.rotate, crop=wp.crop,
            appends=wp.appends or None))

    metadata_obj: bytes | None = None
    extra_catalog: dict[str, Any] = {}
    output_intent_icc: bytes | None = None
    if archive is not None:
        xmp = _XMP_TEMPLATE.encode()
        metadata_obj = (
            b"<< /Type /Metadata /Subtype /XML /Length "
            + str(len(xmp)).encode() + b" >>\nstream\n" + xmp
            + b"\nendstream")
        extra_catalog["/MarkInfo"] = {"/Marked": True}
        extra_catalog["/Lang"] = b"en-US"
        # strip non-archival features the fresh catalog does not copy
        # anyway; page-level /AA is dropped from each page dict
        stripped: list[str] = []
        for out_page in out_pages:
            dpart, _payload = _split_stream(out_page.body)
            d = _parse_dict(dpart)
            if "/AA" in d:
                d.pop("/AA")
                stripped.append("AA")
                out_page.body = _render_value(d)
        arch_labels = [f"pdfa_level: {archive.level}",
                       "xmp_metadata: embedded",
                       "markinfo: marked",
                       "lang: en-US"]
        if stripped:
            arch_labels.append("stripped: " + ",".join(sorted(set(
                stripped))))
        fonts = _unembedded_fonts([wp.closure for wp in work])
        for name in fonts:
            arch_labels.append(f"font not embedded: {name}")
        icc = _find_srgb_icc()
        if icc is not None:
            output_intent_icc = icc
            arch_labels.append("output_intent: sRGB embedded")
        else:
            arch_labels.append(
                "output_intent_missing: no sRGB ICC profile installed "
                "on this machine — archival shape without OutputIntent")
        arch_labels.append("validated: false (conformance validation "
                           "is a separate, unbuilt rung)")
        measurements.append({"op": "archive", "level": archive.level,
                             "labels": arch_labels})
        labels.extend(arch_labels)

    encrypt_body: bytes | None = None
    file_key: bytes | None = None
    file_id: bytes | None = None
    if protect is not None:
        file_id = secrets.token_bytes(16)
        owner_key = _rc4_owner_key(
            protect.owner_password or protect.password, protect.password)
        file_key = _rc4_file_key(protect.password, owner_key,
                                 protect.permissions, file_id)
        user_key = _rc4(file_key, _PADDING)
        encrypt_body = (
            b"<< /Filter /Standard /V 1 /R 3 /Length 40 /O <"
            + owner_key.hex().encode() + b"> /U <"
            + user_key.hex().encode() + b"> /P "
            + str(protect.permissions).encode() + b" >>")
        labels.append("encrypted: RC4 40-bit (V=1, R=3)")

    data = _emit_pdf(
        out_pages, title=job.title, author=job.author,
        extra_catalog=extra_catalog or None,
        metadata_obj=metadata_obj, encrypt=encrypt_body,
        file_key=file_key, file_id=file_id,
        compress_level=compress.level if compress else 0,
        jpeg_quality=compress.jpeg_quality if compress else None,
        output_intent_icc=output_intent_icc)

    if compress is not None:
        measurements.append({
            "op": "compress",
            "level": compress.level,
            "jpeg_quality": compress.jpeg_quality,
            "size_before": size_before,
            "size_after": len(data),
            "saved_bytes": size_before - len(data),
        })
    return PagewiseResult(data=data, pages=len(out_pages),
                          size_before=size_before, size_after=len(data),
                          measurements=measurements, labels=labels)


# ── convenience entry points ────────────────────────────────────────────────


def pdf_info(data: bytes) -> dict[str, Any]:
    src = _load_source(data)
    info: dict[str, Any] = {
        "pages": len(src.pages),
        "encrypted": src.encrypt is not None,
        "size_bytes": len(data),
        "corruption": list(src.corruption),
        "lost_pages": src.lost_pages,
    }
    if src.encrypt is not None:
        info["encryption"] = {"version": src.encrypt.get("/V"),
                              "revision": src.encrypt.get("/R")}
    elif src.info:
        title = src.info.get("/Title")
        if isinstance(title, bytes):
            info["title"] = title.decode("latin-1", "replace")
        author = src.info.get("/Author")
        if isinstance(author, bytes):
            info["author"] = author.decode("latin-1", "replace")
    return info


def extract_page_texts(data: bytes) -> list[str]:
    src = _load_source(data)
    if src.encrypt is not None:
        raise _fail("this PDF is encrypted; unlock it first",
                    kind="invalid_argument")
    return [_page_text(page.body, src.objects).strip()
            for page in src.pages]


def repair_pdf(data: bytes) -> bytes:
    return run_pagewise(PagewiseJob(sources=[data],
                                    ops=[RepairOp()])).data


def protect_pdf(data: bytes, password: str,
                owner_password: str | None = None) -> bytes:
    if not password:
        raise _fail("protect needs a non-empty password",
                    kind="invalid_argument")
    return run_pagewise(PagewiseJob(
        sources=[data],
        ops=[ProtectOp(password,
                       owner_password=owner_password)])).data


def unlock_pdf(data: bytes, password: str) -> bytes:
    return run_pagewise(PagewiseJob(sources=[data],
                                    ops=[UnlockOp(password)])).data


def archive_pdf(data: bytes, password: str | None = None) -> bytes:
    return run_pagewise(PagewiseJob(
        sources=[data], ops=[ArchiveOp(password=password)])).data


def scan_to_pdf(images: list[bytes | str], *,
                title: str | None = None) -> bytes:
    if not images:
        raise _fail("scan needs at least one image",
                    kind="invalid_argument")
    return run_pagewise(PagewiseJob(
        sources=[], ops=[ScanOp(images=tuple(images))],
        title=title)).data


def format_pagewise_reply(result: PagewiseResult, *,
                          what: str = "PDF") -> str:
    """Plain-words summary of a pagewise result, for chat."""
    lines = [f"Done — {what}: {result.pages} page"
             f"{'s' if result.pages != 1 else ''}, "
             f"{result.size_after} bytes."]
    for measurement in result.measurements:
        op = measurement.get("op")
        if op == "compress":
            lines.append(
                f"Compressed: saved {measurement['saved_bytes']} bytes "
                f"({measurement['size_before']} → "
                f"{measurement['size_after']}).")
        elif op == "repair":
            recovered = measurement["recovered_pages"]
            lost = measurement.get("lost_pages", 0)
            if lost:
                lines.append(f"Repaired: recovered {recovered} page(s); "
                             f"{lost} page(s) were lost — their bytes "
                             "are gone and cannot be rebuilt.")
            else:
                lines.append(f"Repaired: all {recovered} page(s) "
                             "recovered.")
        elif op == "archive":
            lines.append("Archived: PDF/A shape written (XMP metadata, "
                         "marked, language set).")
    if any("encrypted" in label for label in result.labels):
        lines.append("The file is password-protected (RC4 40-bit).")
    return "\n".join(lines)
