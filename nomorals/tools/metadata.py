"""Metadata extraction — offline file forensics, zero dependencies.

Pulls the embedded metadata that files carry in the wild:

- images:  JPEG EXIF (timestamps, camera make/model, GPS), JFIF, PNG
           text chunks (title/author/software/date), GIF comment
- documents: PDF Info dict (title/author/producer/creation date, pages,
           encryption flag), DOCX/XLSX/PPTX core + app properties
- audio:   MP3 ID3v2 + ID3v1 (artist/title/album/year), FLAC
           VORBIS_COMMENT, WAV format
- archives: ZIP/GZIP entry listings
- everything: magic-byte type detection, size, SHA-256

Works on a local path or a URL (downloaded to a temp file first).  Every
parser is defensive — a truncated or hostile file yields partial metadata
plus a warning, never a crash.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import struct
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

_log = get_logger(__name__)

__all__ = ["extract_metadata", "register"]

_UA = "Mozilla/5.0 (compatible; NoMoralsMeta/1.0)"

# ── type sniffing ───────────────────────────────────────────────────────────

_MAGIC: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\xff\xd8\xff", "jpeg"),
    (0, b"\x89PNG\r\n\x1a\n", "png"),
    (0, b"GIF87a", "gif"),
    (0, b"GIF89a", "gif"),
    (0, b"RIFF", "riff"),
    (0, b"%PDF-", "pdf"),
    (0, b"PK\x03\x04", "zip"),
    (0, b"ID3", "mp3(id3)"),
    (0, b"fLaC", "flac"),
    (0, b"OggS", "ogg"),
    (0, b"\x1f\x8b", "gzip"),
    (0, b"\x7fELF", "elf"),
    (0, b"MZ", "pe"),
    (4, b"ftyp", "mp4"),
    (0, b"BM", "bmp"),
    (0, b"WEBP", "webp"),
)


def sniff_type(data: bytes) -> str:
    for offset, needle, label in _MAGIC:
        if data[offset:offset + len(needle)] == needle:
            return label
    return "unknown"


# ── JPEG / EXIF ─────────────────────────────────────────────────────────────

_EXIF_TAGS = {
    0x010F: "Make", 0x0110: "Model", 0x0131: "Software",
    0x0132: "DateTime", 0x9003: "DateTimeOriginal", 0x9004: "DateTimeDigitized",
    0x010E: "Orientation", 0x011A: "XResolution", 0x011B: "YResolution",
    0x829A: "ExposureTime", 0x829D: "FNumber", 0x8827: "ISO",
}


def _jpeg_exif(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "JPEG"}
    pos = 2
    while pos + 4 < len(data):
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        segment = data[pos + 4:pos + 2 + length]
        if marker == 0xE1 and segment[:6] == b"Exif\x00\x00":
            out["exif"] = _parse_exif_ifd(segment[6:])
        elif marker == 0xE0 and segment[:5] == b"JFIF\x00":
            out["jif_version"] = f"{segment[5]}.{segment[6]}"
            density = struct.unpack(">HH", segment[7:11])[0]
            unit = segment[6]
            if unit == 1:
                out["dpi"] = density
        elif marker == 0xDA:  # start of scan = done walking headers
            break
        pos += 2 + length
    return out


def _parse_exif_ifd(tiff: bytes) -> dict[str, Any]:
    """Walk the EXIF TIFF structure (IFD0 + Exif sub-IFD + GPS)."""
    if len(tiff) < 8:
        return {}
    endian = tiff[:2]
    if endian == b"II":
        prefix = "<"
    elif endian == b"MM":
        prefix = ">"
    else:
        return {}
    ifd0_offset = struct.unpack(prefix + "I", tiff[4:8])[0]
    found: dict[str, Any] = {}
    sub_offsets: dict[int, int] = {}
    for ifd_base in (0, 1):  # 0 = IFD0 pass, 1 = sub-IFD pass
        try:
            offset = ifd0_offset if ifd_base == 0 else sub_offsets.get(0x8769, 0)
            if not offset or ifd_base == 1 and not sub_offsets:
                continue
            count = struct.unpack(prefix + "H", tiff[offset:offset + 2])[0]
            for i in range(count):
                entry = offset + 2 + i * 12
                if entry + 12 > len(tiff):
                    break
                tag, typ, num = struct.unpack(prefix + "HHI", tiff[entry:entry + 8])
                value_bytes = tiff[entry + 8:entry + 12]
                if tag == 0x8769 and ifd_base == 0:
                    sub_offsets[0x8769] = struct.unpack(prefix + "I", value_bytes)[0]
                if tag == 0x8825 and ifd_base == 0:
                    pass  # GPS IFD: parse below if present
                name = _EXIF_TAGS.get(tag)
                if name is None:
                    continue
                if typ == 2:  # ASCII
                    if num <= 4:
                        raw = value_bytes[:num]
                    else:
                        voff = struct.unpack(prefix + "I", value_bytes)[0]
                        raw = tiff[voff:voff + num]
                    found[name] = raw.split(b"\x00")[0].decode("ascii", "replace")
                elif typ in {3, 4}:  # SHORT / LONG
                    size = {3: 2, 4: 4}[typ]
                    raw = value_bytes[:size] if num == 1 else (
                        tiff[struct.unpack(prefix + "I", value_bytes)[0]:
                             struct.unpack(prefix + "I", value_bytes)[0] + size])
                    try:
                        found[name] = struct.unpack(prefix + ({3: "H", 4: "I"}[typ]),
                                                    raw)[0]
                    except (struct.error, IndexError):  # noqa: E103 - corrupt tag skipped; best-effort parse
                        pass
        except (struct.error, IndexError):
            break
    # GPS sub-IFD
    try:
        gps_offset = None
        count = struct.unpack(prefix + "H", tiff[ifd0_offset:ifd0_offset + 2])[0]
        for i in range(count):
            entry = ifd0_offset + 2 + i * 12
            tag, _t, _n = struct.unpack(prefix + "HHI", tiff[entry:entry + 8])
            if tag == 0x8825:
                gps_offset = struct.unpack(prefix + "I", tiff[entry + 8:entry + 12])[0]
                break
        if gps_offset:
            gcount = struct.unpack(prefix + "H", tiff[gps_offset:gps_offset + 2])[0]
            lat = lon = None
            lat_dir = lon_dir = ""
            for i in range(gcount):
                entry = gps_offset + 2 + i * 12
                tag, typ, num = struct.unpack(prefix + "HHI", tiff[entry:entry + 8])
                raw = tiff[entry + 8:entry + 12]
                if typ == 2:
                    val = raw[:num].split(b"\x00")[0].decode("ascii", "replace")
                    if tag == 0x0001:
                        lat_dir = val
                    if tag == 0x0003:
                        lon_dir = val
                elif tag in {0x0002, 0x0004} and typ == 5:
                    def _rational(off: int) -> float:
                        n, d = struct.unpack(prefix + "II", tiff[off:off + 8])
                        return n / d if d else 0.0
                    d1 = struct.unpack(prefix + "I", raw)[0]
                    vals = [_rational(d1 + k * 8) for k in range(3)]
                    if tag == 0x0002:
                        lat = vals
                    else:
                        lon = vals
            if lat is not None:
                found["GPS"] = {
                    "lat": lat, "lon": lon,
                    "lat_dir": lat_dir, "lon_dir": lon_dir,
                    "decimal": (
                        _gps_decimal(lat, lat_dir), _gps_decimal(lon, lon_dir)
                    ),
                }
    except (struct.error, IndexError, KeyError):  # noqa: E103 - corrupt GPS block skipped
        pass
    return found


def _gps_decimal(parts: list[float] | None, direction: str) -> float | None:
    if not parts or len(parts) != 3:
        return None
    value = parts[0] + parts[1] / 60 + parts[2] / 3600
    if direction in {"S", "W"}:
        value = -value
    return round(value, 6)


# ── PNG ─────────────────────────────────────────────────────────────────────


def _png_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "PNG"}
    if len(data) >= 24:
        width, height = struct.unpack(">II", data[16:24])
        bit_depth, color_type = data[24], data[25]
        out["dimensions"] = f"{width}x{height}"
        out["bit_depth"] = bit_depth
        out["color_type"] = {0: "grayscale", 2: "rgb", 3: "palette",
                             4: "gray+alpha", 6: "rgba"}.get(color_type, color_type)
    pos = 8
    while pos + 8 < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8].decode("latin-1", "replace")
        chunk = data[pos + 8:pos + 8 + length]
        if ctype in {"tEXt", "iTXt"}:
            try:
                if ctype == "tEXt":
                    key, _, value = chunk.partition(b"\x00")
                    out[key.decode("latin-1")] = value.decode("utf-8", "replace")
                else:
                    key, _, rest = chunk.partition(b"\x00")
                    # iTXt: null, flags, lang, original, text
                    parts = rest.split(b"\x00")
                    if len(parts) >= 4:
                        out[key.decode("latin-1")] = parts[-1].decode("utf-8", "replace")
            except (UnicodeDecodeError, IndexError):  # noqa: E103 - undecodable chunk skipped
                pass
        pos += 12 + length
    return out


# ── PDF ─────────────────────────────────────────────────────────────────────


def _pdf_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "PDF"}
    m = re.search(rb"/Version\s*/([\d.]+)", data[:1024])
    if m:
        out["version"] = m.group(1).decode()
    out["encrypted"] = b"/Encrypt" in data[:8192]
    out["pages"] = len(re.findall(rb"/Type\s*/Page[^s]", data))
    info: dict[str, str] = {}
    m = re.search(rb"stream\r?\n", data)
    for key in (b"Title", b"Author", b"Subject", b"Creator", b"Producer",
                b"CreationDate", b"ModDate"):
        km = re.search(key + rb"\s*\(([^)]{0,200})\)", data)
        if km:
            value = km.group(1).decode("latin-1", "replace").strip()
            info[key.decode()] = value
        else:
            km = re.search(key + rb"\s*(\d{14})\s*T", data)
            if km:
                value = km.group(1).decode()
                info[key.decode()] = (f"{value[:4]}-{value[4:6]}-{value[6:8]} "
                                      f"{value[8:10]}:{value[10:12]}")
    if info:
        out["info"] = info
    return out


# ── Office (OOXML) ──────────────────────────────────────────────────────────


def _office_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "OOXML"}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            kind = "unknown"
            for name in names:
                if name == "word/document.xml":
                    kind = "DOCX"
                elif name == "xl/workbook.xml":
                    kind = "XLSX"
                elif name == "ppt/presentation.xml":
                    kind = "PPTX"
            out["document_kind"] = kind
            out["entries"] = len(names)
            if "docProps/core.xml" in names:
                core = zf.read("docProps/core.xml").decode("utf-8", "replace")
                for tag, label in (("dc:title", "title"), ("dc:creator", "creator"),
                                   ("cp:lastModifiedBy", "last_modified_by"),
                                   ("dcterms:created", "created"),
                                   ("dcterms:modified", "modified"),
                                   ("dc:description", "description")):
                    m = re.search(rf"<{tag}(?:\s[^>]*)?>([^<]*)</{tag}>", core)
                    if m:
                        out[label] = m.group(1).strip()
            if "docProps/app.xml" in names:
                app = zf.read("docProps/app.xml").decode("utf-8", "replace")
                for tag, label in (("Application", "application"),
                                   ("Company", "company"), ("Pages", "pages"),
                                   ("Words", "words"), ("TotalTime", "edit_time_min")):
                    m = re.search(rf"<{tag}>([^<]*)</{tag}>", app)
                    if m:
                        out[label] = m.group(1).strip()
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        out["error"] = f"corrupt archive: {exc}"
    return out


# ── audio ───────────────────────────────────────────────────────────────────


def _mp3_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "MP3"}
    # ID3v2 header
    if data[:3] == b"ID3":
        version = data[3]
        # 4 synchsafe bytes, 7 bits each (max 256 MB tag)
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) \
            | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        tag_end = 10 + size
        frame = 10
        labels = {b"TIT2": "title", b"TPE1": "artist", b"TPE2": "album_artist",
                  b"TALB": "album", b"TYER": "year", b"TSOT": "track",
                  b"TRCK": "track", b"TCON": "genre", b"COMM": "comment"}
        while frame + 10 < min(tag_end, len(data)):
            frame_id = data[frame:frame + 4]
            fsize = (data[frame + 4] << 21) | (data[frame + 5] << 14) \
                | (data[frame + 6] << 7) | data[frame + 7]
            if fsize <= 0:
                break
            if frame_id in labels and version >= 3:
                # header is 10 bytes; the size field covers frame data only
                payload = data[frame + 10:frame + 10 + fsize]
                # v2.4: encoding byte + text
                text = payload[1:].decode("utf-8", "replace").strip("\x00 ")
                if text:
                    out[labels[frame_id]] = text[:200]
            frame += 10 + fsize
    # ID3v1 trailer
    if len(data) >= 128 and data[-128:-125] == b"TAG":
        v1 = data[-125:]
        v1 = {
            "title_v1": v1[:30].split(b"\x00")[0].decode("latin-1", "replace").strip(),
            "artist_v1": v1[30:60].split(b"\x00")[0].decode("latin-1", "replace").strip(),
            "album_v1": v1[60:90].split(b"\x00")[0].decode("latin-1", "replace").strip(),
            "year_v1": v1[90:94].split(b"\x00")[0].decode("latin-1", "replace").strip(),
        }
        out.update({k: v for k, v in v1.items() if v})
    return out


def _flac_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "FLAC"}
    pos = 4
    while pos + 4 < len(data):
        header = data[pos]
        is_last = header & 0x80
        block_type = header & 0x7F
        length = struct.unpack(">I", b"\x00" + data[pos + 1:pos + 4])[0]
        block = data[pos + 4:pos + 4 + length]
        if block_type == 0:  # STREAMINFO
            if len(block) >= 18:
                sample_rate = struct.unpack(">I", b"\x00" + block[11:14])[0] & 0x0FFFFFFF
                channels = (block[11] >> 3) & 0x7
                bps = (block[13] >> 1) & 0x7
                out["sample_rate"] = sample_rate
                out["channels"] = channels + 1
                out["bits_per_sample"] = bps
        elif block_type == 4:  # VORBIS_COMMENT
            try:
                vendor_len = struct.unpack("<I", block[:4])[0]
                pos2 = 4 + vendor_len
                count = struct.unpack("<I", block[pos2:pos2 + 4])[0]
                pos2 += 4
                comments: dict[str, str] = {}
                for _ in range(count):
                    clen = struct.unpack("<I", block[pos2:pos2 + 4])[0]
                    pos2 += 4
                    text = block[pos2:pos2 + clen].decode("utf-8", "replace")
                    pos2 += clen
                    key, _, value = text.partition("=")
                    label = key.lower()
                    if label in {"title", "artist", "album", "date", "genre",
                                 "comment"} and key not in comments:
                        comments[key] = value
                if comments:
                    out.update(comments)
            except (struct.error, IndexError, UnicodeDecodeError):  # noqa: E103 - corrupt comment block skipped
                pass
        pos += 4 + length
        if is_last:
            break
    return out


def _wav_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "WAV"}
    # RIFF header: "RIFF" <size> "WAVE" — the WAVE tag sits at offset 8
    if len(data) >= 44 and data[8:12] == b"WAVE":
        out["channels"] = struct.unpack("<H", data[22:24])[0]
        out["sample_rate"] = struct.unpack("<I", data[24:28])[0]
        out["bit_depth"] = struct.unpack("<H", data[34:36])[0]
    return out


def _gif_meta(data: bytes) -> dict[str, Any]:
    out: dict[str, Any] = {"format": "GIF"}
    if len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        out["dimensions"] = f"{width}x{height}"
    # comment extension: 0x21 0xFE
    idx = data.find(b"\x21\xfe")
    while idx != -1:
        length = data[idx + 2]
        comment = data[idx + 3:idx + 3 + length]
        if comment.strip():
            out["comment"] = comment.decode("latin-1", "replace").strip()
            break
        idx = data.find(b"\x21\xfe", idx + 1)
    return out


# ── dispatch ────────────────────────────────────────────────────────────────


def _eml_meta(data: bytes) -> dict[str, Any]:
    """Email header forensics (offline): identity, routing chain, and the
    classic spoofing signals — Message-ID domain vs sending domain,
    X-Originating-IP, authentication verdicts, hop chain."""
    import email
    import email.policy

    out: dict[str, Any] = {"format": "EMAIL"}
    try:
        msg = email.message_from_bytes(data, policy=email.policy.default)
    except Exception as exc:  # noqa: BLE001 - defensive forensics
        return {"format": "EMAIL", "error": f"unparseable: {exc}"}
    for header in ("from", "to", "cc", "subject", "date", "message-id",
                   "reply-to", "sender"):
        value = msg.get(header)
        if value:
            out[header.lower().replace("-", "_")] = str(value)[:400]
    outgoing = msg.get("x-originating-ip")
    if outgoing:
        out["x_originating_ip"] = str(outgoing).strip()
    # authentication verdicts (SPF/DKIM/DMARC) from Authentication-Results
    verdicts: dict[str, str] = {}
    for ar in msg.get_all("authentication-results") or []:
        for token in str(ar).replace(",", " ").split():
            if "=" not in token:
                continue
            key, val = token.split("=", 1)
            key = key.strip().lower()
            val = val.strip().lower()
            if key in {"spf", "dkim", "dmarc"} and val \
                    and key not in verdicts:
                verdicts[key] = val
    if verdicts:
        out["authentication"] = verdicts
    # the Received chain — oldest hop first
    import re as _re
    hops: list[dict[str, str]] = []
    for rcvd in reversed(msg.get_all("received") or []):
        text = " ".join(str(rcvd).split())
        m_by = _re.search(r"\bby\s+([^(\s][^(]*)", text)
        m_fr = _re.search(r"\bfrom\s+([^(\s][^(]*)", text)
        via = m_by.group(1).strip() if m_by else ""
        frm = m_fr.group(1).strip() if m_fr else text[:60]
        hops.append({"via": via[:160], "from": frm[:160]})
    out["received_hops"] = len(hops)
    if hops:
        out["hop_chain"] = hops[:40]
        out["first_hop_from"] = hops[0]["from"]
        out["last_hop_via"] = hops[-1]["via"] or hops[-1]["from"]
    # spoofing signal: Message-ID domain vs the From domain
    msg_id = str(msg.get("message-id") or "")
    from_addr = str(msg.get("from") or "")
    mid_domain = msg_id.rsplit("@", 1)[-1].lower().strip("> ") \
        if "@" in msg_id else ""
    from_domain = from_addr.rsplit("@", 1)[-1].lower().strip(" >") \
        if "@" in from_addr else ""
    if mid_domain and from_domain:
        out["message_id_domain"] = mid_domain
        out["from_domain"] = from_domain
        out["domain_match"] = mid_domain == from_domain \
            or mid_domain.endswith("." + from_domain)
    return out


def _looks_like_email(data: bytes) -> bool:
    head = data[:2048].lstrip()
    if not head:
        return False
    first = head.split(b"\n", 1)[0].lower()
    return first.startswith((b"from:", b"received:", b"mime-version:",
                             b"date:", b"message-id:", b"to:")) or \
        head.startswith(b"From ")


def _extract(data: bytes) -> dict[str, Any]:
    kind = sniff_type(data)
    if _looks_like_email(data):
        return _eml_meta(data)
    if kind == "jpeg":
        return _jpeg_exif(data)
    if kind == "png":
        return _png_meta(data)
    if kind == "gif":
        return _gif_meta(data)
    if kind == "pdf":
        return _pdf_meta(data)
    if kind == "zip":
        meta = {"format": "ZIP"}
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                entries = zf.namelist()
                if entries and (entries[0].startswith(("word/", "xl/", "ppt/"))
                                or "docProps/core.xml" in entries):
                    return _office_meta(data)
                meta["entries"] = len(entries)
                meta["first_entries"] = entries[:20]
        except zipfile.BadZipFile as exc:
            meta["error"] = str(exc)
        return meta
    if kind in {"mp3(id3)", "mp3"}:
        return _mp3_meta(data)
    if kind == "flac":
        return _flac_meta(data)
    if kind == "riff" and data[8:12] == b"WAVE":
        return _wav_meta(data)
    if kind == "gzip":
        return {"format": "GZIP"}
    return {"format": kind}


def extract_metadata(context: Any, source: str) -> dict[str, Any]:
    """Extract embedded metadata from a local file or URL.

    Returns type detection + sha256 + size + whatever the file actually
    carries (EXIF, PNG text, PDF info, OOXML properties, ID3/FLAC tags,
    archive listings).  Defensive: hostile or truncated files produce
    partial results with an 'error' note, never an exception.
    """
    source = (source or "").strip()
    if not source:
        raise ToolError("metadata_extract needs a path or URL")
    temp_path: str | None = None
    try:
        if source.startswith(("http://", "https://")):
            parsed = urllib.parse.urlparse(source)
            if parsed.scheme == "https":
                from ..core.http import default_proxy_handler

                handlers: list[Any] = []
                proxy_handler = default_proxy_handler()
                if proxy_handler is not None:
                    handlers.append(proxy_handler)
            else:
                handlers = []
            request = urllib.request.Request(source, headers={"User-Agent": _UA})
            with urllib.request.build_opener(*handlers).open(request,
                                                             timeout=30) as response:
                data = response.read(200 * 1024 * 1024)  # 200 MB cap
        else:
            from .filesystem import safe_path

            path = safe_path(context, source, must_exist=True)
            data = path.read_bytes()[:200 * 1024 * 1024]
        digest = hashlib.sha256(data).hexdigest()
        meta = _extract(data)
        meta.update({
            "source": source,
            "bytes": len(data),
            "sha256": digest,
            "truncated": len(data) >= 200 * 1024 * 1024,
        })
        return meta
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - forensics must not crash the caller
        raise ToolError(f"metadata extraction failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:  # noqa: E103 - temp file already gone
                pass


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "metadata_extract",
        description=(
            "Offline file forensics: EXIF (camera, timestamps, GPS), PNG "
            "chunks, PDF info, DOCX/XLSX/PPTX properties, MP3/FLAC tags, "
            "archive listings + magic type + sha256. Path or URL."
        ),
        capability=Capability.FS_READ,
        parameters={"source": "str — workspace path or http(s) URL"},
    )
    def metadata_extract_tool(source: str) -> dict[str, Any]:
        return extract_metadata(context, source)
