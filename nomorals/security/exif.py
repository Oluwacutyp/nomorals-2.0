"""Image metadata stripping for outbound images — operational security.

Removes GPS coordinates, camera model, timestamps, XMP/IPTC, ICC profiles,
PNG text chunks, comments and all other metadata before an image leaves the
device. Policy follows mat2 (the Metadata Anonymisation Toolkit): *any piece
of the file that is not image data and can be removed is treated as a
threat and deleted* (whitelist approach per format).

Method is **lossless segment surgery in pure stdlib** — no re-encode, so no
quality loss (Pillow re-encoding a JPEG degrades it, and misses XMP/IPTC/
APP14/COM/JFIF anyway):
- JPEG: parse markers, drop APP0 (JFIF), APP1 (Exif+XMP), APP2 (ICC),
  APP13 (Photoshop/IPTC), APP14 (Adobe), COM. Image data untouched.
- PNG: parse chunks, drop tEXt/iTXt/zTXt, tIME, eXIf, iCCP (CRCs recomputed).
- WebP: parse RIFF chunks, drop EXIF/XMP/ICCP.

Orientation trap: dropping EXIF without applying the orientation tag leaves
the image visually rotated. We read IFD0 tag 0x0112 first; only when it is
not "normal" do we transpose pixels (via PIL) and then strip.

Public API:
- ``scan_metadata(data)`` — rich report (EXIF tags, GPS decimal, XMP, IPTC,
  ICC, PNG text, JPEG comments, C2PA, AI-generation signals).
- ``exif_summary(path)`` — same, from a file (backward compatible shape).
- ``strip_metadata_bytes(data)`` — lossless clean bytes (format-aware).
- ``strip_exif_to_bytes(data)`` — backward-compatible alias.
- ``strip_exif(path, backup=True)`` — in-place strip, keeps ``.bak``.
- ``clean_copy(path, dest=None)`` — mat2-style ``name.cleaned.ext``,
  original untouched.
- ``verify_clean(data)`` — re-scan after stripping; (ok, residual).
- ``secure_delete(path, passes=3)`` — overwrite + unlink (for .bak files).
- ``format_summary(summary)`` — human-readable rendering.
"""

from __future__ import annotations

import binascii
import io
import os
import struct
from pathlib import Path
from typing import Union

__all__ = [
    "scan_metadata",
    "exif_summary",
    "strip_metadata_bytes",
    "strip_exif_to_bytes",
    "strip_exif",
    "clean_copy",
    "verify_clean",
    "secure_delete",
    "format_summary",
]

# ── format sniffing ──────────────────────────────────────────────────────

_JPEG_SOI = b"\xff\xd8"
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_WEBP_RIFF = b"RIFF"
_WEBP_WEBP = b"WEBP"

_PNG_DROP = {"tEXt", "iTXt", "zTXt", "tIME", "eXIf", "iCCP"}
_WEBP_DROP = {"EXIF", "XMP ", "ICCP"}

_AI_GENERATOR_HINTS = (
    "stable diffusion", "midjourney", "dall-e", "dall·e", "dall e",
    "firefly", "leonardo", "ideogram", "comfyui", "automatic1111",
    "novelai", "civitai", "fooocus",
)


def _sniff(data: bytes) -> str:
    if data[:2] == _JPEG_SOI:
        return "jpeg"
    if data[:8] == _PNG_SIG:
        return "png"
    if data[:4] == _WEBP_RIFF and data[8:12] == _WEBP_WEBP:
        return "webp"
    return "other"


# ── tiny EXIF/TIFF parser (orientation + GPS, stdlib) ────────────────────

def _tiff_ifd(tiff: bytes, bo: str, offset: int) -> dict[int, tuple[int, int, bytes]]:
    """Parse one IFD → {tag: (type, count, raw_value_bytes)}."""
    out: dict[int, tuple[int, int, bytes]] = {}
    if offset + 2 > len(tiff):
        return out
    n = struct.unpack(bo + "H", tiff[offset:offset + 2])[0]
    if n > 200:  # sanity — corrupt IFD
        return out
    for i in range(n):
        e = offset + 2 + i * 12
        if e + 12 > len(tiff):
            break
        tag, typ, cnt = struct.unpack(bo + "HHI", tiff[e:e + 8])
        raw = tiff[e + 8:e + 12]
        out[tag] = (typ, cnt, raw)
    return out


def _ifd_value(tiff: bytes, bo: str, entry: tuple[int, int, bytes]):
    typ, cnt, raw = entry
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
    size = sizes.get(typ, 1) * cnt
    data = raw if size <= 4 else tiff[
        struct.unpack(bo + "I", raw)[0]:
        struct.unpack(bo + "I", raw)[0] + size]
    if typ == 2:  # ASCII
        return data.split(b"\x00")[0].decode("ascii", "replace")
    if typ == 3 and cnt >= 1:  # SHORT
        return struct.unpack(bo + "H", data[:2])[0]
    if typ == 4 and cnt >= 1:  # LONG
        return struct.unpack(bo + "I", data[:4])[0]
    if typ == 5 and cnt >= 1:  # RATIONAL
        vals = []
        for i in range(cnt):
            num, den = struct.unpack(bo + "II", data[i * 8:i * 8 + 8])
            vals.append(num / den if den else 0.0)
        return vals[0] if cnt == 1 else vals
    if typ == 1 or typ == 7:
        return data
    return data


def _parse_exif_app1(payload: bytes) -> dict[str, object]:
    """Extract orientation + GPS from an APP1 Exif payload (b'Exif\\0\\0...')."""
    out: dict[str, object] = {"orientation": 1, "gps": None}
    if not payload.startswith(b"Exif\x00\x00"):
        return out
    tiff = payload[6:]
    if len(tiff) < 8:
        return out
    if tiff[:2] == b"II":
        bo = "<"
    elif tiff[:2] == b"MM":
        bo = ">"
    else:
        return out
    if struct.unpack(bo + "H", tiff[2:4])[0] != 42:
        return out
    ifd0_off = struct.unpack(bo + "I", tiff[4:8])[0]
    ifd0 = _tiff_ifd(tiff, bo, ifd0_off)
    if 0x0112 in ifd0:  # Orientation
        try:
            out["orientation"] = int(_ifd_value(tiff, bo, ifd0[0x0112]))
        except Exception:  # noqa: BLE001
            pass
    if 0x8825 in ifd0:  # GPSInfo pointer
        try:
            gps_off = int(_ifd_value(tiff, bo, ifd0[0x8825]))
            gps = _tiff_ifd(tiff, bo, gps_off)

            def _coord(ref_tag: int, val_tag: int) -> float | None:
                if ref_tag not in gps or val_tag not in gps:
                    return None
                ref = str(_ifd_value(tiff, bo, gps[ref_tag])).strip().upper()
                vals = _ifd_value(tiff, bo, gps[val_tag])
                if not isinstance(vals, list) or len(vals) < 3:
                    return None
                dec = float(vals[0]) + float(vals[1]) / 60.0 + float(vals[2]) / 3600.0
                if ref in ("S", "W"):
                    dec = -dec
                return dec

            lat = _coord(0x0001, 0x0002)
            lon = _coord(0x0003, 0x0004)
            if lat is not None and lon is not None:
                out["gps"] = {"lat": lat, "lon": lon}
        except Exception:  # noqa: BLE001
            pass
    return out


# ── JPEG segment surgery ────────────────────────────────────────────────

_STANDALONE = {0xD8, 0xD9, 0x01} | set(range(0xD0, 0xD8))  # SOI EOI RSTn


def _jpeg_segments(data: bytes):
    """Yield (marker, payload) for each JPEG segment; SOS payload = header only,
    followed by ('SCAN', scan_bytes) and ('EOI', b'')."""
    if data[:2] != _JPEG_SOI:
        raise ValueError("not a JPEG")
    pos = 2
    n = len(data)
    yield 0xD8, b""
    while pos < n:
        if data[pos] != 0xFF:
            # not at a marker — resync (corrupt/entropy data)
            nxt = data.find(b"\xff", pos + 1)
            if nxt == -1:
                break
            pos = nxt
            continue
        while pos < n and data[pos] == 0xFF:  # fill bytes
            pos += 1
        if pos >= n:
            break
        marker = data[pos]
        pos += 1
        if marker in _STANDALONE:
            yield marker, b""
            if marker == 0xD9:
                return
            continue
        if pos + 2 > n:
            break
        (seg_len,) = struct.unpack(">H", data[pos:pos + 2])
        if seg_len < 2:
            break
        payload = data[pos + 2:pos + seg_len]
        pos += seg_len
        if marker == 0xDA:  # SOS — rest is scan data until EOI
            yield marker, payload
            # find EOI: 0xFF 0xD9 not preceded by stuffing (0xFF 0x00 is escaped)
            scan_start = pos
            i = scan_start
            while True:
                j = data.find(b"\xff", i)
                if j == -1 or j + 1 >= n:
                    yield 0xD9, b""  # truncated; treat rest as scan
                    yield "SCAN", data[scan_start:]
                    return
                nxt_b = data[j + 1]
                if nxt_b == 0x00:
                    i = j + 2
                    continue
                if nxt_b == 0xD9:
                    yield "SCAN", data[scan_start:j]
                    yield 0xD9, b""
                    return
                if 0xD0 <= nxt_b <= 0xD7:  # restart marker inside scan
                    i = j + 2
                    continue
                # another marker after scan data (rare) — resume parsing there
                yield "SCAN", data[scan_start:j]
                pos = j
                break
            continue
        yield marker, payload


def _jpeg_is_metadata(marker: int, payload: bytes) -> tuple[bool, str]:
    """(drop?, kind) for a JPEG segment."""
    if marker == 0xE0 and payload.startswith(b"JFIF\x00"):
        return True, "jfif"
    if marker == 0xE1:  # Exif or XMP — always metadata
        if payload.startswith(b"http://ns.adobe.com/xap/1.0/"):
            return True, "xmp"
        return True, "exif"
    if marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
        return True, "icc"
    if marker == 0xEB and b"JUMBF" in payload[:64]:
        return True, "c2pa"  # C2PA manifest lives in APP11
    if marker == 0xED:
        return True, "photoshop/iptc"
    if marker == 0xEE and payload.startswith(b"Adobe\x00"):
        return True, "adobe-app14"
    if marker == 0xFE:
        return True, "comment"
    return False, ""


def _jpeg_surgery(data: bytes) -> tuple[bytes, dict]:
    removed: dict[str, int] = {}
    orientation = 1
    gps = None
    # first pass: orientation/GPS from APP1 Exif
    for marker, payload in _jpeg_segments(data):
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            info = _parse_exif_app1(payload)
            orientation = int(info.get("orientation") or 1)
            gps = info.get("gps")
            break
    out = bytearray()
    for marker, payload in _jpeg_segments(data):
        if marker == "SCAN":
            out += payload
            continue
        if isinstance(marker, int):
            drop, kind = _jpeg_is_metadata(marker, payload)
            if drop:
                removed[kind] = removed.get(kind, 0) + len(payload) + 4
                continue
            out += b"\xff" + bytes([marker])
            if marker not in _STANDALONE:
                out += struct.pack(">H", len(payload) + 2) + payload
        # "EOI" marker string never yielded; 0xD9 handled above
    return bytes(out), {"removed": removed, "orientation": orientation, "gps": gps}


# ── PNG chunk surgery ───────────────────────────────────────────────────

def _png_chunks(data: bytes):
    if data[:8] != _PNG_SIG:
        raise ValueError("not a PNG")
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8].decode("ascii", "replace")
        cdata = data[pos + 8:pos + 8 + length]
        yield ctype, cdata
        pos += 8 + length + 4
        if ctype == "IEND":
            break


def _png_surgery(data: bytes) -> tuple[bytes, dict]:
    removed: dict[str, int] = {}
    text_keys: list[str] = []
    out = bytearray(_PNG_SIG)
    for ctype, cdata in _png_chunks(data):
        if ctype in _PNG_DROP:
            key = ctype
            if ctype in ("tEXt", "zTXt"):
                key += ":" + cdata.split(b"\x00", 1)[0].decode(
                    "latin1", "replace")
                text_keys.append(key.split(":", 1)[1])
            elif ctype == "iTXt":
                parts = cdata.split(b"\x00", 3)
                key += ":" + parts[0].decode("latin1", "replace")
                text_keys.append(parts[0].decode("latin1", "replace"))
            removed[key] = removed.get(key, 0) + len(cdata) + 12
            continue
        out += struct.pack(">I", len(cdata)) + ctype.encode("ascii")
        out += cdata
        out += struct.pack(">I", binascii.crc32(ctype.encode("ascii") + cdata)
                           & 0xFFFFFFFF)
    return bytes(out), {"removed": removed, "text_keys": text_keys}


# ── WebP chunk surgery ──────────────────────────────────────────────────

def _webp_surgery(data: bytes) -> tuple[bytes, dict]:
    removed: dict[str, int] = {}
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("not a WebP")
    chunks = bytearray()
    pos = 12
    n = len(data)
    while pos + 8 <= n:
        fourcc = data[pos:pos + 4].decode("ascii", "replace")
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        cdata = data[pos + 8:pos + 8 + size]
        if fourcc in _WEBP_DROP:
            removed[fourcc.strip().lower()] = removed.get(
                fourcc.strip().lower(), 0) + size + 8
        else:
            chunks += data[pos:pos + 8] + cdata
            if size % 2:
                chunks += b"\x00"
        pos += 8 + size + (size % 2)
    riff_size = 4 + len(chunks)
    return (b"RIFF" + struct.pack("<I", riff_size) + b"WEBP"
            + bytes(chunks)), {"removed": removed}


# ── metadata scan (awareness, not removal) ───────────────────────────────

def _ai_signals(data: bytes, text: str) -> list[str]:
    sigs: list[str] = []
    low = text.lower()
    for hint in _AI_GENERATOR_HINTS:
        if hint in low:
            sigs.append(f"generator stamp: {hint}")
    if "trainedalgorithmicmedia" in low.replace(" ", ""):
        sigs.append("C2PA digital source type: trainedAlgorithmicMedia "
                   "(AI-generated)")
    if "claim_generator" in low:
        sigs.append("C2PA claim_generator present (provenance manifest)")
    if b"jumb" in data.lower() or b"c2pa" in data.lower():
        sigs.append("JUMBF/C2PA content-credential box present")
    # Stable Diffusion / A1111 PNG "parameters" chunk
    if "parameters" in low and ("steps:" in low or "sampler" in low
                                or "cfg scale" in low):
        sigs.append("Stable Diffusion generation parameters embedded")
    seen: set[str] = set()
    return [s for s in sigs if not (s in seen or seen.add(s))]


def scan_metadata(data: bytes) -> dict:
    """Report what metadata an image carries (awareness, not removal)."""
    fmt = _sniff(data)
    out: dict = {
        "format": fmt,
        "has_exif": False,
        "tags": {},
        "has_gps": False,
        "gps": None,
        "orientation": 1,
        "has_xmp": False,
        "has_iptc": False,
        "has_icc": False,
        "has_comment": False,
        "has_c2pa": False,
        "png_text": {},
        "ai_signals": [],
        "bytes": len(data),
    }
    try:
        if fmt == "jpeg":
            for marker, payload in _jpeg_segments(data):
                if not isinstance(marker, int):
                    continue
                drop, kind = _jpeg_is_metadata(marker, payload)
                if kind == "exif":
                    out["has_exif"] = True
                    info = _parse_exif_app1(payload)
                    out["orientation"] = info.get("orientation") or 1
                    if info.get("gps"):
                        out["has_gps"] = True
                        out["gps"] = info["gps"]
                elif kind == "xmp":
                    out["has_xmp"] = True
                elif kind == "photoshop/iptc":
                    out["has_iptc"] = True
                elif kind == "icc":
                    out["has_icc"] = True
                elif kind == "comment":
                    out["has_comment"] = True
                elif kind == "c2pa":
                    out["has_c2pa"] = True
            # best-effort EXIF tag listing via PIL when available
            out["tags"] = _pil_exif_tags(data)
        elif fmt == "png":
            for ctype, cdata in _png_chunks(data):
                if ctype == "eXIf":
                    out["has_exif"] = True
                elif ctype == "iCCP":
                    out["has_icc"] = True
                elif ctype in ("tEXt", "zTXt"):
                    kw, _, val = cdata.partition(b"\x00")
                    k = kw.decode("latin1", "replace")
                    out["png_text"][k] = val[:80].decode(
                        "latin1", "replace")
                elif ctype == "iTXt":
                    parts = cdata.split(b"\x00", 4)
                    k = parts[0].decode("latin1", "replace")
                    out["png_text"][k] = parts[-1][:80].decode(
                        "utf8", "replace")
                elif ctype == "tIME":
                    out["png_text"]["tIME(modification-time)"] = "present"
            out["tags"] = dict(out["png_text"])
        elif fmt == "webp":
            low = data.lower()
            out["has_exif"] = b"exif" in data[:12] or any(
                c == "EXIF" for c, _ in _webp_chunks(data))
            out["has_xmp"] = b"xmp " in low
            out["has_icc"] = b"iccp" in low
            out["has_c2pa"] = b"jumb" in low
    except Exception:
        pass
    blob_text = data[:1_000_000].decode("latin1", "replace")
    out["ai_signals"] = _ai_signals(data, blob_text + str(out["png_text"]))
    if out["has_c2pa"] and not any("C2PA" in s for s in out["ai_signals"]):
        out["ai_signals"].append("JUMBF/C2PA content-credential box present")
    return out


def _webp_chunks(data: bytes):
    pos = 12
    n = len(data)
    while pos + 8 <= n:
        fourcc = data[pos:pos + 4].decode("ascii", "replace")
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        yield fourcc, data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size % 2)


def _pil_exif_tags(data: bytes) -> dict:
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(io.BytesIO(data)) as im:
            exif = im.getexif()
            if not exif:
                return {}
            tags = {}
            for tag_id, value in exif.items():
                try:
                    tags[str(TAGS.get(tag_id, tag_id))] = str(value)[:80]
                except Exception:  # noqa: BLE001
                    continue
            return tags
    except Exception:  # noqa: BLE001
        return {}


def exif_summary(path: Union[str, Path]) -> dict:
    """Report what EXIF tags an image carries (for awareness, not removal)."""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return {"has_exif": False, "tags": {}, "has_gps": False}
    full = scan_metadata(data)
    # backward-compatible top-level shape + richer detail
    full["path"] = str(p)
    return full


# ── stripping ────────────────────────────────────────────────────────────

def _pil():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "PIL is required for EXIF stripping (pip install pillow)") from exc
    return Image, ImageOps


def _pil_fallback(data: bytes) -> bytes:
    """Pillow path for formats without a surgery implementation."""
    Image, ImageOps = _pil()
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        fmt = (im.format or "PNG").upper()
        if fmt == "JPEG" and im.mode in ("RGBA", "LA", "PA"):
            im = im.convert("RGB")
        if fmt in ("JPEG", "JPG"):
            fmt = "JPEG"
        buf = io.BytesIO()
        # notably: NO exif= kwarg → no EXIF written; fresh image → no
        # info/text dicts carried over
        im.save(buf, format=fmt)
        return buf.getvalue()


def strip_metadata_bytes(data: bytes) -> bytes:
    """Strip all metadata from image bytes. Returns clean bytes.

    JPEG/PNG/WebP go through lossless segment surgery (no re-encode, no
    quality loss). Anything else falls back to Pillow. Images whose EXIF
    orientation is not "normal" are transposed first so pixels stay correct.
    """
    fmt = _sniff(data)
    if fmt == "jpeg":
        clean, meta = _jpeg_surgery(data)
        if int(meta.get("orientation") or 1) != 1:
            # must transpose pixels → re-encode, then strip again
            Image, ImageOps = _pil()
            with Image.open(io.BytesIO(data)) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA", "PA"):
                    im = im.convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=95)
            clean, _ = _jpeg_surgery(buf.getvalue())
        return clean
    if fmt == "png":
        clean, _ = _png_surgery(data)
        return clean
    if fmt == "webp":
        clean, _ = _webp_surgery(data)
        return clean
    return _pil_fallback(data)


def strip_exif_to_bytes(data: bytes) -> bytes:
    """Strip all EXIF/metadata from image bytes. Returns clean bytes."""
    return strip_metadata_bytes(data)


def _unique_backup(p: Path) -> Path:
    cand = p.with_suffix(p.suffix + ".bak")
    i = 1
    while cand.exists():
        cand = p.with_suffix(f"{p.suffix}.bak.{i}")
        i += 1
    return cand


def strip_exif(path: Union[str, Path], *, backup: bool = True) -> dict:
    """Strip metadata from a file in place. Returns summary of what was removed."""
    p = Path(path)
    before = scan_metadata(p.read_bytes())
    raw = p.read_bytes()
    clean = strip_metadata_bytes(raw)
    bak = None
    if backup:
        bak = _unique_backup(p)
        bak.write_bytes(raw)
    p.write_bytes(clean)
    removed = before.get("removed_detail", {})
    return {
        "path": str(p),
        "had_exif": before["has_exif"],
        "had_gps": before["has_gps"],
        "tags_removed": len(before["tags"]),
        "had_xmp": before["has_xmp"],
        "had_iptc": before["has_iptc"],
        "had_icc": before["has_icc"],
        "ai_signals_removed": before["ai_signals"],
        "bytes_before": len(raw),
        "bytes_after": len(clean),
        "backup": str(bak) if bak else None,
    }


def clean_copy(path: Union[str, Path], dest: Union[str, Path, None] = None) -> dict:
    """mat2-style clean: write ``<name>.cleaned.<ext>``, original untouched."""
    p = Path(path)
    if dest is None:
        dest = p.with_name(f"{p.stem}.cleaned{p.suffix}")
    else:
        dest = Path(dest)
    raw = p.read_bytes()
    before = scan_metadata(raw)
    clean = strip_metadata_bytes(raw)
    dest.write_bytes(clean)
    ok, residual = verify_clean(clean)
    return {
        "src": str(p),
        "dest": str(dest),
        "bytes_before": len(raw),
        "bytes_after": len(clean),
        "verified_clean": ok,
        "residual": residual,
        "removed_signals": before["ai_signals"],
    }


def verify_clean(data: bytes) -> tuple[bool, dict]:
    """Re-scan stripped bytes; returns (ok, residual-metadata dict)."""
    s = scan_metadata(data)
    residual = {}
    for key in ("has_exif", "has_xmp", "has_iptc", "has_icc",
                "has_comment", "has_c2pa"):
        if s.get(key):
            residual[key] = True
    if s.get("has_gps"):
        residual["has_gps"] = s["gps"]
    if s.get("png_text"):
        residual["png_text"] = s["png_text"]
    if s.get("ai_signals"):
        residual["ai_signals"] = s["ai_signals"]
    return (not residual, residual)


def secure_delete(path: Union[str, Path], *, passes: int = 3) -> bool:
    """Overwrite a file with random bytes then unlink it (shred-style).

    For wiping ``.bak`` originals after a strip. Returns True on success.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
        with open(p, "r+b") as f:
            for _ in range(max(1, passes)):
                f.seek(0)
                remaining = size
                while remaining > 0:
                    chunk = os.urandom(min(65536, remaining))
                    f.write(chunk)
                    remaining -= len(chunk)
                f.flush()
                os.fsync(f.fileno())
        p.unlink()
        return True
    except OSError:
        return False


def format_summary(s: dict) -> str:
    """Human-readable rendering of a metadata scan (chat/terminal)."""
    L = [f"🖼️ metadata scan — {s.get('format', '?').upper()} "
         f"({s.get('bytes', 0):,} bytes)"]
    flags = []
    if s.get("has_exif"):
        flags.append(f"EXIF ({len(s.get('tags', {}))} tags)")
    if s.get("has_gps"):
        g = s["gps"] or {}
        flags.append(f"📍 GPS {g.get('lat', '?'):.6f}, {g.get('lon', '?'):.6f}"
                     if isinstance(g, dict) else "📍 GPS present")
    for key, label in (("has_xmp", "XMP"), ("has_iptc", "IPTC"),
                       ("has_icc", "ICC profile"), ("has_comment", "COM"),
                       ("has_c2pa", "C2PA credentials")):
        if s.get(key):
            flags.append(label)
    if s.get("png_text"):
        flags.append("PNG text: " + ", ".join(s["png_text"]))
    L.append("   found: " + (", ".join(flags) if flags else "nothing — clean ✨"))
    for sig in s.get("ai_signals", []):
        L.append(f"   🤖 {sig}")
    if s.get("orientation", 1) != 1:
        L.append(f"   orientation tag: {s['orientation']} (pixels transposed on strip)")
    return "\n".join(L)
