"""EXIF/metadata stripping for outbound images.

Removes GPS coordinates, camera model, timestamps, and all other EXIF
before an image leaves the device — operational security for every
photo Devon sends. Uses PIL (already a repo dependency).

``strip_exif(path)`` rewrites the file in place (keeping a ``.bak``).
``strip_exif_to_bytes(data)`` works on in-memory bytes for the send
path. Both preserve pixel data and orientation (applies the EXIF
orientation first, so the image doesn't come out rotated).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

__all__ = ["strip_exif", "strip_exif_to_bytes", "exif_summary"]


def _pil():
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "PIL is required for EXIF stripping (pip install pillow)") from exc
    return Image, ImageOps


def exif_summary(path: Union[str, Path]) -> dict:
    """Report what EXIF tags an image carries (for awareness, not removal)."""
    Image, _ = _pil()
    out: dict = {"has_exif": False, "tags": {}, "has_gps": False}
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if exif:
                out["has_exif"] = True
                # tag 34853 = GPSInfo
                out["has_gps"] = 34853 in exif
                for tag_id, value in exif.items():
                    try:
                        tag = Image.ExifTags.TAGS.get(tag_id, str(tag_id))
                        v = str(value)
                        out["tags"][str(tag)] = v[:80]
                    except Exception:  # noqa: BLE001
                        continue
    except Exception:
        pass
    return out


def strip_exif_to_bytes(data: bytes) -> bytes:
    """Strip all EXIF/metadata from image bytes. Returns clean bytes."""
    Image, ImageOps = _pil()
    with Image.open(io.BytesIO(data)) as im:
        # apply orientation first so pixels are correct without the tag
        im = ImageOps.exif_transpose(im)
        # convert to a mode that saves cleanly; drop alpha for JPEG
        fmt = (im.format or "JPEG").upper()
        if fmt == "JPEG" and im.mode in ("RGBA", "LA", "PA"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        save_kw: dict = {}
        if fmt in ("JPEG", "JPG"):
            fmt = "JPEG"
        # notably: NO exif= kwarg → no metadata written
        im.save(buf, format=fmt, **save_kw)
        return buf.getvalue()


def strip_exif(path: Union[str, Path], *, backup: bool = True) -> dict:
    """Strip EXIF from a file in place. Returns summary of what was removed."""
    p = Path(path)
    before = exif_summary(p)
    raw = p.read_bytes()
    clean = strip_exif_to_bytes(raw)
    if backup:
        p.with_suffix(p.suffix + ".bak").write_bytes(raw)
    p.write_bytes(clean)
    return {
        "path": str(p),
        "had_exif": before["has_exif"],
        "had_gps": before["has_gps"],
        "tags_removed": len(before["tags"]),
        "backup": str(p.with_suffix(p.suffix + ".bak")) if backup else None,
    }
