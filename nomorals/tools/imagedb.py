"""Image lookup + reverse image search.

* ``image_lookup(path_or_url)`` — real image intelligence with the standard
  library: format sniffing, PNG/JPEG dimension parsing, blake2b content
  hash, optional dHash perceptual hash when PIL is installed, and an index
  (``image_index``) that answers "have we seen this image before, where,
  when?" — plus near-duplicate detection by dHash distance.
* ``reverse_image_search(path_or_url)`` — builds the deep links the big
  engines need (Google Lens / Bing / Yandex / TinEye) and, for public
  image URLs, best-effort scans Bing's visual-search page for related-image
  media URLs (labelled unverified — scrapes are brittle by nature).

URLs are downloaded to the media dir first (size-capped), so both tools
accept either a local path or a public URL.
"""

from __future__ import annotations

import hashlib
import re
import struct
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.http import HttpClient

_log = get_logger(__name__)

__all__ = ["register", "sniff_format", "image_dimensions", "dhash", "hamming"]


# ── pure image primitives (stdlib) ──────────────────────────────────────────


def sniff_format(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] == b"BM":
        return "bmp"
    if data[:4] == b"II*\x00":
        return "tiff"
    return "unknown"


def png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    except struct.error:
        return None


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG markers to the first SOF (start of frame) with dimensions."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD9:
            i += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            try:
                height = int.from_bytes(data[i + 5:i + 7], "big")
                width = int.from_bytes(data[i + 7:i + 9], "big")
                return int(width), int(height)
            except Exception:  # noqa: BLE001
                return None
        if i + 4 > n:
            return None
        seg_len = int.from_bytes(data[i + 2:i + 4], "big")
        if seg_len < 2:
            return None
        i += 2 + seg_len
    return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    fmt = sniff_format(data)
    if fmt == "png":
        return png_dimensions(data)
    if fmt == "jpeg":
        return jpeg_dimensions(data)
    return None


def dhash(path: str | Path) -> str | None:
    """64-bit difference hash. Returns None when PIL is unavailable — the
    content hash still works, near-duplicate detection just degrades."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((9, 8), Image.LANCZOS)
            pixels = list(img.getdata())
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (pixels[row * 9 + col] < pixels[row * 9 + col + 1])
        return f"{bits:016x}"
    except Exception:  # noqa: BLE001 - dhash is best-effort
        return None


def hamming(a: str, b: str) -> int | None:
    if not a or not b or len(a) != len(b):
        return None
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return None


# ── the tools ────────────────────────────────────────────────────────────────


def _resolve_image(context: Any, ref: str, client: HttpClient) -> tuple[Path | None, str]:
    """Path or URL -> (local_path, public_url_or_empty)."""
    ref = (ref or "").strip()
    if not ref:
        return None, ""
    if ref.startswith(("http://", "https://")):
        tools = getattr(getattr(context, "settings", None), "tools", None)
        cap_mb = int(getattr(tools, "max_download_mb", 200)) if tools else 200
        name = "img-" + hashlib.blake2b(ref.encode(), digest_size=8).hexdigest()
        media_dir = Path(getattr(getattr(context, "settings", None), "home", "~/.nomorals"))
        media_dir = media_dir / "media" / "inbound"
        media_dir.mkdir(parents=True, exist_ok=True)
        target = media_dir / (name + ".bin")
        if not target.exists():
            response = client.get(ref)
            if len(response.body or b"") > cap_mb * 1024 * 1024:
                target.unlink(missing_ok=True)
                return None, ref
            target.write_bytes(response.body or b"")
        return target, ref
    path = Path(ref).expanduser()
    return (path if path.exists() else None), ""


def _index_row(context: Any, digest: str, path: Path, size: int, fmt: str,
               seen_in: str) -> dict[str, Any] | None:
    """Upsert into image_index; returns the previous sighting (if any)."""
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        row = db.query_one("SELECT * FROM image_index WHERE hash = ?", (digest,))
        with db.transaction():
            db.execute(
                """INSERT INTO image_index (hash, path, size, mime, first_seen, last_seen, seen_in)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hash) DO UPDATE SET last_seen = excluded.last_seen,
                                                  seen_in = CASE WHEN excluded.seen_in != ''
                                                                 THEN excluded.seen_in ELSE image_index.seen_in END""",
                (digest, str(path), size, fmt, time.time() if row is None else row["first_seen"],
                 time.time(), seen_in),
            )
        return row
    except Exception:  # noqa: BLE001 - indexing must never break a lookup
        return None


def _lookup_impl(context: Any, ref: str, seen_in: str) -> dict[str, Any]:
    tools = getattr(getattr(context, "settings", None), "tools", None)
    client = HttpClient(timeout=getattr(tools, "http_timeout", 30.0),
                        user_agent=getattr(tools, "user_agent", "NoMoralsCore/0.1"),
                        proxy_url=getattr(tools, "proxy_url", ""))
    local, public = _resolve_image(context, ref, client)
    if local is None or not local.exists():
        return {"ok": False, "error": f"could not resolve image: {ref}"}
    data = local.read_bytes()
    fmt = sniff_format(data)
    dims = image_dimensions(data)
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()
    previous = _index_row(context, digest, local, len(data), fmt, seen_in)
    seen_before = previous is not None
    seen_where = (previous or {}).get("seen_in", "")
    seen_days = ""
    if seen_before:
        age = time.time() - float(previous.get("first_seen", 0) or 0)
        if age > 86400:
            seen_days = f" first seen {int(age // 86400)} day(s) ago"
    phash = dhash(local)
    near: list[str] = []
    if phash:
        db = getattr(context, "db", None)
        if db is not None:
            try:
                for row in db.query("SELECT hash, path FROM image_index"):
                    distance = hamming(phash, row.get("hash", ""))
                    if distance is not None and 0 < distance <= 10:
                        near.append(f"{row['path']} (distance {distance})")
            except Exception:  # noqa: BLE001
                pass
    return {
        "ok": True,
        "ref": ref,
        "path": str(local),
        "format": fmt,
        "width": dims[0] if dims else None,
        "height": dims[1] if dims else None,
        "bytes": len(data),
        "hash": digest,
        "dhash": phash,
        "seen_before": seen_before,
        "seen_where": seen_where if seen_before else "",
        "seen_days": seen_days,
        "near_duplicates": near[:5],
    }


def _reverse_impl(context: Any, ref: str) -> dict[str, Any]:
    tools = getattr(getattr(context, "settings", None), "tools", None)
    client = HttpClient(timeout=getattr(tools, "http_timeout", 30.0),
                        user_agent=getattr(tools, "user_agent", "NoMoralsCore/0.1"),
                        proxy_url=getattr(tools, "proxy_url", ""))
    local, public = _resolve_image(context, ref, client)
    if local is None or not local.exists():
        return {"ok": False, "error": f"could not resolve image: {ref}"}
    out: dict[str, Any] = {"ok": True, "ref": ref}
    if public:
        enc = urllib.parse.quote(public, safe="")
        out["lookup_links"] = {
            "google_lens": f"https://lens.google.com/uploadbyurl?url={enc}",
            "bing": f"https://www.bing.com/images/search?view=detailv2&iss=sbi&qimgurl={enc}",
            "yandex": f"https://yandex.com/images/?rpt=imageview&url={enc}",
            "tineye": f"https://tineye.com/search?url={enc}",
        }
        out["unverified_matches"] = _scrape_bing(client, public)
    else:
        out["note"] = ("no public URL for this image — upload it somewhere public "
                       "and re-run, or open these with the file directly: "
                       "https://lens.google.com  https://tineye.com")
        out["lookup_links"] = {
            "google_lens": "https://lens.google.com (upload)",
            "tineye": "https://tineye.com (upload)",
        }
    return out


_MEDIA_URL = re.compile(rb'mediaurl="([^"]{10,300})"', re.IGNORECASE)


def _scrape_bing(client: HttpClient, image_url: str) -> list[str]:
    """Best-effort: related-image media URLs from Bing's visual search page."""
    try:
        page = client.get(
            "https://www.bing.com/images/search?view=detailv2&iss=sbi&qimgurl="
            + urllib.parse.quote(image_url, safe="")
        )
    except Exception:  # noqa: BLE001 - scraping is optional garnish
        return []
    found = _MEDIA_URL.findall(page.body or b"")
    seen: set[bytes] = set()
    out: list[str] = []
    for match in found:
        try:
            url = match.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= 8:
            break
    return out


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "image_lookup",
        description="Inspect an image (path or URL): format, dimensions, hash, "
                    "whether we've seen it before, and near-duplicates.",
        capability="fs.read",
    )
    def image_lookup(path: str, *, seen_in: str = "") -> dict[str, Any]:
        return _lookup_impl(context, path, seen_in)

    @registry.register(
        "reverse_image_search",
        description="Reverse image search: deep links to Lens/Bing/Yandex/TinEye "
                    "plus best-effort related-image scan for public URLs.",
        capability="net.out",
    )
    def reverse_image_search(path: str) -> dict[str, Any]:
        return _reverse_impl(context, path)
