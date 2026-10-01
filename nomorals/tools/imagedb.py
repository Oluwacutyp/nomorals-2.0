"""Image lookup + reverse image search.

* ``image_lookup(path_or_url)`` — real image intelligence with the standard
  library: format sniffing, PNG/JPEG dimension parsing, blake2b content
  hash, optional dHash perceptual hash when PIL is installed, and an index
  (``image_index``) that answers "have we seen this image before, where,
  when?" — plus near-duplicate detection by dHash distance.
* ``image_exif(path)`` — EXIF camera metadata (make/model/datetime/GPS)
  when Pillow is available; {} otherwise.
* ``image_colors(path, n=6)`` — dominant color palette as hex strings;
  [] when Pillow is unavailable.
* ``find_dupes(context, threshold=6)`` — library-wide near-duplicate
  clustering by dHash hamming distance.
* ``image_stats(context)`` — library overview (counts, formats, bytes,
  dupe-group count).
* ``reverse_image_search(path_or_url)`` — builds the deep links the big
  engines need (Google Lens / Bing / Yandex / TinEye) and, for public
  image URLs, best-effort scans Bing's and Yandex's visual-search pages
  for related-image media URLs (labelled unverified — scrapes are brittle
  by nature; each engine runs isolated so one failure never kills the
  other).

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

__all__ = ["register", "sniff_format", "image_dimensions", "dhash", "hamming",
           "image_exif", "image_colors", "find_dupes", "image_stats"]


def _pil_image():
    """Lazy Pillow import — returns the Image class or None when absent."""
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def _pil_exif_tags():
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
    except ImportError:
        return None, None
    return TAGS, GPSTAGS


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


# ── EXIF + visual analysis (Pillow, graceful degradation) ──────────────────


_GPS_IFD = 0x8825


def _gps_decimal(gps: dict) -> tuple[float, float] | None:
    """GPS IFD (name-keyed) -> (lat, lon) decimal degrees, or None."""
    try:
        lat = gps.get("GPSLatitude")
        lat_ref = gps.get("GPSLatitudeRef")
        lon = gps.get("GPSLongitude")
        lon_ref = gps.get("GPSLongitudeRef")
        if lat is None or lon is None:
            return None

        def dms_to_deg(dms):
            deg, minutes, seconds = (float(x) for x in dms)
            return deg + minutes / 60.0 + seconds / 3600.0

        lat_dec = dms_to_deg(lat)
        if str(lat_ref).upper() == "S":
            lat_dec = -lat_dec
        lon_dec = dms_to_deg(lon)
        if str(lon_ref).upper() == "W":
            lon_dec = -lon_dec
        return round(lat_dec, 6), round(lon_dec, 6)
    except Exception:  # noqa: BLE001 - GPS parsing is best-effort
        return None


def image_exif(path: str | Path) -> dict[str, Any]:
    """Camera/format metadata from EXIF: make, model, datetime, software,
    orientation, exposure, ISO, GPS (when present).

    Returns {} gracefully when Pillow is unavailable, the file has no
    EXIF, or the file cannot be read.
    """
    Image = _pil_image()
    if Image is None:
        return {}
    TAGS, GPSTAGS = _pil_exif_tags()
    if TAGS is None:
        return {}
    try:
        with Image.open(path) as img:
            raw = img.getexif()
            if not raw:
                return {}
    except Exception:  # noqa: BLE001
        return {}

    tags = {TAGS.get(tag, tag): value for tag, value in raw.items()}
    out: dict[str, Any] = {}

    def grab(*names):
        for name in names:
            value = tags.get(name)
            if value not in (None, ""):
                return str(value).strip()
        return None

    make = grab("Make")
    model = grab("Model")
    dt = grab("DateTimeOriginal", "DateTime")
    software = grab("Software")
    orientation = tags.get("Orientation")
    exposure = tags.get("ExposureTime")
    fnumber = tags.get("FNumber")
    iso = grab("ISOSpeedRatings", "PhotographicSensitivity")

    if make:
        out["make"] = make
    if model:
        out["model"] = model
    if dt:
        out["datetime"] = dt
    if software:
        out["software"] = software
    if orientation:
        out["orientation"] = int(orientation)
    if exposure:
        try:
            out["exposure_time"] = str(exposure)
        except Exception:  # noqa: BLE001
            pass
    if fnumber:
        try:
            out["f_number"] = float(fnumber)
        except Exception:  # noqa: BLE001
            pass
    if iso:
        out["iso"] = iso

    try:
        gps_ifd = raw.get_ifd(_GPS_IFD) if hasattr(raw, "get_ifd") else {}
        if gps_ifd:
            gps = {GPSTAGS.get(tag, tag): value for tag, value in gps_ifd.items()}
            coords = _gps_decimal(gps)
            if coords:
                out["gps"] = {"lat": coords[0], "lon": coords[1]}
    except Exception:  # noqa: BLE001
        pass
    return out


def image_colors(path: str | Path, n: int = 6) -> list[str]:
    """Dominant color palette: resize small, count quantized colors,
    return top ``n`` as ``#rrggbb`` hex strings.

    Returns [] gracefully when Pillow is unavailable or the file cannot
    be read.
    """
    Image = _pil_image()
    if Image is None:
        return []
    n = max(1, min(int(n), 32))
    try:
        with Image.open(path) as img:
            small = img.convert("RGB").resize((128, 128), Image.BILINEAR)
            colors = small.getcolors(128 * 128)
            if not colors:
                # too many distinct colors: quantize down first
                small = small.quantize(colors=n, method=Image.MEDIANCUT)
                colors = small.convert("RGB").getcolors(128 * 128) or []
            top = sorted(colors, key=lambda c: c[0], reverse=True)[:n]
        return [f"#{r:02x}{g:02x}{b:02x}" for _, (r, g, b) in top]
    except Exception:  # noqa: BLE001
        return []


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
                # compare perceptual hashes (not content digests — dhash is
                # 16 hex chars, blake2b digests are 32, so they can never match)
                for row in db.query("SELECT hash, path FROM image_index"):
                    other = dhash(row.get("path", "")) if row.get("path") else None
                    distance = hamming(phash, other or "")
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


# ── library intelligence ─────────────────────────────────────────────────────


def _library_rows(context: Any) -> list[dict[str, Any]]:
    db = getattr(context, "db", None)
    if db is None:
        return []
    try:
        return list(db.query("SELECT hash, path, size, mime FROM image_index"))
    except Exception:  # noqa: BLE001
        return []


def find_dupes(context: Any, threshold: int = 6) -> dict[str, Any]:
    """Cluster every indexed image by dHash hamming distance.

    Files that no longer exist on disk (or can't be hashed — e.g. Pillow
    missing) are excluded. Returns groups of near-duplicates, each with
    paths, content digests, and max pairwise distance.
    """
    rows = _library_rows(context)
    if not rows:
        return {"ok": True, "threshold": threshold, "scanned": 0, "groups": []}

    items: list[dict[str, Any]] = []
    for row in rows:
        path = row.get("path") or ""
        ph = dhash(path) if path and Path(path).exists() else None
        items.append({"path": path, "digest": row.get("hash", ""), "dhash": ph})

    hashable = [i for i in items if i["dhash"]]
    parent = list(range(len(hashable)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(hashable)):
        for j in range(i + 1, len(hashable)):
            dist = hamming(hashable[i]["dhash"], hashable[j]["dhash"])
            if dist is not None and dist <= threshold:
                union(i, j)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for idx, item in enumerate(hashable):
        clusters.setdefault(find(idx), []).append(item)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        max_dist = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                dist = hamming(members[i]["dhash"], members[j]["dhash"]) or 0
                max_dist = max(max_dist, dist)
        groups.append({
            "count": len(members),
            "max_distance": max_dist,
            "paths": [m["path"] for m in members],
            "digests": [m["digest"] for m in members],
        })
    groups.sort(key=lambda g: g["count"], reverse=True)
    return {"ok": True, "threshold": threshold, "scanned": len(hashable),
            "groups": groups}


def image_stats(context: Any) -> dict[str, Any]:
    """Library overview: count, per-format breakdown, total bytes, and
    near-duplicate group count (via find_dupes, threshold 6)."""
    rows = _library_rows(context)
    formats: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        fmt = row.get("mime") or "unknown"
        formats[fmt] = formats.get(fmt, 0) + 1
        try:
            total_bytes += int(row.get("size") or 0)
        except (TypeError, ValueError) as e:
            _log.debug("skipping bad size value %r: %s", row.get("size"), e)
    try:
        dupe_groups = len(find_dupes(context, threshold=6)["groups"])
    except Exception:  # noqa: BLE001 - stats must never fail on clustering
        dupe_groups = 0
    return {
        "ok": True,
        "count": len(rows),
        "formats": formats,
        "total_bytes": total_bytes,
        "dupe_groups": dupe_groups,
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
        matches = _scrape_engines(client, public)
        out["unverified_matches"] = [u for urls in matches.values() for u in urls]
        out["unverified_by_engine"] = matches
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


def _dedupe_urls(raw: list[bytes], cap: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in raw:
        try:
            url = match.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= cap:
            break
    return out


def _scrape_bing(client: HttpClient, image_url: str) -> list[str]:
    """Best-effort: related-image media URLs from Bing's visual search page."""
    try:
        page = client.get(
            "https://www.bing.com/images/search?view=detailv2&iss=sbi&qimgurl="
            + urllib.parse.quote(image_url, safe="")
        )
    except Exception:  # noqa: BLE001 - scraping is optional garnish
        return []
    return _dedupe_urls(_MEDIA_URL.findall(page.body or b""))


_YANDEX_ORIGINAL = re.compile(rb'"originalImage"\s*:\s*\{\s*"url"\s*:\s*"([^"]{10,300})"', re.IGNORECASE)
_YANDEX_IMG_SRC = re.compile(
    rb'<img[^>]{0,500}?src="(https?://[^"]{10,300}\.(?:jpe?g|png|webp|gif)[^"]{0,60})"',
    re.IGNORECASE)


def _scrape_yandex(client: HttpClient, image_url: str) -> list[str]:
    """Best-effort: similar-image URLs from Yandex's image-search-by-image
    page. Yandex embeds the matched image JSON as "originalImage":{"url":...}
    and similar hits as <img> tags — we fish out both."""
    try:
        page = client.get(
            "https://yandex.com/images/search?rpt=imageview&url="
            + urllib.parse.quote(image_url, safe="")
        )
    except Exception:  # noqa: BLE001 - scraping is optional garnish
        return []
    body = page.body or b""
    hits = _YANDEX_ORIGINAL.findall(body) + _YANDEX_IMG_SRC.findall(body)
    cleaned: list[bytes] = []
    for hit in hits:
        hit = hit.replace(b"\\/", b"/")
        if hit not in cleaned:
            cleaned.append(hit)
    return _dedupe_urls(cleaned)


def _scrape_engines(client: HttpClient, image_url: str) -> dict[str, list[str]]:
    """Run each reverse-search scraper isolated: one engine failing
    (network, block, markup change) never kills the others."""
    results: dict[str, list[str]] = {}
    for name, scraper in (("bing", _scrape_bing), ("yandex", _scrape_yandex)):
        try:
            results[name] = scraper(client, image_url) or []
        except Exception:  # noqa: BLE001 - per-engine isolation
            _log.debug("reverse image scrape failed for %s", name, exc_info=True)
            results[name] = []
    return results


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
                    "plus best-effort related-image scans (Bing + Yandex, "
                    "engine-isolated) for public URLs.",
        capability="net.out",
    )
    def reverse_image_search(path: str) -> dict[str, Any]:
        return _reverse_impl(context, path)

    @registry.register(
        "image_exif",
        description="Read EXIF camera metadata from an image file "
                    "(make, model, datetime, GPS when present).",
        capability="fs.read",
    )
    def image_exif_tool(path: str) -> dict[str, Any]:
        return {"ok": True, "path": path, "exif": image_exif(path)}

    @registry.register(
        "image_colors",
        description="Dominant color palette of an image file, returned as "
                    "hex strings (#rrggbb).",
        capability="fs.read",
    )
    def image_colors_tool(path: str, n: int = 6) -> dict[str, Any]:
        return {"ok": True, "path": path, "colors": image_colors(path, n=n)}

    @registry.register(
        "find_dupes",
        description="Cluster the whole indexed image library into "
                    "near-duplicate groups by dHash hamming distance.",
        capability="fs.read",
    )
    def find_dupes_tool(threshold: int = 6) -> dict[str, Any]:
        return find_dupes(context, threshold=threshold)

    @registry.register(
        "image_stats",
        description="Image library overview: count, formats, total bytes, "
                    "near-duplicate group count.",
        capability="fs.read",
    )
    def image_stats_tool() -> dict[str, Any]:
        return image_stats(context)
