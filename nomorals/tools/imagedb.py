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

Pass #92 (owner's open-sources law — images named explicitly): the
reverse-search roster above is an image SOURCE class, and it now
lives in the ONE source registry (``media/source_registry.py``) the
#86–#90 passes built — as its own class, for the same reason web
search joined in #90: every scraped engine is one host serving
image search.  The two scraped hosts (``bing.com``,
``yandex.com``) are already registry descriptors (web-search
seeds, #90) and both classes fetch the literal same hosts
(``www.bing.com/search`` vs ``www.bing.com/images/search``;
``yandex.com/search`` vs ``yandex.com/images/search``), so this
class SHARES those host descriptors and extends their kinds —
one host, one descriptor, two class memberships (the #88
``youtube.com`` download + music-search precedent).  Every scrape
attempt now records its outcome against the engine's descriptor —
served / no-result (weak evidence, #88) / failure with an error
class / bot-wall (:meth:`record_blocked`, #89 — a 200 challenge
page counts too) / rate-limited (:meth:`record_rate_limited`,
#90 — a 429 is recorded once and never retried around) — the
chain order is outcome-aware with a day-one pin (zero learned
state → ``("bing", "yandex")`` byte-for-byte), and the ONE
registry health loop / health job covers the class.  Google Lens
and TinEye are deep LINKS this tool hands the user; Devon never
fetches them, so they get NO descriptor and earn no credit (the
#90 ddgs precedent) — they surface as link-only rows in the
class status (``image_sources`` tool / ``download_sources``
status).  Discovery verdict — named absence: reverse-search
RESULT hosts are image content, not engines (a host holding a
matching picture cannot be queried BY picture), so no candidate
intake exists on this path.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.errors import http_status_of
from ..core.logging_setup import get_logger
from ..core.http import HttpClient

_log = get_logger(__name__)

__all__ = ["register", "sniff_format", "image_dimensions", "dhash", "hamming",
           "image_exif", "image_colors", "find_dupes", "image_stats",
           "REVERSE_IMAGE_ENGINES", "IMAGE_ENGINE_DOMAINS",
           "IMAGE_LINK_ONLY_DOMAINS", "image_registry_for_context",
           "ordered_image_engines"]


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
        # settings.home defaults to "~/.nomorals" — expand "~" or the
        # media dir is a literal "~" folder under the CWD (Path does
        # not expand "~" on its own). On Windows that literal folder
        # would also be created next to the process, not under the
        # user's profile.
        home = os.path.expanduser(
            str(getattr(getattr(context, "settings", None), "home",
                        "~/.nomorals")))
        media_dir = Path(home)
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


# ── image source class: registry plumbing (Pass #92) ─────────────────

#: The reverse image search engines Devon actually SCRAPES, in
#: day-one order.  This tuple is the class roster: the chain runs
#: every engine isolated (one failure never kills the other), and
#: Pass #92 makes the order outcome-aware through the registry
#: (see :func:`ordered_image_engines`) with this order as the
#: zero-evidence pin.
REVERSE_IMAGE_ENGINES: tuple[str, ...] = ("bing", "yandex")

#: The host each scraped engine actually fetches from — the
#: engine's source in the ONE source registry (Pass #92).  This
#: map is the image class's membership list: the scrape path
#: credits outcomes to the registry descriptor for this domain,
#: the registry's status surface renders the class from it, and
#: the health loop verifies these hosts.  Domains are evidence
#: from the scrape URL construction below (``www.bing.com`` /
#: ``yandex.com``, www-canonicalised by the registry exactly as
#: in ``tools/web.py``), NOT the engines' marketing domains —
#: and they are the SAME hosts the web-search class fetches its
#: search pages from (#90), which is why the descriptors are
#: shared rather than duplicated.
IMAGE_ENGINE_DOMAINS: dict[str, str] = {
    "bing": "bing.com",
    "yandex": "yandex.com",
}

#: Engines that appear in the product only as deep LINKS handed
#: to the user (see ``_reverse_impl``'s ``lookup_links``).  Devon
#: never fetches them, so no outcome can ever be recorded for
#: them honestly: they get NO registry descriptor and earn no
#: credit (the #90 ddgs precedent) and surface in the class
#: status as link-only rows.
IMAGE_LINK_ONLY_DOMAINS: dict[str, str] = {
    "google_lens": "lens.google.com",
    "tineye": "tineye.com",
}


def image_registry_for_context(context: Any = None,
                               injected: Any = None) -> Any:
    """The source registry image-search outcomes land in, or None.

    Mirrors ``tools/web.py``'s #90 resolution exactly: an injected
    registry wins; otherwise the shared ``DEFAULT_REGISTRY`` bound
    to the context's KV store — and only when the context carries
    a database, so db-less callers (pure/offline contexts, the
    fake-client test harnesses) keep the pre-#92 behaviour
    exactly.  Never raises."""
    if injected is not None:
        return injected
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return None
        from ..media.source_registry import DEFAULT_REGISTRY
        reg = DEFAULT_REGISTRY
        try:
            from ..storage.kv import KVStore
            reg.bind_store(KVStore(db))
        except Exception:  # noqa: BLE001 - persistence is an
            # enhancement; its absence never sinks a search
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _image_engine_desc(reg: Any, engine: str) -> Any:
    """The registry descriptor for one scraped engine, or None.
    Never raises."""
    try:
        domain = IMAGE_ENGINE_DOMAINS.get(engine)
        if not domain or reg is None:
            return None
        return reg.for_domain(domain)
    except Exception:  # noqa: BLE001
        return None


def ordered_image_engines(reg: Any,
                          engines: tuple[str, ...] | None = None) -> list[str]:
    """Outcome-aware engine order (Pass #92), through the
    registry's score/health semantics, mirroring the #88 resolver
    / #89 adapter / #90 chain ordering:

    * NO learned state anywhere in the roster → the list is
      returned untouched: day-one behaviour is
      :data:`REVERSE_IMAGE_ENGINES`, byte-for-byte;
    * otherwise active engines sort by registry score (stable —
      ties keep roster order) and stood-down / disabled / dead
      engines are demoted to the end, in roster order: they leave
      the active ordering but are still tried last rather than
      pretending they do not exist.
    Never raises; a registry hiccup returns the roster as-is."""
    pool = list(engines) if engines is not None \
        else list(REVERSE_IMAGE_ENGINES)
    try:
        if reg is None:
            return pool
        items: list[tuple[str, Any]] = []
        evidence = False
        for engine in pool:
            desc = _image_engine_desc(reg, engine)
            items.append((engine, desc))
            if desc is None:
                continue
            st = reg.stats_for(desc)
            if st["ok"] or st["fail"] or st["no_result"] \
                    or st.get("blocked") or st.get("rate_limited") \
                    or not reg.is_active(desc):
                evidence = True
        if not evidence:
            return pool
        active = [it for it in items
                  if it[1] is None or reg.is_active(it[1])]
        inactive = [it for it in items
                    if it[1] is not None and not reg.is_active(it[1])]
        active.sort(key=lambda it: -reg.score(it[1])
                    if it[1] is not None else 0.0)
        return [e for e, _d in active] + [e for e, _d in inactive]
    except Exception:  # noqa: BLE001
        return pool


def _record_image_outcome(reg: Any, engine: str, kind: str,
                            latency_s: float = 0.0,
                            error_class: str = "") -> None:
    """Record one scrape attempt's outcome against its engine's
    registry source.  ``kind``: served | no_result | failure |
    blocked | rate_limited.  Best-effort; never raises —
    bookkeeping must never break a search."""
    try:
        if reg is None:
            return
        desc = _image_engine_desc(reg, engine)
        if desc is None:
            return
        if kind == "served":
            reg.record_outcome(desc, True, latency_s=latency_s)
        elif kind == "no_result":
            reg.record_no_result(desc)
        elif kind == "blocked":
            reg.record_blocked(desc, latency_s=latency_s)
        elif kind == "rate_limited":
            # recorded once — a 429 is a hard stop for this engine,
            # never a cue to retry, rotate or re-query around it
            reg.record_rate_limited(desc, latency_s=latency_s)
        else:
            reg.record_outcome(desc, False, latency_s=latency_s,
                               error_class=error_class or "site_error")
    except Exception:  # noqa: BLE001
        pass


def _image_error_class(exc: BaseException) -> str:
    """The error class a scrape transport exception earns: a
    timeout is its own class; anything else is a network failure
    (mirrors ``tools/web.py``'s chain vocabulary)."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "timed out" in msg or "timeout" in msg:
        return "timeout"
    return "network_failure"


def _reverse_impl(context: Any, ref: str,
                  registry: Any = None) -> dict[str, Any]:
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
        # Pass #92: the scrape attempts record their outcomes in
        # the ONE source registry and run in its learned order
        # (day-one pin: no evidence → ("bing", "yandex") exactly,
        # and the payload below is unchanged either way).
        reg = registry if registry is not None \
            else image_registry_for_context(context)
        matches = _scrape_engines(client, public, registry=reg)
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


def _bing_reverse_url(image_url: str) -> str:
    return ("https://www.bing.com/images/search?view=detailv2&iss=sbi&qimgurl="
            + urllib.parse.quote(image_url, safe=""))


def _parse_bing_body(body: bytes) -> list[str]:
    """Related-image media URLs from a Bing visual-search page body."""
    return _dedupe_urls(_MEDIA_URL.findall(body or b""))


def _scrape_bing(client: HttpClient, image_url: str) -> list[str]:
    """Best-effort: related-image media URLs from Bing's visual search page."""
    urls, _kind, _err = _scrape_one("bing", client, image_url)
    return urls


_YANDEX_ORIGINAL = re.compile(rb'"originalImage"\s*:\s*\{\s*"url"\s*:\s*"([^"]{10,300})"', re.IGNORECASE)
_YANDEX_IMG_SRC = re.compile(
    rb'<img[^>]{0,500}?src="(https?://[^"]{10,300}\.(?:jpe?g|png|webp|gif)[^"]{0,60})"',
    re.IGNORECASE)


def _yandex_reverse_url(image_url: str) -> str:
    return ("https://yandex.com/images/search?rpt=imageview&url="
            + urllib.parse.quote(image_url, safe=""))


def _parse_yandex_body(body: bytes) -> list[str]:
    """Similar-image URLs from a Yandex search-by-image page body.
    Yandex embeds the matched image JSON as "originalImage":{"url":...}
    and similar hits as <img> tags — we fish out both."""
    body = body or b""
    hits = _YANDEX_ORIGINAL.findall(body) + _YANDEX_IMG_SRC.findall(body)
    cleaned: list[bytes] = []
    for hit in hits:
        hit = hit.replace(b"\\/", b"/")
        if hit not in cleaned:
            cleaned.append(hit)
    return _dedupe_urls(cleaned)


def _scrape_yandex(client: HttpClient, image_url: str) -> list[str]:
    """Best-effort: similar-image URLs from Yandex's image-search-by-image
    page."""
    urls, _kind, _err = _scrape_one("yandex", client, image_url)
    return urls


#: engine → (reverse-search URL builder, page body parser)
_ENGINE_SCRAPERS = {
    "bing": (_bing_reverse_url, _parse_bing_body),
    "yandex": (_yandex_reverse_url, _parse_yandex_body),
}


def _bot_signalled(body: bytes) -> bool:
    """True when a 200 page is a bot-wall/challenge, not results —
    the registry's one bot-signal vocabulary
    (``media/source_registry._BOT_SIGNALS``, mirrored in #89's
    health classification: a wall is ``bot-blocked``, never
    ``dead`` and never a plain no-result)."""
    try:
        from ..media.source_registry import _BOT_SIGNALS

        signals = _BOT_SIGNALS
    except Exception:  # noqa: BLE001 - vocabulary fallback
        signals = ("captcha", "cloudflare", "just a moment",
                   "verify you are human", "are you a robot",
                   "access denied", "attention required")
    text = (body or b"")[:4000].decode("utf-8", errors="replace").lower()
    return any(sig in text for sig in signals)


def _status_outcome(status: int) -> tuple[str, str]:
    """(kind, error_class) for an HTTP status on the scrape path:
    429 is its own class (#90), 403 is a bot wall (#89), anything
    else non-2xx is an operational failure with its status as the
    error class."""
    if status == 429:
        return "rate_limited", ""
    if status == 403:
        return "blocked", ""
    return "failure", f"http_{status}"


def _scrape_one(engine: str, client: HttpClient,
                image_url: str) -> tuple[list[str], str, str]:
    """One engine's scrape attempt, classified (Pass #92).

    Returns ``(urls, kind, error_class)`` where ``kind`` is the
    registry vocabulary: ``served`` (results landed) |
    ``no_result`` (the engine answered with an empty page — weak
    evidence) | ``failure`` (transport/HTTP/parser failure, with
    ``error_class``) | ``blocked`` (HTTP 403, or a 200 page
    carrying a bot-wall challenge) | ``rate_limited`` (HTTP 429 —
    recorded once by the caller, never retried around).

    The real ``HttpClient`` RAISES on HTTP status, so the status
    is read from the exception (``http_status_of``) as well as
    from duck-typed responses that carry one.  Never raises:
    scraping is optional garnish, exactly as before."""
    url_for, parse = _ENGINE_SCRAPERS[engine]
    try:
        page = client.get(url_for(image_url))
    except Exception as exc:  # noqa: BLE001 - per-engine isolation
        status = http_status_of(exc)
        if status is not None:
            kind, error_class = _status_outcome(status)
            return [], kind, error_class
        return [], "failure", _image_error_class(exc)
    status = getattr(page, "status", None)
    if status is not None and not 200 <= int(status) < 300:
        kind, error_class = _status_outcome(int(status))
        return [], kind, error_class
    body = getattr(page, "body", b"")
    if body is None:
        # a response with no body at all is malformed, not an
        # empty result page (the #90 chain's parser-failure class)
        return [], "failure", "malformed_response"
    try:
        urls = parse(body)
    except Exception:  # noqa: BLE001 - a markup surprise is a
        # malformed response, its own class (#90 precedent)
        return [], "failure", "malformed_response"
    if urls:
        return urls, "served", ""
    if body and _bot_signalled(body if isinstance(body, bytes)
                               else str(body).encode("utf-8", "replace")):
        return [], "blocked", ""
    return [], "no_result", ""


def _scrape_engines(client: HttpClient, image_url: str, *,
                    registry: Any = None) -> dict[str, list[str]]:
    """Run each reverse-search scraper isolated: one engine failing
    (network, block, markup change) never kills the others.

    Pass #92: when *registry* is given, every attempt records its
    outcome against the engine's registry source (served /
    no-result / failure with error class / blocked on a bot wall /
    rate-limited on a 429 — recorded once, never retried), and the
    iteration order is the registry's learned order over the
    roster (day-one pin: no learned state →
    :data:`REVERSE_IMAGE_ENGINES` byte-for-byte).  With no
    registry this is the pre-#92 function, exactly."""
    results: dict[str, list[str]] = {}
    for name in ordered_image_engines(registry):
        t0 = time.time()
        try:
            urls, kind, error_class = _scrape_one(name, client, image_url)
        except Exception:  # noqa: BLE001 - per-engine isolation
            _log.debug("reverse image scrape failed for %s", name, exc_info=True)
            urls, kind, error_class = [], "failure", "network_failure"
        _record_image_outcome(registry, name, kind,
                              time.time() - t0, error_class=error_class)
        results[name] = urls or []
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

    @registry.register(
        "image_sources",
        description=(
            "Image source registry — the reverse image search "
            "engines (Bing visual search, Yandex search-by-image) "
            "in the ONE source registry (pass #92, same registry "
            "as download_sources): action=status (per-engine "
            "origin/health, served / failed / blocked / "
            "rate-limited / no-result counts, plus the link-only "
            "engines Google Lens / TinEye — deep links never "
            "fetched, so no outcomes recorded for them) | health "
            "(run the shared source health loop: the shared hosts "
            "are probed through their real search endpoint; the "
            "reverse endpoints are judged by real-search outcomes)."
        ),
        capability="net.out",
    )
    def image_sources(*, action: str = "status") -> dict[str, Any]:
        reg = image_registry_for_context(context)
        if reg is None:
            from ..media.source_registry import DEFAULT_REGISTRY

            reg = DEFAULT_REGISTRY
        act = (action or "status").strip().lower()
        if act == "health":
            return reg.verify_health()
        if act == "status":
            return reg.image_status()
        return {"error": f"unknown action {act!r} — status|health"}
