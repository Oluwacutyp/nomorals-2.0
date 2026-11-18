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

AA-58 (assembly audit, owner's open-sources law): the absence is
now closed with the channel that DOES exist for engines — a
catalog/manual candidate channel.  :data:`REVERSE_IMAGE_CATALOG`
names additional reverse-search engines whose result pages are
server-rendered and URL-addressable (IQDB, SauceNAO, ascii2d);
:func:`note_reverse_image_candidate` intakes any engine with the
URL-template shape (``{image_url}`` placeholder) as an untrusted
registry candidate (kind ``image-search``).  The shared health
loop probes the candidate's root (the #92 doctrine: a synthetic
reverse probe is impossible, so the endpoint itself is judged by
real reverse-search outcomes); promoted engines join
:func:`ordered_image_engines` and are scraped through the generic
result-page shape (image URLs harvested from the page, the
engine's own host excluded).  Google Lens / TinEye stay link-only:
they remain pages Devon hands the user, never scraped engines.

52-websites #46 (TinEye, 2026-11-17): the fingerprint LADDER.
dHash is the fast rung and stays; beside it a block-mean/gradient
descriptor rung (:func:`block_descriptor`) — 8x8 normalized block
means over the central 60% of the frame (crop-tolerant by
construction) plus horizontal/vertical gradient-sign bits, so a
copy that survives a crop, a resize, a recolor or a watermark
still matches.  Each rung's robustness is MEASURED against a
mutation corpus (:func:`fingerprint_benchmark` — 6 bases x 12
mutations: resize 50%/200%, center crop 80%/60%, hue swap,
grayscale, brightness ±30%, contrast, watermark bar/corner,
JPEG 60) and recorded in :func:`fingerprint_robustness`, never
asserted in a comment: block rung 69/72 overall (crop80 5/6,
crop60 4/6 — the two misses are the periodic checkerboard, whose
phase flips under a crop, and the noise base), dHash 61/72
(crop80 2/6, crop60 0/6 — the crop fragility the report names);
distinct pairs stay separate (block distance min 0.349 over the
corpus, dHash hamming min 24).  The web half now returns a
uniform :class:`ReverseMatch` per engine (``engine, page_url,
image_url, score, verified``) — scraped results are unverified
unless :func:`verify_reverse_matches` downloads the candidate
and fingerprints it against the query.  And the local index
grows by discovery: :func:`note_image_file` fingerprints any
landed image into ``image_index`` — the downloader calls it on
every landed image file, so her private TinEye gets bigger with
use.  The honest claim stays MatchEngine-shaped (her library,
robust matching), never TinEye-shaped (the whole web).
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import re
import shutil
import struct
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any

from ..core.errors import http_status_of
from ..core.logging_setup import get_logger
from ..core.http import HttpClient
from nomorals.tools.imagedb_failures import record_failure

_log = get_logger(__name__)

__all__ = ["register", "sniff_format", "image_dimensions", "dhash", "hamming",
           "image_exif", "image_colors", "find_dupes", "image_stats",
           "REVERSE_IMAGE_ENGINES", "IMAGE_ENGINE_DOMAINS",
           "IMAGE_LINK_ONLY_DOMAINS", "image_registry_for_context",
           "ordered_image_engines", "REVERSE_IMAGE_CATALOG",
           "grown_image_engines", "note_reverse_image_candidate",
           "discover_reverse_image_engines",
           # 52-websites #46 (TinEye): the fingerprint ladder, the
           # uniform reverse-match shape, the verification rung and
           # index growth by discovery — appended in __all__ order.
           "ReverseMatch", "block_descriptor", "block_distance",
           "block_match", "BLOCK_MATCH_THRESHOLD", "fingerprint",
           "fingerprint_distance", "fingerprints_match",
           "fingerprint_benchmark", "fingerprint_robustness",
           "note_image_file", "verify_reverse_matches"]


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
            except Exception as e:  # noqa: BLE001
                record_failure("jpeg_dimensions", e)
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


# ── fingerprint ladder, second rung: block-mean/gradient descriptor ─────────
# 52-websites #46 (TinEye). dHash is the fast rung and stays; this is the
# robust rung.  A descriptor is 8x8 normalized block-mean luminance over
# the CENTRAL 60% of the frame — a crop that trims the edges leaves the
# center content in place, which is exactly where dHash falls over —
# plus horizontal/vertical gradient-sign bits, so the descriptor still
# separates two different pictures that happen to share a mean tone.
# Normalization (mean-subtract, divide by std) makes it invariant to
# linear brightness/contrast shifts; grayscale makes it invariant to
# hue.  Encoded "bm1:<64 nibbles>:<16 hex h-bits>:<16 hex v-bits>".
# BLOCK_MATCH_THRESHOLD (0.25) is the measured cut: on the mutation
# corpus (see fingerprint_benchmark) mutated copies sit at <= 0.25
# except the periodic checkerboard under a crop, and distinct pairs
# sit at >= 0.349.

BLOCK_DESCRIPTOR_VERSION = "bm1"

#: Measured match threshold for the block rung (see
#: :func:`fingerprint_benchmark`): a mutated copy of the same image
#: lands at or under this distance; distinct images land above it.
BLOCK_MATCH_THRESHOLD = 0.25

#: dHash match threshold — the pre-existing near-duplicate bar
#: (find_dupes default 6, lookup near-dupes 10).  Named here so the
#: benchmark and the ladder read the same number.
DHASH_MATCH_THRESHOLD = 10


def _block_grid(img: Any) -> list[float] | None:
    """8x8 normalized block-mean luminance over the central 60%."""
    Image = _pil_image()
    if Image is None:
        return None
    try:
        w, h = img.size
        crop = img.convert("L").crop(
            (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
        small = crop.resize((8, 8), Image.LANCZOS)
        vals = [float(v) for v in small.getdata()]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        if std < 1e-6:
            return [0.0] * 64
        return [max(-2.0, min(2.0, (v - mean) / std)) for v in vals]
    except Exception as exc:  # noqa: BLE001 - descriptor is best-effort
        assert isinstance(exc, Exception)
        return None


def _gradient_bits(img: Any) -> tuple[int, int] | None:
    """Horizontal + vertical gradient-sign bits over the same 8x8
    central grid (64 bits each)."""
    Image = _pil_image()
    if Image is None:
        return None
    try:
        w, h = img.size
        crop = img.convert("L").crop(
            (int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)))
        grid = list(crop.resize((8, 8), Image.LANCZOS).getdata())
        hbits = 0
        for r in range(8):
            for c in range(8):
                nxt = grid[r * 8 + (c + 1) % 8]
                hbits = (hbits << 1) | (1 if nxt > grid[r * 8 + c] else 0)
        vbits = 0
        for r in range(8):
            for c in range(8):
                nxt = grid[((r + 1) % 8) * 8 + c]
                vbits = (vbits << 1) | (1 if nxt > grid[r * 8 + c] else 0)
        return hbits, vbits
    except Exception as exc:  # noqa: BLE001 - descriptor is best-effort
        assert isinstance(exc, Exception)
        return None


def block_descriptor(path: str | Path) -> str | None:
    """Robust perceptual descriptor for one image file.

    Returns None when Pillow is unavailable or the file cannot be
    read — the content hash still works, robust matching just
    degrades to the dHash rung."""
    Image = _pil_image()
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            norm = _block_grid(img)
            grads = _gradient_bits(img)
        if norm is None or grads is None:
            return None
        nibbles = "".join(
            f"{max(0, min(15, int((v + 2.0) / 4.0 * 15 + 0.5))):x}"
            for v in norm)
        hbits, vbits = grads
        return (f"{BLOCK_DESCRIPTOR_VERSION}:{nibbles}:"
                f"{hbits:016x}:{vbits:016x}")
    except Exception as exc:  # noqa: BLE001 - descriptor is best-effort
        assert isinstance(exc, Exception)
        return None


def _parse_block_descriptor(desc: str) -> tuple[list[int], int, int] | None:
    """"bm1:<64 nibbles>:<16 hex>:<16 hex>" -> (nibbles, hbits, vbits)."""
    try:
        if not desc or not isinstance(desc, str):
            return None
        parts = desc.split(":")
        if len(parts) != 4 or parts[0] != BLOCK_DESCRIPTOR_VERSION:
            return None
        nibbles, hbits, vbits = parts[1], parts[2], parts[3]
        if len(nibbles) != 64 or len(hbits) != 16 or len(vbits) != 16:
            return None
        return [int(c, 16) for c in nibbles], int(hbits, 16), int(vbits, 16)
    except (ValueError, AttributeError):
        return None


def block_distance(a: str | None, b: str | None) -> float | None:
    """Normalized distance in [0, 1] between two block descriptors:
    0.6 * mean nibble difference (/15) + 0.2 * horizontal-gradient
    hamming (/64) + 0.2 * vertical-gradient hamming (/64).
    None when either descriptor is missing or malformed."""
    pa = _parse_block_descriptor(a) if a else None
    pb = _parse_block_descriptor(b) if b else None
    if pa is None or pb is None:
        return None
    na, ha, va = pa
    nb, hb, vb = pb
    mean_diff = sum(abs(x - y) for x, y in zip(na, nb, strict=True)) / 64.0 / 15.0
    hdist = bin(ha ^ hb).count("1") / 64.0
    vdist = bin(va ^ vb).count("1") / 64.0
    return 0.6 * mean_diff + 0.2 * hdist + 0.2 * vdist


def block_match(a: str | None, b: str | None,
                threshold: float = BLOCK_MATCH_THRESHOLD) -> bool | None:
    """True when two block descriptors are within ``threshold``.
    None when either descriptor is missing or malformed (the rung
    cannot speak — never a silent match)."""
    dist = block_distance(a, b)
    if dist is None:
        return None
    return dist <= threshold


def fingerprint(path: str | Path) -> dict[str, str | None] | None:
    """Both rungs for one image: {"dhash": ..., "block": ...}.
    Returns None when Pillow is unavailable (neither rung can run);
    a rung that cannot be computed for a given file is None inside
    the dict, never an invented value."""
    if _pil_image() is None:
        return None
    return {"dhash": dhash(path), "block": block_descriptor(path)}


def fingerprint_distance(a: dict[str, Any] | None,
                         b: dict[str, Any] | None) -> dict[str, Any]:
    """Per-rung distances between two :func:`fingerprint` dicts:
    {"dhash": int|None, "block": float|None}.  A rung that cannot
    be computed on either side is None, never zero."""
    if not a or not b:
        return {"dhash": None, "block": None}
    return {
        "dhash": hamming(a.get("dhash") or "", b.get("dhash") or ""),
        "block": block_distance(a.get("block"), b.get("block")),
    }


def fingerprints_match(a: dict[str, Any] | None,
                       b: dict[str, Any] | None, *,
                       dhash_threshold: int = DHASH_MATCH_THRESHOLD,
                       block_threshold: float = BLOCK_MATCH_THRESHOLD
                       ) -> dict[str, bool]:
    """Per-rung match verdicts between two :func:`fingerprint`
    dicts.  A rung that cannot be computed does NOT match (False)
    — an uncomputed rung is never evidence of sameness.
    "either" is the near-duplicate bar (one rung carrying the
    pair is enough); "both" is the strict bar."""
    dists = fingerprint_distance(a, b)
    d_match = dists["dhash"] is not None and dists["dhash"] <= dhash_threshold
    b_match = dists["block"] is not None and dists["block"] <= block_threshold
    return {"dhash": bool(d_match), "block": bool(b_match),
            "either": bool(d_match or b_match),
            "both": bool(d_match and b_match)}


# ── the mutation corpus: measured robustness, never asserted ───────────────
# 52-websites #46 (TinEye).  The corpus below is deliberately the same
# shape as the one in tests/test_gapfix_tineye.py — the test measures
# INDEPENDENTLY with its own builders and pins floors on this
# function's output, so the two measurements check each other.
# Six textured bases x twelve TinEye-style mutations (resize 50%/200%,
# center crop 80%/60%, hue swap, grayscale, brightness ±30%, contrast,
# watermark bar/corner, JPEG 60).  Measured on this box (2026-11-17):
# block rung 69/72 overall (crop80 5/6, crop60 4/6 — the misses are
# the periodic checkerboard, whose phase flips under a crop, and the
# noise base), dHash 61/72 (crop80 2/6, crop60 0/6); distinct pairs
# stay separate (block distance min 0.349, dHash hamming min 24).

_BENCHMARK_MUTATIONS: tuple[str, ...] = (
    "resize50", "resize200", "crop80", "crop60", "hue", "grayscale",
    "bright+30", "bright-30", "contrast", "watermark_bar",
    "watermark_corner", "jpeg60")


def _benchmark_base(name: str, path: Path) -> Path:
    """One textured base image.  Pillow is required — the caller
    checks and returns None without it."""
    import random

    from PIL import Image, ImageDraw

    if name == "grad":
        img = Image.new("RGB", (256, 256))
        px = img.load()
        for y in range(256):
            for x in range(256):
                px[x, y] = ((x * 3) % 256, (y * 2) % 256, ((x + y) % 256))
    elif name == "stripes":
        img = Image.new("RGB", (256, 256))
        px = img.load()
        for y in range(256):
            for x in range(256):
                on = (x // 16) % 2 == 0
                px[x, y] = (220, 30, 30) if on else (20, 20, 220)
    elif name == "rings":
        img = Image.new("RGB", (256, 256), (10, 10, 40))
        dr = ImageDraw.Draw(img)
        for r in range(120, 0, -20):
            dr.ellipse([128 - r, 128 - r, 128 + r, 128 + r],
                       fill=((r * 2) % 256, 120, 200 - r))
    elif name == "noise":
        rnd = random.Random(42)
        img = Image.new("RGB", (256, 256))
        px = img.load()
        for y in range(256):
            for x in range(256):
                px[x, y] = (rnd.randrange(256), rnd.randrange(256),
                            rnd.randrange(256))
        img = img.resize((32, 32), Image.LANCZOS).resize(
            (256, 256), Image.NEAREST)
    elif name == "scene":
        img = Image.new("RGB", (256, 256), (90, 160, 220))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 170, 256, 256], fill=(40, 120, 50))
        dr.ellipse([180, 30, 230, 80], fill=(250, 220, 80))
        dr.rectangle([20, 120, 90, 170], fill=(120, 70, 40))
    elif name == "check":
        img = Image.new("RGB", (256, 256))
        px = img.load()
        for y in range(256):
            for x in range(256):
                on = ((x // 32) + (y // 32)) % 2 == 0
                px[x, y] = (240, 240, 240) if on else (15, 15, 15)
    else:
        raise ValueError(f"unknown benchmark base {name!r}")
    img.save(path, "PNG")
    return path


def _benchmark_mutate(img: Any, kind: str, out_path: Path) -> None:
    """Apply one TinEye-style mutation to an open Pillow image."""
    from PIL import Image, ImageDraw, ImageEnhance

    if kind == "resize50":
        img.resize((128, 128), Image.LANCZOS).save(out_path)
    elif kind == "resize200":
        img.resize((512, 512), Image.LANCZOS).save(out_path)
    elif kind == "crop80":
        w, h = img.size
        img.crop((int(w * 0.1), int(h * 0.1),
                  int(w * 0.9), int(h * 0.9))).save(out_path)
    elif kind == "crop60":
        w, h = img.size
        img.crop((int(w * 0.2), int(h * 0.2),
                  int(w * 0.8), int(h * 0.8))).save(out_path)
    elif kind == "hue":
        r, g, b = img.split()
        Image.merge("RGB", (b, r, g)).save(out_path)
    elif kind == "grayscale":
        img.convert("L").convert("RGB").save(out_path)
    elif kind == "bright+30":
        ImageEnhance.Brightness(img).enhance(1.3).save(out_path)
    elif kind == "bright-30":
        ImageEnhance.Brightness(img).enhance(0.7).save(out_path)
    elif kind == "contrast":
        ImageEnhance.Contrast(img).enhance(1.6).save(out_path)
    elif kind == "watermark_bar":
        m = img.copy()
        dr = ImageDraw.Draw(m, "RGBA")
        dr.rectangle([0, 220, 256, 256], fill=(255, 255, 255, 140))
        m.convert("RGB").save(out_path)
    elif kind == "watermark_corner":
        m = img.copy()
        dr = ImageDraw.Draw(m, "RGBA")
        dr.rectangle([180, 180, 256, 256], fill=(255, 255, 255, 160))
        m.convert("RGB").save(out_path)
    elif kind == "jpeg60":
        img.convert("RGB").save(out_path, "JPEG", quality=60)
    else:
        raise ValueError(f"unknown benchmark mutation {kind!r}")


def fingerprint_benchmark() -> dict[str, Any]:
    """Measure both fingerprint rungs against the mutation corpus.

    Returns {"corpus": {"bases", "mutations", "pairs"}, "thresholds":
    {"dhash", "block"}, "rungs": {"dhash": {overall + per-mutation
    match rates}, "block": {...}}, "separation": {"block_min",
    "dhash_min"}} — rates are fractions of mutated copies that
    still match their original; separation is the closest distinct
    pair.  Runs in ~1s (small synthetic images).  When Pillow is
    unavailable the rungs cannot be measured: the report says so
    with empty rates rather than inventing them."""
    Image = _pil_image()
    report: dict[str, Any] = {
        "corpus": {"bases": 6, "mutations": len(_BENCHMARK_MUTATIONS),
                   "pairs": 6 * len(_BENCHMARK_MUTATIONS)},
        "thresholds": {"dhash": DHASH_MATCH_THRESHOLD,
                       "block": BLOCK_MATCH_THRESHOLD},
        "rungs": {"dhash": {}, "block": {}},
        "separation": {"block_min": None, "dhash_min": None},
    }
    if Image is None:
        report["error"] = "Pillow unavailable — rungs cannot be measured"
        return report
    bases = ("grad", "stripes", "rings", "noise", "scene", "check")
    root = Path(tempfile.mkdtemp(prefix="tineye_bench_"))
    try:
        base_paths = {name: _benchmark_base(name, root / f"{name}.png")
                      for name in bases}
        dh = {n: dhash(p) for n, p in base_paths.items()}
        bd = {n: block_descriptor(p) for n, p in base_paths.items()}
        # distinct-pair separation first: the closest two different
        # bases on each rung.
        block_mins: list[float] = []
        dhash_mins: list[int] = []
        for i in range(len(bases)):
            for j in range(i + 1, len(bases)):
                dist = block_distance(bd[bases[i]], bd[bases[j]])
                if dist is not None:
                    block_mins.append(dist)
                ham = hamming(dh[bases[i]], dh[bases[j]])
                if ham is not None:
                    dhash_mins.append(ham)
        report["separation"] = {
            "block_min": round(min(block_mins), 3) if block_mins else None,
            "dhash_min": min(dhash_mins) if dhash_mins else None,
        }
        for kind in _BENCHMARK_MUTATIONS:
            d_ok = b_ok = 0
            for name, path in base_paths.items():
                mutated = root / f"{name}_{kind}.png"
                with Image.open(path) as im:
                    _benchmark_mutate(im, kind, mutated)
                ham = hamming(dh[name], dhash(mutated))
                if ham is not None and ham <= DHASH_MATCH_THRESHOLD:
                    d_ok += 1
                dist = block_distance(bd[name], block_descriptor(mutated))
                if dist is not None and dist <= BLOCK_MATCH_THRESHOLD:
                    b_ok += 1
            report["rungs"]["dhash"][kind] = round(d_ok / len(bases), 3)
            report["rungs"]["block"][kind] = round(b_ok / len(bases), 3)
        for rung in ("dhash", "block"):
            rates = report["rungs"][rung].values()
            report["rungs"][rung]["overall"] = round(
                sum(rates) / len(rates), 3) if rates else 0.0
        return report
    except Exception as e:  # noqa: BLE001 - measurement never raises
        record_failure("fingerprint_benchmark", e)
        report["error"] = f"{type(e).__name__}: {e}"[:200]
        return report
    finally:
        shutil.rmtree(root, ignore_errors=True)


_ROBUSTNESS_CACHE: dict[str, Any] | None = None


def fingerprint_robustness() -> dict[str, Any]:
    """The measured robustness table, computed once per process
    (the benchmark builds its corpus each run; the status surface
    calls this, not the benchmark directly).  Never raises."""
    global _ROBUSTNESS_CACHE
    if _ROBUSTNESS_CACHE is None:
        try:
            _ROBUSTNESS_CACHE = fingerprint_benchmark()
        except Exception as e:  # noqa: BLE001
            record_failure("fingerprint_robustness", e)
            _ROBUSTNESS_CACHE = {
                "error": f"{type(e).__name__}: {e}"[:200],
                "rungs": {"dhash": {}, "block": {}},
                "thresholds": {"dhash": DHASH_MATCH_THRESHOLD,
                               "block": BLOCK_MATCH_THRESHOLD},
                "separation": {"block_min": None, "dhash_min": None},
            }
    return _ROBUSTNESS_CACHE


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
    except Exception as e:  # noqa: BLE001
        record_failure("image_exif:open", e)
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
        except Exception as e:  # noqa: BLE001
            record_failure("image_exif:exposure", e)
    if fnumber:
        try:
            out["f_number"] = float(fnumber)
        except Exception as e:  # noqa: BLE001
            record_failure("image_exif:f_number", e)
    if iso:
        out["iso"] = iso

    try:
        gps_ifd = raw.get_ifd(_GPS_IFD) if hasattr(raw, "get_ifd") else {}
        if gps_ifd:
            gps = {GPSTAGS.get(tag, tag): value for tag, value in gps_ifd.items()}
            coords = _gps_decimal(gps)
            if coords:
                out["gps"] = {"lat": coords[0], "lon": coords[1]}
    except Exception as e:  # noqa: BLE001
        record_failure("image_exif:gps", e)
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


def _image_index(context: Any):
    """The seam-backed image_index table (Mongo P3), or None on the
    SQLite default, where the native statements run unchanged."""
    db = getattr(context, "db", None)
    if db is None:
        return None
    from ..storage.rowstore import attached_rows
    return attached_rows(db, "image_index")


def _index_row(context: Any, digest: str, path: Path, size: int, fmt: str,
               seen_in: str) -> dict[str, Any] | None:
    """Upsert into image_index; returns the previous sighting (if any)."""
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        table = _image_index(context)
        if table is not None:
            row = table.get(digest)
            now = time.time()
            table.put({
                "hash": digest, "path": str(path), "size": size,
                "mime": fmt,
                "first_seen": now if row is None else row["first_seen"],
                "last_seen": now,
                "seen_in": seen_in if seen_in != ""
                else (row or {}).get("seen_in", ""),
            })
            return row
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
    except Exception as e:  # noqa: BLE001 - indexing must never break a lookup
        record_failure("_index_row", e)
        return None


def note_image_file(context: Any, path: str | Path,
                    seen_in: str = "download") -> dict[str, Any]:
    """Fingerprint one image file into the local index (52-websites
    #46, index growth by discovery): content digest + both
    fingerprint rungs, upserted into ``image_index`` with
    ``seen_in`` naming where it came from.  Only the first 16
    bytes are read to sniff the format — a non-image (or a
    multi-gigabyte video) is refused without reading the file.
    Returns {"ok": True, "digest", "dhash", "block",
    "seen_before", "path"} or {"ok": False, "reason": ...}.
    Never raises: indexing must never break the caller."""
    try:
        if context is None or getattr(context, "db", None) is None:
            return {"ok": False, "reason": "no database in reach"}
        p = Path(path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "reason": f"not a file: {path}"}
        with p.open("rb") as fh:
            head = fh.read(16)
        if sniff_format(head) == "unknown":
            return {"ok": False, "reason": "not an image file"}
        data = p.read_bytes()
        fmt = sniff_format(data)
        digest = hashlib.blake2b(data, digest_size=16).hexdigest()
        previous = _index_row(context, digest, p, len(data), fmt, seen_in)
        return {"ok": True, "digest": digest,
                "dhash": dhash(p), "block": block_descriptor(p),
                "seen_before": previous is not None, "path": str(p)}
    except Exception as e:  # noqa: BLE001
        record_failure("note_image_file", e)
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"[:200]}


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
    query_fp = fingerprint(local)
    phash = query_fp["dhash"] if query_fp else None
    query_block = query_fp["block"] if query_fp else None
    near: list[str] = []
    if phash or query_block:
        db = getattr(context, "db", None)
        if db is not None:
            try:
                table = _image_index(context)
                index_rows = table.all() if table is not None else \
                    db.query("SELECT hash, path FROM image_index")
                # compare perceptual hashes (not content digests — dhash is
                # 16 hex chars, blake2b digests are 32, so they can never match)
                # 52-websites #46: both rungs speak — a row matches when
                # EITHER rung matches, and the entry names the rung(s)
                # that carried it.
                for row in index_rows:
                    row_path = row.get("path", "")
                    if not row_path:
                        continue
                    other_fp = fingerprint(row_path)
                    if not other_fp:
                        continue
                    distance = hamming(phash or "", other_fp["dhash"] or "")
                    d_hit = distance is not None and 0 < distance <= 10
                    block_dist = block_distance(
                        query_block, other_fp["block"])
                    b_hit = block_dist is not None and \
                        block_dist <= BLOCK_MATCH_THRESHOLD and \
                        block_dist > 0.0
                    if d_hit and b_hit:
                        near.append(
                            f"{row['path']} (distance {distance}, "
                            f"block {block_dist:.2f})")
                    elif d_hit:
                        near.append(f"{row['path']} (distance {distance})")
                    elif b_hit:
                        near.append(
                            f"{row['path']} (block {block_dist:.2f})")
            except Exception as e:  # noqa: BLE001
                record_failure("_lookup_impl:near_dupes", e)
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
        table = _image_index(context)
        if table is not None:
            return table.all()
        return list(db.query("SELECT hash, path, size, mime FROM image_index"))
    except Exception as e:  # noqa: BLE001
        record_failure("_library_rows", e)
        return []


def find_dupes(context: Any, threshold: int = 6) -> dict[str, Any]:
    """Cluster every indexed image by perceptual-hash distance.

    Files that no longer exist on disk (or can't be hashed — e.g. Pillow
    missing) are excluded. Returns groups of near-duplicates, each with
    paths, content digests, and max pairwise distance.

    52-websites #46: two images cluster when EITHER rung matches —
    dHash hamming <= ``threshold`` OR block distance <=
    :data:`BLOCK_MATCH_THRESHOLD`.  The dHash rung alone misses
    cropped copies (crop80: dHash distance 16 on the benchmark
    gradient); the block rung carries those pairs.  ``max_distance``
    stays the dHash figure (it can exceed ``threshold`` when the
    block rung carried the pair); ``max_block_distance`` is the
    block figure beside it."""
    rows = _library_rows(context)
    if not rows:
        return {"ok": True, "threshold": threshold, "scanned": 0, "groups": []}

    items: list[dict[str, Any]] = []
    for row in rows:
        path = row.get("path") or ""
        fp = fingerprint(path) if path and Path(path).exists() else None
        items.append({"path": path, "digest": row.get("hash", ""),
                      "dhash": fp["dhash"] if fp else None,
                      "block": fp["block"] if fp else None})

    hashable = [i for i in items if i["dhash"] or i["block"]]
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

    def _pair_matches(a: dict[str, Any], b: dict[str, Any]) -> bool:
        dist = hamming(a["dhash"] or "", b["dhash"] or "")
        if dist is not None and dist <= threshold:
            return True
        return block_match(a["block"], b["block"]) is True

    for i in range(len(hashable)):
        for j in range(i + 1, len(hashable)):
            if _pair_matches(hashable[i], hashable[j]):
                union(i, j)

    clusters: dict[int, list[dict[str, Any]]] = {}
    for idx, item in enumerate(hashable):
        clusters.setdefault(find(idx), []).append(item)

    groups = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        max_dist = 0
        max_block = 0.0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                dist = hamming(members[i]["dhash"] or "",
                               members[j]["dhash"] or "") or 0
                max_dist = max(max_dist, dist)
                bdist = block_distance(members[i]["block"],
                                       members[j]["block"])
                if bdist is not None:
                    max_block = max(max_block, bdist)
        groups.append({
            "count": len(members),
            "max_distance": max_dist,
            "max_block_distance": round(max_block, 3),
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
    except Exception as e:  # noqa: BLE001 - stats must never fail on clustering
        record_failure("image_stats:dupe_groups", e)
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

#: The reverse-image engine CATALOG (AA-58): additional engines
#: whose search-by-image pages are server-rendered and addressed
#: by a URL template carrying the image URL — the one shape the
#: generic result-page scraper can serve honestly.  A catalog row
#: is a CANDIDATE, never an engine: it intakes through
#: :func:`note_reverse_image_candidate` and only the registry
#: health loop can promote it into the scrape roster.
REVERSE_IMAGE_CATALOG: tuple[dict[str, str], ...] = (
    {"id": "iqdb", "name": "IQDB", "domain": "iqdb.org",
     "template": "https://iqdb.org/?url={image_url}"},
    {"id": "saucenao", "name": "SauceNAO", "domain": "saucenao.com",
     "template": "https://saucenao.com/search.php?url={image_url}"},
    {"id": "ascii2d", "name": "ascii2d", "domain": "ascii2d.net",
     "template": "https://ascii2d.net/search/url/{image_url}"},
)

#: Grown reverse engines (AA-58) carry chain names ``grown:<host>``
#: — the host rides in the name so every name-keyed seam resolves
#: the descriptor statelessly (the AA-54 SearXNG precedent).
_GROWN_IMAGE_PREFIX = "grown:"

#: The registry kind the image class's grown sources carry — the
#: same label ``image_status`` renders its grown rows from.
_IMAGE_KIND = "image-search"


def note_reverse_image_candidate(reg: Any, url_or_domain: str,
                                 template: str, *,
                                 via: str = "manual") -> bool:
    """Intake one reverse-image engine as an UNTRUSTED registry
    candidate (AA-58): the manual/catalog candidate channel.

    ``template`` is the engine's search-by-image URL with an
    ``{image_url}`` placeholder — the shape the generic scraper
    serves.  The candidate carries the template so a promotion
    keeps it; the health loop probes the host root (the reverse
    endpoint itself is judged by real outcomes, #92).  Returns
    True when a NEW candidate was noted.  Never raises."""
    try:
        if reg is None or "{image_url}" not in (template or ""):
            return False
        host = reg._host_of(url_or_domain)
        if not host or "." not in host:
            return False
        is_new = reg.note_candidate(
            host, via=f"reverse-image {via}", kinds=(_IMAGE_KIND,))
        for cand in reg.candidates():
            if cand.domain == host \
                    and cand.search_url_template != template:
                # SourceDescriptor is frozen: replace the pending
                # candidate (its key is template-independent).
                import dataclasses

                reg._candidates[cand.key] = dataclasses.replace(
                    cand, search_url_template=template)
                break
        try:
            reg._persist()
        except Exception as exc:  # noqa: BLE001 - enhancement only,
            # but never silent
            record_failure("note_reverse_image_candidate.persist", exc)
        return bool(is_new)
    except Exception as e:  # noqa: BLE001
        record_failure("note_reverse_image_candidate", e)
        return False


def discover_reverse_image_engines(reg: Any) -> dict[str, Any]:
    """Intake the whole reverse-image catalog as candidates
    (AA-58).  Promotion is the health loop's act, not this
    function's.  Never raises."""
    summary: dict[str, Any] = {"listed": len(REVERSE_IMAGE_CATALOG),
                               "noted": 0, "already": 0}
    try:
        if reg is None:
            summary["error"] = "no source registry in reach"
            return summary
        for row in REVERSE_IMAGE_CATALOG:
            if note_reverse_image_candidate(
                    reg, row["domain"], row["template"],
                    via="catalog"):
                summary["noted"] += 1
            else:
                summary["already"] += 1
        return summary
    except Exception as e:  # noqa: BLE001
        record_failure("discover_reverse_image_engines", e)
        summary["error"] = f"{type(e).__name__}: {e}"[:200]
        return summary


def grown_image_engines(reg: Any) -> dict[str, str]:
    """Grown reverse-image engines (AA-58): chain name → URL
    template, for registry sources the health loop promoted from
    image candidates (origin ``grown``, kind ``image-search``)
    carrying a usable ``{image_url}`` template.  Candidates serve
    nothing.  Never raises."""
    try:
        if reg is None:
            return {}
        out: dict[str, str] = {}
        for desc in reg.by_kind(_IMAGE_KIND):
            if reg.origin_for(desc) != "grown":
                continue
            template = getattr(desc, "search_url_template", "") or ""
            if "{image_url}" not in template:
                continue
            out[_GROWN_IMAGE_PREFIX + desc.domain] = template
        return dict(sorted(out.items()))
    except Exception as e:  # noqa: BLE001
        record_failure("grown_image_engines", e)
        return {}


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
        except Exception as e:  # noqa: BLE001 - persistence is an
            record_failure("image_registry_for_context:bind_store", e)
            # enhancement; its absence never sinks a search
            pass
        return reg
    except Exception as e:  # noqa: BLE001
        record_failure("image_registry_for_context", e)
        return None


def _image_engine_desc(reg: Any, engine: str) -> Any:
    """The registry descriptor for one scraped engine, or None.
    Never raises."""
    try:
        domain = IMAGE_ENGINE_DOMAINS.get(engine)
        if not domain and engine.startswith(_GROWN_IMAGE_PREFIX) \
                and reg is not None:
            # a grown engine's name carries its host (AA-58)
            domain = engine[len(_GROWN_IMAGE_PREFIX):]
        if not domain or reg is None:
            return None
        return reg.for_domain(domain)
    except Exception as e:  # noqa: BLE001
        record_failure("_image_engine_desc", e)
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
    if engines is not None:
        pool = list(engines)
    else:
        # AA-58: the default roster is the seed engines PLUS the
        # grown engines the health loop promoted (learned order
        # applies identically).  An explicit ``engines`` argument
        # stays exactly as passed.
        pool = list(REVERSE_IMAGE_ENGINES)
        if reg is not None:
            for name in grown_image_engines(reg):
                if name not in pool:
                    pool.append(name)
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
    except Exception as e:  # noqa: BLE001
        record_failure("ordered_image_engines", e)
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
    except Exception as e:  # noqa: BLE001
        record_failure("_record_image_outcome", e)


def _image_error_class(exc: BaseException) -> str:
    """The error class a scrape transport exception earns: a
    timeout is its own class; anything else is a network failure
    (mirrors ``tools/web.py``'s chain vocabulary)."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if "timed out" in msg or "timeout" in msg:
        return "timeout"
    return "network_failure"


# ── uniform reverse-match shape + verification (52-websites #46) ────────────

@dataclasses.dataclass(frozen=True)
class ReverseMatch:
    """One reverse-image result in the uniform shape across engines:
    the engine that served it, the search page it came from, the
    image URL itself, a rank score (1.0 for the engine's first
    result, decaying by 0.1 per position, floor 0.1 — a RANK score,
    never a similarity), and whether Devon VERIFIED it by
    downloading the candidate and fingerprinting it against the
    query (:func:`verify_reverse_matches`).  A scraped result is
    unverified unless that rung ran: a URL harvested from a page
    is evidence a page shows the picture, not proof it IS the
    picture."""
    engine: str
    page_url: str
    image_url: str
    score: float
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "page_url": self.page_url,
                "image_url": self.image_url, "score": self.score,
                "verified": self.verified}


def _rank_score(index: int) -> float:
    """Rank score for the ``index``-th result of one engine."""
    return round(max(0.1, 1.0 - 0.1 * index), 2)


def _engine_page_url(engine: str, image_url: str,
                     registry: Any = None) -> str:
    """The reverse-search page URL one engine is fetched from, or
    "" when the engine has no usable scraper.  Never raises."""
    try:
        scraper = _ENGINE_SCRAPERS.get(engine)
        if scraper is None:
            scraper = _grown_scraper(registry, engine)
        if scraper is None:
            return ""
        return scraper[0](image_url)
    except Exception as e:  # noqa: BLE001
        record_failure("_engine_page_url", e)
        return ""


def _scrape_matches(client: HttpClient, image_url: str, *,
                    registry: Any = None) -> dict[str, list[ReverseMatch]]:
    """The scrape, in the uniform :class:`ReverseMatch` shape:
    one list per engine, each match carrying its engine, the
    search page it came from, a rank score, and verified=False
    (scraped results stay unverified until the verification rung
    runs).  Reuses :func:`_scrape_engines` — one scrape, one
    outcome recording — and only reshapes its output.  Never
    raises: a shaping failure yields that engine's list empty."""
    try:
        by_engine = _scrape_engines(client, image_url, registry=registry)
    except Exception as e:  # noqa: BLE001 - shaping never sinks a search
        record_failure("_scrape_matches", e)
        return {}
    out: dict[str, list[ReverseMatch]] = {}
    for engine, urls in by_engine.items():
        page_url = _engine_page_url(engine, image_url, registry)
        matches: list[ReverseMatch] = []
        for index, url in enumerate(urls):
            try:
                matches.append(ReverseMatch(
                    engine=engine, page_url=page_url, image_url=url,
                    score=_rank_score(index), verified=False))
            except Exception as e:  # noqa: BLE001 - one bad URL is not the engine
                record_failure("_scrape_matches:match", e)
                continue
        out[engine] = matches
    return out


def verify_reverse_matches(context: Any, query_path: str | Path,
                           matches: list[ReverseMatch], *,
                           client: HttpClient | None = None,
                           block_threshold: float = BLOCK_MATCH_THRESHOLD,
                           dhash_threshold: int = DHASH_MATCH_THRESHOLD
                           ) -> list[ReverseMatch]:
    """Verify scraped matches against the query image for real:
    each candidate is downloaded (size-capped, into media/inbound
    like any resolved image) and fingerprinted; a match verifies
    only when the BLOCK rung matches — the dHash rung is the
    fallback when the block descriptor cannot be computed (no
    Pillow).  Returns a NEW list (the input is untouched);
    unreachable/unreadable candidates stay unverified.  Never
    raises: verification is garnish, exactly like scraping."""
    try:
        query_fp = fingerprint(query_path) if Path(query_path).exists() else None
    except Exception as e:  # noqa: BLE001
        record_failure("verify_reverse_matches:query", e)
        query_fp = None
    if not query_fp:
        return list(matches)
    if client is None:
        tools = getattr(getattr(context, "settings", None), "tools", None)
        client = HttpClient(timeout=getattr(tools, "http_timeout", 30.0),
                            user_agent=getattr(tools, "user_agent", "NoMoralsCore/0.1"),
                            proxy_url=getattr(tools, "proxy_url", ""))
    out: list[ReverseMatch] = []
    for match in matches:
        verified = False
        try:
            local, _public = _resolve_image(context, match.image_url, client)
            if local is not None and local.exists():
                cand_fp = fingerprint(local)
                if cand_fp:
                    dists = fingerprint_distance(query_fp, cand_fp)
                    if dists["block"] is not None:
                        verified = dists["block"] <= block_threshold
                    elif dists["dhash"] is not None:
                        verified = dists["dhash"] <= dhash_threshold
        except Exception as e:  # noqa: BLE001 - one bad candidate is not the run
            record_failure("verify_reverse_matches:candidate", e)
        out.append(match if verified is match.verified else
                   dataclasses.replace(match, verified=verified))
    return out


def _reverse_impl(context: Any, ref: str,
                  registry: Any = None, verify: bool = False) -> dict[str, Any]:
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
        # 52-websites #46: the scrape now lands in the uniform
        # ReverseMatch shape; the old keys are derived from it so
        # their shape is byte-identical to before.
        matched = _scrape_matches(client, public, registry=reg)
        order = ordered_image_engines(reg)
        flat = [m for engine in order for m in matched.get(engine, [])]
        flat += [m for engine, ms in matched.items()
                 if engine not in order for m in ms]
        if verify:
            flat = verify_reverse_matches(context, local, flat, client=client)
            matched = {}
            for m in flat:
                matched.setdefault(m.engine, []).append(m)
        out["matches"] = [m.to_dict() for m in flat]
        out["matches_by_engine"] = {
            engine: [m.to_dict() for m in ms]
            for engine, ms in matched.items()}
        out["unverified_matches"] = [m.image_url for m in flat
                                     if not m.verified]
        out["unverified_by_engine"] = {
            engine: [m.image_url for m in ms if not m.verified]
            for engine, ms in matched.items()}
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
        except Exception as e:  # noqa: BLE001
            record_failure("_dedupe_urls", e)
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

#: Absolute image URLs anywhere in a page body — the generic
#: grown-engine harvester's widest net (AA-58).
_GENERIC_IMG_URL = re.compile(
    rb'https?://[^\s"\'<>\\]{10,300}?\.(?:jpe?g|png|webp|gif)'
    rb'(?:\?[^\s"\'<>\\]{0,60})?', re.IGNORECASE)


def _parse_generic_body(body: bytes, own_host: str = "") -> list[str]:
    """Similar-image URLs from a GROWN engine's result page
    (AA-58): the union of the seed engines' evidence patterns
    (``mediaurl=`` attributes, ``originalImage`` JSON, <img>
    sources) plus bare absolute image URLs, minus anything hosted
    on the engine's own host (its chrome/thumbnails are not
    matches).  Best-effort by design — the tool labels every
    scraped match unverified."""
    body = body or b""
    raw: list[bytes] = []
    raw += _MEDIA_URL.findall(body)
    raw += _YANDEX_ORIGINAL.findall(body)
    raw += _YANDEX_IMG_SRC.findall(body)
    raw += _GENERIC_IMG_URL.findall(body)
    cleaned: list[bytes] = []
    for hit in raw:
        hit = hit.replace(b"\\/", b"/")
        if hit not in cleaned:
            cleaned.append(hit)
    urls = _dedupe_urls(cleaned, cap=24)
    if own_host:
        urls = [u for u in urls
                if (urllib.parse.urlparse(u).hostname or "").lower()
                not in (own_host, f"www.{own_host}")]
    return urls[:8]


def _grown_scraper(reg: Any, engine: str) -> Any:
    """The (url builder, body parser) pair for one grown engine,
    from its registry template — or None when the engine is not a
    usable grown engine.  Never raises."""
    try:
        template = grown_image_engines(reg).get(engine)
        if not template:
            return None
        domain = engine[len(_GROWN_IMAGE_PREFIX):]

        def url_for(image_url: str, _t: str = template) -> str:
            return _t.replace(
                "{image_url}",
                urllib.parse.quote(image_url, safe=""))

        def parse(body: bytes, _d: str = domain) -> list[str]:
            return _parse_generic_body(body, own_host=_d)

        return url_for, parse
    except Exception as e:  # noqa: BLE001
        record_failure("_grown_scraper", e)
        return None


def _bot_signalled(body: bytes) -> bool:
    """True when a 200 page is a bot-wall/challenge, not results —
    the registry's one bot-signal vocabulary
    (``media/source_registry._BOT_SIGNALS``, mirrored in #89's
    health classification: a wall is ``bot-blocked``, never
    ``dead`` and never a plain no-result)."""
    try:
        from ..media.source_registry import _BOT_SIGNALS

        signals = _BOT_SIGNALS
    except Exception as e:  # noqa: BLE001 - vocabulary fallback
        record_failure("_bot_signalled:vocabulary", e)
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
                image_url: str, registry: Any = None) -> tuple[
        list[str], str, str]:
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
    scraper = _ENGINE_SCRAPERS.get(engine)
    if scraper is None:
        # AA-58: a grown engine resolves its scraper from the
        # registry template; an unresolvable name is this engine's
        # failure, never the chain's.
        scraper = _grown_scraper(registry, engine)
        if scraper is None:
            return [], "failure", "unknown_engine"
    url_for, parse = scraper
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
            urls, kind, error_class = _scrape_one(
                name, client, image_url, registry=registry)
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
                    "engine-isolated) for public URLs, in the uniform "
                    "ReverseMatch shape (engine, page_url, image_url, score, "
                    "verified). Scraped matches are unverified unless "
                    "verify=true, which downloads each candidate and "
                    "fingerprints it against the query.",
        capability="net.out",
    )
    def reverse_image_search(path: str, *, verify: bool = False) -> dict[str, Any]:
        return _reverse_impl(context, path, verify=verify)

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
            "reverse endpoints are judged by real-search outcomes) "
            "| discover (AA-58: intake the reverse-engine catalog "
            "— IQDB / SauceNAO / ascii2d — as untrusted candidates, "
            "then run the health loop that can promote them)."
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
        if act == "discover":
            summary = discover_reverse_image_engines(reg)
            health = reg.verify_health()
            return {"discovery": summary, "health": health,
                    "status": reg.image_status()}
        if act == "status":
            return reg.image_status()
        return {"error": f"unknown action {act!r} — status|health|discover"}
