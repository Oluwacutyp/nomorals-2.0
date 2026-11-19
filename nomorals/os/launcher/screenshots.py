"""The launcher asset pipeline: the pictures on an app's details page.

Pattern port of the ArtCraft Launcher's picture handling — upstream
``crates/core/src/details.rs`` (the pinned-tag URL shape, the JPEG
checks, the budgets), ``crates/native/src/screenshots.rs`` (fetch
once, cache, re-validate a saved copy) and ``crates/native/src/
paths.rs`` (the cache layout), plus the publishing rules and the
per-picture terms in the pictures repository's README and
``ATTRIBUTION.md``. Devon's own code throughout: no upstream file or
text is vendored or copied. The shot captions are derived from the
shot keys (a key with hyphens turned into words), never copied from
upstream's page copy.

The design, as mined:

* Pictures are NOT bundled: the launcher reads
  ``https://raw.githubusercontent.com/storytold/craft-launcher-assets/
  <tag>/<app>/<key>.jpg`` at a pinned tag (``SCREENSHOTS_REV``), and
  keeps what it downloads under that tag, so a published tag never
  changes and a saved copy never goes stale.
* An address comes ONLY from a page's own declared shots — an unknown
  key, or a key carrying ``/``, ``.``, ``?`` or ``#``, has no address
  at all, so a request can never escape the pictures repository.
* What comes back is hostile until proven otherwise: it must start
  ``FF D8 FF`` (a captive portal's login page arrives as a 200 and is
  refused), its frame header must declare dimensions (read from the
  SOF marker without decoding any pixels), and those dimensions must
  fit the pixel budget — a tiny JPEG can declare 65,535 x 65,535 and
  the image loader allocates from the header before decoding, so the
  budget is checked before anything is shown or saved (CWE-789). A
  JPEG whose frame size cannot be read is refused, not trusted.
* A saved copy is re-validated on every read: a damaged, swapped or
  oversized file on disk is ignored and the picture is fetched again
  over it.
* Fetching is lead-first: saved copies need no network; once the
  network or the server has failed, the pictures still to fetch fail
  at once with the same reason instead of each waiting for its own
  timeout; a picture that is merely unpublished does not hold the
  others back.
* Publishing rules (the pictures repository's README): a picture is
  a JPEG, 1200 x 750 px (8:5), no metadata, ideally under 150 KB;
  the launcher refuses anything over 2 MB or not a JPEG. Changing a
  picture means updating ``ATTRIBUTION.md`` in the same change,
  tagging the result and bumping the revision — older launchers keep
  reading the tag they know.

Devon's additions, kept separate from the mined design: a manifest
recording what was fetched (with the per-asset attribution row, so
any use carries its terms), and a publish-side ``prepare`` step that
validates a picture against the publishing rules and strips its
metadata segments.

Terms: every picture's licence is in the pictures repository's
``ATTRIBUTION.md`` — the app UI under each app's own licence (MIT OR
Apache-2.0, except PhotoCraft which is Apache-2.0), and the artwork,
photos and footage shown are public domain or were made with the apps
themselves. Devon enforces the repository's own rule — every asset
gets an attribution row: a picture with no attribution row (unknown
to the catalog) is refused for any use, and every use must state its
purpose so the manifest can record what the picture was used for.
The pictures themselves are never vendored into this repo: the
launcher downloads them at runtime, exactly as upstream does, so the
repository stays small.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import LauncherError

__all__ = [
    "IDEAL_MAX_BYTES",
    "MAX_SCREENSHOT_BYTES",
    "MAX_SCREENSHOT_PIXELS",
    "SCREENSHOTS_BASE",
    "SCREENSHOTS_REV",
    "SHOTS",
    "SHOT_HEIGHT",
    "SHOT_WIDTH",
    "TERMS",
    "NetworkFailure",
    "PictureUnavailable",
    "ScreenshotError",
    "Shot",
    "caption_for",
    "check_use",
    "fetch_all",
    "jpeg_dimensions",
    "looks_like_jpeg",
    "manifest_path",
    "prepare_picture",
    "read_manifest",
    "resize_picture",
    "saved_copy",
    "screenshot_path",
    "screenshot_url",
    "screenshot_within_budget",
    "strip_metadata",
    "terms_for",
    "write_atomic",
]

#: Where the pictures live: one 1200 px wide, 8:5 JPEG per shot.
SCREENSHOTS_BASE = (
    "https://raw.githubusercontent.com/storytold/craft-launcher-assets"
)

#: The tag of the pictures repository the launcher reads. Also part
#: of the saved copies' path, so a new revision never reads an old
#: picture. To change a picture, publish a new tag and bump this.
SCREENSHOTS_REV = "v1"

#: The largest picture the launcher accepts. The real ones are 60 to
#: 150 KB.
MAX_SCREENSHOT_BYTES = 2 << 20

#: The largest decoded picture the launcher will hand to the image
#: loader, in pixels. 8192 x 8192 (67 MP) is far above any real
#: screenshot yet refuses the pathological headers.
MAX_SCREENSHOT_PIXELS = 8192 * 8192

#: The publishing rules: 1200 x 750 px, 8:5, ideally under 150 KB.
SHOT_WIDTH = 1200
SHOT_HEIGHT = 750
IDEAL_MAX_BYTES = 150 * 1024

#: Markers with no length field: RSTn, SOI, EOI, TEM.
_STANDALONE_MARKERS = frozenset(
    [0x01, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9]
)

#: Frame markers the standard decoders support.
_SUPPORTED_SOF = frozenset([0xC0, 0xC1, 0xC2])

#: Frame markers no standard decoder renders: accepting their
#: dimensions would let a subsequent large frame bypass the budget.
_UNSUPPORTED_SOF = frozenset(
    [0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF]
)

#: Metadata segments the publishing rules strip: EXIF/XMP (APP1),
#: IPTC (APP13) and comments (COM). APP0 (JFIF) and APP2 (ICC) stay:
#: the format header and the colour profile are not metadata.
_METADATA_MARKERS = {0xE1: "APP1 (EXIF/XMP)", 0xED: "APP13 (IPTC)",
                     0xFE: "COM (comment)"}


class ScreenshotError(LauncherError):
    """A picture could not be had, or a picture failed its checks."""


class PictureUnavailable(ScreenshotError):
    """This picture only: it isn't published, or what came back isn't
    a picture."""


class NetworkFailure(ScreenshotError):
    """The network or the server: the other pictures would fail the
    same way."""


@dataclass(frozen=True)
class Shot:
    """One picture of one app's details page.

    ``key`` names the file to fetch (``<app>/<key>.jpg`` in the
    pictures repository). ``devon_component`` is the Devon organ the
    app maps to, or None where Devon has no port of that app — the
    picture is still fetchable for the app's details page.
    """

    key: str
    devon_component: str | None = None


def caption_for(key: str) -> str:
    """The shot's caption, derived from its key — never copied text."""
    return key.replace("-", " ").capitalize()


# The shot catalog: the 11 apps with pictures, 39 shots. The keys
# are the asset file names in the pictures repository; the captions
# are derived from them. Devon components are mapped only where the
# mapping is exact (PhotoCraft and LightCraft have no Devon organ —
# the photo editor is not the PhotoCraft port — so they map to None).
SHOTS: dict[str, tuple[Shot, ...]] = {
    "photocraft": (
        Shot("demo"), Shot("adjustments"), Shot("layer-styles"),
        Shot("masks"),
    ),
    "vectorcraft": (
        Shot("neon", "nomorals.media.vector"),
        Shot("ribbons", "nomorals.media.vector"),
        Shot("sheet", "nomorals.media.vector"),
        Shot("bezier", "nomorals.media.vector"),
    ),
    "filmcraft": (
        Shot("color", "nomorals.media.film"),
        Shot("assembly", "nomorals.media.film"),
        Shot("keyframes", "nomorals.media.film"),
    ),
    "lightcraft": (
        Shot("hero"), Shot("curve"), Shot("grading"), Shot("masking"),
    ),
    "effectcraft": (
        Shot("hero", "nomorals.media.effects"),
        Shot("graph", "nomorals.media.effects"),
        Shot("effects", "nomorals.media.effects"),
        Shot("3d", "nomorals.media.effects"),
    ),
    "designcraft": (Shot("spread", "nomorals.media.design"),),
    "wordcraft": (
        Shot("hero", "nomorals.documents.word"),
        Shot("review", "nomorals.documents.word"),
        Shot("design", "nomorals.documents.word"),
        Shot("layout", "nomorals.documents.word"),
    ),
    "deckcraft": (
        Shot("editor", "nomorals.documents.decks"),
        Shot("shape", "nomorals.documents.decks"),
        Shot("sorter", "nomorals.documents.decks"),
        Shot("format", "nomorals.documents.decks"),
    ),
    "gridcraft": (
        Shot("hero", "nomorals.documents.sheets"),
        Shot("budget", "nomorals.documents.sheets"),
        Shot("pivot", "nomorals.documents.sheets"),
        Shot("chart", "nomorals.documents.sheets"),
    ),
    "cadcraft": (
        Shot("apartment", "nomorals.media.cad"),
        Shot("bracket", "nomorals.media.cad"),
        Shot("layout", "nomorals.media.cad"),
    ),
    "soundcraft": (
        Shot("edit", "nomorals.media.sound"),
        Shot("mix", "nomorals.media.sound"),
        Shot("automation", "nomorals.media.sound"),
        Shot("score", "nomorals.media.sound"),
    ),
}

# The per-picture terms, from the pictures repository's
# ATTRIBUTION.md: the app UI under each app's own licence, and the
# artwork, photos and footage shown public domain or made with the
# apps themselves.
TERMS: dict[str, dict[str, str]] = {
    "photocraft": {
        "ui_licence": "Apache-2.0",
        "artwork_licence": "public domain",
        "artwork_note": "Hokusai, Monet, Anders/NASA, Vermeer",
        "source": "storytold/photocraft docs/images (see its SOURCES.md)",
    },
    "vectorcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "made with VectorCraft",
        "artwork_note": "the Neon Drive poster, the feature sheet",
        "source": "storytold/vectorcraft docs/images",
    },
    "filmcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "public domain",
        "artwork_note": "NASA Apollo 11 and ISS films, public-domain films",
        "source": "storytold/filmcraft docs/images",
    },
    "lightcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "public domain",
        "artwork_note": "Ansel Adams, Dorothea Lange (NARA / Library of "
                        "Congress); the masking.jpg landscape is "
                        "procedurally generated",
        "source": "storytold/lightcraft docs/images",
    },
    "effectcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "made with EffectCraft",
        "artwork_note": "the compositions shown",
        "source": "storytold/effectcraft docs/images",
    },
    "designcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "made with DesignCraft",
        "artwork_note": "the magazine layout shown",
        "source": "storytold/designcraft docs/images",
    },
    "wordcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "original text written for WordCraft",
        "artwork_note": "the sample document",
        "source": "storytold/wordcraft docs/images",
    },
    "deckcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "generated in code",
        "artwork_note": "the sample deck: no third-party artwork, fonts "
                        "or themes",
        "source": "storytold/deckcraft docs/images",
    },
    "gridcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "made with GridCraft",
        "artwork_note": "the sample workbooks",
        "source": "storytold/gridcraft docs/images",
    },
    "cadcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "generated in code",
        "artwork_note": "the sample drawings",
        "source": "storytold/cadcraft docs/images",
    },
    "soundcraft": {
        "ui_licence": "MIT OR Apache-2.0",
        "artwork_licence": "synthesised by SoundCraft",
        "artwork_note": "the demo session's sounds: no samples or "
                        "artwork from elsewhere",
        "source": "storytold/soundcraft docs/images",
    },
}

# A key is a file stem: lowercase, digits and hyphens only. Anything
# else (a slash, a dot, a query, a fragment) has no address.
_KEY_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789-")


# ── JPEG checks ──────────────────────────────────────────────────────

def looks_like_jpeg(data: bytes) -> bool:
    """True when ``data`` starts like a JPEG file.

    A captive portal's login page, a proxy's error page or a
    truncated answer is refused before it is saved or decoded.
    """
    return data[:3] == b"\xff\xd8\xff"


def jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """The pixel dimensions ``(width, height)`` declared in a JPEG's
    frame header, without decoding any pixels. ``None`` when the
    bytes aren't a JPEG a frame size can be read from.

    Walks the marker segments to the Start-Of-Frame (SOFn) marker,
    whose payload begins with precision (1) then height (2,
    big-endian) then width (2). Only SOF0/1/2 are supported by
    standard decoders; an unsupported frame marker, or more than one
    frame marker, is refused rather than trusted.
    """
    if not looks_like_jpeg(data):
        return None
    i = 2  # past the SOI (FF D8)
    dims: tuple[int, int] | None = None
    while i + 1 < len(data):
        # Markers are FF, with any number of FF fill bytes before the
        # type byte.
        if data[i] != 0xFF:
            break
        j = i
        while j < len(data) and data[j] == 0xFF:
            j += 1
        if j >= len(data):
            return None
        marker = data[j]
        i = j + 1
        if marker in _STANDALONE_MARKERS:
            continue
        if marker == 0xDA:  # SOS: marker headers end, scan data begins
            break
        if i + 1 >= len(data):
            return None
        length = (data[i] << 8) | data[i + 1]
        if length < 2:
            return None
        if marker in _SUPPORTED_SOF:
            if dims is not None:
                # Ambiguous / multiple frame markers: reject to
                # prevent frame smuggling (CWE-789).
                return None
            if i + 6 >= len(data):
                return None
            height = (data[i + 3] << 8) | data[i + 4]
            width = (data[i + 5] << 8) | data[i + 6]
            dims = (width, height)
        elif marker in _UNSUPPORTED_SOF:
            # Unsupported frame marker (e.g. lossless 0xC3): the
            # rendering decoder will skip it or fail, so accepting
            # its dimensions would let a subsequent large frame
            # bypass the budget (CWE-789).
            return None
        i += length
    return dims


def screenshot_within_budget(data: bytes) -> bool:
    """Whether ``data`` is a JPEG whose declared dimensions are
    within ``MAX_SCREENSHOT_PIXELS``, so it is safe to hand to an
    image loader that allocates from the header before decoding.

    A JPEG whose frame size cannot be read is refused, not trusted.
    """
    dims = jpeg_dimensions(data)
    if dims is None:
        return False
    width, height = dims
    return (width != 0 and height != 0
            and width * height <= MAX_SCREENSHOT_PIXELS)


# ── Addresses and cache ──────────────────────────────────────────────

def _shot(app: str, key: str) -> Shot | None:
    """The catalogued shot ``(app, key)``, or None."""
    if not key or any(ch not in _KEY_CHARS for ch in key):
        return None
    for shot in SHOTS.get(app, ()):
        if shot.key == key:
            return shot
    return None


def screenshot_url(app: str, key: str,
                   base_url: str = SCREENSHOTS_BASE,
                   rev: str = SCREENSHOTS_REV) -> str | None:
    """Where the picture ``key`` of ``app`` is published, or None.

    The only way to turn a request into an address: callers never
    build one from their own text.
    """
    if _shot(app, key) is None:
        return None
    return f"{base_url}/{rev}/{app}/{key}.jpg"


def screenshot_path(home: str | Path, app: str, key: str,
                    rev: str = SCREENSHOTS_REV) -> Path:
    """Where a picture is saved after its first download: under the
    revision of the pictures repository it was fetched from, so a new
    revision never reads an old picture."""
    return (Path(home) / "cache" / "screenshots" / rev / app
            / f"{key}.jpg")


def saved_copy(path: str | Path) -> bytes | None:
    """The picture saved at ``path``, if there is one and it is still
    a JPEG of a sane size. A damaged or swapped file is ignored, and
    the picture is fetched again over it."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_SCREENSHOT_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if not looks_like_jpeg(data) or not screenshot_within_budget(data):
        return None
    return data


def write_atomic(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".screenshot-",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


# ── Manifest ─────────────────────────────────────────────────────────

def manifest_path(home: str | Path) -> Path:
    """The manifest recording what was fetched, under ``home``."""
    return Path(home) / "cache" / "screenshots" / "manifest.json"


def read_manifest(home: str | Path) -> dict[str, Any]:
    """The manifest, or an empty one. A missing or corrupt manifest
    is an empty manifest, never a crash — the cache is the truth,
    the manifest is the record of it."""
    path = manifest_path(home)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"assets": {}}
    if not isinstance(data, dict) or not isinstance(
            data.get("assets"), dict):
        return {"assets": {}}
    return data


def _record_manifest(home: str | Path, entry: dict[str, Any]) -> None:
    """Record one fetched picture in the manifest (best effort: a
    manifest that cannot be written is a missed record, never a
    failed fetch)."""
    manifest = read_manifest(home)
    manifest.setdefault("assets", {})[f"{entry['app']}/{entry['key']}"] = entry
    path = manifest_path(home)
    with contextlib.suppress(OSError):
        write_atomic(path, json.dumps(manifest, indent=2,
                                      sort_keys=True).encode("utf-8"))


def _manifest_entry(app: str, key: str, url: str, data: bytes,
                    source: str) -> dict[str, Any]:
    """The manifest row for one picture, carrying its terms."""
    width, height = jpeg_dimensions(data) or (0, 0)
    terms = TERMS[app]
    return {
        "app": app, "key": key, "rev": SCREENSHOTS_REV, "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "width": width, "height": height,
        "ui_licence": terms["ui_licence"],
        "artwork_licence": terms["artwork_licence"],
        "artwork_note": terms["artwork_note"],
        "source": terms["source"], "used_for": source,
        "fetched_at": time.time(),
    }


# ── Terms ────────────────────────────────────────────────────────────

def terms_for(app: str, key: str) -> dict[str, Any] | None:
    """The attribution row for one picture, or None when the picture
    is not catalogued (no row, no use)."""
    if _shot(app, key) is None:
        return None
    terms = TERMS[app]
    return {"app": app, "key": key, "caption": caption_for(key),
            "ui_licence": terms["ui_licence"],
            "artwork_licence": terms["artwork_licence"],
            "artwork_note": terms["artwork_note"],
            "source": terms["source"]}


def check_use(app: str, key: str, purpose: str) -> dict[str, Any]:
    """Whether ``app``/``key`` may be used for ``purpose``.

    The repository's own rule is that every asset gets an
    attribution row: a picture with no row is refused for any use,
    and a use must state its purpose so the manifest can record what
    the picture was used for. A purpose that is empty is refused
    naming that, not silently accepted.
    """
    row = terms_for(app, key)
    if row is None:
        return {"ok": False,
                "reason": f"no attribution row for {app}/{key}: "
                          "unknown picture — refused"}
    if not str(purpose or "").strip():
        return {"ok": False,
                "reason": f"{app}/{key} needs a stated purpose: the "
                          "use is recorded in the manifest"}
    return {"ok": True, "reason": "", "attribution": row}


# ── Fetch ────────────────────────────────────────────────────────────

#: A fetcher: ``url`` -> the picture's bytes, or None when the
#: picture isn't published. Raises NetworkFailure on a transport
#: failure. Injected (never imported here) so this organ stays
#: stdlib-only and offline-testable; the tool supplies the HttpClient
#: fetcher.
Fetcher = Callable[[str], bytes | None]


def _validate(data: bytes) -> None:
    """Refuse ``data`` as a picture, naming why (PictureUnavailable)."""
    if len(data) > MAX_SCREENSHOT_BYTES:
        raise PictureUnavailable(
            f"the picture is {len(data)} bytes, over the "
            f"{MAX_SCREENSHOT_BYTES} byte cap")
    # A captive portal's login page or a proxy's error page arrives
    # as a 200: don't keep it.
    if not looks_like_jpeg(data):
        raise PictureUnavailable("the answer isn't a JPEG picture")
    # A small JPEG can declare enormous dimensions; the image loader
    # allocates from the header before decoding, so refuse anything
    # outside the pixel budget before it is shown or saved.
    if not screenshot_within_budget(data):
        raise PictureUnavailable(
            "the picture's dimensions are too large")


def fetch_all(home: str | Path, app: str, *,
              rev: str = SCREENSHOTS_REV,
              base_url: str = SCREENSHOTS_BASE,
              fetcher: Fetcher) -> dict[str, Any]:
    """Every picture of ``app``, lead first, one result per shot.

    Saved copies need no network. Once the network or the server has
    failed, the pictures still to fetch fail at once with the same
    reason instead of each waiting for its own timeout; saved ones
    are still served. A picture that is merely missing doesn't hold
    the others back. A saved file that can't be written is only a
    missed cache: the picture is still returned.
    """
    if app not in SHOTS:
        raise ScreenshotError(f"unknown app: {app}")
    shots: list[dict[str, Any]] = []
    network_failed: str | None = None
    fetched = cached = failed = 0
    for shot in SHOTS[app]:
        url = screenshot_url(app, shot.key, base_url, rev)
        if url is None:
            # Unreachable: the key comes from the catalog.
            raise ScreenshotError(
                f"no address for catalogued shot {app}/{shot.key}")
        path = screenshot_path(home, app, shot.key, rev)
        data = saved_copy(path)
        if data is not None:
            _record_manifest(home, _manifest_entry(
                app, shot.key, url, data, "cache"))
            shots.append({"key": shot.key, "status": "cached",
                          "path": str(path), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
            cached += 1
            continue
        if network_failed is not None:
            shots.append({"key": shot.key, "status": "network",
                          "error": network_failed})
            failed += 1
            continue
        try:
            data = fetcher(url)
        except NetworkFailure as exc:
            network_failed = str(exc)
            shots.append({"key": shot.key, "status": "network",
                          "error": network_failed})
            failed += 1
            continue
        if data is None:
            shots.append({"key": shot.key, "status": "missing",
                          "error": "this picture isn't published"})
            failed += 1
            continue
        try:
            _validate(data)
        except PictureUnavailable as exc:
            shots.append({"key": shot.key, "status": "invalid",
                          "error": str(exc)})
            failed += 1
            continue
        try:
            write_atomic(path, data)
            saved = True
        except OSError:
            saved = False
        _record_manifest(home, _manifest_entry(
            app, shot.key, url, data, "download"))
        shots.append({"key": shot.key, "status": "downloaded",
                      "path": str(path), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest(),
                      "saved": saved})
        fetched += 1
    return {"app": app, "rev": rev, "shots": shots,
            "fetched": fetched, "cached": cached, "failed": failed}


# ── Publish side ─────────────────────────────────────────────────────

def strip_metadata(data: bytes) -> tuple[bytes, list[str]]:
    """A copy of the JPEG ``data`` with its metadata segments
    removed (EXIF/XMP, IPTC, comments), and the list of what was
    stripped. The format header (JFIF) and the colour profile (ICC)
    stay: they are not metadata. The scan data is copied verbatim.
    """
    if not looks_like_jpeg(data):
        raise PictureUnavailable("not a JPEG picture")
    out = bytearray(data[:2])  # SOI
    stripped: list[str] = []
    i = 2
    while i + 1 < len(data):
        if data[i] != 0xFF:
            # Not a marker: copy the rest verbatim (defensive — the
            # caller validates the frame first, so this should not
            # happen on a well-formed file).
            out.extend(data[i:])
            break
        start = i
        j = i
        while j < len(data) and data[j] == 0xFF:
            j += 1
        if j >= len(data):
            break
        marker = data[j]
        i = j + 1
        if marker in _STANDALONE_MARKERS:
            out.extend(data[start:i])
            continue
        if marker == 0xDA:  # SOS: scan data to EOF, verbatim
            out.extend(data[start:])
            break
        if i + 1 >= len(data):
            raise PictureUnavailable("truncated JPEG segment")
        length = (data[i] << 8) | data[i + 1]
        if length < 2 or i + length > len(data):
            raise PictureUnavailable("truncated JPEG segment")
        if marker in _METADATA_MARKERS:
            stripped.append(_METADATA_MARKERS[marker])
        else:
            out.extend(data[start:i + length])
        i += length
    return bytes(out), stripped


def resize_picture(data: bytes, *,
                   ffmpeg_path: str | None = None) -> bytes:
    """Resize a JPEG to 1200 x 750 with ffmpeg, or refuse honestly
    when ffmpeg is not installed. The output is re-checked: a resize
    that does not come back 1200 x 750 is refused, not trusted."""
    ffmpeg = ffmpeg_path or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ScreenshotError(
            "ffmpeg is not installed: cannot resize a picture to "
            "1200x750 — install ffmpeg or supply a 1200x750 JPEG")
    try:
        proc = subprocess.run(
            [ffmpeg, "-v", "error", "-i", "pipe:0", "-vf",
             f"scale={SHOT_WIDTH}:{SHOT_HEIGHT}", "-q:v", "2",
             "-f", "image2pipe", "pipe:1"],
            input=data, capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ScreenshotError(
            f"ffmpeg resize failed: {type(exc).__name__}: {exc}") from exc
    if proc.returncode != 0:
        raise ScreenshotError(
            "ffmpeg resize failed: "
            f"{proc.stderr.decode('utf-8', errors='replace').strip()}")
    if jpeg_dimensions(proc.stdout) != (SHOT_WIDTH, SHOT_HEIGHT):
        raise ScreenshotError(
            "ffmpeg resize did not come back 1200x750")
    return proc.stdout


def prepare_picture(data: bytes, *, app: str, key: str,
                    resize: bool = False,
                    ffmpeg_path: str | None = None) -> dict[str, Any]:
    """Validate ``data`` against the publishing rules for a new
    picture of ``app``/``key`` and strip its metadata.

    The picture must be a JPEG, 1200 x 750 px (8:5), under the 2 MB
    cap. A picture of other dimensions is refused naming its actual
    size — unless ``resize`` is passed, in which case ffmpeg scales
    it first. The result carries the cleaned bytes and the
    attribution row to record in ATTRIBUTION.md with the change.
    """
    if app not in SHOTS:
        raise ScreenshotError(
            f"unknown app: {app} — a picture is prepared for an app "
            "in the shot catalog")
    if not looks_like_jpeg(data):
        raise PictureUnavailable("not a JPEG picture")
    if len(data) > MAX_SCREENSHOT_BYTES:
        raise PictureUnavailable(
            f"the picture is {len(data)} bytes, over the "
            f"{MAX_SCREENSHOT_BYTES} byte cap")
    dims = jpeg_dimensions(data)
    if dims is None:
        raise PictureUnavailable("no readable JPEG frame header")
    if dims != (SHOT_WIDTH, SHOT_HEIGHT):
        if not resize:
            raise PictureUnavailable(
                f"the picture must be {SHOT_WIDTH}x{SHOT_HEIGHT}, "
                f"got {dims[0]}x{dims[1]}")
        data = resize_picture(data, ffmpeg_path=ffmpeg_path)
        dims = (SHOT_WIDTH, SHOT_HEIGHT)
    cleaned, stripped = strip_metadata(data)
    if len(cleaned) > MAX_SCREENSHOT_BYTES:
        raise PictureUnavailable(
            "the picture is over the 2 MB cap after stripping")
    warnings: list[str] = []
    if len(cleaned) > IDEAL_MAX_BYTES:
        warnings.append(
            f"{len(cleaned)} bytes is over the {IDEAL_MAX_BYTES} byte "
            "ideal")
    terms = TERMS[app]
    return {
        "app": app, "key": key, "ok": True,
        "width": dims[0], "height": dims[1],
        "bytes": len(cleaned),
        "sha256": hashlib.sha256(cleaned).hexdigest(),
        "stripped": stripped, "warnings": warnings,
        "data": cleaned,
        "attribution": {
            "path": f"{app}/{key}.jpg",
            "ui_licence": terms["ui_licence"],
            "artwork_licence": terms["artwork_licence"],
            "artwork_note": terms["artwork_note"],
            "source": terms["source"],
        },
    }
