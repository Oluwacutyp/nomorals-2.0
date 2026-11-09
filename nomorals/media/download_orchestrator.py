"""Shared download orchestration + per-type acceptance contracts
(Play & Downloads Phase 2 — doc 7 §4, doc 8 P1).

Phase 1 gave every download ONE measured progress stream.  This
module gives every download type ONE journey shape and ONE place
where "did we actually get the thing?" is answered:

    goal → candidates → attempt → retrieve → VERIFY → process → deliver

* The **acceptance contracts** (:func:`accept_artifact`) are per
  type — song, video, image, document, book, archive, software,
  web capture, generic file — and run BEFORE delivery.  A file is
  accepted on positive evidence: it exists, has bytes, its
  container positively identifies as the family the type needs,
  and it parses where an in-repo parser exists (PDF text, ZIP/tar
  structure, image headers, the full-track duration verdict for
  songs).  A file is rejected on positive counter-evidence: an
  HTML error page renamed ``.mp3``, a zero-byte landing, a corrupt
  archive, an "APK" with no manifest.  What cannot be checked is
  recorded as *unverified* in the facts — never silently upgraded
  to a pass, and (except where a type's contract is strict, like
  images) never treated as a failure either.

* The **orchestrator** (:class:`DownloadOrchestrator`) runs the
  candidate loop on the Phase-1 progress stream: a new source per
  candidate, VERIFYING per landed file, PROCESSING per real
  post-processing step.  A failed acceptance moves to the next
  candidate with the reason journaled into the event stream (a
  tracker note) and into the returned attempt story.  Delivery is
  deliberately NOT done here — Phase-1 law: only the layer that
  actually handed the file over may mark ``delivered``, so callers
  deliver ``result.path`` themselves and finish the tracker.

The existing organs keep their jobs: ``MediaDownloader`` runs the
per-URL stage chain, ``PlaybackEngine`` fetches a queue item,
``ResumeDownloader`` moves bytes with validator discipline.  This
module is the contract + loop they share, not a second downloader.
"""

from __future__ import annotations

import gzip
import re
import tarfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.progress_events import PROCESSING, VERIFYING, ProgressTracker, current_tracker
from .download_kit_failures import record_failure

_log = get_logger(__name__)

__all__ = [
    "KIND_SONG", "KIND_AUDIO", "KIND_VIDEO", "KIND_IMAGE",
    "KIND_DOCUMENT", "KIND_BOOK", "KIND_ARCHIVE", "KIND_SOFTWARE",
    "KIND_WEB_CAPTURE", "KIND_FILE", "KINDS",
    "Acceptance", "accept_artifact", "sniff_format", "infer_kind",
    "image_dimensions",
    "CandidateAttempt", "OrchestrationResult", "DownloadOrchestrator",
]

KIND_SONG = "song"
KIND_AUDIO = "audio"
KIND_VIDEO = "video"
KIND_IMAGE = "image"
KIND_DOCUMENT = "document"
KIND_BOOK = "book"
KIND_ARCHIVE = "archive"
KIND_SOFTWARE = "software"
KIND_WEB_CAPTURE = "web_capture"
KIND_FILE = "file"

KINDS = (KIND_SONG, KIND_AUDIO, KIND_VIDEO, KIND_IMAGE, KIND_DOCUMENT,
         KIND_BOOK, KIND_ARCHIVE, KIND_SOFTWARE, KIND_WEB_CAPTURE,
         KIND_FILE)


# ── content sniffing ──────────────────────────────────────────────────

def _head_bytes(path: str | Path, n: int = 512) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _bytes_at(path: str | Path, offset: int, n: int) -> bytes:
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            return fh.read(n)
    except OSError:
        return b""


def sniff_format(path: str | Path) -> str:
    """The container the bytes positively identify as, from magic
    bytes — never from the name.  One of: mp3, flac, ogg, wav, m4a,
    mp4, mkv, webm, avi, flv, wmv, mpeg, ts, png, jpeg, gif, webp,
    bmp, svg, pdf, zip, tar, gzip, 7z, rar, epub (never — see the
    book contract), mobi, rtf, elf, pe, deb, ole, iso, html, text,
    empty, missing, unknown.  Never raises."""
    try:
        p = Path(path)
        if not p.is_file():
            return "missing"
        if p.stat().st_size == 0:
            return "empty"
        head = _head_bytes(p, 512)
        if not head:
            return "unknown"
        if head.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if head.startswith((b"GIF87a", b"GIF89a")):
            return "gif"
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "webp"
        if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
            return "wav"
        if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
            return "avi"
        if head.startswith(b"BM") and len(head) > 26:
            return "bmp"
        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"fLaC"):
            return "flac"
        if head.startswith(b"OggS"):
            return "ogg"
        if head.startswith(b"ID3") or (
                len(head) > 1 and head[0] == 0xFF
                and (head[1] & 0xE0) == 0xE0):
            return "mp3"
        if head[4:8] == b"ftyp":
            brand = head[8:12].decode("ascii", "replace")
            if brand.strip() in ("M4A", "M4B", "M4P", "F4A"):
                return "m4a"
            return "mp4"
        if head.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm" if b"webm" in head else "mkv"
        if head.startswith(b"FLV"):
            return "flv"
        if head.startswith(b"\x30\x26\xb2\x75\x8e\x66\xcf\x11"):
            return "wmv"
        if head.startswith(b"\x00\x00\x01\xba"):
            return "mpeg"
        if head[0] == 0x47 and len(head) > 189 and head[188] == 0x47:
            return "ts"
        if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return "zip"
        if head.startswith(b"\x1f\x8b"):
            return "gzip"
        if head.startswith(b"7z\xbc\xaf\x27\x1c"):
            return "7z"
        if head.startswith(b"Rar!\x1a\x07"):
            return "rar"
        if _bytes_at(p, 257, 5) == b"ustar":
            return "tar"
        if head.startswith(b"\x7fELF"):
            return "elf"
        if head.startswith(b"MZ"):
            return "pe"
        if head.startswith(b"!<arch>\n"):
            return "deb"
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            return "ole"
        if _bytes_at(p, 60, 8) == b"BOOKMOBI":
            return "mobi"
        if head.startswith(b"{\\rtf"):
            return "rtf"
        if p.stat().st_size > 32775 and _bytes_at(p, 32769, 5) == b"CD001":
            return "iso"
        low = head.lower()
        if b"<html" in low or b"<!doctype html" in low or (
                b"<head" in low and b"<body" in low):
            return "html"
        if b"<svg" in low:
            return "svg"
        sample = head[:4096]
        if b"\x00" not in sample:
            try:
                sample.decode("utf-8")
                return "text"
            except UnicodeDecodeError:
                pass
            try:
                sample.decode("latin-1")
                if sum(1 for b in sample if b >= 32 or b in (9, 10, 13)) \
                        > len(sample) * 0.95:
                    return "text"
            except Exception as exc:  # noqa: BLE001 - sniffing is best-effort
                record_failure("sniff_format.text", exc)
        return "unknown"
    except Exception as exc:  # noqa: BLE001 - a sniff never kills a download
        record_failure("sniff_format", exc)
        return "unknown"


_AUDIO_FORMATS = frozenset({"mp3", "flac", "ogg", "wav", "m4a"})
_VIDEO_FORMATS = frozenset({"mkv", "webm", "avi", "flv", "wmv", "mpeg",
                            "ts"})
_IMAGE_FORMATS = frozenset({"png", "jpeg", "gif", "webp", "bmp", "svg"})
_ARCHIVE_FORMATS = frozenset({"zip", "tar", "gzip", "7z", "rar"})
_SOFTWARE_FORMATS = frozenset({"pe", "elf", "deb", "ole", "iso"})

#: Formats that positively identify a file as NOT media of any kind.
_NOT_MEDIA = _IMAGE_FORMATS | {"pdf", "zip", "tar", "gzip", "7z", "rar",
                               "pe", "elf", "deb", "ole", "iso", "mobi",
                               "rtf"}


def image_dimensions(path: str | Path) -> tuple[int, int] | None:
    """(width, height) from the image itself: Pillow when importable
    (it is the repo's imaging rung), else the PNG/JPEG/GIF headers.
    None when neither can read it.  Never raises."""
    try:
        try:
            from PIL import Image

            with Image.open(str(path)) as img:
                return int(img.width), int(img.height)
        except ImportError:
            pass
        head = _head_bytes(path, 64 * 1024)
        if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
            w = int.from_bytes(head[16:20], "big")
            h = int.from_bytes(head[20:24], "big")
            return (w, h) if w and h else None
        if head.startswith((b"GIF87a", b"GIF89a")) and len(head) >= 10:
            w = int.from_bytes(head[6:8], "little")
            h = int.from_bytes(head[8:10], "little")
            return (w, h) if w and h else None
        if head.startswith(b"\xff\xd8\xff"):
            import struct

            i = 2
            data = _head_bytes(path, 256 * 1024)
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):
                    h = struct.unpack(">H", data[i + 5:i + 7])[0]
                    w = struct.unpack(">H", data[i + 7:i + 9])[0]
                    return (w, h) if w and h else None
                seg = struct.unpack(">H", data[i + 2:i + 4])[0]
                if seg < 2:
                    return None
                i += 2 + seg
        return None
    except Exception as exc:  # noqa: BLE001
        record_failure("image_dimensions", exc)
        return None


# ── the acceptance contracts ──────────────────────────────────────────

@dataclass
class Acceptance:
    """The verdict on one landed file.  ``facts`` carries only what
    was actually measured; ``unverified`` names the properties the
    contract could not check — recorded, never hidden."""

    ok: bool = False
    kind: str = ""
    reason: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    unverified: list[str] = field(default_factory=list)


def _zip_members(path: str) -> list[str] | None:
    """Member names of a real ZIP, or None when it does not parse."""
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return None


def _pdf_facts(path: str, facts: dict[str, Any]) -> str:
    """Parse a PDF with the in-repo reader.  Returns "" when it
    parses (facts gain the page count), else the failure reason."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return f"the PDF could not be read: {exc}"
    try:
        from ..core.pdf import read_pdf_text

        read_pdf_text(data)
    except Exception as exc:  # noqa: BLE001 - the parser's verdict
        return f"the PDF does not parse ({type(exc).__name__}: {exc})"
    pages = len(re.findall(rb"/Type\s*/Page(?![\w])", data))
    if pages:
        facts["pages"] = pages
    return ""


def _accept_archive(path: str, fmt: str, acc: Acceptance) -> Acceptance:
    if fmt == "zip" or zipfile.is_zipfile(path):
        members = _zip_members(path)
        if members is None:
            acc.reason = "the archive is corrupt — it does not parse as a ZIP"
            return acc
        if not members:
            acc.reason = "the archive is empty — no members inside"
            return acc
        try:
            with zipfile.ZipFile(path) as zf:
                bad = zf.testzip()
        except (zipfile.BadZipFile, OSError) as exc:
            acc.reason = f"the archive is corrupt ({exc})"
            return acc
        if bad is not None:
            acc.reason = f"the archive is corrupt — member {bad!r} fails its CRC"
            return acc
        acc.ok = True
        acc.facts["members"] = len(members)
        return acc
    if fmt == "tar" or tarfile.is_tarfile(path):
        try:
            with tarfile.open(path) as tf:
                names = tf.getnames()
        except (tarfile.TarError, OSError) as exc:
            acc.reason = f"the archive is corrupt ({exc})"
            return acc
        if not names:
            acc.reason = "the archive is empty — no members inside"
            return acc
        acc.ok = True
        acc.facts["members"] = len(names)
        return acc
    if fmt == "gzip":
        try:
            with gzip.open(path, "rb") as fh:
                fh.read(1)
        except (OSError, EOFError) as exc:
            acc.reason = f"the gzip stream is corrupt ({exc})"
            return acc
        acc.ok = True
        acc.unverified.append("gzip contents not listed (single-stream "
                              "format; first bytes decompress)")
        return acc
    if fmt in ("7z", "rar"):
        acc.ok = True
        acc.unverified.append(f"{fmt} contents not listed — no in-repo "
                              "parser for this format")
        return acc
    acc.reason = f"a {fmt} file, not an archive"
    return acc


def _accept_song(acc: Acceptance, p: Path,
                 expected_duration: float,
                 min_bytes: int) -> Acceptance:
    """The song/audio contract: the full-track verdict judges first.

    `media.library.full_track_verdict` is the identity authority the
    chat handlers have used all along — it reads metadata (tags /
    ffprobe / filename) and returns verified / rejected / unverified.
    A REJECTED verdict (a 30-second clip against a 213-second listing,
    a different track entirely) refuses the artifact with the
    verdict's own reason, whether or not byte probes can run.  Byte
    evidence then hardens the judgment where the file is on disk:
    zero bytes and positive wrong-family evidence refuse; an
    unidentifiable container is recorded as unverified, not refused.
    A path that is not on disk at all is judged by the verdict alone
    (the download step already vouched that it landed; if it truly
    vanished, the send itself fails honestly downstream, as it did
    before this contract existed)."""
    path = str(p)
    exists = p.is_file()
    if exists:
        size = p.stat().st_size
        acc.facts["size_bytes"] = size
        if size == 0:
            acc.reason = "zero bytes — nothing actually came down"
            return acc
        if min_bytes and size < min_bytes:
            acc.reason = (f"only {size} bytes — below the "
                          f"{min_bytes}-byte floor for a {acc.kind}")
            return acc
    if expected_duration:
        from .library import full_track_verdict

        state, why = full_track_verdict(path, expected_duration)
        acc.facts["verify_state"] = state
        acc.facts["verify_why"] = why
        if state == "rejected":
            acc.reason = why
            return acc
        if state == "unverified":
            acc.unverified.append(why)
    if not exists:
        acc.unverified.append(
            "file not on disk to probe; identity judged by the "
            "metadata verdict alone")
        acc.ok = True
        return acc
    fmt = sniff_format(path)
    acc.facts["format"] = fmt
    if fmt == "html":
        acc.reason = ("an HTML page (an error page served instead "
                      f"of the {acc.kind}), not a {acc.kind} file")
        return acc
    if fmt in _NOT_MEDIA or fmt in ("avi", "flv"):
        acc.reason = f"a {fmt} file, not audio"
        return acc
    if fmt in ("unknown", "text"):
        acc.unverified.append(
            "container not positively identified as audio")
    acc.ok = True
    return acc


def accept_artifact(path: str, kind: str, *,
                    expected_duration: float = 0.0,
                    min_bytes: int = 0,
                    expected_format: str = "") -> Acceptance:
    """Check one landed file against its type's contract.

    For non-song kinds the common gate runs first: the file must
    exist and have bytes.  Type rules then judge positive evidence
    (see the module docstring).  The song contract is verdict-first
    (see `_accept_song`).  Never raises."""
    kind = (kind or KIND_FILE).strip().lower()
    if kind not in KINDS:
        kind = KIND_FILE
    acc = Acceptance(kind=kind)
    try:
        p = Path(path)
        if kind in (KIND_SONG, KIND_AUDIO):
            # The song contract is verdict-first (see _accept_song):
            # the library's three-valued full-track verdict is the
            # identity authority the chat handlers have always used,
            # and it judges from metadata even where the byte-level
            # probes cannot run.
            return _accept_song(acc, p, expected_duration, min_bytes)
        if not p.is_file():
            acc.reason = "the file is missing — the download produced nothing"
            return acc
        size = p.stat().st_size
        acc.facts["size_bytes"] = size
        if size == 0:
            acc.reason = "zero bytes — nothing actually came down"
            return acc
        if min_bytes and size < min_bytes:
            acc.reason = (f"only {size} bytes — below the "
                          f"{min_bytes}-byte floor for a {kind}")
            return acc
        fmt = sniff_format(path)
        acc.facts["format"] = fmt
        if fmt == "html" and kind != KIND_FILE:
            acc.reason = ("an HTML page (an error page served instead "
                          f"of the {kind}), not a {kind} file")
            return acc

        if kind == KIND_VIDEO:
            if fmt in _NOT_MEDIA or fmt == "text":
                acc.reason = f"a {fmt} file, not video"
                return acc
            if fmt in _VIDEO_FORMATS or fmt in ("mp4", "ogg"):
                acc.ok = True
                return acc
            ext = p.suffix.lower()
            if ext in _VIDEO_EXTS:
                acc.ok = True
                acc.unverified.append(
                    f"container not positively identified ({fmt}); "
                    f"accepted on the {ext} extension")
                return acc
            acc.reason = (f"could not identify the file as video "
                          f"(format: {fmt})")
            return acc

        if kind == KIND_IMAGE:
            if fmt == "svg":
                acc.ok = True
                acc.facts["format"] = "svg"
                return acc
            if fmt in _IMAGE_FORMATS:
                dims = image_dimensions(path)
                if dims:
                    acc.facts["width"], acc.facts["height"] = dims
                else:
                    acc.unverified.append("dimensions not readable")
                acc.ok = True
                return acc
            acc.reason = f"not a recognizable image (format: {fmt})"
            return acc

        if kind in (KIND_DOCUMENT, KIND_BOOK):
            if fmt == "pdf":
                err = _pdf_facts(path, acc.facts)
                if err:
                    acc.reason = err
                    return acc
                acc.ok = True
                return acc
            if fmt == "zip":
                members = _zip_members(path)
                if members is None:
                    acc.reason = ("the file does not parse as a "
                                  "document package (bad ZIP)")
                    return acc
                names = set(members)
                if kind == KIND_BOOK and (
                        "META-INF/container.xml" in names
                        or "mimetype" in names):
                    acc.ok = True
                    acc.facts["format"] = "epub"
                    acc.facts["members"] = len(members)
                    return acc
                if "word/document.xml" in names:
                    acc.ok = True
                    acc.facts["format"] = "docx"
                    return acc
                if "xl/workbook.xml" in names:
                    acc.ok = True
                    acc.facts["format"] = "xlsx"
                    return acc
                if "ppt/presentation.xml" in names:
                    acc.ok = True
                    acc.facts["format"] = "pptx"
                    return acc
                if "content.xml" in names:
                    acc.ok = True
                    acc.facts["format"] = "odt"
                    return acc
                acc.reason = ("a ZIP package, but not a document/book "
                              "structure (no document, book or office "
                              "members inside)")
                return acc
            if fmt == "mobi" and kind == KIND_BOOK:
                acc.ok = True
                return acc
            if fmt == "rtf" and kind == KIND_DOCUMENT:
                acc.ok = True
                return acc
            if fmt == "text":
                head = _head_bytes(path, 1024)
                if kind == KIND_BOOK and b"<FictionBook" in head:
                    acc.facts["format"] = "fb2"
                acc.ok = True
                return acc
            if kind == KIND_BOOK:
                acc.reason = f"a {fmt} file, not a book"
            else:
                acc.reason = f"a {fmt} file, not a readable document"
            return acc

        if kind == KIND_ARCHIVE:
            return _accept_archive(path, fmt, acc)

        if kind == KIND_SOFTWARE:
            ext = p.suffix.lower()
            if ext == ".apk" or expected_format == "apk":
                members = _zip_members(path)
                if members is None:
                    acc.reason = "not an APK — the package does not parse as a ZIP"
                    return acc
                if "AndroidManifest.xml" not in members:
                    acc.reason = ("no AndroidManifest.xml in the "
                                  "package — not an APK")
                    return acc
                acc.ok = True
                acc.facts["format"] = "apk"
                acc.facts["members"] = len(members)
                return acc
            if fmt in _SOFTWARE_FORMATS:
                acc.ok = True
                return acc
            if fmt == "zip":
                members = _zip_members(path) or []
                if "META-INF/MANIFEST.MF" in members:
                    acc.facts["format"] = "jar"
                else:
                    acc.unverified.append(
                        "ZIP-packaged software; contents not verified")
                acc.ok = True
                return acc
            if fmt in _NOT_MEDIA or fmt in _AUDIO_FORMATS \
                    or fmt in _VIDEO_FORMATS:
                acc.reason = f"a {fmt} file, not recognizable software"
                return acc
            acc.ok = True
            acc.unverified.append(
                f"software format not positively identified ({fmt})")
            return acc

        if kind == KIND_WEB_CAPTURE:
            want = (expected_format or "").strip().lower()
            if want in ("png", "jpg", "jpeg", "pdf"):
                want_fmt = "jpeg" if want in ("jpg", "jpeg") else want
                if fmt != want_fmt:
                    acc.reason = (f"the capture is a {fmt} file, not "
                                  f"the expected {want}")
                    return acc
                acc.ok = True
                return acc
            if want in ("mp4", "webm", "video"):
                return accept_artifact(path, KIND_VIDEO,
                                       min_bytes=min_bytes)
            if want == "html":
                if fmt not in ("html", "text"):
                    acc.reason = f"the capture is a {fmt} file, not HTML"
                    return acc
                acc.ok = True
                return acc
            acc.ok = True
            return acc

        # KIND_FILE: existence + bytes is the whole contract.
        acc.ok = True
        return acc
    except Exception as exc:  # noqa: BLE001 - a contract never raises
        record_failure("accept_artifact", exc)
        acc.ok = False
        acc.reason = f"acceptance check failed: {exc}"
        return acc


# ── kind inference ────────────────────────────────────────────────────

_AUDIO_EXTS = frozenset({".mp3", ".m4a", ".ogg", ".opus", ".wav",
                         ".flac", ".aac", ".wma"})
_VIDEO_EXTS = frozenset({".mp4", ".mkv", ".webm", ".avi", ".mov",
                         ".flv", ".wmv", ".mpg", ".mpeg", ".ts",
                         ".m2ts", ".3gp", ".m4v"})
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp",
                         ".bmp", ".svg", ".tif", ".tiff", ".avif"})
_DOCUMENT_EXTS = frozenset({".pdf", ".doc", ".docx", ".xls", ".xlsx",
                            ".ppt", ".pptx", ".odt", ".ods", ".odp",
                            ".txt", ".md", ".rtf", ".csv"})
_BOOK_EXTS = frozenset({".epub", ".mobi", ".azw", ".azw3", ".fb2",
                        ".djvu"})
_ARCHIVE_EXTS = frozenset({".zip", ".tar", ".gz", ".tgz", ".bz2",
                           ".xz", ".7z", ".rar"})
_SOFTWARE_EXTS = frozenset({".apk", ".exe", ".msi", ".deb", ".rpm",
                            ".dmg", ".iso", ".img", ".appimage", ".jar"})

_HINT_KINDS = {
    "song": KIND_SONG, "music": KIND_SONG, "track": KIND_SONG,
    "audio": KIND_AUDIO, "sound": KIND_AUDIO,
    "video": KIND_VIDEO, "movie": KIND_VIDEO, "clip": KIND_VIDEO,
    "image": KIND_IMAGE, "photo": KIND_IMAGE, "picture": KIND_IMAGE,
    "document": KIND_DOCUMENT, "doc": KIND_DOCUMENT,
    "book": KIND_BOOK, "ebook": KIND_BOOK,
    "archive": KIND_ARCHIVE, "zip": KIND_ARCHIVE,
    "software": KIND_SOFTWARE, "app": KIND_SOFTWARE,
    "capture": KIND_WEB_CAPTURE, "screenshot": KIND_WEB_CAPTURE,
    "web_capture": KIND_WEB_CAPTURE,
    "file": KIND_FILE,
}


def infer_kind(path: str = "", *, url: str = "", hint: str = "",
               audio_only: bool = False) -> str:
    """Best kind for a download whose type was not declared: the
    caller's hint first, then the extension of the landed path (or
    the URL), then the sniffed container.  Never raises."""
    try:
        if audio_only:
            return KIND_AUDIO
        h = (hint or "").strip().lower()
        if h in _HINT_KINDS:
            return _HINT_KINDS[h]
        if h in KINDS:
            return h
        ext = ""
        for source in (path, url):
            if source:
                ext = Path(str(source).split("?")[0]).suffix.lower()
                if ext:
                    break
        if ext in _AUDIO_EXTS:
            return KIND_AUDIO
        if ext in _VIDEO_EXTS:
            return KIND_VIDEO
        if ext in _IMAGE_EXTS:
            return KIND_IMAGE
        if ext in _BOOK_EXTS:
            return KIND_BOOK
        if ext in _DOCUMENT_EXTS:
            return KIND_DOCUMENT
        if ext in _ARCHIVE_EXTS:
            return KIND_ARCHIVE
        if ext in _SOFTWARE_EXTS:
            return KIND_SOFTWARE
        if path:
            fmt = sniff_format(path)
            if fmt in _AUDIO_FORMATS:
                return KIND_AUDIO
            if fmt in _VIDEO_FORMATS or fmt == "mp4":
                return KIND_VIDEO
            if fmt in _IMAGE_FORMATS:
                return KIND_IMAGE
            if fmt == "pdf":
                return KIND_DOCUMENT
            if fmt == "zip":
                return KIND_ARCHIVE
            if fmt in _SOFTWARE_FORMATS:
                return KIND_SOFTWARE
        return KIND_FILE
    except Exception as exc:  # noqa: BLE001
        record_failure("infer_kind", exc)
        return KIND_FILE


# ── the orchestrator loop ─────────────────────────────────────────────

@dataclass
class CandidateAttempt:
    """One candidate's story: how far it got and why it ended."""

    label: str = ""
    ok: bool = False
    reason: str = ""
    path: str = ""


@dataclass
class OrchestrationResult:
    """The outcome of one orchestrated fetch.  ``attempts`` is the
    journal-shaped story — every candidate, in order, with its
    ending — so the chat report and the journal can never diverge:
    both are rendered from this."""

    ok: bool = False
    path: str = ""
    kind: str = ""
    candidate: Any = None
    acceptance: Acceptance | None = None
    attempts: list[CandidateAttempt] = field(default_factory=list)

    def story(self) -> str:
        lines = []
        for a in self.attempts:
            lines.append(f"{a.label} — "
                         f"{'delivered-ready' if a.ok else a.reason}")
        return "; ".join(lines)


class DownloadOrchestrator:
    """The shared candidate loop on the Phase-1 progress stream.

    ``tracker`` defaults to the context-bound tracker at call time,
    so chat jobs (which bind one per operation) get the full stage
    story and standalone callers get a silent no-op stream.
    """

    def __init__(self, tracker: ProgressTracker | None = None) -> None:
        self._tracker = tracker

    def _bound(self) -> ProgressTracker | None:
        if self._tracker is not None:
            return self._tracker
        try:
            return current_tracker()
        except Exception as exc:  # noqa: BLE001 - progress never breaks
            record_failure("DownloadOrchestrator._bound", exc)
            return None

    def verify(self, path: str, kind: str, *,
               expected_duration: float = 0.0,
               min_bytes: int = 0,
               expected_format: str = "") -> Acceptance:
        """Run one type contract, journaled on the stream: the
        VERIFYING stage opens, and a rejection lands as a tracker
        note carrying its reason — the stream is the journal the
        chat report is rendered from."""
        tracker = self._bound()
        if tracker is not None:
            tracker.set_stage(VERIFYING, note="checking the landed file")
        acc = accept_artifact(
            path, kind, expected_duration=expected_duration,
            min_bytes=min_bytes, expected_format=expected_format)
        if tracker is not None and not acc.ok:
            tracker.note(f"rejected: {acc.reason}")
        return acc

    def run(self, candidates: Iterable[Any],
            retrieve: Callable[[Any], tuple[str | None, str]],
            *, kind: str,
            label_of: Callable[[Any], str] | None = None,
            expected_duration_of: Callable[[Any], float] | None = None,
            process_steps: Iterable[tuple[str, Callable[[str], str]]] = (),
            manage_sources: bool = True) -> OrchestrationResult:
        """Try candidates in order until one retrieves AND verifies.

        ``retrieve(cand)`` returns ``(path, "")`` on a landed file
        or ``(None, reason)``.  Accepted files run ``process_steps``
        — ``(label, fn)`` pairs, each emitting PROCESSING with its
        own label — before the result returns.  Delivery is the
        caller's step (see the module docstring)."""
        result = OrchestrationResult(kind=kind)
        tracker = self._bound()
        label_of = label_of or (lambda c: str(c))
        steps = list(process_steps)
        for cand in candidates:
            label = label_of(cand)
            if tracker is not None and manage_sources:
                tracker.begin_source(label)
            try:
                path, reason = retrieve(cand)
            except Exception as exc:  # noqa: BLE001 - a route failure
                path, reason = None, f"{type(exc).__name__}: {exc}"
            if not path:
                result.attempts.append(CandidateAttempt(
                    label=label, ok=False,
                    reason=reason or "retrieve produced no file"))
                continue
            expected = 0.0
            if expected_duration_of is not None:
                try:
                    expected = float(expected_duration_of(cand) or 0.0)
                except (TypeError, ValueError):
                    expected = 0.0
            acc = self.verify(path, kind, expected_duration=expected)
            if not acc.ok:
                result.attempts.append(CandidateAttempt(
                    label=label, ok=False, reason=acc.reason, path=path))
                continue
            out = path
            failed_step = ""
            for step_label, fn in steps:
                if tracker is not None:
                    tracker.set_stage(PROCESSING, note=step_label)
                try:
                    out = fn(out) or out
                except Exception as exc:  # noqa: BLE001
                    failed_step = (f"processing failed at "
                                   f"{step_label!r}: {exc}")
                    break
            if failed_step:
                result.attempts.append(CandidateAttempt(
                    label=label, ok=False, reason=failed_step, path=path))
                continue
            result.ok = True
            result.path = out
            result.candidate = cand
            result.acceptance = acc
            result.attempts.append(CandidateAttempt(
                label=label, ok=True, path=out))
            return result
        return result
