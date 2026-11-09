"""Zip / unzip archives plus send wiring for the social chats.

This module is the chat send-fusion surface; the archive ENGINE is the
Archivist (:mod:`nomorals.archives`) — the same organ behind the
``archive`` brain tool and the ``/zip`` command (WS1-P1 unification).
The callables here keep their long-standing signatures and result
shapes, but create/extract ride the Archivist underneath, so there is
exactly ONE archive semantics in the system:

* ``zip_create`` — zip a file set or whole directory into
  ``workspace/archives/<name>.zip`` (deflate, max compression; member
  names workspace-relative, as they have always been here).
* ``zip_extract`` — unzip an archive back into the workspace on the
  Archivist's skip-and-report semantics: an entry whose name would
  escape the destination (``../`` / absolute / backslash traversal)
  is SKIPPED and named in the result's ``skipped`` list — never
  silently, and the good entries still land. The whole archive is
  refused only when nothing in it could be extracted, or when a
  zip-bomb shape trips a cap: entry count / declared total size are
  pre-flighted (``NM_ARCHIVE_MAX_EXTRACT_MB``, default 2048;
  ``NM_ARCHIVE_MAX_EXTRACT_FILES``, default 10000 — both also
  monkeypatchable module constants for tests) and the byte budget is
  enforced live while streaming, since zip header sizes can be lies.
  Entries stream in chunks, never read() whole into memory. Because
  the engine identifies by magic bytes, a ``.zip`` path holding a
  tarball now extracts as what it really is.
* ``zip_send`` — zip then send the archive to a chat in one call
  (``zip → send`` works from a single chat command).
* ``unzip_send`` — extract an archive, then send each resulting file to a
  chat (up to ``max_files`` per call; the rest is reported, not silently
  dropped; extraction-skipped entries are reported too, under
  ``unsafe_skipped``).

All paths are sandboxed to the workspace via :func:`filesystem.safe_path`;
sending reuses :func:`filesend.send_file` — the same live-gateway path the
rest of the bot uses, with its auto-compression for large files.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from .filesystem import safe_path

_log = get_logger(__name__)

__all__ = [
    "zip_create",
    "zip_extract",
    "zip_send",
    "unzip_send",
    "register",
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


#: refuse to extract archives whose total uncompressed size exceeds this
#: (env-overridable; also enforced live while writing, since the declared
#: per-entry sizes in a zip header can be lies)
MAX_EXTRACT_BYTES = _env_int("NM_ARCHIVE_MAX_EXTRACT_MB", 2048) * 1024 * 1024
#: refuse archives with more entries than this (zip-bomb guard)
MAX_EXTRACT_FILES = _env_int("NM_ARCHIVE_MAX_EXTRACT_FILES", 10_000)
#: how many extracted files one unzip_send call will forward
DEFAULT_MAX_SEND_FILES = 10


def _workspace_root(context: Any) -> Path:
    settings = getattr(context, "settings", None)
    root = Path(getattr(settings, "workspace_dir", "") or "workspace")
    return root.resolve()


def _collect_sources(context: Any, sources: list[str]) -> list[Path]:
    """Resolve ``sources`` inside the workspace; expand directories."""
    if not sources:
        raise ToolError("zip: needs at least one source path")
    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in sources:
        target = safe_path(context, raw, must_exist=True)
        if target.is_dir():
            for entry in sorted(target.rglob("*")):
                if entry.is_file() and entry not in seen:
                    seen.add(entry)
                    resolved.append(entry)
        elif target.is_file():
            if target not in seen:
                seen.add(target)
                resolved.append(target)
        else:
            raise ToolError(f"zip: cannot zip {raw!r}: not a file or directory")
    if not resolved:
        raise ToolError("zip: found no files to zip")
    return resolved


def zip_create(context: Any, sources: list[str], *, name: str = "") -> dict[str, Any]:
    """Zip ``sources`` (workspace paths; directories included recursively).

    Returns the archive path, file count, and sizes. Fail-fast: an empty
    source list or a missing path raises instead of producing an empty zip.
    The bytes are written by the Archivist (the one archive engine);
    member names stay workspace-relative, as this surface has always
    produced, via the engine's ``arc_base``.
    """
    from ..archives import Archivist
    from ..core.text import slugify

    started = time.perf_counter()
    files = _collect_sources(context, sources)
    root = _workspace_root(context)
    slug = (slugify(name or "archive", limit=60, keep_case=True, extra="._-",
                    strip="-.") or f"archive-{int(time.time())}")
    if not slug.endswith(".zip"):
        slug += ".zip"
    archive: Path = safe_path(context, f"archives/{slug}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    info = Archivist(context).create(
        [str(f) for f in files], str(archive), fmt="zip",
        arc_base=str(root))
    total_in = int(info.get("total_bytes") or 0)
    total_out = int(info.get("compressed_bytes") or archive.stat().st_size)
    _log.info("zip created: %s (%d files, %d → %d bytes)",
              archive, len(files), total_in, total_out)
    return {
        "path": str(archive),
        "files": len(files),
        "input_bytes": total_in,
        "archive_bytes": total_out,
        "ratio": round(total_out / total_in, 3) if total_in else 1.0,
        "seconds": round(time.perf_counter() - started, 2),
    }


def _preflight_bomb_shapes(archive: Path) -> None:
    """Refuse zip-bomb SHAPES before anything is written.

    Only the cost shape is pre-flighted — entry count and the DECLARED
    total uncompressed size against this module's caps. Entry SAFETY
    is not decided here: that is the Archivist's per-entry
    skip-and-report, applied while streaming (the live byte budget
    still guards against headers that lie about their sizes).
    """
    if not zipfile.is_zipfile(archive):
        raise ToolError(f"unzip: {archive.name} is not a zip archive")
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
    if len(infos) > MAX_EXTRACT_FILES:
        raise ToolError(
            f"unzip: archive has {len(infos)} entries — over the "
            f"{MAX_EXTRACT_FILES} file cap (zip-bomb guard); refusing to extract")
    total = sum(info.file_size for info in infos)
    if total > MAX_EXTRACT_BYTES:
        raise ToolError(
            f"unzip: archive would extract {total} bytes — over the "
            f"{MAX_EXTRACT_BYTES}-byte cap (zip-bomb guard); refusing to extract")


def zip_extract(context: Any, archive: str, *, destination: str = "") -> dict[str, Any]:
    """Extract an archive inside the workspace, on the Archivist's
    skip-and-report semantics.

    Unsafe entries are skipped and named in the result's ``skipped``
    list (with ``skipped_count``); the good entries still land. The
    whole archive is refused only when nothing in it could be
    extracted, or a zip-bomb cap trips. Returns the destination, file
    count, and the extracted relative paths (capped at 200 in the
    report — the count is always exact).
    """
    from ..archives import Archivist, detect_format

    started = time.perf_counter()
    src = safe_path(context, archive, must_exist=True)
    if src.is_dir():
        raise ToolError(f"unzip: {archive!r} is a directory, not a zip archive")
    if detect_format(src) == "zip":
        _preflight_bomb_shapes(src)

    dest_name = destination or f"extracted/{src.stem}"
    try:
        result = Archivist(context).extract(
            str(src), dest=dest_name,
            max_files=MAX_EXTRACT_FILES, max_bytes=MAX_EXTRACT_BYTES)
    except ToolError as exc:
        msg = str(exc)
        raise ToolError(msg if msg.startswith("unzip")
                        else f"unzip: {msg}") from exc
    skipped = list(result.get("skipped") or [])
    skipped_count = int(result.get("skipped_count", len(skipped)))
    extracted = int(result.get("extracted") or 0)
    if extracted == 0 and skipped_count:
        details = list(result.get("skipped_details") or [])
        if details and not any(
                "traversal" in str(d.get("reason", ""))
                for d in details):
            # The skips were not unsafe paths — name the real
            # reasons (corrupt entries, unwritable names, …)
            # instead of calling everything a zip-slip.
            shown = "; ".join(
                f"{d.get('name')!r}: {d.get('reason')}"
                for d in details[:3])
            raise ToolError(
                f"unzip: refusing {src.name}: nothing could be "
                f"extracted ({skipped_count} skipped: {shown})")
        shown = ", ".join(repr(s) for s in skipped[:5])
        raise ToolError(
            f"unzip: refusing {src.name}: every file entry was a "
            f"zip-slip / unsafe path ({skipped_count} skipped: {shown}); "
            "nothing could be extracted")
    files = list(result.get("files") or [])
    dest_dir = Path(result.get("dest") or safe_path(context, dest_name))
    _log.info("zip extracted: %s → %s (%d files, %d skipped)",
              src.name, dest_dir, extracted, skipped_count)
    return {
        "archive": str(src),
        "destination": str(dest_dir),
        "files": extracted,
        "paths": files,
        "truncated": extracted > len(files),
        "skipped": skipped,
        "skipped_count": skipped_count,
        "format": result.get("format", ""),
        "seconds": round(time.perf_counter() - started, 2),
    }


def zip_send(
    context: Any,
    sources: list[str],
    platform: str,
    chat_id: str,
    *,
    name: str = "",
    caption: str = "",
) -> dict[str, Any]:
    """Zip ``sources`` then send the archive to a chat — one call, one flow."""
    from .filesend import send_file

    if not (platform or "").strip():
        raise ToolError("zip: send needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("zip: send needs a chat_id")
    created = zip_create(context, sources, name=name)
    try:
        sent = send_file(context, platform, chat_id, created["path"],
                         caption=caption or f"🗜 {Path(created['path']).name}")
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"zip_send: archive created but send failed: {exc}") from exc
    if not getattr(sent, "get", lambda k, d=None: None)("sent", False) \
            and isinstance(sent, dict) and sent.get("sent") is False:
        raise ToolError(f"zip_send: archive created but send reported failure: {sent}")
    out = dict(created)
    out.update(sent if isinstance(sent, dict) else {})
    out["ok"] = True
    return out


def unzip_send(
    context: Any,
    archive: str,
    platform: str,
    chat_id: str,
    *,
    destination: str = "",
    max_files: int = DEFAULT_MAX_SEND_FILES,
    caption_prefix: str = "",
) -> dict[str, Any]:
    """Extract an archive, then send each resulting file to a chat.

    Sends up to ``max_files`` files; anything beyond that is listed under
    ``skipped`` with the reason — never silently dropped. Entries the
    extractor skipped as unsafe are reported under ``unsafe_skipped``.
    """
    from .filesend import send_file

    if not (platform or "").strip():
        raise ToolError("unzip: send needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("unzip: send needs a chat_id")
    if max_files < 1:
        raise ToolError("unzip: max_files must be ≥ 1")
    extracted = zip_extract(context, archive, destination=destination)
    dest_dir = Path(extracted["destination"])
    files = sorted(p for p in dest_dir.rglob("*") if p.is_file())
    to_send = files[:max_files]
    skipped = [str(p.relative_to(dest_dir)) for p in files[max_files:]]

    sent: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for path in to_send:
        rel = str(path.relative_to(dest_dir))
        try:
            result = send_file(
                context, platform, chat_id, str(path),
                caption=f"{caption_prefix}{rel}" if caption_prefix else rel)
            sent.append({"path": rel, "message_id": result.get("message_id", ""),
                         "bytes": result.get("bytes", 0)})
        except Exception as exc:  # noqa: BLE001 - one bad file must not sink the rest
            failed.append({"path": rel, "error": str(exc)})
    report = {
        "ok": not failed and bool(sent),
        "archive": extracted["archive"],
        "destination": extracted["destination"],
        "extracted_files": extracted["files"],
        "unsafe_skipped": extracted.get("skipped", []),
        "unsafe_skipped_count": extracted.get("skipped_count", 0),
        "sent": sent,
        "failed": failed,
        "skipped": skipped,
        "skipped_reason": (
            f"over the per-call max_files={max_files} cap" if skipped else ""),
    }
    if failed and not sent:
        raise ToolError(
            f"unzip_send: extracted {extracted['files']} files but every send "
            f"failed: {failed[0]['error']}")
    return report


def register(registry: Any) -> None:
    """Attach the archive tools to a registry."""
    context = registry.context

    @registry.register(
        "zip_create",
        description=(
            "Zip files or directories (workspace paths; directories zipped "
            "recursively) into workspace/archives/<name>.zip with max "
            "deflate. Returns path, file count, and sizes."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "sources": "list[str] — workspace file/dir paths to include",
            "name": "str (optional) — archive base name",
        },
    )
    def create(sources: list[str], *, name: str = "") -> dict[str, Any]:
        return zip_create(context, sources, name=name)

    @registry.register(
        "zip_extract",
        description=(
            "Unzip an archive into the workspace (the Archivist engine; "
            "a mislabeled tarball extracts as what it really is). "
            "Entries with unsafe paths are skipped and reported in the "
            "result's 'skipped' list — never silently; the archive is "
            "refused only when nothing in it can be extracted or a "
            "zip-bomb cap (entry count / size) trips."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "archive": "str — workspace path to the .zip",
            "destination": "str (optional) — workspace dir to extract into",
        },
    )
    def extract(archive: str, *, destination: str = "") -> dict[str, Any]:
        return zip_extract(context, archive, destination=destination)

    @registry.register(
        "zip_send",
        description=(
            "Zip → send in one call: zip a file set or directory, then send "
            "the archive to an active chat (telegram, whatsapp, …) through "
            "the live gateway."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "sources": "list[str] — workspace file/dir paths to include",
            "platform": "str — telegram | whatsapp | discord | console | …",
            "chat_id": "str — the target chat id",
            "name": "str (optional) — archive base name",
            "caption": "str (optional)",
        },
    )
    def send(sources: list[str], platform: str, chat_id: str, *,
             name: str = "", caption: str = "") -> dict[str, Any]:
        return zip_send(context, sources, platform, chat_id,
                        name=name, caption=caption)

    @registry.register(
        "unzip_send",
        description=(
            "Extract a zip archive, then send each resulting file to an "
            "active chat (up to max_files per call; the rest is reported "
            "under 'skipped', never silently dropped)."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "archive": "str — workspace path to the .zip",
            "platform": "str — telegram | whatsapp | discord | console | …",
            "chat_id": "str — the target chat id",
            "destination": "str (optional) — workspace dir to extract into",
            "max_files": "int (optional, 10) — per-call send cap",
            "caption_prefix": "str (optional)",
        },
    )
    def extract_and_send(archive: str, platform: str, chat_id: str, *,
                         destination: str = "",
                         max_files: int = DEFAULT_MAX_SEND_FILES,
                         caption_prefix: str = "") -> dict[str, Any]:
        return unzip_send(context, archive, platform, chat_id,
                          destination=destination, max_files=max_files,
                          caption_prefix=caption_prefix)
