"""Zip / unzip archives plus send wiring for the social chats.

Tools:

* ``zip_create`` — zip a file set or whole directory into
  ``workspace/archives/<name>.zip`` (deflate, max compression).
* ``zip_extract`` — unzip an archive back into the workspace. Extraction
  is guarded: zip-slip traversal (``../`` / absolute entries) is rejected
  outright, and a total-uncompressed-size cap plus a file-count cap keep a
  zip bomb from filling the disk.
* ``zip_send`` — zip then send the archive to a chat in one call
  (``zip → send`` works from a single chat command).
* ``unzip_send`` — extract an archive, then send each resulting file to a
  chat (up to ``max_files`` per call; the rest is reported, not silently
  dropped).

All paths are sandboxed to the workspace via :func:`filesystem.safe_path`;
sending reuses :func:`filesend.send_file` — the same live-gateway path the
rest of the bot uses, with its auto-compression for large files.
"""

from __future__ import annotations

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

#: refuse to extract archives whose total uncompressed size exceeds this
MAX_EXTRACT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
#: refuse archives with more entries than this (zip-bomb guard)
MAX_EXTRACT_FILES = 10_000
#: how many extracted files one unzip_send call will forward
DEFAULT_MAX_SEND_FILES = 10


def _workspace_root(context: Any) -> Path:
    settings = getattr(context, "settings", None)
    root = Path(getattr(settings, "workspace_dir", "") or "workspace")
    return root.resolve()


def _collect_sources(context: Any, sources: list[str]) -> list[Path]:
    """Resolve ``sources`` inside the workspace; expand directories."""
    if not sources:
        raise ToolError("zip_create needs at least one source path")
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
            raise ToolError(f"cannot zip {raw!r}: not a file or directory")
    if not resolved:
        raise ToolError("zip_create found no files to zip")
    return resolved


def zip_create(context: Any, sources: list[str], *, name: str = "") -> dict[str, Any]:
    """Zip ``sources`` (workspace paths; directories included recursively).

    Returns the archive path, file count, and sizes. Fail-fast: an empty
    source list or a missing path raises instead of producing an empty zip.
    """
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
    total_in = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for entry in files:
            # sources are workspace-sandboxed, so arcnames stay relative
            arcname = str(entry.relative_to(root))
            zf.write(entry, arcname=arcname)
            total_in += entry.stat().st_size
    total_out = archive.stat().st_size
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


def _guard_entries(archive: Path) -> list[zipfile.ZipInfo]:
    """Open ``archive`` and validate every entry before anything is written.

    Raises :class:`ToolError` on zip-slip entries, non-zip input, or
    zip-bomb shapes. Returns the validated entry list.
    """
    if not zipfile.is_zipfile(archive):
        raise ToolError(f"{archive.name} is not a zip archive")
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
    if len(infos) > MAX_EXTRACT_FILES:
        raise ToolError(
            f"archive has {len(infos)} entries — over the "
            f"{MAX_EXTRACT_FILES} file cap; refusing to extract")
    total = sum(info.file_size for info in infos)
    if total > MAX_EXTRACT_BYTES:
        raise ToolError(
            f"archive would extract {total} bytes — over the "
            f"{MAX_EXTRACT_BYTES}-byte cap; refusing to extract")
    for info in infos:
        name = info.filename
        if not name or name.startswith("/") or name.startswith("\\"):
            raise ToolError(f"refusing zip-slip entry: {name!r}")
        parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise ToolError(f"refusing zip-slip entry: {name!r}")
    return infos


def zip_extract(context: Any, archive: str, *, destination: str = "") -> dict[str, Any]:
    """Extract a zip archive inside the workspace (zip-slip guarded).

    Returns the destination, file count, and the extracted relative paths
    (capped at 200 in the report — the count is always exact).
    """
    started = time.perf_counter()
    src = safe_path(context, archive, must_exist=True)
    if src.is_dir():
        raise ToolError(f"{archive!r} is a directory, not a zip archive")
    infos = _guard_entries(src)

    dest_name = destination or f"extracted/{src.stem}"
    dest_dir = safe_path(context, dest_name)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()

    written: list[str] = []
    with zipfile.ZipFile(src, "r") as zf:
        for info in infos:
            parts = [p for p in info.filename.replace("\\", "/").split("/")
                     if p not in ("", ".")]
            target = dest_resolved.joinpath(*parts).resolve()
            try:
                target.relative_to(dest_resolved)
            except ValueError as exc:
                raise ToolError(
                    f"refusing zip-slip entry: {info.filename!r}") from exc
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src_f, target.open("wb") as dst_f:
                dst_f.write(src_f.read())
            written.append(str(target.relative_to(dest_resolved)))
    _log.info("zip extracted: %s → %s (%d files)",
              src.name, dest_dir, len(written))
    return {
        "archive": str(src),
        "destination": str(dest_dir),
        "files": len(written),
        "paths": written[:200],
        "truncated": len(written) > 200,
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
        raise ToolError("zip_send needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("zip_send needs a chat_id")
    created = zip_create(context, sources, name=name)
    try:
        sent = send_file(context, platform, chat_id, created["path"],
                         caption=caption or f"🗜 {Path(created['path']).name}")
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"zip created but send failed: {exc}") from exc
    if not getattr(sent, "get", lambda k, d=None: None)("sent", False) \
            and isinstance(sent, dict) and sent.get("sent") is False:
        raise ToolError(f"zip created but send reported failure: {sent}")
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
    ``skipped`` with the reason — never silently dropped.
    """
    from .filesend import send_file

    if not (platform or "").strip():
        raise ToolError("unzip_send needs a platform (telegram | whatsapp | …)")
    if not (chat_id or "").strip():
        raise ToolError("unzip_send needs a chat_id")
    if max_files < 1:
        raise ToolError("max_files must be ≥ 1")
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
        "sent": sent,
        "failed": failed,
        "skipped": skipped,
        "skipped_reason": (
            f"over the per-call max_files={max_files} cap" if skipped else ""),
    }
    if failed and not sent:
        raise ToolError(
            f"extracted {extracted['files']} files but every send failed: "
            f"{failed[0]['error']}")
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
            "Unzip an archive into the workspace. Zip-slip entries are "
            "rejected and zip-bomb shapes (entry count / total size caps) "
            "refuse to extract."
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
