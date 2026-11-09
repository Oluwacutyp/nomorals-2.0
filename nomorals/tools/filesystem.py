"""Filesystem tools.

Every path is resolved against the workspace and then checked to be *inside* it.
That check is the difference between a tool an agent can use and a tool an agent
can use to read ``~/.ssh/id_rsa`` because a web page told it to. Path traversal is
not a hypothetical: it is the first thing a prompt-injected agent tries.
"""

from __future__ import annotations

import errno
import fnmatch
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..core.errors import NotFound, ToolError, ValidationError
from ..core.policy import Capability
from ..core.text import approx_token_count

__all__ = ["register", "safe_path"]

TEXT_SUFFIXES = {
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".csv", ".tsv", ".html", ".htm", ".xml", ".css", ".sh", ".rs", ".go",
    ".c", ".h", ".cpp", ".java", ".rb", ".php", ".sql", ".env", ".log", ".rst",
}
MAX_READ_BYTES = 8 * 1024 * 1024
DEFAULT_RANGE_BYTES = 64 * 1024
MAX_RANGE_LINES = 5000
DEFAULT_SEARCH_MATCHES = 100
MAX_SEARCH_MATCHES = 1000
SEARCH_SNIFF_BYTES = 8192
SNIPPET_CHARS = 240
HASH_ALGORITHMS = ("sha256", "sha1", "md5")


def _tree_bytes(target: Path) -> int:
    """Total bytes of a file, or of every file under a directory."""
    if target.is_file():
        return target.stat().st_size
    total = 0
    for entry in target.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return total


def safe_path(context: Any, path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """Resolve ``path`` inside the workspace, rejecting traversal.

    Absolute paths are permitted only when they already sit under the workspace.
    ``..`` segments that escape are rejected rather than silently normalized, so a
    rejected attempt is visible in the audit log instead of looking like success.
    """
    settings = getattr(context, "settings", None) if context is not None else None
    if settings is not None:
        root = Path(settings.workspace_dir)
    else:
        root = Path.cwd() / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()

    candidate = Path(os.path.expanduser(str(path)))
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidationError(
            f"path {path!r} escapes the workspace {root}", field="path"
        ) from exc
    if must_exist and not resolved.exists():
        raise NotFound(f"no such path: {resolved}")
    return resolved


def safe_media_path(context: Any, path: str | os.PathLike[str],
                    *, must_exist: bool = False) -> Path:
    """Resolve ``path`` inside the workspace OR the chat media dirs.

    Chat adapters download inbound media (voice notes, photos) to
    ``~/.nomorals/data/media/<platform>/`` — outside the workspace
    sandbox by design.  Tools that consume inbound media (transcribe,
    vision) use this instead of :func:`safe_path` so Devon's own
    downloads are reachable without opening the sandbox to arbitrary
    absolute paths.
    """
    candidate = Path(os.path.expanduser(str(path)))
    if candidate.is_absolute():
        resolved = candidate.resolve()
        # allowlist: the chat media roots
        home = Path(os.path.expanduser("~/.nomorals")).resolve()
        media_roots = [home / "data" / "media", home / "data"]
        for root in media_roots:
            try:
                resolved.relative_to(root)
                if must_exist and not resolved.exists():
                    raise NotFound(f"no such path: {resolved}")
                return resolved
            except ValueError:
                continue
    return safe_path(context, path, must_exist=must_exist)


def register(registry: Any) -> None:
    """Attach the filesystem tools to a registry."""
    context = registry.context

    @registry.register(
        "fs_read",
        description="Read a text file from the workspace.",
        capability=Capability.FS_READ,
    )
    def fs_read(path: str, *, encoding: str = "utf-8", max_bytes: int = MAX_READ_BYTES) -> dict[str, Any]:
        target = safe_path(context, path, must_exist=True)
        if target.is_dir():
            raise ToolError(f"{target} is a directory")
        size = target.stat().st_size
        if size > max_bytes:
            raise ToolError(f"{target.name} is {size} bytes, over the {max_bytes} byte limit")
        raw = target.read_bytes()
        try:
            content = raw.decode(encoding)
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
        return {
            "path": str(target),
            "bytes": size,
            "lines": content.count("\n") + 1,
            "tokens": approx_token_count(content),
            "content": content,
        }

    @registry.register(
        "fs_write",
        description="Write text to a file in the workspace, creating parents.",
        capability=Capability.FS_WRITE,
    )
    def fs_write(path: str, content: str, *, append: bool = False, encoding: str = "utf-8") -> dict[str, Any]:
        target = safe_path(context, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding=encoding) as handle:
            handle.write(content)
        return {"path": str(target), "bytes": target.stat().st_size, "appended": append}

    @registry.register(
        "fs_list",
        description="List directory entries, optionally recursive and filtered by glob.",
        capability=Capability.FS_READ,
    )
    def fs_list(path: str = ".", *, pattern: str = "*", recursive: bool = False, limit: int = 500) -> dict[str, Any]:
        root = safe_path(context, path, must_exist=True)
        if not root.is_dir():
            raise ToolError(f"{root} is not a directory")
        walker = root.rglob(pattern) if recursive else root.glob(pattern)
        entries = []
        for entry in walker:
            if len(entries) >= limit:
                break
            try:
                stat = entry.stat()
            except OSError:
                continue
            entries.append(
                {
                    "path": str(entry.relative_to(root)),
                    "is_dir": entry.is_dir(),
                    "bytes": stat.st_size if entry.is_file() else 0,
                }
            )
        entries.sort(key=lambda e: (not e["is_dir"], e["path"]))
        return {"root": str(root), "count": len(entries), "entries": entries}

    @registry.register(
        "fs_glob",
        description="Find files matching a glob pattern anywhere in the workspace.",
        capability=Capability.FS_READ,
    )
    def fs_glob(pattern: str, *, limit: int = 200) -> dict[str, Any]:
        root = safe_path(context, ".")
        matches = []
        for entry in root.rglob("*"):
            if len(matches) >= limit:
                break
            if entry.is_file() and fnmatch.fnmatch(entry.name, pattern):
                matches.append(str(entry.relative_to(root)))
        return {"pattern": pattern, "count": len(matches), "matches": sorted(matches)}

    @registry.register(
        "fs_delete",
        description="Delete a file or directory inside the workspace. Requires fingerprint approval.",
        capability=Capability.FS_DELETE,
        confirm="biometric",
    )
    def fs_delete(path: str, *, recursive: bool = False) -> dict[str, Any]:
        target = safe_path(context, path, must_exist=True)
        if target.is_dir():
            if not recursive:
                raise ToolError("refusing to delete a directory without recursive=True")
            shutil.rmtree(target)
        else:
            target.unlink()
        return {"deleted": str(target), "recursive": recursive}

    @registry.register(
        "fs_copy",
        description="Copy a file within the workspace.",
        capability=Capability.FS_WRITE,
    )
    def fs_copy(source: str, destination: str) -> dict[str, Any]:
        src = safe_path(context, source, must_exist=True)
        dst = safe_path(context, destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {"from": str(src), "to": str(dst), "bytes": dst.stat().st_size}

    @registry.register(
        "fs_info",
        description="Stat a path: size, kind, and modification time.",
        capability=Capability.FS_READ,
    )
    def fs_info(path: str) -> dict[str, Any]:
        target = safe_path(context, path, must_exist=True)
        stat = target.stat()
        return {
            "path": str(target),
            "is_dir": target.is_dir(),
            "bytes": stat.st_size,
            "modified": stat.st_mtime,
            "suffix": target.suffix,
            "is_text": target.suffix.lower() in TEXT_SUFFIXES,
        }

    @registry.register(
        "fs_move",
        description=(
            "Move or rename a file or directory inside the workspace. "
            "Refuses to overwrite an existing destination unless "
            "overwrite=True; a directory destination receives the source "
            "under its own name."
        ),
        capability=Capability.FS_WRITE,
    )
    def fs_move(source: str, destination: str, *, overwrite: bool = False) -> dict[str, Any]:
        src = safe_path(context, source, must_exist=True)
        root = safe_path(context, ".")
        if src == root:
            raise ToolError("refusing to move the workspace root")
        dst = safe_path(context, destination)
        if dst.is_dir():
            # mv semantics: an existing directory destination receives
            # the source under the source's own name.
            dst = dst / src.name
        if dst == src:
            return {
                "from": str(src), "to": str(dst), "moved": False,
                "note": "source and destination are the same path",
            }
        if src.is_dir() and dst.is_relative_to(src):
            raise ToolError(f"cannot move {src} into its own subtree at {dst}")
        overwrote = False
        if dst.exists():
            if not overwrite:
                raise ToolError(
                    f"destination exists: {dst} "
                    "(pass overwrite=True to replace it)"
                )
            overwrote = True
        is_dir = src.is_dir()
        size = _tree_bytes(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cross_device = False
        try:
            os.replace(src, dst) if overwrote else os.rename(src, dst)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise ToolError(f"move failed: {exc}") from exc
            # Cross-device: copy, then remove the source.  With
            # overwrite, the old destination goes first — this path
            # is not atomic, and the result says so.
            cross_device = True
            try:
                if overwrote:
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                if is_dir:
                    shutil.copytree(src, dst)
                    shutil.rmtree(src)
                else:
                    shutil.copy2(src, dst)
                    src.unlink()
            except OSError as copy_exc:
                raise ToolError(
                    f"cross-device move failed: {copy_exc}"
                ) from copy_exc
        return {
            "from": str(src),
            "to": str(dst),
            "moved": True,
            "is_dir": is_dir,
            "bytes": size,
            "overwrote": overwrote,
            "cross_device": cross_device,
        }

    @registry.register(
        "fs_mkdir",
        description="Create a directory inside the workspace.",
        capability=Capability.FS_WRITE,
    )
    def fs_mkdir(path: str, *, parents: bool = True) -> dict[str, Any]:
        target = safe_path(context, path)
        if target.exists():
            if target.is_dir():
                return {
                    "path": str(target), "created": False,
                    "note": "directory already exists",
                }
            raise ToolError(f"{target} exists and is not a directory")
        try:
            target.mkdir(parents=parents, exist_ok=False)
        except OSError as exc:
            raise ToolError(f"mkdir failed: {exc}") from exc
        return {"path": str(target), "created": True}

    @registry.register(
        "fs_hash",
        description="Hash a workspace file (sha256/sha1/md5), streamed.",
        capability=Capability.FS_READ,
    )
    def fs_hash(path: str, *, algorithm: str = "sha256") -> dict[str, Any]:
        target = safe_path(context, path, must_exist=True)
        if target.is_dir():
            raise ToolError(f"{target} is a directory")
        if algorithm not in HASH_ALGORITHMS:
            raise ValidationError(
                f"unsupported hash algorithm {algorithm!r}; "
                f"use one of {', '.join(HASH_ALGORITHMS)}",
                field="algorithm",
            )
        digest = hashlib.new(algorithm)
        total = 0
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                total += len(chunk)
        return {
            "path": str(target),
            "algorithm": algorithm,
            "hexdigest": digest.hexdigest(),
            "bytes": total,
        }

    @registry.register(
        "fs_read_range",
        description=(
            "Read part of a workspace file too big for fs_read: a byte "
            "range (offset/length) or a line range (start_line/"
            "line_count). Reports the file's total size so the caller "
            "can page deliberately."
        ),
        capability=Capability.FS_READ,
    )
    def fs_read_range(path: str, *, offset: int = 0,
                      length: int = DEFAULT_RANGE_BYTES,
                      start_line: int | None = None,
                      line_count: int | None = None,
                      encoding: str = "utf-8") -> dict[str, Any]:
        target = safe_path(context, path, must_exist=True)
        if target.is_dir():
            raise ToolError(f"{target} is a directory")
        total = target.stat().st_size
        if start_line is not None:
            if start_line < 1:
                raise ValidationError("start_line is 1-based", field="start_line")
            wanted = line_count if line_count is not None else 200
            if wanted < 1:
                raise ValidationError("line_count must be positive", field="line_count")
            capped = wanted > MAX_RANGE_LINES
            wanted = min(wanted, MAX_RANGE_LINES)
            picked: list[str] = []
            total_lines = 0
            with target.open("r", encoding=encoding, errors="replace") as handle:
                for number, line in enumerate(handle, start=1):
                    total_lines = number
                    if start_line <= number < start_line + wanted:
                        picked.append(line.rstrip("\n"))
            beyond = start_line > total_lines
            result: dict[str, Any] = {
                "path": str(target),
                "mode": "lines",
                "total_bytes": total,
                "total_lines": total_lines,
                "start_line": start_line,
                "lines_returned": len(picked),
                "line_count_capped": capped,
                "eof": start_line + len(picked) - 1 >= total_lines,
                "content": "\n".join(picked),
            }
            if beyond:
                result["note"] = (
                    f"start_line {start_line} is beyond the end of the "
                    f"file ({total_lines} lines)"
                )
            return result
        if offset < 0:
            raise ValidationError("offset must be >= 0", field="offset")
        if length < 1:
            raise ValidationError("length must be positive", field="length")
        capped = length > MAX_READ_BYTES
        length = min(length, MAX_READ_BYTES)
        if offset >= total:
            return {
                "path": str(target),
                "mode": "bytes",
                "total_bytes": total,
                "offset": offset,
                "returned_bytes": 0,
                "length_capped": capped,
                "eof": True,
                "next_offset": None,
                "content": "",
                "note": (
                    f"offset {offset} is at or beyond the end of the "
                    f"file ({total} bytes)"
                ),
            }
        with target.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read(length)
        try:
            content = raw.decode(encoding)
            lossy = False
        except UnicodeDecodeError:
            content = raw.decode(encoding, errors="replace")
            lossy = True
        end = offset + len(raw)
        return {
            "path": str(target),
            "mode": "bytes",
            "total_bytes": total,
            "offset": offset,
            "returned_bytes": len(raw),
            "length_capped": capped,
            "eof": end >= total,
            "next_offset": None if end >= total else end,
            "lossy_decode": lossy,
            "content": content,
        }

    @registry.register(
        "fs_search",
        description=(
            "Search file contents under a workspace path (literal or "
            "regex), with line numbers. Binary and oversized files are "
            "skipped and counted; the match cap is stated when hit."
        ),
        capability=Capability.FS_READ,
    )
    def fs_search(pattern: str, *, path: str = ".", glob: str = "*",
                  regex: bool = False, case_sensitive: bool = True,
                  max_matches: int = DEFAULT_SEARCH_MATCHES,
                  max_file_bytes: int = MAX_READ_BYTES) -> dict[str, Any]:
        if not pattern:
            raise ValidationError("pattern must not be empty", field="pattern")
        root = safe_path(context, path, must_exist=True)
        if max_matches < 1:
            raise ValidationError("max_matches must be positive", field="max_matches")
        cap = min(max_matches, MAX_SEARCH_MATCHES)
        matcher: Any
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                raise ValidationError(f"invalid regex: {exc}", field="pattern") from exc
            matcher = compiled.search
        else:
            needle = pattern if case_sensitive else pattern.lower()
            if case_sensitive:
                matcher = lambda line: needle in line  # noqa: E731
            else:
                matcher = lambda line: needle in line.lower()  # noqa: E731
        if root.is_file():
            candidates = [root]
            display_root = root.parent
        else:
            display_root = root
            candidates = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames.sort()
                for name in sorted(filenames):
                    if fnmatch.fnmatch(name, glob):
                        candidates.append(Path(dirpath) / name)
        matches: list[dict[str, Any]] = []
        searched = skipped_binary = skipped_large = skipped_unreadable = 0
        truncated = False
        for candidate in candidates:
            if truncated:
                break
            try:
                if candidate.stat().st_size > max_file_bytes:
                    skipped_large += 1
                    continue
                raw = candidate.read_bytes()
            except OSError:
                skipped_unreadable += 1
                continue
            if b"\x00" in raw[:SEARCH_SNIFF_BYTES]:
                skipped_binary += 1
                continue
            searched += 1
            text = raw.decode("utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if matcher(line):
                    snippet = line.strip()
                    if len(snippet) > SNIPPET_CHARS:
                        snippet = snippet[:SNIPPET_CHARS] + "…"
                    matches.append({
                        "file": str(candidate.relative_to(display_root)),
                        "line": number,
                        "text": snippet,
                    })
                    if len(matches) >= cap:
                        truncated = True
                        break
        return {
            "pattern": pattern,
            "root": str(root),
            "regex": regex,
            "case_sensitive": case_sensitive,
            "matches": matches,
            "count": len(matches),
            "truncated": truncated,
            "cap": cap,
            "files_searched": searched,
            "files_skipped_binary": skipped_binary,
            "files_skipped_large": skipped_large,
            "files_skipped_unreadable": skipped_unreadable,
        }
