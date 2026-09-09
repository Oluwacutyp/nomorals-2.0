"""Filesystem tools.

Every path is resolved against the workspace and then checked to be *inside* it.
That check is the difference between a tool an agent can use and a tool an agent
can use to read ``~/.ssh/id_rsa`` because a web page told it to. Path traversal is
not a hypothetical: it is the first thing a prompt-injected agent tries.
"""

from __future__ import annotations

import fnmatch
import os
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
        description="Delete a file or directory inside the workspace. Requires confirmation.",
        capability=Capability.FS_DELETE,
        confirm=True,
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
