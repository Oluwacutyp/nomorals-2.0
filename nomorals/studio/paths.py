"""Paths of files inside a package. Always relative, forward slashes,
plain names.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
path.rs``, dual MIT/Apache): a validated path cannot leave its
folder and is a legal file name on Windows, macOS and Linux, so
unpacking never has to guess.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["PathError", "validate_all", "validate_path"]

MAX_PATH_BYTES = 200
MAX_SEGMENT_BYTES = 64
MAX_DEPTH = 8

_RESERVED = frozenset(["con", "prn", "aux", "nul"])


class PathError(Enum):
    """Why a path was refused."""

    EMPTY = "a file path is empty"
    TOO_LONG = "a file path is too long"
    TOO_DEEP = "a file path has too many folders"
    EMPTY_SEGMENT = "a file path has an empty folder name"
    DOT_SEGMENT = "a file path steps out of its folder"
    BAD_CHARACTER = "a file path uses a character that is not allowed"
    TRAILING_DOT = "a file name ends with a dot"
    RESERVED = "a file name is reserved by the system"
    CASE_COLLISION = "two files differ only in letter case"
    DUPLICATE = "a file is listed twice"


def _is_reserved_device(segment: str) -> bool:
    # Windows reserves the name before the first dot: "nul.txt" is
    # as dead as "nul".
    stem = segment.split(".")[0].lower()
    if stem in _RESERVED:
        return True
    for prefix in ("com", "lpt"):
        rest = stem[len(prefix):] if stem.startswith(prefix) else ""
        if rest and len(rest) == 1 and rest in "123456789":
            return True
    return False


def validate_path(path: str) -> PathError | None:
    """Check one path; the refusal reason, or None."""
    if not path:
        return PathError.EMPTY
    if len(path.encode("utf-8")) > MAX_PATH_BYTES:
        return PathError.TOO_LONG
    segments = path.split("/")
    if len(segments) > MAX_DEPTH:
        return PathError.TOO_DEEP
    for segment in segments:
        if not segment:
            return PathError.EMPTY_SEGMENT
        if segment in (".", ".."):
            return PathError.DOT_SEGMENT
        if not all(c.isascii() and (c.isalnum() or c in ".-_")
                   for c in segment):
            return PathError.BAD_CHARACTER
        if segment.endswith("."):
            return PathError.TRAILING_DOT
        if len(segment.encode("utf-8")) > MAX_SEGMENT_BYTES:
            return PathError.TOO_LONG
        if _is_reserved_device(segment):
            return PathError.RESERVED
    return None


def validate_all(paths) -> PathError | None:
    """Check a whole file list: every path valid, none repeated,
    none differing only by case (or being a file where another has
    a folder)."""
    exact: set[str] = set()
    files: set[str] = set()
    # Folder prefix (lower case) -> the spelling first seen. A second
    # spelling is a collision.
    folders: dict[str, str] = {}
    for path in paths:
        err = validate_path(path)
        if err is not None:
            return err
        if path in exact:
            return PathError.DUPLICATE
        exact.add(path)
        segments = path.split("/")
        end = 0
        for i, segment in enumerate(segments):
            end += len(segment) + (1 if i > 0 else 0)
            prefix = path[:end]
            key = prefix.lower()
            if i + 1 < len(segments):
                if key in files:
                    return PathError.CASE_COLLISION
                seen = folders.get(key)
                if seen is not None and seen != prefix:
                    return PathError.CASE_COLLISION
                folders.setdefault(key, prefix)
            else:
                if key in files or key in folders:
                    return PathError.CASE_COLLISION
                files.add(key)
    return None
